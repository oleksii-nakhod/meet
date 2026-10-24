"""Per-speaker audio front-end (F6): noise floor, voice activity, echo guard, segment boundaries.

Noise is handled in four layers, cheapest first:
  1. the browser's noiseSuppression / echoCancellation / autoGainControl (web/room.js),
  2. this front-end: a frame counts as speech only if it is well above this person's own noise floor (calibrated in
     the first 2 s and tracked during silence), has its energy in the speech band, and is not spectrally flat
     (fans, hiss, hum are flat or low; voiced speech is peaky); speech must last 100 ms+ to start a segment, so
     keyboard clicks and bumps never do,
  3. the AI's own voice leaking back from someone's speakers (well below what the AI is playing) is ignored,
  4. Boson's server-side noise reduction (near_field) on every transcription session, and a transcript guard that
     drops empty or non-word results (stt/).

push() takes one 20 ms frame and returns the events it caused plus the frames to forward to transcription: from
200 ms before speech starts until the segment ends.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

SR = 24000
FRAME = 480
FRAME_S = FRAME / SR

_FREQS = np.fft.rfftfreq(FRAME, 1 / SR)
_BAND = (_FREQS >= 200) & (_FREQS <= 4000)
_LOW = _FREQS < 200
_WINDOW = np.hanning(FRAME).astype(np.float32)


@dataclass
class Event:
    kind: str            # "start" | "clause" | "end" | "blip"
    t: float             # stream time (s) of the event
    duration: float = 0  # voiced seconds in the segment so far


@dataclass
class Out:
    events: list[Event] = field(default_factory=list)
    audio: list[np.ndarray] = field(default_factory=list)

    def __iter__(self):
        return iter(self.events)


def power(frame: np.ndarray) -> np.ndarray:
    x = frame.astype(np.float32) / 32768.0
    return np.abs(np.fft.rfft(x * _WINDOW)) ** 2 + 1e-12


def level_db(frame: np.ndarray) -> float:
    x = frame.astype(np.float32) / 32768.0
    return float(20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9))


def features(frame: np.ndarray, noise: np.ndarray | None = None) -> tuple[float, float, float, float]:
    """(level dBFS, SNR in 200-4000 Hz against the noise spectrum, share of the cleaned energy in that band,
    spectral flatness of the cleaned band). Cleaning = spectral subtraction of the tracked noise spectrum."""
    p = power(frame)
    db = level_db(frame)
    if noise is None:
        noise = np.full_like(p, 1e-9)
    clean = np.maximum(p - 1.5 * noise, 0.02 * p)
    band = clean[_BAND]
    snr = float(10 * np.log10(band.sum() / (noise[_BAND].sum() + 1e-12)))
    share = float(band.sum() / clean.sum())
    flat = float(np.exp(np.mean(np.log(band))) / np.mean(band))
    return db, snr, share, flat


class SpeakerFrontEnd:
    def __init__(self, *, clause_silence: float = 0.28, line_silence: float = 0.45, margin_db: float = 6.0,
                 min_db: float = -52.0, echo_margin_db: float = 25.0, preroll: float = 0.2,
                 onset_frames: int = 5, min_voiced: float = 0.15, max_segment: float = 30.0):
        self.clause_silence, self.line_silence = clause_silence, line_silence
        self.margin_db, self.min_db, self.echo_margin_db = margin_db, min_db, echo_margin_db
        self.onset_frames, self.min_voiced, self.max_segment = onset_frames, min_voiced, max_segment
        self.pre: deque[np.ndarray] = deque(maxlen=max(1, int(preroll / FRAME_S)))
        self.recent: deque[bool] = deque(maxlen=onset_frames + 2)
        self.calib: list[float] = []
        self.floor = -70.0
        self.noise: np.ndarray | None = None   # per-bin noise power, tracked in silence
        self.t = 0.0
        self.active = False
        self.voiced = 0.0
        self.seg_len = 0.0
        self.silence = 0.0
        self.clause_sent = False
        self.ai_db: float | None = None   # level the AI is playing right now (None = silent)
        self.last_db = -100.0
        # near-field gate: once this person's own speaking level is known, speech far below it is someone else in
        # their room (cafe chatter, a TV), not them
        self.near_field_db = 14.0
        self.voice_levels: deque[float] = deque(maxlen=500)   # dB of their voiced frames in finished segments
        self._seg_levels: list[float] = []
        self._last_seg_n = 0

    # -------------------------------------------------------------- state
    @property
    def noisy(self) -> bool:
        return len(self.calib) >= 50 and self.floor > -45.0

    def set_ai_level(self, db: float | None) -> None:
        self.ai_db = db

    def is_speech(self, frame: np.ndarray) -> bool:
        p = power(frame)
        db = level_db(frame)
        self.last_db = db
        if self.noise is None:
            self.noise = p.copy()
        calibrating = len(self.calib) < 100             # first 2 s: learn the room
        if calibrating:
            self.calib.append(db)
            self.floor = float(np.percentile(self.calib, 20))
            self.noise = np.minimum(self.noise * 1.02, p) * 0.5 + self.noise * 0.5 if len(self.calib) > 1 else p
        _, snr, share, flat = features(frame, self.noise)
        speech = (snr > self.margin_db and db > self.min_db and share > 0.5 and flat < 0.45
                  and db > self.floor + 3.0)
        if speech and self.voice_db is not None and db < self.voice_db - self.near_field_db:
            speech = False                              # far-field: someone else in their room
        if speech and self.ai_db is not None and db < self.ai_db - self.echo_margin_db:
            speech = False                              # the AI's own voice coming back through someone's speakers
        if not speech and not self.active:              # track the noise only in silence: falls fast, rises slowly
            rate = 0.2 if db < self.floor else 0.02
            self.floor += rate * (db - self.floor)
            up = p > self.noise
            self.noise = np.where(up, self.noise + 0.02 * (p - self.noise), self.noise + 0.2 * (p - self.noise))
        return speech

    # -------------------------------------------------------------- per frame
    def push(self, frame: np.ndarray) -> Out:
        out = Out()
        self.t += FRAME_S
        speech = self.is_speech(frame)
        self.recent.append(speech)
        if not self.active:
            self.pre.append(frame)
            if sum(self.recent) >= self.onset_frames:
                self.active, self.voiced, self.silence, self.clause_sent = True, sum(self.recent) * FRAME_S, 0.0, False
                self.seg_len = len(self.pre) * FRAME_S
                out.events.append(Event("start", self.t - len(self.pre) * FRAME_S, self.voiced))
                out.audio.extend(self.pre)
                self.pre.clear()
            return out
        out.audio.append(frame)
        self.seg_len += FRAME_S
        if speech:
            self._seg_levels.append(self.last_db)
            self.voiced += FRAME_S
            self.silence = 0.0
            self.clause_sent = False
        else:
            self.silence += FRAME_S
        if not self.clause_sent and self.silence >= self.clause_silence:
            self.clause_sent = True
            out.events.append(Event("clause", self.t, self.voiced))
        if self.silence >= self.line_silence or self.seg_len >= self.max_segment:
            self.active = False
            self.recent.clear()
            kind = "end" if self.voiced >= self.min_voiced else "blip"
            out.events.append(Event(kind, self.t, self.voiced))
            if kind == "end":
                self.voice_levels.extend(self._seg_levels)
                self._last_seg_n = min(len(self._seg_levels), self.voice_levels.maxlen or 0)
            self._seg_levels = []
        return out

    @property
    def voice_db(self) -> float | None:
        """This person's speaking level, once 1 s of voiced frames is known. The 90th percentile: background talkers
        that slipped into earlier segments are quieter than the person at the mic, so they don't drag it down."""
        if len(self.voice_levels) < 50:
            return None
        return float(np.percentile(self.voice_levels, 90))

    def confirm_voice(self, ok: bool) -> None:
        """The transcript guard's verdict on the last segment: non-speech segments must not teach the voice level."""
        if not ok:
            for _ in range(min(self._last_seg_n, len(self.voice_levels))):
                self.voice_levels.pop()
        self._last_seg_n = 0

    @property
    def speaking(self) -> bool:
        """Currently voicing (not in a pause)."""
        return self.active and self.silence < 0.1
