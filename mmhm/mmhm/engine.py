"""The Mm-hm engine: a turn-taking layer between a microphone/speaker and Higgs Realtime.

It runs Higgs with `turn_detection: null` and decides the turns itself:
- **When to answer** (speculative end-of-turn): after a short silence, if the turn-taking model says the
  user is done, commit the audio and let Higgs start generating *without playing it yet*. Play it once the
  model is confident, or throw it away if the user keeps talking. That hides most of Higgs's ~530 ms
  generation latency without answering half-sentences.
- **Whether a sound is an interruption** (duck, then decide): when the user makes a sound while Higgs is
  talking, drop Higgs's volume at once, then within ~250 ms decide: a backchannel ("mm-hm", "yeah")
  restores the volume and is discarded; a real interruption stops playback and truncates Higgs's turn
  to what was actually heard.
- **Listening noises** (optional): while the user tells a long story, say "mm-hm" where a listener would.

Modes `server_vad` / `semantic_vad` run plain Higgs (its own turn detection; barge-in on any speech
as the docs prescribe) so the same harness measures the baseline.

The engine is clocked by the caller: every 20 ms, `await engine.tick(mic)` takes 20 ms of 24 kHz int16 mic
audio and returns 20 ms of speaker audio. The same code drives the web app and the offline evaluation."""
import asyncio
import base64
import re
import time
from dataclasses import dataclass, field

import json
import os
from collections import deque

import numpy as np
from scipy.signal import resample_poly

from .expressive import EXPRESSIVE_RULES, SentenceSpeaker, spoken
from .higgs import SR, Session

TICK = 0.02
N = int(SR * TICK)  # 480 samples at 24 kHz


# A turn that ends on one of these words is almost never finished ("...play the", "...wanted to", "um").
INCOMPLETE = set("""the a an my your our their his her its this that these those to of in on at for with about from by
into like as than and but or because so if when then while although though is was are were be been being am have has
had do does did will would can could should might must um uh er like i we they he she which who whom whose where
we're you're they're i'm it's there's that's he's she's i've we've you've i'd we'd i'll we'll""".split())


def unfinished(transcript):
    """True if the transcript of what the user said so far ends mid-sentence."""
    t = (transcript or "").strip()
    if not t or t.endswith("?"):  # a question is complete even if it ends on a preposition ("which team are you on?")
        return False
    if t.endswith((",", "...", "…", "-", "—")):
        return True
    words = re.findall(r"[a-zA-Z']+", t.lower())
    return bool(words) and words[-1] in INCOMPLETE


@dataclass
class Policy:
    min_silence: float = 0.10      # silence before the gate may speculate
    th_start: float = 0.30         # p_now below this -> start generating (speculate)
    th_play: float = 0.12          # p_now below this -> play the speculative answer
    max_wait: float = 0.80         # silence after which we play / commit if the model isn't sure they're mid-thought
    give_up: float = 1.60          # silence after which we answer even if the model thinks they'll go on
    hold_p: float = 0.12           # p_now above this at max_wait counts as "mid-thought": wait until give_up
    min_turn: float = 0.25         # seconds of user speech needed before a turn can end
    resume_cancel: float = 0.12    # user speech this long during speculation cancels it
    duck_gain: float = 0.25
    bc_threshold: float = 0.5      # bc_det above this within the decision window -> backchannel
    interrupt_after: float = 0.40  # sustained non-backchannel speech this long (and VAP agrees) -> interruption
    floor_p: float = 0.35          # VAP p_now above this = the user is taking the floor (not laughing along)
    interrupt_max: float = 0.80    # sustained speech this long is an interruption whatever the models say
    interrupt_total: float = 1.0   # speech adding up to this across fragments ("yeah, I think we, um, it") -> interruption
    syntax_hold: bool = True       # Higgs STT says the words so far end mid-sentence -> don't answer yet
    hold_model: bool = False       # experimental: logistic model on VAP's trend 200 ms into a pause (hold_model.json)
    backchannels: bool = False     # say "mm-hm" while the user talks
    bc_prob: float = 0.60         # chosen on 24 AMI meetings, confirmed on 12 others (eval/listen_ami.py)
    bc_min_gap: float = 3.0
    bc_min_talk: float = 1.5
    bc_in_pauses_only: bool = False  # real listeners mostly backchannel over the speaker, not in pauses


@dataclass
class Turn:
    response_id: str = None
    item_id: str = None
    audio: bytearray = field(default_factory=bytearray)
    done: bool = False
    pos: int = 0                  # samples played
    discarded: bool = False
    speaker: object = None        # SentenceSpeaker when the reply is voiced by TTS 3
    text_done: bool = False
    tts_done: bool = False


class VAD:
    """Energy VAD against a noise floor taken by minimum statistics (a low percentile of the last few seconds of
    frame energies), so it adapts to a room that never goes quiet (babble, fans), plus a short hangover."""

    def __init__(self, margin_db=12.0, min_db=-48.0, hang=0.06, window_s=3.0, pct=10):
        self.margin, self.min_db, self.hang, self.pct = margin_db, min_db, hang, pct
        self.hist = np.full(int(window_s / TICK), -90.0)
        self.k = 0
        self.since_voice = 99.0

    def __call__(self, pcm):
        x = pcm.astype(np.float32) / 32768
        db = 20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9)
        self.hist[self.k % len(self.hist)] = db
        self.k += 1
        seen = self.hist[:min(self.k, len(self.hist))]
        floor = np.percentile(seen, self.pct)
        if self.k < 25:  # first 0.5 s: assume a quiet room unless shown otherwise, so speech from frame one counts
            floor = min(floor, -60.0)
        voiced = db > max(floor + self.margin, self.min_db)
        self.since_voice = 0.0 if voiced else self.since_voice + TICK
        return self.since_voice < self.hang, db


class Engine:
    def __init__(self, mode="mmhm", policy=None, instructions="", voice="chloe", signals=None,
                 bc_clips=None, log=None, session_extra=None, expressive=False):
        assert mode in ("mmhm", "server_vad", "semantic_vad")
        self.mode, self.p = mode, policy or Policy()
        self.expressive, self.voice = expressive, voice
        td = None if mode == "mmhm" else {"type": mode}
        if expressive:  # Realtime writes, TTS 3 speaks (laughs, sighs, emotions)
            instructions += EXPRESSIVE_RULES
        cfg = {"instructions": instructions, "output_modalities": ["text"] if expressive else ["audio"],
               "audio": {"input": {"turn_detection": td, "transcription": {"model": "higgs-stt-3.1"}},
                         "output": {"voice": voice}}}
        cfg.update(session_extra or {})
        self.session = Session(cfg)
        self.signals = signals
        self.bc_clips = bc_clips or []
        self.log = log if log is not None else []
        self.vad = VAD()
        self.t = 0.0
        self.state = "listening"
        self.turns = {}            # response_id -> Turn
        self.current = None        # Turn being spoken (or speculated)
        self.gain, self.target_gain = 1.0, 1.0
        self.user_speech = 0.0     # speech seconds since last commit
        self.silence = 0.0         # seconds since user stopped
        self.resume = 0.0          # speech seconds during speculation
        self.onset = None          # barge-in decision window: start time
        self.bc_max = 0.0
        self.bc_out = None         # (clip, pos) of our own backchannel being played
        self.last_bc = -99.0
        self.last_heard = -99.0    # when we last let a user backchannel pass
        self.frag_speech, self.frag_quiet = 0.0, 99.0  # user speech while Higgs talks, summed across short pauses
        self.talk_start = None
        self.sig = {}
        self._ev = 0
        self._discard_next = False  # a speculation was aborted before Higgs acknowledged it
        self.stage_direction = None
        self.on_transcript = None   # optional hook(text) -> stage direction: re-plan a speculative reply
        self.unfinished = False     # the latest committed words end mid-sentence (from Higgs input transcription)
        self.model_hold = None      # the VAP-trend model's verdict for the current pause (None = not decided yet)
        self.vap_hist = deque(maxlen=8)
        hm = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hold_model.json")
        self.hold_m = json.load(open(hm)) if os.path.exists(hm) else None

    async def __aenter__(self):
        await self.session.__aenter__()
        return self

    async def __aexit__(self, *a):
        await self.session.__aexit__(*a)

    def emit(self, kind, **kw):
        self.log.append({"t": round(self.t, 3), "wall": round(time.time(), 3), "kind": kind, **kw})

    # --- Higgs events -------------------------------------------------------------------------------
    def _drain_events(self):
        evs = self.session.events
        while self._ev < len(evs):
            _, ev = evs[self._ev]
            self._ev += 1
            typ = ev.get("type", "")
            if typ == "response.created":
                rid = ev["response"]["id"]
                self.turns[rid] = Turn(response_id=rid)
                if self.expressive:
                    t = self.turns[rid]
                    t.speaker = SentenceSpeaker(self.voice, t.audio, lambda t=t: setattr(t, "tts_done", True))
                if self._discard_next:
                    self._discard_next = False
                    self.turns[rid].discarded = True
                    if self.turns[rid].speaker:
                        self.turns[rid].speaker.cancel()
                    asyncio.ensure_future(self.session.send({"type": "response.cancel", "response_id": rid}))
                elif self.mode != "mmhm" or self.state == "speculating":
                    self.current = self.turns[rid]
                    if self.mode != "mmhm":
                        self.state = "playing"
                        self.emit("respond", rid=rid)
            elif typ == "response.output_item.added":
                rid = ev.get("response_id")
                if rid in self.turns and ev.get("item", {}).get("type") == "message":
                    self.turns[rid].item_id = ev["item"]["id"]
            elif typ == "response.output_text.delta" and self.expressive:
                rid = ev.get("response_id")
                if rid in self.turns and not self.turns[rid].discarded:
                    self.turns[rid].speaker.feed(ev.get("delta", ""))
            elif typ == "response.output_audio.delta":
                rid = ev.get("response_id")
                if rid in self.turns:
                    self.turns[rid].audio.extend(self.session.audio.get(rid, b"")[len(self.turns[rid].audio):])
            elif typ == "response.done":
                rid = ev["response"]["id"]
                if rid in self.turns:
                    t = self.turns[rid]
                    if self.expressive:
                        t.text_done = True
                        t.speaker.close()
                    else:
                        t.done = True
                        t.audio = bytearray(self.session.audio.get(rid, b""))
                    if t.discarded and t.item_id:
                        asyncio.ensure_future(self.session.send({"type": "conversation.item.delete",
                                                                 "item_id": t.item_id}))
                    self.emit("response_done", rid=rid, status=ev["response"].get("status"),
                              text=spoken(self.session.transcript(rid)), raw=self.session.transcript(rid))
            elif typ == "input_audio_buffer.speech_started" and self.mode != "mmhm":
                # Plain Higgs: the docs say to stop local playback on speech_started.
                if self.current and self.state == "playing":
                    self._stop_playback("speech_started")
            elif typ == "conversation.item.input_audio_transcription.completed":
                self.emit("user_text", text=ev.get("transcript"))
                if self.state == "speculating" and self.on_transcript:
                    direction = self.on_transcript(ev.get("transcript") or "")
                    if direction:
                        asyncio.ensure_future(self._replan(direction))
                if self.state == "speculating":
                    self.unfinished = unfinished(ev.get("transcript"))
                    if self.unfinished:
                        self.emit("syntax_hold", text=ev.get("transcript"))

    # --- actions ------------------------------------------------------------------------------------
    async def _commit_and_respond(self, why):
        await self.session.send({"type": "input_audio_buffer.commit"})
        if self.stage_direction:
            # Steering mid-conversation: a mid-session session.update of the instructions is outweighed by the
            # conversation history, and per-response `instructions` make Higgs drop the history entirely. A short
            # user-role stage direction right before the reply is followed and keeps the context.
            await self.session.send({"type": "conversation.item.create", "item": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": f"[stage direction for your next reply only, never "
                                                            f"mention it: {self.stage_direction}]"}]}})
        await self.session.send({"type": "response.create"})
        self.user_speech = 0.0
        self.state = "speculating"
        self.current = None
        self.unfinished = False
        self.model_hold = None
        self.emit("commit", why=why, p_now=round(self.sig.get("p_now", -1), 3), silence=round(self.silence, 3))

    def _stop_playback(self, why):
        t = self.current
        if t is None:
            return
        played_ms = int(t.pos / SR * 1000)
        if not t.done:
            asyncio.ensure_future(self.session.send({"type": "response.cancel"}))
        if self.expressive and t.speaker:
            # a text reply can't be truncated by audio time: replace it with the sentences actually heard
            t.speaker.cancel()
            heard = t.speaker.heard_text(t.pos)
            if t.item_id:
                asyncio.ensure_future(self._replace_reply(t.item_id, heard + " —"))
        elif t.item_id:
            asyncio.ensure_future(self.session.send({"type": "conversation.item.truncate", "item_id": t.item_id,
                                                     "content_index": 0, "audio_end_ms": played_ms}))
        self.emit("stop", why=why, rid=t.response_id, played_ms=played_ms,
                  total_ms=int(len(t.audio) / 2 / SR * 1000))
        self.current = None
        self.state = "listening"
        self.target_gain = self.gain = 1.0
        self.onset = None

    async def _replan(self, direction):
        """The transcript changed what the reply should be: drop the speculative one and generate it again."""
        t = self.current
        if t is None:
            self._discard_next = True
        else:
            t.discarded = True
            if t.speaker:
                t.speaker.cancel()
            if not t.done:
                await self.session.send({"type": "response.cancel"})
            elif t.item_id:
                await self.session.send({"type": "conversation.item.delete", "item_id": t.item_id})
        self.current = None
        await self.session.send({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": f"[stage direction for your next reply only, never mention it: "
                                                        f"{direction}]"}]}})
        await self.session.send({"type": "response.create"})
        self.emit("replan", direction=direction[:60])

    async def _replace_reply(self, item_id, text):
        await self.session.send({"type": "conversation.item.delete", "item_id": item_id})
        await self.session.send({"type": "conversation.item.create", "item": {
            "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}})

    # --- the clock ----------------------------------------------------------------------------------
    async def tick(self, mic):
        """mic: int16[480] at 24 kHz. Returns int16[480] for the speaker."""
        self._drain_events()
        for t in self.turns.values():  # an expressive turn is done when both the text and its speech are
            if t.speaker is not None and not t.done and t.text_done and t.tts_done:
                t.done = True
        await self.session.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(mic.tobytes()).decode()})
        speaking, db = self.vad(mic)
        out = np.zeros(N, np.int16)

        # agent audio this tick (before gain) -> also feeds channel 2 of the models
        if self.state == "playing" and self.current is not None:
            t = self.current
            a = np.frombuffer(bytes(t.audio[t.pos * 2:(t.pos + N) * 2]), np.int16)
            if len(a):
                if t.pos == 0:
                    self.emit("play", rid=t.response_id)
                out[:len(a)] = a
                t.pos += len(a)
            elif t.done:
                self.emit("play_end", rid=t.response_id, played_ms=int(t.pos / SR * 1000))
                self.current = None
                self.state = "listening"
                self.onset = None
                self.target_gain = self.gain = 1.0
                if self.mode == "mmhm":
                    await self.session.send({"type": "input_audio_buffer.clear"})
                    self.user_speech = 0.0
        if self.bc_out is not None:
            clip, pos = self.bc_out
            seg = clip[pos:pos + N]
            out[:len(seg)] = np.clip(out[:len(seg)].astype(np.int32) + seg, -32768, 32767).astype(np.int16)
            self.bc_out = (clip, pos + N) if pos + N < len(clip) else None

        if self.signals is not None and self.mode == "mmhm":
            u16 = resample_poly(mic.astype(np.float32) / 32768, 2, 3).astype(np.float32)
            if not speaking:
                # noise gate: in a room full of other voices the turn-taking model hears "speech" in the user's
                # channel all the time and never lets them finish; show it silence between the user's words
                u16 *= 0.05
            s16 = resample_poly(out.astype(np.float32) / 32768, 2, 3).astype(np.float32)
            self.sig = self.signals.push(u16, s16)

        if self.mode == "mmhm":
            self.vap_hist.append((self.sig.get("p_now", 0.0), self.sig.get("p_future", 0.0)))
            await self._policy(speaking)

        # gain ramps (ducking) are applied to agent audio only; 10 ms-ish smoothing per tick
        self.gain += (self.target_gain - self.gain) * 0.6
        if self.gain != 1.0:
            out = (out.astype(np.float32) * self.gain).astype(np.int16)
        self.t += TICK
        return out

    async def _policy(self, speaking):
        p, now = self.p, self.t
        p_now = self.sig.get("p_now", 0.0)
        if speaking:
            self.silence = 0.0
            if self.talk_start is None:
                self.talk_start = now
        else:
            self.silence += TICK
            if self.silence > 1.5:
                self.talk_start = None

        if self.state == "listening":
            if speaking:
                self.user_speech += TICK
            elif self.user_speech >= p.min_turn:
                if self.silence >= p.min_silence and p_now < p.th_start:
                    await self._commit_and_respond("gate")
                elif self.silence >= p.give_up or (self.silence >= p.max_wait and p_now < p.hold_p):
                    await self._commit_and_respond("timeout")
            # our own listening noises
            if (p.backchannels and self.bc_clips and self.bc_out is None and self.talk_start is not None
                    and now - self.talk_start > p.bc_min_talk and now - self.last_bc > p.bc_min_gap
                    and self.sig.get("p_bc", 0) > p.bc_prob
                    and (not p.bc_in_pauses_only or 0.08 <= self.silence <= 0.35)):
                clip = self.bc_clips[int(now * 10) % len(self.bc_clips)]
                self.bc_out = (clip, 0)
                self.last_bc = now
                self.emit("backchannel", p_bc=round(self.sig["p_bc"], 3))

        elif self.state == "speculating":
            if speaking:
                self.resume += TICK
                if self.resume >= p.resume_cancel:
                    # they weren't done: drop the speculative answer, keep listening
                    if self.current is None:
                        self._discard_next = True
                    else:
                        self.current.discarded = True
                        if self.current.speaker:
                            self.current.speaker.cancel()
                        if not self.current.done:
                            await self.session.send({"type": "response.cancel"})
                        elif self.current.item_id:
                            await self.session.send({"type": "conversation.item.delete",
                                                     "item_id": self.current.item_id})
                    self.emit("abort", rid=self.current and self.current.response_id)
                    self.state, self.current, self.resume = "listening", None, 0.0
                    self.user_speech = p.min_turn  # they've already said something this turn
            else:
                self.resume = 0.0
                ready = self.current is not None and len(self.current.audio) > 0
                if p.syntax_hold and self.unfinished and self.silence < p.give_up:
                    ready = False
                if p.hold_model and self.hold_m and self.model_hold is None and self.silence >= 0.2 \
                        and len(self.vap_hist) >= 6:
                    m = self.hold_m
                    (n1, f1), (n2, f2) = self.vap_hist[-6], self.vap_hist[-1]  # 100 ms apart
                    z = (np.array([n1, n2, f1, f2]) - m["mean"]) / m["scale"]
                    p_hold = 1 / (1 + np.exp(-(float(np.dot(z, m["coef"])) + m["intercept"])))
                    self.model_hold = p_hold > m["threshold"]
                    if self.model_hold:
                        self.emit("model_hold", p_hold=round(p_hold, 3))
                if p.hold_model and self.model_hold and self.silence < p.give_up:
                    ready = False
                if ready and (p_now < p.th_play or self.silence >= p.give_up
                              or (self.silence >= p.max_wait and p_now < p.hold_p)):
                    self.state = "playing"
                    self.frag_speech, self.frag_quiet = 0.0, 99.0
                    await self.session.send({"type": "input_audio_buffer.clear"})
                    self.emit("respond", rid=self.current.response_id, p_now=round(p_now, 3),
                              silence=round(self.silence, 3))

        elif self.state == "playing":
            p_bc = self.sig.get("p_bcdet", 0.0)
            # hesitant interruptions come in fragments; add them up until a second of quiet
            self.frag_quiet = 0.0 if speaking else self.frag_quiet + TICK
            self.frag_speech = (self.frag_speech + TICK) if speaking else (0.0 if self.frag_quiet > 1.0 else self.frag_speech)
            if self.frag_speech >= p.interrupt_total and speaking:
                self.emit("decide", dur=round(self.frag_speech, 3), why="fragments")
                self._stop_playback("interrupt")
                self.user_speech = self.frag_speech
                self.frag_speech = 0.0
                return
            if self.onset is None and speaking:
                self.onset, self.bc_max = now, 0.0
                self.target_gain = p.duck_gain
                self.emit("duck")
            if self.onset is not None:
                self.bc_max = max(self.bc_max, p_bc)
                dur = now - self.onset
                p_floor = self.sig.get("p_now", 0.0)  # VAP: is the user taking the floor?
                # right after a backchannel, a second "mhm" is likely too: give it a longer look
                patience = p.interrupt_after + (0.25 if now - self.last_heard < 1.0 else 0.0)
                if self.bc_max >= p.bc_threshold and dur < 1.0:
                    if not speaking:  # the backchannel is over: carry on at full volume, forget it
                        self.emit("backchannel_heard", p=round(self.bc_max, 3), dur=round(dur, 3))
                        self.onset, self.target_gain, self.last_heard = None, 1.0, now
                        await self.session.send({"type": "input_audio_buffer.clear"})
                elif speaking and ((dur >= patience and p_floor > p.floor_p) or dur >= p.interrupt_max):
                    self.emit("decide", dur=round(dur, 3), bc_max=round(self.bc_max, 3), p_now=round(p_floor, 3))
                    self._stop_playback("interrupt")
                    self.user_speech = dur
                elif not speaking and self.silence > 0.25:
                    # a short non-backchannel noise (cough, laugh): resume
                    self.emit("noise_ignored", dur=round(dur, 3), bc_max=round(self.bc_max, 3), p_now=round(p_floor, 3))
                    self.onset, self.target_gain = None, 1.0
                    await self.session.send({"type": "input_audio_buffer.clear"})
