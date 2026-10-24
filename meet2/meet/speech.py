"""Speech output: utterances synthesized sentence by sentence, and one supervised worker that plays them.

Design rules (from the Meet 1 runtime audit):
  * output runs in its own task, separate from listening; a send error is logged and the loop carries on,
  * an utterance's audio only ever grows: once a sample is released it is never replaced (a sentence whose speech
    fails is retried only if none of it has played; otherwise the utterance ends cleanly there),
  * generated, sent and heard are separate: "heard" is the transport's playback-finished report when it has one,
    else sent time + the browser's jitter buffer. The floor is released on heard, never on sent.
Sentence-by-sentence synthesis gives exact sentence boundaries, so an interrupted reply knows what was heard.
"""
from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .transport.base import FRAME_SAMPLES

SR = 24000
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> tuple[list[str], str]:
    """(complete sentences, the unfinished rest)."""
    parts = _SENT_END.split(text)
    if len(parts) == 1:
        return [], text
    return [p.strip() for p in parts[:-1] if p.strip()], parts[-1]


@dataclass
class Sentence:
    text: str
    audio: bytearray = field(default_factory=bytearray)
    done: bool = False
    failed: bool = False

    @property
    def samples(self) -> int:
        return len(self.audio) // 2


class Utterance:
    """What the AI is about to say or is saying. Text arrives with feed() (streaming from the writer) and ends with
    close(); speech for each complete sentence is synthesized in order by the Synthesizer."""

    _n = 0

    def __init__(self, kind: str = "reply", *, meta: dict | None = None):
        Utterance._n += 1
        self.id = f"ai{Utterance._n}"
        self.kind = kind                     # "reply" | "correction"
        self.meta = meta or {}
        self.sentences: list[Sentence] = []
        self._rest = ""
        self.text_done = False
        self.synth_done = False
        self.cancelled = False
        self.interrupted = False
        self.sent = 0                        # samples handed to the transport
        self.started_at: float | None = None # monotonic time of the first frame sent
        self.finished_at: float | None = None
        self.heard = asyncio.Event()
        self.changed = asyncio.Event()
        self.created = time.monotonic()

    # ------------------------------------------------------------ text
    def feed(self, delta: str) -> None:
        if self.text_done or self.cancelled:
            return
        sents, self._rest = split_sentences(self._rest + delta)
        for s in sents:
            self.sentences.append(Sentence(s))
        if sents:
            self.changed.set()

    def close(self) -> None:
        if self.text_done:
            return
        if self._rest.strip():
            self.sentences.append(Sentence(self._rest.strip()))
        self._rest = ""
        self.text_done = True
        self.changed.set()

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences) + ((" " + self._rest) if self._rest else "")

    # ------------------------------------------------------------ audio
    @property
    def total_samples(self) -> int:
        return sum(s.samples for s in self.sentences)

    def available(self) -> int:
        """Samples that may be played: every sentence in order, up to the first one still being made."""
        n = 0
        for s in self.sentences:
            n += s.samples
            if not s.done:
                break
        return n

    def read(self, start: int, n: int) -> bytes:
        out, pos = bytearray(), 0
        for s in self.sentences:
            a, b = pos, pos + s.samples
            if b > start and a < start + n:
                lo, hi = max(start, a) - a, min(start + n, b) - a
                out += s.audio[lo * 2:hi * 2]
            pos = b
            if pos >= start + n:
                break
        return bytes(out)

    @property
    def exhausted(self) -> bool:
        """Everything that will ever be said has been sent."""
        return self.synth_done and self.sent >= self.total_samples

    def heard_text(self, heard_samples: int) -> str:
        """The words a listener had heard when the utterance stopped at heard_samples."""
        out, pos = [], 0
        for s in self.sentences:
            if pos + s.samples <= heard_samples:
                out.append(s.text)
            elif pos < heard_samples and s.samples:
                ws = s.text.split()
                k = int(len(ws) * (heard_samples - pos) / s.samples)
                if k:
                    out.append(" ".join(ws[:k]) + "…")
                break
            else:
                break
            pos += s.samples
        return " ".join(out)

    def cancel(self) -> None:
        self.cancelled = True
        self.changed.set()


class Synthesizer:
    """Turns an utterance's sentences into speech in order (TTS streams faster than real time, so sentence n+1 is
    ready before sentence n has finished playing)."""

    def __init__(self, tts: Callable, *, on_error: Callable[[str], None] | None = None, retries: int = 1):
        self.tts, self.on_error, self.retries = tts, on_error or (lambda e: None), retries

    async def run(self, u: Utterance) -> None:
        i = 0
        try:
            while not u.cancelled:
                if i < len(u.sentences):
                    s = u.sentences[i]
                    await self._speak(u, s)
                    if s.failed and s.samples:       # part of it is out: never rewind, end the utterance here
                        u.sentences[i + 1:] = []
                        break
                    i += 1
                    continue
                if u.text_done:
                    break
                u.changed.clear()
                await u.changed.wait()
        finally:
            u.synth_done = True
            u.changed.set()

    async def _speak(self, u: Utterance, s: Sentence) -> None:
        for attempt in range(self.retries + 1):
            try:
                async for chunk in self.tts(s.text):
                    if u.cancelled:
                        return
                    s.audio += chunk
                s.done = True
                return
            except Exception as e:  # noqa: BLE001
                self.on_error(f"speech: {e}"[:200])
                if s.samples:                        # some of it already exists: keep it, stop here
                    s.failed = s.done = True
                    return
                await asyncio.sleep(0.1 * (attempt + 1))
        s.failed = s.done = True                     # nothing produced: skip the sentence


def level_db(pcm: bytes) -> float | None:
    if not pcm:
        return None
    x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(x * x)))
    return 20 * np.log10(rms) if rms > 1e-4 else None


class OutputWorker:
    """Plays one utterance at a time to the transport in real-time 20 ms frames. Never blocks the caller."""

    def __init__(self, transport, *, log: Callable[[dict], None] = lambda e: None,
                 on_level: Callable[[float | None], None] = lambda db: None):
        self.transport, self.log, self.on_level = transport, log, on_level
        self.current: Utterance | None = None
        self.task: asyncio.Task | None = None
        self.wake = asyncio.Event()
        self._finished_cb: Callable[[Utterance], None] | None = None
        self._playback_waiter: asyncio.Future | None = None
        self.max_late = 0.0
        self.errors = 0

    def start(self) -> None:
        if self.task is None or self.task.done():
            self.task = asyncio.ensure_future(self._supervise())
        self.transport.on_playback_finished(self._playback_finished)

    async def stop(self) -> None:
        if self.current:
            self.current.cancel()
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self.task = None

    def play(self, u: Utterance, on_heard: Callable[[Utterance], None]) -> None:
        if self.current and not self.current.cancelled and not self.current.heard.is_set():
            raise RuntimeError("already playing")
        self.current, self._finished_cb = u, on_heard
        self.wake.set()

    async def interrupt(self) -> int:
        """Stop now. Returns the samples the listeners had heard."""
        u = self.current
        if u is None:
            return 0
        heard = self.heard_samples()
        u.cancel()
        u.interrupted = True
        try:
            await self.transport.clear_ai_audio()
        except Exception as e:  # noqa: BLE001
            self.log({"type": "ai_error", "error": f"clear: {e}"[:200]})
        self.on_level(None)
        self._finish(u)
        return heard

    def heard_samples(self) -> int:
        u = self.current
        if u is None or u.started_at is None:
            return 0
        lag = int(self.transport.playout_delay() * SR)
        elapsed = int((time.monotonic() - u.started_at) * SR)
        return max(0, min(u.sent, elapsed) - lag)

    @property
    def speaking(self) -> bool:
        u = self.current
        return u is not None and not u.cancelled and not u.heard.is_set()

    def _playback_finished(self) -> None:
        if self._playback_waiter is not None and not self._playback_waiter.done():
            self._playback_waiter.set_result(True)

    def _finish(self, u: Utterance) -> None:
        if u.heard.is_set():
            return
        u.finished_at = time.monotonic()
        u.heard.set()
        cb = self._finished_cb
        if self.current is u:
            self.current = None
        if cb:
            try:
                cb(u)
            except Exception as e:  # noqa: BLE001
                self.log({"type": "ai_error", "error": f"after speech: {e}"[:200]})

    async def _supervise(self) -> None:
        while True:
            try:
                await self._loop()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001  (never die silently: log, reset, carry on)
                self.errors += 1
                self.log({"type": "ai_error", "error": f"output loop: {e!r}"[:200]})
                if self.current:
                    self._finish(self.current)
                await asyncio.sleep(0.1)

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            u = self.current
            if u is None or u.cancelled or u.heard.is_set():
                self.wake.clear()
                await self.wake.wait()
                continue
            t0 = loop.time()
            k = 0
            while not u.cancelled:
                if u.sent + FRAME_SAMPLES <= u.available() or (u.synth_done and u.sent < u.total_samples):
                    frame = u.read(u.sent, FRAME_SAMPLES)
                    if len(frame) < FRAME_SAMPLES * 2:
                        frame += bytes(FRAME_SAMPLES * 2 - len(frame))
                    if u.started_at is None:
                        u.started_at = time.monotonic()
                        self.log({"type": "speech_start", "id": u.id, "kind": u.kind,
                                  "wait_s": round(u.started_at - u.created, 3)})
                    try:
                        await self.transport.send_ai_audio(frame)
                    except Exception as e:  # noqa: BLE001
                        self.errors += 1
                        self.log({"type": "ai_error", "error": f"send: {e!r}"[:200]})
                    u.sent += FRAME_SAMPLES
                    self.on_level(level_db(frame))
                    k += 1
                    late = loop.time() - (t0 + k * 0.02)
                    self.max_late = max(self.max_late, late)
                    if late < -0.001:
                        await asyncio.sleep(-late)
                    elif late > 0.2:                 # fell far behind (a stall): don't burst to catch up
                        t0, k = loop.time(), 0
                    continue
                if u.exhausted:
                    break
                # waiting for speech (start, or a slow sentence): restart the clock when it comes
                self.on_level(None)
                u.changed.clear()
                try:
                    await asyncio.wait_for(u.changed.wait(), 0.05)
                except asyncio.TimeoutError:
                    pass
                t0, k = loop.time(), 0
            self.on_level(None)
            if u.cancelled:
                continue
            try:
                await self.transport.end_ai_segment()
            except Exception as e:  # noqa: BLE001
                self.log({"type": "ai_error", "error": f"end segment: {e!r}"[:200]})
            await self._wait_heard(u)
            self._finish(u)

    async def _wait_heard(self, u: Utterance) -> None:
        delay = self.transport.playout_delay()
        remaining = 0.0
        if u.started_at is not None:                 # frames are paced, so sent ~= elapsed; add the buffer
            remaining = max(0.0, u.started_at + u.sent / SR - time.monotonic()) + delay
        self._playback_waiter = asyncio.get_running_loop().create_future()
        try:
            await asyncio.wait_for(asyncio.shield(self._playback_waiter), remaining + (2.0 if delay > 0.5 else 0.0))
        except asyncio.TimeoutError:
            pass
        finally:
            self._playback_waiter = None
