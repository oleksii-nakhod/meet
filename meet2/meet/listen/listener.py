"""One person in the meeting: their audio front-end, their own transcription stream, their turns (F3), and the
barge-in probe while the AI talks (F2).

Audio is cut into pieces at clause pauses (~0.28 s) and turns at longer pauses (~0.45 s). Each piece is committed to
this person's Higgs STT session as soon as it ends (~0.3 s to its words), so a claim can be fact-checked while the
person is still talking; the pieces of a turn join into one transcript line. Speaker identity is exact: every
person has their own microphone stream.

While the AI is speaking, a person who starts talking is transcribed every ~0.3 s on a separate probe session (the
main stream is left alone) and each transcript gets a barge-in verdict: fillers never stop the AI, a stop request or
a real interruption does.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

import numpy as np

from ..audio.frontend import FRAME_S, SpeakerFrontEnd
from ..boson import STTSession
from .words import bargein

STT_FAILED = re.compile(r"transcription is not available", re.I)


@dataclass
class Turn:
    id: str
    slot: int
    speaker: str
    started: float
    parts: list[str] = field(default_factory=list)
    hold_until: float = 0.0          # an unfinished turn waits this long for the speaker to go on
    during_correction: bool = False  # said while the AI was correcting: held, then handled with this note

    @property
    def text(self) -> str:
        out = ""
        for p in self.parts:
            out = join_pieces(out, p) if out else p
        return out.strip()


def join_pieces(first: str, rest: str) -> str:
    """STT ends every piece with punctuation. If the first stopped mid-sentence ("what does the lease say about
    having a?" + "Dog.") it was one sentence: drop the stop and continue in lower case."""
    from .words import DANGLING
    first, rest = first.rstrip(), rest.strip()
    m = re.search(r"([A-Za-z']+)[.?!,]+$", first)
    if m and m.group(1).lower() in DANGLING and rest[:1].isupper() and not re.match(r"I\b", rest):
        return f"{first[:m.end(1)]} {rest[:1].lower()}{rest[1:]}"
    return f"{first} {rest}"


class Listener:
    _n = 0

    def __init__(self, agent, slot: int, name: str):
        s = agent.settings
        self.agent, self.slot, self.name = agent, slot, name
        self.fe = SpeakerFrontEnd(clause_silence=s.clause_silence_s, line_silence=s.line_silence_s)
        self.stt: STTSession = agent.models.open_stt()
        self.q: asyncio.Queue = asyncio.Queue()
        self.results: asyncio.Queue = asyncio.Queue()
        self.piece: list[np.ndarray] = []       # audio since the last commit (for the local fallback)
        self.segment: list[np.ndarray] = []     # audio of the current segment (for barge-in probes)
        self.turn: Turn | None = None
        self.last_voice = 0.0
        self.probe_task: asyncio.Task | None = None
        self.tasks = [asyncio.ensure_future(self._sender()), asyncio.ensure_future(self._reader())]
        self.closed = False

    # ------------------------------------------------------------ per frame (sync, cheap)
    def feed(self, frame: np.ndarray) -> None:
        self.fe.set_ai_level(self.agent.echo_level())
        out = self.fe.push(frame)
        kinds = [e.kind for e in out.events]
        if "start" in kinds:
            self.segment = []
            self._on_start()
        voiced = bool(self.fe.recent and self.fe.recent[-1])
        for a in out.audio:
            self.segment.append(a)
            self.q.put_nowait(("audio", (a, voiced or "start" in kinds)))
        if self.fe.speaking:
            self.last_voice = time.monotonic()
        for e in out.events:
            if e.kind == "clause":
                self.q.put_nowait(("commit", "clause"))
            elif e.kind == "end":
                self.q.put_nowait(("commit", "end"))
            elif e.kind == "blip":
                self.q.put_nowait(("clear", None))

    def _on_start(self) -> None:
        now = time.monotonic()
        if self.turn is None or now > self.turn.hold_until or not self.turn.hold_until:
            if self.turn is not None and self.turn.hold_until:          # a held turn that nobody continued
                self.agent.finish_turn(self, self.turn, force=True)
            Listener._n += 1
            self.turn = Turn(f"L{Listener._n}", self.slot, self.name, now)
        else:
            self.turn.hold_until = 0.0                                   # they went on: same line
        self.agent.on_speech_start(self)
        if (self.agent.ai_talking() and getattr(self.agent, "mode", "smart") == "smart"
                and (self.probe_task is None or self.probe_task.done())):
            self.probe_task = asyncio.ensure_future(self._probe())

    # ------------------------------------------------------------ transcription stream (ordered)
    async def _sender(self) -> None:
        voiced = 0.0                     # voiced seconds since the last commit: silence alone is never committed
        while True:
            kind, val = await self.q.get()
            audio: list[np.ndarray] = []
            try:
                if kind == "audio":
                    frame, is_voiced = val
                    self.piece.append(frame)
                    voiced += FRAME_S if is_voiced else 0.0
                    await self.stt.append(frame.tobytes())
                elif kind == "clear":
                    self.piece, voiced = [], 0.0
                    await self.stt.clear()
                    await self.results.put(("end", None, []))
                elif kind == "commit":
                    audio, self.piece = self.piece, []
                    if voiced >= 0.12:
                        fut = await self.stt.commit_nowait()
                        await self.results.put((val, fut, audio))
                    else:
                        if audio:
                            await self.stt.clear()   # only silence since the last piece: drop it
                        if val == "end":
                            await self.results.put(("end", None, []))
                    voiced = 0.0
            except Exception as e:  # noqa: BLE001
                self.agent.log({"type": "ai_error", "error": f"stt send ({self.name}): {e}"[:200]})
                if kind == "commit":
                    await self.results.put((val, None, audio))

    async def _reader(self) -> None:
        while True:
            kind, fut, audio = await self.results.get()
            text = ""
            if fut is not None or audio:
                text = await self._transcript(fut, audio)
            turn = self.turn
            self.agent.log({"type": "piece", "slot": self.slot, "kind": kind, "text": text,
                            "audio_s": round(len(audio) * FRAME_S, 2), "turn": turn.id if turn else None})
            if turn is None:
                continue
            if text:
                turn.parts.append(text)
                self.agent.on_piece(self, turn, text, final=kind == "end")
            if kind == "end":
                self.agent.finish_turn(self, turn)

    async def _transcript(self, fut, audio) -> str:
        t0 = time.monotonic()
        failed = None
        text = ""
        if fut is not None:
            try:
                text = await STTSession.result(fut, timeout=2.5)
                if STT_FAILED.search(text):
                    failed = text
            except Exception as e:  # noqa: BLE001
                failed = repr(e)
        else:
            failed = "no session"
        if failed and audio:
            try:
                text = await self.agent.models.transcribe(np.concatenate(audio), timeout=6)
                self.agent.log({"type": "stt_fallback", "slot": self.slot, "why": failed[:120], "text": text,
                                "s": round(time.monotonic() - t0, 2)})
            except Exception as e:  # noqa: BLE001
                self.agent.log({"type": "ai_error", "error": f"transcription ({self.name}): {e}"[:200]})
                return ""
        return self.guard(text)

    @staticmethod
    def guard(text: str) -> str:
        """Drop what isn't speech: empty, no letters, or a lone hum transcribed in another script."""
        t = (text or "").strip()
        if not re.search(r"[A-Za-z]", t):
            return ""
        return t

    # ------------------------------------------------------------ barge-in probe
    async def _probe(self) -> None:
        a, lex = self.agent, self.agent.lex
        verdict, text, last_voiced = "unclear", "", 0.0
        try:
            while self.fe.active and a.ai_talking():
                voiced = self.fe.voiced
                if voiced < lex.probe_first_s or voiced - last_voiced < 0.1:   # only when there is new speech
                    await asyncio.sleep(0.05)
                    continue
                last_voiced = voiced
                t0 = time.monotonic()
                text = await a.probe_transcribe(np.concatenate(self.segment))
                verdict = bargein(text, voiced, lex, a.ai_name)
                a.log({"type": "bargein_probe", "slot": self.slot, "text": text, "verdict": verdict,
                       "voiced_s": round(voiced, 2), "probe_s": round(time.monotonic() - t0, 2)})
                if verdict in ("stop", "interrupt") and a.on_bargein(self, verdict, text):
                    return                      # it stopped; otherwise (a correction's guard) keep listening
                await asyncio.sleep(max(0.0, lex.probe_every_s - (time.monotonic() - t0)))
            if verdict == "filler" or (verdict == "unclear" and text):
                a.on_bargein(self, "filler", text)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            a.log({"type": "ai_error", "error": f"barge-in probe: {e!r}"[:200]})

    # ------------------------------------------------------------ state for the floor
    @property
    def speaking(self) -> bool:
        return self.fe.speaking

    @property
    def in_turn(self) -> bool:
        return self.fe.active

    @property
    def pause(self) -> float:
        return self.fe.silence if self.fe.active else 99.0

    async def close(self) -> None:
        self.closed = True
        for t in self.tasks + ([self.probe_task] if self.probe_task else []):
            t.cancel()
        await self.stt.close()
