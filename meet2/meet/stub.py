"""A stand-in AI for UI work and browser tests: no model calls.

It beeps on slot 0 when someone joins, echoes typed chat, and reports each person's speech as a transcript line
("[speech 1.2 s]") using the same audio front-end as the real agent.
"""
from __future__ import annotations

import asyncio

import numpy as np

from .audio.frontend import SpeakerFrontEnd
from .transport.base import FRAME_SAMPLES

SR = 24000


def beep(seconds: float = 0.3, hz: float = 660.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    env = np.minimum(1, np.minimum(t, seconds - t) / 0.02)
    return (np.sin(2 * np.pi * hz * t) * env * 0.3 * 32767).astype(np.int16)


class StubAgent:
    def __init__(self, room):
        self.room = room
        self.fronts: dict[int, SpeakerFrontEnd] = {}
        self.queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.n = 0

    async def start(self) -> None:
        if self.task is None:
            self.task = asyncio.ensure_future(self._play())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            self.task = None

    async def _play(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            pcm = await self.queue.get()
            self.room.event({"type": "ai_state", "state": "speaking"}, log=False)
            t0 = loop.time()
            for i, k in enumerate(range(0, len(pcm), FRAME_SAMPLES)):
                frame = pcm[k:k + FRAME_SAMPLES]
                if len(frame) < FRAME_SAMPLES:
                    frame = np.pad(frame, (0, FRAME_SAMPLES - len(frame)))
                await self.room.transport.send_ai_audio(frame.tobytes())
                await asyncio.sleep(max(0, t0 + (i + 1) * 0.02 - loop.time()))
            self.room.event({"type": "ai_state", "state": "idle"}, log=False)

    def feed(self, slot: int, pcm: bytes) -> None:
        fe = self.fronts.get(slot)
        if fe is None:
            fe = self.fronts[slot] = SpeakerFrontEnd()
        for ev in fe.push(np.frombuffer(pcm, np.int16)):
            if ev.kind == "end":
                person = self.room.people.get(slot)
                self.n += 1
                self.room.event({"type": "line", "id": f"stub{self.n}", "slot": slot,
                                 "speaker": person.name if person else "?", "text": f"[speech {ev.duration:.1f} s]",
                                 "final": True})

    def on_roster(self, people: list[dict]) -> None:
        if len(people) > len(self.fronts):
            self.queue.put_nowait(beep())

    def on_chat(self, slot: int, name: str, text: str) -> None:
        self.room.event({"type": "chat", "slot": 0, "name": self.room.brief.ai, "text": f"(stub) {name} said: {text}"})

    async def set_avatar(self, mode: str) -> None:
        self.room.avatar = mode

    def set_mode(self, mode: str) -> None:
        pass

    def on_kb_changed(self) -> None:
        pass
