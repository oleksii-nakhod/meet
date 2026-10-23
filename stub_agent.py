"""Placeholder AI participant, used when meet/agent.py (the real Higgs) is missing. It proves the plumbing: a 440 Hz
beep on slot 0 when the room starts, a chat echo, and a few chat commands the call-layer test uses:
  /beep  /flush  /gain <0..1>  /line  /lab  /say <text>  /avatar (streams $MEET_STUB_FMP4, a fragmented MP4, if set)"""
import asyncio
import os

import numpy as np

FRAME = 480  # samples per 20 ms at 24 kHz


class HiggsAgent:
    def __init__(self, room):
        self.room = room
        self.task = None
        self.tasks = set()

    def _spawn(self, coro):
        t = asyncio.ensure_future(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)
        return t

    async def start(self):
        self._spawn(self.beep())

    async def stop(self):
        for t in list(self.tasks):
            t.cancel()

    def feed(self, slot, pcm):
        pass

    def on_roster(self, people):
        pass

    def on_chat(self, name, text):
        cmd, _, arg = text.partition(" ")
        r = self.room
        if cmd == "/beep":
            self._spawn(self.beep())
        elif cmd == "/flush":
            self._spawn(r.ai_flush())
        elif cmd == "/gain":
            self._spawn(r.ai_gain(float(arg or 1)))
        elif cmd == "/line":
            self._spawn(r.event({"type": "line", "slot": 1, "name": name, "text": arg or "Higgs, what time is it?",
                                 "decision": {"speak": True, "why": "asked me"}}))
        elif cmd == "/lab":
            self._spawn(self.lab())
        elif cmd == "/say":
            self._spawn(r.event({"type": "ai_say", "text": arg or "Hello, I'm only a stub."}))
        elif cmd == "/avatar":
            self._spawn(self.avatar())
        else:
            self._spawn(r.event({"type": "chat", "from": f"{r.brief['ai']} (stub)", "slot": 0, "text": f"heard {text}"}))

    def set_mode(self, mode):
        pass

    def set_avatar(self, on):
        pass

    async def beep(self, secs=0.6, hz=440):
        """A short tone through send_ai_audio, paced at 20 ms against the clock (not sleep-drift)."""
        t = np.arange(int(secs * 24000)) / 24000
        x = (0.25 * np.sin(2 * np.pi * hz * t) * np.minimum(1, np.minimum(t, secs - t) / 0.02) * 32767).astype("<i2")
        await self.room.ai_status("speaking")
        await self.room.event({"type": "ai_say", "text": "(beep)"})
        t0 = asyncio.get_running_loop().time()
        for i in range(0, len(x) - FRAME + 1, FRAME):
            await self.room.send_ai_audio(x[i:i + FRAME].tobytes())
            await asyncio.sleep(max(0, t0 + (i // FRAME + 1) * 0.02 - asyncio.get_running_loop().time()))
        await self.room.ai_status("listening")

    async def lab(self):
        for k in range(10):
            for p in self.room.people():
                await self.room.event({"type": "lab", "slot": p["slot"], "p_now": (k + p["slot"]) % 10 / 10, "p_bc": 0.3})
            await asyncio.sleep(0.1)

    async def avatar(self):
        path = os.environ.get("MEET_STUB_FMP4")
        if not path or not os.path.exists(path):
            return
        data = open(path, "rb").read()
        await self.room.avatar_start('video/mp4; codecs="avc1.42E01E"')
        for i in range(0, len(data), 16384):
            await self.room.send_avatar_chunk(data[i:i + 16384])
            await asyncio.sleep(0.02)
        await self.room.avatar_stop()
