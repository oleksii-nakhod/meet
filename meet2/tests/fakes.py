"""A scripted stand-in for Boson, so the whole agent pipeline runs without network calls.

Test "speech" is a tone whose pitch encodes a phrase: phrase i is a 400 + 40*i Hz tone. The fake STT finds the
dominant pitch of the audio it was given and returns that phrase, so the real front-end, segmenting, per-person
transcription stream, barge-in probes, judge, writer, floor and output all run for real.
"""
from __future__ import annotations

import asyncio
import json
import re

import numpy as np

SR = 24000


class Phrases:
    def __init__(self, phrases: list[str]):
        self.phrases = phrases

    def audio(self, i: int, seconds: float | None = None, level: float = 6000) -> np.ndarray:
        n = int(SR * (seconds or max(0.6, 0.07 * len(self.phrases[i].split()) + 0.4)))
        t = np.arange(n) / SR
        f = 400 + 40 * i
        x = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2 * f * t)
        env = np.minimum(1, np.minimum(t, t[-1] - t) / 0.03)
        return (x * env * level / 1.4).astype(np.int16)

    def decode(self, pcm: np.ndarray) -> str:
        if len(pcm) < 2400:
            return ""
        x = pcm.astype(np.float32)
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), 1 / SR)
        band = (freqs > 380) & (freqs < 4000)
        f = freqs[band][np.argmax(spec[band])]
        i = int(round((f - 400) / 40))
        return self.phrases[i] if 0 <= i < len(self.phrases) and spec[band].max() > 1e5 else ""


class FakeSTT:
    def __init__(self, models: "FakeModels"):
        self.m = models
        self.buf: list[bytes] = []
        self.ready = asyncio.Event()
        self.ready.set()

    async def append(self, pcm: bytes) -> None:
        self.buf.append(pcm)

    async def commit_nowait(self, timeout: float = 6.0):
        audio = np.frombuffer(b"".join(self.buf), np.int16)
        self.buf = []
        fut = asyncio.get_running_loop().create_future()

        async def later():
            await asyncio.sleep(self.m.stt_delay)
            fut.set_result(self.m.phrases.decode(audio))
        asyncio.ensure_future(later())
        self.m.calls["stt"] += 1
        return fut

    @staticmethod
    async def result(fut, timeout: float = 6.0) -> str:
        return await asyncio.wait_for(fut, timeout)

    async def commit(self, timeout: float = 6.0) -> str:
        return await self.result(await self.commit_nowait())

    async def clear(self) -> None:
        self.buf = []

    async def close(self) -> None:
        pass


class FakeModels:
    def __init__(self, phrases: Phrases, *, reply: str = "Sure. The rent is due on the first of the month.",
                 contradictions: dict[str, dict] | None = None, tts_seconds_per_word: float = 0.12,
                 stt_delay: float = 0.15):
        self.phrases, self.reply = phrases, reply
        self.contradictions = contradictions or {}
        self.tts_seconds_per_word, self.stt_delay = tts_seconds_per_word, stt_delay
        self.calls = {"stt": 0, "judge": 0, "check": 0, "write": 0, "tts": 0, "notes": 0}

    def open_stt(self):
        return FakeSTT(self)

    async def transcribe(self, pcm, timeout: float = 10.0) -> str:
        return self.phrases.decode(np.asarray(pcm))

    async def chat(self, system: str, user: str, **kw) -> str:
        await asyncio.sleep(0.05)
        latest = user.rsplit("tatement:\n", 1)[-1] if "LATEST statement" in user else user.rsplit("utterance:\n", 1)[-1]
        latest = latest.split("\n")[0]
        if "Analyse the latest utterance" in system:
            self.calls["judge"] += 1
            asks_ai = "higgs" in latest.lower() or (latest.rstrip().endswith("?") and "1:1" in system) or "one-to-one" in system
            if re.search(r":\s*(Mei|Alex|Jordan),", latest):
                asks_ai = False
            return json.dumps({"to": "me" if asks_ai else "person", "ask": latest.rstrip().endswith("?"),
                               "answered": False, "can": True, "reply": asks_ai, "wrong": False})
        if "You check one statement" in system:
            self.calls["check"] += 1
            for needle, verdict in self.contradictions.items():
                if needle in latest:
                    return json.dumps(verdict)
            return json.dumps({"contradicts": False})
        if "live notes" in system:
            self.calls["notes"] += 1
            return json.dumps({"topic": "rent", "decisions": [], "actions": [], "questions": []})
        return "[]"

    async def chat_stream(self, system: str, user: str, **kw):
        self.calls["write"] += 1
        for w in re.findall(r"\S+\s*", self.reply):
            await asyncio.sleep(0.01)
            yield w

    async def tts_stream(self, text: str, **kw):
        self.calls["tts"] += 1
        n = int(SR * self.tts_seconds_per_word * max(1, len(text.split())))
        t = np.arange(n) / SR
        audio = (np.sin(2 * np.pi * 220 * t) * 3000).astype(np.int16).tobytes()
        await asyncio.sleep(0.05)
        for k in range(0, len(audio), 4800):
            yield audio[k:k + 4800]
            await asyncio.sleep(0)

    async def close(self) -> None:
        pass
