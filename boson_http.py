"""Stateless Boson calls over HTTP: chat completions (~0.25 s to the first token) and one-shot transcription.

Used by the contradiction monitor (fact_monitor.py), the knowledge base's fact extraction (kb/) and the barge-in
probe. Same key and base URL as the Realtime client (mmhm/higgs.py). Every call goes through `spend` so the
server-wide request budget still applies.
"""
import io
import ssl
import wave

import aiohttp
import certifi
import numpy as np

from mmhm.higgs import BASE, KEY, SR

TLS = ssl.create_default_context(cafile=certifi.where())
LLM = "higgs-realtime"
STT = "higgs-stt-3.1"


class Models:
    def __init__(self, spend=lambda n=1: True, language="en"):
        self.spend, self.language = spend, language
        self._http = None

    @property
    def http(self):
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(headers={"Authorization": f"Bearer {KEY}"},
                                               connector=aiohttp.TCPConnector(ssl=TLS, limit=32))
        return self._http

    async def chat(self, system, user, *, max_tokens=300, temperature=0.0, timeout=15.0):
        if not self.spend():
            raise RuntimeError("request budget used up")
        body = {"model": LLM, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        async with self.http.post(f"{BASE}/chat/completions", json=body,
                                  timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                raise RuntimeError(f"chat {r.status}: {(await r.text())[:200]}")
            data = await r.json()
        return (data["choices"][0]["message"].get("content") or "").strip()

    async def transcribe(self, pcm, *, timeout=6.0):
        """int16 24 kHz mono -> text (~0.6 s for a few seconds of speech)."""
        if not self.spend():
            return ""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(np.asarray(pcm, np.int16).tobytes())
        form = aiohttp.FormData()
        form.add_field("file", buf.getvalue(), filename="a.wav", content_type="audio/wav")
        form.add_field("model", STT)
        if self.language:
            form.add_field("language", self.language)
        async with self.http.post(f"{BASE}/audio/transcriptions", data=form,
                                  timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                raise RuntimeError(f"stt {r.status}: {(await r.text())[:200]}")
            return ((await r.json()).get("text") or "").strip()

    async def close(self):
        if self._http is not None:
            await self._http.close()
