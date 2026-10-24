"""Boson AI client: chat completions, streaming TTS, transcription (HTTP and Realtime sessions).

Measured 2026-10-23 with higgs-realtime / higgs-stt-3.1 / higgs-tts-3:
  chat completion first token ~0.23 s; STT transcript ~0.3 s after a commit (no partial results are sent, so a
  "partial" is made by committing early); TTS first audio ~0.5 s, faster than real time.
Everything else talks to Boson only through `Models`, so tests can swap in a fake with the same methods.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import ssl
import wave
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import certifi
import numpy as np

from .obs import Budget

SR = 24000
TLS = ssl.create_default_context(cafile=certifi.where())


class ModelError(RuntimeError):
    pass


def wav_bytes(pcm: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.asarray(pcm, np.int16).tobytes())
    return buf.getvalue()


class Models:
    def __init__(self, api_key: str, base_url: str, *, llm: str, stt: str, tts: str, voice: str, language: str,
                 noise_reduction: bool = True, budget: Budget | None = None):
        if not api_key:
            raise ModelError("BOSON_API_KEY is not set (put it in .env)")
        self.key, self.base = api_key, base_url.rstrip("/")
        self.llm, self.stt, self.tts, self.voice, self.language = llm, stt, tts, voice, language
        self.noise_reduction = noise_reduction
        self.budget = budget or Budget(10**9)
        self._http: aiohttp.ClientSession | None = None

    @property
    def http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.key}"},
                connector=aiohttp.TCPConnector(ssl=TLS, limit=64),
                timeout=aiohttp.ClientTimeout(total=60, sock_read=30))
        return self._http

    async def close(self) -> None:
        if self._http is not None:
            await self._http.close()

    def _spend(self) -> None:
        if not self.budget.spend():
            raise ModelError("request budget used up")

    # ------------------------------------------------------------------ text
    async def chat(self, system: str, user: str, *, max_tokens: int = 300, temperature: float = 0.2,
                   timeout: float = 15.0) -> str:
        self._spend()
        body = {"model": self.llm, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        async with self.http.post(f"{self.base}/chat/completions", json=body,
                                  timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                raise ModelError(f"chat {r.status}: {(await r.text())[:200]}")
            data = await r.json()
        return (data["choices"][0]["message"].get("content") or "").strip()

    async def chat_stream(self, system: str, user: str, *, max_tokens: int = 200, temperature: float = 0.5,
                          timeout: float = 20.0) -> AsyncIterator[str]:
        self._spend()
        body = {"model": self.llm, "max_tokens": max_tokens, "temperature": temperature, "stream": True,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        async with self.http.post(f"{self.base}/chat/completions", json=body,
                                  timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                raise ModelError(f"chat {r.status}: {(await r.text())[:200]}")
            async for raw in r.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta

    # ------------------------------------------------------------------ speech out
    async def tts_stream(self, text: str, *, voice: str | None = None, timeout: float = 30.0) -> AsyncIterator[bytes]:
        """int16 mono 24 kHz PCM, in chunks with an even byte count."""
        self._spend()
        body = {"model": self.tts, "input": text, "response_format": "pcm", "stream": True,
                "voice": voice or self.voice}
        async with self.http.post(f"{self.base}/audio/speech", json=body,
                                  timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                raise ModelError(f"tts {r.status}: {(await r.text())[:200]}")
            carry = b""
            async for chunk in r.content.iter_any():
                chunk = carry + chunk
                cut = len(chunk) // 2 * 2
                carry = chunk[cut:]
                if cut:
                    yield chunk[:cut]

    # ------------------------------------------------------------------ speech in
    async def transcribe(self, pcm: np.ndarray, *, timeout: float = 10.0) -> str:
        """One-shot HTTP transcription (~0.6 s for 4 s of speech)."""
        self._spend()
        form = aiohttp.FormData()
        form.add_field("file", wav_bytes(pcm), filename="a.wav", content_type="audio/wav")
        form.add_field("model", self.stt)
        if self.language:
            form.add_field("language", self.language)
        async with self.http.post(f"{self.base}/audio/transcriptions", data=form,
                                  timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                raise ModelError(f"stt {r.status}: {(await r.text())[:200]}")
            return ((await r.json()).get("text") or "").strip()

    def open_stt(self) -> "STTSession":
        return STTSession(self)


class STTSession:
    """A Realtime session used only for transcription (no turn detection: we decide when a segment ends).

    append() never blocks; audio sent while (re)connecting is kept (up to 30 s) and replayed. commit() returns the
    transcript of everything appended since the previous commit. Higgs closes idle sessions after 5 minutes, and
    opens can fail under load, so the session reopens itself with backoff.
    """

    BACKLOG_MAX = 30 * SR

    def __init__(self, models: Models):
        self.m = models
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self.ready = asyncio.Event()
        self.closed = False
        self.backlog: list[bytes] = []
        self._backlog_n = 0
        self._waiters: dict[str, asyncio.Future] = {}
        self._commits: asyncio.Queue[asyncio.Future] = asyncio.Queue()
        self._runner = asyncio.ensure_future(self._run())
        self._lock = asyncio.Lock()

    async def _connect(self) -> None:
        url = self.m.base.replace("https://", "wss://").replace("http://", "ws://") + "/realtime?model=higgs-realtime"
        ws = await self.m.http.ws_connect(url, max_msg_size=0, heartbeat=20)
        tr: dict[str, Any] = {"model": self.m.stt}
        if self.m.language:
            tr["language"] = self.m.language
        audio_in: dict[str, Any] = {"turn_detection": None, "transcription": tr}
        if self.m.noise_reduction:
            audio_in["noise_reduction"] = {"type": "near_field"}
        await ws.send_str(json.dumps({"type": "session.update", "session": {
            "model": "higgs-realtime", "instructions": "Transcription only.", "output_modalities": ["text"],
            "audio": {"input": audio_in}}}))
        msg = await asyncio.wait_for(ws.receive(), 15)
        if msg.type != aiohttp.WSMsgType.TEXT or json.loads(msg.data).get("type") not in ("session.created",
                                                                                          "session.updated"):
            await ws.close()
            raise ModelError(f"stt session did not open: {str(msg.data)[:200]}")
        self.ws = ws

    async def _run(self) -> None:
        attempt = 0
        while not self.closed:
            try:
                await self._connect()
                attempt = 0
                backlog, self.backlog, self._backlog_n = self.backlog, [], 0
                for b in backlog:
                    await self._send_audio(b)
                self.ready.set()
                await self._read()
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: BLE001  (reconnect below)
                pass
            self.ready.clear()
            self.ws = None
            for f in list(self._waiters.values()):
                if not f.done():
                    f.set_exception(ModelError("stt session closed"))
            self._waiters.clear()
            attempt += 1
            await asyncio.sleep(min(8.0, 0.5 * 2 ** attempt))

    async def _read(self) -> None:
        assert self.ws is not None
        async for msg in self.ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            ev = json.loads(msg.data)
            typ = ev.get("type", "")
            if typ == "input_audio_buffer.committed":
                fut = self._commits.get_nowait() if not self._commits.empty() else None
                if fut is not None and not fut.done():
                    self._waiters[ev.get("item_id", "")] = fut
            elif typ == "conversation.item.input_audio_transcription.completed":
                fut = self._waiters.pop(ev.get("item_id", ""), None)
                if fut is not None and not fut.done():
                    fut.set_result((ev.get("transcript") or "").strip())
            elif typ == "error":
                err = json.dumps(ev.get("error"))
                if ("commit" in err.lower() or "buffer" in err.lower()) and not self._commits.empty():
                    fut = self._commits.get_nowait()   # this commit produced no item
                    if not fut.done():
                        fut.set_exception(ModelError(err[:200]))

    async def _send_audio(self, pcm: bytes) -> None:
        assert self.ws is not None
        await self.ws.send_str(json.dumps({"type": "input_audio_buffer.append",
                                           "audio": base64.b64encode(pcm).decode()}))

    async def append(self, pcm: bytes) -> None:
        if self.ready.is_set() and self.ws is not None:
            try:
                await self._send_audio(pcm)
                return
            except Exception:  # noqa: BLE001
                pass
        if self._backlog_n < self.BACKLOG_MAX:
            self.backlog.append(pcm)
            self._backlog_n += len(pcm) // 2

    async def commit_nowait(self, timeout: float = 6.0) -> asyncio.Future:
        """Send the commit now (so audio appended afterwards goes to the next item) and return a future for the
        transcript of everything appended since the previous commit."""
        async with self._lock:
            await asyncio.wait_for(self.ready.wait(), timeout)
            self.m._spend()
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            await self._commits.put(fut)
            assert self.ws is not None
            await self.ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
        return fut

    @staticmethod
    async def result(fut: asyncio.Future, timeout: float = 6.0) -> str:
        try:
            return await asyncio.wait_for(fut, timeout)
        except ModelError as e:
            if "empty" in str(e).lower() or "too small" in str(e).lower():
                return ""
            raise

    async def commit(self, timeout: float = 6.0) -> str:
        """Transcript of the audio since the last commit; "" if there was none."""
        return await self.result(await self.commit_nowait(timeout), timeout)

    async def clear(self) -> None:
        """Drop audio appended since the last commit (a noise blip, not speech)."""
        self.backlog, self._backlog_n = [], 0
        if self.ready.is_set() and self.ws is not None:
            try:
                await self.ws.send_str(json.dumps({"type": "input_audio_buffer.clear"}))
            except Exception:  # noqa: BLE001
                pass

    async def close(self) -> None:
        self.closed = True
        self._runner.cancel()
        if self.ws is not None:
            await self.ws.close()
