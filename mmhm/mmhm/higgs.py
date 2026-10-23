"""Higgs API helpers: TTS, a logged Realtime session with paced audio streaming, WAV I/O."""
import asyncio
import base64
import json
import os
import time

import numpy as np
import requests
import soundfile as sf
import websockets

ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_key():
    """BOSON_API_KEY from the environment, else from the nearest .env up the tree."""
    d = ROOT
    while "BOSON_API_KEY" not in os.environ and d != os.path.dirname(d):
        env = os.path.join(d, ".env")
        if os.path.exists(env):
            for line in open(env):
                k, _, v = line.strip().partition("=")
                if k == "BOSON_API_KEY":
                    os.environ[k] = v
        d = os.path.dirname(d)
    return os.environ["BOSON_API_KEY"]


KEY = _load_key()
BASE = "https://api.boson.ai/v1"
WS = "wss://api.boson.ai/v1/realtime?model=higgs-realtime"
HDR = {"Authorization": f"Bearer {KEY}"}
SR = 24000


def outdir(name):
    d = os.path.join(ROOT, "..", "out", name)
    os.makedirs(d, exist_ok=True)
    return d


def save_wav(path, pcm, sr=SR):
    sf.write(path, np.asarray(pcm, dtype=np.int16), sr, subtype="PCM_16")


def load_wav(path, sr=SR):
    x, r = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if r != sr:
        from scipy.signal import resample_poly
        x = resample_poly(x, sr, r)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def tts(text, voice="chloe", stream=True, **extra):
    """Returns (int16 pcm at 24 kHz, time-to-first-audio s, total s)."""
    t0 = time.perf_counter()
    body = {"model": "higgs-tts-3", "input": text, "response_format": "pcm", "stream": stream, **extra}
    if voice:
        body["voice"] = voice
    r = requests.post(f"{BASE}/audio/speech", headers=HDR, stream=True, timeout=120, json=body)
    if r.status_code != 200:
        raise RuntimeError(f"TTS {r.status_code}: {r.text[:300]}")
    ttfa, buf = None, bytearray()
    for chunk in r.iter_content(4096):
        if chunk:
            ttfa = ttfa or time.perf_counter() - t0
            buf.extend(chunk)
    if len(buf) % 2:
        buf = buf[:-1]
    return np.frombuffer(bytes(buf), dtype=np.int16), ttfa, time.perf_counter() - t0


def silence(sec):
    return np.zeros(int(sec * SR), dtype=np.int16)


def trim(pcm, thresh=300):
    """Strip leading/trailing near-silence."""
    idx = np.flatnonzero(np.abs(pcm.astype(np.int32)) > thresh)
    return pcm[idx[0]:idx[-1] + 1] if len(idx) else pcm


class Session:
    """A Realtime WebSocket with a background reader that timestamps every server event."""

    # Session opens validate the preset voice against the voices API, which answers 429 when ~6+ sessions open
    # at once and then stays rate-limited for tens of seconds. Open one session at a time, spaced out.
    _open_lock = None
    _last_open = 0.0
    _free_sem = None

    def __init__(self, session_cfg, log_audio=False):
        self.cfg = session_cfg
        self.events = []  # (t, ev) with t relative to self.t0
        self.t0 = None
        self.audio = {}  # response_id -> bytearray
        self.log_audio = log_audio
        self.close_code = None
        self._waiters = []

    async def __aenter__(self, attempts=4):
        if not (self.cfg.get("audio") or {}).get("output", {}).get("voice"):
            # no preset voice to validate: no need to queue, but cap bursts (42 at once drew upstream_errors)
            if Session._free_sem is None:
                Session._free_sem = asyncio.Semaphore(8)
            async with Session._free_sem:
                return await self._open(attempts)
        if Session._open_lock is None:
            Session._open_lock = asyncio.Lock()
        async with Session._open_lock:
            wait = Session._last_open + 0.3 - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                return await self._open(attempts)
            finally:
                Session._last_open = time.perf_counter()

    async def _open(self, attempts):
        for k in range(attempts):
            self.events, self.audio, self._waiters, self.close_code = [], {}, [], None
            self.ws = await websockets.connect(WS, additional_headers=HDR, max_size=None)
            self.t0 = time.perf_counter()
            self.reader = asyncio.create_task(self._read())
            # Higgs sends nothing until the client speaks first; the first session.update is answered with
            # session.created (carrying the applied config), later updates with session.updated.
            await self.send({"type": "session.update", "session": {"model": "higgs-realtime", **self.cfg}})
            ok = lambda ev: ev.get("type") in ("session.created", "session.updated")  # noqa: E731
            failed = lambda ev: ev.get("type") == "error" and "terminated" in ev.get("error", {}).get("message", "")  # noqa: E731
            try:
                _, ev = await self.wait(lambda ev: ok(ev) or failed(ev), timeout=15, since=0.0)
                if ok(ev):
                    return self
            except asyncio.TimeoutError:
                pass
            # Opening several sessions at once can trip a 429 on preset-voice validation: back off and retry.
            self.reader.cancel()
            await self.ws.close()
            await asyncio.sleep(2.0 + 4.0 * k)
        raise RuntimeError("could not open a Higgs Realtime session")

    async def __aexit__(self, *a):
        self.reader.cancel()
        await self.ws.close()

    def now(self):
        return time.perf_counter() - self.t0

    async def send(self, ev):
        await self.ws.send(json.dumps(ev))

    async def _read(self):
        try:
            async for raw in self.ws:
                ev = json.loads(raw)
                t = self.now()
                typ = ev.get("type", "")
                if typ == "response.output_audio.delta":
                    self.audio.setdefault(ev.get("response_id"), bytearray()).extend(base64.b64decode(ev["delta"]))
                    if not self.log_audio:
                        ev = {"type": typ, "response_id": ev.get("response_id"), "n": len(ev["delta"]) * 3 // 4}
                elif typ == "error":
                    print("  !! server error:", json.dumps(ev)[:400])
                self.events.append((t, ev))
                for w in list(self._waiters):
                    if w[0](ev):
                        w[1].set_result((t, ev))
                        self._waiters.remove(w)
        except websockets.ConnectionClosed as e:
            self.close_code = e.code
        except asyncio.CancelledError:
            pass

    async def wait(self, pred, timeout=60, since=None):
        """Wait for an event whose type equals pred (str) or for which pred(ev) is true.
        If since is given, an already-received event after that time also matches."""
        f = (lambda ev: ev.get("type") == pred) if isinstance(pred, str) else pred
        if since is not None:
            for t, ev in self.events:
                if t >= since and f(ev):
                    return t, ev
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append((f, fut))
        return await asyncio.wait_for(fut, timeout)

    async def say_text(self, text, create=True):
        await self.send({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}})
        if create:
            await self.send({"type": "response.create"})

    async def stream_audio(self, pcm, chunk_ms=20, paced=True):
        """Append int16 24 kHz PCM in chunk_ms pieces, in real time if paced. Returns the session time when
        the last chunk was sent."""
        n = int(SR * chunk_ms / 1000)
        start = time.perf_counter()
        for i in range(0, len(pcm), n):
            await self.send({"type": "input_audio_buffer.append",
                             "audio": base64.b64encode(pcm[i:i + n].tobytes()).decode()})
            if paced:
                target = start + (i + n) / SR
                d = target - time.perf_counter()
                if d > 0:
                    await asyncio.sleep(d)
        return self.now()

    async def silence_until(self, pred, max_s=20.0, since=None, chunk_ms=20):
        """Keep a live 'mic' open: stream real-time silence until an event matching pred arrives (or max_s).
        Returns (t, ev) or (None, None)."""
        f = (lambda ev: ev.get("type") == pred) if isinstance(pred, str) else pred
        since = self.now() if since is None else since
        n = int(SR * chunk_ms / 1000)
        z = base64.b64encode(np.zeros(n, dtype=np.int16).tobytes()).decode()
        start = time.perf_counter()
        k = 0
        while time.perf_counter() - start < max_s:
            for t, ev in self.events:
                if t >= since and f(ev):
                    return t, ev
            await self.send({"type": "input_audio_buffer.append", "audio": z})
            k += 1
            d = start + k * chunk_ms / 1000 - time.perf_counter()
            if d > 0:
                await asyncio.sleep(d)
        return None, None

    def types(self, since=0.0):
        return [(round(t, 3), ev["type"]) for t, ev in self.events if t >= since]

    def first(self, typ, since=0.0, pred=None):
        for t, ev in self.events:
            if t >= since and ev.get("type") == typ and (pred is None or pred(ev)):
                return t, ev
        return None, None

    def transcript(self, response_id):
        return "".join(ev.get("delta", "") for _, ev in self.events
                       if ev.get("response_id") == response_id and ev.get("type") in (
                           "response.output_audio_transcript.delta", "response.output_text.delta"))

    def response_audio(self, response_id):
        return np.frombuffer(bytes(self.audio.get(response_id, b"")), dtype=np.int16)

    def dump(self, path):
        with open(path, "w") as f:
            for t, ev in self.events:
                f.write(json.dumps({"t": round(t, 4), **ev}) + "\n")
