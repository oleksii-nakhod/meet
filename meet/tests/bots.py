"""Bot participants for the Meet server: no browser, the real /ws/room protocol.

A Bot joins a room like a browser tab does, sends 20 ms int16 frames in real time (its queued speech, otherwise a
very quiet noise floor), and records what it receives: Higgs's slot-0 audio (as onset/offset times) and every JSON
event (`line`, `ai_say`, `ai_state`, `ai_flush`, ...). Scenarios (tests/scenarios.py) drive bots and read these.

Speech comes from Higgs TTS (cached under out/clips/, one file per voice + text) or from real recordings (AMI cues
in ../mmhm/out/ami_cues: listener backchannels bc_*.wav, interruptions int_*.wav)."""
import asyncio
import hashlib
import json
import os
import secrets
import sys
import time
import urllib.request

import numpy as np
import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
MEET = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(MEET, "..", "mmhm"))
from mmhm.higgs import SR, load_wav, save_wav, trim, tts  # noqa: E402

CLIPS = os.path.join(MEET, "out", "clips")
AMI = os.path.join(MEET, "..", "mmhm", "out", "ami_cues")
N = 480
AI_RMS = 150  # slot-0 frame louder than this counts as Higgs speaking


def clip(text, voice="marcus", gain_db=0.0, **extra):
    """TTS of `text` in `voice`, trimmed, cached on disk. gain_db scales it (a soft "mm-hm" is -10)."""
    os.makedirs(CLIPS, exist_ok=True)
    h = hashlib.sha1(json.dumps([text, voice, extra], sort_keys=True).encode()).hexdigest()[:16]
    path = os.path.join(CLIPS, f"{voice}_{h}.wav")
    if not os.path.exists(path):
        for k in range(4):
            try:
                pcm, _, _ = tts(text, voice=voice, **extra)
                break
            except Exception:  # noqa: BLE001
                if k == 3:
                    raise
                time.sleep(2 + 3 * k)
        save_wav(path, trim(pcm))
    pcm = load_wav(path)
    if gain_db:
        pcm = np.clip(pcm.astype(np.float32) * 10 ** (gain_db / 20), -32768, 32767).astype(np.int16)
    return pcm


def ami(name, gain_db=0.0):
    pcm = trim(load_wav(os.path.join(AMI, name)), 200)
    if gain_db:
        pcm = np.clip(pcm.astype(np.float32) * 10 ** (gain_db / 20), -32768, 32767).astype(np.int16)
    return pcm


def http(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def create_room(base, admin, title="test", ai="Higgs", setting="", facts=(), passcode="pw"):
    return http(base + "/api/rooms", {"admin": admin, "title": title, "ai": ai, "setting": setting,
                                      "facts": list(facts), "passcode": passcode})["id"]


class Bot:
    def __init__(self, base, room, name, passcode="pw", noise_db=-70.0, echo=0.0):
        self.base, self.room, self.name, self.k = base, room, name, passcode
        self.pid = secrets.token_hex(6)
        self.noise = 32768 * 10 ** (noise_db / 20)
        self.echo = echo                 # >0: this bot's mic picks up Higgs's voice at this gain (leaky AEC)
        self.queue = []                  # [ [pcm, pos, future] ]
        self.events = []                 # (t, msg)
        self.ai_frames = []              # (t, rms) of slot-0 frames
        self.ai_tail = np.zeros(0, np.int16)  # recent Higgs audio for the echo path
        self.ai_audio = []               # (t, pcm) all slot-0 frames, for saving
        self.slot = None
        self.ws = None
        self.tasks = []
        self.speaking_until = 0.0
        self.t0 = time.perf_counter()

    def now(self):
        return time.perf_counter() - self.t0

    async def join(self):
        u = self.base.replace("http", "ws") + f"/ws/room?room={self.room}&pid={self.pid}&name={self.name}&k={self.k}"
        self.ws = await websockets.connect(u, max_size=None)
        self.tasks = [asyncio.ensure_future(self._send()), asyncio.ensure_future(self._recv())]

    async def leave(self):
        for t in self.tasks:
            t.cancel()
        if self.ws:
            await self.ws.close()

    async def json(self, obj):
        await self.ws.send(json.dumps(obj))

    def say(self, pcm):
        """Queue speech; returns a future that resolves (to the end time) when the last frame has been sent."""
        f = asyncio.get_running_loop().create_future()
        self.queue.append([np.asarray(pcm, np.int16), 0, f])
        return f

    def silence_frame(self):
        return (np.random.randn(N) * self.noise).astype(np.int16)

    async def _send(self):
        start, k = time.perf_counter(), 0
        rng = np.random.default_rng(int(self.pid, 16) % 2**32)
        while True:
            if self.queue:
                item = self.queue[0]
                pcm, pos, fut = item
                frame = np.zeros(N, np.int16)
                seg = pcm[pos:pos + N]
                frame[:len(seg)] = seg
                item[1] = pos + N
                self.speaking_until = self.now() + 0.02
                if item[1] >= len(pcm):
                    self.queue.pop(0)
                    if not fut.done():
                        fut.set_result(self.now())
                frame = frame.astype(np.int32) + (rng.standard_normal(N) * self.noise).astype(np.int32)
            else:
                frame = (rng.standard_normal(N) * self.noise).astype(np.int32)
            if self.echo and len(self.ai_tail) >= N:
                frame = frame + (self.ai_tail[:N].astype(np.float32) * self.echo).astype(np.int32)
                self.ai_tail = self.ai_tail[N:]
            try:
                await self.ws.send(np.clip(frame, -32768, 32767).astype(np.int16).tobytes())
            except Exception:  # noqa: BLE001
                return
            k += 1
            d = start + k * 0.02 - time.perf_counter()
            if d > 0:
                await asyncio.sleep(d)

    async def _recv(self):
        try:
            async for m in self.ws:
                t = self.now()
                if isinstance(m, bytes):
                    if m[0] == 0 and len(m) == 961:
                        pcm = np.frombuffer(m[1:], np.int16)
                        rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
                        self.ai_frames.append((t, rms))
                        self.ai_audio.append((t, pcm))
                        if self.echo:
                            self.ai_tail = np.concatenate([self.ai_tail[-SR:], pcm])
                    continue
                msg = json.loads(m)
                if msg.get("type") == "roster":
                    self.slot = msg.get("you")
                self.events.append((t, msg))
        except Exception:  # noqa: BLE001
            pass

    # ---- reading what happened ----
    def ai_onsets(self, since=0.0, gap=0.6):
        """[(start, end)] of Higgs's audible speech after `since` (frames louder than AI_RMS, gaps < `gap` merged)."""
        spans = []
        for t, rms in self.ai_frames:
            if t < since or rms < AI_RMS:
                continue
            if spans and t - spans[-1][1] < gap:
                spans[-1][1] = t
            else:
                spans.append([t, t])
        return [(a, b) for a, b in spans]

    def ai_speaking(self, within=0.25):
        return bool(self.ai_frames) and self.now() - self.ai_frames[-1][0] < within and self.ai_frames[-1][1] >= AI_RMS

    def evs(self, typ, since=0.0):
        return [(t, m) for t, m in self.events if t >= since and m.get("type") == typ]

    async def wait_ai(self, timeout, since=None):
        """Wait until Higgs is audible; returns the onset time or None."""
        since = self.now() if since is None else since
        end = self.now() + timeout
        while self.now() < end:
            on = self.ai_onsets(since)
            if on:
                return on[0][0]
            await asyncio.sleep(0.02)
        return None

    async def wait_ai_quiet(self, quiet=1.0, timeout=30.0):
        """Wait until Higgs has been silent for `quiet` seconds; returns the time its audio ended."""
        end = self.now() + timeout
        while self.now() < end:
            loud = [t for t, r in self.ai_frames if r >= AI_RMS]
            last = loud[-1] if loud else 0.0
            if self.now() - last >= quiet:
                return last
            await asyncio.sleep(0.05)
        return None

    def save_ai(self, path):
        if self.ai_audio:
            pcm = np.concatenate([p for _, p in self.ai_audio])
            save_wav(path, pcm)
