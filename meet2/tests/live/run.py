"""Live end-to-end scenarios against real Boson models, through the real server, with bot participants.

    python -m tests.live.run [--only s1 s2 ...] [--port 8722]

Each bot joins over /ws/room exactly as a browser does and speaks TTS-rendered lines in real time (lines are cached
in tests/data/tts/). It records the AI's audio (slot 0) and the room's events, and each scenario is scored from
what a participant would hear and see. Needs BOSON_API_KEY. Makes real API calls (~40-80 per scenario).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path

import certifi
import numpy as np
import websockets
import ssl

ROOT = Path(__file__).resolve().parents[2]
TTS_DIR = ROOT / "tests" / "data" / "tts"
SR = 24000
ALEX = "chloe"
MEI = "chloe"   # the only preset voice on this key; identity comes from each bot's own mic stream
FILLERS = ("Umm.", "Okay.", "Mm-hm, yeah.", "Ahh, right.")


def key() -> str:
    k = os.environ.get("BOSON_API_KEY", "")
    if not k:
        sys.path.insert(0, str(ROOT))
        from meet import config
        k = config.settings().boson_api_key
    if not k:
        sys.exit("BOSON_API_KEY is not set")
    return k


def tts(text: str, voice: str) -> np.ndarray:
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TTS_DIR / f"{hashlib.sha1((voice + text).encode()).hexdigest()[:12]}.wav"
    if not path.exists():
        body = json.dumps({"model": "higgs-tts-3", "input": text, "response_format": "pcm", "voice": voice}).encode()
        req = urllib.request.Request("https://api.boson.ai/v1/audio/speech", data=body, headers={
            "Authorization": f"Bearer {key()}", "content-type": "application/json"})
        pcm = urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context(cafile=certifi.where())).read()
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm[:len(pcm) // 2 * 2])
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


class Bot:
    def __init__(self, base: str, rid: str, name: str, host_key: str = ""):
        self.url = f"{base.replace('http', 'ws')}/ws/room?room={rid}&pid={name.lower()}&name={name}&h={host_key}"
        self.name = name
        self.events: list[tuple[float, dict]] = []
        self.ai_frames: list[float] = []          # arrival times of AI audio frames
        self.ws = None
        self.reader = None
        self.sending = asyncio.Lock()

    async def join(self):
        self.ws = await websockets.connect(self.url, max_size=None)
        self.reader = asyncio.ensure_future(self._read())
        self.idle = asyncio.ensure_future(self._silence())

    async def _read(self):
        async for m in self.ws:
            t = time.monotonic()
            if isinstance(m, bytes):
                if m[0] == 1 and m[1] == 0:
                    self.ai_frames.append(t)
            else:
                self.events.append((t, json.loads(m)))

    async def _silence(self):
        """A live microphone sends frames all the time: quiet room noise between lines."""
        rng = np.random.default_rng(abs(hash(self.name)) % 1000)
        t0, k = time.monotonic(), 0
        while True:
            if not self.sending.locked():
                frame = rng.normal(0, 25, 480).astype(np.int16)
                await self.ws.send(bytes([1]) + frame.tobytes())
            k += 1
            await asyncio.sleep(max(0, t0 + k * 0.02 - time.monotonic()))

    async def say(self, audio: np.ndarray) -> float:
        """Send audio in real time; returns the time the last frame was sent."""
        async with self.sending:
            t0 = time.monotonic()
            for i, k in enumerate(range(0, len(audio) - 479, 480)):
                await self.ws.send(bytes([1]) + audio[k:k + 480].tobytes())
                await asyncio.sleep(max(0, t0 + (i + 1) * 0.02 - time.monotonic()))
            return time.monotonic()

    def ev(self, kind: str, since: float = 0.0) -> list[dict]:
        return [e for t, e in self.events if e.get("type") == kind and t >= since]

    def ai_audio_between(self, a: float, b: float) -> float:
        return sum(1 for t in self.ai_frames if a <= t < b) * 0.02

    async def reply_after(self, t0: float, timeout: float = 20.0, kind: str | None = None) -> dict | None:
        """The first finished AI turn (of this kind) that ended after t0 (waits for it)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            done = [e for t, e in self.events if e.get("type") == "ai_say" and e.get("done") and t >= t0
                    and (kind is None or e.get("kind") == kind)]
            if done:
                return done[0]
            await asyncio.sleep(0.1)
        return None

    def first_ai_frame_after(self, t0: float) -> float | None:
        return next((t for t in self.ai_frames if t > t0), None)

    async def leave(self):
        self.idle.cancel()
        self.reader.cancel()
        await self.ws.close()


def post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


HOUSE_FACTS = [
    "Rent is $2,400 a month, split equally ($800 each), due on the 1st to the landlord, Mrs. Okafor.",
    "Jordan pays the internet bill, $75 a month; Alex and Mei each owe Jordan $25.",
    "Trash goes out Monday and Thursday nights; recycling every other Monday.",
    "Quiet hours are 11 pm to 7 am on weeknights.",
]


async def room(base: str, people: list[str], facts=HOUSE_FACTS, title="House meeting"):
    r = post(base, "/api/rooms", {"title": title, "ai": "Higgs", "setting": "monthly house meeting of roommates",
                                  "facts": facts})
    bots = [Bot(base, r["id"], n, r["host_key"] if i == 0 else "") for i, n in enumerate(people)]
    for b in bots:
        await b.join()
    await asyncio.sleep(2.5)                     # front-ends learn the room; sessions open
    return r, bots


class Score:
    def __init__(self, name):
        self.name, self.checks = name, []

    def check(self, ok: bool, what: str, detail: str = ""):
        self.checks.append((bool(ok), what, detail))
        print(f"   {'PASS' if ok else 'FAIL'}  {what}{('  — ' + detail) if detail else ''}", flush=True)

    @property
    def ok(self):
        return all(c[0] for c in self.checks)


# ------------------------------------------------------------------ scenarios
async def s1_one_to_one(base) -> Score:
    sc = Score("S1 one-to-one answer (UC2)")
    _, (alex,) = await room(base, ["Alex"])
    end = await alex.say(tts("Hey Higgs, when is the rent due, and who do we pay it to?", ALEX))
    reply = await alex.reply_after(end)
    first = alex.first_ai_frame_after(end)
    said = [reply["text"]] if reply else []
    sc.check(first is not None, "the AI answered", f"first audio {first - end:.2f} s after the question" if first else "")
    if first:
        sc.check(first - end < 2.5, "end of question -> first AI audio under 2.5 s", f"{first - end:.2f} s")
    ok = any(("1st" in s or "first" in s.lower()) and ("okafor" in s.lower() or "landlord" in s.lower()) for s in said)
    sc.check(ok, "the answer has the right day and payee", said[-1] if said else "")
    await alex.leave()
    return sc


async def s2_fillers_then_interrupt(base) -> Score:
    sc = Score("S2 fillers don't stop it, an interruption does (F2)")
    _, (alex,) = await room(base, ["Alex"])
    end = await alex.say(tts("Higgs, can you walk me through all the house rules and the bills, in detail please?", ALEX))
    for _ in range(80):
        await asyncio.sleep(0.1)
        if alex.first_ai_frame_after(end):
            break
    start = alex.first_ai_frame_after(end)
    sc.check(start is not None, "the AI started answering")
    if not start:
        await alex.leave()
        return sc
    await asyncio.sleep(1.2)
    t_f = time.monotonic()
    for filler in FILLERS:
        await alex.say(tts(filler, ALEX))
        await asyncio.sleep(0.35)
    await asyncio.sleep(0.8)
    flushes = alex.ev("ai_flush", t_f)
    talking = alex.ai_audio_between(time.monotonic() - 0.6, time.monotonic())
    sc.check(not flushes, "fillers did not stop the AI", f"{len(flushes)} flush(es)")
    sc.check(talking > 0.3, "the AI is still talking after the fillers", f"{talking:.1f} s of audio in the last 0.6 s")
    if not talking:
        await alex.leave()
        return sc
    audio = tts("Wait, what about the internet bill?", ALEX)
    t_i = time.monotonic()
    await alex.say(audio)
    await asyncio.sleep(1.5)
    fl = alex.ev("ai_flush", t_i)
    sc.check(bool(fl), "the interruption stopped the AI")
    if fl:
        t_stop = next(t for t, e in alex.events if e.get("type") == "ai_flush" and t >= t_i)
        sc.check(t_stop - t_i < 1.2, "stopped within 1.2 s of the interruption starting", f"{t_stop - t_i:.2f} s")
    t_stop = time.monotonic()
    reply = await alex.reply_after(t_stop - 0.5)
    text = reply["text"] if reply else ""
    sc.check("internet" in text.lower() or "75" in text or "25" in text, "then it answered the interruption", text)
    await alex.leave()
    return sc


async def s3_group_quiet(base) -> Score:
    sc = Score("S3 people talking to each other: the AI stays quiet (UC3)")
    _, (alex, mei) = await room(base, ["Alex", "Mei"])
    t0 = time.monotonic()
    await alex.say(tts("Mei, did you get a chance to clean the bathroom yesterday?", ALEX))
    await asyncio.sleep(0.5)
    await mei.say(tts("Yeah, I did it last night after dinner.", MEI))
    await asyncio.sleep(0.5)
    await alex.say(tts("Nice, thanks. I'll do the kitchen tonight then.", ALEX))
    await asyncio.sleep(4)
    sc.check(alex.ai_audio_between(t0, time.monotonic()) == 0, "the AI said nothing")
    lines = [e for e in alex.ev("line") if e.get("final")]
    speakers = {e["speaker"] for e in lines}
    sc.check(speakers == {"Alex", "Mei"} and len(lines) >= 3, "every line was transcribed under the right name",
             "; ".join(f"{e['speaker']}: {e['text']}" for e in lines[:3]))
    await alex.leave()
    await mei.leave()
    return sc


async def s4_contradiction(base) -> Score:
    sc = Score("S4 a contradiction: the AI cuts in, ducks, buffers (F4)")
    _, (alex, mei) = await room(base, ["Alex", "Mei"])
    t0 = time.monotonic()
    await mei.say(tts("Okay so rent is due on the fifteenth, so I'll transfer my eight hundred then.", MEI))
    for _ in range(60):
        await asyncio.sleep(0.1)
        if alex.ev("correction", t0):
            break
    fixes = alex.ev("correction", t0)
    sc.check(bool(fixes), "the AI corrected the wrong date", fixes[0]["text"] if fixes else "")
    if not fixes:
        await alex.leave()
        await mei.leave()
        return sc
    t_fix = next(t for t, e in alex.events if e.get("type") == "correction" and t >= t0)
    gains = alex.ev("human_gain", t0)
    sc.check(gains and gains[0]["gain"] <= 0.2, "everyone's playback of the humans was ducked",
             f"gain {gains[0]['gain'] if gains else '-'}")
    sc.check(fixes[0].get("source"), "the correction names its source", str(fixes[0].get("source")))
    await alex.say(tts("Hmm, are you sure about that?", ALEX))         # talks over the correction
    await asyncio.sleep(0.3)
    flushes = alex.ev("ai_flush", t_fix)
    sc.check(not flushes, "talking over the correction did not stop it")
    for _ in range(100):
        await asyncio.sleep(0.1)
        if any(e["gain"] >= 1.0 for e in alex.ev("human_gain", t_fix)):
            break
    sc.check(any(e["gain"] >= 1.0 for e in alex.ev("human_gain", t_fix)), "volume came back after the correction")
    buffered = [e for e in alex.ev("line", t_fix) if e.get("kind") == "buffered"]
    sc.check(bool(buffered), "what Alex said meanwhile was held, not acted on",
             buffered[0]["text"] if buffered else "")
    t_rel = next((t for t, e in alex.events if e.get("type") == "human_gain" and e["gain"] >= 1.0 and t >= t_fix),
                 time.monotonic())
    reply = await alex.reply_after(t_rel, timeout=15, kind="reply")
    text = reply["text"] if reply else ""
    sc.check("1st" in text or "first" in text.lower(), "afterwards it answered the doubt with the evidence", text)
    await alex.leave()
    await mei.leave()
    return sc


async def s5_noise(base) -> Score:
    sc = Score("S5 background noise makes no lines and no replies (F6)")
    _, (alex,) = await room(base, ["Alex"])
    rng = np.random.default_rng(3)
    t0 = time.monotonic()
    white = (rng.normal(0, 1, SR * 4) * 1500).astype(np.int16)
    b = np.cumsum(rng.normal(0, 1, SR * 4))
    b = b - np.convolve(b, np.ones(2400) / 2400, "same")
    fan = (b / np.abs(b).max() * 6000).astype(np.int16)
    clicks = rng.normal(0, 30, SR * 3)
    for k in range(12):
        s = int((0.2 + k * 0.23) * SR)
        clicks[s:s + 240] += rng.normal(0, 9000, 240)
    for noise in (white, fan, clicks.astype(np.int16)):
        await alex.say(noise)
    await asyncio.sleep(2)
    lines = [e for e in alex.ev("line", t0)]
    sc.check(not lines, "no transcript lines from noise", "; ".join(e["text"] for e in lines[:3]))
    sc.check(alex.ai_audio_between(t0, time.monotonic()) == 0, "the AI said nothing")
    await alex.leave()
    return sc


LEASE = ("Residential lease for 14 Elm Street. Section 7, pets: no pets are allowed without the landlord's written "
         "approval. Section 9, guests: a guest staying more than 7 nights in a row needs the landlord's notice in "
         "advance. Section 11, repairs: tenants report repairs through the landlord's portal within 48 hours.")


async def s6_knowledge_base(base) -> Score:
    sc = Score("S6 an answer from an uploaded document, with its source (F5)")
    r = post(base, "/api/rooms", {"title": "House", "ai": "Higgs", "setting": "house meeting", "facts": []})
    post(base, f"/api/rooms/{r['id']}/kb/sources", {"title": "Lease", "text": LEASE})
    for _ in range(60):
        st = json.loads(urllib.request.urlopen(f"{base}/api/rooms/{r['id']}/kb").read())
        if st["docs"] and st["docs"][0]["status"] in ("ready", "error"):
            break
        await asyncio.sleep(0.5)
    sc.check(st["docs"][0]["status"] == "ready", "the document was ingested",
             f"{st['docs'][0]['status']}, {len(st['facts'])} facts extracted")
    alex = Bot(base, r["id"], "Alex", r["host_key"])
    await alex.join()
    await asyncio.sleep(2.5)
    end = await alex.say(tts("Higgs, what does the lease say about having a dog?", ALEX))
    reply = await alex.reply_after(end)
    said = [reply] if reply else []
    text = reply["text"] if reply else ""
    sc.check("approval" in text.lower() or "written" in text.lower() or "permission" in text.lower(),
             "the answer comes from the lease", text)
    sc.check(bool(said and said[-1].get("source")), "the answer names its source", str(said[-1].get("source") if said else ""))
    await alex.leave()
    return sc


async def s7_avatar(base) -> Score:
    sc = Score("S7 live avatar over LiveKit, and back to voice only (F1)")
    from livekit import rtc
    r = post(base, "/api/rooms", {"title": "House", "ai": "Higgs", "setting": "house meeting", "facts": HOUSE_FACTS,
                                  "avatar": "live"})
    alex = Bot(base, r["id"], "Alex", r["host_key"])
    await alex.join()
    cred = post(base, f"/api/rooms/{r['id']}/livekit-token", {"pid": "alex"})
    lk = rtc.Room()
    got = {"video": 0, "audio": 0, "audio_frames": [], "first_video": None}

    async def pull_audio(track):
        async for ev in rtc.AudioStream(track, sample_rate=24000, num_channels=1):
            x = np.frombuffer(bytes(ev.frame.data), np.int16)
            if np.abs(x).max() > 300:
                got["audio_frames"].append(time.monotonic())

    @lk.on("track_subscribed")
    def on_track(track, pub, participant):
        if participant.identity != cred["avatar_identity"]:
            return
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            got["video"] += 1
            got["first_video"] = got["first_video"] or time.monotonic()
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            got["audio"] += 1
            asyncio.ensure_future(pull_audio(track))
    await lk.connect(cred["url"], cred["token"])
    for _ in range(150):
        if got["video"] and got["audio"]:
            break
        await asyncio.sleep(0.2)
    sc.check(got["video"] and got["audio"], "the avatar published video and audio tracks")
    await asyncio.sleep(1.0)
    end = await alex.say(tts("Hey Higgs, when is the rent due, and who do we pay it to?", ALEX))
    reply = await alex.reply_after(end, timeout=30)
    first = next((t for t in got["audio_frames"] if t > end), None)
    sc.check(reply is not None, "the AI answered", reply["text"] if reply else "")
    sc.check(first is not None, "its voice came through the avatar over LiveKit",
             f"first avatar audio {first - end:.2f} s after the question" if first else "")
    sc.check(alex.ai_audio_between(end, time.monotonic()) == 0, "and not over the WebSocket")
    await alex.ws.send(json.dumps({"type": "set_avatar", "mode": "off"}))
    await asyncio.sleep(1.5)
    sc.check(any(e.get("mode") == "off" for e in alex.ev("avatar")), "the avatar turned off")
    end = await alex.say(tts("Higgs, what nights does the trash go out?", ALEX))
    reply = await alex.reply_after(end, timeout=20)
    ws_audio = alex.ai_audio_between(end, time.monotonic())
    sc.check(reply is not None and ws_audio > 0.5, "voice only again: the answer came over the WebSocket",
             f"{ws_audio:.1f} s of audio; {reply['text'] if reply else ''}")
    await lk.disconnect()
    await alex.leave()
    return sc


PRERENDER = [
    *((f, ALEX) for f in FILLERS),
    ("Hey Higgs, when is the rent due, and who do we pay it to?", ALEX),
    ("Higgs, can you walk me through all the house rules and the bills, in detail please?", ALEX),
    ("Wait, what about the internet bill?", ALEX),
    ("Mei, did you get a chance to clean the bathroom yesterday?", ALEX),
    ("Yeah, I did it last night after dinner.", MEI),
    ("Nice, thanks. I'll do the kitchen tonight then.", ALEX),
    ("Okay so rent is due on the fifteenth, so I'll transfer my eight hundred then.", MEI),
    ("Hmm, are you sure about that?", ALEX),
    ("Higgs, what does the lease say about having a dog?", ALEX),
    ("Higgs, what nights does the trash go out?", ALEX),
]

SCENARIOS = {"s1": s1_one_to_one, "s2": s2_fillers_then_interrupt, "s3": s3_group_quiet, "s4": s4_contradiction,
             "s5": s5_noise, "s6": s6_knowledge_base, "s7": s7_avatar}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[k for k in SCENARIOS if k != "s7"],
                    help="scenarios to run (s7, the live avatar, needs LiveKit settings and runs only when named)")
    ap.add_argument("--port", type=int, default=8722)
    ap.add_argument("--base", help="use a server that is already running")
    args = ap.parse_args()
    print("Rendering lines with TTS (cached in tests/data/tts)...", flush=True)
    for text, voice in PRERENDER:
        tts(text, voice)
    proc = None
    base = args.base
    if not base:
        import socket
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", args.port)) == 0:
                sys.exit(f"port {args.port} is already in use (an old server?): stop it or pass --port")
        out = ROOT / "out" / "live"
        out.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "BOSON_API_KEY": key(), "MEET_PORT": str(args.port), "MEET_AGENT": "higgs",
               "MEET_LOG_DIR": str(out / "rooms"), "MEET_KB_DIR": str(out / "kb")}
        proc = subprocess.Popen([sys.executable, "-m", "meet.server"], cwd=ROOT, env=env,
                                stdout=open(out / "server.log", "w"), stderr=subprocess.STDOUT)
        base = f"http://127.0.0.1:{args.port}"
        for _ in range(50):
            try:
                urllib.request.urlopen(base + "/api/config", timeout=1)
                break
            except Exception:  # noqa: BLE001
                if proc.poll() is not None:
                    sys.exit("the server exited at start-up: see out/live/server.log")
                time.sleep(0.2)
    results = []
    try:
        for name in args.only:
            print(f"\n== {name}: {SCENARIOS[name].__doc__ or ''}".rstrip(), flush=True)
            sc = await SCENARIOS[name](base)
            print(f"   -> {sc.name}: {'PASS' if sc.ok else 'FAIL'}")
            results.append(sc)
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
    print("\nSummary")
    for sc in results:
        print(f"  {'PASS' if sc.ok else 'FAIL'}  {sc.name}  ({sum(c[0] for c in sc.checks)}/{len(sc.checks)})")
    sys.exit(0 if all(sc.ok for sc in results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
