"""Live checks for the junaid3 features against real Boson, through the real server, with bot participants.

    python tests/live_junaid3.py [--only l1 l2 ...] [--port 8723]     (l6, the live avatar, runs only when named)

Bots join over /ws/room exactly as a browser does (20 ms int16 frames; the server sends 1 byte slot + PCM back) and
speak TTS-rendered lines in real time (cached in tests/data/tts3/). Needs BOSON_API_KEY. Real API calls.
"""
import argparse
import asyncio
import hashlib
import json
import os
import ssl
import subprocess
import sys
import time
import urllib.request
import wave

import certifi
import numpy as np
import websockets

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from meet_config import load_environment  # noqa: E402

load_environment()
TTS_DIR = os.path.join(HERE, "tests", "data", "tts3")
SR = 24000
CTX = ssl.create_default_context(cafile=certifi.where())
FILLERS = ("Umm.", "Okay.", "Mm-hm, yeah.", "Ahh, right.")


def tts(text):
    os.makedirs(TTS_DIR, exist_ok=True)
    path = os.path.join(TTS_DIR, hashlib.sha1(text.encode()).hexdigest()[:12] + ".wav")
    if not os.path.exists(path):
        body = json.dumps({"model": "higgs-tts-3", "input": text, "response_format": "pcm", "voice": "chloe"}).encode()
        req = urllib.request.Request("https://api.boson.ai/v1/audio/speech", data=body, headers={
            "Authorization": f"Bearer {os.environ['BOSON_API_KEY']}", "content-type": "application/json"})
        pcm = urllib.request.urlopen(req, timeout=60, context=CTX).read()
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm[:len(pcm) // 2 * 2])
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def get(base, path):
    return json.loads(urllib.request.urlopen(base + path, timeout=30).read())


class Bot:
    def __init__(self, base, rid, name):
        self.url = f"{base.replace('http', 'ws')}/ws/room?room={rid}&pid={name.lower()}&name={name}"
        self.name, self.events, self.ai_frames = name, [], []
        self.sending = asyncio.Lock()

    async def join(self):
        self.ws = await websockets.connect(self.url, max_size=None)
        self.reader = asyncio.ensure_future(self._read())
        self.idle = asyncio.ensure_future(self._room_noise())

    async def _read(self):
        async for m in self.ws:
            t = time.monotonic()
            if isinstance(m, bytes):
                if m[0] == 0:
                    self.ai_frames.append(t)
            else:
                self.events.append((t, json.loads(m)))

    async def _room_noise(self):
        rng = np.random.default_rng(abs(hash(self.name)) % 1000)
        t0, k = time.monotonic(), 0
        while True:
            if not self.sending.locked():
                await self.ws.send(rng.normal(0, 25, 480).astype(np.int16).tobytes())
            k += 1
            await asyncio.sleep(max(0, t0 + k * 0.02 - time.monotonic()))

    async def say(self, audio):
        async with self.sending:
            t0 = time.monotonic()
            for i, k in enumerate(range(0, len(audio) - 479, 480)):
                await self.ws.send(audio[k:k + 480].tobytes())
                await asyncio.sleep(max(0, t0 + (i + 1) * 0.02 - time.monotonic()))
            return time.monotonic()

    def ev(self, kind, since=0.0):
        return [e for t, e in self.events if e.get("type") == kind and t >= since]

    def t_of(self, kind, since=0.0, pred=lambda e: True):
        return next((t for t, e in self.events if e.get("type") == kind and t >= since and pred(e)), None)

    def ai_audio(self, a, b):
        return sum(1 for t in self.ai_frames if a <= t < b) * 0.02

    async def reply_after(self, t0, timeout=20, correction=None):
        """The first thing the AI said after t0, once it has finished speaking: (text, source) or (None, None)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            says = [(t, e) for t, e in self.events if e.get("type") == "ai_say" and t >= t0
                    and (correction is None or bool(e.get("correction")) == correction)]
            if says:
                t_say, e = says[0]
                if any(x.get("type") == "ai_state" and x.get("status") == "listening" and t > t_say
                       for t, x in self.events):
                    return e["text"], e.get("source")
            await asyncio.sleep(0.1)
        return None, None

    async def leave(self):
        self.idle.cancel()
        self.reader.cancel()
        await self.ws.close()


HOUSE = ["Rent is $2,400 a month, split equally ($800 each), due on the 1st to the landlord, Mrs. Okafor.",
         "Jordan pays the internet bill, $75 a month; Alex and Mei each owe Jordan $25.",
         "Trash goes out Monday and Thursday nights."]


async def room(base, people, facts=HOUSE, mode="improved"):
    r = post(base, "/api/rooms", {"title": "House meeting", "ai": "Higgs", "facts": facts, "mode": mode,
                                  "setting": "monthly house meeting of roommates"})
    bots = [Bot(base, r["id"], n) for n in people]
    for b in bots:
        await b.join()
    await bots[0].ws.send(json.dumps({"type": "set_mode", "mode": mode}))
    await asyncio.sleep(4)            # sessions open, the noise gate learns the room
    return r["id"], bots


class Score:
    def __init__(self, name):
        self.name, self.checks = name, []

    def check(self, ok, what, detail=""):
        self.checks.append(bool(ok))
        print(f"   {'PASS' if ok else 'FAIL'}  {what}{('  — ' + str(detail)) if detail else ''}", flush=True)

    @property
    def ok(self):
        return all(self.checks)


async def l1(base):
    sc = Score("L1 fillers never stop it; a real interruption does")
    _, (alex,) = await room(base, ["Alex"])
    end = await alex.say(tts("Higgs, how exactly are the rent and the internet bill split between the three of us?"))
    for _ in range(100):
        await asyncio.sleep(0.05)
        if alex.ai_audio(end, time.monotonic()) > 0.2:
            break
    sc.check(alex.ai_audio(end, time.monotonic()) > 0.2, "the AI started answering")
    t_f = time.monotonic()
    for f in FILLERS[:2]:
        await alex.say(tts(f))
        await asyncio.sleep(0.2)
    sc.check(not alex.ev("ai_flush", t_f), "the fillers did not stop it", f"{len(alex.ev('ai_flush', t_f))} stops")
    talking = alex.ai_audio(time.monotonic() - 0.6, time.monotonic())
    sc.check(talking > 0.3, "still talking after the fillers", f"{talking:.1f} s in the last 0.6 s")
    t_i = time.monotonic()
    await alex.say(tts("Wait, what about the internet bill?"))
    await asyncio.sleep(1.0)
    t_stop = alex.t_of("ai_flush", t_i)
    sc.check(t_stop is not None and t_stop - t_i < 1.5, "the interruption stopped it",
             f"{t_stop - t_i:.2f} s after the interruption started" if t_stop else "never stopped")
    text, _ = await alex.reply_after(t_i + 0.5)
    sc.check(text and ("internet" in text.lower() or "75" in text or "25" in text), "then it answered the interruption", text)
    await alex.leave()
    return sc


async def l2(base):
    sc = Score("L2 a wrong fact: cut in with the source, ducked, talk-over held, then answered")
    _, (alex, mei) = await room(base, ["Alex", "Mei"])
    t0 = time.monotonic()
    await mei.say(tts("Okay so rent is due on the fifteenth, so I'll transfer my eight hundred then."))
    for _ in range(80):
        await asyncio.sleep(0.1)
        if alex.ev("correction", t0):
            break
    fix = (alex.ev("correction", t0) or [None])[0]
    sc.check(fix is not None, "it corrected the wrong date", fix and fix["text"])
    if not fix:
        await alex.leave(), await mei.leave()
        return sc
    sc.check(fix.get("source"), "the correction names its source", fix.get("source"))
    for _ in range(40):  # ducking starts with the correction's first audio
        if alex.ev("human_gain", t0):
            break
        await asyncio.sleep(0.1)
    gains = alex.ev("human_gain", t0)
    sc.check(gains and gains[0]["gain"] < 0.5, "everyone's playback of the humans was ducked", gains and gains[0]["gain"])
    t_g = alex.t_of("human_gain", t0) or time.monotonic()
    await alex.say(tts("Hmm, are you sure about that?"))
    sc.check(not alex.ev("ai_flush", t_g), "talking over the correction did not stop it")
    for _ in range(150):
        if any(e["gain"] >= 1 for e in alex.ev("human_gain", t_g)):
            break
        await asyncio.sleep(0.1)
    t_rel = alex.t_of("human_gain", t_g, lambda e: e["gain"] >= 1)
    sc.check(t_rel is not None, "the volume came back after the correction")
    text, _ = await alex.reply_after(t_rel or time.monotonic(), correction=False)
    sc.check(text and ("1st" in text or "first" in text.lower()), "then it answered the doubt with the evidence", text)
    await alex.leave()
    await mei.leave()
    return sc


async def l3(base):
    sc = Score("L3 person vs person: a contradiction of what someone else said")
    _, (jordan, mei) = await room(base, ["Jordan", "Mei"], facts=[])
    await jordan.say(tts("Just so everyone knows, the plumber is coming on Friday morning at nine."))
    await asyncio.sleep(3)
    t0 = time.monotonic()
    await mei.say(tts("Right, and since the plumber is coming on Monday, I'll stay home on Monday."))
    for _ in range(80):
        await asyncio.sleep(0.1)
        if jordan.ev("correction", t0):
            break
    fix = (jordan.ev("correction", t0) or [None])[0]
    sc.check(fix is not None, "it pointed out the contradiction", fix and fix["text"])
    sc.check(fix and "jordan" in str(fix.get("source", "")).lower(), "and said it came from Jordan", fix and fix.get("source"))
    await jordan.leave()
    await mei.leave()
    return sc


LEASE = ("Residential lease for 14 Elm Street. Section 7, pets: no pets are allowed without the landlord's written "
         "approval. Section 9, guests: a guest staying more than 7 nights in a row needs the landlord's notice in "
         "advance. Section 11, repairs: tenants report repairs through the landlord's portal within 48 hours.")


async def l4(base):
    sc = Score("L4 an answer from an uploaded document, with its source")
    r = post(base, "/api/rooms", {"title": "House", "ai": "Higgs", "facts": [], "mode": "improved"})
    post(base, f"/api/rooms/{r['id']}/kb/sources", {"title": "Lease", "text": LEASE})
    for _ in range(80):
        st = get(base, f"/api/rooms/{r['id']}/kb")
        if st["docs"] and st["docs"][0]["status"] in ("ready", "error"):
            break
        await asyncio.sleep(0.5)
    sc.check(st["docs"][0]["status"] == "ready", "the document was indexed", f"{len(st['facts'])} facts extracted")
    alex = Bot(base, r["id"], "Alex")
    await alex.join()
    await alex.ws.send(json.dumps({"type": "set_mode", "mode": "improved"}))
    await asyncio.sleep(4)
    end = await alex.say(tts("Higgs, what does the lease say about having a dog?"))
    text, source = await alex.reply_after(end)
    sc.check(text and any(w in text.lower() for w in ("approval", "written", "permission")), "the answer comes from the lease", text)
    sc.check(source, "the answer names its source", source)
    await alex.leave()
    return sc


async def l5(base):
    sc = Score("L5 background noise makes no lines and no replies")
    _, (alex,) = await room(base, ["Alex"])
    rng = np.random.default_rng(3)
    t0 = time.monotonic()
    b = np.cumsum(rng.normal(0, 1, SR * 4))
    b = b - np.convolve(b, np.ones(2400) / 2400, "same")
    clicks = rng.normal(0, 30, SR * 3)
    for k in range(12):
        s = int((0.2 + k * 0.23) * SR)
        clicks[s:s + 240] += rng.normal(0, 9000, 240)
    for noise in ((rng.normal(0, 1, SR * 4) * 1500), b / np.abs(b).max() * 6000, clicks):
        await alex.say(noise.astype(np.int16))
    await asyncio.sleep(3)
    lines = alex.ev("line", t0)
    sc.check(not lines, "no transcript lines from noise", "; ".join(e["text"] for e in lines[:3]))
    sc.check(alex.ai_audio(t0, time.monotonic()) == 0, "the AI said nothing")
    await alex.leave()
    return sc


async def l6(base):
    sc = Score("L6 the live avatar over LiveKit (only while it's on), and back")
    from livekit import rtc
    rid, (alex,) = await room(base, ["Alex"])
    await alex.ws.send(json.dumps({"type": "set_avatar", "on": True}))
    for _ in range(100):
        await asyncio.sleep(0.2)
        if any(e.get("avatar") for e in alex.ev("ai_state")) or alex.ev("ai_error"):
            break
    sc.check(not alex.ev("ai_error"), "the avatar started", [e["error"] for e in alex.ev("ai_error")][:1])
    cred = post(base, f"/api/rooms/{rid}/livekit-token", {"pid": "alex", "name": "Alex"})
    lk, got = rtc.Room(), {"video": 0, "audio": [], "tracks": 0}

    async def pull(track):
        async for ev in rtc.AudioStream(track, sample_rate=24000, num_channels=1):
            if np.abs(np.frombuffer(bytes(ev.frame.data), np.int16)).max() > 300:
                got["audio"].append(time.monotonic())

    @lk.on("track_subscribed")
    def on_track(track, _pub, participant):
        if participant.identity != cred["avatar_identity"]:
            return
        got["tracks"] += 1
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            got["video"] += 1
        else:
            asyncio.ensure_future(pull(track))
    await lk.connect(cred["url"], cred["token"])
    for _ in range(150):
        if got["tracks"] >= 2:
            break
        await asyncio.sleep(0.2)
    sc.check(got["tracks"] >= 2, "the avatar published audio and video", f"identity {cred['avatar_identity']}")
    end = await alex.say(tts("Hey Higgs, when is the rent due, and who do we pay it to?"))
    text, _ = await alex.reply_after(end, timeout=30)
    first = next((t for t in got["audio"] if t > end), None)
    sc.check(text, "it answered", text)
    sc.check(first, "its voice came through the avatar", f"first avatar audio {first - end:.2f} s after" if first else "")
    sc.check(alex.ai_audio(end, time.monotonic()) == 0, "and not over the WebSocket")
    await alex.ws.send(json.dumps({"type": "set_avatar", "on": False}))
    await asyncio.sleep(2)
    end = await alex.say(tts("Higgs, what nights does the trash go out?"))
    text, _ = await alex.reply_after(end)
    sc.check(text and alex.ai_audio(end, time.monotonic()) > 0.5, "voice only again: the answer came over the WebSocket", text)
    await lk.disconnect()
    await alex.leave()
    return sc


FLAT = ("So I was reading about old navigation, and since the Earth is flat, ships could only sail so far before "
        "the edge, which is why old maps just stop, and honestly I think that explains a lot about history.")
DOG = ("So I checked the lease last night and pets are totally allowed without asking anyone, so I'm going to bring "
       "my dog over next weekend and he can stay here with us for a few weeks, that should be fine.")


async def speak_and_watch(speaker, watcher, line, t0):
    """Say a long line; return (when the correction came, when the speaker finished)."""
    task = asyncio.ensure_future(speaker.say(tts(line)))
    t_fix = None
    while not task.done():
        await asyncio.sleep(0.05)
        t_fix = t_fix or watcher.t_of("correction", t0)
    t_end = task.result()
    for _ in range(60):
        t_fix = t_fix or watcher.t_of("correction", t0)
        if t_fix:
            break
        await asyncio.sleep(0.1)
    return t_fix, t_end


async def l7(base):
    sc = Score("L7 a general-knowledge error is corrected mid-sentence")
    _, (alex, mei) = await room(base, ["Alex", "Mei"], facts=[])
    t0 = time.monotonic()
    t_fix, t_end = await speak_and_watch(alex, mei, FLAT, t0)
    fix = (mei.ev("correction", t0) or [None])[0]
    sc.check(fix is not None, "it corrected the flat Earth", fix and fix["text"])
    sc.check(fix and fix.get("source") == "general knowledge", "source: general knowledge", fix and fix.get("source"))
    sc.check(t_fix and t_fix < t_end, "it cut in while Alex was still talking",
             f"{t_end - t_fix:.1f} s before Alex finished" if t_fix and t_fix < t_end else "after Alex finished")
    await asyncio.sleep(6)
    ends = [e.get("reason") for e in mei.ev("human_gain", t0) if e["gain"] >= 1]
    sc.check(ends and ends[0] == "fact_correction_done", "the correction played to the end", ends[:1])
    await alex.leave()
    await mei.leave()
    return sc


async def l8(base):
    sc = Score("L8 a statement contradicting an uploaded document is corrected mid-sentence")
    r = post(base, "/api/rooms", {"title": "House", "ai": "Higgs", "facts": [], "mode": "improved"})
    post(base, f"/api/rooms/{r['id']}/kb/sources", {"title": "Lease", "text": LEASE})
    for _ in range(80):
        st = get(base, f"/api/rooms/{r['id']}/kb")
        if st["docs"] and st["docs"][0]["status"] in ("ready", "error"):
            break
        await asyncio.sleep(0.5)
    alex, mei = Bot(base, r["id"], "Alex"), Bot(base, r["id"], "Mei")
    await alex.join()
    await mei.join()
    await alex.ws.send(json.dumps({"type": "set_mode", "mode": "improved"}))
    await asyncio.sleep(4)
    t0 = time.monotonic()
    t_fix, t_end = await speak_and_watch(mei, alex, DOG, t0)
    fix = (alex.ev("correction", t0) or [None])[0]
    sc.check(fix is not None, "it corrected the pets claim", fix and fix["text"])
    sc.check(fix and "lease" in str(fix.get("source", "")).lower(), "source: the lease", fix and fix.get("source"))
    sc.check(t_fix and t_fix < t_end, "it cut in while Mei was still talking",
             f"{t_end - t_fix:.1f} s before Mei finished" if t_fix and t_fix < t_end else "after Mei finished")
    await asyncio.sleep(6)
    ends = [e.get("reason") for e in alex.ev("human_gain", t0) if e["gain"] >= 1]
    sc.check(ends and ends[0] == "fact_correction_done", "the correction played to the end", ends[:1])
    await alex.leave()
    await mei.leave()
    return sc


SCENARIOS = {"l1": l1, "l2": l2, "l3": l3, "l4": l4, "l5": l5, "l6": l6, "l7": l7, "l8": l8}
LINES = ["Higgs, how exactly are the rent and the internet bill split between the three of us?", *FILLERS,
         "Wait, what about the internet bill?", "Okay so rent is due on the fifteenth, so I'll transfer my eight hundred then.",
         "Hmm, are you sure about that?", "Just so everyone knows, the plumber is coming on Friday morning at nine.",
         "Right, and since the plumber is coming on Monday, I'll stay home on Monday.",
         "Higgs, what does the lease say about having a dog?", "Hey Higgs, when is the rent due, and who do we pay it to?",
         "Higgs, what nights does the trash go out?", FLAT, DOG]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[k for k in SCENARIOS if k != "l6"])
    ap.add_argument("--port", type=int, default=8723)
    args = ap.parse_args()
    for line in LINES:
        tts(line)
    import socket
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", args.port)) == 0:
            sys.exit(f"port {args.port} is in use")
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    proc = subprocess.Popen([sys.executable, "server.py", "--port", str(args.port)], cwd=HERE,
                            stdout=open(os.path.join(HERE, "out", "live3_server.log"), "w"), stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{args.port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(base + "/api/config", timeout=1)
            break
        except Exception:  # noqa: BLE001
            if proc.poll() is not None:
                sys.exit("server exited: see out/live3_server.log")
            time.sleep(0.2)
    results = []
    try:
        print("agent:", get(base, "/api/config").get("agent"))
        for name in args.only:
            print(f"\n== {name}", flush=True)
            sc = await SCENARIOS[name](base)
            results.append(sc)
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("\nSummary")
    for sc in results:
        print(f"  {'PASS' if sc.ok else 'FAIL'}  {sc.name}  ({sum(sc.checks)}/{len(sc.checks)})")
    sys.exit(0 if all(sc.ok for sc in results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
