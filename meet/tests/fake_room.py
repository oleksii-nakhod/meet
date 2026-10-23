"""Call-layer test without humans: 3 headless Chromium pages with different fake voices (and the fake camera) join
one room; the stub agent (MEET_AGENT=stub, no Higgs calls) plays slot 0. Then the same over --https.

Usage: python tests/fake_room.py [--port 8611] [--url http://host:port (use a running server; needs MEET_AGENT=stub)]
       [--no-https]
Writes tests/shot_room.png. Exit code 1 if any check fails."""
import argparse
import asyncio
import json
import os
import ssl
import subprocess
import sys
import time
import urllib.request

import websockets
from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
MEET = os.path.dirname(HERE)
sys.path.insert(0, MEET)
WAVS = [os.path.join(MEET, "..", "mmhm", "tests", w) for w in ("fake_mic.wav", "demo_mic.wav", "fake_rule.wav")]
CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome")
PY = sys.executable
ADMIN = "test-admin"
FMP4 = "/tmp/meet_test_fmp4.mp4"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def voice(wav):
    """The fake-mic wavs are ~3 s of speech then long silence; for the test, loop just the speech (30 s)."""
    import wave
    import numpy as np
    out = f"/tmp/meet_voice_{os.path.basename(wav)}"
    if not os.path.exists(out):
        with wave.open(wav) as w:
            sr, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), "<i2")
        f = x[:len(x) // 480 * 480].reshape(-1, 480)
        x = f[np.abs(f).max(1) > 300].ravel()
        x = np.tile(x, int(30 * sr / len(x)) + 1)
        with wave.open(out, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
            w.writeframes(x.astype("<i2").tobytes())
    return out


def args(wav):
    return ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={voice(wav)}", "--autoplay-policy=no-user-gesture-required"]


def http(url, body=None, headers=None):
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def start_server(port, https=False):
    if not os.path.exists(FMP4):  # a 3 s H.264 baseline fragmented MP4 for the avatar path
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        "testsrc=size=320x240:rate=25", "-t", "3", "-c:v", "libx264", "-profile:v", "baseline",
                        "-level", "3.0", "-pix_fmt", "yuv420p", "-g", "25", "-movflags",
                        "frag_keyframe+empty_moov+default_base_moof", FMP4], check=False)
    env = {**os.environ, "MEET_AGENT": "stub", "MEET_ADMIN": ADMIN, "MEET_STUB_FMP4": FMP4}
    p = subprocess.Popen([PY, "server.py", "--port", str(port)] + (["--https"] if https else []), cwd=MEET, env=env,
                         stdout=open(f"/tmp/meet_server_{port}.log", "w"), stderr=subprocess.STDOUT)
    p.log = f"/tmp/meet_server_{port}.log"
    base = f"{'https' if https else 'http'}://127.0.0.1:{port}"
    for _ in range(100):
        try:
            http(base + "/api/config")
            return p, base
        except Exception:
            time.sleep(0.2)
    p.kill()
    raise RuntimeError("server didn't start: " + open(p.log).read())


async def ev(page, js):
    return await page.evaluate(js)


async def wait(page, js, timeout=10000):
    try:
        await page.wait_for_function(js, timeout=timeout)
        return True
    except Exception:
        return False


async def join(page, url, name):
    await page.goto(url)
    await page.fill("#jname", name)
    await page.click("#jgo")
    await page.wait_for_selector("#call:not(.hidden)", timeout=10000)


async def chat(page, text):
    await page.fill("#chatin", text)
    await page.press("#chatin", "Enter")


async def main_test(pw, base, wsbase):
    # rooms need the admin token
    st, _ = http(base + "/api/rooms", {"title": "x"})
    check("POST /api/rooms without admin token -> 403", st == 403, st)
    st, d = http(base + "/api/rooms", {"title": "Fake room", "ai": "Higgs", "setting": "a test call",
                                       "facts": ["The demo is on Friday."], "passcode": "pw"}, {"X-Admin": ADMIN})
    check("POST /api/rooms with X-Admin -> id + link", st == 200 and d["link"] == f"/r/{d['id']}", d)
    rid = d["id"]
    st, info = http(f"{base}/api/rooms/{rid}")
    check("GET /api/rooms/<id> says passcode needed", st == 200 and info["passcode"] is True, info)

    browsers = [await pw.chromium.launch(executable_path=CHROME, args=args(w)) for w in WAVS]
    ctxs = [await b.new_context(viewport={"width": 1280, "height": 800}) for b in browsers]
    pages = [await c.new_page() for c in ctxs]
    errors = {i: [] for i in range(3)}
    for i, pg in enumerate(pages):
        pg.on("pageerror", lambda e, i=i: errors[i].append(str(e)))
        pg.on("console", lambda m, i=i: m.type == "error" and print(f"  page{i} console error:", m.text))
    names = ["Ana", "Ben", "Chen"]
    for i, pg in enumerate(pages):
        await join(pg, f"{base}/r/{rid}?k=pw" + ("&lab" if i == 0 else ""), names[i])
    A, B, C = pages

    ok = all([await wait(p, "meet.roster && meet.roster.people.length === 3") for p in pages])
    tiles = [await p.eval_on_selector_all("#grid .tile", "els => els.length") for p in pages]
    ai_tile = [await p.eval_on_selector_all("#grid .tile.ai", "els => els.length") for p in pages]
    check("roster shows 3 people + Higgs tile on every page", ok and tiles == [4, 4, 4] and ai_tile == [1, 1, 1], f"tiles {tiles}")
    slots = [await ev(p, "meet.slot") for p in pages]
    check("slots 1, 2, 3 assigned in join order", slots == [1, 2, 3], slots)

    await A.wait_for_timeout(5000)
    for i, p in enumerate(pages):
        rx = await ev(p, "meet.audioRx")
        others = [s for s in slots if s != slots[i]]
        check(f"page {names[i]} gets audio frames from both others", all(rx.get(str(s), 0) > 100 for s in others)
              and rx.get(str(slots[i]), 0) == 0, rx)
        vx = await ev(p, "meet.videoRx")
        check(f"page {names[i]} gets camera frames from both others", all(vx.get(str(s), 0) >= 10 for s in others), vx)
        peak = await ev(p, "meet.peak")
        check(f"page {names[i]} mixer plays the others' voices", all(peak[s] > 0.01 for s in others), [round(x, 3) for x in peak])
        imgs = await p.eval_on_selector_all(".tile.has-video:not(.self) img", "els => els.filter(e => e.naturalWidth === 320).length")
        check(f"page {names[i]} shows 2 remote 320x240 camera images", imgs == 2, imgs)
    rx0 = await ev(A, "meet.audioRx[0] || 0")
    check("stub beep on start reached the first joiner on slot 0", rx0 >= 25, rx0)

    await chat(B, "hello from Ben")
    ok = all([await wait(p, "[...document.querySelectorAll('#chat li')].some(l => l.textContent.includes('hello from Ben'))") for p in pages])
    ok2 = all([await wait(p, "[...document.querySelectorAll('#chat li')].some(l => l.textContent.includes('heard hello from Ben'))") for p in pages])
    check("chat roundtrip to all pages (input cleared)", ok and await B.input_value("#chatin") == "")
    check("stub agent's chat reply (Higgs (stub): heard ...) on all pages", ok2)

    before = [await ev(p, "meet.audioRx[0] || 0") for p in pages]
    await chat(C, "/beep")
    await A.wait_for_timeout(1500)
    after = [await ev(p, "meet.audioRx[0] || 0") for p in pages]
    peak0 = [await ev(p, "meet.peak[0]") for p in pages]
    check("/beep: slot-0 frames received on all pages", all(a - b >= 25 for a, b in zip(after, before)), [a - b for a, b in zip(after, before)])
    check("/beep: slot 0 audible in the mixer on all pages", all(x > 0.1 for x in peak0), [round(x, 3) for x in peak0])

    await chat(C, "/gain 0.3")
    ok = all([await wait(p, "meet.msgs.ai_gain >= 1") for p in pages])
    await chat(C, "/beep")
    await A.wait_for_timeout(1200)
    g = [await ev(p, "meet.gain0") for p in pages]
    check("ai_gain 0.3 applied in the worklet on all pages", ok and all(abs(x - 0.3) < 0.01 for x in g), g)
    await chat(C, "/flush")
    await chat(C, "/gain 1")
    ok = all([await wait(p, "meet.msgs.ai_flush >= 1 && meet.gain0 > 0.99") for p in pages])
    check("ai_flush and ai_gain back to 1 handled", ok)

    await chat(A, "/line Higgs, are you there?")
    ok = all([await wait(p, "[...document.querySelectorAll('#lines li')].some(l => l.textContent.includes('are you there') && l.textContent.includes('SPEAK') && l.textContent.includes('asked me'))") for p in pages])
    check("transcript line with Higgs decision on all pages", ok)
    await chat(A, "/say Nice to meet you all.")
    ok = all([await wait(p, "document.querySelector('.tile.ai .cap').textContent.includes('Nice to meet you')") for p in pages])
    check("ai_say caption on the Higgs tile", ok)
    await chat(A, "/lab")
    ok = await wait(A, "[...document.querySelectorAll('.tile .lab i')].some(i => parseFloat(i.style.width) > 0)")
    nolab = await ev(B, "document.querySelectorAll('.tile .lab').length")
    check("lab bars on tiles with ?lab (and none without)", ok and nolab == 0, nolab)

    await B.click("#modesw")
    ok = all([await wait(p, "document.querySelector('.tile.ai .badge').textContent === 'Mm-hm + Third Voice'") for p in pages])
    toast = await wait(A, "[...document.querySelectorAll('.toast')].some(t => t.textContent.includes('Ben switched Higgs to Mm-hm + Third Voice'))", 3000)
    check("mode switch: badge 'Mm-hm + Third Voice' on all pages + toast", ok and toast)
    shot_lab = A
    await A.wait_for_timeout(300)
    await shot_lab.screenshot(path=os.path.join(HERE, "shot_room.png"))
    await B.click("#modesw")
    ok = all([await wait(p, "document.querySelector('.tile.ai .badge').textContent === 'default Higgs'") for p in pages])
    check("mode switch back: badge 'default Higgs' on all pages", ok)

    await C.click("#avatar")
    ok = all([await wait(p, "meet.ai.avatar === true && document.querySelector('#avatar').textContent === 'Avatar: on'") for p in pages])
    check("avatar toggle reaches all pages", ok)
    supported = await ev(A, "MediaSource.isTypeSupported('video/mp4; codecs=\"avc1.42E01E\"')")
    await chat(C, "/avatar")
    ok = all([await wait(p, "meet.avatar.starts >= 1 && meet.avatar.chunks >= 1") for p in pages])
    played = all([await wait(p, "meet.avatar.played >= 1", 6000) for p in pages]) if supported else None
    check("avatar fMP4: avatar_start + chunks on all pages", ok)
    check(f"avatar fMP4 plays in the Higgs tile (MSE H.264 supported: {supported})", played or not supported, played)
    await C.click("#avatar")

    # network drop on the control socket: the page reconnects by itself, same slot, video keeps flowing both ways
    v_in, v_out = await ev(A, "meet.videoRx[2] || 0"), await ev(B, "meet.videoRx[1] || 0")
    await ev(A, "ws.close()")
    ok = await wait(A, "meet.msgs.roster >= 1 && ws.readyState === 1", 8000)
    await A.wait_for_timeout(3000)
    v_in2, v_out2 = await ev(A, "meet.videoRx[2] || 0"), await ev(B, "meet.videoRx[1] || 0")
    check("control socket drop: auto-reconnect, same slot, video both ways after", ok and await ev(A, "meet.slot") == 1
          and v_in2 - v_in > 5 and v_out2 - v_out > 5 and await ev(A, "meet.videoRx[1] || 0") == 0, (v_in2 - v_in, v_out2 - v_out))

    # reload = same pid (sessionStorage) -> same slot back
    await C.reload()
    await C.fill("#jname", "Chen")
    await C.click("#jgo")
    await C.wait_for_selector("#call:not(.hidden)")
    ok = await wait(A, "meet.roster.people.length === 3")
    check("reload rejoins with the same slot", ok and await ev(C, "meet.slot") == slots[2], await ev(C, "meet.slot"))
    # leave -> slot freed; a new person gets the lowest free slot
    await B.click("#leave")
    ok = await wait(A, "meet.roster.people.length === 2")
    tl = await wait(A, "[...document.querySelectorAll('.toast')].some(t => t.textContent.includes('Ben left'))", 3000)
    check("leave: roster drops to 2 (+ toast)", ok and tl)
    d2 = await browsers[1].new_context(viewport={"width": 390, "height": 844})
    D = await d2.new_page()
    D.on("pageerror", lambda e: errors[1].append(str(e)))
    await join(D, f"{base}/r/{rid}?k=pw", "Dee")
    ok = await wait(A, "meet.roster.people.length === 3")
    check("new person reuses the freed slot 2", ok and await ev(D, "meet.slot") == 2, await ev(D, "meet.slot"))
    await D.wait_for_timeout(2500)
    vx = await ev(D, "meet.videoRx")
    check("rejoined slot gets audio + video", (await ev(D, "meet.audioRx[1] || 0")) > 50 and vx.get("1", 0) > 5, vx)
    await D.screenshot(path="/tmp/meet_shot_phone.png", full_page=True)

    # raw websocket clients: frame format, 5-person cap, passcode
    sslctx = ssl._create_unverified_context() if wsbase.startswith("wss") else None
    extra = [await websockets.connect(f"{wsbase}/ws/room?room={rid}&pid=x{i}&name=Bot{i}&k=pw", ssl=sslctx) for i in range(2)]
    frames = []
    t0 = time.time()
    while time.time() - t0 < 2 and len(frames) < 20:
        m = await asyncio.wait_for(extra[0].recv(), 2)
        if isinstance(m, bytes):
            frames.append(m)
    check("server audio frame = 1 byte slot + 960 bytes", frames and all(len(f) == 961 and 1 <= f[0] <= 5 for f in frames),
          sorted({f[0] for f in frames}))
    code = None
    try:
        w6 = await websockets.connect(f"{wsbase}/ws/room?room={rid}&pid=x9&name=Six&k=pw", ssl=sslctx)
        await asyncio.wait_for(w6.recv(), 3)
    except websockets.ConnectionClosed as e:
        code = e.rcvd.code if e.rcvd else None
    check("6th human refused (close 4001)", code == 4001, code)
    code = None
    try:
        w = await websockets.connect(f"{wsbase}/ws/room?room={rid}&pid=y&name=Bad&k=nope", ssl=sslctx)
        await asyncio.wait_for(w.recv(), 3)
    except websockets.ConnectionClosed as e:
        code = e.rcvd.code if e.rcvd else None
    check("wrong passcode refused (close 4003)", code == 4003, code)
    for w in extra:
        await w.close()

    werr = [await ev(p, "meet.errors") for p in (A, C, D)]
    check("no page errors", not any(errors.values()) and not any(werr), (errors, werr))
    for b in browsers:
        await b.close()
    log = os.path.join(MEET, "out", "rooms", f"{rid}.jsonl")
    types = [json.loads(x)["type"] for x in open(log)]
    check("room log has join/chat/line/ai_state/leave", all(t in types for t in ("join", "chat", "line", "ai_state", "leave")), log)


async def https_test(pw, base, wsbase):
    st, d = http(base + "/api/rooms", {"title": "TLS room", "admin": ADMIN})
    rid = d["id"]
    browsers = [await pw.chromium.launch(executable_path=CHROME, args=args(w)) for w in WAVS[:2]]
    pages = [await (await b.new_context(ignore_https_errors=True)).new_page() for b in browsers]
    errs = []
    for p in pages:
        p.on("pageerror", lambda e: errs.append(str(e)))
    for p, n in zip(pages, ("Tia", "Uri")):
        await join(p, f"{base}/r/{rid}", n)
    ok = all([await wait(p, "meet.roster && meet.roster.people.length === 2") for p in pages])
    await pages[0].wait_for_timeout(3000)
    a = [await ev(p, "meet.audioRx") for p in pages]
    v = [await ev(p, "meet.videoRx") for p in pages]
    check("HTTPS: 2 pages join, roster 2", ok)
    check("HTTPS: audio both ways over wss", a[0].get("2", 0) > 50 and a[1].get("1", 0) > 50, a)
    check("HTTPS: video both ways over wss", v[0].get("2", 0) > 5 and v[1].get("1", 0) > 5, v)
    check("HTTPS: no page errors", not errs, errs)
    for b in browsers:
        await b.close()


async def amain(a):
    procs = []
    try:
        if a.url:
            base = a.url.rstrip("/")
        else:
            p, base = start_server(a.port)
            procs.append(p)
        async with async_playwright() as pw:
            await main_test(pw, base, base.replace("http", "ws", 1))
            if not a.no_https:
                p, hbase = start_server(a.port + 1, https=True)
                procs.append(p)
                await https_test(pw, hbase, hbase.replace("http", "ws", 1))
    finally:
        for i, p in enumerate(procs):
            p.terminate()
            p.wait(timeout=5)
            out = open(p.log).read()
            if "Traceback" in out or "Error" in out:
                print("server output:\n" + out)
            if i == 0:
                check("server log prints the room link with ?k=<passcode>", "?k=pw" in out,
                      [x for x in out.splitlines() if "?k=" in x][:1])


def unit_out():
    """Out queues without a network: a client that never drains keeps <= 15 audio frames and one video frame per sender."""
    from room import Out

    class Stuck:
        async def send_bytes(self, b):
            await asyncio.sleep(3600)
        send_text = send_bytes

        async def close(self, code=1000):
            pass

    async def run():
        o = Out(Stuck())
        await asyncio.sleep(0)
        for _ in range(100):
            o.lossy(b"a")
        for i in range(10):
            o.latest(3, bytes([i]))
        o.send("ctl")
        n, dropped, latest = o.n_lossy, o.dropped, dict(o.latest_q)
        o.close()
        o.task.cancel()
        return n, dropped, latest

    n, dropped, latest = asyncio.run(run())
    check("slow client: audio queue bounded (<=15 queued, rest dropped)", n <= 15 and dropped >= 84, (n, dropped))
    check("slow client: one pending video frame per sender, newest wins", latest == {3: bytes([9])}, latest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8611)
    ap.add_argument("--url")
    ap.add_argument("--no-https", action="store_true")
    a = ap.parse_args()
    os.environ["MEET_AGENT"] = "stub"
    unit_out()
    asyncio.run(amain(a))
    bad = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)
