"""Does a listener's "mm-hm" stop Higgs when it comes through a real browser (Chrome's echo cancellation, noise
suppression and auto gain on the mic), not only over the bot socket? One headless Chromium participant whose fake
microphone plays rounds of [a question that gets a long answer, then a backchannel `--cue-at` s after the question
ends, then quiet]. The room log says whether Higgs stopped after each cue.

Usage: python tests/browser_bc.py --mode default [--cue-at 3.0] [--gain-db 0] [--port 8660]"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
MEET = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from bots import SR, ami, clip, create_room, http  # noqa: E402
from fake_room import CHROME, join  # noqa: E402

ADMIN = "bc-admin"
QS = ["Higgs, can you explain in a few sentences how we should structure the demo video?",
      "Higgs, how should we split the work between the three of us this week?",
      "Higgs, what's the riskiest part of our plan, and why?",
      "Higgs, what should the last thirty seconds of the video show?",
      "Higgs, how do we make the judges remember us?",
      "Higgs, walk me through what happens when someone joins the call.",
      "Higgs, what would you cut if we run out of time?",
      "Higgs, how should we test it before Saturday?"]
CUES = [("tts", "Mm-hm."), ("ami", "bc_00.wav"), ("tts", "Yeah."), ("ami", "bc_03.wav"), ("tts", "Uh-huh."),
        ("ami", "bc_05.wav"), ("tts", "Right."), ("ami", "bc_07.wav")]


def track(a):
    parts, t, plan = [np.zeros(int(a.lead * SR), np.int16)], a.lead, []
    for q, (kind, c) in zip(QS, CUES):
        qa = clip(q, "marcus")
        cue = clip(c, "marcus") if kind == "tts" else ami(c)
        if a.gain_db:
            cue = np.clip(cue.astype(np.float32) * 10 ** (a.gain_db / 20), -32768, 32767).astype(np.int16)
        parts += [qa, np.zeros(int(a.cue_at * SR), np.int16), cue, np.zeros(int(14 * SR), np.int16)]
        plan.append({"q": q, "cue": c, "q_end": round(t + len(qa) / SR, 2), "cue_t": round(t + len(qa) / SR + a.cue_at, 2)})
        t += (len(qa) + int(a.cue_at * SR) + len(cue) + int(14 * SR)) / SR
    pcm = np.concatenate(parts + [np.zeros(int(120 * SR), np.int16)])
    path = os.path.join(MEET, "out", f"bc_track_{a.gain_db:+.0f}.wav")
    sf.write(path, resample_poly(pcm.astype(np.float32) / 32768, 2, 1), 48000, subtype="PCM_16")
    return path, plan, t


async def run(a):
    from playwright.async_api import async_playwright
    path, plan, total = track(a)
    env = {**os.environ, "MEET_ADMIN": ADMIN, "MEET_BUDGET": "3000"}
    env.pop("MEET_AGENT", None)
    srv = subprocess.Popen([sys.executable, "server.py", "--port", str(a.port)], cwd=MEET, env=env,
                           stdout=open(os.path.join(MEET, "out", f"browser_bc_{a.port}.log"), "w"), stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{a.port}"
    try:
        for _ in range(300):
            try:
                http(base + "/api/config")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        rid = create_room(base, ADMIN, title="browser bc", setting="hackathon team meeting",
                          facts=["Submissions close Saturday at 3 pm."])
        async with async_playwright() as pw:
            b = await pw.chromium.launch(executable_path=CHROME, args=[
                "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={path}", "--autoplay-policy=no-user-gesture-required"])
            page = await (await b.new_context()).new_page()
            t0 = time.time()
            await join(page, f"{base}/r/{rid}?k=pw", "Alex")
            if a.mode == "improved":
                await asyncio.sleep(1)
                await page.click("#modesw")
            while time.time() - t0 < total + 5:
                await asyncio.sleep(1)
            await b.close()
    finally:
        srv.terminate()
    log = [json.loads(x) for x in open(os.path.join(MEET, "out", "rooms", f"{rid}.jsonl"))]
    stops = [e for e in log if e.get("type") == "ai_state" and e.get("stopped")]
    lines = [e["text"] for e in log if e.get("type") == "line"]
    says = [e for e in log if e.get("type") == "ai_say"]
    res = {"mode": a.mode, "gain_db": a.gain_db, "cue_at": a.cue_at, "rounds": len(plan), "replies": len(says),
           "stops": len(stops), "stopped_why": [s.get("stopped") for s in stops], "lines": lines, "room": rid}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(MEET, "out", f"browser_bc_{a.mode}_{a.gain_db:+.0f}.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="default")
    ap.add_argument("--cue-at", type=float, default=3.0)
    ap.add_argument("--gain-db", type=float, default=0.0)
    ap.add_argument("--lead", type=float, default=12.0)
    ap.add_argument("--port", type=int, default=8660)
    asyncio.run(run(ap.parse_args()))
