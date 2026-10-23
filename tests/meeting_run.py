"""A scripted meeting through the real call with the real Higgs agent: 3 headless Chromium participants, each
playing one person's track from tests/make_tracks.py as its microphone, join at the same moment; the room runs in
one mode (default Higgs or improved) for the whole meeting; the room log is scored against the gold labels.

Makes Higgs calls (about 100 in default mode, ~270 in improved mode with MEET_SAMPLES=3); MEET_BUDGET caps them.
-> out/runs/<meeting>_<mode>_<time>.json (summary + per-line rows); prints the summary.
Usage: python tests/meeting_run.py --mode improved [--meeting B2] [--port 8620] [--samples 3] [--budget 600]"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from difflib import SequenceMatcher

from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
MEET = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from fake_room import CHROME, http, join  # noqa: E402

ADMIN = "run-admin"


def chrome_args(wav):
    return ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={wav}", "--autoplay-policy=no-user-gesture-required"]


def similar(a, b):
    norm = lambda x: re.sub(r"[^a-z0-9 ]", "", x.lower())  # noqa: E731
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def score(gold, log, mode, t0):
    """Match each gold line to the transcript line Higgs heard (same speaker, most similar text, in order)."""
    lines = [e for e in log if e.get("type") == "line"]
    says = [e for e in log if e.get("type") == "ai_say"]
    states = [e for e in log if e.get("type") == "ai_state"]
    rows, used = [], set()
    for g in gold["lines"]:
        cands = [(similar(g["text"], e["text"]), i) for i, e in enumerate(lines)
                 if i not in used and (mode == "default" or e.get("name") == g["speaker"])]
        best = max(cands, default=(0, None))
        e = lines[best[1]] if best[1] is not None and best[0] > 0.45 else None
        if e is not None:
            used.add(best[1])
        rows.append({"speaker": g["speaker"], "gold": g["label"], "category": g.get("category"), "said": g["text"],
                     "heard": e["text"] if e else None,
                     "speak": (e.get("decision") or {}).get("speak") if e else None,
                     "decided_s": (e.get("decision") or {}).get("decided_s") if e else None})
    played = [s for s in states if s.get("status") == "speaking"]
    stopped = [s for s in states if s.get("stopped")]
    dropped = [s for s in states if s.get("dropped")]
    errors = [e for e in log if e.get("type") == "ai_error"]
    out = {"mode": mode, "gold_lines": len(gold["lines"]), "heard_lines": sum(r["heard"] is not None for r in rows),
           "higgs_spoke": len(says), "stopped_by_humans": len(stopped), "dropped_before_playing": len(dropped),
           "calls": max([s.get("calls", 0) for s in states] or [0]), "errors": [e["error"] for e in errors][:5]}
    if mode == "improved":
        dec = [r for r in rows if r["speak"] is not None]
        tp = sum(r["speak"] and r["gold"] == "SPEAK" for r in dec)
        fn = sum((not r["speak"]) and r["gold"] == "SPEAK" for r in dec)
        tn = sum((not r["speak"]) and r["gold"] == "SILENT" for r in dec)
        fp = sum(r["speak"] and r["gold"] == "SILENT" for r in dec)
        out.update(spoke_when_it_should=f"{tp}/{tp + fn}", silent_when_it_should=f"{tn}/{tn + fp}",
                   balanced_acc=round((tp / max(1, tp + fn) + tn / max(1, tn + fp)) / 2, 3),
                   after_s=[s.get("after_s") for s in played if s.get("after_s") is not None],
                   decided_s=[r["decided_s"] for r in dec if r["decided_s"] is not None])
    return out, rows


async def run(a):
    gold = json.load(open(os.path.join(HERE, "tracks", a.meeting, "gold.json")))
    env = {**os.environ, "MEET_ADMIN": ADMIN, "MEET_BUDGET": str(a.budget), "MEET_SAMPLES": str(a.samples)}
    env.pop("MEET_AGENT", None)
    logf = f"/tmp/meet_run_{a.port}.log"
    srv = subprocess.Popen([sys.executable, "server.py", "--port", str(a.port)], cwd=MEET, env=env,
                           stdout=open(logf, "w"), stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{a.port}"
    try:
        for _ in range(100):
            try:
                http(base + "/api/config")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        _, r = http(base + "/api/rooms", {"admin": ADMIN, "title": f"test {a.meeting}", "ai": gold["ai"],
                                          "setting": gold["setting"], "facts": gold["facts"], "passcode": "pw"})
        rid = r["id"]
        async with async_playwright() as pw:
            browsers, pages = [], []
            for p in gold["people"]:
                b = await pw.chromium.launch(executable_path=CHROME,
                                             args=chrome_args(os.path.join(HERE, "tracks", a.meeting, f"{p}.wav")))
                browsers.append(b)
                pages.append(await (await b.new_context()).new_page())
            t0 = time.time()
            await asyncio.gather(*[join(pg, f"{base}/r/{rid}?k=pw&lab", p) for pg, p in zip(pages, gold["people"])])
            print(f"joined {len(pages)} at +{time.time() - t0:.1f}s; room {rid}", flush=True)
            if a.mode == "improved":
                await asyncio.sleep(2)
                await pages[0].click("#modesw")
            if a.avatar:
                await asyncio.sleep(1)
                await pages[1].click("#avatar")
            switched = a.switch_at is None
            end = gold["lines"][-1]["end"] + a.tail
            shot = False
            while time.time() - t0 < end:
                await asyncio.sleep(0.5)
                if not switched and time.time() - t0 >= a.switch_at:  # someone flips the switch mid-meeting
                    await pages[2].click("#modesw")
                    switched = True
                    print(f"  switched at +{time.time() - t0:.0f}s", flush=True)
                if a.avatar and not shot and await pages[2].evaluate("document.querySelector('.av-on') !== null"):
                    await asyncio.sleep(1.0)  # mid-reply: the face talking on everyone's Higgs tile
                    await pages[2].screenshot(path=os.path.join(HERE, "shot_run_avatar.png"))
                    shot = True
                if int(time.time() - t0) % 20 == 0:
                    print(f"  +{time.time() - t0:.0f}s / {end:.0f}s", flush=True)
            if a.avatar:
                av = [await pg.evaluate("meet.avatar") for pg in pages]
                print("avatar per page:", av, flush=True)
            await pages[0].screenshot(path=os.path.join(HERE, f"shot_run_{a.mode}.png"))
            for b in browsers:
                await b.close()
        await asyncio.sleep(1)
    finally:
        srv.terminate()
    log = [json.loads(x) for x in open(os.path.join(MEET, "out", "rooms", f"{rid}.jsonl"))]
    summary, rows = score(gold, log, a.mode, t0)
    if a.switch_at is not None:  # per mode: what Higgs said, and how often it was cut off
        for m in ("default", "improved"):
            st = [e for e in log if e.get("type") == "ai_state" and e.get("mode") == m]
            summary[f"{m}_spoke"] = sum(s.get("status") == "speaking" for s in st)
            summary[f"{m}_stopped"] = sum(bool(s.get("stopped")) for s in st)
            summary[f"{m}_lines"] = sum(e.get("type") == "line" and e.get("mode") == m for e in log)
        summary["switch_states"] = [s.get("status") for s in log if s.get("type") == "ai_state"
                                    and s.get("status") == "switching"]
    os.makedirs(os.path.join(MEET, "out", "runs"), exist_ok=True)
    dst = os.path.join(MEET, "out", "runs", f"{a.meeting}_{a.mode}_{time.strftime('%H%M%S')}.json")
    json.dump({"summary": summary, "rows": rows, "room": rid}, open(dst, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("->", dst, "| server log:", logf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["default", "improved"], default="improved")
    ap.add_argument("--meeting", default="B2")
    ap.add_argument("--port", type=int, default=8620)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--tail", type=float, default=20.0)
    ap.add_argument("--avatar", action="store_true", help="turn the Higgs Avatar on (improved mode)")
    ap.add_argument("--switch-at", type=float, default=None, help="flip the mode switch this many seconds in")
    asyncio.run(run(ap.parse_args()))
