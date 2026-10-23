"""A real meeting through the call: N minutes of an AMI scenario meeting (four people designing a TV remote), each
person a bot playing their own headset channel in real time (muted outside their own words, so it sounds like four
people on four mics, with the real overlaps, backchannels and laughter). Nobody in AMI talks to an AI, and nobody
reacts to Higgs, so this measures false alarms: how often Higgs speaks, after what, and whether it talks over anyone.
Higgs knows the project brief the AMI participants were given (third_voice/ami_chatty.py).

Usage: python tests/ami_meeting.py --meeting ES2009c --minutes 10 [--start 0] [--mode improved] [--port 8731]
-> out/ami_meeting_<meeting>_<mode>.json"""
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
sys.path.insert(0, os.path.join(MEET, "..", "mmhm", "eval"))
sys.path.insert(0, os.path.join(MEET, "..", "third_voice"))
from bots import SR, Bot, clip, create_room, http  # noqa: E402

ADMIN = "ami-admin"
NAMES = {"A": "Ana", "B": "Ben", "C": "Cleo", "D": "Dev"}
# --ask: a fifth person, Eve, asks these in the meeting's lulls (what a good answer contains, lowercase)
ASKS = [("Higgs, what's our production cost limit again?", ["12.5", "twelve", "12"]),
        ("Does anyone remember what price the remote will sell for?", ["25", "twenty-five", "twenty five"]),
        ("So we can spend up to twenty euros per remote, right?", ["12.5", "twelve", "12"]),
        ("Higgs, remind me, are we still doing teletext?", ["no", "only", "tv", "outdated"]),
        ("Higgs, what's the profit target?", ["50", "fifty"])]


def lulls(tracks, t0, t1, need=2.0, gap=40.0):
    """Start times (s from t0) of stretches of at least `need` s in which nobody says a word, `gap` s apart."""
    words = sorted((w["s"], w["e"]) for _, ws in tracks.values() for w in ws)
    out, last_end, prev = [], t0, -1e9
    for s, e in words + [(t1, t1)]:
        if s - last_end >= need and last_end - t0 - prev >= gap and last_end - t0 > 20:
            out.append(round(last_end - t0 + 0.4, 2))
            prev = out[-1]
        last_end = max(last_end, e)
    return out


def track(meeting, spk, ch, t0, t1):
    from ami_stimuli import AMI, words
    x, sr = sf.read(f"{AMI}/amicorpus/{meeting}/audio/{meeting}.Headset-{ch}.wav", dtype="float32", start=int(t0 * 16000),
                    stop=int(t1 * 16000))
    mask = np.zeros(len(x), bool)
    ws = [w for w in words(meeting, spk) if w["e"] >= t0 and w["s"] <= t1]
    for w in ws:
        mask[max(0, int((w["s"] - t0 - 0.2) * sr)):int((w["e"] - t0 + 0.2) * sr)] = True
    x = np.where(mask, x, x * 0.003)  # the other headsets' bleed: as if each person had their own room
    x = resample_poly(x, SR, sr)
    sp = x[np.abs(x) > 1e-3]
    x = x / (np.sqrt(np.mean(sp ** 2)) + 1e-9) * 0.08 if len(sp) else x  # ~ -22 dBFS speech, like a browser's AGC
    return (np.clip(x, -1, 1) * 32767).astype(np.int16), ws


async def run(a):
    from ami_chatty import FACTS
    t0, t1 = a.start * 60, (a.start + a.minutes) * 60
    tracks = {}
    for ch, spk in enumerate("ABCD"):
        try:
            tracks[spk] = track(a.meeting, spk, ch, t0, t1)
        except (FileNotFoundError, RuntimeError):
            pass
    env = {**os.environ, "MEET_ADMIN": ADMIN, "MEET_BUDGET": "20000"}
    env.pop("MEET_AGENT", None)
    srv = subprocess.Popen([sys.executable, "server.py", "--port", str(a.port)], cwd=MEET, env=env,
                           stdout=open(os.path.join(MEET, "out", f"ami_meeting_{a.port}.log"), "w"), stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{a.port}"
    try:
        for _ in range(300):
            try:
                http(base + "/api/config")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        rid = create_room(base, ADMIN, title=f"AMI {a.meeting}", ai=a.ai, setting="product design meeting", facts=FACTS)
        bs = {spk: Bot(base, rid, NAMES[spk]) for spk in tracks}
        plan = []
        if a.ask:  # Eve's track: silence, with a question at the start of each lull
            starts = lulls(tracks, t0, t1)[:a.ask]
            n = int((t1 - t0) * SR)
            eve = np.zeros(n, np.int16)
            for (q, want), at in zip(ASKS, starts):
                c = clip(q, "eleanor")
                i = int(at * SR)
                eve[i:i + len(c)] = c[:max(0, n - i)]
                plan.append({"q": q, "want": want, "at": at, "end": round(at + len(c) / SR, 2)})
            tracks["E"] = (eve, [])
            NAMES["E"] = "Eve"
            bs["E"] = Bot(base, rid, "Eve")
        for b in bs.values():
            await b.join()
            await asyncio.sleep(0.2)
        ear = next(iter(bs.values()))
        await asyncio.sleep(1.0)
        if a.mode != "default":
            await ear.json({"type": "set_mode", "mode": a.mode})
        await asyncio.sleep(8.0)
        start = ear.now()
        done = [bs[spk].say(pcm) for spk, (pcm, _) in tracks.items()]  # everyone starts together
        await asyncio.gather(*done)
        await asyncio.sleep(6.0)
        spans = ear.ai_onsets(start)
        for p in plan:  # did Higgs start talking within 6 s of the question's end?
            on = [sp for sp in spans if p["end"] - 0.5 <= sp[0] - start <= p["end"] + 6.0]
            p["answered_after"] = round(on[0][0] - start - p["end"], 2) if on else None
        # talking over someone: Higgs audible while one of the real speakers is mid-word
        talk = {spk: [(w["s"] - t0 + start, w["e"] - t0 + start) for w in ws] for spk, (_, ws) in tracks.items()}
        over = []
        for s, e in spans:
            for spk, iv in talk.items():
                ov = sum(max(0.0, min(e, b) - max(s, a_)) for a_, b in iv)
                if ov > 0.3:
                    over.append({"ai_at": round(s - start, 1), "who": NAMES[spk], "overlap_s": round(ov, 2)})
        for b in bs.values():
            await b.leave()
    finally:
        srv.terminate()
    log = [json.loads(x) for x in open(os.path.join(MEET, "out", "rooms", f"{rid}.jsonl"))]
    lines = [e for e in log if e.get("type") == "line"]
    says = [e for e in log if e.get("type") == "ai_say"]
    speak_lines = [e for e in lines if (e.get("decision") or {}).get("speak")]
    kinds = {}
    for e in lines:
        w = (e.get("decision") or {}).get("why")
        kinds[w] = kinds.get(w, 0) + 1
    out = {"meeting": a.meeting, "minutes": a.minutes, "start": a.start, "mode": a.mode, "room": rid, "lines": len(lines),
           "decided_speak": len(speak_lines), "spoke": len(says), "spoke_per_min": round(len(says) / a.minutes, 2),
           "talked_over": over, "line_kinds": kinds,
           "said": [{"after": next((x["name"] + ": " + x["text"] for x in reversed(lines) if x["t"] <= s["t"]), None),
                     "ai": s["text"]} for s in says],
           "skipped": [e for e in log if e.get("type") == "reply_skipped"],
           "dropped": [e.get("dropped") for e in log if e.get("type") == "ai_state" and e.get("dropped")],
           "errors": [e.get("error") for e in log if e.get("type") == "ai_error"][:10], "asks": plan}
    for p in plan:
        lines_q = [x for x in lines if x.get("name") == "Eve" and x["t"] > 0]
        p["heard"] = next((x["text"] for x in lines_q if x["text"][:15].lower() in p["q"].lower() or
                           p["q"][:15].lower() in x["text"].lower()), None)
        t_line = next((y["t"] for y in lines_q if y["text"] == p["heard"]), None)
        said = next((x for x in says if t_line and t_line <= x["t"] <= t_line + 8.0), None)
        p["ai"] = said["text"] if said else None
        p["right"] = bool(said) and any(w in said["text"].lower() for w in p["want"])
    path = os.path.join(MEET, "out", f"ami_meeting_{a.meeting}_{a.start}_{a.mode}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("said", "asks")}, indent=1)[:2500])
    for p in out["asks"]:
        print(f"  ASK at {p['at']}s: {p['q']!r} heard {p.get('heard')!r} -> answered after {p['answered_after']} s, "
              f"right={p['right']}: {p.get('ai')!r}")
    for x in out["said"]:
        print(f"  after {x['after']!r}\n     -> {x['ai']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting", default="ES2009c")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--mode", default="improved")
    ap.add_argument("--ai", default="Higgs")
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--ask", type=int, default=0, help="a fifth person asks this many brief questions in the lulls")
    asyncio.run(run(ap.parse_args()))
