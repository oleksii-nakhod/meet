"""MSI-Bench (Boson AI's multi-speaker benchmark) through the live call: each case is a room, each human a bot playing
the case's rendered line audio (speaker distance, reverb, room ambience, sometimes a bystander's voice) one after the
other with short gaps; the AI is "Assistant". In every case only the final line is the assistant's cue (people talk
to each other before it), so: did Higgs speak before the cue (premature), and did it answer the cue?
Third Voice scored these as text and as Higgs STT transcripts (Third Voice's msi_eval.py); this adds the call's timing.

Usage: python tests/msi_meeting.py [--n 24] [--par 3] [--port 8941] [--gap 0.6]
-> out/msi_meeting.json"""
import argparse
import asyncio
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MEET = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from bots import Bot, create_room, load_wav  # noqa: E402
from scenarios import start_server  # noqa: E402

MSI = os.environ.get("MSI_DIR", os.path.join(MEET, "data", "msi_bench"))
SHORT = {"speaker_authority_constraint": "auth", "sequential_constraint_integration": "seq", "selective_disclosure":
         "disc", "scope_tracking": "scope", "background_speech_retrieval": "retr", "constraint_prioritization": "prior"}
ADMIN = "scen-admin"


def cases(n, seed=int(os.environ.get("MSI_SEED", "0"))):
    out = []
    for c in map(json.loads, open(os.path.join(MSI, "index", "cases.jsonl"))):
        if c["language"] != "en" or any(t["role"] == "assistant" for t in c["turns"]):
            continue
        found = glob.glob(os.path.join(MSI, "data", "en", SHORT[c["pattern"]], "*", c["case_id"]))
        if not found:
            continue
        d = found[0]
        lines = [(t, os.path.join(d, t["audio"])) for t in c["turns"] if t["role"] == "human"]
        if lines and all(os.path.exists(p) for _, p in lines):
            out.append((c, lines))
    rng = random.Random(seed)
    by = {}
    for c in out:
        by.setdefault(c[0]["pattern"], []).append(c)
    for v in by.values():
        rng.shuffle(v)
    picked = []
    while len(picked) < n and any(by.values()):  # round-robin over the patterns
        for v in by.values():
            if v and len(picked) < n:
                picked.append(v.pop())
    return picked


async def run_case(base, c, lines, gap):
    people = list(dict.fromkeys(t["speaker"] for t, _ in lines))
    facts = []
    pm = c["eval"].get("private_memory")
    if isinstance(pm, str) and pm.strip():
        facts = [pm.strip()[:500]]
    elif isinstance(pm, (list, dict)):
        facts = [json.dumps(pm)[:500]]
    rid = create_room(base, ADMIN, title=c["case_id"], ai="Assistant", setting=c["scene_context"][:300], facts=facts)
    bs = {p: Bot(base, rid, p) for p in people}
    for b in bs.values():
        await b.join()
        await asyncio.sleep(0.2)
    ear = bs[people[0]]
    await asyncio.sleep(1.0)
    await ear.json({"type": "set_mode", "mode": "improved"})
    await asyncio.sleep(6.0)
    t_first = ear.now()
    t_cue = t_cue_end = None
    for i, (t, path) in enumerate(lines):
        if i == len(lines) - 1:
            t_cue = ear.now()
        t_cue_end = await bs[t["speaker"]].say(load_wav(path))
        await asyncio.sleep(gap)
    await ear.wait_ai(8.0, since=t_cue_end - 0.5)
    await ear.wait_ai_quiet(1.0, 25)
    await asyncio.sleep(0.5)
    spans = ear.ai_onsets(t_first)
    early = [round(a - t_first, 1) for a, _ in spans if a < t_cue]
    answer = next((a for a, _ in spans if a >= t_cue - 0.2), None)
    says = [m["text"] for t_, m in ear.evs("ai_say", t_cue - 0.2)]
    lines_heard = [(m.get("name"), m.get("text"), (m.get("decision") or {}).get("speak"))
                   for t_, m in ear.evs("line", t_first)]
    for b in bs.values():
        await b.leave()
    return {"case": c["case_id"], "pattern": c["pattern"], "people": len(people), "lines": len(lines),
            "premature": early, "answered": answer is not None,
            "latency": round(answer - t_cue_end, 2) if answer else None, "said": says[:1],
            "cue": lines[-1][0]["text"], "standard": c["eval"].get("standard_answer"), "heard": lines_heard}


async def main_async(a):
    todo = cases(a.n)
    servers = [start_server(a.port + i) for i in range(a.par)]
    q = asyncio.Queue()
    for x in todo:
        q.put_nowait(x)
    res = []

    async def worker(base):
        while not q.empty():
            c, lines = q.get_nowait()
            try:
                r = await run_case(base, c, lines, a.gap)
            except Exception as e:  # noqa: BLE001
                r = {"case": c["case_id"], "error": repr(e)}
            res.append(r)
            print(f"{r['case']}: premature {r.get('premature')} answered {r.get('answered')} ({r.get('latency')} s) "
                  f"-> {r.get('said')}", flush=True)
    try:
        await asyncio.gather(*[worker(base) for _, base in servers])
    finally:
        for p, _ in servers:
            p.terminate()
    ok = [r for r in res if "error" not in r]
    summ = {"cases": len(ok), "premature": sum(bool(r["premature"]) for r in ok),
            "answered": sum(r["answered"] for r in ok),
            "latency_median": sorted(r["latency"] for r in ok if r["latency"] is not None)[len(ok) // 2] if ok else None}
    json.dump({"summary": summ, "results": res}, open(os.path.join(MEET, "out", "msi_meeting.json"), "w"), indent=1)
    print("SUMMARY", json.dumps(summ))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--par", type=int, default=3)
    ap.add_argument("--port", type=int, default=8941)
    ap.add_argument("--gap", type=float, default=0.6)
    asyncio.run(main_async(ap.parse_args()))
