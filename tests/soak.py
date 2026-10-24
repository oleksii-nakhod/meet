"""A long call: one 3-person room, N cycles of [two lines between the humans, a question to Higgs], with the server's
memory (its own RSS + its children's) sampled every minute. Checks that Higgs keeps answering late in a meeting and
that nothing grows without bound.

Usage: python tests/soak.py [--cycles 25] [--port 9191] -> out/soak.json"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MEET = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import scenarios  # noqa: E402

CHAT = [("Priya", "Did anyone see the new mockups?"), ("Sam", "Yeah, they look good, the colors are better."),
        ("Priya", "Sam, can you send me the link?"), ("Sam", "Sure, I'll post it in the chat."),
        ("Priya", "I think we should keep the intro short."), ("Sam", "Agreed, twenty seconds at most.")]
ASK = ["Higgs, when do submissions close?", "Higgs, how long can the video be?", "Higgs, what are the judging criteria?",
       "Higgs, how much is innovation worth?", "Higgs, remind me what our project is?"]


def rss_mb(pid):
    def one(p):
        try:
            return int(open(f"/proc/{p}/status").read().split("VmRSS:")[1].split()[0]) / 1024
        except (OSError, IndexError):
            return 0.0
    kids = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    return round(one(pid), 1), round(sum(one(k) for k in kids), 1), len(kids)


async def main_async(a):
    steps = []
    for i in range(a.cycles):
        for who, text in CHAT[(2 * i) % len(CHAT):(2 * i) % len(CHAT) + 2]:
            steps.append(("say", who, text, "silent"))
        steps.append(("say", "Alex", ASK[i % len(ASK)], "reply"))
    sc = dict(name="soak", people=["Alex", "Priya", "Sam"], steps=steps, avatar=a.avatar)
    os.makedirs(os.path.join(MEET, "out", "scen", "soak"), exist_ok=True)
    p, base = scenarios.start_server(a.port)
    mem = []

    async def sample():
        t0 = time.time()
        while True:
            mem.append((round(time.time() - t0), *rss_mb(p.pid)))
            await asyncio.sleep(60)
    task = asyncio.ensure_future(sample())
    try:
        res = await scenarios.run_scenario(base, sc, "improved", os.path.join(MEET, "out", "scen", "soak"))
    finally:
        task.cancel()
        mem.append(("end", *rss_mb(p.pid)))
        p.terminate()
    asks = [r for r in res["rows"] if r.get("expect") == "reply"]
    out = {"cycles": a.cycles, "secs": res["secs"], "ok": res["ok"], "scored": res["scored"],
           "answered": sum(r["got"] == "reply" for r in asks), "asked": len(asks),
           "latency_by_cycle": [r.get("latency") for r in asks], "memory_mb (t, server, children, n)": mem}
    json.dump(out, open(os.path.join(MEET, "out", "soak.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=25)
    ap.add_argument("--port", type=int, default=9191)
    ap.add_argument("--avatar", action="store_true", help="the face on for the whole call (live with LiveKit set)")
    asyncio.run(main_async(ap.parse_args()))
