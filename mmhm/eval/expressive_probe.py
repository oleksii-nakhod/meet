"""Expressive path check: the simulated user says things that invite laughter, sympathy or surprise; Higgs Realtime
(text) + TTS 3 answers with tags. Reports tag use, response gap, and saves the session (left user, right agent).
Usage: python eval/expressive_probe.py [--native]   (--native: same prompts through native Realtime audio)"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.personas import HUMANIZED  # noqa: E402
from eval.stimuli import get  # noqa: E402
from eval.turns import Clock  # noqa: E402
from mmhm.engine import Engine, Policy  # noqa: E402
from mmhm.higgs import outdir, silence  # noqa: E402
from mmhm.models import Signals  # noqa: E402

OUT = outdir("expressive")
PROMPTS = [
    "So my teammate just spilled an entire coffee on the router, and now nobody has Wi-Fi.",
    "I've been debugging the same bug for six hours and it was a missing comma.",
    "My dog died last week, so I'm a bit out of it, sorry.",
    "Guess what, we just won the first prize at our last hackathon!",
    "The pizza here is honestly kind of terrible.",
    "What are you working on this weekend?",
]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", action="store_true")
    a = ap.parse_args()
    signals = Signals(vap=True, bcdet=True, bc=False)
    log = []
    async with Engine(mode="mmhm", policy=Policy(), instructions=HUMANIZED, signals=signals, log=log,
                      expressive=not a.native) as e:
        clk = Clock(e)
        await clk.feed(silence(0.5))
        for i, text in enumerate(PROMPTS):
            k0 = len(log)
            await clk.feed(get(text, ["jake", "nora"][i % 2]))
            t_end = e.t
            await clk.idle_until(lambda: any(r["kind"] in ("play_end", "stop") and r["t"] > t_end for r in log[k0:]), 25)
            await clk.feed(silence(0.6))
            plays = [r["t"] for r in log[k0:] if r["kind"] == "play" and r["t"] >= t_end]
            done = [r for r in log[k0:] if r["kind"] == "response_done" and r.get("status") == "completed"]
            print(json.dumps({"prompt": text[:40], "gap": round(plays[0] - t_end, 2) if plays else None,
                              "raw": done[-1].get("raw") if done else None}, ensure_ascii=False), flush=True)
        tag = "native" if a.native else "expressive"
        clk.save(f"{OUT}/session_{tag}.wav")
        json.dump(log, open(f"{OUT}/log_{tag}.json", "w"))
    signals.close()


if __name__ == "__main__":
    asyncio.run(main())
