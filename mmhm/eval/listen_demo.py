"""Listening noises: a simulated user tells a ~30 s story in sentences with natural pauses; the engine (with
backchannels on) says "mm-hm" where the MaAI backchannel model predicts a listener would, then answers.
Writes out/listen/listen_demo.wav (left: user, right: agent) and prints when each backchannel fired relative
to the user's speech (it should land in pauses, never on top of words).

Usage: python eval/listen_demo.py [--voice nora]"""
import argparse
import asyncio
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.server import bc_clips  # noqa: E402
from eval.stimuli import get  # noqa: E402
from eval.turns import Clock  # noqa: E402
from mmhm.engine import Engine, Policy  # noqa: E402
from mmhm.higgs import outdir, silence  # noqa: E402
from mmhm.models import Signals  # noqa: E402

OUT = outdir("listen")
STORY = [
    "So last weekend I finally went to that cabin up north with my brother.",
    "We drove for like four hours, and the last part was this awful dirt road.",
    "And of course, halfway down the road, the car gets a flat tire.",
    "Neither of us had changed a tire in maybe ten years.",
    "So we're out there with a phone flashlight and a manual from two thousand nine.",
    "Anyway, after about an hour we got it done, and we were so proud of ourselves.",
    "And then we get to the cabin, and we realize we forgot the keys at home.",
]
PERSONA = "You are a friendly person chatting. When someone finishes a story, react in one or two short sentences."


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="nora")
    a = ap.parse_args()
    signals = Signals(vap=True, bcdet=True, bc=True)
    log = []
    pol = Policy(backchannels=True)
    spans = []  # (start, end) of user speech in engine time
    async with Engine(mode="mmhm", policy=pol, instructions=PERSONA, signals=signals, bc_clips=bc_clips(),
                      log=log) as e:
        clk = Clock(e)
        await clk.feed(silence(0.5))
        rng = np.random.default_rng(3)
        for line in STORY:
            pcm = get(line, a.voice)
            t0 = e.t
            await clk.feed(pcm)
            spans.append((t0, e.t))
            await clk.feed(silence(float(rng.uniform(0.35, 0.6))))  # breath between sentences: a story, not a turn end
        await clk.idle_until(lambda: any(r["kind"] == "play_end" for r in log), 20)
        await clk.feed(silence(0.5))
        clk.save(f"{OUT}/listen_demo.wav")
    signals.close()
    bcs = [r for r in log if r["kind"] == "backchannel"]
    overlap = sum(any(s <= r["t"] <= en for s, en in spans) for r in bcs)
    print(f"backchannels: {len(bcs)} over {spans[-1][1] - spans[0][0]:.1f} s of story; fired during user speech: {overlap}")
    for r in bcs:
        print(f"  t={r['t']:.2f}  p_bc={r['p_bc']}")
    print("speculative commits during the story:", sum(r["kind"] == "commit" for r in log) - 1,
          "aborted:", sum(r["kind"] == "abort" for r in log))
    print("reply:", [r.get("text") for r in log if r["kind"] == "response_done"])
    json.dump({"spans": spans, "log": log}, open(f"{OUT}/listen_demo.json", "w"))


if __name__ == "__main__":
    asyncio.run(main())
