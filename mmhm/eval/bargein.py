"""Live barge-in evaluation: does a listener's "mm-hm" kill Higgs's turn? Does a real interruption still work?

The simulated user asks for a story; 2.5 s into playback it makes a sound:
- BACKCHANNELS (mm-hm, yeah, uh-huh, right, okay, wow, a laugh, "mhm mhm") in 2 voices: the right behaviour
  is to keep talking, and not to answer the backchannel afterwards.
- INTERRUPTS ("wait, sorry, where was that?" ...): the right behaviour is to stop quickly and answer.

Plain Higgs (server_vad / semantic_vad, client stops playback on `speech_started` as the docs say) vs Mm-hm.

Usage: python eval/bargein.py [--conds server_vad mmhm] [--delay 2.5]"""
import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval import stimuli  # noqa: E402
from eval.stimuli import get  # noqa: E402
from eval.turns import Clock  # noqa: E402
from mmhm.engine import Engine, Policy  # noqa: E402
from mmhm.higgs import outdir, silence  # noqa: E402

OUT = outdir("bargein")
PERSONA = ("You are a chatty friend at a hackathon. When asked for a story, tell it in five or six sentences. "
           "Otherwise reply in one short sentence.")
ASKS = [
    "Tell me about the best trip you ever took.",
    "What's the funniest thing that happened to you at work?",
    "Tell me how you got into programming.",
    "What was your worst ever travel day?",
    "Tell me about your favourite teacher.",
    "How did you end up living in Toronto?",
]


async def run(cond, delay, test=False, ami=False):
    signals = None
    if cond == "mmhm":
        from mmhm.models import Signals
        signals = Signals(vap=True, bcdet=True, bc=False)
    if ami:  # real cues cut from AMI meetings (eval/ami_cues.py)
        d = os.path.join(os.path.dirname(OUT), "ami_cues")
        meta = json.load(open(os.path.join(d, "cues.json")))
        BACKCHANNELS = {c["file"]: os.path.join(d, c["file"]) for c in meta if c["kind"] == "backchannel"}
        INTERRUPTS = {c["file"]: os.path.join(d, c["file"]) for c in meta if c["kind"] == "interrupt"}
        trials = [("backchannel", k, "ami") for k in BACKCHANNELS] + [("interrupt", k, "ami") for k in INTERRUPTS]
    elif test:
        BACKCHANNELS, INTERRUPTS = stimuli.BACKCHANNELS_TEST, stimuli.INTERRUPTS_TEST
        trials = [("backchannel", k, v) for k in BACKCHANNELS for v in ("oliver", "chloe")]
        trials += [("interrupt", k, v) for k in INTERRUPTS for v in ("chloe",)]
    else:
        BACKCHANNELS, INTERRUPTS = stimuli.BACKCHANNELS, stimuli.INTERRUPTS
        trials = [("backchannel", k, v) for k in BACKCHANNELS for v in ("jake", "nora")]
        trials += [("interrupt", k, "marcus") for k in INTERRUPTS]
    trials = [trials[i] for i in np.random.default_rng(1).permutation(len(trials))]
    log, rows = [], []
    async with Engine(mode=cond, policy=Policy(), instructions=PERSONA, signals=signals, log=log) as e:
        clk = Clock(e)
        await clk.feed(silence(0.5))
        for i, (kind, cue, voice) in enumerate(trials):
            k0 = len(log)
            await clk.feed(get(ASKS[i % len(ASKS)], "eleanor"))
            await clk.idle_until(lambda: any(r["kind"] == "play" for r in log[k0:]), 8)
            plays = [r for r in log[k0:] if r["kind"] == "play"]
            if not plays:
                rows.append({"cond": cond, "kind": kind, "cue": cue, "error": "no reply"}); continue
            rid = plays[0]["rid"]
            await clk.feed(silence(delay))
            t_cue = e.t
            if voice == "ami":
                from mmhm.higgs import load_wav
                await clk.feed(load_wav(BACKCHANNELS.get(cue) or INTERRUPTS[cue]))
            else:
                await clk.feed(get(BACKCHANNELS.get(cue) or INTERRUPTS[cue], voice))
            # settle: no agent audio playing and nothing new for 1.5 s (or 40 s max)
            last = [e.t]

            def settled():
                if e.state != "listening" or e.current is not None or e.session.events and \
                        e.session.events[-1][1].get("type", "").startswith("response.output"):
                    last[0] = e.t
                return e.t - last[0] > 1.5

            await clk.idle_until(settled, 40)
            ev = log[k0:]
            stop = next((r for r in ev if r["kind"] == "stop" and r["t"] >= t_cue), None)
            duck = next((r for r in ev if r["kind"] == "duck" and r["t"] >= t_cue), None)
            end = next((r for r in ev if r["kind"] == "play_end" and r.get("rid") == rid), None)
            later = [r for r in ev if r["kind"] == "response_done" and r.get("rid") != rid and r["t"] >= t_cue]
            row = {"cond": cond, "kind": kind, "cue": cue, "voice": voice, "t_play": plays[0]["t"], "t_cue": round(t_cue, 3),
                   "t_settled": round(e.t, 3),
                   "stopped": stop is not None,
                   "stop_latency": round(stop["t"] - t_cue, 3) if stop else None,
                   "duck_latency": round(duck["t"] - t_cue, 3) if duck else None,
                   "story_finished": end is not None,
                   "played_ms": (stop or end or {}).get("played_ms"),
                   "replied_after": [r.get("text") for r in later if r.get("status") == "completed"],
                   "heard": [r["kind"] for r in ev if r["kind"] in ("backchannel_heard", "noise_ignored")]}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            await clk.feed(silence(0.5))
        tag = cond + ("_ami" if ami else "_test" if test else "")
        clk.save(f"{OUT}/session_{tag}.wav")
        with open(f"{OUT}/log_{tag}.jsonl", "w") as f:
            for r in log:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{cond}] late ticks: {clk.late}/{len(clk.mic)}", flush=True)
    if signals is not None:
        signals.close()
    return rows


def summarize(rows):
    out = {}
    for cond in dict.fromkeys(r["cond"] for r in rows):
        bc = [r for r in rows if r["cond"] == cond and r["kind"] == "backchannel" and "error" not in r]
        it = [r for r in rows if r["cond"] == cond and r["kind"] == "interrupt" and "error" not in r]
        lat = [r["stop_latency"] for r in it if r["stop_latency"] is not None]
        duck = [r["duck_latency"] for r in it if r["duck_latency"] is not None]
        out[cond] = {
            "backchannel_killed_story": f"{sum(r['stopped'] for r in bc)}/{len(bc)}",
            "backchannel_answered": f"{sum(bool(r['replied_after']) for r in bc)}/{len(bc)}",
            "interrupt_stopped": f"{sum(r['stopped'] for r in it)}/{len(it)}",
            "interrupt_stop_ms_median": int(np.median(lat) * 1000) if lat else None,
            "interrupt_duck_ms_median": int(np.median(duck) * 1000) if duck else None,
        }
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="+", default=["server_vad", "mmhm"])
    ap.add_argument("--delay", type=float, default=2.5)
    ap.add_argument("--test", action="store_true", help="use the held-out cues and voices")
    ap.add_argument("--ami", action="store_true", help="real cues from AMI meetings (eval/ami_cues.py)")
    args = ap.parse_args()
    rows = []
    for c in args.conds:
        rows += await run(c, args.delay, args.test, args.ami)
    tag = f"{'ami_' if args.ami else 'test_' if args.test else ''}{int(time.time())}"
    with open(f"{OUT}/results_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = summarize(rows)
    print(json.dumps(s, indent=1))
    json.dump(s, open(f"{OUT}/summary_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
