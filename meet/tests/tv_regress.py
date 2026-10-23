"""Does a judge prompt change hurt Third Voice's group-meeting numbers? Runs agent.decide (3 judge + 3 fact-check
samples, the live agent's rule) on third_voice/testset/points_<set>.jsonl and prints balanced accuracy with a
bootstrap CI, plus a paired comparison when two prompts are given. Cache shared with judge_cases.py.

Usage: python tests/tv_regress.py --set test --prompts v4b v5"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MEET = os.path.dirname(HERE)
sys.path.insert(0, MEET)


def bal(rows, pred):
    sp = [r for r in rows if r["gold"] == "SPEAK"]
    si = [r for r in rows if r["gold"] != "SPEAK"]
    tpr = sum(pred[id(r)] for r in sp) / max(1, len(sp))
    tnr = sum(not pred[id(r)] for r in si) / max(1, len(si))
    return (tpr + tnr) / 2, tpr, tnr


def boot(rows, preds, n=2000):
    rng = random.Random(0)
    meets = sorted({r["meeting"] for r in rows})
    out = []
    for _ in range(n):
        pick = [rng.choice(meets) for _ in meets]  # resample meetings, not lines
        rs = [r for m in pick for r in rows if r["meeting"] == m]
        out.append([bal(rs, p)[0] for p in preds])
    return out


async def main_async(a):
    import agent
    import tv_eval
    cache_path = os.path.join(MEET, "out", "judge_cache.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for x in open(cache_path):
            d = json.loads(x)
            cache[d["k"]] = d["v"]
    sem = asyncio.Semaphore(a.par)

    async def ask(sys_, user, i):
        k = hashlib.sha1(json.dumps([sys_, user, i]).encode()).hexdigest()
        if k not in cache or not cache[k].strip():
            async with sem:
                v = await tv_eval.ask_higgs(sys_, user)
            cache[k] = v
            if v.strip():
                with open(cache_path, "a") as f:
                    f.write(json.dumps({"k": k, "v": v}) + "\n")
        return cache[k]

    rows = [json.loads(x) for x in open(os.path.join(MEET, "..", "third_voice", "testset", f"points_{a.set}.jsonl"))]
    async def one(r, prompt):
        hist = r["context_turns"]
        text, kind = agent.prefilter(r["current_turn"]["text"], r["target_speaker"], hist)
        if kind and prompt != "v4b":  # the live agent's pre-filters (added with v5)
            return False, {"reading": {"prefilter": kind}}
        r2 = {**r, "current_turn": {**r["current_turn"], "text": text}}
        return await agent.decide(r2, lambda _k, s, u, i: ask(s, u, i), samples=3, prompt=prompt)

    preds = []
    for prompt in a.prompts:
        res = await asyncio.gather(*[one(r, prompt) for r in rows])
        p = {id(r): sp for r, (sp, _) in zip(rows, res)}
        preds.append(p)
        b, tpr, tnr = bal(rows, p)
        print(f"{prompt}: balanced {b:.3f} (speak {tpr:.3f}, silent {tnr:.3f}) on {len(rows)} {a.set} decisions", flush=True)
        if a.verbose:
            for r, (sp, d) in zip(rows, res):
                if sp != (r["gold"] == "SPEAK"):
                    print(f"   XX {r['gold']:6s} {r['current_turn']['speaker']}: {r['current_turn']['text'][:90]!r} "
                          f"{json.dumps(d['reading'])}")
    bs = boot(rows, preds)
    for i, prompt in enumerate(a.prompts):
        v = sorted(x[i] for x in bs)
        print(f"  {prompt} 95% CI [{v[50]:.3f}, {v[1949]:.3f}]")
    if len(preds) == 2:
        d = sorted(x[1] - x[0] for x in bs)
        print(f"  {a.prompts[1]} - {a.prompts[0]}: {bal(rows, preds[1])[0] - bal(rows, preds[0])[0]:+.3f} "
              f"[{d[50]:+.3f}, {d[1949]:+.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="test")
    ap.add_argument("--prompts", nargs="+", default=["v4b", "v5"])
    ap.add_argument("--par", type=int, default=10)
    ap.add_argument("-v", "--verbose", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
