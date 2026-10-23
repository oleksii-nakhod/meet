"""Paired bootstrap over meetings: change in balanced accuracy between two policies on the same decisions.
Usage: python paired.py test vote:higgs_dec2x3 heuristic"""
import asyncio
import random
import sys

import tv_eval


async def preds(pol, rows):
    if pol.startswith("vote:"):
        base, n = pol[5:].rsplit("x", 1)
        ps = [(await tv_eval.run_higgs(base + (f"@{i}" if i else ""), rows, 4))[0] for i in range(int(n))]
        return {k: sum(p[k] for p in ps) * 2 > len(ps) for k in ps[0]}
    if pol in tv_eval.PROMPTS:
        return (await tv_eval.run_higgs(pol, rows, 4))[0]
    return {tv_eval.key(r): getattr(tv_eval, pol)(r) for r in rows}


async def main(split, a, b):
    rows = tv_eval.load(split)
    pa, pb = await preds(a, rows), await preds(b, rows)
    by = {}
    for r in rows:
        by.setdefault(r["meeting"], []).append(r)
    rng, ms, d = random.Random(0), list(by), []
    for _ in range(4000):
        smp = [r for m in (rng.choice(ms) for _ in ms) for r in by[m]]
        if any(r["gold"] == "SPEAK" for r in smp):
            d.append(tv_eval.metrics(smp, pa)["bacc"] - tv_eval.metrics(smp, pb)["bacc"])
    d.sort()
    full = tv_eval.metrics(rows, pa)["bacc"] - tv_eval.metrics(rows, pb)["bacc"]
    print(f"{split}: {a} - {b} = {full:+.3f} [{d[int(.025 * len(d))]:+.3f}, {d[int(.975 * len(d)) - 1]:+.3f}]")


asyncio.run(main(*sys.argv[1:4]))
