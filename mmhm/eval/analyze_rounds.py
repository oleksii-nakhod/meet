"""Analyze Turing Clock rounds (out/game/rounds.jsonl) for the video, honestly.

- Time until players catch the AI, per version. Rounds where nobody pressed the button are censored (we only
  know they lasted at least that long), so the median comes from a Kaplan-Meier curve, not from caught rounds only.
- Difference between versions: permutation test on the restricted mean time-to-catch (area under the KM curve up
  to the shortest version's longest round).
- How often each version (and real people, in party mode) was judged human, with Wilson 95% intervals.
Usage: python eval/analyze_rounds.py [rounds.jsonl]"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"plain": "plain Higgs", "mmhm": "Higgs + Mm-hm", "human": "real person"}


def km(times, events):
    """Kaplan-Meier survival curve: list of (t, S(t)) at event times."""
    order = np.argsort(times)
    t, e = np.asarray(times)[order], np.asarray(events)[order]
    s, at_risk, curve = 1.0, len(t), [(0.0, 1.0)]
    for ti in np.unique(t):
        d = int(((t == ti) & (e == 1)).sum())
        if d:
            s *= 1 - d / at_risk
            curve.append((float(ti), s))
        at_risk -= int((t == ti).sum())
    return curve


def km_median(curve):
    return next((t for t, s in curve if s <= 0.5), None)


def rmst(curve, tau):
    area, (pt, ps) = 0.0, curve[0]
    for t, s in curve[1:]:
        if t > tau:
            break
        area += ps * (t - pt); pt, ps = t, s
    return area + ps * (tau - pt)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0, c - h), min(1, c + h)


def main(path):
    rows = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    rows = [r for r in rows if r.get("seconds") and not r.get("forced")]
    print(f"{len(rows)} rounds\n")
    curves = {}
    for m in ("plain", "mmhm"):
        rs = [r for r in rows if r.get("mode") == m]
        if not rs:
            continue
        t = [r["seconds"] for r in rs]; e = [1 if r.get("pressed") else 0 for r in rs]
        curves[m] = (t, e, km(t, e))
        med = km_median(curves[m][2])
        print(f"{NAMES[m]:15s} n={len(rs):3d}  caught {sum(e)}/{len(rs)}  KM median time-to-catch: "
              f"{f'{med:.0f} s' if med else 'not reached (most never caught)'}")
    if len(curves) == 2:
        tau = min(max(curves[m][0]) for m in curves)
        obs = rmst(curves["mmhm"][2], tau) - rmst(curves["plain"][2], tau)
        pooled = [(t, e) for m in curves for t, e in zip(curves[m][0], curves[m][1])]
        n1 = len(curves["plain"][0])
        rng = np.random.default_rng(0)
        more = 0
        for _ in range(5000):
            idx = rng.permutation(len(pooled))
            a = [pooled[i] for i in idx[:n1]]; b = [pooled[i] for i in idx[n1:]]
            d = rmst(km(*zip(*b)), tau) - rmst(km(*zip(*a)), tau)
            more += abs(d) >= abs(obs)
        print(f"\nrestricted mean time-to-catch (first {tau:.0f} s): Mm-hm − plain = {obs:+.1f} s, "
              f"permutation p = {more / 5000:.3f}")
    print()
    for m in ("plain", "mmhm", "human"):
        g = [r for r in rows if r.get("mode") == m and r.get("guess") in ("human", "ai")]
        w = wilson(sum(r["guess"] == "human" for r in g), len(g))
        if w:
            print(f"judged human, {NAMES[m]:15s}: {w[0]:.0%} of {len(g)} (95% CI {w[1]:.0%}–{w[2]:.0%})")
    tells = {}
    for r in rows:
        if r.get("tell"):
            tells.setdefault(r["mode"], {}).setdefault(r["tell"], 0)
            tells[r["mode"]][r["tell"]] += 1
    for m, d in tells.items():
        print(f"tells, {NAMES.get(m, m)}: " + ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "out", "game", "rounds.jsonl"))
