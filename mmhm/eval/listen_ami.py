"""Does Mm-hm say "mm-hm" where real listeners do? Offline, on AMI (12 test meetings, no API calls).

Stretches where speaker Y holds the floor >= 8 s (others only backchannel). The engine's listening-noise trigger is
replayed on Y's headset audio with MaAI's backchannel model: fire when p_bc > 0.45 during a 80-350 ms pause, Y has
been talking > 1.5 s, and the last trigger was > 3 s ago. Ground truth: onsets of the real listeners' backchannels.
A hit is a trigger within 0.6 s of a real backchannel onset. Baseline: the same rule without the model (fire at
every 80-350 ms pause, same spacing), i.e. "just nod at pauses".
Usage: python eval/listen_ami.py"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from eval.ami_stimuli import AMI, MEETINGS, MEETINGS_2, words  # noqa: E402
from eval.bargein_ami import BC_WORDS  # noqa: E402


def stretches(m):
    ws = {a: words(m, a) for a in "ABCD"}
    is_bc = lambda w: w["w"].lower().strip(".,?!") in BC_WORDS  # noqa: E731
    out = []
    for yi, y in enumerate("ABCD"):
        ys = ws[y]
        if not ys:
            continue
        others = sorted((w["s"], w["e"], is_bc(w), o) for o in "ABCD" if o != y for w in ws[o])
        # walk Y's words; a stretch ends when someone else says a non-backchannel word or Y pauses > 1.5 s
        start = prev_e = ys[0]["s"]
        for w in ys[1:] + [None]:
            gap = (w["s"] - prev_e) if w else 99
            intr = [o for o in others if prev_e - 0.2 < o[0] < (w["s"] if w else prev_e + 1) and not o[2]]
            if gap > 1.5 or intr or w is None:
                if prev_e - start >= 8.0:
                    bcs = [o[0] for o in others if start < o[0] < prev_e and o[2]]
                    out.append((yi, y, start, prev_e, bcs))
                if w:
                    start = w["s"]
            if w:
                prev_e = w["e"]
    return out


# name: (p_bc threshold or None for "no model", fire only in a 80-350 ms pause?)
RULES = {"pause": (None, True), "model": (0.45, True), "model_p35": (0.35, True), "model_p25": (0.25, True),
         "any_p45": (0.45, False), "any_p60": (0.60, False), "any_p75": (0.75, False)}


def run(m):
    import mmhm.models  # noqa: F401  (installs the stubs MaAI needs)
    from mmhm.models import _Stream
    res = []
    for yi, y, s, e, bcs in stretches(m):
        x, sr = sf.read(f"{AMI}/amicorpus/{m}/audio/{m}.Headset-{yi}.wav", dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        yw = [w for w in words(m, y) if s - 0.1 <= w["s"] <= e]
        seg = np.zeros(int((e - s + 0.5) * sr), np.float32)
        for w in yw:
            a, b = int((w["s"] - s) * sr), int((w["e"] - s) * sr)
            seg[max(0, a):b] = x[int(w["s"] * sr):int(w["s"] * sr) + (b - max(0, a))][:len(seg[max(0, a):b])]
        u = resample_poly(seg, 16000, sr).astype(np.float32)
        bc = _Stream("bc", 20, key="p_bc")
        z = np.zeros(320, np.float32)
        p = [bc.push(u[i:i + 320], z) for i in range(0, len(u) - 320, 320)]
        ends = np.array([w["e"] - s for w in yw]); starts = np.array([w["s"] - s for w in yw])
        trig = {k: [] for k in RULES}
        last = {k: -99.0 for k in RULES}
        for k_, pk in enumerate(p):
            t = k_ * 0.02
            if t < 1.5:
                continue
            done = ends[ends <= t]
            speaking = ((starts <= t) & (ends > t)).any()
            sil = (t - done.max()) if len(done) and not speaking else 0.0
            in_pause = 0.08 <= sil <= 0.35
            for name, (th, need_pause) in RULES.items():
                if t - last[name] <= 3.0 or (need_pause and not in_pause):
                    continue
                if th is None or pk > th:
                    trig[name].append(t); last[name] = t
        res.append({"meeting": m, "speaker": y, "dur": e - s, "real": [b - s for b in bcs], **trig})
    return res


def score(rows, key, tol=0.6):
    tp = sum(any(abs(t - r) <= tol for r in x["real"]) for x in rows for t in x[key])
    n_trig = sum(len(x[key]) for x in rows)
    found = sum(any(abs(t - r) <= tol for t in x[key]) for x in rows for r in x["real"])
    n_real = sum(len(x["real"]) for x in rows)
    mins = sum(x["dur"] for x in rows) / 60
    return f"{n_trig / mins:4.1f}/min, precision {tp / max(1, n_trig):.2f}, recall {found / max(1, n_real):.2f}"


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    if split == "train":
        from eval.turnend_features import meetings
        ms = meetings("train")
    else:
        ms = MEETINGS + MEETINGS_2
    with mp.get_context("spawn").Pool(24) as pool:
        rows = [r for rs in pool.imap_unordered(run, ms) for r in rs]
    json.dump(rows, open(os.path.join(HERE, "..", "out", f"listen_ami_{split}.json"), "w"))
    mins = sum(x["dur"] for x in rows) / 60
    print(f"{len(rows)} stretches, {mins:.1f} min of monologue, real backchannels "
          f"{sum(len(x['real']) for x in rows) / mins:.1f}/min")
    chance = sum(len(x["real"]) for x in rows) / sum(x["dur"] for x in rows) * 1.2
    print(f"chance precision (random timing): {chance:.2f}")
    for k in RULES:
        print(f"{k:10s}", score(rows, k))
