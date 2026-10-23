"""Barge-in decisions on real human sounds (AMI headsets, offline, no API calls).

Events, from manual word timings in the 12 test meetings:
  BACKCHANNEL  - X says 1-2 backchannel words (yeah, mm-hmm, right, okay, ...) lasting <= 0.8 s while Y is talking,
                 and Y keeps talking for >= 0.5 s after it.
  INTERRUPTION - X starts a turn of >= 3 words / >= 1 s while Y is talking, and Y stops within 1.5 s.
For each event, Y plays the role of the agent (channel 2) and X the user (channel 1), each headset muted outside
its speaker's words. The engine's rule is replayed frame by frame with the real models: backchannel if
p_bc_det >= 0.5 while the sound lasts (and < 1 s); interruption if speech lasts >= 0.4 s with VAP p_now > 0.35, or
>= 0.8 s whatever the models say.
Usage: python eval/bargein_ami.py [--per_meeting 40]"""
import argparse
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

BC_WORDS = {"yeah", "mm-hmm", "mm", "hmm", "right", "okay", "uh-huh", "yep", "yes", "sure", "wow", "oh", "mm-hm",
            "uh-hmm", "aha", "ah"}


def find_events(m, per):
    ws = {a: words(m, a) for a in "ABCD"}
    ev = []
    for xi, x in enumerate("ABCD"):
        xs = ws[x]
        # group X's words into IPUs
        ipus, cur = [], []
        for w in xs:
            if cur and w["s"] - cur[-1]["e"] > 0.3:
                ipus.append(cur); cur = []
            cur.append(w)
        if cur:
            ipus.append(cur)
        for ipu in ipus:
            s, e = ipu[0]["s"], ipu[-1]["e"]
            for yi, y in enumerate("ABCD"):
                if y == x:
                    continue
                talking = [w for w in ws[y] if w["s"] <= s + 0.1 and w["e"] >= s - 0.3]
                if not talking:
                    continue
                y_after = [w for w in ws[y] if w["s"] > e]
                y_end = max((w["e"] for w in ws[y] if w["s"] < s + 2.0), default=s)
                toks = [w["w"].lower().strip(".,?!") for w in ipu]
                if len(ipu) <= 2 and e - s <= 0.8 and all(t in BC_WORDS for t in toks) and \
                        y_after and y_after[0]["s"] - e < 0.6 and max(w["e"] for w in y_after[:3]) - e >= 0.5:
                    ev.append(("backchannel", xi, yi, s, e, " ".join(toks)))
                elif (len(ipu) >= 3 or e - s >= 1.0) and not all(t in BC_WORDS for t in toks[:2]) \
                        and y_end - s < 1.5 and y_end > s:
                    ev.append(("interrupt", xi, yi, s, e, " ".join(toks[:6])))
                break
    rng = np.random.default_rng(1)
    out = []
    for kind in ("backchannel", "interrupt"):
        k = [e for e in ev if e[0] == kind]
        out += [k[i] for i in sorted(rng.choice(len(k), min(per, len(k)), replace=False))] if k else []
    return out


def run_meeting(args):
    m, per = args
    import mmhm.models  # noqa: F401  (installs the stubs MaAI needs)
    from mmhm.models import _Stream
    evs = find_events(m, per)
    audio = {}
    for ch, a in enumerate("ABCD"):
        x, sr = sf.read(f"{AMI}/amicorpus/{m}/audio/{m}.Headset-{ch}.wav", dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        mask = np.zeros(len(x), bool)
        for w in words(m, a):
            mask[max(0, int((w["s"] - 0.05) * sr)):int((w["e"] + 0.05) * sr)] = True
        audio[ch] = (np.where(mask, x, 0.0).astype(np.float32), sr)
    rows = []
    for kind, xi, yi, s, e, text in evs:
        (xa, sr), (ya, _) = audio[xi], audio[yi]
        t0 = max(0.0, s - 3.0)
        u = resample_poly(xa[int(t0 * sr):int((s + 1.2) * sr)], 16000, sr).astype(np.float32)
        g = resample_poly(ya[int(t0 * sr):int((s + 1.2) * sr)], 16000, sr).astype(np.float32)
        bcd, vap = _Stream("bc_det", 12.5, model_type="normal-ver2", key="p_bc_det"), _Stream("vap", 20, key="p_now")
        n = min(len(u), len(g))
        p_bc, p_now = [], []
        for i in range(0, n - 320, 320):  # 20 ms ticks, like the engine
            p_bc.append(bcd.push(u[i:i + 320], g[i:i + 320])); p_now.append(vap.push(u[i:i + 320], g[i:i + 320]))
        onset = int((s - t0) / 0.02)
        dur = e - s
        decision, bc_max = None, 0.0
        for k in range(onset, len(p_bc)):
            t = (k - onset) * 0.02
            speaking = t <= dur
            bc_max = max(bc_max, p_bc[k])
            if bc_max >= 0.5 and t < 1.0:
                if not speaking:
                    decision = "backchannel"; break
            elif speaking and ((t >= 0.4 and p_now[k] > 0.35) or t >= 0.8):
                decision = "interrupt"; break
            elif not speaking and t - dur > 0.25:
                decision = "ignored"; break
        rows.append({"meeting": m, "kind": kind, "text": text, "dur": round(dur, 2), "decision": decision or "none",
                     "bc_max": round(bc_max, 3)})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_meeting", type=int, default=40)
    a = ap.parse_args()
    with mp.get_context("spawn").Pool(12) as pool:
        rows = [r for rs in pool.imap_unordered(run_meeting, [(m, a.per_meeting) for m in MEETINGS + MEETINGS_2]) for r in rs]
    json.dump(rows, open(os.path.join(HERE, "..", "out", "bargein_ami.json"), "w"), indent=1)
    for kind in ("backchannel", "interrupt"):
        rs = [r for r in rows if r["kind"] == kind]
        c = {d: sum(r["decision"] == d for r in rs) for d in ("backchannel", "ignored", "interrupt", "none")}
        print(f"{kind:12s} n={len(rs):4d}  kept talking (backchannel/ignored): {c['backchannel'] + c['ignored']}"
              f"  stopped: {c['interrupt']}  undecided: {c['none']}")
    bc = [r for r in rows if r["kind"] == "backchannel"]
    print("most common backchannels:", sorted({r["text"] for r in bc}, key=lambda t: -sum(r["text"] == t for r in bc))[:10])
