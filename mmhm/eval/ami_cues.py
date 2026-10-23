"""Real barge-in cues from AMI headsets: listener backchannels and real interruptions (eval/bargein_ami.py's
definitions), cut from the test meetings, muted outside the speaker's words, level-normalized.
-> out/ami_cues/{bc,int}_NN.wav + cues.json. Usage: python eval/ami_cues.py [--bc 40 --int 20]"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.ami_stimuli import AMI, MEETINGS, MEETINGS_2, words  # noqa: E402
from eval.bargein_ami import find_events  # noqa: E402
from mmhm.higgs import SR, outdir  # noqa: E402

OUT = outdir("ami_cues")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", type=int, default=40)
    ap.add_argument("--int", type=int, default=20)
    a = ap.parse_args()
    evs = [e + (m,) for m in MEETINGS + MEETINGS_2 for e in find_events(m, 10)]
    rng = np.random.default_rng(3)
    pick = {k: [e for e in evs if e[0] == k] for k in ("backchannel", "interrupt")}
    pick = {k: [v[i] for i in sorted(rng.choice(len(v), min(n, len(v)), replace=False))]
            for k, v, n in (("backchannel", pick["backchannel"], a.bc), ("interrupt", pick["interrupt"], a.int))}
    meta = []
    for kind, es in pick.items():
        for i, (_, xi, yi, s, e, text, m) in enumerate(es):
            x, sr = sf.read(f"{AMI}/amicorpus/{m}/audio/{m}.Headset-{xi}.wav", dtype="float32")
            x = x[:, 0] if x.ndim > 1 else x
            end = e if kind == "backchannel" else min(e, s + 2.5)
            seg = x[int((s - 0.05) * sr):int((end + 0.1) * sr)].copy()
            mask = np.zeros(len(seg), bool)
            for w in words(m, "ABCD"[xi]):
                if w["e"] >= s - 0.05 and w["s"] <= end + 0.1:
                    mask[max(0, int((w["s"] - s + 0.05 - 0.03) * sr)):int((w["e"] - s + 0.05 + 0.03) * sr)] = True
            seg = np.where(mask, seg, 0.0)
            seg = resample_poly(seg, SR, sr)
            seg = seg / (np.sqrt(np.mean(seg[np.abs(seg) > 1e-4] ** 2)) + 1e-9) * 0.08
            name = f"{'bc' if kind == 'backchannel' else 'int'}_{i:02d}.wav"
            sf.write(os.path.join(OUT, name), (np.clip(seg, -1, 1) * 32767).astype(np.int16), SR, subtype="PCM_16")
            meta.append({"kind": kind, "file": name, "text": text, "meeting": m})
    json.dump(meta, open(os.path.join(OUT, "cues.json"), "w"), indent=1)
    print(len(meta), "cues;", ", ".join(sorted({x["text"] for x in meta if x["kind"] == "backchannel"})[:12]))
