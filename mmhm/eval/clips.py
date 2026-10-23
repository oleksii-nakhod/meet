"""Cut listenable A/B examples out of recorded evaluation sessions (user on the left, agent on the right).

  python eval/clips.py turns <results.jsonl> <cond> <pause>   -> out/clips/turn_<cond>_<i>.wav for HOLD items
  python eval/clips.py bargein <results.jsonl> <cond>         -> out/clips/bc_<cond>_<cue>_<voice>.wav"""
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mmhm.higgs import outdir  # noqa: E402

OUT = outdir("clips")


def cut(wav, t0, t1, path):
    x, sr = sf.read(wav, dtype="float32")
    seg = x[max(0, int(t0 * sr)):int(t1 * sr)]
    # gentle stereo: user mostly left, agent mostly right, so both are audible on speakers too
    mix = np.stack([0.85 * seg[:, 0] + 0.35 * seg[:, 1], 0.35 * seg[:, 0] + 0.85 * seg[:, 1]], 1)
    fade = np.minimum(1, np.minimum(np.arange(len(mix)), np.arange(len(mix))[::-1]) / (0.05 * sr))[:, None]
    sf.write(path, mix * fade, sr, subtype="PCM_16")


if __name__ == "__main__":
    kind, res, cond = sys.argv[1:4]
    rows = [json.loads(line) for line in open(res) if json.loads(line)["cond"] == cond]
    n = 0
    if kind == "turns":
        pause = sys.argv[4]
        wav = os.path.join(os.path.dirname(OUT), "turns", f"session_{cond}_{pause}{sys.argv[5] if len(sys.argv) > 5 else ''}.wav")
        for i, r in enumerate(rows):
            if r["kind"] == "hold":
                cut(wav, r["t_a_end"] - 2.5, r["t_end"] + 3.5, f"{OUT}/turn_{cond}_{pause}_{i:02d}.wav"); n += 1
    else:
        wav = os.path.join(os.path.dirname(OUT), "bargein", f"session_{cond}.wav")
        for r in rows:
            if "t_cue" in r:
                cut(wav, r["t_cue"] - 2.5, min(r["t_settled"], r["t_cue"] + 9), f"{OUT}/bc_{cond}_{r['cue']}_{r['voice']}.wav"); n += 1
    print(f"{n} clips -> {OUT}")
