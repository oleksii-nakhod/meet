"""Which MaAI turn-taking model separates real thinking pauses from real turn ends best? Offline, on the AMI clips.
For each clip, p(user holds the floor) at fixed times after the pause (holds) or after the end (completes, padded
with silence). Reports AUROC per silence duration. No API calls."""
import json
import os
import sys

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mmhm.models  # noqa: E402,F401  (stubs + quiet bc model)
from maai import Maai, MaaiInput  # noqa: E402
from mmhm.higgs import load_wav, outdir  # noqa: E402

T = [0.1, 0.2, 0.4, 0.6, 0.8]
VARIANTS = {"cpc 20Hz (current)": dict(frame_rate=20, model_type="normal"),
            "cpc 10Hz": dict(frame_rate=10, model_type="normal"),
            "mimi 12.5Hz": dict(frame_rate=12.5, model_type="normal-ver2"),
            "cpc 20Hz mc": dict(frame_rate=20, model_type="normal", mc=True),
            "kyoto 10Hz": dict(frame_rate=10, model_type="normal", lang="en_kyoto")}


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def run(variant, clips):
    kw = dict(VARIANTS[variant]); lang = kw.pop("lang", "en")
    out = {"hold": [], "complete": []}
    for kind, x16, t_pause in clips:
        m = Maai(mode="vap", lang=lang, audio_ch1=MaaiInput.Zero(), audio_ch2=MaaiInput.Zero(), device="cpu", **kw)
        hop = int(16000 / kw["frame_rate"])
        z = np.zeros(hop, np.float32)
        ps = []
        for i in range(0, len(x16) - hop, hop):
            m.process(x16[i:i + hop], z)
            while not m.result_dict_queue.empty():
                r = m.result_dict_queue.get()
            ps.append(float(np.ravel(r["p_now"])[0]))
        ps = np.array(ps)
        out[kind].append([ps[min(len(ps) - 1, int((t_pause + t) * kw["frame_rate"]))] for t in T])
    return {k: np.array(v) for k, v in out.items()}


def load(set_dir):
    d = os.path.join(outdir(set_dir))
    clips = []
    for it in json.load(open(os.path.join(d, "items.json"))):
        x = load_wav(os.path.join(d, it["file"])).astype(np.float32) / 32768
        if it["kind"] == "complete":
            x = np.concatenate([x, np.zeros(int(1.0 * 24000), np.float32)])
        x16 = resample_poly(x, 2, 3).astype(np.float32)
        clips.append((it["kind"], x16, it["pause_at"]))
    return clips


if __name__ == "__main__":
    clips = load("ami_stim") + load("ami_stim2")
    print(f"{len(clips)} clips; AUROC (hold > complete) at silence " + " ".join(f"{t}s" for t in T))
    for v in VARIANTS:
        try:
            r = run(v, clips)
        except Exception as e:
            print(f"{v:22s} failed: {str(e)[:80]}"); continue
        a = [auroc(r["hold"][:, j], r["complete"][:, j]) for j in range(len(T))]
        med = [f"{np.median(r['hold'][:, j]):.2f}/{np.median(r['complete'][:, j]):.2f}" for j in range(len(T))]
        print(f"{v:22s} " + "  ".join(f"{x:.2f}" for x in a) + "   medians hold/complete: " + " ".join(med), flush=True)
