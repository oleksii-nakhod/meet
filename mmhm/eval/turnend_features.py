"""Features for a learned turn-end classifier, from real AMI conversation.

Events (per speaker, from manual word timings, IPUs merged across < 200 ms gaps):
  HOLD  - the speaker pauses >= 0.5 s, nobody else starts talking, and the same speaker continues.
  SHIFT - the speaker stops for >= 1.5 s and someone else starts within 1.5 s.
Decision time: 200 ms into the silence (when Mm-hm first considers answering).
Features at that time, using only what the live engine can know:
  VAP p_now / p_future at +100 and +200 ms (user channel = this speaker's headset, other people's bleed muted),
  IPU duration, words, speech rate, last-word duration, energy drop over the last 150 ms, pitch slope over the
  last 300 ms, and the last word (function word / filler / closing word).
Meetings: `--split train` uses series not used anywhere in the evaluation; `--split test` uses the 12 meetings
behind AMI stimulus sets 1 and 2.

Usage: python eval/turnend_features.py --split train|test [--procs 32] -> out/turnend/<split>.npz"""
import argparse
import glob
import multiprocessing as mp
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from eval.ami_stimuli import AMI, MEETINGS, MEETINGS_2, ipus, words  # noqa: E402
from mmhm.engine import INCOMPLETE  # noqa: E402

OUT = os.path.join(HERE, "..", "out", "turnend")
FILLERS = {"um", "uh", "er", "erm", "hmm", "mm", "eh"}
CLOSERS = {"yeah", "okay", "right", "yes", "no", "so", "alright", "sure", "good", "great", "fine", "exactly"}
TEST_SERIES = {m[:6] for m in MEETINGS + MEETINGS_2} | {"ES2002", "IS1001", "TS3003"}
FEATS = ["p_now_1", "p_now_2", "p_fut_1", "p_fut_2", "ipu_dur", "n_words", "rate", "last_dur", "drop_db",
         "f0_slope", "voiced_end", "w_incomplete", "w_filler", "w_closer"]


def meetings(split):
    if split == "test":
        return MEETINGS + MEETINGS_2
    out = []
    for d in sorted(glob.glob(f"{AMI}/amicorpus/*")):
        m = os.path.basename(d)
        if m[:6] in TEST_SERIES:
            continue
        if all(os.path.exists(f"{d}/audio/{m}.Headset-{i}.wav") for i in range(4)) and \
                all(os.path.exists(f"{AMI}/ami_public_manual_1.5/words/{m}.{a}.words.xml") for a in "ABCD"):
            out.append(m)
    return out


def f0_track(x, sr=16000, fmin=75, fmax=400):
    """Crude autocorrelation pitch per 40 ms frame (10 ms hop); 0 where unvoiced."""
    n, hop = int(0.04 * sr), int(0.01 * sr)
    out = []
    for i in range(0, len(x) - n, hop):
        f = x[i:i + n] - x[i:i + n].mean()
        if np.sqrt(np.mean(f ** 2)) < 1e-3:
            out.append(0.0); continue
        ac = np.correlate(f, f, "full")[n - 1:]
        lo, hi = int(sr / fmax), int(sr / fmin)
        k = lo + int(np.argmax(ac[lo:hi]))
        out.append(sr / k if ac[k] > 0.45 * ac[0] else 0.0)
    return np.array(out)


def events_for(m):
    import mmhm.models  # noqa: F401  (installs the stubs MaAI needs)
    from maai import Maai, MaaiInput
    rows = []
    for ch, agent in enumerate("ABCD"):
        mine = words(m, agent)
        others = np.array(sorted(w["s"] for o in "ABCD" if o != agent for w in words(m, o)))
        segs = ipus(mine)
        x, sr = sf.read(f"{AMI}/amicorpus/{m}/audio/{m}.Headset-{ch}.wav", dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        mask = np.zeros(len(x), bool)
        for w in mine:
            mask[max(0, int((w["s"] - 0.05) * sr)):int((w["e"] + 0.05) * sr)] = True
        x = np.where(mask, x, x * 0.003)
        evs = []
        for k in range(len(segs) - 1):
            cur, nxt = segs[k], segs[k + 1]
            gap = nxt["s"] - cur["e"]
            if gap >= 0.5 and not ((others > cur["e"] - 0.2) & (others < nxt["s"])).any():
                evs.append((cur, 0))
            elif gap >= 1.5 and ((others > cur["e"] - 0.2) & (others < cur["e"] + 1.5)).any():
                evs.append((cur, 1))
        rng = np.random.default_rng(hash(m + agent) % 2**32)
        if len(evs) > 80:
            evs = [evs[i] for i in sorted(rng.choice(len(evs), 80, replace=False))]
        vap = Maai(mode="vap", lang="en", frame_rate=20, audio_ch1=MaaiInput.Zero(), audio_ch2=MaaiInput.Zero(),
                   device="cpu")
        for cur, label in evs:
            if cur["e"] - cur["s"] < 0.3:
                continue
            t_dec = cur["e"] + 0.2
            t0 = max(0.0, t_dec - 5.0)
            seg = x[int(t0 * sr):int(t_dec * sr)]
            x16 = resample_poly(seg, 16000, sr).astype(np.float32)
            vap.reset_runtime_state()
            hop, z, ps = 800, np.zeros(800, np.float32), []
            for i in range(0, len(x16) - hop + 1, hop):
                vap.process(x16[i:i + hop], z)
                r = None
                while not vap.result_dict_queue.empty():
                    r = vap.result_dict_queue.get()
                if r is not None:
                    ps.append((float(np.ravel(r["p_now"])[0]), float(np.ravel(r["p_future"])[0])))
            if len(ps) < 3:
                continue
            ipu = x16[int((cur["s"] - t0) * 16000):int((cur["e"] - t0) * 16000)]
            if len(ipu) < 4800:
                continue
            db = lambda y: 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)  # noqa: E731
            f0 = f0_track(ipu[-4800:])
            v = np.where(f0 > 0)[0]
            slope = float(np.polyfit(v, np.log2(f0[v]), 1)[0] * 100) if len(v) >= 5 else 0.0
            ws = cur["words"]
            last = ws[-1]["w"].lower().strip(".,?!")
            rows.append({"meeting": m, "speaker": agent, "label": label, "punc": ws[-1]["punc"],
                         "p_now_1": ps[-2][0], "p_now_2": ps[-1][0], "p_fut_1": ps[-2][1], "p_fut_2": ps[-1][1],
                         "ipu_dur": cur["e"] - cur["s"], "n_words": len(ws), "rate": len(ws) / (cur["e"] - cur["s"]),
                         "last_dur": ws[-1]["e"] - ws[-1]["s"], "drop_db": db(ipu[-2400:]) - db(ipu),
                         "f0_slope": slope, "voiced_end": float(len(v) / max(1, len(f0))),
                         "w_incomplete": float(last in INCOMPLETE), "w_filler": float(last in FILLERS),
                         "w_closer": float(last in CLOSERS), "last_word": last})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--max_meetings", type=int, default=40)
    a = ap.parse_args()
    ms = meetings(a.split)[:a.max_meetings]
    print(f"{a.split}: {len(ms)} meetings", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with mp.get_context("spawn").Pool(a.procs) as pool:
        rows = [r for rs in pool.imap_unordered(events_for, ms) for r in rs]
    import json
    json.dump(rows, open(os.path.join(OUT, f"{a.split}.json"), "w"), default=float)
    y = np.array([r["label"] for r in rows])
    print(f"{len(rows)} events: {int((y == 0).sum())} holds, {int(y.sum())} shifts", flush=True)
