"""Real human speech for the turn-taking evaluation, cut from AMI meeting headset recordings (CC BY 4.0).

HOLD: the speaker stops mid-sentence (no sentence-final punctuation), is silent 0.8-1.6 s while nobody else
      speaks, then continues. Clip = up to 5 s before the pause + the pause + the continuation (<= 3 s).
COMPLETE: the speaker finishes (sentence-final punctuation), stays silent >= 2 s, and someone else takes over
      within 1.5 s. Clip = up to 5 s before the end.
Other people's speech bleeding into the headset is muted outside the target speaker's word timings.
Meetings are ones not used on Oct 10 (ES2002a/b, IS1001a, TS3003a).

Usage: python eval/ami_stimuli.py [--n 12]  -> out/ami_stim/*.wav + items.json"""
import argparse
import json
import os
import random
import sys
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mmhm.higgs import SR, outdir  # noqa: E402

AMI = os.environ.get("AMI_DIR", "/w/150/JHUworkshop/AMI")  # amicorpus/ (headset audio) + ami_public_manual_1.5/
MEETINGS = ["ES2008a", "ES2011c", "ES2015b", "ES2005a", "ES2013b", "ES2016c"]
# a second, independent set (different meetings and speakers), for testing after tuning on the first
MEETINGS_2 = ["ES2004b", "ES2007c", "ES2009d", "ES2012a", "ES2014b", "ES2006c"]
OUT = outdir("ami_stim")


def words(meeting, agent):
    path = f"{AMI}/ami_public_manual_1.5/words/{meeting}.{agent}.words.xml"
    out = []
    for w in ET.parse(path).getroot():
        if w.tag != "w":
            continue
        s, e = w.get("starttime"), w.get("endtime")
        if w.get("punc") == "true":
            if out:
                out[-1]["punc"] = (w.text or "").strip()
            continue
        if s and e:
            out.append({"s": float(s), "e": float(e), "w": (w.text or "").strip(), "punc": ""})
    return out


def ipus(ws, gap=0.2):
    res = []
    for w in ws:
        if res and w["s"] - res[-1]["e"] < gap:
            res[-1]["e"] = max(res[-1]["e"], w["e"]); res[-1]["words"].append(w)
        else:
            res.append({"s": w["s"], "e": w["e"], "words": [w]})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--set2", action="store_true", help="the second, independent set of meetings")
    a = ap.parse_args()
    global OUT
    meetings = MEETINGS_2 if a.set2 else MEETINGS
    if a.set2:
        OUT = outdir("ami_stim2")
    holds, comps = [], []
    for m in meetings:
        for ch, agent in enumerate("ABCD"):
            wav = f"{AMI}/amicorpus/{m}/audio/{m}.Headset-{ch}.wav"
            if not os.path.exists(wav):
                continue
            mine = words(m, agent)
            others = sorted(w["s"] for o in "ABCD" if o != agent for w in words(m, o))
            others = np.array(others)
            segs = ipus(mine)
            for k in range(len(segs) - 1):
                cur, nxt = segs[k], segs[k + 1]
                gap = nxt["s"] - cur["e"]
                last = cur["words"][-1]
                quiet = not ((others > cur["e"] - 0.2) & (others < nxt["s"] + 0.2)).any()
                if 0.8 <= gap <= 1.6 and quiet and last["punc"] not in (".", "?", "!") and cur["e"] - cur["s"] >= 1.0 \
                        and nxt["e"] - nxt["s"] >= 0.8:
                    holds.append((m, ch, agent, cur, nxt))
                after = nxt["s"] - cur["e"]
                if last["punc"] in (".", "?") and after >= 2.0 and cur["e"] - cur["s"] >= 1.0 and \
                        ((others > cur["e"] - 0.2) & (others < cur["e"] + 1.5)).any():
                    comps.append((m, ch, agent, cur, None))
    rng = random.Random(0)
    rng.shuffle(holds); rng.shuffle(comps)
    print(f"candidates: {len(holds)} holds, {len(comps)} completes")
    items = []
    for kind, pool in (("hold", holds[:a.n]), ("complete", comps[:a.n])):
        for i, (m, ch, agent, cur, nxt) in enumerate(pool):
            x, sr = sf.read(f"{AMI}/amicorpus/{m}/audio/{m}.Headset-{ch}.wav", dtype="float32")
            t0 = max(0.0, cur["e"] - 5.0)
            t1 = min(nxt["e"], nxt["s"] + 3.0) if nxt else cur["e"] + 0.05
            seg = x[int(t0 * sr):int(t1 * sr)].copy()
            # mute everything outside this speaker's words (bleed from the other headsets), 50 ms margins
            mask = np.zeros(len(seg), bool)
            for w in words(m, agent):
                if w["e"] >= t0 and w["s"] <= t1:
                    mask[max(0, int((w["s"] - t0 - 0.05) * sr)):int((w["e"] - t0 + 0.05) * sr)] = True
            seg = np.where(mask, seg, seg * 0.003)
            seg = resample_poly(seg, SR, sr)
            seg = seg / (np.sqrt(np.mean(seg[np.abs(seg) > 1e-4] ** 2)) + 1e-9) * 0.08  # ~ -22 dBFS speech
            pcm = (np.clip(seg, -1, 1) * 32767).astype(np.int16)
            name = f"{kind}_{i:02d}_{m}_{agent}.wav"
            sf.write(os.path.join(OUT, name), pcm, SR, subtype="PCM_16")
            text = " ".join(w["w"] for w in cur["words"][-12:]) + (" ... " + " ".join(w["w"] for w in nxt["words"][:8]) if nxt else "")
            items.append({"kind": kind, "file": name, "pause_at": round(cur["e"] - t0, 3),
                          "duration": round(len(pcm) / SR, 3), "text": text, "meeting": m, "speaker": agent})
    json.dump(items, open(os.path.join(OUT, "items.json"), "w"), indent=1)
    print(f"wrote {len(items)} clips to {OUT}")
    for it in items[:4] + items[-3:]:
        print(" ", it["kind"], it["meeting"], it["speaker"], "|", it["text"][:110])


if __name__ == "__main__":
    main()
