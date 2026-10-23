"""Do TTS 3 laughter tags produce audible laughter? 10 lines, each rendered with "<|sfx:laughter|>Haha, ..." and
as the same text without the tag; an AudioSet classifier (AST, MIT/ast-finetuned-audioset-10-10-0.4593) scores
"Laughter" on each. Also sighs. Usage: python eval/laugh_check.py [--voice <id>]"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import ASTFeatureExtractor, ASTForAudioClassification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mmhm import voices  # noqa: E402
from mmhm.higgs import tts  # noqa: E402

LINES = ["oh my god, that's hilarious.", "no way, you did not just say that.", "okay that's actually so funny.",
         "wait, the coffee went everywhere?", "stop, I can't breathe.", "that's the worst idea I've ever heard.",
         "you're kidding me, right?", "I can't believe it worked.", "that is so dumb, I love it.",
         "honestly, same thing happened to me."]
M = "MIT/ast-finetuned-audioset-10-10-0.4593"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=None)
    a = ap.parse_args()
    voice = a.voice or voices.get("maya")["voice"]
    fe, model = ASTFeatureExtractor.from_pretrained(M), ASTForAudioClassification.from_pretrained(M).eval()
    labels = model.config.id2label
    idx = {name: [i for i, l in labels.items() if l == name][0] for name in ("Laughter", "Sigh", "Speech")}

    def scores(pcm):
        x = resample_poly(pcm.astype(np.float32) / 32768, 2, 3)
        with torch.inference_mode():
            p = torch.sigmoid(model(**fe(x, sampling_rate=16000, return_tensors="pt")).logits)[0]
        return {k: float(p[i]) for k, i in idx.items()}

    rows = {"laugh_tag": [], "laugh_plain": [], "sigh_tag": [], "sigh_plain": []}
    for line in LINES:
        rows["laugh_tag"].append(scores(tts("<|sfx:laughter|>Haha, " + line, voice=voice)[0])["Laughter"])
        rows["laugh_plain"].append(scores(tts("Haha, " + line, voice=voice)[0])["Laughter"])
        rows["sigh_tag"].append(scores(tts("<|sfx:sigh|>Uh, " + line, voice=voice)[0])["Sigh"])
        rows["sigh_plain"].append(scores(tts("Uh, " + line, voice=voice)[0])["Sigh"])
    for k, v in rows.items():
        v = np.array(v)
        print(f"{k:12s} median {np.median(v):.3f}  >0.3 in {int((v > 0.3).sum())}/{len(v)}")


if __name__ == "__main__":
    main()
