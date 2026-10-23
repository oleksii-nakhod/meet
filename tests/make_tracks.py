"""Turn a labelled Third Voice test meeting into one fake-microphone track per person, for tests/fake_room.py.

Each line's audio is the TTS rendering already on disk (tests/data/tts/<meeting>/NN.flac, one preset voice per
person, made by stt_meetings.py), so this makes no Higgs calls. Lines follow each other with a short gap; after a
line where the AI should speak, the next person waits long enough for it to answer (a polite room). Every track
starts with LEAD seconds of silence (joining, opening sessions) and ends with a long silence, because Chromium loops
the fake-mic file.

-> tests/tracks/<meeting>/<person>.wav (48 kHz, as browser_test.py uses) + gold.json (per line: who, when, label)
Usage: python tests/make_tracks.py [B2] [--lead 20]"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GAP, AFTER_SPEAK, TAIL = 0.7, 6.0, 120.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meeting", nargs="?", default="B2")
    ap.add_argument("--lead", type=float, default=20.0)
    a = ap.parse_args()
    m = next(json.loads(x) for f in "BC" for x in open(os.path.join(DATA, "testset", f"meetings_{f}.jsonl"))
             if json.loads(x)["meeting_id"] == a.meeting)
    clips = [sf.read(os.path.join(DATA, "tts", a.meeting, f"{i:02d}.flac"), dtype="float32")[0]
             for i in range(len(m["lines"]))]
    t, gold = a.lead, []
    for d, c in zip(m["lines"], clips):
        gold.append({"speaker": d["speaker"], "text": d["text"], "label": d["label"], "category": d.get("category"),
                     "start": round(t, 2), "end": round(t + len(c) / 16000, 2)})
        t += len(c) / 16000 + (AFTER_SPEAK if d["label"] == "SPEAK" else GAP)
    total = int((t + TAIL) * 16000)
    out = os.path.join(HERE, "tracks", a.meeting)
    os.makedirs(out, exist_ok=True)
    for p in m["people"]:
        track = np.zeros(total, np.float32)
        for g, c in zip(gold, clips):
            if g["speaker"] == p:
                s = int(g["start"] * 16000)
                track[s:s + len(c)] = c
        sf.write(os.path.join(out, f"{p}.wav"), resample_poly(track, 3, 1), 48000, subtype="PCM_16")
    json.dump({"meeting": a.meeting, "ai": m["ai_name"], "people": m["people"], "setting": m["setting"],
               "facts": m["facts"], "lead": a.lead, "lines": gold}, open(os.path.join(out, "gold.json"), "w"), indent=1)
    print(f"{a.meeting}: {len(gold)} lines, {sum(g['label'] == 'SPEAK' for g in gold)} SPEAK, "
          f"meeting ends at {t:.0f} s -> {out}")


if __name__ == "__main__":
    main()
