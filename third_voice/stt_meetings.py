"""Speech round trip for the speak/silent test sets: every human line is spoken by Higgs TTS 3 (one preset voice per
person) and transcribed back by Higgs STT (higgs-stt-3.1, Realtime session, manual commits), so the policies can be
scored on what a live system would actually read (punctuation, names and "?" as the recogniser hears them).
The AI's own earlier replies stay as text (it knows what it said).

-> out/stt/<meeting>/<line>.flac (16 kHz), out/stt/transcripts.json, testset/points_{dev,test}_stt.jsonl,
   testset/points_meeting_stt.jsonl
Usage: python stt_meetings.py [--par 3]"""
import argparse
import asyncio
import base64
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mmhm"))
from mmhm.higgs import Session, tts  # noqa: E402

import meeting  # noqa: E402
from testset.build_points import points  # noqa: E402

VOICES = ["eleanor", "marcus", "nora", "oliver", "chloe", "jake"]
OUT = os.path.join(HERE, "out", "stt")


def meetings():
    ms = [json.loads(line) for f in "ABC" for line in open(os.path.join(HERE, "testset", f"meetings_{f}.jsonl"))]
    old = {"meeting_id": "M0", "ai_name": "Higgs", "people": ["Ana", "Ben", "Chen"], "setting": "hackathon team meeting",
           "facts": [meeting.FACTS.split(": ", 1)[1]],
           "lines": [{"speaker": s, "text": t, "label": g, "category": "?", "ai_reply": ai} for s, t, g, ai in meeting.TURNS]}
    return [old] + ms


async def transcribe_meeting(m, sem):
    os.makedirs(os.path.join(OUT, m["meeting_id"]), exist_ok=True)
    voice = {p: VOICES[i % len(VOICES)] for i, p in enumerate(m["people"])}
    out = []
    async with sem:
        async with Session({"instructions": "Transcription only.", "output_modalities": ["text"],
                            "audio": {"input": {"turn_detection": None, "transcription": {"model": "higgs-stt-3.1"}}}}) as s:
            for i, d in enumerate(m["lines"]):
                path = os.path.join(OUT, m["meeting_id"], f"{i:02d}.flac")
                if os.path.exists(path):
                    pcm16 = sf.read(path, dtype="int16")[0]
                else:
                    pcm, _, _ = await asyncio.to_thread(tts, d["text"], voice.get(d["speaker"], "eleanor"))
                    pcm16 = resample_poly(pcm.astype(np.float32), 2, 3).clip(-32768, 32767).astype(np.int16)
                    sf.write(path, pcm16, 16000)
                pcm24 = resample_poly(pcm16.astype(np.float32), 3, 2).clip(-32768, 32767).astype(np.int16)
                since = s.now()
                for k in range(0, len(pcm24), 4800):  # 200 ms chunks, unpaced
                    await s.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm24[k:k + 4800].tobytes()).decode()})
                await s.send({"type": "input_audio_buffer.commit"})
                try:
                    _, ev = await s.wait("conversation.item.input_audio_transcription.completed", timeout=30, since=since)
                    text = (ev.get("transcript") or "").strip()
                except Exception as e:  # noqa: BLE001
                    text = ""
                    print("no transcript", m["meeting_id"], i, e, flush=True)
                out.append(text)
    print(m["meeting_id"], "done", flush=True)
    return m["meeting_id"], out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--par", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ms = meetings()
    tpath = os.path.join(OUT, "transcripts.json")
    done = json.load(open(tpath)) if os.path.exists(tpath) else {}
    sem = asyncio.Semaphore(a.par)
    res = await asyncio.gather(*[transcribe_meeting(m, sem) for m in ms if m["meeting_id"] not in done])
    done.update(dict(res))
    json.dump(done, open(tpath, "w"), indent=1)
    for split, prefixes in (("dev", "A"), ("test", "BC"), ("meeting", "M")):
        rows = []
        for m in ms:
            if m["meeting_id"][0] not in prefixes:
                continue
            m2 = json.loads(json.dumps(m))
            for d, t in zip(m2["lines"], done[m["meeting_id"]]):
                d["text"] = t
            rows += points(m2)
        dst = os.path.join(HERE, "testset", f"points_{split}_stt.jsonl")
        with open(dst, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(split, len(rows))


if __name__ == "__main__":
    asyncio.run(main())
