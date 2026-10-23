"""Higgs STT transcripts of MSI-Bench line audio (the live app's input: one mic, no speaker labels), for msi_eval.py
--sets stt. Each line wav (24 kHz mono PCM16, background ambience and, in background-speech-retrieval cases, a
bystander talking over it) is sent to a transcription-only Realtime session (higgs-stt-3.1, manual commits).

-> out/msi/stt_transcripts.json {"<case_id>/lines/NN.wav": text}
Usage: python msi_stt.py --pattern background_speech_retrieval [--data /tmp/msi/data] [--par 3]"""
import argparse
import asyncio
import base64
import json
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mmhm"))
from mmhm.higgs import Session  # noqa: E402

OUT = os.path.join(HERE, "out", "msi", "stt_transcripts.json")


async def transcribe_case(c, data, sem):
    out = {}
    async with sem:
        async with Session({"instructions": "Transcription only.", "output_modalities": ["text"],
                            "audio": {"input": {"turn_detection": None, "transcription": {"model": "higgs-stt-3.1"}}}}) as s:
            for t in c["turns"]:
                if not t.get("audio"):
                    continue
                pcm, sr = sf.read(os.path.join(data, c["path"], t["audio"]), dtype="int16")
                assert sr == 24000, sr
                pcm = np.asarray(pcm if pcm.ndim == 1 else pcm[:, 0])
                since = s.now()
                for k in range(0, len(pcm), 4800):  # 200 ms chunks, unpaced
                    await s.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[k:k + 4800].tobytes()).decode()})
                await s.send({"type": "input_audio_buffer.commit"})
                try:
                    _, ev = await s.wait("conversation.item.input_audio_transcription.completed", timeout=30, since=since)
                    text = (ev.get("transcript") or "").strip()
                except Exception as e:  # noqa: BLE001
                    text = ""
                    print("no transcript", c["case_id"], t["audio"], e, flush=True)
                out[f"{c['case_id']}/{t['audio']}"] = text
    print(c["case_id"], "done", flush=True)
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/w/20251/oleksii/data/msi_bench")
    ap.add_argument("--pattern", default="background_speech_retrieval")
    ap.add_argument("--par", type=int, default=3)
    a = ap.parse_args()
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cases = [c for c in map(json.loads, open(os.path.join(a.data, "index", "cases.jsonl")))
             if c["language"] == "en" and c["pattern"] == a.pattern
             and not all(f"{c['case_id']}/{t['audio']}" in done for t in c["turns"] if t.get("audio"))]
    sem = asyncio.Semaphore(a.par)
    for res in await asyncio.gather(*[transcribe_case(c, a.data, sem) for c in cases]):
        done.update(res)
    json.dump(done, open(OUT, "w"), indent=1)
    print(len(done), "transcripts")


if __name__ == "__main__":
    asyncio.run(main())
