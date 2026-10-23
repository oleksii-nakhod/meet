"""Third Voice on real meeting *audio*: each segment of the AMI meeting (manual segmentation) is cut from the room's
far-field microphone (Array1-01, the closest thing to one shared mic on the table), transcribed by Higgs STT, and
labelled "Someone" (no diarisation). -> testset/points_ami_<meeting>_far.jsonl, for tv_eval.py / ami_chatty.py --show.
Usage: python ami_farfield.py ES2009c"""
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
from mmhm.higgs import Session  # noqa: E402

AMI = os.environ.get("AMI_DIR", "/w/150/JHUworkshop/AMI")


async def main(meeting):
    rows = [json.loads(line) for line in open(os.path.join(HERE, "testset", f"points_ami_{meeting}.jsonl"))]
    audio, sr = sf.read(f"{AMI}/amicorpus/{meeting}/audio/{meeting}.Array1-01.wav", dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    peak = np.abs(audio).max() or 1.0
    texts = []
    async with Session({"instructions": "Transcription only.", "output_modalities": ["text"],
                        "audio": {"input": {"turn_detection": None, "transcription": {"model": "higgs-stt-3.1"}}}}) as s:
        for r in rows:
            seg = audio[int(r["t"] * sr):int((r["t_end"] + 0.15) * sr)] / peak * 0.8
            pcm = (resample_poly(seg, 24000, sr) * 32767).clip(-32768, 32767).astype(np.int16)
            since = s.now()
            for k in range(0, len(pcm), 4800):
                await s.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[k:k + 4800].tobytes()).decode()})
            await s.send({"type": "input_audio_buffer.commit"})
            try:
                _, ev = await s.wait("conversation.item.input_audio_transcription.completed", timeout=30, since=since)
                texts.append((ev.get("transcript") or "").strip())
            except Exception:  # noqa: BLE001
                texts.append("")
    out, history = [], []
    for r, t in zip(rows, texts):
        cur = {"speaker": "Someone", "text": t}
        out.append({**r, "context_turns": list(history), "current_turn": cur, "manual_text": r["current_turn"]["text"]})
        if t:
            history.append(cur)
    out = [o for o in out if o["current_turn"]["text"]]
    with open(os.path.join(HERE, "testset", f"points_ami_{meeting}_far.jsonl"), "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(meeting, len(out), "transcribed segments; e.g.:")
    for o in out[40:45]:
        print("  manual:", o["manual_text"][:90], "\n  far STT:", o["current_turn"]["text"][:90])


asyncio.run(main(sys.argv[1]))
