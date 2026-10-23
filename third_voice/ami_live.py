"""Real room audio through the live app's own pipeline: stream an AMI meeting's far-field microphone (Array1-01) in
real time into app.py's WebSocket (server VAD segments it, Higgs STT transcribes, v4 decides), with the AMI project
brief as the fact sheet, and log every line, decision and reply. Nobody is labelled; there is no gold.
Usage: python ami_live.py ES2009c --start 600 --minutes 15 [--url ws://127.0.0.1:8500/ws]  -> out/ami_live_<m>.jsonl"""
import argparse
import asyncio
import json
import os
import time

import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample_poly

from ami_chatty import FACTS

HERE = os.path.dirname(os.path.abspath(__file__))
AMI = os.environ.get("AMI_DIR", "/w/150/JHUworkshop/AMI")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meeting")
    ap.add_argument("--start", type=float, default=600)
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--url", default="ws://127.0.0.1:8500/ws")
    a = ap.parse_args()
    x, sr = sf.read(f"{AMI}/amicorpus/{a.meeting}/audio/{a.meeting}.Array1-01.wav", dtype="float32",
                    start=int(a.start * 16000), frames=int(a.minutes * 60 * 16000))
    x = x[:, 0] if x.ndim > 1 else x
    pcm = (resample_poly(x / (np.abs(x).max() or 1) * 0.8, 3, 2) * 32767).astype(np.int16)
    out = open(os.path.join(HERE, "out", f"ami_live_{a.meeting}.jsonl"), "w")
    async with websockets.connect(a.url, max_size=None) as ws:
        await ws.send(json.dumps({"ai": "Nova", "setting": "a product design team meeting", "people": [], "facts": FACTS}))
        t0 = time.perf_counter()

        async def reader():
            async for msg in ws:
                if isinstance(msg, bytes):
                    continue
                m = json.loads(msg)
                if m.get("type") in ("line", "ai"):
                    m["t_min"] = round(a.start / 60 + (time.perf_counter() - t0) / 60, 2)
                    out.write(json.dumps(m) + "\n")
                    out.flush()
                    if m["type"] == "ai" or m.get("speak"):
                        print(m["t_min"], m["type"], (m.get("text") or "")[:120], flush=True)

        rd = asyncio.ensure_future(reader())
        for k in range(0, len(pcm), 480):  # 20 ms frames in real time
            await ws.send(pcm[k:k + 480].tobytes())
            await asyncio.sleep(max(0, t0 + (k + 480) / 24000 - time.perf_counter()))
        await asyncio.sleep(5)
        rd.cancel()
    rows = [json.loads(line) for line in open(os.path.join(HERE, "out", f"ami_live_{a.meeting}.jsonl"))]
    lines = [r for r in rows if r["type"] == "line"]
    print(f"{a.meeting} live: {len(lines)} lines, AI spoke {sum(r['speak'] for r in lines)} times in {a.minutes:.0f} min")


if __name__ == "__main__":
    asyncio.run(main())
