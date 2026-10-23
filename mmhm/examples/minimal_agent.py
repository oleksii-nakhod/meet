"""The smallest useful Mm-hm agent: your laptop's mic and speakers, Higgs Realtime behind the turn-taking layer.

    pip install -e .[mic] && pip install --no-deps maai==0.2.25
    BOSON_API_KEY=... python examples/minimal_agent.py            # talk (use headphones)
    python examples/minimal_agent.py --wav some_speech.wav          # no mic: play a file in as the "mic"
"""
import argparse
import asyncio
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mmhm.engine import N, Engine, Policy  # noqa: E402
from mmhm.higgs import SR, load_wav, save_wav  # noqa: E402
from mmhm.models import Signals  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", help="use this file as the microphone (then save what the speaker played)")
    ap.add_argument("--instructions", default="You are a friendly person. Reply in one or two short sentences.")
    a = ap.parse_args()
    log = []
    async with Engine(mode="mmhm", policy=Policy(backchannels=False), instructions=a.instructions,
                      signals=Signals(), log=log) as engine:
        if a.wav:  # offline: feed the file in real time, then a few seconds of silence for the reply
            mic = np.concatenate([load_wav(a.wav), np.zeros(SR * 8, np.int16)])
            out = []
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            for i in range(0, len(mic) - N, N):
                out.append(await engine.tick(mic[i:i + N]))
                await asyncio.sleep(max(0, t0 + (i + N) / SR - loop.time()))
            save_wav("minimal_agent_out.wav", np.concatenate(out))
            print("\n".join(f"{r['t']:6.2f}  {r['kind']}  {r.get('text', '')}" for r in log
                            if r["kind"] in ("commit", "respond", "user_text", "response_done", "stop")))
            return
        import sounddevice as sd  # live: one duplex stream, 20 ms blocks, the engine is the clock
        q_in = asyncio.Queue()
        loop = asyncio.get_running_loop()
        speaker = np.zeros(N, np.int16)

        def callback(indata, outdata, frames, time_info, status):
            outdata[:, 0] = speaker
            loop.call_soon_threadsafe(q_in.put_nowait, indata[:, 0].copy())

        with sd.Stream(samplerate=SR, blocksize=N, dtype="int16", channels=1, callback=callback):
            print("Talking to Higgs through Mm-hm. Ctrl-C to stop.")
            while True:
                speaker[:] = await engine.tick(await q_in.get())


if __name__ == "__main__":
    asyncio.run(main())
