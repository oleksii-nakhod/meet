"""Where the live avatar's ~1.3 s lag comes from (a real Boson session, ~1 min): is it per utterance (the avatar
buffering the start of each data stream) or steady (its pipeline)? Pushes, straight through DataStreamAudioOutput:

  cold     a 1 s tone as a new utterance, paced at 20 ms
  burst    a 1 s tone as a new utterance, all at once
  inside   tone 1 s, silence 1 s, tone 1 s in one utterance: the second tone's lag is the steady-state lag
  warm     3 s of silence then a 1 s tone in one utterance (a stream kept open with silence before speech)

    python tests/live_avatar_lag.py [--reps 2]"""
import argparse
import asyncio
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import live_avatar  # noqa: E402
from livekit import rtc  # noqa: E402
from live_avatar_probe import Viewer, onset  # noqa: E402
from mmhm.higgs import SR  # noqa: E402

N = SR // 50


def tone(s, f=440):
    return (0.3 * 32767 * np.sin(2 * np.pi * f * np.arange(int(s * SR)) / SR)).astype(np.int16)


async def push(out, pcm, paced=True, marks=()):
    """Frames of pcm into the open utterance; returns {sample index in marks: perf_counter when that frame went}."""
    t0, at = time.perf_counter(), {}
    for k in range(0, len(pcm) // N):
        if k * N in marks:
            at[k * N] = time.perf_counter()
        f = pcm[k * N:(k + 1) * N].tobytes()
        await out.capture_frame(rtc.AudioFrame(f, SR, 1, N))
        if paced:
            d = t0 + (k + 1) * 0.02 - time.perf_counter()
            if d > 0:
                await asyncio.sleep(d)
    return at


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    av = live_avatar.LiveAvatar("lag")
    try:
        print(f"session start {await av.start():.2f} s", flush=True)
        av.task.cancel()  # this probe drives the output itself
        out = av.out
        started = []  # does the avatar say when it starts playing? (lk.playback_started, an optional RPC)
        av.room.local_participant.register_rpc_method(
            "lk.playback_started", lambda data: started.append(time.perf_counter()) or "")
        vw = Viewer(av.avatar_identity)
        v = av.viewer("lag")
        await vw.join(v["url"], v["token"])
        await asyncio.sleep(3.0)
        res = {k: [] for k in ("cold", "burst", "inside", "warm")}
        for _ in range(a.reps):
            at = await push(out, tone(1.0), marks=(0,))
            out.flush()
            await asyncio.sleep(2.5)
            res["cold"].append(onset(vw.audio, at[0]) - at[0])
            at = await push(out, tone(1.0), paced=False, marks=(0,))
            out.flush()
            await asyncio.sleep(3.0)
            res["burst"].append(onset(vw.audio, at[0]) - at[0])
            pcm = np.concatenate([tone(1.0), np.zeros(SR, np.int16), tone(1.0)])
            at = await push(out, pcm, marks=(0, 2 * SR))
            out.flush()
            await asyncio.sleep(2.5)
            # the second tone's lag inside one utterance: its heard onset (after the heard 1 s gap) minus its push
            on1 = onset(vw.audio, at[0])
            quiet = next((t for t, r in vw.audio if t > on1 + 0.9 and r < 300), None)
            on2 = onset(vw.audio, quiet) if quiet else None
            res["inside"].append(on2 - at[2 * SR] if on2 else float("nan"))
            pcm = np.concatenate([np.zeros(3 * SR, np.int16), tone(1.0)])
            at = await push(out, pcm, marks=(3 * SR,))
            out.flush()
            await asyncio.sleep(2.5)
            res["warm"].append(onset(vw.audio, at[3 * SR]) - at[3 * SR])
            print({k: [round(x, 3) for x in v] for k, v in res.items()}, f"size {live_avatar.SIZE}",
                  f"playback_started RPCs: {len(started)}", flush=True)
        for x in vw.tasks:
            x.cancel()
        await vw.room.disconnect()
    finally:
        await av.close()


if __name__ == "__main__":
    asyncio.run(main())
