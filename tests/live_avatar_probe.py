"""The live avatar on its own (live_avatar.py; a real Boson session and LiveKit room, ~1 min): how long a session takes
to start, how long after a frame is pushed the avatar plays it, whether its audio runs dry while Meet's tick loop
sends 20 ms frames (on time and with jitter), how fast clear() silences it, and the video frame rate.

    python tests/live_avatar_probe.py [--lead 0 5 10] [--jitter 0.015]

A viewer joins the room as a page would (subscribe only) and records the avatar's audio level and video frames. The
pushed sound is a 440 Hz tone, so a dropout is any 20 ms window inside it that is far quieter than the tone."""
import argparse
import asyncio
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import live_avatar  # noqa: E402
from livekit import rtc  # noqa: E402
from mmhm.higgs import SR  # noqa: E402

N = SR // 50


class Viewer:
    """Joins the avatar's LiveKit room subscribe-only (as a page does) and records its audio level and video frames."""

    def __init__(self, identity):
        self.identity, self.room = identity, rtc.Room()
        self.audio, self.video = [], []  # (t, rms) per 20 ms; t per video frame
        self.tasks = []
        self.room.on("track_subscribed", self._on_track)

    def _on_track(self, track, _pub, p):
        if p.identity != self.identity:
            return
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            self.tasks.append(asyncio.ensure_future(self._audio(track)))
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            self.tasks.append(asyncio.ensure_future(self._video(track)))

    async def _audio(self, track):
        buf = np.zeros(0, np.int16)
        async for ev in rtc.AudioStream(track, sample_rate=SR, num_channels=1):
            buf = np.concatenate([buf, np.frombuffer(bytes(ev.frame.data), np.int16)])
            while len(buf) >= N:
                f, buf = buf[:N].astype(np.float32), buf[N:]
                self.audio.append((time.perf_counter(), float(np.sqrt(np.mean(f * f)))))

    async def _video(self, track):
        async for _ev in rtc.VideoStream(track):
            self.video.append(time.perf_counter())

    async def join(self, url, token):
        await self.room.connect(url, token)

    def fps(self, t0, t1):
        n = sum(t0 <= t < t1 for t in self.video)
        return n / max(1e-6, t1 - t0)


async def push_tone(av, seconds, jitter=0.0, clear_at=None):
    """Push `seconds` of tone, one 20 ms frame per tick like agent._run (a late tick catches up). Returns (t_first, t_clear)."""
    k, tone = 0, (0.3 * 32767 * np.sin(2 * np.pi * 440 * np.arange(int(seconds * SR)) / SR)).astype(np.int16)
    t0, t_clear = time.perf_counter(), None
    while k * N < len(tone):
        if clear_at is not None and k * 0.02 >= clear_at:
            av.clear()
            t_clear = time.perf_counter()
            break
        av.push(tone[k * N:(k + 1) * N].tobytes())
        k += 1
        delay = t0 + k * 0.02 - time.perf_counter() + (random.uniform(0, jitter) if jitter else 0.0)
        if delay > 0:
            await asyncio.sleep(delay)
    return t0, t_clear


def onset(audio, after, thr=1500.0):
    return next((t for t, r in audio if t >= after and r > thr), None)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, nargs="+", default=[0, 5])
    ap.add_argument("--jitter", type=float, default=0.015, help="extra random lateness per tick, seconds")
    ap.add_argument("--seconds", type=float, default=3.0)
    a = ap.parse_args()
    if not live_avatar.settings():
        raise SystemExit("LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET are not set")
    av = live_avatar.LiveAvatar("probe")
    t = time.perf_counter()
    try:
        print(f"session start: {await av.start():.2f} s (avatar {live_avatar.AVATAR_ID}, room {av.room_name})", flush=True)
        vw = Viewer(av.avatar_identity)
        v = av.viewer("probe")
        await vw.join(v["url"], v["token"])
        await asyncio.sleep(3.0)
        t_idle = time.perf_counter()
        await asyncio.sleep(2.0)
        print(f"idle video: {vw.fps(t_idle, time.perf_counter()):.1f} fps, {len(vw.audio)} audio windows so far", flush=True)
        for lead in a.lead:
            for jit in (0.0, a.jitter):
                live_avatar.LEAD_FRAMES = lead
                t0, _ = await push_tone(av, a.seconds, jitter=jit)
                await asyncio.sleep(2.0)
                on = onset(vw.audio, t0)
                if on is None:
                    print(f"lead {lead} jitter {jit}: no audio heard", flush=True)
                    continue
                inside = [r for t, r in vw.audio if on + 0.1 <= t < on + a.seconds - 0.1]
                quiet = sum(r < 0.3 * 0.3 * 32767 / np.sqrt(2) for r in inside)
                print(f"lead {lead} jitter {jit}: heard {on - t0:.3f} s after the first push; dropouts {quiet}/{len(inside)} "
                      f"windows; video while speaking {vw.fps(on, on + a.seconds):.1f} fps", flush=True)
        live_avatar.LEAD_FRAMES = 5
        t0, tc = await push_tone(av, 4.0, clear_at=1.5)
        await asyncio.sleep(2.0)
        stop = next((t for t, r in vw.audio if t >= tc and r < 300), None)
        print(f"clear: silent {stop - tc:.3f} s after clear()" if stop else "clear: still loud after 2 s", flush=True)
        t0, _ = await push_tone(av, 1.0)  # a new utterance right after a clear still plays
        await asyncio.sleep(2.0)
        on = onset(vw.audio, t0)
        print(f"after clear: next utterance heard {on - t0:.3f} s after its push" if on else "after clear: next utterance never heard")
        print(f"dead: {av.dead.is_set()}, frames sent {av.frames}, total {time.perf_counter() - t:.0f} s", flush=True)
        for x in vw.tasks:
            x.cancel()
        await vw.room.disconnect()
    finally:
        await av.close()


if __name__ == "__main__":
    asyncio.run(main())
