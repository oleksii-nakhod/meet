"""Manual integration check using real LiveKit mic/received avatar audio and configured Boson credentials.

Run against the development server: .venv/bin/python tests/livekit_audio_smoke.py
This calls paid upstream APIs. No browser, keys or tokens are printed.
"""
import asyncio
import json
import os
import secrets
import sys
import time
from urllib.parse import urlencode

import numpy as np
import websockets
from livekit import rtc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meet_config import load_environment

load_environment()
from bots import clip, http

BASE = "http://127.0.0.1:8600"
SR, N = 24000, 480


async def main():
    question, acknowledgement, stop = await asyncio.gather(*[
        asyncio.to_thread(clip, text) for text in (
            "Higgs, explain in two sentences why the Earth has seasons.",
            "Umm, ahh, okay, okay, yeah.", "Higgs, stop talking.")])
    created = await asyncio.to_thread(http, BASE + "/api/rooms", {
        "title": "LiveKit audio regression", "mode": "improved", "setting": "a conversation",
        "facts": ["The project deadline is Tuesday at 3 PM."]})
    rid, pid = created["id"], secrets.token_hex(12)
    events, audio, tasks = [], [], []
    queue = asyncio.Queue()
    room = rtc.Room()
    q = urlencode({"room": rid, "pid": pid, "name": "Audio test"})
    async with websockets.connect(BASE.replace("http", "ws") + "/ws/room?" + q) as ws:
        async def controls():
            async for raw in ws:
                if isinstance(raw, str):
                    events.append((time.perf_counter(), json.loads(raw)))
        tasks.append(asyncio.create_task(controls()))

        async def wait_event(predicate, since=0.0, timeout=35):
            async def wait():
                while True:
                    for at, event in events:
                        if at >= since and predicate(event):
                            return at, event
                    await asyncio.sleep(0.05)
            return await asyncio.wait_for(wait(), timeout)

        async def read_audio(track, identity):
            async for ev in rtc.AudioStream(track, sample_rate=SR, num_channels=1):
                pcm = np.frombuffer(ev.frame.data, np.int16).astype(np.float32)
                audio.append((time.perf_counter(), identity, float(np.sqrt(np.mean(pcm * pcm)))))

        @room.on("track_subscribed")
        def subscribed(track, publication, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                tasks.append(asyncio.create_task(read_audio(track, participant.identity)))

        try:
            await wait_event(lambda e: e.get("type") == "roster")
            credentials = await asyncio.to_thread(http, BASE + f"/api/rooms/{rid}/livekit-token", {
                "pid": pid, "name": "Audio test"})
            await room.connect(credentials["url"], credentials["token"])
            source = rtc.AudioSource(SR, 1, queue_size_ms=200)
            track = rtc.LocalAudioTrack.create_audio_track("regression-mic", source)
            opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            await room.local_participant.publish_track(track, opts)

            async def microphone():
                next_at = time.perf_counter()
                while True:
                    try:
                        pcm = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pcm = np.zeros(N, np.int16)
                    await source.capture_frame(rtc.AudioFrame(pcm.tobytes(), SR, 1, N))
                    next_at += 0.02
                    await asyncio.sleep(max(0, next_at - time.perf_counter()))
            tasks.append(asyncio.create_task(microphone()))

            async def say(pcm):
                for i in range(0, len(pcm), N):
                    frame = np.zeros(N, np.int16)
                    piece = pcm[i:i + N]
                    frame[:len(piece)] = piece
                    queue.put_nowait(frame)
                while not queue.empty():
                    await asyncio.sleep(0.02)

            await ws.send(json.dumps({"type": "set_avatar", "on": True}))
            await wait_event(lambda e: e.get("type") == "ai_state" and e.get("avatar") is True)
            await asyncio.sleep(2)
            started = time.perf_counter()
            await say(question)
            _, answer = await wait_event(lambda e: e.get("type") == "ai_say", since=started)
            audible_by = time.perf_counter() + 20
            while not any(t >= started and identity == "boson-avatar" and rms > 150 for t, identity, rms in audio):
                if time.perf_counter() > audible_by:
                    raise AssertionError("No audible avatar audio after reply")
                await asyncio.sleep(0.05)
            ack_start = time.perf_counter()
            await say(acknowledgement)
            await wait_event(lambda e: e.get("type") == "line" and
                             e.get("decision", {}).get("why") == "listener acknowledgement", since=ack_start)
            await wait_event(lambda e: e.get("type") == "ai_state" and e.get("status") == "listening",
                             since=ack_start)
            assert not any(t >= started and (e.get("type") == "interruption_confirmed" or
                           (e.get("type") == "ai_state" and e.get("stopped"))) for t, e in events)
            print(json.dumps({"stage": "acknowledgement", "result": "PASS", "room": rid,
                              "reply": answer["text"], "audible_avatar_frames": sum(
                                  identity == "boson-avatar" and rms > 150 for _, identity, rms in audio)}, indent=2), flush=True)

            start_stop = time.perf_counter()
            await ws.send(json.dumps({"type": "chat", "text": "Explain in two sentences how ocean tides work."}))
            await wait_event(lambda e: e.get("type") == "ai_say", since=start_stop)
            await asyncio.sleep(0.4)
            await say(stop)
            _, confirmed = await wait_event(lambda e: e.get("type") == "interruption_confirmed", since=start_stop)
            print(json.dumps({"stage": "explicit_stop", "result": "PASS", "text": confirmed["text"]}), flush=True)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
