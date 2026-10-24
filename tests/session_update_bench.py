"""How long does a Realtime session.update that only changes the instructions take on an open voice session (does it
validate the preset voice again)? Then one answer with the new instructions, to see they apply.

    python tests/session_update_bench.py"""
import asyncio
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmhm.higgs import Session  # noqa: E402

VOICE = "chloe"
A = "You are a text-to-speech engine. Read the user's text aloud exactly as written."
B = "You are Nova, an AI in a meeting. Answer the latest line in one short sentence."


async def main():
    s = Session({"instructions": A, "output_modalities": ["audio"], "audio": {"output": {"voice": VOICE}}})
    t = time.perf_counter()
    await s.__aenter__()
    print(f"open {time.perf_counter() - t:.2f} s", flush=True)
    ts = []
    for k in range(6):
        since = s.now()
        t = time.perf_counter()
        await s.send({"type": "session.update", "session": {"instructions": B if k % 2 == 0 else A}})
        _, ev = await s.wait(lambda e: e.get("type") in ("session.updated", "error"), timeout=10, since=since)
        ts.append(time.perf_counter() - t)
        print(ev.get("type"), f"{ts[-1]:.3f} s", flush=True)
    await s.send({"type": "session.update", "session": {"instructions": B}})
    await asyncio.sleep(0.3)
    since = s.now()
    t = time.perf_counter()
    await s.say_text("Sam: Nova, what's the capital of Australia?")
    _, ev = await s.wait("response.done", timeout=20, since=since)
    first = next((tt for tt, e in s.events if tt >= since and e.get("type") == "response.output_audio.delta"), None)
    print("answer:", s.transcript(ev["response"]["id"]), f"first audio {first - since:.2f} s" if first else "")
    print("update median", round(statistics.median(ts), 3))
    await s.__aexit__()


if __name__ == "__main__":
    asyncio.run(main())
