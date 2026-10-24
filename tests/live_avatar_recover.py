"""The live face when its avatar goes away mid-call (a real Boson session and LiveKit room, stub agent, ~1 min): the
avatar participant is removed from its LiveKit room, as when Boson ends a session (expired, capacity, a crash). The
room should notice, start a new session, send every page the new one, and the face should speak again.

    python tests/live_avatar_recover.py [--port 8615]"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bots  # noqa: E402
import live_avatar  # noqa: E402
from fake_room import ADMIN, MEET, start_server  # noqa: E402
from live_avatar_probe import Viewer  # noqa: E402
from livekit import api  # noqa: E402

RESULTS = []


def check(name, ok, info=""):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}", flush=True)


async def until(pred, timeout):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        v = pred()
        if v:
            return v
        await asyncio.sleep(0.1)
    return None


async def heard_beep(b, vw, timeout=8.0):
    """Ask the stub for a beep; True if the face's own audio carries it."""
    n0 = len(vw.audio)
    await b.json({"type": "chat", "text": "/beep"})
    return bool(await until(lambda: any(r > 1500 for _, r in vw.audio[n0:]), timeout))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8615)
    a = ap.parse_args()
    cfg = live_avatar.settings()
    if not cfg:
        raise SystemExit("LIVEKIT_* not set")
    srv, base = start_server(a.port)
    try:
        rid = bots.create_room(base, ADMIN, title="recover")
        b = bots.Bot(base, rid, "Ana")
        await b.join()
        await asyncio.sleep(1.0)
        await b.json({"type": "set_avatar", "on": True})
        msg = await until(lambda: next((m for _, m in b.evs("live_avatar") if m.get("on")), None), 40)
        check("the face starts", bool(msg))
        if not msg:
            return
        vw = Viewer(msg["identity"])
        await vw.join(msg["url"], msg["token"])
        await asyncio.sleep(2.0)
        check("…and speaks", await heard_beep(b, vw))
        lk = api.LiveKitAPI(cfg["LIVEKIT_URL"].replace("wss://", "https://"), cfg["LIVEKIT_API_KEY"], cfg["LIVEKIT_API_SECRET"])
        t0 = time.perf_counter()
        await lk.room.remove_participant(api.RoomParticipantIdentity(room=msg["room"], identity=msg["identity"]))
        off = await until(lambda: next((m for _, m in b.evs("live_avatar") if not m.get("on")), None), 15)
        check("the page is told the face is gone", bool(off), f"after {time.perf_counter() - t0:.1f} s" if off else "")
        new = await until(lambda: next((m for _, m in b.evs("live_avatar") if m.get("on") and m["room"] != msg["room"]),
                                       None), 60)
        check("a new session reaches the page", bool(new), f"after {time.perf_counter() - t0:.1f} s" if new else "")
        if new:
            await vw.room.disconnect()
            vw = Viewer(new["identity"])
            await vw.join(new["url"], new["token"])
            await asyncio.sleep(2.0)
            check("…and the new face speaks", await heard_beep(b, vw))
        await b.leave()
        await vw.room.disconnect()
        await lk.aclose()
        await asyncio.sleep(3.0)
        log = [json.loads(x) for x in open(os.path.join(MEET, "out", "rooms", f"{rid}.jsonl"))]
        lv = [(x["on"], x.get("start_s") or x.get("why")) for x in log if x["type"] == "live_avatar"]
        check("room log: started, ended, started, released", [x[0] for x in lv] == [True, False, True, False], lv)
    finally:
        srv.kill()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    asyncio.run(main())
