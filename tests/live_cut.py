"""Interruptions with the live face on (real Higgs and a real avatar session, ~5 min): a bot asks for a long answer and,
1.5 s after the face starts saying it, either cuts in ("stop" rounds: how soon does Meet stop Higgs?) or says
"Mm-hm." ("bc" rounds: Higgs should keep going). For an A/B of MEET_LIVE_ALIGN, run once with each.

    MEET_LIVE_ALIGN=1 python tests/live_cut.py [--rounds 8] [--port 8694]"""
import argparse
import asyncio
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bots  # noqa: E402
from live_avatar_e2e import heard, quiet_from, start_server, until  # noqa: E402
from live_avatar_probe import Viewer  # noqa: E402

ASKS = ["Higgs, can you explain in detail how a bicycle's gears work?", "Higgs, walk me through how to plan a road trip.",
        "Higgs, explain in a few sentences how a dishwasher works.", "Higgs, tell me in detail how bread is made."]
CUTS = ["Wait, wait, stop. Let me ask you something else.", "Sorry, hold on, I have a different question."]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--port", type=int, default=8694)
    a = ap.parse_args()
    for t in ASKS + CUTS + ["Mm-hm."]:
        bots.clip(t)
    srv, base = start_server(a.port)
    try:
        rid = bots.create_room(base, "e2e-admin", title="live cut", setting="a quick one-to-one chat")
        b = bots.Bot(base, rid, "Sam")
        await b.join()
        await asyncio.sleep(1.0)
        await b.json({"type": "set_mode", "mode": "improved"})
        await b.json({"type": "set_avatar", "on": True})
        msg = await until(lambda: next((m for _, m in b.evs("live_avatar") if m.get("on")), None), 40)
        if not msg:
            raise SystemExit("no live face")
        vw = Viewer(msg["identity"])
        await vw.join(msg["url"], msg["token"])
        await asyncio.sleep(4.0)
        stops, bcs = [], []
        for k in range(a.rounds):
            kind = "stop" if k % 2 == 0 else "bc"
            await b.wait_ai_quiet(1.5, 30)
            await until(lambda: quiet_from(vw, b.now() - 1.5, 1.2) is not None, 20)
            t_end = await b.say(bots.clip(ASKS[k % len(ASKS)]))
            face = await until(lambda: heard(vw, t_end), 15)
            if not face:
                print(f"round {k}: no answer heard", flush=True)
                continue
            await asyncio.sleep(1.5)
            t_cue = b.now()
            b.say(bots.clip(CUTS[(k // 2) % len(CUTS)] if kind == "stop" else "Mm-hm."))
            flush = await until(lambda: next((t for t, _ in b.evs("ai_flush", t_cue)), None), 5)
            lat = None if flush is None else round(flush - t_cue, 2)
            (stops if kind == "stop" else bcs).append(lat)
            print(f"round {k} {kind}: Meet stopped Higgs {'+' + str(lat) + ' s' if lat is not None else 'never'}", flush=True)
            await asyncio.sleep(3.0)
        got = [x for x in stops if x is not None]
        print(f"\nLIVE_ALIGN={os.environ.get('MEET_LIVE_ALIGN', '0')}: stops {len(got)}/{len(stops)}, median "
              f"{statistics.median(got) if got else None} s ({stops}); mm-hm stopped it {sum(x is not None for x in bcs)}/{len(bcs)}")
        await b.leave()
        await vw.room.disconnect()
    finally:
        srv.kill()


if __name__ == "__main__":
    asyncio.run(main())
