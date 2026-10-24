"""The live face with real Higgs (real API calls, ~3 min a mode): a bot asks Higgs questions in a one-to-one call with
the avatar on, and a LiveKit viewer (joined with the bot's own `live_avatar` token, as its page would) records when
the face actually says the reply. Then the bot cuts into a long answer.

    python tests/live_avatar_e2e.py [--modes improved default] [--port 8693]

Prints, per question: when the server started sending the reply (slot 0 on the socket) and when the face was heard,
both after the end of the question; for the interruption: when Meet stopped Higgs (ai_flush: pages mute the face
then) and when the face's own audio went quiet."""
import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bots  # noqa: E402
from live_avatar_probe import Viewer  # noqa: E402

MEET = bots.MEET
ADMIN = "e2e-admin"
QUESTIONS = ["Higgs, what's the capital of Australia?", "Higgs, how many legs does a spider have?",
             "Higgs, what is a good name for a pet goldfish?", "Higgs, can you suggest a quick lunch idea?"]
LONG = "Higgs, can you explain in detail how a bicycle's gears work?"
CUT = "Wait, wait, stop. Let me ask you something else."
LOUD = 150.0  # int16 RMS per 20 ms: Higgs is talking (bots.AI_RMS)


def start_server(port):
    env = {**os.environ, "MEET_ADMIN": ADMIN, "MEET_BUDGET": "100000"}
    env.pop("MEET_AGENT", None)
    os.makedirs(os.path.join(MEET, "out"), exist_ok=True)
    log = open(os.path.join(MEET, "out", f"live_e2e_server_{port}.log"), "w")
    p = subprocess.Popen([sys.executable, "server.py", "--port", str(port)], cwd=MEET, env=env, stdout=log,
                         stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    for _ in range(300):
        try:
            bots.http(base + "/api/config")
            return p, base
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    raise RuntimeError("server did not start")


def heard(vw, since, until=None):
    """First time (bot clock) after `since` the face's audio is loud, or None."""
    return next((t - bots.T0 for t, r in vw.audio if t - bots.T0 >= since and r > LOUD
                 and (until is None or t - bots.T0 < until)), None)


def quiet_from(vw, since, hold=0.4):
    """First time after `since` the face's audio stays quiet for `hold` s."""
    run = None
    for t, r in vw.audio:
        t -= bots.T0
        if t < since:
            continue
        if r > LOUD:
            run = None
        elif run is None:
            run = t
        elif t - run >= hold:
            return run
    return None


async def until(pred, timeout):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        v = pred()
        if v:
            return v
        await asyncio.sleep(0.05)
    return None


async def run_mode(base, mode):
    rid = bots.create_room(base, ADMIN, title=f"live e2e {mode}", setting="a quick one-to-one chat")
    b = bots.Bot(base, rid, "Sam")
    await b.join()
    await asyncio.sleep(1.0)
    await b.json({"type": "set_mode", "mode": mode})
    await b.json({"type": "set_avatar", "on": True})
    msg = await until(lambda: next((m for _, m in b.evs("live_avatar") if m.get("on")), None), 40)
    if not msg:
        print(f"[{mode}] no live_avatar message (is LiveKit configured?)", flush=True)
        await b.leave()
        return None
    vw = Viewer(msg["identity"])
    await vw.join(msg["url"], msg["token"])
    await until(lambda: any(m.get("status") == "listening" for _, m in b.evs("ai_state")), 30)
    await asyncio.sleep(3.0)
    rows = []
    for q in QUESTIONS:
        await b.wait_ai_quiet(1.0, 20)
        await until(lambda: quiet_from(vw, b.now() - 1.5, 1.0) is not None, 10)
        t_end = await b.say(bots.clip(q))
        push = await b.wait_ai(12, since=t_end)
        await asyncio.sleep(4.0)
        face = heard(vw, t_end)
        rows.append((q, None if push is None else round(push - t_end, 2), None if face is None else round(face - t_end, 2)))
        print(f"[{mode}] {q!r}: sent +{rows[-1][1]} s, face heard +{rows[-1][2]} s", flush=True)
        await b.wait_ai_quiet(1.0, 25)
    # interruption: cut in once the face has talked for ~1.5 s
    await until(lambda: quiet_from(vw, b.now() - 1.5, 1.0) is not None, 15)
    t_end = await b.say(bots.clip(LONG))
    face = await until(lambda: heard(vw, t_end), 15)
    cut = None
    if face:
        await asyncio.sleep(1.5)
        t_cut = b.now()
        b.say(bots.clip(CUT))
        flush = await until(lambda: next((t for t, _ in b.evs("ai_flush", t_cut)), None), 6)
        await asyncio.sleep(4.0)
        silent = quiet_from(vw, t_cut)
        stopped = next((m for _, m in b.evs("ai_state", t_cut) if m.get("stopped")), {})
        cut = {"flush_s": None if flush is None else round(flush - t_cut, 2),
               "face_quiet_s": None if silent is None else round(silent - t_cut, 2), "heard": stopped.get("heard")}
        print(f"[{mode}] interrupted: Meet stopped Higgs +{cut['flush_s']} s (pages mute then); the face's own audio "
              f"quiet +{cut['face_quiet_s']} s; remembered as heard: {cut['heard']!r}", flush=True)
    else:
        print(f"[{mode}] the long answer was never heard", flush=True)
    await b.leave()
    await vw.room.disconnect()
    await asyncio.sleep(2.0)
    log = [json.loads(x) for x in open(os.path.join(MEET, "out", "rooms", f"{rid}.jsonl"))]
    lv = [(x["on"], x.get("start_s") or x.get("why")) for x in log if x["type"] == "live_avatar"]
    errs = [x.get("error") for x in log if x["type"] in ("ai_error", "agent_error")]
    timings = [x.get("timing") for x in log if x["type"] == "ai_state" and x.get("timing")]
    return {"mode": mode, "rows": rows, "cut": cut, "live": lv, "errors": errs[:5], "timings": timings}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["improved", "default"])
    ap.add_argument("--port", type=int, default=8693)
    a = ap.parse_args()
    for q in QUESTIONS + [LONG, CUT]:
        bots.clip(q)
    srv, base = start_server(a.port)
    out = []
    try:
        for mode in a.modes:
            out.append(await run_mode(base, mode))
    finally:
        srv.kill()
    print()
    for r in filter(None, out):
        sent = [x[1] for x in r["rows"] if x[1] is not None]
        face = [x[2] for x in r["rows"] if x[2] is not None]
        print(f"{r['mode']}: answered {len(face)}/{len(r['rows'])}; reply sent median "
              f"{statistics.median(sent) if sent else None} s, face heard median {statistics.median(face) if face else None} s "
              f"after the question (all: {face}); live sessions {r['live']}; errors {r['errors']}")
        if r["timings"]:
            print("  improved timing (s after speech end):", [{k: t.get(k) for k in ("decided", "play", "live_lag")} for t in r["timings"]])
    os.makedirs(os.path.join(MEET, "out"), exist_ok=True)
    json.dump(out, open(os.path.join(MEET, "out", "live_e2e.json"), "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
