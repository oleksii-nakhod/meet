"""Interactive scenarios against the real Meet server + Higgs agent, with bots (tests/bots.py) instead of browsers.

Each scenario is a room (people, facts) and a list of steps with what should happen:
  ("say", who, speech, expect[, opts])   speech: text, or a list of text / pause seconds / {"ami": file} pieces.
                                         expect: "reply" (Higgs should answer), "silent", "any".
                                         opts: {"window": s, "gap": s (start before Higgs is quiet / after the previous
                                         line), "gain_db": x}
  ("talk_over", who, speech, then, expect[, opts])
                                         `who` says `speech`, which should make Higgs talk; `at` s into Higgs's audio,
                                         then = (who2, speech2) is played. expect: "continue" (Higgs keeps talking,
                                         e.g. through a "mm-hm") or "stop" (a real interruption).
  ("chat", who, text, expect)            a typed line.
  ("wait", seconds)
Every step is scored from what a participant hears (Higgs's slot-0 audio) plus the room's events, so a failure says
where it came from: what Higgs's STT heard, the judge's reading, a floor drop ("stale", "newer line"), or a stop.

Usage: python tests/scenarios.py [--modes improved default] [--only greet_1to1 ...] [--par 4] [--tag NAME]
-> out/scen/<tag>/<scenario>_<mode>.json + summary.json; prints a table."""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEET = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bots  # noqa: E402
from bots import SR, Bot, ami, clip, create_room  # noqa: E402

ADMIN = "scen-admin"
VOICES = {"Alex": "marcus", "Priya": "nora", "Sam": "jake", "Jordan": "oliver", "Mei": "eleanor"}
HACK = ["The demo video must be 2 to 3 minutes and show the project running.", "Submissions close Saturday at 3 pm.",
        "Judging: use of Higgs 30%, user experience 25%, innovation 25%, technical execution 20%.",
        "The team's project is a video call where Higgs joins as a participant."]
SET = "hackathon team meeting"
LONG_Q = "Higgs, can you explain in a few sentences how we should structure the demo video?"

SCENARIOS = [
    # ---------------- one person and Higgs ----------------
    dict(name="greet_1to1", people=["Alex"], steps=[
        ("say", "Alex", "Hey, Higgs.", "reply"),
        ("say", "Alex", "How's it going?", "reply"),
        ("say", "Alex", "Can you hear me okay?", "reply")]),
    dict(name="questions_1to1", people=["Alex"], steps=[
        ("say", "Alex", "When do submissions close?", "reply"),
        ("say", "Alex", "And how long can the video be?", "reply"),
        ("say", "Alex", "What's the capital of Australia?", "reply")]),
    dict(name="followups_1to1", people=["Alex"], steps=[
        ("say", "Alex", "Higgs, give me one idea for a demo opening.", "reply"),
        ("say", "Alex", "Anything else?", "reply"),
        ("say", "Alex", "Go on.", "reply"),
        ("say", "Alex", "Tell me more about that.", "reply"),
        ("say", "Alex", "Okay, thanks.", "any")]),
    dict(name="statements_1to1", people=["Alex"], steps=[
        ("say", "Alex", "I think we should open the video with the problem, not the product.", "reply"),
        ("say", "Alex", "I'm a bit worried we won't finish the avatar part in time.", "reply"),
        ("say", "Alex", "Give me your thoughts.", "reply")]),
    dict(name="pause_1to1", people=["Alex"], steps=[
        ("say", "Alex", ["So I was thinking that maybe we could", 1.3, "start the video with a live call."], "reply"),
        ("say", "Alex", ["What do you think about", 1.0, "using the avatar in the intro?"], "reply")]),
    dict(name="fillers_1to1", people=["Alex"], steps=[
        ("say", "Alex", "Um.", "silent"),
        ("say", "Alex", "Hmm, where did I put that file.", "silent"),
        ("say", "Alex", "Higgs, are you there?", "reply")]),
    dict(name="bc_1to1", people=["Alex"], steps=[
        ("talk_over", "Alex", LONG_Q, ("Alex", {"ami": "bc_00.wav"}), "continue"),
        ("talk_over", "Alex", "Higgs, and what should the last thirty seconds show?", ("Alex", "Mm-hm."), "continue")]),
    dict(name="bc2_1to1", people=["Alex"], steps=[
        ("talk_over", "Alex", "Higgs, how should we split the work between us this week?", ("Alex", {"ami": "bc_03.wav"}), "continue"),
        ("talk_over", "Alex", "Higgs, what's the riskiest part of our plan?", ("Alex", "Yeah."), "continue")]),
    dict(name="barge_1to1", people=["Alex"], steps=[
        ("talk_over", "Alex", LONG_Q, ("Alex", "Wait, sorry, can I stop you there? I have a question."), "stop"),
        ("say", "Alex", "Should the intro be shorter than twenty seconds?", "reply")]),
    dict(name="chat_1to1", people=["Alex"], steps=[
        ("chat", "Alex", "Higgs, when do submissions close?", "reply"),
        ("say", "Alex", "Thanks. And what do the judges weigh the most?", "reply")]),
    # ---------------- groups ----------------
    dict(name="humans_talk", people=["Alex", "Priya", "Sam"], steps=[
        ("say", "Alex", "Priya, did you push the new layout?", "silent"),
        ("say", "Priya", "Yeah, I pushed it this morning. Sam, can you check it on your phone?", "silent"),
        ("say", "Sam", "Sure, I'll look at it after lunch.", "silent"),
        ("say", "Alex", "Cool. I like the new colors.", "silent")]),
    dict(name="group_fact", people=["Alex", "Priya", "Sam"], steps=[
        ("say", "Alex", "Wait, when do submissions close again?", "reply"),
        ("say", "Priya", "Does anyone remember how long the video can be?", "reply"),
        ("say", "Sam", "Okay, so we have time.", "silent")]),
    dict(name="group_address", people=["Alex", "Priya"], steps=[
        ("say", "Alex", "Higgs, what do you think we should call the project?", "reply"),
        ("say", "Priya", "Hey Higgs, quick one: how much is the user experience worth in judging?", "reply"),
        ("say", "Alex", "Thanks, Higgs.", "any")]),
    dict(name="group_wrong", people=["Alex", "Priya", "Sam"], steps=[
        ("say", "Alex", "So submissions close Sunday at noon, right? We have the whole weekend.", "reply"),
        ("say", "Priya", "And the video can be up to five minutes, so let's show everything.", "reply")]),
    dict(name="group_answered", people=["Alex", "Priya"], steps=[
        ("say", "Alex", "Priya, when's your flight tomorrow?", "silent"),
        ("say", "Priya", "At nine in the morning, so I'll join from the airport.", "silent"),
        ("say", "Alex", "Okay, Higgs, can you remind us what the judges care about?", "reply")]),
    dict(name="group_bc", people=["Alex", "Priya"], steps=[
        ("talk_over", "Alex", LONG_Q, ("Priya", {"ami": "bc_05.wav"}), "continue"),
        ("talk_over", "Priya", "Higgs, what's the one thing we must not forget?", ("Alex", "Uh-huh."), "continue")]),
    dict(name="group_barge", people=["Alex", "Priya"], steps=[
        ("talk_over", "Alex", LONG_Q, ("Priya", "Sorry Higgs, hold on, Alex, we already decided that yesterday."), "stop"),
        ("say", "Alex", "Right, sorry, I forgot.", "silent")]),
    dict(name="group_fast", people=["Alex", "Priya", "Sam"], steps=[
        ("say", "Alex", "What do we still need for the demo?", "any"),
        ("say", "Priya", "The avatar shot and the voice-over.", "silent", {"gap": 0.4}),
        ("say", "Sam", "I can record the voice-over tonight.", "silent", {"gap": 0.4}),
        ("say", "Alex", "Great, and Higgs, how long should the voice-over be?", "reply", {"gap": 0.4})]),
    dict(name="group_pause", people=["Alex", "Priya"], steps=[
        ("say", "Alex", ["Higgs, I was wondering whether", 1.3, "we should show the switch between the two modes."], "reply"),
        ("say", "Priya", ["I think the judges will", 1.0, "want to see it running live."], "silent")]),
]


def speech(spec, voice):
    """A text / [text, pause, {"ami": f}, ...] spec -> int16 pcm."""
    parts = spec if isinstance(spec, list) else [spec]
    out = []
    for p in parts:
        if isinstance(p, (int, float)):
            out.append(np.zeros(int(p * SR), np.int16))
        elif isinstance(p, dict):
            out.append(ami(p["ami"], p.get("gain_db", 0.0)))
        else:
            out.append(clip(p, voice))
    return np.concatenate(out)


def text_of(spec):
    parts = spec if isinstance(spec, list) else [spec]
    return " ".join(f"[{p}s]" if isinstance(p, (int, float)) else (f"<{p['ami']}>" if isinstance(p, dict) else p)
                    for p in parts)


def window_events(b, t0, t1):
    ev = [(t, m) for t, m in b.events if t0 <= t <= t1]
    lines = [{"name": m.get("name"), "text": m.get("text"), "speak": (m.get("decision") or {}).get("speak"),
              "why": (m.get("decision") or {}).get("why"), "reading": (m.get("decision") or {}).get("reading"),
              "decided_s": (m.get("decision") or {}).get("decided_s")} for t, m in ev if m.get("type") == "line"]
    says = [m.get("text") for t, m in ev if m.get("type") == "ai_say"]
    states = [{k: m.get(k) for k in ("status", "dropped", "stopped", "after_s", "why", "heard", "timing") if m.get(k) is not None}
              for t, m in ev if m.get("type") == "ai_state" and (m.get("dropped") or m.get("stopped") or m.get("after_s"))]
    flushes = [round(t, 2) for t, m in ev if m.get("type") == "ai_flush"]
    errors = [m.get("error") for t, m in ev if m.get("type") == "ai_error"]
    errors += [f"skipped: {m.get('why')}" for t, m in ev if m.get("type") == "reply_skipped"]
    return {"lines": lines, "ai_say": says, "states": states, "flush": flushes, "errors": errors}


async def wait_quiet(b, quiet=0.8, timeout=25):
    await b.wait_ai_quiet(quiet, timeout)


async def run_step(step, bots_, ear, mode):
    kind = step[0]
    row = {"kind": kind}
    if kind == "wait":
        await asyncio.sleep(step[1])
        return None
    if kind == "say":
        _, who, sp, expect, *rest = step
        opts = rest[0] if rest else {}
        if "gap" in opts:
            await asyncio.sleep(opts["gap"])
        else:
            await wait_quiet(ear, 1.0)
            await asyncio.sleep(0.3)
        b = bots_[who]
        pcm = speech(sp, VOICES[who])
        t_start = b.now()
        t_end = await b.say(pcm)
        win = opts.get("window", 9.0 if expect == "reply" else 5.0)
        on = ear.ai_onsets(t_start)
        if not on and expect != "silent":
            await ear.wait_ai(win, since=t_start)
        elif not on:
            await asyncio.sleep(win)
        on = [o for o in ear.ai_onsets(t_start) if o[0] <= t_end + win]
        cut_in = [o for o in on if o[0] < t_end - 0.1]
        got = "reply" if on else "silent"
        ok = expect == "any" or got == expect
        if cut_in:
            ok = False
            got = "talked over"
        row.update(who=who, said=text_of(sp), expect=expect, got=got, ok=ok,
                   latency=round(on[0][0] - t_end, 2) if on else None)
        if on:
            await wait_quiet(ear, 1.0)
        await asyncio.sleep(0.5)  # late events (a decision) belong to this step
        row.update(window_events(ear, t_start, ear.now()))
        return row
    if kind == "chat":
        _, who, text, expect = step
        await wait_quiet(ear, 1.0)
        b = bots_[who]
        t_start = b.now()
        await b.json({"type": "chat", "text": text})
        await ear.wait_ai(9.0, since=t_start)
        on = ear.ai_onsets(t_start)
        got = "reply" if on else "silent"
        row.update(who=who, said=f"(chat) {text}", expect=expect, got=got, ok=expect == "any" or got == expect,
                   latency=round(on[0][0] - t_start, 2) if on else None)
        if on:
            await wait_quiet(ear, 1.0)
        await asyncio.sleep(0.5)
        row.update(window_events(ear, t_start, ear.now()))
        return row
    if kind == "talk_over":
        _, who, sp, (who2, sp2), expect, *rest = step
        opts = rest[0] if rest else {}
        at = opts.get("at", 1.2)
        await wait_quiet(ear, 1.0)
        await asyncio.sleep(0.3)
        b, b2 = bots_[who], bots_[who2]
        pcm, pcm2 = speech(sp, VOICES[who]), speech(sp2, VOICES[who2])
        t_start = b.now()
        t_end = await b.say(pcm)
        on = await ear.wait_ai(10.0, since=t_start)
        row.update(who=who, said=text_of(sp), then_who=who2, then=text_of(sp2), expect=expect)
        if on is None:
            row.update(got="no reply to talk over", ok=None, latency=None)
            await asyncio.sleep(0.5)
            row.update(window_events(ear, t_start, ear.now()))
            return row
        row["latency"] = round(on - t_end, 2)
        while ear.now() < on + at:
            await asyncio.sleep(0.01)
        if not ear.ai_speaking(0.3):
            row.update(got="reply too short to talk over", ok=None)
            await wait_quiet(ear, 1.0)
            row.update(window_events(ear, t_start, ear.now()))
            return row
        t_c = b2.now()
        t_c_end = await b2.say(pcm2)
        await asyncio.sleep(2.5)
        # stopped = a flush after the cue started, or Higgs's first reply ended within 1.5 s of the cue's end while a
        # stop was announced
        ev = window_events(ear, t_c, ear.now())
        stopped = bool(ev["flush"]) or any(s.get("stopped") for s in ev["states"])
        spans = ear.ai_onsets(on)
        first_end = spans[0][1] if spans else on
        # Higgs answering the cue as a new turn (plain Higgs does): audio that starts after the stop
        answered_cue = [round(a - t_c_end, 2) for a, _ in spans[1:] if a > t_c]
        got = "stopped" if stopped else "continued"
        row.update(got=got, ok=(got == "stopped") == (expect == "stop"), cue_at=round(t_c - on, 2),
                   stop_latency=round(ev["flush"][0] - t_c, 2) if ev["flush"] else None,
                   first_reply_len=round(first_end - on, 2), answered_cue_after=answered_cue[:1])
        await wait_quiet(ear, 1.2, 30)
        await asyncio.sleep(0.5)
        row.update(window_events(ear, t_start, ear.now()))
        return row
    raise ValueError(kind)


async def run_scenario(base, sc, mode, outdir):
    people = sc["people"]
    rid = create_room(base, ADMIN, title=sc["name"], ai=sc.get("ai", "Higgs"), setting=sc.get("setting", SET),
                      facts=sc.get("facts", HACK))
    bs = {p: Bot(base, rid, p, echo=sc.get("echo", 0.0)) for p in people}
    for b in bs.values():
        await b.join()
        await asyncio.sleep(0.2)
    ear = bs[people[0]]  # what one participant hears
    await asyncio.sleep(1.0)
    if mode != "default":
        await ear.json({"type": "set_mode", "mode": mode})
    if sc.get("avatar"):
        await ear.json({"type": "set_avatar", "on": True})
    await asyncio.sleep(sc.get("settle", 6.0))  # sessions open, turn-taking models load
    rows = []
    t0 = time.time()
    for step in sc["steps"]:
        try:
            r = await run_step(step, bs, ear, mode)
        except Exception as e:  # noqa: BLE001
            r = {"kind": step[0], "said": str(step[2])[:80], "ok": False, "got": f"harness error {e!r}"}
        if r:
            rows.append(r)
    ear.save_ai(os.path.join(outdir, f"{sc['name']}_{mode}_ai.wav"))
    for b in bs.values():
        await b.leave()
    res = {"scenario": sc["name"], "mode": mode, "room": rid, "secs": round(time.time() - t0, 1), "rows": rows,
           "ok": sum(r.get("ok") is True for r in rows), "scored": sum(r.get("ok") is not None for r in rows)}
    json.dump(res, open(os.path.join(outdir, f"{sc['name']}_{mode}.json"), "w"), indent=1)
    return res


def start_server(port, extra_env=None):
    env = {**os.environ, "MEET_ADMIN": ADMIN, "MEET_BUDGET": "100000", **(extra_env or {})}
    env.pop("MEET_AGENT", None)
    log = open(os.path.join(MEET, "out", f"scen_server_{port}.log"), "w")
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


def short(r):
    s = f"{'OK ' if r.get('ok') else ('-- ' if r.get('ok') is None else 'XX ')}{r.get('said', '')[:60]:60s}"
    if r["kind"] == "talk_over":
        s += f" | {r.get('then', '')[:30]:30s} want {r.get('expect')}, got {r.get('got')}"
    else:
        s += f" | want {r.get('expect')}, got {r.get('got')}"
    if r.get("latency") is not None:
        s += f" ({r['latency']}s)"
    heard = [f"{x['name']}: {x['text']!r}->{'S' if x['speak'] else 's'}({x['why']})" for x in r.get("lines", [])]
    if heard:
        s += "\n      heard " + "; ".join(heard)
    for st in r.get("states", []):
        if st.get("dropped") or st.get("stopped"):
            s += f"\n      {st}"
    for e in r.get("errors", []):
        s += f"\n      !! {e}"
    return s


async def main_async(a):
    outdir = os.path.join(MEET, "out", "scen", a.tag)
    os.makedirs(outdir, exist_ok=True)
    todo = [(sc, m) for sc in SCENARIOS if not a.only or sc["name"] in a.only for m in a.modes
            if m in sc.get("modes", ("default", "improved"))]
    # warm the TTS cache first (parallel, outside the timed part)
    texts = set()
    for sc, _ in todo:
        for st in sc["steps"]:
            specs = []
            if st[0] == "say":
                specs = [(st[1], st[2])]
            elif st[0] == "talk_over":
                specs = [(st[1], st[2]), st[3]]
            for who, sp in specs:
                for p in (sp if isinstance(sp, list) else [sp]):
                    if isinstance(p, str):
                        texts.add((p, VOICES[who]))
    await asyncio.gather(*[asyncio.to_thread(clip, t, v) for t, v in texts])
    servers = [start_server(a.port + i, {"MEET_SAMPLES": str(a.samples)}) for i in range(a.par)]
    q = asyncio.Queue()
    for item in todo:
        q.put_nowait(item)
    results = []

    async def worker(base):
        while not q.empty():
            sc, m = q.get_nowait()
            r = await run_scenario(base, sc, m, outdir)
            results.append(r)
            print(f"\n== {sc['name']} [{m}] {r['ok']}/{r['scored']}", flush=True)
            for row in r["rows"]:
                print("   " + short(row), flush=True)

    try:
        await asyncio.gather(*[worker(base) for _, base in servers])
    finally:
        for p, _ in servers:
            p.terminate()
    summ = {}
    for r in results:
        s = summ.setdefault(r["mode"], {"ok": 0, "scored": 0})
        s["ok"] += r["ok"]
        s["scored"] += r["scored"]
    json.dump({"summary": summ, "results": sorted(results, key=lambda r: (r["scenario"], r["mode"]))},
              open(os.path.join(outdir, "summary.json"), "w"), indent=1)
    print("\nSUMMARY", json.dumps(summ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["improved", "default"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--port", type=int, default=8641)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--tag", default=time.strftime("%m%d_%H%M"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
