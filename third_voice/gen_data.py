"""Synthetic in-domain training data for the speak/silent policy: meetings where an AI participant listens in.
Higgs Realtime (text mode) writes each transcript with labels; we turn them into decision points in the same
format as meeting.records(). The AI's name is drawn from a list that excludes "Higgs", so the hand-written test
meeting (meeting.py, AI = "Higgs") stays out of distribution.

Usage: python gen_data.py [--n 200] [--par 6]  -> out/synth_meetings.jsonl, out/synth_points.jsonl"""
import argparse
import asyncio
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mmhm"))
from mmhm.higgs import Session  # noqa: E402

SETTINGS = ["software team stand-up", "family planning a summer road trip", "university study group before an exam",
            "startup founders preparing a pitch", "hospital nurses at shift handover", "wedding planning call",
            "podcast hosts recording an episode", "friends planning a birthday party", "landlord and tenants meeting",
            "high-school robotics club", "restaurant kitchen before service", "book club discussion",
            "marketing team brainstorm", "roommates splitting chores", "hackathon team in the last hours",
            "film crew planning a shoot", "parents at a school council meeting", "research lab weekly meeting",
            "band rehearsal break", "volunteers organizing a food drive"]
AI_NAMES = ["Nova", "Juno", "Atlas", "Echo", "Iris", "Orbit", "Pixel", "Sage", "Kai", "Ember", "Remy", "Tess"]
PEOPLE = ["Ana", "Ben", "Chen", "Dara", "Eli", "Fatima", "Gus", "Hana", "Ivan", "Jade", "Kofi", "Lena", "Marco",
          "Nia", "Omar", "Priya", "Quinn", "Raj", "Sofia", "Tom", "Uma", "Viktor", "Wen", "Yusuf", "Zoe"]

SPEC = """Write a realistic spoken transcript of a {setting} with {people} and an AI assistant called {ai} that listens in and can talk.
{n} lines. Spoken style: fillers, fragments, people trailing off mid-sentence, short replies.
Mix these situations naturally:
- people talking to each other by name (the AI must stay silent),
- someone thinking aloud or stopping mid-sentence (silent),
- someone addressing {ai} by name with a request or question (the AI should speak),
- someone asking the whole group a factual question nobody in the room answers, which {ai} could answer (the AI should speak),
- someone mentioning {ai} in passing without asking it anything (silent),
- small talk and jokes between people (silent).
The AI speaks after about one line in five. Output ONLY JSON lines, one per human line, like:
{{"speaker": "<name>", "text": "<what they said>", "ai_speaks": true or false, "ai_reply": "<what {ai} says, only if ai_speaks>"}}"""


async def one(i, rng):
    setting = rng.choice(SETTINGS)
    people = rng.sample(PEOPLE, rng.choice([2, 3, 3, 4]))
    ai = rng.choice(AI_NAMES)
    n = rng.randint(14, 22)
    prompt = SPEC.format(setting=setting, people=", ".join(people), ai=ai, n=n)
    # Realtime answers in short conversational turns, so the transcript is written one line per request.
    lines = []
    async with Session({"instructions": "You write realistic dialogue transcripts, one JSON line per request. " + prompt,
                        "output_modalities": ["text"]}) as s:
        for k in range(n):
            await s.say_text(f"Line {k + 1} of {n}. Output only that one JSON object.")
            _, ev = await s.wait(lambda e: e.get("type") == "response.done", timeout=60)
            raw = s.transcript(ev["response"]["id"]).strip()
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                continue
            try:
                d = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict) and d.get("speaker") and d.get("text") and d["speaker"] != ai:
                lines.append(d)
    return {"id": i, "setting": setting, "people": people, "ai": ai, "lines": lines}


def points(m):
    history, out = [], []
    for j, d in enumerate(m["lines"]):
        if d["speaker"] == m["ai"]:
            continue
        speak = bool(d.get("ai_speaks")) and bool(d.get("ai_reply"))
        out.append({"meeting": m["id"], "turn": j, "target_speaker": m["ai"], "context_turns": list(history),
                    "current_turn": {"speaker": d["speaker"], "text": d["text"]},
                    "decision": "SPEAK" if speak else "SILENT"})
        history.append({"speaker": d["speaker"], "text": d["text"]})
        if speak:
            history.append({"speaker": m["ai"], "text": d["ai_reply"]})
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--par", type=int, default=6)
    a = ap.parse_args()
    rng = random.Random(0)
    seeds = [random.Random(rng.random()) for _ in range(a.n)]
    sem = asyncio.Semaphore(a.par)
    meetings = []

    async def run(i):
        async with sem:
            try:
                m = await one(i, seeds[i])
            except Exception as e:
                print("fail", i, e, flush=True)
                return
            if len(m["lines"]) >= 8:
                meetings.append(m)
            if len(meetings) % 20 == 0:
                print(len(meetings), "meetings", flush=True)

    await asyncio.gather(*[run(i) for i in range(a.n)])
    meetings.sort(key=lambda m: m["id"])
    with open(os.path.join(HERE, "out", "synth_meetings.jsonl"), "w") as f:
        for m in meetings:
            f.write(json.dumps(m) + "\n")
    pts = [p for m in meetings for p in points(m)]
    with open(os.path.join(HERE, "out", "synth_points.jsonl"), "w") as f:
        for p in pts:
            f.write(json.dumps(p) + "\n")
    sp = sum(p["decision"] == "SPEAK" for p in pts)
    print(f"{len(meetings)} meetings, {len(pts)} decision points, SPEAK {sp} ({sp / max(1, len(pts)):.0%})")


if __name__ == "__main__":
    asyncio.run(main())
