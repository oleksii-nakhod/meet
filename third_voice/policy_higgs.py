"""Baselines for the speak/silent decision on the scripted meeting:
- zero-shot Higgs Realtime (text mode) asked to answer SPEAK or SILENT with the same transcript,
- a heuristic: speak after any question, or whenever "Higgs" is said.
Writes out/decisions_baselines.jsonl."""
import asyncio
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mmhm"))
from mmhm.higgs import Session  # noqa: E402

from meeting import FACTS  # noqa: E402

JUDGE = ("You are Higgs, an AI teammate sitting in on a hackathon team meeting with Ana, Ben and Chen. "
         + FACTS + " After each utterance there is a short pause and you must decide whether to speak now. Speak "
         "only when someone addresses you, or when the group needs something you know and nobody else can answer. "
         "Stay silent when people talk to each other or are mid-thought. Answer with exactly one word: SPEAK or SILENT.")


def transcript(r):
    ctx = "\n".join(f"{t['speaker']}: {t['text']}" for t in r["context_turns"][-12:]) or "(meeting just started)"
    return f"Transcript so far:\n{ctx}\n\nJust said (now there's a pause):\n{r['current_turn']['speaker']}: {r['current_turn']['text']}\n\nSPEAK or SILENT?"


async def higgs_decision(r):
    async with Session({"instructions": JUDGE, "output_modalities": ["text"]}) as s:
        await s.say_text(transcript(r))
        _, ev = await s.wait("response.done", timeout=30)
        text = s.transcript(ev["response"]["id"]).strip().upper()
        return "SPEAK" if "SPEAK" in text else "SILENT", text


async def main():
    rows = [json.loads(line) for line in open(os.path.join(HERE, "out", "decision_points.jsonl"))]
    out = []
    for r in rows:
        d, raw = await higgs_decision(r)
        cur = r["current_turn"]["text"]
        heur = "SPEAK" if ("?" in cur or re.search(r"\bhiggs\b", cur, re.I)) else "SILENT"
        out.append({"id": r["id"], "gold": r["gold"], "higgs_zero_shot": d, "higgs_raw": raw[:40], "heuristic": heur})
        print(out[-1], flush=True)
    with open(os.path.join(HERE, "out", "decisions_baselines.jsonl"), "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
