"""How chatty is the speak/silent judge in a *real* meeting? AMI scenario meetings (four people designing a TV
remote; manual word-level transcripts) with an AI participant "Nova" that knows the project brief everyone in
AMI was given. There is no gold label: we measure how often the judge would speak and read every SPEAK it makes.

-> testset/points_ami_<meeting>.jsonl (gold "SILENT" is a placeholder), then
   python tv_eval.py --sets ami_<meeting> --policies or:vote:higgs_dec2x3+all:higgs_factx3 ; python ami_chatty.py --show <meeting>
Usage: python ami_chatty.py --build ES2009c [--minutes 20]"""
import argparse
import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mmhm", "eval"))
sys.path.insert(0, os.path.join(HERE, "..", "mmhm"))
FACTS = ["The team is designing a new TV remote control for the company Real Reaction.",
         "The remote will sell for 25 euros.", "Production cost must be at most 12.50 euros per remote.",
         "The profit target is 50 million euros, selling internationally.",
         "Management decided teletext is outdated and the remote should only control the TV.",
         "The company's colour and slogan ('We put the fashion in electronics') must be recognisable on the remote.",
         "The meeting has four roles: project manager, marketing expert, user interface designer, industrial designer."]


def build(meeting, minutes):
    from ami_stimuli import ipus, words
    segs = []
    for spk in "ABCD":
        try:
            ws = words(meeting, spk)
        except FileNotFoundError:
            continue
        for u in ipus(ws, gap=0.6):
            text = " ".join(w["w"] + w.get("punc", "") for w in u["words"]).replace(" ,", ",").strip()
            if len(u["words"]) >= 2 and u["s"] < minutes * 60:
                segs.append({"s": u["s"], "e": u["e"], "speaker": spk, "text": text})
    segs.sort(key=lambda x: x["s"])
    rows, history = [], []
    for i, sg in enumerate(segs):
        rows.append({"meeting": meeting, "line": i, "target_speaker": "Nova", "facts": FACTS, "people": list("ABCD"),
                     "setting": "a product design team meeting", "context_turns": list(history),
                     "current_turn": {"speaker": sg["speaker"], "text": sg["text"]}, "gold": "SILENT", "category": "real",
                     "t": round(sg["s"], 1), "t_end": round(sg["e"], 2)})
        history.append({"speaker": sg["speaker"], "text": sg["text"]})
    path = os.path.join(HERE, "testset", f"points_ami_{meeting}.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(meeting, len(rows), "segments in", minutes, "min ->", path)


async def show(meeting):
    import tv_eval
    rows = tv_eval.load(f"ami_{meeting}")
    preds = []
    for part in ("higgs_dec2", "higgs_dec2@1", "higgs_dec2@2"):
        preds.append((await tv_eval.run_higgs(part, rows, 6))[0])
    facts = [(await tv_eval.run_higgs(p, rows, 6))[0] for p in ("higgs_fact", "higgs_fact@1", "higgs_fact@2")]
    n = 0
    for r in rows:
        k = tv_eval.key(r)
        judge, fact = sum(p[k] for p in preds) >= 2, all(p[k] for p in facts)
        if judge or fact:
            n += 1
            ctx = " / ".join(f"{t['speaker']}: {t['text']}" for t in r["context_turns"][-2:])
            print(f"[{r['t'] / 60:5.1f} min] {'fact' if fact and not judge else 'judge'} | ...{ctx[-160:]}\n"
                  f"          >> {r['current_turn']['speaker']}: {r['current_turn']['text']}")
    mins = max(r["t"] for r in rows) / 60
    print(f"{meeting}: speaks at {n} of {len(rows)} segments ({n / mins:.2f} per minute over {mins:.0f} min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build")
    ap.add_argument("--show")
    ap.add_argument("--minutes", type=float, default=20)
    a = ap.parse_args()
    if a.build:
        build(a.build, a.minutes)
    if a.show:
        asyncio.run(show(a.show))
