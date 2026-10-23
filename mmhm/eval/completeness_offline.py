"""Can Higgs (text mode) tell a finished utterance from one that stopped mid-sentence, from the STT transcript alone?
Data: the first input transcription after each item's start in the AMI runs (holds: words up to the thinking
pause; completes: the whole turn). Compares Higgs's judgement with the word-list rule in mmhm.engine.unfinished."""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mmhm.engine import unfinished  # noqa: E402
from mmhm.higgs import Session, outdir  # noqa: E402

OUT = outdir("turns")
JUDGE = ("You judge transcripts of live speech. The speaker has just paused. Decide whether what they said so far is a "
         "finished thought, or whether they stopped in the middle of a sentence and will probably continue. "
         "Answer with one word: FINISHED or UNFINISHED.")


def collect(set_tag):
    rows = [json.loads(line) for line in open(f"{OUT}/run_{set_tag}_mmhm.log") if line.startswith('{"cond"')]
    log = [json.loads(line) for line in open(f"{OUT}/log_mmhm_{set_tag}.jsonl")]
    out = []
    for r in rows:
        t0 = r["t_a_end"] - 0.2 if r["kind"] == "hold" else r["t_end"] - 0.2
        txt = next((x["text"] for x in log if x["kind"] == "user_text" and x["t"] >= t0), None)
        if txt:
            out.append({"kind": r["kind"], "text": txt})
    return out


async def judge(items):
    async with Session({"instructions": JUDGE, "output_modalities": ["text"]}) as s:
        for it in items:
            t = time.perf_counter()
            await s.say_text(f'Transcript: "{it["text"]}"')
            _, ev = await s.wait("response.done", timeout=30)
            ans = s.transcript(ev["response"]["id"]).strip().upper()
            it["higgs"] = "UNFINISHED" if "UNFINISHED" in ans else "FINISHED"
            it["ms"] = int((time.perf_counter() - t) * 1000)
            # a fresh context per judgement: delete the two items we just added
            for item in [e for _, e in s.events if e.get("type") == "conversation.item.added"][-2:]:
                await s.send({"type": "conversation.item.delete", "item_id": item["item"]["id"]})
    return items


def score(items, key):
    h = [i for i in items if i["kind"] == "hold"]; c = [i for i in items if i["kind"] == "complete"]
    return (f"holds caught {sum(i[key] for i in h)}/{len(h)}, completes wrongly held {sum(i[key] for i in c)}/{len(c)}")


async def main():
    items = collect("ami1") + collect("ami2")
    items = await judge(items)
    for i in items:
        i["rule"] = unfinished(i["text"])
        i["llm"] = i["higgs"] == "UNFINISHED"
    print("word-list rule :", score(items, "rule"))
    print("Higgs judge    :", score(items, "llm"))
    print("either         :", score([{**i, "both": i["rule"] or i["llm"]} for i in items], "both"))
    print("median judge latency ms:", sorted(i["ms"] for i in items)[len(items) // 2])
    json.dump(items, open(f"{OUT}/completeness_offline.json", "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
