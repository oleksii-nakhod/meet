"""Keep the judge's fact sheet short when the AI knows a lot: pick the k facts with the most word overlap with the
latest utterance and the two before it (lowercased content words, crude suffix stripping). No API calls.
Usage: python retrieve.py test_bigfacts 8  -> testset/points_test_bigfacts_top8.jsonl"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STOP = set("""a an the and or but if so to of in on at for with from by as is are was were be been it its it's this that
these those i you he she we they me him her us them my your our their do does did not no yes okay ok um uh like just
yeah right well oh what when where who how why which can could would should will about there here then than""".split())


def toks(s):
    return {re.sub(r"(ing|ed|es|s)$", "", w) for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in STOP and len(w) > 1}


def top_facts(r, k):
    q = toks(" ".join([t["text"] for t in r["context_turns"][-2:]] + [r["current_turn"]["text"]]))
    scored = sorted(enumerate(r["facts"]), key=lambda x: (-len(q & toks(x[1])), x[0]))
    return [f for _, f in scored[:k]]


if __name__ == "__main__":
    split, k = sys.argv[1], int(sys.argv[2])
    rows = [json.loads(line) for line in open(os.path.join(HERE, "testset", f"points_{split}.jsonl"))]
    with open(os.path.join(HERE, "testset", f"points_{split}_top{k}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({**r, "facts": top_facts(r, k)}) + "\n")
    print(split, len(rows), "rows, top", k, "facts each")
