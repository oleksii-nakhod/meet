"""Turn the independently written test meetings (meetings_[ABC].jsonl) into decision points in the same format as
meeting.records(): speaker-labelled context (incl. the AI's earlier replies) + the current utterance + gold.
Split (fixed before any policy was tuned on them): A = dev (prompt/threshold design), B + C = held-out test.
D (dev2, written on the night of 2026-10-22 after the first test run) is a second dev set rich in corrections.
-> points_dev.jsonl, points_test.jsonl"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def points(m):
    history, out = [], []
    for i, d in enumerate(m["lines"]):
        out.append({"meeting": m["meeting_id"], "line": i, "target_speaker": m["ai_name"], "facts": m["facts"],
                    "people": m["people"], "setting": m["setting"], "context_turns": list(history),
                    "current_turn": {"speaker": d["speaker"], "text": d["text"]}, "gold": d["label"],
                    "category": d["category"]})
        history.append({"speaker": d["speaker"], "text": d["text"]})
        if d["label"] == "SPEAK" and d.get("ai_reply"):
            history.append({"speaker": m["ai_name"], "text": d["ai_reply"]})
    return out


if __name__ == "__main__":
    splits = [("dev", "A"), ("test", "BC")]
    if os.path.exists(os.path.join(HERE, "meetings_D.jsonl")):
        splits.append(("dev2", "D"))  # written later, rich in corrections, to tune v3 without touching test
    for split, files in splits:
        rows = [p for f in files for line in open(os.path.join(HERE, f"meetings_{f}.jsonl")) for p in points(json.loads(line))]
        with open(os.path.join(HERE, f"points_{split}.jsonl"), "w") as fo:
            for r in rows:
                fo.write(json.dumps(r) + "\n")
        print(split, len(rows), "points,", sum(r["gold"] == "SPEAK" for r in rows), "SPEAK")
