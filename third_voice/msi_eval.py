"""Third Voice on MSI-Bench (Boson AI, arXiv 2609.24812; data CC BY 4.0: huggingface.co/datasets/M2cha4l1124/MSI-Bench).

MSI-Bench scenes are 2-3 people talking near a voice assistant ("Assistant"); every human line is labelled with whom it
is addressed to, and the assistant answers only lines addressed to it. Every English human line becomes one
speak/stay-silent decision for Third Voice (gold SPEAK iff the line is addressed to the assistant):
  - the 192 English hear-time probes (dialogue cut after a human-to-human line; the paper's premature-response rate,
    PRR = how often a model answers there) are a subset of these lines;
  - the 576 final lines (the assistant's cue in every case) are another subset;
  - the AI's fact sheet = the scene date, the participants' roles and capabilities, and the case's private memory.
The assistant's earlier answers stay in the context as text (they have no audio in MSI-Bench either).

Sets: text (gold transcripts, speaker names), anon (gold text, every human "Someone"), stt (Higgs STT of the
line audio, every human "Someone" = the live app's setting; run msi_stt.py first).
Policies: keyword (speak iff the line says "assistant"), heuristic2, higgs_zs (ask SPEAK/SILENT), v4 (the live
app: majority of 3 decomposed judges OR all 3 fact-checks), cheap (1 judge OR 1 fact-check), v4b (v4 with
"ask" covering commands; designed after seeing the v4 errors on MSI-Bench, so a second look).

Usage: python msi_eval.py --sets text --policies keyword heuristic2 v4 [--only probes,finals] [--par 6]
       [--data /tmp/msi/data]  -> out/msi/report_<set>.json"""
import argparse
import asyncio
import json
import os
import random
import re

import tv_eval

HERE = os.path.dirname(os.path.abspath(__file__))
AI = "Assistant"


def facts_of(c):
    f = [f"Today is {c['scene_date']}."]
    for p in c["participants"]:
        caps = f"; can {', '.join(p['capabilities'])}" if p.get("capabilities") else ""
        f.append(f"{p['name']} is the {p['role']}{caps}.")
    if c["eval"].get("private_memory"):
        f.append(f"You privately know: {c['eval']['private_memory']}")
    return f


def rows_from(data, anon=False, transcripts=None):
    probes = {}
    for line in open(os.path.join(data, "probes", "hear_probe.jsonl")):
        h = json.loads(line)
        probes[(h["case_id"], h["n_lines"] - 1)] = h["competing_speech"]
    rows = []
    for line in open(os.path.join(data, "index", "cases.jsonl")):
        c = json.loads(line)
        if c["language"] != "en":
            continue
        T = c["turns"]

        def said(t):
            if t["role"] == "assistant":
                return {"speaker": AI, "text": t["text"]}
            text = t["text"] if transcripts is None else transcripts.get(f"{c['case_id']}/{t['audio']}", "")
            return {"speaker": "Someone" if anon else t["speaker"], "text": text}

        for i, t in enumerate(T):
            if t["role"] != "human":
                continue
            to = t["addressed_to"]
            if to == "assistant":
                cat = "to_ai_named" if re.search(r"\bassistant\b", t["text"], re.I) else "to_ai_unnamed"
            else:
                cat = "to_group" if to == "group" else "to_person"
            rows.append({"meeting": c["case_id"], "line": i, "target_speaker": AI, "facts": facts_of(c),
                         "people": [p["name"] for p in c["participants"]], "setting": c["scene_context"],
                         "context_turns": [said(u) for u in T[:i]], "current_turn": said(t),
                         "gold": "SPEAK" if to == "assistant" else "SILENT", "category": cat,
                         "pattern": c["pattern"], "final": i == len(T) - 1,
                         "probe": probes.get((c["case_id"], i)), "noisy": c["acoustic"]["competing_speech"] != "none"})
    return rows


def keyword(r):
    return re.search(r"\bassistant\b", r["current_turn"]["text"], re.I) is not None


async def predict(policy, rows, par):
    key = tv_eval.key
    if policy == "keyword":
        return {key(r): keyword(r) for r in rows}
    if policy == "heuristic2":
        return {key(r): tv_eval.heuristic2(r) for r in rows}
    if policy == "higgs_zs":
        return (await tv_eval.run_higgs("higgs_zs", rows, par))[0]
    if policy in ("v4", "cheap", "v4b"):
        n = 1 if policy == "cheap" else 3
        judge = "higgs_dec4" if policy == "v4b" else "higgs_dec2"
        judges = [(await tv_eval.run_higgs(judge + (f"@{i}" if i else ""), rows, par))[0] for i in range(n)]
        checks = [(await tv_eval.run_higgs("higgs_fact" + (f"@{i}" if i else ""), rows, par))[0] for i in range(n)]
        return {key(r): sum(j[key(r)] for j in judges) * 2 > n or all(c[key(r)] for c in checks) for r in rows}
    raise ValueError(policy)


def score(rows, pred):
    k = tv_eval.key
    m = tv_eval.metrics(rows, pred)
    m["ci"] = tv_eval.boot_ci(rows, pred, n=1000)
    m["by_category"] = tv_eval.by_category(rows, pred)
    pr = [r for r in rows if r["probe"]]
    m["PRR_probes"] = f"{sum(pred[k(r)] for r in pr)}/{len(pr)}"
    m["PRR_probes_noisy"] = f"{sum(pred[k(r)] for r in pr if r['probe'] != 'none')}/{sum(r['probe'] != 'none' for r in pr)}"
    fi = [r for r in rows if r["final"]]
    m["responds_on_cue"] = f"{sum(pred[k(r)] for r in fi)}/{len(fi)}"
    sil = [r for r in rows if r["gold"] == "SILENT"]
    m["unprompted_per_case"] = round(sum(pred[k(r)] for r in sil) / max(1, len({r['meeting'] for r in rows})), 3)
    by = {}
    for r in rows:
        a, n = by.get(r["pattern"], (0, 0))
        by[r["pattern"]] = (a + (pred[k(r)] == (r["gold"] == "SPEAK")), n + 1)
    m["by_pattern"] = {p: f"{a}/{n}" for p, (a, n) in sorted(by.items())}
    return m


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/w/20251/oleksii/data/msi_bench")
    ap.add_argument("--sets", nargs="+", default=["text"])
    ap.add_argument("--policies", nargs="+", default=["keyword", "heuristic2", "v4"])
    ap.add_argument("--only", default="", help="probes,finals,unnamed: restrict to those lines (cheap first pass)")
    ap.add_argument("--sample", type=int, default=0, help="score every line of N random cases (seed 0) plus --only lines")
    ap.add_argument("--par", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "out", "msi"), exist_ok=True)
    for s in a.sets:
        tr = json.load(open(os.path.join(HERE, "out", "msi", "stt_transcripts.json"))) if s == "stt" else None
        rows = rows_from(a.data, anon=s in ("anon", "stt"), transcripts=tr)
        if tr is not None:  # only the cases msi_stt.py transcribed
            have = {k.split("/")[0] for k in tr}
            rows = [r for r in rows if r["meeting"] in have]
        if a.only or a.sample:
            want = set(a.only.split(",")) if a.only else set()
            cases = sorted({r["meeting"] for r in rows})
            picked = set(random.Random(0).sample(cases, a.sample)) if a.sample else set()
            rows = [r for r in rows if ("probes" in want and r["probe"]) or ("finals" in want and r["final"])
                    or ("unnamed" in want and r["category"] == "to_ai_unnamed") or r["meeting"] in picked]
        print(s, len(rows), "decisions", flush=True)
        report = {}
        for pol in a.policies:
            pred = await predict(pol, rows, a.par)
            report[pol] = score(rows, pred)
            print(s, pol, json.dumps(report[pol]), flush=True)
        tag = s + (f"_{a.only.replace(',', '+')}" if a.only else "") + (f"_sample{a.sample}" if a.sample else "")
        json.dump(report, open(os.path.join(HERE, "out", "msi", f"report_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
