"""Speak/stay-silent policies on three sets, with the AI's fact sheet in context:
  meeting : the original hand-written 20-turn meeting (meeting.py, AI = "Higgs")
  dev     : independently written meetings A1-A4 (testset/), used to design prompts and rules
  test    : independently written meetings B1-B4, C1-C4, held out; scored once per final policy

Policies:
  heuristic   : speak after any "?" or when the AI's name is said (the old baseline)
  heuristic2  : speak when the AI's name is said, or after a "?" that doesn't name a person in the room
  higgs_zs    : Higgs Realtime (text mode) asked SPEAK or SILENT (old zero-shot prompt + facts)
  higgs_dec   : Higgs Realtime (text mode) answers five small questions about the utterance as JSON
                (who it is aimed at, does it ask for something, already answered, can the AI answer, does it
                contradict the facts); a fixed rule turns them into SPEAK/SILENT
Higgs answers are cached in out/cache_<policy>.jsonl.

Usage: python tv_eval.py --sets meeting dev [--policies heuristic heuristic2 higgs_dec] [--par 6]"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mmhm"))
from mmhm.higgs import Session  # noqa: E402

import meeting  # noqa: E402


def load(name):
    if name == "meeting":
        rows = meeting.records()
        for r in rows:
            r.update(meeting="M0", line=r["id"], facts=[meeting.FACTS.split(": ", 1)[1]], people=["Ana", "Ben", "Chen"],
                     setting="hackathon team meeting", category="?")
        return rows
    return [json.loads(line) for line in open(os.path.join(HERE, "testset", f"points_{name}.jsonl"))]


def key(r):
    return f"{r['meeting']}:{r['line']}"


def ctx(r, n=14):
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in r["context_turns"][-n:]) or "(the conversation just started)"


# ---------------------------------------------------------------- heuristics
def heuristic(r):
    cur = r["current_turn"]["text"]
    return "?" in cur or re.search(rf"\b{re.escape(r['target_speaker'])}\b", cur, re.I) is not None


def heuristic2(r):
    cur = r["current_turn"]["text"]
    if re.search(rf"\b{re.escape(r['target_speaker'])}\b", cur, re.I):
        return True
    others = [p for p in r["people"] if p != r["current_turn"]["speaker"]]
    return "?" in cur and not any(re.search(rf"\b{re.escape(p)}\b", cur) for p in others)


# ---------------------------------------------------------------- Higgs judges
def zs_prompt(r):
    ai = r["target_speaker"]
    sys_ = (f"You are {ai}, an AI participant in a group conversation ({r['setting']}) with {', '.join(r['people'])}. "
            f"What you know: {' '.join(r['facts'])} After each utterance there is a short pause and you must decide "
            "whether to speak now. Speak only when someone addresses you, or when the group needs something you know and "
            "nobody else can answer. Stay silent when people talk to each other or are mid-thought. Answer with exactly "
            "one word: SPEAK or SILENT.")
    user = (f"Conversation so far:\n{ctx(r)}\n\nJust said (now there's a pause):\n{r['current_turn']['speaker']}: "
            f"{r['current_turn']['text']}\n\nSPEAK or SILENT?")
    return sys_, user


DEC_SYS = """You are {ai}, an AI participant with a voice in a group conversation ({setting}) with {people}.
What you know: {facts} You also have ordinary general knowledge.
You will see the conversation so far and the latest utterance. Do not reply to it. Analyse the latest utterance and output ONLY one compact JSON object with these keys:
"to": who the speaker is talking to: "me" if it is aimed at you, {ai} (by name, or clearly by context, e.g. a follow-up to what you just said); "person" if aimed at a specific human; "group" if aimed at everyone; "self" if thinking aloud or stopping mid-sentence.
"ask": true if it asks for information, a check, a reminder or help (including indirect forms like "I wonder when..." or "I can't remember..."); false for statements, opinions, jokes, chat and rhetorical questions.
"answered": true if someone in the conversation already answered it.
"can": true if you could usefully answer it from what you know or general knowledge; false if only the humans can (their own preferences, plans, feelings).
"wrong": true if it states something that contradicts what you know and the group may act on it.
Example output: {{"to": "group", "ask": true, "answered": false, "can": true, "wrong": false}}"""


# v2 (after looking at dev errors only): "can" = a fact the AI knows, not a decision/plan that is up to the humans;
# "answered" includes answers given earlier that the asker missed.
DEC_SYS2 = DEC_SYS.replace(
    '"answered": true if someone in the conversation already answered it.',
    '"answered": true if it was already answered earlier in the conversation (even if the speaker missed the answer) or the speaker answers it themselves.').replace(
    '"can": true if you could usefully answer it from what you know or general knowledge; false if only the humans can (their own preferences, plans, feelings).',
    '"can": true only if the answer is a fact you know (from what you know above or general knowledge); false if it is a decision, plan, preference, feeling or opinion that is up to the humans, or something you cannot know.')


# v3 (designed on dev2): facts numbered, and "wrong" must name the contradicted fact (0 if none)
DEC_SYS3 = DEC_SYS2.replace("What you know: {facts}", "What you know (numbered facts):\n{facts}\n").replace(
    '"wrong": true if it states something that contradicts what you know and the group may act on it.',
    '"wrong": the number of the fact it contradicts (a wrong day, date, time, number, address, dose, name or rule that '
    'the group may act on), or 0 if it contradicts none. Rounding, rewording or extra detail is not a contradiction.').replace(
    '"can": true, "wrong": false}}', '"can": true, "wrong": 0}}')


# v4b (after MSI-Bench, where most requests are commands: "Assistant, put that through"): "ask" includes asking for an
# action, not only for information or help
DEC_SYS4 = DEC_SYS2.replace(
    '"ask": true if it asks for information, a check, a reminder or help (including indirect forms like "I wonder when..." or "I can\'t remember...");',
    '"ask": true if it asks for information, a check, a reminder, help or an action, including instructions and commands such as "go ahead and book it" (and indirect forms like "I wonder when..." or "I can\'t remember...");')
assert DEC_SYS4 != DEC_SYS2


def dec_prompt(r, template=DEC_SYS):
    if template is DEC_SYS3:
        facts = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(r["facts"]))
        sys_ = template.format(ai=r["target_speaker"], setting=r["setting"], people=", ".join(r["people"]), facts=facts)
        user = (f"Conversation so far:\n{ctx(r)}\n\nLatest utterance:\n{r['current_turn']['speaker']}: "
                f"{r['current_turn']['text']}\n\nOutput the JSON object only.")
        return sys_, user
    sys_ = template.format(ai=r["target_speaker"], setting=r["setting"], people=", ".join(r["people"]), facts=" ".join(r["facts"]))
    user = (f"Conversation so far:\n{ctx(r)}\n\nLatest utterance:\n{r['current_turn']['speaker']}: {r['current_turn']['text']}"
            "\n\nOutput the JSON object only.")
    return sys_, user


def dec_rule(d, ai=None, v=1):
    if not isinstance(d, dict):
        return False
    w = d.get("wrong")
    if w is True or (isinstance(w, (int, float)) and not isinstance(w, bool) and w > 0):
        return True
    to = str(d.get("to", "")).lower()
    if v >= 2 and ai and to == ai.lower():  # Higgs sometimes writes the AI's name instead of "me"
        to = "me"
    if to == "me":
        return v < 2 or d.get("ask") is True  # v2: being talked to without a request ("okay, thanks") needs no reply
    return to == "group" and d.get("ask") is True and d.get("answered") is not True and d.get("can") is True


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


async def ask_higgs(sys_, user, attempts=3):
    """One text-mode answer. Under sustained load Higgs sometimes fails a response ("upstream_error"): retry."""
    for k in range(attempts):
        async with Session({"instructions": sys_, "output_modalities": ["text"]}) as s:
            await s.say_text(user)
            _, ev = await s.wait("response.done", timeout=45)
            text = s.transcript(ev["response"]["id"]).strip()
        if text:
            return text
        await asyncio.sleep(1.0 + k)
    return text


FACT_SYS = """You are {ai}, an AI participant in a group conversation ({setting}) with {people}.
Facts you know for certain:
{facts}
You will see the conversation so far and the latest utterance. Do not reply to it. Decide whether the latest utterance states or assumes something that contradicts one of the numbered facts (a wrong day, date, time, number, address, name or rule) in a way that matters for what the group is doing. Differences in wording, rounding or extra detail are not contradictions, and a question is not a contradiction.
Output ONLY one compact JSON object: {{"contradicts": <the number of the contradicted fact, or 0 if none>}}"""


def fact_prompt(r):
    facts = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(r["facts"]))
    sys_ = FACT_SYS.format(ai=r["target_speaker"], setting=r["setting"], people=", ".join(r["people"]), facts=facts)
    user = (f"Conversation so far:\n{ctx(r)}\n\nLatest utterance:\n{r['current_turn']['speaker']}: {r['current_turn']['text']}"
            "\n\nOutput the JSON object only.")
    return sys_, user


def _facts_to_user(sys_, user, r, numbered=False):
    """Variant with the facts in the request instead of the session instructions, so one pre-opened session pool can
    serve per-line (retrieved) fact sheets. The instructions keep a placeholder sentence."""
    facts = "\n".join((f"{i + 1}. " if numbered else "- ") + f for i, f in enumerate(r["facts"]))
    return sys_, f"What you know right now:\n{facts}\n\n{user}"


def dec_prompt_u(r):
    nofacts = {**r, "facts": ["(listed in each request)"]}
    return _facts_to_user(dec_prompt(nofacts, DEC_SYS2)[0], dec_prompt(r, DEC_SYS2)[1], r)


def fact_prompt_u(r):
    nofacts = {**r, "facts": ["(listed in each request, numbered)"]}
    return _facts_to_user(fact_prompt(nofacts)[0], fact_prompt(r)[1], r, numbered=True)


PROMPTS = {"higgs_zs": zs_prompt, "higgs_dec": dec_prompt, "higgs_dec2": lambda r: dec_prompt(r, DEC_SYS2),
           "higgs_dec3": lambda r: dec_prompt(r, DEC_SYS3),
           "higgs_dec4": lambda r: dec_prompt(r, DEC_SYS4),
           "higgs_fact": fact_prompt,
           "higgs_dec2u": dec_prompt_u, "higgs_factu": fact_prompt_u}


async def run_higgs(policy, rows, par):
    """policy@n = an independent repeat sample of the same prompt (own cache)."""
    path = os.path.join(HERE, "out", f"cache_{policy}.jsonl")
    policy = policy.split("@")[0]
    cache = {}
    if os.path.exists(path):
        for line in open(path):
            d = json.loads(line)
            cache[d["key"]] = d["raw"]
    # cache by prompt content, so the same utterance as gold text and as an STT transcript are separate entries
    ck = {key(r): key(r) + ":" + hashlib.sha1(json.dumps(PROMPTS[policy](r)).encode()).hexdigest()[:12] for r in rows}
    todo = [r for r in rows if not cache.get(ck[key(r)], "").strip() or cache[ck[key(r)]].startswith("ERROR")]
    sem = asyncio.Semaphore(par)

    async def one(r):
        async with sem:
            for attempt in range(3):
                try:
                    raw = await ask_higgs(*PROMPTS[policy](r))
                    break
                except Exception as e:  # noqa: BLE001
                    raw = f"ERROR {e}"
                    await asyncio.sleep(2 + 3 * attempt)
            cache[ck[key(r)]] = raw
            with open(path, "a") as f:
                f.write(json.dumps({"key": ck[key(r)], "raw": raw}) + "\n")

    await asyncio.gather(*[one(r) for r in todo])
    if policy == "higgs_zs":
        return {key(r): "SPEAK" in cache[ck[key(r)]].upper() for r in rows}, cache
    if policy in ("higgs_fact", "higgs_factu"):
        return {key(r): bool((parse_json(cache[ck[key(r)]]) or {}).get("contradicts")) for r in rows}, cache
    v = 2 if policy in ("higgs_dec2", "higgs_dec3", "higgs_dec4", "higgs_dec2u") else 1
    return {key(r): dec_rule(parse_json(cache[ck[key(r)]]), r["target_speaker"], v) for r in rows}, cache


# ---------------------------------------------------------------- metrics
def metrics(rows, pred):
    y = [r["gold"] == "SPEAK" for r in rows]
    p = [pred[key(r)] for r in rows]
    tp = sum(a and b for a, b in zip(y, p)); fn = sum(a and not b for a, b in zip(y, p))
    tn = sum(not a and not b for a, b in zip(y, p)); fp = sum(not a and b for a, b in zip(y, p))
    rec, spec = tp / max(1, tp + fn), tn / max(1, tn + fp)
    return {"bacc": round((rec + spec) / 2, 3), "speak": f"{tp}/{tp + fn}", "silent": f"{tn}/{tn + fp}"}


def boot_ci(rows, pred, n=2000, seed=0):
    """95% CI of balanced accuracy, resampling whole meetings (decisions within a meeting are correlated)."""
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r["meeting"], []).append(r)
    ms = list(by)
    vals = []
    for _ in range(n):
        sample = [r for m in (rng.choice(ms) for _ in ms) for r in by[m]]
        if any(r["gold"] == "SPEAK" for r in sample) and any(r["gold"] != "SPEAK" for r in sample):
            vals.append(metrics(sample, pred)["bacc"])
    if not vals:  # one class only (e.g. the unlabelled AMI sets)
        return None
    vals.sort()
    return round(vals[int(.025 * len(vals))], 3), round(vals[int(.975 * len(vals)) - 1], 3)


def by_category(rows, pred):
    out = {}
    for r in rows:
        c = r.get("category", "?")
        ok = pred[key(r)] == (r["gold"] == "SPEAK")
        a, n = out.get(c, (0, 0))
        out[c] = (a + ok, n + 1)
    return {c: f"{a}/{n}" for c, (a, n) in sorted(out.items())}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["meeting", "dev"])
    ap.add_argument("--policies", nargs="+", default=["heuristic", "heuristic2", "higgs_dec"])
    ap.add_argument("--par", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    report = {}
    for s in a.sets:
        rows = load(s)
        for pol in a.policies:
            if pol.startswith("or:"):  # or:a+b = speak if either policy says speak (each may be a vote:)
                preds = []
                for part in pol[3:].split("+"):
                    if part.startswith(("vote:", "all:")):
                        base, n = part.split(":", 1)[1].rsplit("x", 1)
                        ps = [(await run_higgs(base + (f"@{i}" if i else ""), rows, a.par))[0] for i in range(int(n))]
                        agg = all if part.startswith("all:") else (lambda v: sum(v) * 2 > len(v))
                        preds.append({k: agg([p[k] for p in ps]) for k in ps[0]})
                    else:
                        preds.append((await run_higgs(part, rows, a.par))[0])
                pred = {k: any(p[k] for p in preds) for k in preds[0]}
            elif pol.startswith("all:"):  # all:higgs_factx3 = speak only if every sample says so
                base, n = pol[4:].rsplit("x", 1)
                preds = [(await run_higgs(base + (f"@{i}" if i else ""), rows, a.par))[0] for i in range(int(n))]
                pred = {k: all(p[k] for p in preds) for k in preds[0]}
            elif pol.startswith("vote:"):  # vote:higgs_dec2x3 = majority of higgs_dec2, @2, @3
                base, n = pol[5:].rsplit("x", 1)
                preds = [(await run_higgs(base + (f"@{i}" if i else ""), rows, a.par))[0] for i in range(int(n))]
                pred = {k: sum(p[k] for p in preds) * 2 > len(preds) for k in preds[0]}
            elif pol.split("@")[0] in PROMPTS:
                pred, _ = await run_higgs(pol, rows, a.par)
            else:
                pred = {key(r): globals()[pol](r) for r in rows}
            m = metrics(rows, pred)
            m["ci"] = boot_ci(rows, pred) if s != "meeting" else None
            m["by_category"] = by_category(rows, pred) if s != "meeting" else None
            report[f"{s}/{pol}"] = m
            print(s, pol, json.dumps(m), flush=True)
    return report


if __name__ == "__main__":
    asyncio.run(main())
