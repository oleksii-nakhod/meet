"""Third Voice: should the AI speak after this line? (from the Third Voice prototype, tv_eval.py + retrieve.py)

Higgs Realtime (text mode) answers small questions about the latest line as JSON (who it is aimed at, does it ask for
something, was it already answered, can the AI answer, does it contradict the facts) and a fixed rule turns them into
SPEAK/SILENT. DEC_SYS4 ("v4b") is the version scored on Third Voice's test sets (tests/data/testset/,
tests/tv_regress.py); agent.py builds its v5 prompt on it. A second prompt (FACT_SYS) only checks for contradicted facts.
toks/top_facts keep a big fact sheet short: the k facts with the most word overlap with the latest lines. No API calls
here except ask_higgs."""
import asyncio
import json
import re

from mmhm.higgs import Session


def ctx(r, n=14):
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in r["context_turns"][-n:]) or "(the conversation just started)"


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


# v4b (after MSI-Bench, where most requests are commands: "Assistant, put that through"): "ask" includes asking for an
# action, not only for information or help
DEC_SYS4 = DEC_SYS2.replace(
    '"ask": true if it asks for information, a check, a reminder or help (including indirect forms like "I wonder when..." or "I can\'t remember...");',
    '"ask": true if it asks for information, a check, a reminder, help or an action, including instructions and commands such as "go ahead and book it" (and indirect forms like "I wonder when..." or "I can\'t remember...");')
assert DEC_SYS4 != DEC_SYS2


def dec_prompt(r, template=DEC_SYS):
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



# ---------------------------------------------------------------- fact retrieval (retrieve.py)
STOP = set("""a an the and or but if so to of in on at for with from by as is are was were be been it its it's this that
these those i you he she we they me him her us them my your our their do does did not no yes okay ok um uh like just
yeah right well oh what when where who how why which can could would should will about there here then than""".split())


def toks(s):
    return {re.sub(r"(ing|ed|es|s)$", "", w) for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in STOP and len(w) > 1}


def top_facts(r, k):
    q = toks(" ".join([t["text"] for t in r["context_turns"][-2:]] + [r["current_turn"]["text"]]))
    scored = sorted(enumerate(r["facts"]), key=lambda x: (-len(q & toks(x[1])), x[0]))
    return [f for _, f in scored[:k]]
