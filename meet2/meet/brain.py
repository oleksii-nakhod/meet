"""Whether to speak (the judge) and what to say (the writer).

The judge prompt and rule are Meet 1's "v5" (Third Voice's v4b plus greetings, the AI's own name, follow-ups and a
one-to-one mode), which were tuned on Third Voice's group test sets and Meet 1's judge cases. The writer is Meet 1's
v2 reply prompt, now grounded in the knowledge base: relevant facts plus the top document excerpts, with an optional
source marker at the very end ("[[source: 2]]") that is shown in the UI and never spoken.
Calls go through Models.chat / chat_stream (higgs-realtime chat completions, ~0.25 s to the first token).
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

# ------------------------------------------------------------------ judge prompt (Meet 1 v5)
DEC_SYS5 = """You are {ai}, an AI participant with a voice in a group conversation ({setting}) with {people}.
What you know: {facts} You also have ordinary general knowledge.
You will see the conversation so far and the latest utterance. Do not reply to it. Analyse the latest utterance and output ONLY one compact JSON object with these keys:
"to": who the speaker is talking to: "me" if it is aimed at you, {ai} (by name, or clearly by context, e.g. a follow-up to what you just said); "person" if aimed at a specific human; "group" if aimed at everyone; "self" if thinking aloud or stopping mid-sentence. Greeting you, saying your name to get your attention, or asking whether you are there or can hear counts as "me"; talking about you in the third person ("{ai} is great", "{ai} said...") does not.
"ask": true if it asks for information, a check, a reminder, help or an action, including instructions and commands such as "go ahead and book it" (and indirect forms like "I wonder when..." or "I can't remember..."); false for statements, opinions, jokes, chat and rhetorical questions.
"answered": true if it was already answered earlier in the conversation (even if the speaker missed the answer) or the speaker answers it themselves.
"can": true only if the answer is a fact you know (from what you know above or general knowledge); false if it is a decision, plan, preference, feeling or opinion that is up to the humans, or something you cannot know.
"reply": true if someone in your place would be expected to respond now: you were greeted, called, asked something, asked to go on, explain or repeat, or (in a one-to-one call) told something that invites a reaction; false for thanks and acknowledgements ("okay", "got it", "thanks"), backchannels ("mm-hm", "yeah"), fillers, unfinished sentences and remarks to someone outside the call.
"wrong": true if it states something that contradicts what you know and the group may act on it.
Example output: {{"to": "group", "ask": true, "answered": false, "can": true, "reply": false, "wrong": false}}"""

ONE_TO_ONE = ("\nThis is a one-to-one call: {people} and you are the only participants, so anything that is not "
              "thinking aloud or said to someone outside the call is aimed at you (\"me\").")
TO_1TO1 = ('"to": who {people} is talking to: "me" (you, {ai}: in this call questions, requests, opinions, reactions '
           'and small talk are aimed at you); "self" if thinking aloud, searching for something, reading out, or '
           'stopping mid-sentence; "outside" if talking to someone who is not in the call (a person or pet in their '
           'room, a phone call).')

WRITER = ("You are {ai}, an AI participant with a voice in an online meeting ({setting}) with {people}. "
          "What you know: {facts} You decided to speak up after the latest utterance. Reply to the latest utterance "
          "itself. If it states something that contradicts what you know (a wrong day, time, number, place, dose or "
          "rule), start by correcting it plainly. Bring up what you know only when it answers or corrects the latest "
          "utterance, and don't repeat what was already said. A turn of yours marked [interrupted] was cut off: reply to "
          "what the person who interrupted said. If they asked a question or raised a topic, answer it directly; only "
          "if they just said \"wait\" or \"stop\" with nothing else, say something like \"Sure, go ahead.\" Don't "
          "go back to the interrupted turn unless asked. A line marked [said while you were correcting] was said over your "
          "correction: respond to it now (if they doubt the correction, briefly restate the evidence). You can answer, remind, summarize and suggest, but you can't do work "
          "outside the call, so don't volunteer for tasks. Follow the person's latest topic, including general "
          "questions outside the meeting. Say what a helpful teammate would say: one or two short spoken sentences, "
          "under 30 words, no lists, no preamble, no markdown. If your answer comes from one of the numbered "
          "sources, end with [[source: N]] after the last sentence (it is not read aloud).")

STOP_WORDS_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class Turn:
    speaker: str
    text: str


@dataclass
class Context:
    ai: str
    setting: str
    people: list[str]
    facts: list[str]
    turns: list[Turn]            # the conversation before the current turn (oldest first)
    current: Turn
    sources: list[dict] = field(default_factory=list)   # knowledge-base excerpts: {"doc", "page", "text"}

    @property
    def one_to_one(self) -> bool:
        return len(self.people) == 1

    def history(self, n: int = 14) -> str:
        return "\n".join(f"{t.speaker}: {t.text}" for t in self.turns[-n:]) or "(the conversation just started)"


def parse_json(text: str):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge_prompt(c: Context) -> tuple[str, str]:
    t = DEC_SYS5
    if c.one_to_one:
        t = re.sub(r'^"to":.*$', lambda _m: TO_1TO1, t, count=1, flags=re.M)
        t = t.replace("\nYou will see", ONE_TO_ONE + "\nYou will see", 1).replace('{{"to": "group"', '{{"to": "me"')
    system = t.format(ai=c.ai, setting=c.setting, people=", ".join(c.people), facts=" ".join(c.facts) or "(nothing)")
    user = (f"Conversation so far:\n{c.history()}\n\nLatest utterance:\n{c.current.speaker}: {c.current.text}"
            "\n\nOutput the JSON object only.")
    return system, user


def _named_ai_before(turns: list[Turn], speaker: str, ai: str, n: int = 2) -> bool:
    mine = [t for t in turns if t.speaker == speaker][-n:]
    return any(ai.lower() in t.text.lower() for t in mine)


def _vocative_person(text: str, people: list[str], ai: str) -> str | None:
    """"Alex, can you..." / "..., Alex?": said to a person in the call."""
    for p in people:
        if re.match(rf"^\W*(?:hey |so |okay |ok )?{re.escape(p)}\b[,!?]", text, re.I) or re.search(
                rf",\s*{re.escape(p)}[.?!]*\s*$", text, re.I):
            return p
    return None


def judge_rule(d, c: Context) -> bool:
    if not isinstance(d, dict):
        return False
    w = d.get("wrong")
    if w is True or (isinstance(w, (int, float)) and not isinstance(w, bool) and w > 0):
        return True
    to = str(d.get("to", "")).lower()
    if to == c.ai.lower():
        to = "me"
    cur = c.current.text
    if c.one_to_one and to == "group":
        to = "me"
    if _vocative_person(cur, [p for p in c.people if p != c.current.speaker], c.ai):
        to = "person"
    if (to == "me" and not c.one_to_one and c.ai.lower() not in cur.lower()
            and not any(t.speaker == c.ai for t in c.turns[-2:])
            and not _named_ai_before(c.turns, c.current.speaker, c.ai)):
        to = "group"            # in a group "you" is nearly always a person
    if (to == "group" and c.turns and (c.turns[-1].speaker == c.ai or _named_ai_before(c.turns, c.current.speaker, c.ai, 1))
            and (d.get("ask") is True or cur.rstrip().endswith("?"))
            and len(STOP_WORDS_RE.findall(cur)) <= 4 and not any(p.lower() in cur.lower() for p in c.people)):
        to, d = "me", {**d, "ask": True}   # "why?", "anything else?" right after the AI spoke
    if to == "group":
        return d.get("ask") is True and d.get("answered") is not True and d.get("can") is True
    if to != "me":
        return False
    if c.one_to_one and len(STOP_WORDS_RE.findall(cur)) > 3:
        return True
    return d.get("ask") is True or d.get("reply") is True


def reason(d: dict, speak: bool) -> str:
    to = str(d.get("to", "")).lower()
    if speak:
        return "asked me" if to == "me" else "a question I can answer"
    if to not in ("me", "group", "self", ""):
        return f"talking to {d.get('to')}"
    if to == "self":
        return "thinking aloud"
    if not d.get("ask"):
        return "nothing asked"
    if d.get("answered"):
        return "already answered"
    return "not mine to answer"


async def decide(models, c: Context, samples: int = 1) -> tuple[bool, dict]:
    """Majority of the judge votes that came back (a failed request abstains). Decided as soon as a majority says
    speak. Returns (speak, {"reading", "why", "votes"})."""
    system, user = judge_prompt(c)

    async def vote():
        try:
            return parse_json(await models.chat(system, user, max_tokens=80, temperature=0.3 if samples > 1 else 0.0,
                                                timeout=6))
        except Exception:  # noqa: BLE001
            return None
    tasks = [asyncio.ensure_future(vote()) for _ in range(samples)]
    need = samples // 2 + 1
    readings: list[dict] = []
    speak_votes = 0
    for fut in asyncio.as_completed(tasks):
        d = await fut
        if isinstance(d, dict):
            readings.append(d)
            speak_votes += judge_rule(d, c)
            if speak_votes >= need:
                break
    for t in tasks:
        t.cancel()
    speak = speak_votes >= need or (bool(readings) and speak_votes * 2 > len(readings))
    reading = readings[0] if readings else {}
    return speak, {"reading": reading, "why": reason(reading, speak), "votes": len(readings)}


def writer_prompt(c: Context, *, interrupted_note: str = "") -> tuple[str, str]:
    system = WRITER.format(ai=c.ai, setting=c.setting, people=", ".join(c.people), facts=" ".join(c.facts) or "(nothing)")
    src = ""
    if c.sources:
        src = "Numbered sources from the meeting's documents:\n" + "\n".join(
            f"[{i + 1}] {s['doc']}{' p.' + str(s['page']) if s.get('page') else ''}: {s['text'][:700]}"
            for i, s in enumerate(c.sources)) + "\n\n"
    user = (f"{src}Conversation so far:\n{c.history(20)}\n\nLatest utterance:\n{c.current.speaker}: {c.current.text}"
            f"{interrupted_note}\n\nYour spoken reply:")
    return system, user


SOURCE_RE = re.compile(r"\[\[\s*source[s]?\s*:?\s*([0-9, ]+)\s*\]\]", re.I)


class SpokenText:
    """Filters the writer's stream: everything before a "[[" marker is speech; the marker names the source."""

    def __init__(self):
        self.spoken, self.tail, self.held = "", "", ""

    def push(self, delta: str) -> str:
        if self.tail:
            self.tail += delta
            return ""
        text = self.held + delta
        i = text.find("[[")
        if i >= 0:
            self.tail = text[i:]
            out, self.held = text[:i], ""
        elif text.endswith("["):
            out, self.held = text[:-1], "["
        else:
            out, self.held = text, ""
        out = out.replace("*", "")
        self.spoken += out
        return out

    def finish(self) -> str:
        out, self.held = self.held, ""
        self.spoken += out
        return out

    def source_ids(self) -> list[int]:
        m = SOURCE_RE.search(self.tail)
        return [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []
