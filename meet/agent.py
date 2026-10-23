"""Higgs as a participant in a meeting: two brains behind one switch, on the room's slot 0.

default  : plain Higgs Realtime as Boson's docs prescribe. It hears the room (everyone's audio mixed), its own
           server_vad decides the turns, it answers after every turn, and playback stops on any speech_started.
           This is mmhm.engine.Engine(mode="server_vad"), unchanged.
improved : Third Voice decides *whether* to speak (v4b: 3 decomposed judges + 3 fact-checks on each person's
           transcribed line, tv_eval.DEC_SYS4), Mm-hm decides *when* (floor.py): a person's line ends when the
           turn-taking model says they are done, a ready reply waits through a thinking pause or an unfinished
           sentence and is dropped if someone else has spoken since, and while Higgs talks a listener's "mm-hm" only
           ducks it; a real interruption stops it and Higgs remembers only what was heard.

The humans hear each other directly (room.py relays their audio); this module only listens and talks.
Every Higgs request is counted (CALLS); above MEET_BUDGET the agent stops asking and says so in the room log."""
import asyncio
import base64
import dataclasses
import os
import re
import sys
import threading
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mmhm"))
sys.path.insert(0, os.path.join(HERE, "..", "third_voice"))  # tv_eval, retrieve
from mmhm.engine import INCOMPLETE, N, TICK, Engine, Policy, VAD, to16k  # noqa: E402
from mmhm.expressive import SentenceSpeaker  # noqa: E402
from mmhm.higgs import SR, Session  # noqa: E402

import avatar  # noqa: E402
import floor  # noqa: E402
import tv_eval  # noqa: E402

VOICE = os.environ.get("MEET_VOICE", "chloe")      # the same preset voice in both modes, so the voice doesn't tell
# Judge/fact-check samples per line. Default: 1 in a one-to-one call, 3 with more people. On Third Voice's group test
# set 1 sample speaks up wrongly more often (balanced 0.906 vs 0.928, silent 0.924 vs 0.967); on the one-to-one and
# held-out cases (tests/judge_cases.py) 1 and 3 do the same. Each sample is a standing Higgs session, and Higgs caps
# concurrent sessions per API key (close 1013 "Max concurrency exceeded").
SAMPLES = int(os.environ["MEET_SAMPLES"]) if os.environ.get("MEET_SAMPLES") else None


def samples_for(n_people):
    return SAMPLES or (1 if n_people <= 1 else 3)
BUDGET = int(os.environ.get("MEET_BUDGET", "30000"))  # requests per server run (~10 per judged line in improved mode)
LANG = os.environ.get("MEET_LANG", "en")           # transcription language hint ("" = auto: "mm" came back as "嗯")
SPECULATE = os.environ.get("MEET_SPECULATE", "1") == "1"  # voice likely replies while the judge decides
PROMPT = os.environ.get("MEET_PROMPT", "v5")       # judge prompt (see judge_prompt)
SIGNALS = os.environ.get("MEET_SIGNALS", "1") == "1"  # per-person turn-taking models (about one CPU core each)
CALLS = {"n": 0}
# In 1:1 Mm-hm a commit after 0.1 s of silence is a cancellable guess. Here a committed line is transcribed and judged
# (7 requests) and shown to everyone, so a line ends only after a real pause: 0.45 s with the turn-taking model
# agreeing they're done (else its hold rules: max_wait / give_up). First B2 run with 0.1 s: 58 judged pieces for 32
# lines ("So.", "Um.", then the line again), 474 requests. 2026-10-23: 0.45 -> 0.30 (MEET_LINE_SILENCE) once fillers
# and backchannels stopped being judged (prefilter): the commit was 0.46 s of a 1.8 s median reply.
LINE_POLICY = dataclasses.replace(Policy(), min_silence=float(os.environ.get("MEET_LINE_SILENCE", "0.30")))
FLOOR_POLICY = dataclasses.replace(Policy(), min_silence=0.35)  # room quiet before Higgs may start (0.1 s let it
#                                                                 start in the gap between an interrupter's words)
STALE_S = 3.5  # a reply that couldn't start this long after its line, while people kept talking, is about a moment
STALE_ADDRESSED = 6.0  # an answer to a question put to the AI by name waits longer for a gap
STALE_MAX = 8.0  # that has passed; in a quiet room it may come later (a slow Higgs request), but not after this
CORRECT = ("[The latest utterance contradicts what you know. Start by correcting it plainly, as {ai}, not as one of "
           "the people.]\n")

REPLY_SYS = ("You are {ai}, an AI participant with a voice in an online meeting ({setting}) with {people}. "
             "What you know: {facts} You decided to speak up after the latest utterance. If the latest utterance "
             "states something that contradicts what you know (a wrong day, time, number, place, dose or rule), start by "
             "correcting it plainly. Otherwise say what a helpful teammate would say: one or two short spoken sentences, "
             "no lists, no preamble.")  # third_voice/live_demo.py's writer, "group conversation" -> "online meeting"
# v2 (2026-10-23, tests/writer_cases.py): v1 recited the fact sheet unprompted ("How's it going?" -> "...the deadline is
# Saturday at 3 pm"), carried on with its old answer after being interrupted, and promised work it can't do.
REPLY_SYS2 = ("You are {ai}, an AI participant with a voice in an online meeting ({setting}) with {people}. "
              "What you know: {facts} You decided to speak up after the latest utterance. Reply to the latest "
              "utterance itself. If it states something that contradicts what you know (a wrong day, time, number, place, "
              "dose or rule), start by correcting it plainly. Bring up what you know only when it answers or corrects the "
              "latest utterance, and don't repeat what was already said. A turn of yours marked [interrupted] was cut "
              "off: answer what the person who interrupted said (if they only want to speak, say something like "
              "\"Sure, go ahead.\"), and don't go back to it unless asked. You can answer, remind, summarize and "
              "suggest, but you can't do work outside the call, so don't volunteer for tasks. Say what a helpful "
              "teammate would say: one or two short spoken sentences, under 30 words, no lists, no preamble.")
PERSONA = ("You are {ai}, an AI participant with a voice in an online meeting ({setting}) with {people}. "
           "What you know: {facts} Keep each reply to one or two short spoken sentences.")


def spend(n=1):
    """Count Higgs requests; False once the budget is used up (the agent then goes quiet: rooms are told once)."""
    if CALLS["n"] + n > BUDGET:
        CALLS["over"] = True
        return False
    CALLS["n"] += n
    return True


def repeats(reply, previous, threshold=0.8):
    """True if the text is nearly the same as one of the previous lines (third_voice/app.py)."""
    from difflib import SequenceMatcher
    norm = lambda x: " ".join(x.lower().split())  # noqa: E731
    return any(SequenceMatcher(None, norm(reply), norm(p)).ratio() >= threshold for p in previous)


async def close(session):
    """Close a Higgs session that may never have finished opening (a failed open has no socket)."""
    if getattr(session, "ws", None) is not None:
        try:
            await session.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass


def reason(d, speak):
    """The judge's reading of a line, as the few words the transcript panel shows."""
    to = str(d.get("to", "")).lower()
    if d.get("wrong"):
        return "corrects a mistake"
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


# v5 (meet, 2026-10-23; tests/judge_cases.py): v4b ignored greetings and its own name ("Hey, Higgs." -> to me, no
# request -> silent) and, in a one-to-one call, read nearly everything as aimed at "group" ("Anything else?", "Go on.",
# opinions) -> silent. v5 adds "reply" (would someone in your place be expected to answer now?) and says when the call
# is one-to-one.
DEC_SYS5 = tv_eval.DEC_SYS4.replace(
    '"self" if thinking aloud or stopping mid-sentence.',
    '"self" if thinking aloud or stopping mid-sentence. Greeting you, saying your name to get your attention, or asking '
    'whether you are there or can hear counts as "me"; talking about you in the third person ("{ai} is great", '
    '"{ai} said...") does not.').replace(
    '"wrong": true if it states',
    '"reply": true if someone in your place would be expected to respond now: you were greeted, called, asked '
    'something, asked to go on, explain or repeat, or (in a one-to-one call) told something that invites a reaction; '
    'false for thanks and acknowledgements ("okay", "got it", "thanks"), backchannels ("mm-hm", "yeah"), fillers, '
    'unfinished sentences and remarks to someone outside the call.\n"wrong": true if it states').replace(
    '"can": true, "wrong": false}}', '"can": true, "reply": false, "wrong": false}}')
assert DEC_SYS5.count('"reply"') == 2 and "counts as" in DEC_SYS5
ONE_TO_ONE = ("\nThis is a one-to-one call: {people} and you are the only participants, so anything that is not thinking "
              "aloud or said to someone outside the call is aimed at you (\"me\").")
TO_1TO1 = ('"to": who {people} is talking to: "me" (you, {ai}: in this call questions, requests, opinions, reactions and '
           'small talk are aimed at you); "self" if thinking aloud, searching for something, reading out, or stopping '
           'mid-sentence; "outside" if talking to someone who is not in the call (a person or pet in their room, a phone '
           'call).')


def judge_prompt(r, prompt="v4b"):
    """(system, user) for the judge: v4b = Third Voice's tv_eval.DEC_SYS4 (scored on its test sets); v5 = DEC_SYS5."""
    if prompt == "v5":
        if len(r["people"]) == 1:
            t = re.sub(r'^"to":.*$', lambda _m: TO_1TO1, DEC_SYS5, count=1, flags=re.M)
            t = t.replace("\nYou will see", ONE_TO_ONE + "\nYou will see", 1).replace('{{"to": "group"', '{{"to": "me"')
            return tv_eval.dec_prompt(r, t)
        return tv_eval.dec_prompt(r, DEC_SYS5)
    return tv_eval.dec_prompt(r, tv_eval.DEC_SYS4)


def judge_rule(d, r, prompt="v4b"):
    if prompt != "v5" or not isinstance(d, dict):
        return tv_eval.dec_rule(d, r["target_speaker"], 2)
    w = d.get("wrong")
    if w is True or (isinstance(w, (int, float)) and not isinstance(w, bool) and w > 0):
        return True  # v4b's rule, with "to" read more carefully below
    to = str(d.get("to", "")).lower()
    if to == r["target_speaker"].lower():
        to = "me"
    ctx, cur = r["context_turns"], r["current_turn"]["text"]
    who = vocative_person(cur, [p for p in r["people"] if p != r["current_turn"]["speaker"]], r["target_speaker"])
    if who:
        to = "person"  # "Alex, I can barely hear you.": said to Alex, however the judge read it
    if (to == "me" and len(r["people"]) > 1 and r["target_speaker"].lower() not in cur.lower()
            and not any(t["speaker"] == r["target_speaker"] for t in ctx[-2:])):
        to = "group"  # in a group "you" is nearly always a person (AMI: "makes sense for you to take it from here")
    if (to == "group" and d.get("ask") is True and ctx and ctx[-1]["speaker"] == r["target_speaker"]
            and len(re.findall(r"[A-Za-z']+", cur)) <= 4 and not any(p.lower() in cur.lower() for p in r["people"])):
        to = "me"  # "why?", "how small?", "like what?" right after the AI spoke: a follow-up to it
    if to == "group":
        return d.get("ask") is True and d.get("answered") is not True and d.get("can") is True
    if to != "me":
        return False
    if len(r["people"]) == 1 and len(re.findall(r"[A-Za-z']+", cur)) > 3:
        return True  # one-to-one: they're talking to the AI; only short acknowledgements ("okay, thanks") go unanswered
    return d.get("ask") is True or d.get("reply") is True


async def decide(r, ask, samples=3, prompt="v4b"):
    """Should the AI speak after r's current turn? ask(kind, system, user, i) -> Higgs's text, kind "judge"/"check".
    Majority of the judge votes that came back (a failed request is an abstention, not a "no": under load Higgs fails
    a few percent of requests with upstream_error), OR all fact-checks that came back (at least a majority of them) say
    it contradicts a fact. Decided as soon as a majority of the judges says speak (the slowest of 6 requests set the
    pace before); the fact-checks then get 0.25 s more, as they only say whether the reply is a correction.
    Returns (speak, {"reading": first judge reading (+ wrong), "why": the transcript panel's words})."""
    (jsys, juser), (fsys, fuser) = judge_prompt(r, prompt), tv_eval.fact_prompt(r)
    jt = [asyncio.ensure_future(ask("judge", jsys, juser, i)) for i in range(samples)]
    ft = [asyncio.ensure_future(ask("check", fsys, fuser, i)) for i in range(samples)]
    need = samples // 2 + 1  # this many "speak" votes decide it whatever the others say

    def got(t):  # a vote's JSON, None if it failed
        return tv_eval.parse_json(t.result()) if t.done() and not t.cancelled() and t.exception() is None else None
    pending = set(jt)
    while pending:
        _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        if sum(judge_rule(got(t), r, prompt) for t in jt if t.done()) >= need:
            break  # decided to speak; the fact-checks only say whether it's a correction: give them a moment
    early = bool(pending)
    _, late = await asyncio.wait(ft, timeout=0.25 if early else None)
    for t in list(pending) + list(late):
        t.add_done_callback(lambda f: f.exception())
    parsed = [p for p in map(got, jt) if isinstance(p, dict)]
    checks = [p for p in map(got, ft) if isinstance(p, dict)]
    facts_hit = [bool(p.get("contradicts")) for p in checks]
    wrong = len(facts_hit) >= need and all(facts_hit)
    if wrong and FACT_GUARD and not FACTY.search(r["current_turn"]["text"]) and not any(
            p.get("wrong") is True for p in map(got, jt) if isinstance(p, dict)):
        wrong = False  # the fact-check reacting to an earlier wrong claim, not to this line ("I'll just check it.")
    speak = early or (bool(parsed) and sum(judge_rule(p, r, prompt) for p in parsed) * 2 > len(parsed)) or wrong
    reading = dict(parsed[0] if parsed else {})
    if wrong:
        reading["wrong"] = True
    return speak, {"reading": reading, "why": reason(reading, speak), "votes": len(parsed), "checks":
                   [p.get("contradicts") for p in checks]}


BACKCHANNEL = set("""mm mmm mhm mm-hm mmhm hm hmm uh-huh uh huh uhhuh aha ah oh yeah yes yep yup ok okay right sure
cool nice wow true exactly totally ha haha lol i see got it gotcha alright all""".split())


# a short line with one of these may still state a (wrong) fact: "Sunday.", "Five minutes."
FACTY = re.compile(r"\d|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|tonight|noon|midnight|"
                   r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|"
                   r"hundred|thousand|million|percent|minutes?|hours?|days?|weeks?|euros?|dollars?|pounds?)\b", re.I)
OFFER = re.compile(r"\b(let me know|want me to|shall I|should I|would you like|do you want|if you(?:'d)? like)\b", re.I)
FILLERS = set("um umm uh uhh er erm hmm hm mm eh ah like well".split())
# words no English sentence ends on (engine.INCOMPLETE is broader: "go on", "about that" end there; it only delays)
DANGLING = set("the a an to of and but or because if my your our their than um uh".split())


def backchannel(text):
    """True for a short listener sound or acknowledgement ("Mm-hm.", "Yeah, yeah.", "Okay.", "Got it."), but not for
    one asked as a question ("Yeah?", "Right?")."""
    t = text.strip().lower()
    words = re.findall(r"[a-z][a-z'-]*", t)
    return bool(words) and len(words) <= 3 and not t.endswith("?") and all(w in BACKCHANNEL for w in words)


def fix_name(text, ai, people=()):
    """Higgs STT sometimes mishears the AI's name when it's said to get attention ("Pigs, how should we...",
    "thanks, Hicks.", "Pigs. And what should..."): a word that sounds like it (similar spelling, about as long), as
    the first word (after hey/hi/okay/so) set off by punctuation, or the last one after a comma, becomes the name."""
    from difflib import SequenceMatcher

    def sim(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def like(w):  # sounds like the AI's name, and not (more) like a person in the call ("Eva" when the AI is "Ava")
        r = sim(w, ai)
        return (w.lower() != ai.lower() and abs(len(w) - len(ai)) <= 2 and r >= 0.6
                and not any(sim(w, p) >= r for p in people))

    m = re.match(r"^((?:hey|hi|hello|okay|ok|so|and|um|uh)[,]?\s+)?([A-Za-z']+)(?=[,.!?])", text, re.I)
    if m and like(m.group(2)):
        text = text[:m.start(2)] + ai + text[m.end(2):]
    m = re.search(r",\s*([A-Za-z']+)([.?!]*)\s*$", text)
    if m and like(m.group(1)):
        text = text[:m.start(1)] + ai + text[m.end(1):]
    w = re.findall(r"[A-Za-z']+", text)
    if 2 <= len(w) <= 3 and like(w[0]) and text.lstrip().startswith(w[0]):  # "Pigs stop."
        text = text.replace(w[0], ai, 1)
    return text


def join_pieces(first, rest):
    """Join a line's two transcribed pieces. STT ends every piece with punctuation; if the first stopped mid-sentence
    ("I think the judges will." + "Want to see it running live?") it was one sentence."""
    from mmhm.engine import unfinished
    if re.search(r"[.?!]$", first) and unfinished(first[:-1]) and rest[:1].isupper() and not rest.startswith("I "):
        return f"{first[:-1]} {rest[:1].lower()}{rest[1:]}"
    return f"{first} {rest}"


def prefilter(text, ai, history, people=()):
    """Before judging a transcribed line: (text with a misheard name repaired, kind). kind "skip": nothing to judge
    ("...", or a hum transcribed in another script: "嗯。"); "backchannel": an acknowledgement nobody needs answered
    (unless it answers a question the AI just asked: "Want me to go on?" "Yeah."); None: judge it."""
    text = fix_name(text, ai, people)
    if not re.search(r"[A-Za-z]", text):
        return text, "skip"
    bare = re.sub(rf"^{re.escape(ai)}[,.!]?\s*|[,.!]?\s*{re.escape(ai)}[.!]*$", "", text.strip(), flags=re.I)
    if STOP_ASK.match(bare):
        return text, "stop"  # "Higgs, stop.", "hold on": wants quiet, not an answer (barge-in already stopped it)
    last = history[-1]["text"] if history and history[-1]["speaker"] == ai else ""
    answers_me = last.rstrip().endswith("?") or bool(OFFER.search(last))  # "Want me to go on?" / "Let me know if..."
    if backchannel(text) and not answers_me:
        return text, "backchannel"
    words = re.findall(r"[a-z][a-z'-]*", text.lower())
    if (len(people) > 1 and len(words) <= 2 and not text.rstrip().endswith("?") and ai.lower() not in text.lower()
            and not last and not FACTY.search(text)):  # right after the AI spoke, "why" is a follow-up
        return text, "fragment"   # "I think.", "So, um.": in a group, a scrap, not a line to judge or to drop replies for
    if all(w in FILLERS or w in INCOMPLETE for w in words) and not text.rstrip().endswith("?"):
        return text, "filler"      # "um", "so the, uh, the": still thinking
    if words and words[-1] in DANGLING and not text.rstrip().endswith("?"):
        return text, "unfinished"  # "so basically what i'm trying to": stopped mid-sentence (they may go on: merge)
    return text, None


REPEAT_ASK = re.compile(r"\b(say (that|it) again|repeat (that|it|yourself)|come again|what did you (just )?say|"
                        r"(didn't|did not|couldn't) (catch|hear|get) (that|it|you)|pardon)\b", re.I)
VOLUNTEERS = re.compile(r"\bI(?:'ll| will| can| could)\s+(?:handle|take care of|take on|work on|do the|build|code|"
                        r"implement|set up|write the|own the)\b", re.I)
STOP_ASK = re.compile(r"^(?:(?:ok(?:ay)?|hey|um|uh)[,.]?\s+)?(?:stop(?: talking)?|wait|hold on|hang on|shh+|quiet|be quiet|"
                      r"enough|one (?:sec|second|moment))[.!]*$", re.I)
WRITER = REPLY_SYS2
FACT_GUARD = os.environ.get("MEET_FACT_GUARD", "0") == "1"
HEDGE_S = 0.5


def facts_relevant(text, context, facts, names):
    """Does the line (or the two turns before it) share a content word with the fact sheet? Names don't count."""
    import retrieve
    q = retrieve.toks(" ".join([t["text"] for t in context] + [text])) - retrieve.toks(" ".join(names))
    f = retrieve.toks(" ".join(facts)) - retrieve.toks(" ".join(names))
    return bool(q & f)  # a draft still not done this long after the decision gets a second, identical request; first one wins


async def hedged(task, again, delay):
    """task's result, or that of again() started after `delay` s if it finishes first (Higgs latency has a long tail:
    2.6-3.3 s drafts in the scenario runs, against ~0.5 s usually). An empty result waits for the other one."""
    done, _ = await asyncio.wait({task}, timeout=delay)
    if done:
        return task.result()
    other = asyncio.ensure_future(again())
    pending = {task, other}
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            if t.exception() is None and t.result():
                for p in pending:
                    p.add_done_callback(lambda f: f.exception())
                return t.result()
    return ""


ECHO_DB = float(os.environ.get("MEET_ECHO_DB", "25"))  # quieter than Higgs's output by this much while it talks: echo


def first_sentence(text, complete_only=False):
    """The text up to the end of its first sentence of 4+ words ("Sure." alone is too short). complete_only: None until
    that sentence has ended (for streaming)."""
    for m in re.finditer(r"[.!?](?=\s|$)", text):
        head = text[:m.end()]
        if len(re.findall(r"[A-Za-z0-9']+", head)) >= 4:
            return head
    return None if complete_only else text


VOCATIVE_WAIT = 0.9  # silence after a line that only calls the AI before answering it


def vocative_person(text, people, ai):
    """The person a line is addressed to by name, as its first word(s) or after a final comma ("Alex, can you...",
    "Hey Priya, ...", "..., Sam?"), unless the AI is named too. None if no one is."""
    if re.search(rf"\b{re.escape(ai)}\b", text, re.I):
        return None
    for p in people:
        n = re.escape(p)
        if (re.match(rf"^\s*(?:(?:hey|hi|ok|okay|so|and|um|uh|oh|well)[,]?\s+)?{n}\b\s*[,.!?]", text, re.I)
                or re.search(rf",\s*{n}\s*[.?!]*\s*$", text, re.I)):
            return p
    return None


def vocative_only(text, ai):
    """"Hey, Higgs." / "Higgs?" / "Hi Higgs!": the AI's name with at most a greeting word."""
    rest = [w for w in re.findall(r"[a-z']+", text.lower()) if w != ai.lower()]
    return ai.lower() in text.lower() and all(w in ("hey", "hi", "hello", "ok", "okay", "um", "so", "oh", "yo") for w in rest)


def frame_db(pcm):
    x = pcm.astype(np.float32) / 32768
    return float(20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9))


class Spares:
    """Turn-taking model processes loaded ahead of time. Loading one takes seconds (longer on a busy machine), and
    until it's ready a person's lines end on the 0.8 s fallback instead of 0.3 s: in a loaded test run, 29 of 48 lines.
    New listeners (a join, a mode switch) take a ready one; a replacement starts loading at once. Processes are
    started off the event loop: starting one blocks for a while, and a 3-person room starting 6 held up every room's
    20 ms clock by seconds (dropped audio, shredded transcripts)."""

    def __init__(self, n):
        self.n, self.ready, self.filling = n, deque(), False

    @staticmethod
    def _spawn():
        from mmhm.models import ProcSignals
        return ProcSignals(vap=True, bcdet=True, bc=False, wait=False)

    def fill(self):
        """Top up the spares in a background thread (no-op if already doing so)."""
        if self.filling or len(self.ready) >= self.n:
            return
        self.filling = True

        def work():
            try:
                while len(self.ready) < self.n:
                    self.ready.append(self._spawn())
            finally:
                self.filling = False
        threading.Thread(target=work, daemon=True).start()

    async def take(self):
        s = next((x for x in list(self.ready) if x.ready.is_set()), None) or (self.ready[0] if self.ready else None)
        if s is not None:
            self.ready.remove(s)
        else:
            s = await asyncio.to_thread(self._spawn)
        self.fill()
        return s


SPARES = Spares(int(os.environ.get("MEET_SPARE_SIGNALS", "2")))
if SIGNALS:  # import torch + MaAI now, at server start: done during the first call it stalled every room for ~5 s
    import mmhm.models  # noqa: F401


def to16(pcm):
    return to16k(pcm.astype(np.float32) / 32768)


class Pool:
    """Pre-opened Realtime text sessions with fixed instructions (third_voice/live_demo.py), plus a close() for when the
    roster (part of the instructions) changes. A session answers one request, then its two items (the request and
    the answer) are deleted and, once Higgs confirms both deletions, it goes back to the pool for the next one: the
    conversation stays empty, so no request sees another's (checked: a deleted "my name is ..." is forgotten). Before,
    every judged line opened and closed ~9 sessions."""

    MAX_AGE = 240.0  # Higgs closes a session after 5 min without user activity (session.idle_timeout)

    def __init__(self, instructions, size):
        self.instructions, self.q, self.closed, self.size = instructions, asyncio.Queue(), False, size
        self.opens = 0
        for _ in range(size):
            self._refill()

    def _new(self):
        self.opens += 1
        return Session({"instructions": self.instructions, "output_modalities": ["text"]})

    def _refill(self):
        async def open_one():
            for k in range(8):  # an open can fail under load (upstream errors, rate limits, 1013): keep trying
                if self.closed:
                    return
                s = self._new()
                try:
                    await s.__aenter__()
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(min(10.0, 1.0 + 2 * k))
                    continue
                s.opened_at = time.perf_counter()
                await self._put(s)
                return
        asyncio.ensure_future(open_one())

    async def _put(self, s):
        if self.closed or self.q.qsize() >= self.size:
            await close(s)
        else:
            await self.q.put(s)

    async def _take(self):
        """A live pre-opened session, or a freshly opened one if none is ready within a few seconds."""
        while True:
            try:
                s = await asyncio.wait_for(self.q.get(), 3.0)
            except asyncio.TimeoutError:
                s = self._new()
                await asyncio.wait_for(s.__aenter__(), 20)
                s.opened_at = time.perf_counter()
                return s
            if s.close_code is None and time.perf_counter() - s.opened_at < self.MAX_AGE:
                return s
            asyncio.ensure_future(close(s))  # idle-closed or about to be: replace it, take the next one
            self._refill()

    async def _recycle(self, s, since):
        """Delete what this request added and put the session back; close it if Higgs doesn't confirm in 3 s."""
        try:
            ids = list(dict.fromkeys(e["item"]["id"] for t, e in s.events if t >= since and e.get("type") ==
                                     "conversation.item.added" and (e.get("item") or {}).get("id")))
            for i in ids:
                await s.send({"type": "conversation.item.delete", "item_id": i})
            for i in ids:
                await s.wait(lambda e, i=i: e.get("type") == "conversation.item.deleted" and e.get("item_id") == i,
                             timeout=3, since=since)
            if ids and s.close_code is None and time.perf_counter() - s.opened_at < self.MAX_AGE:
                await self._put(s)
                return
        except Exception:  # noqa: BLE001
            pass
        await close(s)
        self._refill()

    async def ask(self, user, retries=0, on_delta=None):
        """Higgs's text answer to `user`, "" if it failed (`retries` more attempts on a failure or empty answer).
        on_delta(text) gets the first attempt's text as it streams in."""
        for attempt in range(retries + 1):
            if not spend():
                return ""
            s, ok = None, False
            stream = on_delta if attempt == 0 else None

            def done(ev):
                if stream and ev.get("type") == "response.output_text.delta":
                    stream(ev.get("delta", ""))
                return ev.get("type") == "response.done"
            try:
                s = await self._take()
                since = s.now()
                await s.say_text(user)
                _, ev = await s.wait(done, timeout=30, since=since)
                ok = ev["response"].get("status") in (None, "completed")
                text = s.transcript(ev["response"]["id"]).strip()
                if text:
                    return text
            except Exception:  # noqa: BLE001  (a failed vote is an abstention, see decide())
                pass
            finally:
                if s is not None:
                    if ok:
                        asyncio.ensure_future(self._recycle(s, since))
                    else:  # a failed or timed-out response: don't reuse that session
                        asyncio.ensure_future(close(s))
                        self._refill()
        return ""

    async def close(self):
        self.closed = True
        while not self.q.empty():
            await close(self.q.get_nowait())


class DefaultBrain:
    """Plain Higgs: one Realtime session hears the mix of everyone and does exactly what it does out of the box."""

    def __init__(self, agent):
        self.a = agent
        self.engine = Engine(mode="server_vad", instructions=agent.persona(), voice=VOICE)
        self.k = 0  # engine.log entries handled

    async def start(self):
        spend()
        await self.engine.__aenter__()
        for h in self.a.history[-14:]:  # switched mid-call: give it the meeting so far
            await self.engine.session.say_text(f"{h['speaker']}: {h['text']}", create=False)

    async def stop(self):
        await close(self.engine.session)

    def dead(self):
        return self.engine.session.close_code is not None

    async def tick(self, frames):
        mix = np.zeros(N, np.int32)
        for f in frames.values():
            mix += f
        out = await self.engine.tick(np.clip(mix, -32768, 32767).astype(np.int16))
        for e in self.engine.log[self.k:]:
            kind = e["kind"]
            if kind == "user_text" and (e.get("text") or "").strip():
                # into the shared history too (who said it is unknown: one mixed stream), so that after a switch
                # the improved brain knows what was asked ("What did Priya just ask you?" got invented otherwise)
                self.a.history.append({"speaker": "someone in the room", "text": e["text"].strip()})
                await self.a.line(None, "the room", e["text"].strip(), {"speak": True, "why": "default Higgs answers every turn"})
            elif kind == "respond":
                spend()
                await self.a.status("speaking")
            elif kind == "stop":
                await self.a.room.ai_flush()
                await self.a.status("listening", stopped=e.get("why"))
            elif kind == "response_done" and e.get("text"):
                self.a.history.append({"speaker": self.a.ai, "text": e["text"]})
                await self.a.room.event({"type": "ai_say", "text": e["text"]})
            elif kind == "play_end":
                await self.a.status("listening")
        self.k = len(self.engine.log)
        return out if self.engine.state == "playing" else None


class Listener:
    """One person in improved mode: their own transcription session, VAD, turn-taking models and gates."""

    def __init__(self, brain, slot, name):
        self.b, self.slot, self.name = brain, slot, name
        self.vad, self.p = VAD(), brain.policy
        self.gate, self.barge = floor.LineGate(LINE_POLICY), floor.BargeIn(self.p)
        self.speaking, self.p_now, self.p_bc = False, None, None
        self.preroll, self.streaming, self.sent = deque(maxlen=10), False, 0
        self.merge = False       # they resumed right after a commit: join the next piece to the last line
        self.signals = None
        self.db, self.echo_frames = -90.0, 0
        self.run, self.quiet = 0.0, 99.0
        self.in_flight = 0       # committed lines not yet transcribed and judged
        self.last_voice = 0.0    # perf_counter of their last voiced frame
        self.session, self.open, self.stopped = None, False, False
        self.backlog = []        # audio to send once a (re)opened transcription session is up
        self.commit_waiting = None  # a line ended while the session was down: commit it after the backlog
        self.opener = None

    def _new_session(self):
        tr = {"model": "higgs-stt-3.1", **({"language": LANG} if LANG else {})}
        return Session({"instructions": "Transcription only.", "output_modalities": ["text"],
                        "audio": {"input": {"turn_detection": None, "transcription": tr}}})

    async def start(self):
        self._reopen()
        if SIGNALS:
            self.signals = await SPARES.take()

    def _reopen(self):
        """(Re)open the transcription session in the background, retrying: an open can fail under load, and Higgs
        closes a session after 5 minutes without speech from this person (session.idle_timeout)."""
        if self.opener and not self.opener.done():
            return
        self.open = False

        async def go():
            k = 0
            while not self.stopped:
                s = self._new_session()
                try:
                    await s.__aenter__()
                except Exception:  # noqa: BLE001
                    k += 1
                    await asyncio.sleep(min(10.0, 1.0 + 2 * k))
                    continue
                if self.stopped:
                    await close(s)
                    return
                old, self.session = self.session, s
                if old is not None:
                    asyncio.ensure_future(close(old))
                backlog, self.backlog = self.backlog, []
                for f in backlog:
                    await s.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(f.tobytes()).decode()})
                self.open = True
                if self.commit_waiting is not None:
                    t_end, self.commit_waiting = self.commit_waiting, None
                    asyncio.ensure_future(self._transcribe(t_end))
                return
        self.opener = asyncio.ensure_future(go())

    async def stop(self):
        self.stopped = True
        if self.opener:
            self.opener.cancel()
        if self.signals:
            self.signals.close()
        self.open = False
        if self.session is not None:
            await close(self.session)

    async def _append(self, pcm):
        self.sent += 1
        if self.open and self.session.close_code is None:
            try:
                await self.session.send({"type": "input_audio_buffer.append",
                                         "audio": base64.b64encode(pcm.tobytes()).decode()})
                return
            except Exception:  # noqa: BLE001  (closed under us: reopen below)
                pass
        if len(self.backlog) < 1500:  # up to 30 s of speech waits for the new session
            self.backlog.append(pcm)
        self._reopen()

    async def tick(self, t, pcm, others, echo_db=None):
        """echo_db: Higgs's loudest recent output frame (dBFS) while it talks, else None. A sound from this person more
        than ECHO_DB below it is Higgs's own voice coming back through their speakers (a leaky echo canceller), not
        them: it neither interrupts Higgs nor becomes a line."""
        self.speaking, self.db = self.vad(pcm)
        if self.speaking and echo_db is not None and self.db < echo_db - ECHO_DB:
            self.speaking = False
            self.echo_frames += 1
        if self.speaking:
            self.last_voice = time.perf_counter()
        # continuous speech (gaps under 0.25 s don't break it): a "mm" is short, taking the floor isn't
        self.quiet = 0.0 if self.speaking else self.quiet + TICK
        self.run = self.run + TICK if self.speaking else (self.run if self.quiet < 0.25 else 0.0)
        if self.signals is not None and self.signals.ready.is_set():
            u16 = to16(pcm) * (1.0 if self.speaking else 0.05)  # noise gate, as in engine.tick
            sig = self.signals.push(u16, to16(others))
            self.p_now, self.p_bc = sig.get("p_now"), sig.get("p_bcdet")
        if self.open and self.session.close_code is not None:
            self._reopen()  # idle timeout / max duration / network: what they say next waits in the backlog
        if self.speaking and not self.streaming:  # only speech (plus 200 ms before it) goes to transcription
            self.streaming = True
            for f in self.preroll:
                await self._append(f)
            self.preroll.clear()
        if self.streaming:
            await self._append(pcm)
        else:
            self.preroll.append(pcm)
        for act in self.gate.update(t, self.speaking, self.p_now):
            if act == "commit" and self.sent >= int(self.p.min_turn / TICK):
                self.streaming, self.sent = False, 0
                self.in_flight += 1  # until on_line has decided it (a reply waiting for the floor checks this)
                self.t_speech_end = time.perf_counter() - self.gate.silence
                self.commit_info = {"why": self.gate.why, "p_now": None if self.p_now is None else round(self.p_now, 2),
                                    "lag": self.signals.lag() if self.signals is not None else None}
                if self.open:
                    asyncio.ensure_future(self._transcribe(time.perf_counter()))
                else:
                    self.commit_waiting = time.perf_counter()
            elif act == "resume":
                self.merge = True
                self.b.on_resume(self)

    async def _transcribe(self, t_end):
        self.decided_early = False
        try:
            await self._transcribe_and_decide(t_end)
        finally:
            if not self.decided_early:
                self.in_flight = max(0, self.in_flight - 1)

    async def _transcribe_and_decide(self, t_end):
        if not spend():
            return
        s = self.session
        since = s.now()
        try:
            await s.send({"type": "input_audio_buffer.commit"})
            _, ev = await s.wait("input_audio_buffer.committed", timeout=10, since=since)
            item = ev.get("item_id")
            _, ev = await s.wait(lambda e: e.get("type") == "conversation.item.input_audio_transcription.completed"
                                 and e.get("item_id") == item, timeout=30, since=since)
        except Exception as e:  # noqa: BLE001
            await self.b.a.room.event({"type": "ai_error", "error": f"transcription ({self.name}): {e!r}"[:200]})
            return
        text = (ev.get("transcript") or "").strip()
        self.t_transcript = time.perf_counter()
        if text:
            merge, self.merge = self.merge, False
            await self.b.on_line(self, text, t_end, merge)


class Reply:
    """A reply being voiced (TTS 3, sentence by sentence). Given `text` it is complete at once; given None, its text
    streams in through feed() (spoken while it arrives) until close()."""

    def __init__(self, line_id, listener, line_text, text, t_end, one=False, ai=None):
        self.line_id, self.listener, self.line_text, self.t_end = line_id, listener, line_text, t_end
        self.text, self.closed = "", False
        self.one = one           # only the first sentence (Avatar on: its render waits for the whole speech)
        self.label, self.head = (f"{ai}:" if ai else None), ""  # the writer sometimes starts with "Higgs:"
        self.vocative = False    # its line only called the AI ("Hey, Higgs."): the request may follow a pause
        self.addressed = False   # its line named the AI (or a one-to-one call): owed an answer despite side chatter
        self.audio, self.done, self.pos, self.hist_idx = bytearray(), False, 0, None
        self.video = True        # False while its Avatar render hasn't sent the first chunk (audio waits for it)
        self.render = None       # Avatar render task, started as soon as the speech is done (before the floor opens)
        self.vbuf = []           # its video chunks, held until the reply is released to the room
        self.released = False    # the floor opened: chunks go to everyone as they come
        self.vstarted = False    # avatar_start sent
        self.cancel = False
        self.started = False     # first audio frame sent
        self.marks = {"created": time.perf_counter()}  # when each stage finished (perf_counter), for timing logs
        self.speaker = SentenceSpeaker(VOICE, self.audio, lambda: setattr(self, "done", True))
        if text is not None:
            self.feed(text)
            self.close()

    def feed(self, delta):
        if self.closed:
            return
        if self.label is not None:  # hold the first characters back until we know they aren't "Higgs:"
            self.head += delta
            if len(self.head) < len(self.label) + 1 and not self.head.endswith((".", "!", "?")):
                return
            delta, self.head = re.sub(rf"^\s*{re.escape(self.label)}\s*", "", self.head, flags=re.I), ""
            self.label = None
        if self.one:
            cut = first_sentence(self.text + delta, complete_only=True)
            if cut is not None:
                self.speaker.feed(cut[len(self.text):])
                self.text = cut
                self.close()
                return
        self.text += delta
        self.speaker.feed(delta)

    def close(self):
        if self.head and not self.closed:  # a reply shorter than the label check
            h, self.head, self.label = self.head, "", None
            self.feed(h)
        if not self.closed:
            self.closed = True
            self.text = self.text.strip().strip('"')
            t = self.text
            spend(max(1, t.count(".") + t.count("?") + t.count("!")))  # roughly one TTS request per sentence
            self.speaker.close()


class ImprovedBrain:
    """Third Voice (whether) + Mm-hm (when), per person."""

    def __init__(self, agent):
        self.a, self.policy = agent, Policy()
        self.listeners = {}
        self.pools = None
        self.samples = 3
        self.take = floor.TakeFloor(FLOOR_POLICY)
        self.t = 0.0
        self.line_no = 0          # id of the newest committed line
        self.pending = None       # Reply waiting for the floor
        self.playing = None       # Reply being heard
        self.last_out = np.zeros(N, np.int16)
        self.gain = 1.0
        self.last_listener = None
        self.recent = deque(maxlen=12)  # (time, slot, text): echo guard
        self.lab_t = 0.0
        self.voice_at = 0.0       # perf_counter when anyone last made a sound
        self.line_by = {}         # line id -> slot of who said it
        self.out_db = deque([-90.0], maxlen=30)  # Higgs's output level over the last 0.6 s (echo gate)

    def point(self):
        a = self.a
        return {"target_speaker": a.ai, "setting": a.setting, "people": a.names(), "facts": a.facts,
                "context_turns": [], "current_turn": {"speaker": "", "text": ""}}

    async def _rebuild_pools(self):
        old, p = self.pools, self.point()
        n = self.samples = samples_for(len(p["people"]))
        self.pools = {"judge": Pool(judge_prompt(p, PROMPT)[0], n),
                      "check": Pool(tv_eval.fact_prompt(p)[0], n),
                      "write": Pool(WRITER.format(ai=p["target_speaker"], setting=p["setting"],
                                                  people=", ".join(p["people"]) or "the group",
                                                  facts=" ".join(p["facts"])), 2),
                      # small talk / greetings get a writer without the fact sheet: it recited it otherwise
                      "write0": Pool(WRITER.format(ai=p["target_speaker"], setting=p["setting"],
                                                   people=", ".join(p["people"]) or "the group",
                                                   facts="nothing specific beyond general knowledge."), 1)}
        if old:
            for pool in old.values():
                await pool.close()

    async def start(self):
        await self.on_roster(self.a.room.people())

    def dead(self):
        return False  # each listener and pool reconnects on its own

    async def stop(self):
        if self.pools:
            await self.a.room.event({"type": "pool_stats", "opens": {k: p.opens for k, p in self.pools.items()},
                                     "lines": self.line_no})
        for li in list(self.listeners.values()):
            await li.stop()
        self.listeners.clear()
        if self.pools:
            for pool in self.pools.values():
                await pool.close()
        self._cancel_pending()

    async def on_roster(self, people):
        now = {p["slot"]: p["name"] for p in people}
        for slot in [s for s in self.listeners if s not in now]:
            await self.listeners.pop(slot).stop()
        for slot, name in now.items():
            if slot not in self.listeners:
                li = Listener(self, slot, name)
                self.listeners[slot] = li
                spend()
                asyncio.ensure_future(li.start())
        await self._rebuild_pools()

    def on_resume(self, li):
        """The same person went on right after a commit: a reply to their half-line is stale."""
        if self.pending and self.pending.listener is li:
            self._cancel_pending()

    def moved_on(self, r):
        """Has the conversation moved past r's line? Any newer line does, except that a question put to the AI by name
        (or in a one-to-one call) is still owed an answer while other people chat: only the asker going on, or a newer
        reply, replaces it (in a busy AMI meeting every answer to a direct question was dropped otherwise)."""
        if r.line_id == self.line_no:
            return False
        if not r.addressed:
            return True
        return any(self.line_by.get(i) == r.listener.slot for i in range(r.line_id + 1, self.line_no + 1))

    def _cancel_pending(self):
        if self.pending:
            self.pending.cancel = True  # also stops its Avatar render (one at a time per API key)
            self.pending.speaker.cancel()
            self.pending = None

    async def on_line(self, li, text, t_end, merge):
        a = self.a
        if not re.search(r"[A-Za-z]", text):
            return  # "...", or a hum transcribed in another script ("嗯。")
        if any(time.time() - t0 < 2.0 and slot != li.slot and repeats(text, [x]) for t0, slot, x in self.recent):
            return  # someone's mic picked up another participant (no headphones): ignore the echo
        self.recent.append((time.time(), li.slot, text))
        merged = merge and bool(a.history) and a.history[-1]["speaker"] == li.name
        if merged:  # they went on after a pause: one line (the transcript panel replaces the fragment)
            text = join_pieces(a.history.pop()["text"], text)
        text, kind = prefilter(text, a.ai, a.history, a.names())
        if kind == "skip":
            return
        context = list(a.history)
        cur = {"speaker": li.name, "text": text}
        a.history.append(cur)
        if kind:  # "mm-hm", "um", "so the, uh,": part of the meeting, but nothing to answer, and it doesn't make a reply
            if kind != "unfinished":  # someone else is waiting for stale; an unfinished line merges with what follows
                li.gate.settle()
            await a.line(li.slot, li.name, text, {"speak": False, "why": kind, "decided_s": 0.0}, merged)
            return
        self.line_no += 1
        line_id = self.line_no
        self.line_by[line_id] = li.slot
        self.last_listener = li
        r = {**self.point(), "meeting": a.room.id, "line": line_id, "context_turns": context, "current_turn": cur}
        if len(a.facts) > 12:  # big fact sheet: the 8 most relevant facts per line (third_voice/retrieve.py)
            import retrieve
            r["facts"] = retrieve.top_facts(r, 8)
        pools = self.pools
        ctx = "\n".join(f"{t['speaker']}: {t['text']}" for t in context[-14:])
        convo = f"Conversation so far:\n{ctx}\n{li.name}: {text}\n\nYour reply:"
        addressed = len(self.listeners) == 1 or re.search(rf"\b{re.escape(a.ai)}\b", text, re.I) is not None
        if a.avatar:  # the face renders only once the whole reply is spoken, and TTS takes about as long as the speech
            convo = "[Answer in one short sentence of at most 12 words.]\n" + convo
        # Likely a reply (a one-to-one call, the AI's name, a question): voice the draft while it streams in and the
        # judge decides (~0.5 s sooner). Otherwise draft only; a silent line costs no speech.
        spec = None
        if SPECULATE and (len(self.listeners) == 1 or a.ai.lower() in text.lower() or text.rstrip().endswith("?")):
            spec = Reply(line_id, li, text, None, t_end, one=a.avatar, ai=a.ai)
        writer = pools["write"] if facts_relevant(text, context[-2:], a.facts, [a.ai] + a.names()) else pools["write0"]
        draft = asyncio.ensure_future(writer.ask(convo, retries=1, on_delta=spec.feed if spec else None))
        # both drafts start with the decision, so a correction costs no extra wait (rewriting it after: ~1 s)
        cdraft = asyncio.ensure_future(pools["write"].ask(CORRECT.format(ai=a.ai) + convo, retries=1)) if a.facts else None
        speak, d = await decide(r, lambda kind, _sys, user, _i: pools[kind].ask(user), samples=self.samples, prompt=PROMPT)
        why = d["reading"]
        t_decided = time.perf_counter()
        li.in_flight = max(0, li.in_flight - 1)  # decided: a reply waiting on this line may go (or drop)
        li.decided_early = True
        await a.line(li.slot, li.name, text, {"speak": speak, "why": d["why"], "reading": why, "votes": d["votes"],
                                              "checks": d["checks"],
                                              "decided_s": round(time.perf_counter() - t_end, 2)}, merged)
        if not speak:
            li.gate.settle()
            if spec:
                spec.speaker.cancel()
            for f in (draft, cdraft):
                if f:
                    f.add_done_callback(lambda f: f.exception())
            return
        convo_used = convo
        if why.get("wrong") and cdraft:  # the plain draft was written without knowing why Higgs speaks
            draft, cdraft = cdraft, draft
            convo_used = CORRECT.format(ai=a.ai) + convo
            if spec:
                spec.speaker.cancel()
                spec = None
        if cdraft:
            cdraft.add_done_callback(lambda f: f.exception())
        if convo_used is not convo:
            writer = pools["write"]
        ans = (await hedged(draft, lambda: writer.ask(convo_used, retries=0), HEDGE_S)).strip().strip('"')
        t_draft = time.perf_counter()
        if VOLUNTEERS.search(ans):  # "I can handle the backend": it's an AI in a call, it can't
            if spec:
                spec.speaker.cancel()
                spec = None
            ans = (await writer.ask("[You are an AI in this call and can't do work outside it (coding, building, "
                                    "booking): don't volunteer for tasks.]\n" + convo_used, retries=1)).strip().strip('"') or ans
            t_draft = time.perf_counter()
        ans = re.sub(rf"^\s*{re.escape(a.ai)}\s*:\s*", "", ans, flags=re.I)  # "Higgs: Judging is ..."
        if a.avatar:  # one sentence with the face on: the render waits for the whole reply's speech
            ans = first_sentence(ans).strip().strip('"')
        if spec and (not ans or spec.text.strip().strip('"') != ans):  # the stream failed and a retry answered
            spec.speaker.cancel()
            spec = None
        mine = [h["text"] for h in a.history if h["speaker"] == a.ai][-3:]
        if REPEAT_ASK.search(text):
            mine = []  # "say that again?": repeating is the answer
        if ans and repeats(ans, mine):  # the writer said the same thing again: once more, asking for something new
            if spec:
                spec.speaker.cancel()
                spec = None
            ans = (await writer.ask("[You already said that. Reply to the latest utterance with something new, "
                                            "briefly.]\n" + convo, retries=1)).strip().strip('"')
            t_draft = time.perf_counter()
        skip = ("no draft (Higgs request failed)" if not ans else "repeat" if repeats(ans, mine)
                else "someone spoke since" if line_id != self.line_no and not (addressed and not any(
                    self.line_by.get(i) == li.slot for i in range(line_id + 1, self.line_no + 1))) else None)
        if skip:  # the next line's decision sees this one in context
            if spec:
                spec.speaker.cancel()
            await a.room.event({"type": "reply_skipped", "line": text[:80], "why": skip})
            return
        self._cancel_pending()
        if spec:
            spec.close()
            self.pending = spec
        else:
            self.pending = Reply(line_id, li, text, ans, t_end)
        self.pending.vocative = vocative_only(text, a.ai)
        self.pending.addressed = addressed
        if a.avatar:  # render ahead, into a buffer: by the time the floor opens the face is (nearly) ready
            self.pending.video = False
            self.pending.render = asyncio.ensure_future(self._render(self.pending))
        self.pending.marks.update(gate=getattr(li, "commit_info", None), speech_end=getattr(li, "t_speech_end", t_end), commit=t_end,
                                  transcript=getattr(li, "t_transcript", t_end), decided=t_decided, draft=t_draft,
                                  speculated=spec is not None)
        await a.status("thinking")

    async def tick(self, frames):
        t = self.t = self.t + TICK
        total = np.zeros(N, np.int32) + self.last_out
        # Higgs's voice can come back through a person's speakers 0.1-0.5 s later: compare with its recent loudest frame
        self.out_db.append(frame_db(self.last_out))
        echo_db = max(self.out_db) if max(self.out_db) > -60 else None
        for f in frames.values():
            total += f
        for slot, li in list(self.listeners.items()):
            pcm = frames.get(slot, np.zeros(N, np.int16))
            others = np.clip(total - pcm, -32768, 32767).astype(np.int16)  # VAP's ch2: everyone but this person
            await li.tick(t, pcm, others, echo_db)
        await self._lab(t)
        anyone = any(li.speaking for li in self.listeners.values())
        self.take.tick(t, anyone)
        if anyone:
            self.voice_at = time.perf_counter()

        r = self.pending
        if r is not None and "audio" not in r.marks and len(r.audio) >= N * 2:
            r.marks["audio"] = time.perf_counter()
        if self.playing is None and r is not None and r.closed and len(r.audio) >= N * 2:
            li = r.listener
            act, why = self.take.may_play(t, {"p_now": li.p_now, "silence": li.gate.silence}, anyone, r.line_text,
                                          self.moved_on(r))
            holder = next((x for x in self.listeners.values() if x is not li and x.gate.silence < FLOOR_POLICY.max_wait
                           and (x.p_now is None or x.p_now >= FLOOR_POLICY.th_play)), None)
            if act == "play" and holder is not None and not r.addressed:  # asked by name: a 0.35 s gap will do
                act, why = "wait", f"{holder.name} may go on"  # someone else just spoke and still holds the floor
            # someone else spoke after this line and what they said isn't judged yet (it may answer it: "Saturday at
            # three"): wait for that verdict; a substantive line then makes this reply stale, a "mm-hm" doesn't
            other = next((x for x in self.listeners.values() if x is not li and (x.in_flight or
                          (x.gate.speech >= LINE_POLICY.min_turn and x.last_voice > r.t_end))), None)
            if act == "play" and other is not None and not r.addressed:
                act, why = "wait", f"{other.name}'s line"
            if act == "play" and r.vocative and li.gate.silence < VOCATIVE_WAIT:
                act, why = "wait", "the request may follow"  # "Hey Higgs, [pause] quick one: ..."
            age = time.perf_counter() - r.t_end
            stale = STALE_ADDRESSED if r.addressed else STALE_S
            if act != "drop" and ((age > stale and self.voice_at > r.t_end) or age > STALE_MAX):
                act, why = "drop", "stale"  # people moved on (or it's very late); a slow reply to a quiet room still plays
            if act == "play":
                self.pending, self.playing = None, r
                r.hist_idx = len(self.a.history)
                self.a.history.append({"speaker": self.a.ai, "text": r.text})
                li.gate.settle()  # the reply is out: a later word from them is a new line, not a resume
                for li2 in self.listeners.values():
                    li2.barge.reset()
                r.play_why = why
                if self.a.avatar:  # the face has to move with the words: the audio waits for the first video chunk
                    if r.render is None:  # the avatar was switched on after this reply was decided
                        r.video = False
                        r.render = asyncio.ensure_future(self._render(r))
                    await self._release(r)
            elif act == "drop":
                self._cancel_pending()
                await self.a.status("listening", dropped=why)

        out = None
        r = self.playing
        if r is not None and not r.video and not self.a.avatar:
            r.video = True  # the avatar was switched off while this reply waited for its face: just speak
        if r is not None and not r.video:
            taker = next((li for li in self.listeners.values() if li.run >= FLOOR_POLICY.interrupt_max), None)
            if taker or self.moved_on(r):  # someone took the floor while the face rendered (a "mm" doesn't)
                r.cancel = True
                r.speaker.cancel()
                if r.hist_idx == len(self.a.history) - 1:  # never heard: not part of the meeting
                    self.a.history.pop()
                self.playing = None
                if r.vstarted:
                    await self.a.room.avatar_stop()
                await self.a.status("listening", dropped="someone spoke while the avatar rendered")
            r = None
        if r is not None:
            if not r.started:
                r.started = True
                m, now = r.marks, time.perf_counter()
                t0 = m.get("speech_end", r.t_end)
                timing = {k: round(m[k] - t0, 2) for k in ("commit", "transcript", "decided", "draft", "created", "audio",
                                                            "render", "video") if k in m}
                timing["play"] = round(now - t0, 2)
                timing["speculated"] = bool(m.get("speculated"))
                timing["gate"] = m.get("gate")
                await self.a.status("speaking", after_s=round(now - r.t_end, 2), why=r.play_why, timing=timing)
                await self.a.room.event({"type": "ai_say", "text": r.text})
            gains = []
            for li in self.listeners.values():
                res = li.barge.update(t, li.speaking, li.p_bc, p_now=li.p_now)  # p_bc is the detector's p_bcdet
                gains.append(res["gain"])
                if res["event"] == "interrupt":
                    await self._interrupt(r, li)
                    return None
                if res["event"] == "backchannel":
                    await self.a.room.event({"type": "lab_event", "slot": li.slot, "event": "backchannel"})
            g = floor.combine(gains) if gains else 1.0
            if abs(g - self.gain) > 0.05 or (g == 1.0 and self.gain != 1.0):
                self.gain = g
                await self.a.room.ai_gain(g)
            a = bytes(r.audio[r.pos * 2:(r.pos + N) * 2])
            if a:
                out = np.zeros(N, np.int16)
                out[:len(a) // 2] = np.frombuffer(a, np.int16)
                r.pos += len(a) // 2
            elif r.done:
                self.playing = None
                if r.vstarted:
                    await self.a.room.avatar_stop()
                await self.a.status("listening")
        self.last_out = out if out is not None else np.zeros(N, np.int16)
        return out

    async def _interrupt(self, r, li):
        r.cancel = True
        r.speaker.cancel()
        if r.vstarted:
            await self.a.room.avatar_stop()
        heard = floor.truncate(r.text, [(a / SR, b / SR, x) for a, b, x in r.speaker.segments], r.pos / SR)
        if r.hist_idx is not None and r.hist_idx < len(self.a.history):
            # marked, so the judge and the writer know it was cut off (unmarked, the writer resumed its old answer)
            self.a.history[r.hist_idx]["text"] = f"{heard} [interrupted]"
        self.playing = None
        self.gain = 1.0
        await self.a.room.ai_flush()
        await self.a.room.ai_gain(1.0)
        await self.a.status("listening", stopped=f"{li.name} interrupted", heard=heard)

    async def _render(self, r):
        """Render the reply through Higgs Avatar once its speech is complete. Chunks are held in r.vbuf until the
        reply is released to the room (_release), then streamed to everyone as they come."""
        while not r.done and not r.cancel:
            await asyncio.sleep(0.05)
        if r.cancel or not spend():
            r.video = True
            return
        pcm = np.frombuffer(bytes(r.audio[:len(r.audio) // 2 * 2]), np.int16)
        loop = asyncio.get_running_loop()
        r.marks["render"] = time.perf_counter()

        def on_chunk(c):
            def deliver():
                if r.cancel:
                    return
                r.marks.setdefault("video", time.perf_counter())
                if r.released:
                    asyncio.ensure_future(self._send_video(r, [c]))
                else:
                    r.vbuf.append(c)
            loop.call_soon_threadsafe(deliver)
        try:
            await asyncio.to_thread(avatar.render, pcm, on_chunk, lambda: r.cancel)
        except Exception as e:  # noqa: BLE001  (no face this time: speak anyway)
            await self.a.room.event({"type": "ai_error", "error": f"avatar: {e!r}"[:300]})
            r.video = True

    async def _send_video(self, r, chunks):
        if not r.vstarted:
            r.vstarted = True
            await self.a.room.avatar_start(avatar.MIME)
        for c in chunks:
            await self.a.room.send_avatar_chunk(c)
        r.video = True  # the audio may start: the face is moving on everyone's screen

    async def _release(self, r):
        """The floor opened for this reply: send the video rendered so far; the rest follows as it comes."""
        r.released = True
        if r.vbuf:
            chunks, r.vbuf = r.vbuf, []
            await self._send_video(r, chunks)

    async def _lab(self, t):
        if t - self.lab_t < 0.2:
            return
        self.lab_t = t
        for li in self.listeners.values():
            if li.p_now is not None:
                await self.a.room.event({"type": "lab", "slot": li.slot, "p_now": round(li.p_now, 2),
                                         "p_bc": round(li.p_bc or 0.0, 2)})


class HiggsAgent:
    """The interface room.py drives (see room.py). One per room."""

    def __init__(self, room):
        self.room = room
        if SIGNALS:
            SPARES.fill()  # the first room starts loading turn-taking models for the people who will join
        b = room.brief
        self.ai, self.setting, self.facts = b.get("ai") or "Higgs", b.get("setting") or "a meeting", list(b.get("facts") or [])
        self.history = []
        self.brain = None
        self.mode = getattr(room, "mode", "improved")
        self.inq = {}
        self.task = None
        self.running = False
        self.switching = None
        self.starting = None
        self.gen = 0             # bumped by every launch / switch / stop
        self.errors_t = 0.0      # last ai_error event from a failing tick (rate-limited)
        self.told_budget = False
        self.stats = {"dropped": 0, "late_max": 0.0, "catchup": 0, "ticks": 0, "tick_ms_sum": 0.0, "tick_ms_max": 0.0}  # input frames lost, worst lateness, catch-up ticks
        self.avatar = bool(getattr(room, "avatar", False))

    def names(self):
        return [p["name"] for p in self.room.people()]

    def persona(self):
        return PERSONA.format(ai=self.ai, setting=self.setting, people=", ".join(self.names()) or "the group",
                              facts=" ".join(self.facts))

    async def status(self, s, **kw):
        await self.room.event({"type": "ai_state", "mode": self.mode, "avatar": getattr(self.room, "avatar", False),
                               "status": s, "calls": CALLS["n"], **kw})

    async def line(self, slot, name, text, decision, merge=False):
        await self.room.event({"type": "line", "slot": slot, "name": name, "text": text, "decision": decision,
                               "mode": self.mode, **({"merge": True} if merge else {})})

    def _brain(self, mode):
        return ImprovedBrain(self) if mode == "improved" else DefaultBrain(self)

    def _launch(self):
        """Start a brain for the current mode in the background, retrying until Higgs answers (opens fail under load:
        voice validation is rate-limited per key, upstream errors). A newer launch (mode switch) or stop() supersedes
        it: a superseded brain is stopped, never installed."""
        self.gen += 1
        gen, mode = self.gen, self.mode

        async def go():
            k = 0
            while self.running and gen == self.gen:
                b = self._brain(mode)
                try:
                    await b.start()
                except Exception as e:  # noqa: BLE001  (Higgs unreachable: the humans can still talk to each other)
                    k += 1
                    try:
                        await b.stop()
                    except Exception:  # noqa: BLE001
                        pass
                    if k == 1 or k % 5 == 0:
                        await self.room.event({"type": "ai_error", "error": f"start ({mode}, try {k}): {e!r}"[:300]})
                    await self.status("connecting")
                    await asyncio.sleep(min(10.0, 2.0 * k))
                    continue
                if gen != self.gen or not self.running:
                    await b.stop()
                    return
                self.brain = b
                await self.status("listening")
                return
        self.starting = asyncio.ensure_future(go())

    async def start(self):
        self.running = True
        self.task = asyncio.ensure_future(self._run())
        self._launch()

    async def stop(self):
        st = dict(self.stats)
        st["tick_ms_mean"] = round(st.pop("tick_ms_sum") / max(1, st["ticks"]), 2)
        await self.room.event({"type": "loop_stats", **st})
        self.running = False
        self.gen += 1
        if self.task:
            self.task.cancel()
        if self.brain:
            b, self.brain = self.brain, None
            await b.stop()

    def feed(self, slot, pcm):
        # Up to 1 s of a person's audio waits for the tick loop. (It used to keep only 120 ms: with three rooms on one
        # server the loop sometimes ran late, the dropped speech reached STT in shreds, "Higgs, what do you think..."
        # -> "Pig.", and Higgs didn't answer.) A late loop catches up by running ticks back to back (_run).
        q = self.inq.setdefault(slot, deque(maxlen=50))
        if len(q) == q.maxlen:
            self.stats["dropped"] += 1
        q.append(np.frombuffer(pcm, np.int16))

    def on_roster(self, people):
        live = {p["slot"] for p in people}
        for s in [s for s in self.inq if s not in live]:
            self.inq.pop(s)
        if isinstance(self.brain, ImprovedBrain):
            asyncio.ensure_future(self.brain.on_roster(people))

    def on_chat(self, name, text):
        """Typed lines count as lines (a fallback when someone's mic is bad): judged in improved mode."""
        if isinstance(self.brain, ImprovedBrain):
            li = next((x for x in self.brain.listeners.values() if x.name == name), None)
            if li:
                asyncio.ensure_future(self.brain.on_line(li, text, time.perf_counter(), False))
        elif isinstance(self.brain, DefaultBrain):
            asyncio.ensure_future(self.brain.engine.session.say_text(f"{name}: {text}"))

    def set_mode(self, mode):
        if mode == self.mode or mode not in ("default", "improved"):
            return
        self.mode = mode
        self.switching = asyncio.ensure_future(self._switch())

    async def _switch(self):
        await self.status("switching")
        self.gen += 1  # a brain still starting for the old mode must not be installed
        old, self.brain = self.brain, None
        if old:
            await old.stop()
        await self.room.ai_flush()
        await self.room.ai_gain(1.0)
        if self.running:
            self._launch()

    def set_avatar(self, on):
        """Improved mode only: plain Realtime audio arrives in bursts and has no reply boundary to render on."""
        self.avatar = bool(on)
        if self.avatar:
            asyncio.ensure_future(self.room.avatar_idle(avatar.IDLE_URL))
        if self.avatar and self.mode == "default":
            asyncio.ensure_future(self.room.event({"type": "ai_note", "text": "The avatar works in improved mode."}))

    def _pop(self, slot):
        q = self.inq.get(slot)
        if not q:
            return np.zeros(N, np.int16)
        return q.popleft()

    async def _run(self):
        t0, k = time.perf_counter(), 0
        while self.running:
            k += 1
            frames = {slot: self._pop(slot) for slot in list(self.inq)}
            if CALLS.get("over") and not self.told_budget:
                self.told_budget = True
                await self.room.event({"type": "ai_note", "text": f"{self.ai} has used this server's Higgs request budget "
                                                                  f"(MEET_BUDGET={BUDGET}) and can't answer any more."})
            brain = self.brain
            if brain is not None and brain.dead():  # its Realtime session closed (idle timeout, max duration, network)
                await self.room.event({"type": "ai_error", "error": "Higgs session closed; reconnecting"})
                self.brain = None
                asyncio.ensure_future(brain.stop())
                self._launch()
                brain = None
            if brain is not None:
                try:
                    t_tick = time.perf_counter()
                    out = await brain.tick(frames)
                    dt = time.perf_counter() - t_tick
                    self.stats["ticks"] += 1
                    self.stats["tick_ms_sum"] += dt * 1000
                    self.stats["tick_ms_max"] = max(self.stats["tick_ms_max"], round(dt * 1000, 1))
                except Exception as e:  # noqa: BLE001  (keep the room alive; the log says what broke)
                    if time.perf_counter() - self.errors_t > 5.0:
                        self.errors_t = time.perf_counter()
                        await self.room.event({"type": "ai_error", "error": repr(e)[:300]})
                    out = None
                if out is not None:
                    await self.room.send_ai_audio(out.astype(np.int16).tobytes())
            delay = t0 + k * TICK - time.perf_counter()
            self.stats["late_max"] = max(self.stats["late_max"], round(-delay, 3))
            if max((len(q) for q in self.inq.values()), default=0) > 2:
                self.stats["catchup"] += 1
                await asyncio.sleep(0)  # audio is waiting: next tick now (let the sockets run in between)
                continue
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -0.5:  # fell far behind (a slow tick): resync instead of racing to catch up
                t0, k = time.perf_counter(), 0
