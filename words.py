"""Word rules: filler-proof barge-in (F2) and the line prefilter. Pure functions, no I/O beyond loading the list.

Barge-in verdicts on a listener's speech while the AI talks:
  "filler"     only hesitations and acknowledgements ("umm", "okay yeah", "I see"): keep talking at full volume
  "stop"       an explicit stop ("wait", "hold on", "Higgs, stop"): stop at once
  "interrupt"  enough content words to be a real turn ("what about Friday"): stop and hand over the floor
  "unclear"    not enough yet (one content word, or nothing transcribed): keep talking and listen on
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import yaml


@dataclass
class Lexicon:
    fillers: set[str] = field(default_factory=set)
    backchannels: set[str] = field(default_factory=set)
    backchannel_phrases: list[str] = field(default_factory=list)
    stop_phrases: list[str] = field(default_factory=list)
    min_content_words: int = 2
    long_speech_s: float = 1.4
    probe_first_s: float = 0.35
    probe_every_s: float = 0.3

    @classmethod
    def load(cls, path: str | Path) -> "Lexicon":
        d = yaml.safe_load(Path(path).read_text())
        bad = [w for key in ("fillers", "backchannels", "backchannel_phrases", "stop_phrases") for w in d[key]
               if not isinstance(w, str)]
        if bad:
            raise ValueError(f"{path}: quote these words, YAML read them as non-text: {bad}")
        return cls(fillers=set(d["fillers"]), backchannels=set(d["backchannels"]),
                   backchannel_phrases=sorted(d["backchannel_phrases"], key=len, reverse=True),
                   stop_phrases=sorted(d["stop_phrases"], key=len, reverse=True),
                   **{k: d[k] for k in ("min_content_words", "long_speech_s", "probe_first_s", "probe_every_s")
                      if k in d})


@lru_cache(maxsize=4)
def lexicon(path: str) -> Lexicon:
    return Lexicon.load(path)


def normalise(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' -]+", " ", t)
    t = re.sub(r"([a-z])\1{2,}", r"\1\1", t)        # "ummmm" -> "umm", "sooo" -> "soo"
    t = re.sub(r"\b(so|no|oh)o+\b", r"\1", t)        # "soo" -> "so"
    return " ".join(t.split())


def words(text: str) -> list[str]:
    return [w for w in normalise(text).replace("-", " - ").split() if w != "-"] if text else []


def _strip_name(text: str, ai: str) -> str:
    if not ai:
        return text
    return re.sub(rf"\b{re.escape(ai.lower())}\b", " ", text)


def content_words(text: str, lex: Lexicon, ai: str = "") -> list[str]:
    """Words left after removing the AI's name, acknowledgement phrases, fillers and backchannels."""
    t = " " + _strip_name(normalise(text), ai) + " "
    for phrase in lex.backchannel_phrases:
        t = t.replace(f" {phrase} ", " ")
    out = []
    for w in t.split():
        w2 = w.strip("'")
        if w2 in lex.fillers or w2 in lex.backchannels or w2.replace("-", "") in lex.backchannels:
            continue
        out.append(w2)
    return out


def is_stop(text: str, lex: Lexicon, ai: str = "") -> bool:
    """An explicit request to stop, at the start of what was said ("wait, ...", "Higgs, hold on")."""
    t = _strip_name(normalise(text), ai).strip()
    t = re.sub(r"^(?:(?:um+|uh+|oh|okay|ok|hey|no)\s+)*", "", t)
    return any(t == p or t.startswith(p + " ") for p in lex.stop_phrases)


def bargein(text: str, speech_s: float, lex: Lexicon, ai: str = "") -> str:
    """Verdict on what a listener has said so far while the AI talks (see module doc)."""
    text = (text or "").strip()
    if not words(text):                       # nothing transcribed (yet)
        return "interrupt" if speech_s >= lex.long_speech_s else "unclear"
    if is_stop(text, lex, ai):
        return "stop"
    content = content_words(text, lex, ai)
    if len(content) >= lex.min_content_words:
        return "interrupt"
    return "filler" if not content else "unclear"


def is_acknowledgement(text: str, lex: Lexicon, ai: str = "") -> bool:
    """Only fillers/backchannels and not asked as a question ("Yeah?" asks something)."""
    return bool(words(text)) and not text.rstrip().endswith("?") and not content_words(text, lex, ai)


# ------------------------------------------------------------------ the line prefilter (ported from Meet 1)
FACTY = re.compile(r"\d|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|tonight|noon|"
                   r"midnight|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|"
                   r"forty|fifty|hundred|thousand|million|percent|minutes?|hours?|days?|weeks?|months?|euros?|"
                   r"dollars?|pounds?|chapters?|room|section|page)\b", re.I)
OFFER = re.compile(r"\b(let me know|want me to|shall i|should i|would you like|do you want|if you(?:'d)? like)\b", re.I)
DANGLING = set("the a an to of and but or because if my your our their than um uh with for".split())


def fix_name(text: str, ai: str, people: list[str] = ()) -> str:
    """STT mishears the AI's name when it's said to get attention ("Pigs, how should we...", "thanks, Hicks"): a
    similar word as the first word (after hey/okay/so) set off by punctuation, or the last after a comma."""
    def skeleton(w):                       # consonants only: "Mei" and "May" both sound like "m"
        return re.sub(r"[aeiouyhw]", "", w.lower()[1:])

    def sim(a, b):
        r = SequenceMatcher(None, a.lower(), b.lower()).ratio()
        if a[:1].lower() == b[:1].lower() and skeleton(a) == skeleton(b):
            r = max(r, 0.7)
        return r

    def like(w):
        r = sim(w, ai)
        return (w.lower() != ai.lower() and abs(len(w) - len(ai)) <= 2 and r >= 0.6
                and not any(sim(w, p) >= r for p in people))

    m = re.match(r"^((?:hey|hi|hello|okay|ok|so|and|um|uh)[,]?\s+)?([A-Za-z']+)(?=[,.!?])", text, re.I)
    if m and like(m.group(2)):
        text = text[:m.start(2)] + ai + text[m.end(2):]
    m = re.search(r",\s*([A-Za-z']+)([.?!]*)\s*$", text)
    if m and like(m.group(1)):
        text = text[:m.start(1)] + ai + text[m.end(1):]
    return text


def prefilter(text: str, ai: str, last_ai_text: str, lex: Lexicon, people: list[str] = ()) -> tuple[str, str | None]:
    """(text with the AI's name repaired, kind). kind None = a line to judge; otherwise it is shown and logged but
    not answered: "skip" (no words), "stop", "backchannel" (unless it answers the AI's question), "filler",
    "fragment" (a group scrap: <= 2 words, no question/name/number), "unfinished" (ends mid-sentence)."""
    text = fix_name(text, ai, list(people))
    for person in people:                      # "May, did you..." -> "Mei, did you..."
        text = fix_name(text, person, [ai] + [p for p in people if p != person])
    if not re.search(r"[A-Za-z]", text):
        return text, "skip"
    if is_stop(text, lex, ai) and len(content_words(text, lex, ai)) <= 2:
        return text, "stop"
    answers_ai = last_ai_text.rstrip().endswith("?") or bool(OFFER.search(last_ai_text))
    ws = words(text)
    if is_acknowledgement(text, lex, ai) and not answers_ai:
        named = ai.lower() in normalise(text)
        if not named:
            toks = [w.strip("'") for w in normalise(text).split()]
            return text, "backchannel" if any(w in lex.backchannels for w in toks) else "filler"
    if (len(people) > 1 and len(ws) <= 2 and not text.rstrip().endswith("?") and ai.lower() not in text.lower()
            and not last_ai_text and not FACTY.search(text)):
        return text, "fragment"
    if ws and ws[-1] in DANGLING and not text.rstrip().endswith("?"):
        return text, "unfinished"
    return text, None
