"""Contradiction monitor (F4): runs beside the conversation, never in a reply's critical path.

Each claim-like clause (not a question, with a number/date/name or enough words) is checked against three kinds of
evidence: the fact sheet (F1..), document excerpts from the knowledge base (D1..) and earlier statements by
people in the meeting (S1..). The checker must quote its evidence; a verdict only becomes an interruption when
  * it says "contradicts" with confidence >= fact_min_confidence (higher for speaker-vs-speaker),
  * the evidence id exists and the quote really appears in that item (a model opinion with no evidence never
    takes the floor: the Meet 1 audit found "verified" corrections that no checker had verified),
  * it is not a repeat of a correction already made, and the cooldown has passed.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from third_voice import parse_json
from words import FACTY

CHECK_SYS = """You check one statement made in a live meeting ({setting}) against evidence, to catch mistakes the group might act on.
Evidence items have ids: F = fact sheet, D = document excerpt, S = something a participant said earlier.
Decide whether the LATEST statement contradicts an evidence item about the same specific thing (a different amount, date, day, time, number, name, place, owner, rule or yes/no fact).
Not contradictions: rounding, rewording, extra or missing detail, plans, guesses explicitly hedged ("maybe", "I think it might"), opinions, questions, hypotheticals, jokes, and changes someone explicitly announces ("we moved it to Friday").
Output ONLY one JSON object:
{{"contradicts": true or false, "evidence_id": "F2" or "D1" or "S3" or "", "evidence": "<the exact words copied from that evidence item>", "confidence": <0.0-1.0>, "correction": "<one short, polite spoken sentence to {speaker} that states what the evidence says and where it comes from, e.g. 'Sorry to jump in, {speaker} — the lease says rent is due on the 1st, not the 5th.'>"}}
If unsure, output contradicts false."""


@dataclass
class Claim:
    id: str
    speaker: str
    text: str
    t: float


@dataclass
class Verdict:
    contradicts: bool
    evidence_id: str = ""
    evidence: str = ""
    confidence: float = 0.0
    correction: str = ""
    source: str = ""          # human-readable: "lease.pdf p.3" / "fact sheet" / "Jordan, earlier"
    kind: str = ""            # "fact" | "document" | "speaker"
    reason: str = ""          # why it did or didn't become an interruption


def claim_like(text: str) -> bool:
    t = text.strip()
    if not t or t.endswith("?"):
        return False
    words = re.findall(r"[A-Za-z0-9']+", t)
    if len(words) < 3:
        return bool(FACTY.search(t)) and len(words) >= 2
    return bool(FACTY.search(t)) or len(words) >= 6


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9$€£.%]+", s.lower()))


def quote_found(quote: str, source: str) -> bool:
    """Does the checker's quote really come from the source? Exact substring, or a close fuzzy match."""
    q, s = _norm(quote), _norm(source)
    if len(q) < 4:
        return False
    if q in s:
        return True
    m = SequenceMatcher(None, q, s, autojunk=False).find_longest_match(0, len(q), 0, len(s))
    if m.size >= 0.8 * len(q):
        return True
    qt, st = set(q.split()), set(s.split())
    return len(qt) >= 3 and len(qt & st) / len(qt) >= 0.85


@dataclass
class Monitor:
    min_confidence: float = 0.8
    cooldown_s: float = 20.0
    claims: list[Claim] = field(default_factory=list)
    corrected: set[str] = field(default_factory=set)
    last_correction: float = -1e9
    checks: int = 0

    def remember(self, claim: Claim) -> None:
        self.claims.append(claim)
        del self.claims[:-60]

    def evidence(self, kb, speaker: str, text: str, setting: str = "") -> list[tuple[str, str, str, str]]:
        """[(id, text, source label, kind)]"""
        items = []
        if kb is not None:
            for i, f in enumerate(kb.relevant_facts(text, k=12)):
                label = f"{f['doc']}{' p.' + str(f['page']) if f.get('page') else ''}" if f.get("doc") else "fact sheet"
                items.append((f"F{i + 1}", f["text"], label, "fact"))
            for i, d in enumerate(kb.search(text, k=4)):
                items.append((f"D{i + 1}", d["text"][:900], f"{d['doc']}{' p.' + str(d['page']) if d.get('page') else ''}",
                              "document"))
        now = time.monotonic()
        earlier = [c for c in self.claims if c.text != text and now - c.t < 900][-12:]
        for i, c in enumerate(earlier):
            label = f"{c.speaker}, earlier" if c.speaker != speaker else f"{speaker} said earlier"
            items.append((f"S{i + 1}", f"{c.speaker}: {c.text}", label, "speaker"))
        return items

    async def check(self, models, kb, *, speaker: str, text: str, setting: str, history: str) -> Verdict:
        items = self.evidence(kb, speaker, text, setting)
        if not items:
            return Verdict(False, reason="no evidence to check against")
        self.checks += 1
        system = CHECK_SYS.format(setting=setting, speaker=speaker)
        ev = "\n".join(f"{i}: {t}" for i, t, _, _ in items)
        user = f"Evidence:\n{ev}\n\nRecent conversation:\n{history}\n\nLATEST statement:\n{speaker}: {text}\n\nJSON only."
        try:
            raw = await models.chat(system, user, max_tokens=220, temperature=0.0, timeout=6)
        except Exception as e:  # noqa: BLE001
            return Verdict(False, reason=f"check failed: {e}"[:120])
        return self.judge(parse_json(raw), items, speaker=speaker, text=text)

    def judge(self, d, items, *, speaker: str, text: str) -> Verdict:
        if not isinstance(d, dict) or not d.get("contradicts"):
            return Verdict(False, reason="no contradiction")
        try:
            conf = float(d.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        eid = str(d.get("evidence_id", "")).strip().upper()
        item = next((it for it in items if it[0] == eid), None)
        v = Verdict(True, eid, str(d.get("evidence", "")).strip(), conf, str(d.get("correction", "")).strip())
        if item is None:
            v.contradicts, v.reason = False, f"cited evidence {eid or '(none)'} does not exist"
            return v
        _, item_text, v.source, v.kind = item
        need = self.min_confidence + (0.05 if v.kind == "speaker" else 0.0)
        if conf < need:
            v.contradicts, v.reason = False, f"confidence {conf:.2f} < {need:.2f}"
        elif not quote_found(v.evidence, item_text):
            v.contradicts, v.reason = False, "the quoted evidence is not in the cited source"
        elif quote_found(v.evidence, text) and v.kind == "speaker" and item_text.endswith(text):
            v.contradicts, v.reason = False, "evidence is the statement itself"
        elif not v.correction:
            v.contradicts, v.reason = False, "no correction to say"
        else:
            key = _norm(v.evidence)[:80] + "|" + speaker.lower()
            now = time.monotonic()
            if key in self.corrected:
                v.contradicts, v.reason = False, "already corrected"
            elif now - self.last_correction < self.cooldown_s:
                v.contradicts, v.reason = False, "cooldown"
            else:
                self.corrected.add(key)
                self.last_correction = now
                v.reason = "verified"
        words = v.correction.split()
        if len(words) > 40:
            v.correction = " ".join(words[:40]).rstrip(",;") + "."
        return v
