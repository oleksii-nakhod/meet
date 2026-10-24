"""Meeting memory: short running notes of the meeting so far, so replies can use what was said before the last 14 lines.

A reply is written from the last 14 lines (agent.py), so by minute ten "what did we decide at the start?" had nothing
to go on. Beside the conversation, never in a reply's path, every `every` new lines one request folds them into the
notes (decisions, numbers, dates, who does what, open questions, with names); a reply then gets the notes plus the
recent lines, so its prompt stays the same size however long the meeting runs. The newest `margin` lines are left
for later: they are in every reply's window anyway, and they can still change (a line merged with what followed, an
AI reply marked [interrupted] or dropped because nobody heard it). A failed request keeps the old notes; the next
batch retries from the same place.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass

NOTES_SYS = """You are the note-taker for a meeting ({setting}). {ai} is an AI participant in it. You keep short notes of what matters so {ai} can answer questions about earlier parts of the meeting.
Each note is one short line starting with "- ", summarising in your own words: decisions, numbers, amounts, dates, times, places, names, who will do what, open questions, and who proposed or objected to what.
Never copy transcript lines. Leave out small talk, greetings, jokes and filler completely (weekends, food, weather, sports, pets, "cool", "okay"), unless it decides something.
When something changes or is corrected, keep only the latest version and note the change ("- Launch moved from Friday to Monday (Mei)").
Use only what the transcript says; never add, guess or explain. At most 15 notes, under 200 words in total.
If there is nothing worth noting yet, answer exactly: none"""
# the model follows the end of the request most closely (small talk crept into the notes without this)
NOTES_TAIL = ("Write the complete updated notes: only decisions, numbers, plans, who does what and open questions; "
              "no small talk; latest versions only; at most 15 lines.")


@dataclass
class Memory:
    every: int = 8          # fold in new lines once this many are waiting
    margin: int = 4         # the newest lines are left out: still in every reply's window, and they can still change
    batch: int = 40         # at most this many lines per request (catching up after a switch from default mode)
    timeout: float = 20.0
    notes: str = ""
    upto: int = 0           # history[:upto] is in the notes
    updates: int = 0
    failures: int = 0
    running: bool = False

    def due(self, n: int) -> bool:
        return not self.running and n - self.margin - self.upto >= self.every

    async def update(self, models, history: list[dict], *, ai: str, setting: str, log=None) -> None:
        """Fold history[upto:len - margin] into the notes, a batch at a time. Only ever one at a time."""
        if self.running:
            return
        self.running = True
        try:
            self.upto = min(self.upto, len(history))
            while (end := min(len(history) - self.margin, self.upto + self.batch)) > self.upto:
                new = history[self.upto:end]
                lines = "\n".join(f"{h['speaker']}: {h['text'][:400]}" for h in new)
                user = (f"Notes so far:\n{self.notes or 'none'}\n\nNext part of the transcript:\n{lines}\n\n"
                        f"{NOTES_TAIL}")
                t0 = time.perf_counter()
                try:
                    out = await models.chat(NOTES_SYS.format(ai=ai, setting=setting), user, max_tokens=400,
                                            temperature=0.0, timeout=self.timeout)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    self.failures += 1
                    if log:
                        log({"type": "memory", "error": str(e)[:200], "upto": self.upto})
                    return
                notes = clean(out, said=[h["text"] for h in new])
                if notes is None:  # an empty or rambling answer: keep the old notes, retry with the next batch
                    self.failures += 1
                    if log:
                        log({"type": "memory", "error": "unusable notes", "raw": out[:300], "upto": self.upto})
                    return
                self.notes, self.upto = notes, end
                self.updates += 1
                if log:
                    log({"type": "memory", "upto": end, "lines": len(new), "s": round(time.perf_counter() - t0, 2),
                         "notes": notes})
        finally:
            self.running = False

    def context(self, n: int, window: int = 14, cap: int = 30) -> tuple[str, int]:
        """For a reply after n lines of history: (notes block for its prompt, how many recent lines to show raw).
        Short meetings get no notes and the usual window. When the notes lag behind, the window widens (up to `cap`)
        to the first line they don't cover, so no line falls between the notes and the window."""
        if n <= window:
            return "", window
        start = max(n - cap, min(n - window, self.upto))
        block = (f"[Notes on earlier parts of this meeting; use them only when the latest utterance asks about or "
                 f"depends on something said earlier:]\n{self.notes}\n\n") if self.notes else ""
        return block, n - start


def _bare(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9$€£%']+", s.lower()))


def clean(out: str, said: list[str] = ()) -> str | None:
    """The notes as returned, trimmed and without lines copied from the transcript (`said`, as "- Sam: ..." or bare);
    "" for none; None when the answer isn't notes."""
    t = (out or "").strip().strip("`").strip()
    if t.lower().rstrip(".") in ("none", "- none", "notes: none"):
        return ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines and lines[0].lower().rstrip(":") in ("updated notes", "notes"):
        lines = lines[1:]
    copies = {_bare(s) for s in said}
    lines = [ln for ln in lines if _bare(re.sub(r"^[-*•\s]*(?:[A-Z][\w' .-]{0,30}:\s+)?", "", ln)) not in copies]
    if not lines or len(" ".join(lines).split()) > 400:
        return None
    return "\n".join(lines[:20])
