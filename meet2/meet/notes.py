"""Rolling meeting state (F3): topic, decisions, action items, open questions.

Refreshed in the background every few final lines (never on a reply's critical path) and shown in the Notes tab;
the writer sees the topic so replies stay on it.
"""
from __future__ import annotations

import time

from .brain import parse_json

NOTES_SYS = """You keep live notes for a meeting ({setting}) between {people} and an AI participant called {ai}.
Given the previous notes and the latest transcript, output ONLY one JSON object:
{{"topic": "<what they are discussing now, under 12 words>", "decisions": ["..."], "actions": ["<who> will <what> (<when>, if said)"], "questions": ["<open questions nobody has answered yet>"]}}
Keep earlier items that still hold, drop answered questions, at most 6 items per list, each under 20 words. Only what was actually said."""


class Notes:
    EVERY_LINES = 4
    EVERY_S = 45.0

    def __init__(self):
        self.state: dict = {"topic": "", "decisions": [], "actions": [], "questions": []}
        self.lines_since = 0
        self.last = 0.0
        self.busy = False

    def due(self) -> bool:
        self.lines_since += 1
        return not self.busy and (self.lines_since >= self.EVERY_LINES or
                                  (self.lines_since >= 1 and time.monotonic() - self.last > self.EVERY_S))

    async def refresh(self, models, *, ai: str, setting: str, people: list[str], transcript: str) -> dict | None:
        self.busy = True
        self.lines_since = 0
        self.last = time.monotonic()
        try:
            system = NOTES_SYS.format(setting=setting, people=", ".join(people) or "the participants", ai=ai)
            user = f"Previous notes: {self.state}\n\nTranscript (latest last):\n{transcript}\n\nJSON only."
            d = parse_json(await models.chat(system, user, max_tokens=400, temperature=0.0, timeout=12))
            if not isinstance(d, dict):
                return None
            self.state = {"topic": str(d.get("topic", ""))[:120],
                          **{k: [str(x)[:160] for x in d.get(k, []) if x][:6] for k in ("decisions", "actions", "questions")}}
            return self.state
        except Exception:  # noqa: BLE001
            return None
        finally:
            self.busy = False
