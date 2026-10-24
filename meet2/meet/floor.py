"""The floor controller: the only thing that starts or stops the AI's speech.

States: idle -> waiting (a reply is ready, waiting for a gap) -> speaking -> idle
        any  -> correcting (a protected correction) -> idle

  * A reply starts only once the room has been quiet for floor_silence and nobody is mid-turn. It expires if the
    conversation moves on (a newer substantive line from someone else) or waits too long.
  * While a reply plays, a listener's speech is classified by the barge-in gate (listen/words.py): fillers and
    acknowledgements change nothing; a stop request or a real interruption stops it at once.
  * A correction (F4) takes the floor at the speaker's next clause pause (or after urgent_wait even if they don't
    pause), cutting off any reply. While it plays every client ducks the other humans, new speech is buffered
    rather than acted on, and only the host saying "stop" can cut it. Release happens when the correction has been
    heard (not merely sent); then the volume returns and the buffered lines are handled in order. For the first
    correction_guard seconds (default 2) only the host's "stop" can cut it; after that someone who keeps
    interrupting for 0.6 s more gets the floor (a line that just runs past the guard doesn't; fillers never do).

Pure logic on an injected clock: tick(now, room) is called every ~50 ms by the agent, so tests drive it directly.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .speech import Utterance


class Player(Protocol):
    speaking: bool

    def play(self, u: Utterance, on_heard: Callable[[Utterance], None]) -> None: ...

    async def interrupt(self) -> int: ...

    def heard_samples(self) -> int: ...


@dataclass
class RoomNow:
    """What the floor needs to know about the humans at one instant."""
    speaking: set[int] = field(default_factory=set)        # slots voicing right now
    in_turn: set[int] = field(default_factory=set)         # slots whose turn hasn't ended (incl. short pauses)
    quiet_for: float = 99.0                                 # seconds since anyone last voiced
    paused: dict[int, float] = field(default_factory=dict)  # slot -> seconds into their current pause


@dataclass
class Pending:
    utt: Utterance
    line_id: str
    slot: int
    addressed: bool
    created: float


class Floor:
    STALE_S = 3.5          # an unaddressed reply that couldn't start this long after its line, people still talking
    STALE_ADDRESSED = 6.0  # an answer to a question put to the AI by name waits longer
    STALE_MAX = 8.0        # never later than this, even in a quiet room

    def __init__(self, player: Player, *, emit: Callable[[dict], None], set_human_gain: Callable[[float, str], None],
                 release_buffer: Callable[[], None], floor_silence: float = 0.35, clause_silence: float = 0.28,
                 urgent_wait: float = 1.2, duck_gain: float = 0.15, correction_guard: float = 2.0):
        self.player, self.emit, self.set_human_gain, self.release_buffer = player, emit, set_human_gain, release_buffer
        self.floor_silence, self.clause_silence, self.urgent_wait, self.duck_gain = (
            floor_silence, clause_silence, urgent_wait, duck_gain)
        self.correction_guard = correction_guard
        self.insist_s = 0.6                    # after the guard, someone must keep interrupting this long
        self._insist: dict[int, float] = {}    # slot -> first post-guard interrupt verdict
        self.state = "idle"
        self.pending: Pending | None = None
        self.urgent: Pending | None = None
        self.current: Pending | None = None
        self.on_interrupted: Callable[[Utterance, str, int | None], None] = lambda u, heard, slot: None
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------ proposals
    def propose(self, utt: Utterance, *, line_id: str, slot: int, addressed: bool, now: float) -> None:
        if self.pending:
            self._drop(self.pending, "replaced by a newer reply")
        self.pending = Pending(utt, line_id, slot, addressed, now)
        self._set("waiting" if self.state == "idle" else self.state)

    def correct(self, utt: Utterance, *, line_id: str, slot: int, now: float) -> None:
        if self.urgent:
            self._drop(self.urgent, "replaced by a newer correction")
        self.urgent = Pending(utt, line_id, slot, True, now)
        if self.pending and self.pending.line_id == line_id:      # the correction is the answer to that line
            self._drop(self.pending, "a correction covers it")
            self.pending = None

    def moved_on(self, line_id: str, slot: int, now: float) -> None:
        """A newer substantive line: an unaddressed pending reply to someone else's earlier line is stale."""
        p = self.pending
        if p and p.line_id != line_id and not p.addressed:
            self._drop(p, "the conversation moved on")
            self.pending = None
            if self.state == "waiting":
                self._set("idle")

    # ------------------------------------------------------------ the clock
    def tick(self, now: float, room: RoomNow) -> None:
        if self.urgent and self.state != "correcting":
            u = self.urgent
            age = now - u.created
            speaker_paused = all(room.paused.get(s, 99.0) >= self.clause_silence for s in room.speaking | room.in_turn)
            if not room.speaking or speaker_paused or age >= self.urgent_wait:
                self._start_correction(u)
            return
        if self.pending and self.state in ("idle", "waiting"):
            p = self.pending
            age = now - p.created
            limit = self.STALE_ADDRESSED if p.addressed else self.STALE_S
            if age > self.STALE_MAX or (age > limit and room.quiet_for < self.floor_silence):
                self._drop(p, "no gap to speak in")
                self.pending = None
                self._set("idle")
                return
            if room.quiet_for >= self.floor_silence and not room.speaking and not room.in_turn:
                self.pending = None
                self._start(p, "speaking")

    # ------------------------------------------------------------ barge-in
    def bargein(self, slot: int, verdict: str, text: str, *, host: bool = False) -> bool:
        """A listener's verdict while the AI talks. True if the AI stopped."""
        if self.state == "correcting":
            if verdict == "stop" and host:
                self._spawn(self._interrupt(slot, "the host stopped the correction"))
                return True
            if verdict in ("stop", "interrupt") and self.correction_age() >= self.correction_guard:
                first = self._insist.setdefault(slot, time.monotonic())
                if time.monotonic() - first >= self.insist_s:      # still pressing: let them have the floor
                    self._spawn(self._interrupt(slot, f"after the guard, {verdict}: {text}"[:120]))
                    return True
            return False                       # inside the guard, a line that just ran past it, or a filler
        if self.state != "speaking":
            return False
        if verdict in ("stop", "interrupt"):
            self._spawn(self._interrupt(slot, f"{verdict}: {text}"[:120]))
            return True
        if verdict == "filler":
            self.emit({"type": "backchannel", "slot": slot, "text": text})
        return False

    def correction_age(self) -> float:
        """Seconds the current correction has been audible (0 until its first frame is sent)."""
        cur = self.current
        if cur is None or cur.utt.started_at is None:
            return 0.0
        return time.monotonic() - cur.utt.started_at

    def stop_now(self, reason: str) -> None:
        """Mode switch / leave / budget: stop speaking and forget what was waiting."""
        for p in (self.pending, self.urgent):
            if p:
                self._drop(p, reason)
        self.pending = self.urgent = None
        if self.player.speaking:
            self._spawn(self._interrupt(None, reason))
        elif self.state == "correcting":
            self._release()

    # ------------------------------------------------------------ internals
    def _start(self, p: Pending, state: str) -> None:
        self.current = p
        self._set(state)
        self.emit({"type": "floor_take", "id": p.utt.id, "kind": p.utt.kind, "line": p.line_id,
                   "waited_s": round(time.monotonic() - p.utt.created, 3)})
        self.player.play(p.utt, self._heard)

    def _start_correction(self, u: Pending) -> None:
        self.urgent = None
        if self.pending:
            self._drop(self.pending, "a correction took the floor")
            self.pending = None
        cur = self.current
        self.set_human_gain(self.duck_gain, "fact_correction")
        self._insist.clear()
        self._set("correcting")
        if self.player.speaking and cur is not None:
            async def cut_then_correct():
                heard = await self.player.interrupt()
                self.on_interrupted(cur.utt, cur.utt.heard_text(heard), None)
                self._start(u, "correcting")
            self._spawn(cut_then_correct())
        else:
            self._start(u, "correcting")

    async def _interrupt(self, slot: int | None, reason: str) -> None:
        cur = self.current
        heard = await self.player.interrupt()
        if cur is not None:
            self.on_interrupted(cur.utt, cur.utt.heard_text(heard), slot)
            self.emit({"type": "interrupted", "id": cur.utt.id, "by": slot, "reason": reason})

    def _heard(self, u: Utterance) -> None:
        """The player says this utterance is over (finished and heard, or interrupted)."""
        if self.current is None or self.current.utt is not u:
            return
        self.current = None
        if u.kind == "correction" and self.state == "correcting":
            self._release()
        elif self.state != "correcting":
            self._set("waiting" if self.pending else "idle")

    def _release(self) -> None:
        self.set_human_gain(1.0, "")
        self._set("waiting" if self.pending else "idle")
        self.release_buffer()

    def _drop(self, p: Pending, reason: str) -> None:
        p.utt.cancel()
        self.emit({"type": "reply_skipped", "id": p.utt.id, "line": p.line_id, "kind": p.utt.kind, "reason": reason})

    def _set(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.emit({"type": "floor_state", "state": state})

    def _spawn(self, coro) -> None:
        t = asyncio.ensure_future(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
