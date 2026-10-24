"""The AI participant: listens to everyone, decides whether and when to speak, and corrects contradictions.

    people -> Listener (front-end, per-person STT, barge-in probe)
           -> pieces  -> Monitor (fact check per clause, beside the conversation) -> Floor.correct
           -> turns   -> prefilter -> Brain (judge + writer, streamed)            -> Floor.propose
    Floor (the one owner of speech) -> OutputWorker -> transport (WebSocket, or the live avatar over LiveKit)

Every background task belongs to the agent's task set and a generation number: stop() (room empty) cancels and
awaits all of them, and work from an older generation is dropped when it finishes.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

from . import config
from .brain import Context, SpokenText, Turn as CTurn, decide, writer_prompt
from .floor import Floor, RoomNow
from .listen.listener import Listener, Turn
from .listen.words import bargein, lexicon, prefilter
from .monitor import Claim, Monitor, claim_like
from .notes import Notes
from .kb import terms
from .speech import OutputWorker, Synthesizer, Utterance


class HiggsAgent:
    def __init__(self, room, models, library):
        self.room, self.models, self.library = room, models, library
        self.settings = s = config.settings()
        self.lex = lexicon(str(s.path(s.fillers_file)))
        self.listeners: dict[int, Listener] = {}
        self.ledger: list[dict] = []                 # final lines and AI turns, oldest first
        self.buffer: list[tuple[Turn, bool]] = []      # (turn, typed) held during a correction
        self.tasks: set[asyncio.Task] = set()
        self.generation = 0
        self.running = False
        self.output = OutputWorker(room.transport, log=self.log, on_level=self._ai_level)
        self.synth = Synthesizer(models.tts_stream, on_error=lambda e: self.log({"type": "ai_error", "error": e}))
        self.floor = Floor(self.output, emit=self._floor_event, set_human_gain=self._human_gain,
                           release_buffer=self._release_buffer, floor_silence=s.floor_silence_s,
                           clause_silence=s.clause_silence_s, duck_gain=s.fact_duck_gain,
                           correction_guard=s.correction_guard_s)
        self.floor.on_interrupted = self._on_interrupted
        self.monitor = Monitor(min_confidence=s.fact_min_confidence, cooldown_s=s.fact_cooldown_s)
        self.notes = Notes()
        self.thinking = 0
        self.last_ai_db: float | None = None
        self.last_ai_t = 0.0
        self.probe_stt = None
        self.probe_lock = asyncio.Lock()
        self.speaking_utt: Utterance | None = None
        self.told_budget = False
        self.avatar = None
        self.mode = "smart"       # "smart" | "plain": plain answers every line and stops at any sound (a baseline)

    # ------------------------------------------------------------ helpers
    @property
    def ai_name(self) -> str:
        return self.room.brief.ai

    def people(self) -> list[str]:
        return self.room.names()

    def kb(self):
        return self.library.get(self.room.collection)

    def log(self, event: dict) -> None:
        self.room.log.write(event)
        if event.get("type") == "ai_error" and "budget" in event.get("error", "") and not self.told_budget:
            self.told_budget = True
            self.room.event({"type": "toast", "level": "error",
                             "text": f"{self.ai_name} has used its request budget and will stay quiet."})

    def spawn(self, coro) -> asyncio.Task:
        t = asyncio.ensure_future(coro)
        self.tasks.add(t)

        def done(f: asyncio.Task):
            self.tasks.discard(f)
            if not f.cancelled() and f.exception() is not None:
                self.log({"type": "ai_error", "error": f"task: {f.exception()!r}"[:200]})
        t.add_done_callback(done)
        return t

    def ai_talking(self) -> bool:
        return self.output.speaking or self.floor.state in ("speaking", "correcting")

    def echo_level(self) -> float | None:
        """The AI's output level, held for the playout delay + 0.3 s so its echo tail is still recognised."""
        if self.last_ai_db is None:
            return None
        hold = self.room.transport.playout_delay() + 0.3
        return self.last_ai_db if time.monotonic() - self.last_ai_t < hold else None

    def _ai_level(self, db: float | None) -> None:
        if db is not None:
            self.last_ai_db, self.last_ai_t = db, time.monotonic()

    # ------------------------------------------------------------ lifecycle (room interface)
    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.generation += 1
        self.output.start()
        self.probe_stt = self.models.open_stt()
        self.on_roster(self.room.roster())
        self.spawn(self._clock())
        if self.room.avatar == "live":
            try:
                await self.set_avatar("live")
            except Exception as e:  # noqa: BLE001
                self.room.event({"type": "toast", "level": "error", "text": f"Avatar failed: {e}"[:200]})

    async def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self.generation += 1
        self.floor.stop_now("the room emptied")
        for t in list(self.tasks):
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        for li in list(self.listeners.values()):
            await li.close()
        self.listeners.clear()
        if self.probe_stt:
            await self.probe_stt.close()
        await self.output.stop()
        if self.avatar:
            await self.avatar.stop()
            self.avatar = None
            self.output.transport = self.room.transport

    def feed(self, slot: int, pcm: bytes) -> None:
        li = self.listeners.get(slot)
        if li is not None:
            li.feed(np.frombuffer(pcm, np.int16))

    def on_roster(self, people: list[dict]) -> None:
        if not self.running:
            return
        live = {p["slot"]: p for p in people if p.get("mic", True)}
        for slot in list(self.listeners):
            if slot not in live or self.listeners[slot].name != live[slot]["name"]:
                self.spawn(self.listeners.pop(slot).close())
        for slot, p in live.items():
            if slot not in self.listeners:
                self.listeners[slot] = Listener(self, slot, p["name"])

    def on_chat(self, slot: int, name: str, text: str) -> None:
        if not self.running:
            return
        Listener._n += 1
        turn = Turn(f"L{Listener._n}", slot, name, time.monotonic(), [text])
        self._final(turn, typed=True)

    async def set_avatar(self, mode: str) -> None:
        if mode == "live":
            if self.avatar is None:
                from .transport.livekit import AvatarBridge
                bridge = AvatarBridge(self.room)
                await bridge.start()
                self.avatar = bridge
                self.output.transport = bridge
                self.output.start()
        elif self.avatar is not None:
            bridge, self.avatar = self.avatar, None
            self.output.transport = self.room.transport
            self.output.start()
            await bridge.stop()

    def set_mode(self, mode: str) -> None:
        self.mode = "plain" if mode == "plain" else "smart"
        self.floor.stop_now(f"switched to {self.mode}")

    def on_kb_changed(self) -> None:
        pass  # the knowledge base is read live on every reply and check

    # ------------------------------------------------------------ the clock
    async def _clock(self) -> None:
        shown_state = None
        last_text: dict[str, str] = {}
        while True:
            await asyncio.sleep(0.05)
            now = time.monotonic()
            ls = list(self.listeners.values())
            room = RoomNow(speaking={li.slot for li in ls if li.speaking},
                           in_turn={li.slot for li in ls if li.in_turn},
                           quiet_for=now - max([li.last_voice for li in ls], default=0.0),
                           paused={li.slot: li.pause for li in ls if li.in_turn})
            self.floor.tick(now, room)
            for li in ls:                                    # unfinished turns nobody continued
                if li.turn and li.turn.hold_until and now > li.turn.hold_until and not li.in_turn:
                    self.finish_turn(li, li.turn, force=True)
            state = ("correcting" if self.floor.state == "correcting" else "speaking" if self.output.speaking
                     else "thinking" if self.thinking or self.floor.state == "waiting"
                     else "listening" if room.speaking else "idle")
            if state != shown_state:
                shown_state = state
                self.room.event({"type": "ai_state", "state": state}, log=False)
            u = self.speaking_utt                            # captions follow the streamed text
            if u is not None and last_text.get(u.id) != u.text:
                last_text[u.id] = u.text
                self._say(u, done=False)
            changed = False
            for li in ls:
                person = self.room.people.get(li.slot)
                if person and person.noisy != li.fe.noisy:
                    person.noisy = li.fe.noisy
                    changed = True
            if changed:
                self.room.broadcast({"type": "roster", "people": self.room.roster()})

    # ------------------------------------------------------------ listening
    def on_speech_start(self, li: Listener) -> None:
        if self.mode == "plain" and self.floor.state == "speaking":
            self.floor.bargein(li.slot, "interrupt", "(any sound stops plain mode)")

    def on_piece(self, li: Listener, turn: Turn, text: str, *, final: bool) -> None:
        """A clause (or the last piece of a turn) has its words: show it, and fact-check it at once."""
        self.room.event({"type": "line", "id": turn.id, "slot": li.slot, "speaker": li.name, "text": turn.text,
                         "final": False}, log=False)
        if self.floor.state == "correcting":
            return                                          # buffered: checked when the correction ends
        if self.mode == "smart" and self.settings.proactive_facts and claim_like(text) and not self.ai_talking_correction():
            self.spawn(self._fact_check(li.slot, li.name, text, turn.id))

    def ai_talking_correction(self) -> bool:
        return self.floor.state == "correcting" or self.floor.urgent is not None

    def finish_turn(self, li: Listener, turn: Turn, force: bool = False) -> None:
        if li.turn is not turn:
            return
        text = turn.text
        if not text:
            li.turn = None
            return
        _, kind = prefilter(text, self.ai_name, self._last_ai_text(), self.lex, self.people())
        if kind == "unfinished" and not force and not turn.hold_until:
            turn.hold_until = time.monotonic() + 1.5          # they may go on: keep the line open
            return
        li.turn = None
        if self.ai_talking() and self.floor.state == "speaking":   # a short "wait" that ended before a probe
            v = bargein(text, li.fe.voiced, self.lex, self.ai_name)
            if v in ("stop", "interrupt"):
                self.floor.bargein(li.slot, v, text, host=self._is_host(li.slot))
        self._final(turn)

    def _final(self, turn: Turn, typed: bool = False) -> None:
        cur = self.floor.current
        if cur is not None and cur.utt.kind == "correction" and cur.line_id == turn.id:
            # the line being corrected ended during its own correction: record it, the correction is the answer
            self.room.event({"type": "line", "id": turn.id, "slot": turn.slot, "speaker": turn.speaker,
                             "text": turn.text, "final": True, "kind": "line", "why": "corrected"})
            self.ledger.append({"id": turn.id, "slot": turn.slot, "speaker": turn.speaker, "text": turn.text,
                                "t": time.monotonic()})
            self.monitor.remember(Claim(turn.id, turn.speaker, turn.text, time.monotonic()))
            return
        if self.floor.state == "correcting":
            self.buffer.append((turn, typed))
            self.room.event({"type": "line", "id": turn.id, "slot": turn.slot, "speaker": turn.speaker,
                             "text": turn.text, "final": True, "kind": "buffered"})
            return
        text, kind = prefilter(turn.text, self.ai_name, self._last_ai_text(), self.lex, self.people())
        line = {"type": "line", "id": turn.id, "slot": turn.slot, "speaker": turn.speaker, "text": text,
                "final": True, "kind": kind or "line", "typed": typed}
        self.room.event(line)
        if kind in ("skip", "filler", "backchannel"):
            return
        if turn.during_correction:            # the reply should know this was said over its correction
            text = f"{text} [said while you were correcting]"
        self.ledger.append({"id": turn.id, "slot": turn.slot, "speaker": turn.speaker, "text": text,
                            "t": time.monotonic()})
        if claim_like(text):
            self.monitor.remember(Claim(turn.id, turn.speaker, text, time.monotonic()))
        if kind is not None and not (self.mode == "plain" and kind in ("fragment", "unfinished")):
            return                                           # stop / fragment: logged, not answered
        self.floor.moved_on(turn.id, turn.slot, time.monotonic())
        if typed and self.mode == "smart" and self.settings.proactive_facts and claim_like(text):
            self.spawn(self._fact_check(turn.slot, turn.speaker, text, turn.id))
        if self.notes.due():
            self.spawn(self._notes())
        self.spawn(self._respond(turn.id, turn.slot, turn.speaker, text, typed))

    def _last_ai_text(self) -> str:
        for line in reversed(self.ledger[-3:]):
            if line["slot"] == 0:
                return line["text"]
        return ""

    def _is_host(self, slot: int) -> bool:
        p = self.room.people.get(slot)
        return bool(p and p.host)

    def on_bargein(self, li: Listener, verdict: str, text: str) -> bool:
        return self.floor.bargein(li.slot, verdict, text, host=self._is_host(li.slot))

    async def probe_transcribe(self, pcm: np.ndarray) -> str:
        """Transcribe a barge-in so far on the shared probe session (~0.3 s); HTTP if that is down."""
        async with self.probe_lock:
            try:
                if self.probe_stt and self.probe_stt.ready.is_set():
                    for k in range(0, len(pcm), 4800):
                        await self.probe_stt.append(pcm[k:k + 4800].tobytes())
                    return Listener.guard(await self.probe_stt.commit(timeout=2.0))
            except Exception:  # noqa: BLE001
                pass
        try:
            return Listener.guard(await self.models.transcribe(pcm, timeout=3))
        except Exception:  # noqa: BLE001
            return ""

    # ------------------------------------------------------------ replying
    def _context(self, text: str, speaker: str, exclude_id: str, *, sources: bool) -> Context:
        kb = self.kb()
        facts = [f["text"] for f in kb.relevant_facts(text, k=12)] or self.room.brief.facts
        turns = [CTurn(line["speaker"], line["text"]) for line in self.ledger if line["id"] != exclude_id][-20:]
        src = kb.search(text + " " + " ".join(t.text for t in turns[-2:]), k=3) if sources else []
        topic = self.notes.state.get("topic")
        setting = self.room.brief.setting + (f"; current topic: {topic}" if topic else "")
        return Context(ai=self.ai_name, setting=setting, people=self.people(), facts=facts, turns=turns,
                       current=CTurn(speaker, text), sources=src)

    async def _respond(self, line_id: str, slot: int, speaker: str, text: str, typed: bool) -> None:
        gen = self.generation
        self.thinking += 1
        try:
            c = self._context(text, speaker, line_id, sources=True)
            addressed = c.one_to_one or self.ai_name.lower() in text.lower() or self.mode == "plain"
            likely = addressed or text.rstrip().endswith("?")
            utt = writer = None
            if likely:                                       # draft while the judge decides
                utt, writer = self._write(c, line_id)
            if self.mode == "plain":
                speak, info = True, {"why": "plain mode answers everything", "votes": 0}
            else:
                samples = self.settings.judge_samples or (1 if c.one_to_one else 3)
                speak, info = await decide(self.models, c, samples)
            self.room.event({"type": "line_reading", "id": line_id, "speak": speak, "why": info["why"],
                             "votes": info["votes"]})
            self.room.event({"type": "line", "id": line_id, "slot": slot, "speaker": speaker, "text": text,
                             "final": True, "kind": "line", "why": info["why"], "typed": typed}, log=False)
            if gen != self.generation or not speak:
                if utt:
                    utt.cancel()
                return
            if utt is None:
                utt, writer = self._write(c, line_id)
            if any(p.line_id == line_id for p in (self.floor.urgent,) if p):
                utt.cancel()                                 # a correction already answers this line
                return
            self.floor.propose(utt, line_id=line_id, slot=slot, addressed=addressed, now=time.monotonic())
            if typed:
                await writer
                if not utt.cancelled and utt.text:
                    self.room.event({"type": "chat", "slot": 0, "name": self.ai_name, "text": utt.text})
        finally:
            self.thinking -= 1

    def _write(self, c: Context, line_id: str) -> tuple[Utterance, asyncio.Task]:
        utt = Utterance("reply", meta={"line": line_id})
        self.spawn(self.synth.run(utt))

        async def write():
            system, user = writer_prompt(c)
            spoken = SpokenText()
            try:
                async for delta in self.models.chat_stream(system, user, max_tokens=160, temperature=0.6):
                    if utt.cancelled:
                        return
                    utt.feed(spoken.push(delta))
                utt.feed(spoken.finish())
                ids = [i for i in spoken.source_ids() if 1 <= i <= len(c.sources)]
                if ids:
                    s = c.sources[ids[0] - 1]
                    utt.meta["source"] = f"{s['doc']}{' p.' + str(s['page']) if s.get('page') else ''}"
                else:
                    utt.meta["source"] = attribute(utt.text, c.sources, self.kb().facts())
                self.log({"type": "reply_written", "id": utt.id, "text": utt.text, "source": utt.meta.get("source"),
                          "sources": len(c.sources), "marker": spoken.tail[:40]})
            except Exception as e:  # noqa: BLE001
                self.log({"type": "ai_error", "error": f"writer: {e}"[:200]})
            finally:
                utt.close()
        return utt, self.spawn(write())

    # ------------------------------------------------------------ corrections (F4)
    async def _fact_check(self, slot: int, speaker: str, text: str, line_id: str) -> None:
        gen = self.generation
        history = "\n".join(f"{line['speaker']}: {line['text']}" for line in self.ledger[-8:])
        t0 = time.monotonic()
        v = await self.monitor.check(self.models, self.kb(), speaker=speaker, text=text,
                                     setting=self.room.brief.setting, history=history)
        self.log({"type": "fact_check", "line": line_id, "speaker": speaker, "text": text,
                  "contradicts": v.contradicts, "reason": v.reason, "evidence_id": v.evidence_id,
                  "evidence": v.evidence, "confidence": v.confidence, "s": round(time.monotonic() - t0, 2)})
        if gen != self.generation or not v.contradicts:
            return
        utt = Utterance("correction", meta={"line": line_id, "source": v.source, "evidence": v.evidence,
                                            "speaker": speaker, "claim": text})
        utt.feed(v.correction)
        utt.close()
        self.spawn(self.synth.run(utt))
        self.floor.correct(utt, line_id=line_id, slot=slot, now=time.monotonic())

    def _human_gain(self, gain: float, reason: str) -> None:
        self.room.event({"type": "human_gain", "gain": gain, "reason": reason})

    def _release_buffer(self) -> None:
        held, self.buffer = self.buffer, []
        self.log({"type": "buffer_release", "lines": len(held)})
        for turn, typed in held:
            turn.during_correction = True
            self._final(turn, typed)
            if self.settings.proactive_facts and claim_like(turn.text) and not typed:
                self.spawn(self._fact_check(turn.slot, turn.speaker, turn.text, turn.id))

    # ------------------------------------------------------------ what the AI says
    def _floor_event(self, e: dict[str, Any]) -> None:
        self.log(e)
        if e["type"] == "floor_take":
            p = self.floor.current
            if p is None:
                return
            u = p.utt
            self.speaking_utt = u
            if u.kind == "correction":
                self.room.event({"type": "correction", "id": u.id, "text": u.text, "source": u.meta.get("source"),
                                 "evidence": u.meta.get("evidence"), "speaker": u.meta.get("speaker"),
                                 "claim": u.meta.get("claim")})
            self._say(u, done=False)
            self.spawn(self._after(u))
        elif e["type"] == "backchannel":
            pass

    def _say(self, u: Utterance, *, done: bool) -> None:
        self.room.event({"type": "ai_say", "id": u.id, "text": u.text, "kind": u.kind, "done": done,
                         "interrupted": u.interrupted, "source": u.meta.get("source")}, log=done)

    async def _after(self, u: Utterance) -> None:
        await u.heard.wait()
        if self.speaking_utt is u:
            self.speaking_utt = None
        if u.interrupted:
            return                                           # _on_interrupted recorded what was heard
        self._say(u, done=True)
        if u.text:
            self.ledger.append({"id": u.id, "slot": 0, "speaker": self.ai_name, "text": u.text, "t": time.monotonic()})

    def _on_interrupted(self, u: Utterance, heard: str, slot: int | None) -> None:
        text = (heard or "") + " [interrupted]"
        self.room.event({"type": "ai_say", "id": u.id, "text": heard or "…", "kind": u.kind, "done": True,
                         "interrupted": True, "source": u.meta.get("source")})
        self.ledger.append({"id": u.id, "slot": 0, "speaker": self.ai_name, "text": text.strip(), "t": time.monotonic()})

    async def _notes(self) -> None:
        transcript = "\n".join(f"{line['speaker']}: {line['text']}" for line in self.ledger[-30:])
        state = await self.notes.refresh(self.models, ai=self.ai_name, setting=self.room.brief.setting,
                                         people=self.people(), transcript=transcript)
        if state:
            claims = [f"{c.speaker}: {c.text}" for c in self.monitor.claims[-5:]]
            self.room.event({"type": "notes", **state, "claims": claims})


def attribute(reply: str, sources: list[dict], facts: list[dict]) -> str | None:
    """Which document excerpt or sourced fact a reply rests on: the one sharing the most specific words with it
    (at least 3, and at least a third of the reply's own). None for small talk and general knowledge."""
    said = set(terms(reply))
    if len(said) < 3:
        return None
    best, label = 0, None
    for f in facts:
        if f.get("doc"):
            n = len(said & set(terms(f["text"])))
            if n > best:
                best, label = n, f"{f['doc']}{' p.' + str(f['page']) if f.get('page') else ''}"
    for s in sources:
        n = len(said & set(terms(s["text"])))
        if n > best:
            best, label = n, f"{s['doc']}{' p.' + str(s['page']) if s.get('page') else ''}"
    return label if best >= 3 and best >= len(said) / 3 else None
