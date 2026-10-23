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
import sys
import time
from collections import deque

import numpy as np
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mmhm"))
sys.path.insert(0, os.path.join(HERE, "..", "third_voice"))
from mmhm.engine import N, TICK, Engine, Policy, VAD  # noqa: E402
from mmhm.expressive import SentenceSpeaker  # noqa: E402
from mmhm.higgs import SR, Session  # noqa: E402

import avatar  # noqa: E402
import floor  # noqa: E402
import tv_eval  # noqa: E402

VOICE = os.environ.get("MEET_VOICE", "chloe")      # the same preset voice in both modes, so the voice doesn't tell
SAMPLES = int(os.environ.get("MEET_SAMPLES", "3"))  # judge/fact-check samples per line (1 for cheap test runs)
BUDGET = int(os.environ.get("MEET_BUDGET", "5000"))
SIGNALS = os.environ.get("MEET_SIGNALS", "1") == "1"  # per-person turn-taking models (about one CPU core each)
CALLS = {"n": 0}
# In 1:1 Mm-hm a commit after 0.1 s of silence is a cancellable guess. Here a committed line is transcribed and judged
# (7 requests) and shown to everyone, so a line ends only after a real pause: 0.45 s with the turn-taking model
# agreeing they're done (else its hold rules: max_wait / give_up). First B2 run with 0.1 s: 58 judged pieces for 32
# lines ("So.", "Um.", then the line again), 474 requests.
LINE_POLICY = dataclasses.replace(Policy(), min_silence=0.45)
FLOOR_POLICY = dataclasses.replace(Policy(), min_silence=0.35)  # room quiet before Higgs may start (0.1 s let it
#                                                                 start in the gap between an interrupter's words)
STALE_S = 3.5  # a reply that couldn't start this long after its line is about a moment that has passed
CORRECT = ("[The latest utterance contradicts what you know. Start by correcting it plainly, as {ai}, not as one of "
           "the people.]\n")

REPLY_SYS = ("You are {ai}, an AI participant with a voice in an online meeting ({setting}) with {people}. "
             "What you know: {facts} You decided to speak up after the latest utterance. If the latest utterance "
             "states something that contradicts what you know (a wrong day, time, number, place, dose or rule), start by "
             "correcting it plainly. Otherwise say what a helpful teammate would say: one or two short spoken sentences, "
             "no lists, no preamble.")  # third_voice/live_demo.py's writer, "group conversation" -> "online meeting"
PERSONA = ("You are {ai}, an AI participant with a voice in an online meeting ({setting}) with {people}. "
           "What you know: {facts} Keep each reply to one or two short spoken sentences.")


def spend(n=1):
    """Count Higgs requests; False once the budget is used up."""
    if CALLS["n"] + n > BUDGET:
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


def to16(pcm):
    return resample_poly(pcm.astype(np.float32) / 32768, 2, 3).astype(np.float32)


class Pool:
    """Pre-opened Realtime text sessions with fixed instructions, one request each, replaced in the background
    (third_voice/live_demo.py), plus a close() for when the roster (part of the instructions) changes."""

    def __init__(self, instructions, size):
        self.instructions, self.q, self.closed = instructions, asyncio.Queue(), False
        for _ in range(size):
            self._refill()

    def _refill(self):
        async def open_one():
            s = Session({"instructions": self.instructions, "output_modalities": ["text"]})
            await s.__aenter__()
            if self.closed:
                await close(s)
            else:
                await self.q.put(s)
        asyncio.ensure_future(open_one())

    async def ask(self, user):
        if not spend():
            return ""
        s = await self.q.get()
        self._refill()
        try:
            since = s.now()
            await s.say_text(user)
            _, ev = await s.wait("response.done", timeout=45, since=since)
            return s.transcript(ev["response"]["id"]).strip()
        except Exception:  # noqa: BLE001  (a failed vote counts as "don't speak": the safe side)
            return ""
        finally:
            asyncio.ensure_future(close(s))

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

    async def tick(self, frames):
        mix = np.zeros(N, np.int32)
        for f in frames.values():
            mix += f
        out = await self.engine.tick(np.clip(mix, -32768, 32767).astype(np.int16))
        for e in self.engine.log[self.k:]:
            kind = e["kind"]
            if kind == "user_text" and (e.get("text") or "").strip():
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
        self.session = Session({"instructions": "Transcription only.", "output_modalities": ["text"],
                                "audio": {"input": {"turn_detection": None, "transcription": {"model": "higgs-stt-3.1"}}}})
        self.open = False

    async def start(self):
        await self.session.__aenter__()
        self.open = True
        if SIGNALS:
            from mmhm.models import ProcSignals
            self.signals = ProcSignals(vap=True, bcdet=True, bc=False, wait=False)

    async def stop(self):
        if self.signals:
            self.signals.close()
        if self.open:
            self.open = False
            await close(self.session)

    async def _append(self, pcm):
        await self.session.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm.tobytes()).decode()})
        self.sent += 1

    async def tick(self, t, pcm, others):
        self.speaking, _ = self.vad(pcm)
        if self.signals is not None and self.signals.ready.is_set():
            u16 = to16(pcm) * (1.0 if self.speaking else 0.05)  # noise gate, as in engine.tick
            sig = self.signals.push(u16, to16(others))
            self.p_now, self.p_bc = sig.get("p_now"), sig.get("p_bcdet")
        if not self.open:
            return
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
                asyncio.ensure_future(self._transcribe(time.perf_counter()))
            elif act == "resume":
                self.merge = True
                self.b.on_resume(self)

    async def _transcribe(self, t_end):
        if not spend():
            return
        since = self.session.now()
        await self.session.send({"type": "input_audio_buffer.commit"})
        try:
            _, ev = await self.session.wait("input_audio_buffer.committed", timeout=10, since=since)
            item = ev.get("item_id")
            _, ev = await self.session.wait(lambda e: e.get("type") == "conversation.item.input_audio_transcription.completed"
                                            and e.get("item_id") == item, timeout=30, since=since)
        except asyncio.TimeoutError:
            return
        text = (ev.get("transcript") or "").strip()
        if text:
            merge, self.merge = self.merge, False
            await self.b.on_line(self, text, t_end, merge)


class Reply:
    def __init__(self, line_id, listener, line_text, text, t_end):
        self.line_id, self.listener, self.line_text, self.text, self.t_end = line_id, listener, line_text, text, t_end
        self.audio, self.done, self.pos, self.hist_idx = bytearray(), False, 0, None
        self.video = True        # False while its Avatar render hasn't sent the first chunk (audio waits for it)
        self.cancel = False
        self.started = False     # first audio frame sent
        self.speaker = SentenceSpeaker(VOICE, self.audio, lambda: setattr(self, "done", True))
        spend(max(1, text.count(".") + text.count("?") + text.count("!")))  # roughly one TTS request per sentence
        self.speaker.feed(text)
        self.speaker.close()


class ImprovedBrain:
    """Third Voice (whether) + Mm-hm (when), per person."""

    def __init__(self, agent):
        self.a, self.policy = agent, Policy()
        self.listeners = {}
        self.pools = None
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

    def point(self):
        a = self.a
        return {"target_speaker": a.ai, "setting": a.setting, "people": a.names(), "facts": a.facts,
                "context_turns": [], "current_turn": {"speaker": "", "text": ""}}

    async def _rebuild_pools(self):
        old, p = self.pools, self.point()
        self.pools = {"judge": Pool(tv_eval.dec_prompt(p, tv_eval.DEC_SYS4)[0], SAMPLES),
                      "check": Pool(tv_eval.fact_prompt(p)[0], SAMPLES),
                      "write": Pool(REPLY_SYS.format(ai=p["target_speaker"], setting=p["setting"],
                                                     people=", ".join(p["people"]) or "the group",
                                                     facts=" ".join(p["facts"])), 1)}
        if old:
            for pool in old.values():
                await pool.close()

    async def start(self):
        await self.on_roster(self.a.room.people())

    async def stop(self):
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

    def _cancel_pending(self):
        if self.pending:
            self.pending.speaker.cancel()
            self.pending = None

    async def on_line(self, li, text, t_end, merge):
        a = self.a
        if any(time.time() - t0 < 2.0 and slot != li.slot and repeats(text, [x]) for t0, slot, x in self.recent):
            return  # someone's mic picked up another participant (no headphones): ignore the echo
        self.recent.append((time.time(), li.slot, text))
        if merge and a.history and a.history[-1]["speaker"] == li.name:
            text = a.history.pop()["text"] + " " + text
        self.line_no += 1
        line_id = self.line_no
        self.last_listener = li
        context = list(a.history)
        cur = {"speaker": li.name, "text": text}
        a.history.append(cur)
        r = {**self.point(), "meeting": a.room.id, "line": line_id, "context_turns": context, "current_turn": cur}
        if len(a.facts) > 12:  # big fact sheet: the 8 most relevant facts per line (third_voice/retrieve.py)
            import retrieve
            r["facts"] = retrieve.top_facts(r, 8)
        pools = self.pools
        ctx = "\n".join(f"{t['speaker']}: {t['text']}" for t in context[-14:])
        draft = asyncio.ensure_future(pools["write"].ask(f"Conversation so far:\n{ctx}\n{li.name}: {text}\n\nYour reply:"))
        user, fuser = tv_eval.dec_prompt(r, tv_eval.DEC_SYS4)[1], tv_eval.fact_prompt(r)[1]
        votes = await asyncio.gather(*[pools["judge"].ask(user) for _ in range(SAMPLES)],
                                     *[pools["check"].ask(fuser) for _ in range(SAMPLES)])
        parsed = [tv_eval.parse_json(v) for v in votes[:SAMPLES]]
        facts_hit = [bool((tv_eval.parse_json(v) or {}).get("contradicts")) for v in votes[SAMPLES:]]
        speak = sum(tv_eval.dec_rule(p, a.ai, 2) for p in parsed) * 2 > SAMPLES or (bool(facts_hit) and all(facts_hit))
        why = dict(parsed[0] or {})
        if facts_hit and all(facts_hit):
            why["wrong"] = True
        await a.line(li.slot, li.name, text, {"speak": speak, "why": reason(why, speak), "reading": why,
                                              "decided_s": round(time.perf_counter() - t_end, 2)})
        if not speak:
            li.gate.settle()
            draft.add_done_callback(lambda f: f.exception())
            return
        if why.get("wrong"):  # the draft was written without knowing why Higgs speaks: say the correction
            draft.add_done_callback(lambda f: f.exception())
            draft = asyncio.ensure_future(pools["write"].ask(CORRECT.format(ai=a.ai) + f"Conversation so far:\n{ctx}\n"
                                                             f"{li.name}: {text}\n\nYour reply:"))
        ans = (await draft).strip().strip('"')
        if not ans or repeats(ans, [h["text"] for h in a.history if h["speaker"] == a.ai][-3:]):
            return
        if line_id != self.line_no:
            return  # someone spoke since: the next line's decision sees this one in context
        self._cancel_pending()
        self.pending = Reply(line_id, li, text, ans, t_end)
        await a.status("thinking")

    async def tick(self, frames):
        t = self.t = self.t + TICK
        total = np.zeros(N, np.int32) + self.last_out
        for f in frames.values():
            total += f
        for slot, li in list(self.listeners.items()):
            pcm = frames.get(slot, np.zeros(N, np.int16))
            others = np.clip(total - pcm, -32768, 32767).astype(np.int16)  # VAP's ch2: everyone but this person
            await li.tick(t, pcm, others)
        await self._lab(t)
        anyone = any(li.speaking for li in self.listeners.values())
        self.take.tick(t, anyone)

        r = self.pending
        if self.playing is None and r is not None and len(r.audio) >= N * 2:
            li = r.listener
            act, why = self.take.may_play(t, {"p_now": li.p_now, "silence": li.gate.silence}, anyone, r.line_text,
                                          r.line_id != self.line_no)
            holder = next((x for x in self.listeners.values() if x is not li and x.gate.silence < FLOOR_POLICY.max_wait
                           and (x.p_now is None or x.p_now >= FLOOR_POLICY.th_play)), None)
            if act == "play" and holder is not None:
                act, why = "wait", f"{holder.name} may go on"  # someone else just spoke and still holds the floor
            if act != "drop" and time.perf_counter() - r.t_end > STALE_S:
                act, why = "drop", "stale"
            if act == "play":
                self.pending, self.playing = None, r
                r.hist_idx = len(self.a.history)
                self.a.history.append({"speaker": self.a.ai, "text": r.text})
                li.gate.settle()  # the reply is out: a later word from them is a new line, not a resume
                for li2 in self.listeners.values():
                    li2.barge.reset()
                r.play_why = why
                if self.a.avatar:  # the face has to move with the words: hold the audio until the video starts
                    r.video = False
                    asyncio.ensure_future(self._render(r))
            elif act == "drop":
                self._cancel_pending()
                await self.a.status("listening", dropped=why)

        out = None
        r = self.playing
        if r is not None and not r.video:
            if anyone:  # someone took the floor while the face was being rendered: let them have it
                r.cancel = True
                r.speaker.cancel()
                if r.hist_idx == len(self.a.history) - 1:  # never heard: not part of the meeting
                    self.a.history.pop()
                self.playing = None
                await self.a.room.avatar_stop()
                await self.a.status("listening", dropped="someone spoke while the avatar rendered")
            r = None
        if r is not None:
            if not r.started:
                r.started = True
                await self.a.status("speaking", after_s=round(time.perf_counter() - r.t_end, 2), why=r.play_why)
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
                if self.a.avatar:
                    await self.a.room.avatar_stop()
                await self.a.status("listening")
        self.last_out = out if out is not None else np.zeros(N, np.int16)
        return out

    async def _interrupt(self, r, li):
        r.cancel = True
        r.speaker.cancel()
        if self.a.avatar:
            await self.a.room.avatar_stop()
        heard = floor.truncate(r.text, [(a / SR, b / SR, x) for a, b, x in r.speaker.segments], r.pos / SR)
        if r.hist_idx is not None and r.hist_idx < len(self.a.history):
            self.a.history[r.hist_idx]["text"] = heard
        self.playing = None
        self.gain = 1.0
        await self.a.room.ai_flush()
        await self.a.room.ai_gain(1.0)
        await self.a.status("listening", stopped=f"{li.name} interrupted", heard=heard)

    async def _render(self, r):
        """Render the whole reply through Higgs Avatar once its speech is complete, and stream it to everyone."""
        while not r.done and not r.cancel:
            await asyncio.sleep(0.05)
        if r.cancel or not spend():
            r.video = True
            return
        pcm = np.frombuffer(bytes(r.audio[:len(r.audio) // 2 * 2]), np.int16)
        loop = asyncio.get_running_loop()
        await self.a.room.avatar_start(avatar.MIME)

        def on_chunk(c):
            def deliver():
                if not r.cancel:
                    asyncio.ensure_future(self.a.room.send_avatar_chunk(c))
                    r.video = True
            loop.call_soon_threadsafe(deliver)
        try:
            await asyncio.to_thread(avatar.render, pcm, on_chunk, lambda: r.cancel)
        except Exception as e:  # noqa: BLE001  (no face this time: speak anyway)
            await self.a.room.event({"type": "ai_error", "error": f"avatar: {e!r}"[:300]})
            r.video = True

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
        b = room.brief
        self.ai, self.setting, self.facts = b.get("ai") or "Higgs", b.get("setting") or "a meeting", list(b.get("facts") or [])
        self.history = []
        self.brain = None
        self.mode = getattr(room, "mode", "improved")
        self.inq = {}
        self.task = None
        self.running = False
        self.switching = None
        self.avatar = bool(getattr(room, "avatar", False))

    def names(self):
        return [p["name"] for p in self.room.people()]

    def persona(self):
        return PERSONA.format(ai=self.ai, setting=self.setting, people=", ".join(self.names()) or "the group",
                              facts=" ".join(self.facts))

    async def status(self, s, **kw):
        await self.room.event({"type": "ai_state", "mode": self.mode, "avatar": getattr(self.room, "avatar", False),
                               "status": s, "calls": CALLS["n"], **kw})

    async def line(self, slot, name, text, decision):
        await self.room.event({"type": "line", "slot": slot, "name": name, "text": text, "decision": decision,
                               "mode": self.mode})

    def _brain(self):
        return ImprovedBrain(self) if self.mode == "improved" else DefaultBrain(self)

    async def start(self):
        self.brain = self._brain()
        try:
            await self.brain.start()
        except Exception as e:  # noqa: BLE001  (Higgs unreachable: the humans can still talk to each other)
            await self.room.event({"type": "ai_error", "error": f"start: {e!r}"[:300]})
            self.brain = None
        self.running = True
        self.task = asyncio.ensure_future(self._run())
        await self.status("listening")

    async def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()
        if self.brain:
            await self.brain.stop()
            self.brain = None

    def feed(self, slot, pcm):
        q = self.inq.setdefault(slot, deque(maxlen=25))
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
        old, self.brain = self.brain, None
        if old:
            await old.stop()
        await self.room.ai_flush()
        await self.room.ai_gain(1.0)
        if self.running:
            b = self._brain()
            try:
                await b.start()
                self.brain = b
            except Exception as e:  # noqa: BLE001
                await self.room.event({"type": "ai_error", "error": f"switch: {e!r}"[:300]})
        await self.status("listening")

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
        while len(q) > 6:  # a burst after network jitter: stay near real time
            q.popleft()
        return q.popleft()

    async def _run(self):
        t0, k = time.perf_counter(), 0
        while self.running:
            k += 1
            frames = {slot: self._pop(slot) for slot in list(self.inq)}
            brain = self.brain
            if brain is not None:
                try:
                    out = await brain.tick(frames)
                except Exception as e:  # noqa: BLE001  (keep the room alive; the log says what broke)
                    await self.room.event({"type": "ai_error", "error": repr(e)[:300]})
                    out = None
                if out is not None:
                    await self.room.send_ai_audio(out.astype(np.int16).tobytes())
            delay = t0 + k * TICK - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -0.5:  # fell far behind (a slow tick): resync instead of racing to catch up
                t0, k = time.perf_counter(), 0
