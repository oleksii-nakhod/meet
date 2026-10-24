"""Floor timing for a group meeting: the Mm-hm engine's turn-taking rules, ported from one user to many.

Each human has their own audio stream, VAD and turn-taking signals (VAP `p_now` = probability they keep the floor,
backchannel detector `p_bc`). Whether Higgs *should* speak is decided elsewhere (an LLM judge); this module only
decides *when*:
- `LineGate` (one per human): when is this person's line finished, so we can commit it for transcription + judging.
  Same rule as the engine's listening branch (gate on `p_now`, fall back to a timeout), plus the speculating branch's
  "they weren't done" cancel.
- `TakeFloor` (one per room): may a ready reply start playing now? The engine's play rule, but nobody in the room may
  be talking, and a line committed after the reply was decided makes it stale.
- `BargeIn` (one per human, while Higgs talks): duck, then decide backchannel / interruption / noise, exactly as the
  engine's playing branch. `combine` mixes the per-person gain targets: the most cautious one wins.
- `truncate`: what the listeners actually heard of an interrupted reply, like `SentenceSpeaker.heard_text`.

Pure and synchronous: no I/O, time is the caller's `t` (seconds), ticked every 20 ms. Parameters are the engine's
`Policy` (same names and defaults), so thresholds tuned on the 1:1 engine carry over."""
from mmhm.engine import TICK, VAD, Policy, unfinished  # noqa: F401  (VAD: one per human stream -> `speaking`)


def _dt(prev, t):
    """Seconds since the previous tick (a first tick counts as one tick), rounded so 50 ticks sum to exactly 1 s."""
    return TICK if prev is None else round(max(0.0, t - prev), 6)


def _get(state, k):
    return state.get(k) if isinstance(state, dict) else getattr(state, k, None)


class LineGate:
    """One per human. Decides when their line is finished.

    update() returns a list of actions:
    - "commit": the line ended; commit the audio since the last commit (`why` is "gate" or "timeout").
    - "resume": they started talking again while the last commit's decision was still pending (speech adding up to
      `resume_cancel`, like the engine's speculation cancel). That decision is stale: drop it, and treat the next
      commit as a continuation (`merge` is True on that commit).
    A commit stays pending until settle() (call it when the reply starts playing or the judge's verdict is final), so a
    late resume after Higgs has answered is a new line, not a continuation."""

    def __init__(self, policy=None):
        self.p = policy or Policy()
        self.t = None
        self.speech = 0.0        # speech seconds since the last commit
        self.silence = 99.0      # seconds since this person last made a sound
        self.last_end = None     # t of their last voiced tick
        self.p_now = None        # latest VAP p_now (None: no model)
        self.pending = False     # a commit whose decision may still be overtaken
        self.resume_s = 0.0      # consecutive speech seconds while pending
        self.resumed = False     # a resume since the last commit: the next commit continues the previous piece
        self.why, self.merge = None, False  # about the latest commit, for logging / merging

    def update(self, t, speaking, p_now):
        p, dt = self.p, _dt(self.t, t)
        self.t, self.p_now = t, p_now
        acts = []
        if speaking:
            self.silence, self.last_end = 0.0, t
            self.speech = round(self.speech + dt, 6)
        else:
            self.silence = round(t - self.last_end, 6) if self.last_end is not None else 99.0

        if self.pending:
            self.resume_s = round(self.resume_s + dt, 6) if speaking else 0.0
            if self.resume_s >= p.resume_cancel:
                # they weren't done: the pending decision is stale, and what follows continues the same line
                self.pending, self.resume_s, self.resumed = False, 0.0, True
                self.speech = max(self.speech, p.min_turn)  # they've already said something this turn
                acts.append("resume")
            return acts

        if not speaking and self.speech >= p.min_turn:
            if self.silence >= p.min_silence and p_now is not None and p_now < p.th_start:
                acts.append(self._commit("gate"))
            elif self.silence >= p.give_up or (self.silence >= p.max_wait and (p_now is None or p_now < p.hold_p)):
                # no model, or the model isn't sure they're mid-thought, by max_wait; whatever it says, by give_up
                acts.append(self._commit("timeout"))
        return acts

    def _commit(self, why):
        self.why, self.merge, self.resumed = why, self.resumed, False
        self.speech, self.pending, self.resume_s = 0.0, True, 0.0
        return "commit"

    def settle(self):
        """The last commit's decision is final (Higgs started playing, or the judge passed): no more resumes."""
        self.pending, self.resume_s = False, 0.0


class TakeFloor:
    """One per room. May a SPEAK reply that is ready start playing?

    may_play() returns (action, why) with action "play" | "wait" | "drop":
    - "drop": a newer line was committed since this reply was decided (the conversation moved on).
    - "wait": someone is talking, or the room hasn't been quiet for `min_silence`, or the last speaker's words end
      mid-sentence (syntactic hold, until `give_up`), or their p_now says they're mid-thought.
    - "play": the engine's rule on the last speaker: p_now < th_play, or give_up silence, or max_wait silence with
      p_now < hold_p (or no model).
    `last_speaker_state` is that person's LineGate (or any object/dict with `silence` and `p_now`). The room's own
    quiet is tracked from `anyone_speaking`; call tick() every tick too so speech before the reply was ready counts."""

    def __init__(self, policy=None):
        self.p = policy or Policy()
        self.last_voice = None   # t when anyone in the room last made a sound

    def tick(self, t, anyone_speaking):
        if anyone_speaking:
            self.last_voice = t

    def may_play(self, t, last_speaker_state, anyone_speaking, transcript, newer_line):
        p = self.p
        self.tick(t, anyone_speaking)
        if newer_line:
            return "drop", "newer line"
        if anyone_speaking:
            return "wait", "someone speaking"
        silence = _get(last_speaker_state, "silence")
        silence = 99.0 if silence is None else silence
        if self.last_voice is not None:
            silence = min(silence, round(t - self.last_voice, 6))
        if silence < p.min_silence:
            return "wait", "silence"
        if p.syntax_hold and unfinished(transcript) and silence < p.give_up:
            return "wait", "unfinished"
        p_now = _get(last_speaker_state, "p_now")
        if p_now is not None and p_now < p.th_play:
            return "play", "p_now"
        if silence >= p.give_up:
            return "play", "give_up"
        if silence >= p.max_wait and (p_now is None or p_now < p.hold_p):
            return "play", "max_wait"
        return "wait", "hold"


class BargeIn:
    """One per human, while Higgs is speaking: duck, then decide.

    On a sound, the target gain drops to `duck_gain` at once; then:
    - backchannel: p_bc reached `bc_threshold` and the sound ended within 1 s -> restore, forget it;
    - interrupt: speech for `interrupt_after` (+0.25 s right after a backchannel: a second "mhm" is likely) with VAP
      p_now > `floor_p` (they're taking the floor, not laughing along), or for `interrupt_max` whatever the models say,
      or speech adding up to `interrupt_total` across fragments separated by under 1 s of quiet ("yeah, I think we,
      um, it") -> stop Higgs and truncate;
    - noise: the sound ended without being a backchannel and they've been quiet 0.25 s (cough, laugh, blip) ->
      restore.
    update() returns {"gain": target, "event": None|"backchannel"|"interrupt"|"noise", "why": str|None,
    "dur": seconds since onset}. After an interrupt Higgs is stopped and the machine stays quiet until reset(); call
    reset() whenever Higgs starts a new reply (the engine also resets the fragment sum then)."""

    def __init__(self, policy=None):
        self.p = policy or Policy()
        self.t = None
        self.silence = 99.0
        self.last_heard = -99.0  # when we last let one of their backchannels pass
        self.reset()

    def reset(self):
        self.onset, self.bc_max, self.gain = None, 0.0, 1.0
        self.frag_speech, self.frag_quiet = 0.0, 99.0
        self.stopped = False     # we interrupted Higgs: nothing more to decide until its next reply

    def _out(self, event=None, why=None, dur=0.0):
        return {"gain": self.gain, "event": event, "why": why, "dur": round(dur, 3)}

    def _decided(self, event, why, dur, now):
        """Restore full volume (the engine also restores after stopping) and close the decision window."""
        self.onset, self.gain = None, 1.0
        self.stopped = event == "interrupt"
        if event == "backchannel":
            self.last_heard = now
        return self._out(event, why, dur)

    def update(self, t, speaking, p_bc, words_so_far=None, p_now=None, bc_lag=0.0):
        """p_bc: backchannel *detector* (the engine's `p_bcdet`). bc_lag: seconds of audio the detector hasn't seen yet.
        p_now: VAP for this person, if available; without it
        only `interrupt_max` and the fragment rule can interrupt. words_so_far (streaming STT, if any): 3+ words is
        more than a backchannel, so it stands in for VAP agreeing once `interrupt_after` has passed."""
        p, now, dt = self.p, t, _dt(self.t, t)
        self.t = t
        self.silence = 0.0 if speaking else round(self.silence + dt, 6)
        if self.stopped:
            return self._out()
        # hesitant interruptions come in fragments; add them up until a second of quiet
        self.frag_quiet = 0.0 if speaking else round(self.frag_quiet + dt, 6)
        if speaking:
            self.frag_speech = round(self.frag_speech + dt, 6)
        elif self.frag_quiet > 1.0:
            self.frag_speech = 0.0
        if self.frag_speech >= p.interrupt_total and speaking:
            dur = self.frag_speech
            self.frag_speech = 0.0
            return self._decided("interrupt", "fragments", dur, now)
        if self.onset is None and speaking:
            self.onset, self.bc_max, self.gain = now, 0.0, p.duck_gain
        if self.onset is None:
            return self._out()
        self.bc_max = max(self.bc_max, p_bc or 0.0)
        dur = round(now - self.onset, 6)
        patience = p.interrupt_after + (0.25 if now - self.last_heard < 1.0 else 0.0)
        # a backchannel detector running behind real time (a busy machine) hasn't heard the sound yet: give it that long
        # (a loaded run stopped Higgs for a 0.55 s "Mm-hm." that passed twice on an idle machine); interrupt_max caps it
        patience += min(max(0.0, bc_lag), 0.4)
        floor = (p_now is not None and p_now > p.floor_p) or (words_so_far is not None and words_so_far >= 3)
        if self.bc_max >= p.bc_threshold and dur < 1.0:
            if not speaking:  # the backchannel is over: carry on at full volume, forget it
                return self._decided("backchannel", "p_bc", dur, now)
        elif speaking and ((dur >= patience and floor) or dur >= p.interrupt_max):
            return self._decided("interrupt", "floor" if dur < p.interrupt_max else "sustained", dur, now)
        elif not speaking and self.silence > 0.25:
            return self._decided("noise", "short", dur, now)
        return self._out(dur=dur)


def combine(gains):
    """Room gain target for Higgs's voice: anyone ducking it wins (min of the per-person targets)."""
    return min(gains, default=1.0)


def truncate(text, segments, pos_s):
    """What the listeners heard of a reply stopped at playback position `pos_s`: the sentences fully or mostly
    (60 %) played, plus a dash, like the engine's `heard_text(...) + " —"`. segments: [(start_s, end_s, sentence)]
    as SentenceSpeaker records them, converted to seconds. `text` (the full reply) is not needed for the cut, the
    segments carry the sentences; it is in the signature so callers log what was lost. Of the sentence it was cut in,
    the words played so far (in proportion to its time) are kept if there are 3 or more: cut early in a long first
    sentence, the reply was remembered as "—", as if nothing had been said."""
    heard = [s for a, b, s in segments if a + 0.6 * (b - a) <= pos_s]
    cur = next(((a, b, s) for a, b, s in segments if a <= pos_s < a + 0.6 * (b - a)), None)
    if cur is not None:
        a, b, s = cur
        words = s.split()
        k = int(len(words) * (pos_s - a) / max(1e-6, b - a))
        if k >= 3:
            heard.append(" ".join(words[:k]).rstrip(",;:"))
    return (" ".join(heard) + " —").lstrip()
