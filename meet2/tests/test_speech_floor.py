"""Speech output and the floor controller, with fake transports and voices (no API calls)."""
import asyncio

import numpy as np

from meet.floor import Floor, RoomNow
from meet.speech import OutputWorker, Synthesizer, Utterance, split_sentences

SR = 24000


class FakeTransport:
    name = "fake"

    def __init__(self, fail_frames=(), delay=0.02):
        self.frames, self.cleared, self.ended = [], 0, 0
        self.fail_frames, self.delay = set(fail_frames), delay
        self.cb = None

    async def send_ai_audio(self, frame):
        n = len(self.frames)
        self.frames.append(frame)
        if n in self.fail_frames:
            raise ConnectionError("network blip")

    async def end_ai_segment(self):
        self.ended += 1

    async def clear_ai_audio(self):
        self.cleared += 1

    def playout_delay(self):
        return self.delay

    def on_playback_finished(self, cb):
        self.cb = cb


def tone(seconds, value):
    return (np.full(int(SR * seconds), value, np.int16)).tobytes()


def voice(seconds_per_sentence=0.1, fail=None):
    """A fake TTS: each sentence becomes a constant tone whose value is the sentence's index (+1)."""
    calls = {"n": 0}

    async def tts(text):
        calls["n"] += 1
        idx = int(text.split()[0].strip("S.")) if text.startswith("S") else 1
        audio = tone(seconds_per_sentence, idx)
        for k in range(0, len(audio), 960):
            if fail and fail(text, k, calls["n"]):
                raise RuntimeError("voice failed")
            yield audio[k:k + 960]
            await asyncio.sleep(0)
    return tts, calls


def test_split_sentences():
    assert split_sentences("Hello there. How are") == (["Hello there."], "How are")
    assert split_sentences("It costs $2.40 each way. OK") == (["It costs $2.40 each way."], "OK")


async def play(u, tts, transport, **kw):
    log = []
    w = OutputWorker(transport, log=log.append)
    w.start()
    synth = Synthesizer(tts, on_error=lambda e: log.append({"type": "ai_error", "error": e}), **kw)
    done = asyncio.Event()
    asyncio.ensure_future(synth.run(u))
    w.play(u, lambda _u: done.set())
    await asyncio.wait_for(done.wait(), 10)
    await w.stop()
    return log, w


def test_plays_every_sentence_in_order_then_reports_heard():
    async def go():
        u = Utterance()
        u.feed("S1 one. S2 two. S3")
        u.feed(" three.")
        u.close()
        t = FakeTransport()
        tts, _ = voice(0.1)
        log, _ = await play(u, tts, t)
        pcm = np.frombuffer(b"".join(t.frames), np.int16)
        values = [int(v) for v in pcm if v]
        assert values == sorted(values) and set(values) == {1, 2, 3}     # in order, nothing repeated
        assert len(t.frames) == 15 and t.ended == 1 and u.heard.is_set()
        assert [e["type"] for e in log] == ["speech_start"]
    asyncio.run(go())


def test_a_send_error_does_not_kill_the_output_loop():
    async def go():
        u = Utterance()
        u.feed("S1 one. S2 two.")
        u.close()
        t = FakeTransport(fail_frames={2, 3, 4})
        log, w = await play(u, voice(0.1)[0], t)
        assert u.heard.is_set() and len(t.frames) == 10 and w.errors == 3
        assert sum(e["type"] == "ai_error" for e in log) == 3
    asyncio.run(go())


def test_a_voice_failure_after_audio_is_out_never_rewinds():
    async def go():
        u = Utterance()
        u.feed("S1 one. S2 two. S3 three.")
        u.close()
        fail = lambda text, k, n: text.startswith("S2") and k >= 960 * 2     # S2 dies after 2 chunks
        t = FakeTransport()
        log, _ = await play(u, voice(0.1, fail)[0], t)
        values = [int(v) for v in np.frombuffer(b"".join(t.frames), np.int16) if v]
        assert values == sorted(values) and 3 not in values               # monotonic; ends cleanly at the failure
        assert values.count(2) == 960                                      # exactly the part of S2 that existed
        assert any("voice failed" in e.get("error", "") for e in log)
    asyncio.run(go())


def test_a_voice_failure_before_any_audio_is_retried():
    async def go():
        u = Utterance()
        u.feed("S1 one.")
        u.close()
        fail = lambda text, k, n: n == 1 and k == 0                        # first attempt fails at once
        tts, calls = voice(0.1, fail)
        t = FakeTransport()
        await play(u, tts, t)
        assert calls["n"] == 2 and len(t.frames) == 5
    asyncio.run(go())


def test_interrupt_reports_what_was_heard():
    async def go():
        u = Utterance()
        u.feed("S1 first sentence here. S2 second sentence goes on and on.")
        u.close()
        t = FakeTransport(delay=0.0)
        w = OutputWorker(t)
        w.start()
        asyncio.ensure_future(Synthesizer(voice(0.5)[0]).run(u))
        w.play(u, lambda _u: None)
        await asyncio.sleep(0.75)                                          # S1 played, about half of S2
        heard = await w.interrupt()
        assert t.cleared == 1 and u.interrupted and not w.speaking
        text = u.heard_text(heard)
        assert text.startswith("S1 first sentence here. S2") and text.endswith("…")
        await w.stop()
    asyncio.run(go())


# ------------------------------------------------------------------ floor
class FakePlayer:
    def __init__(self):
        self.playing, self.cb, self.interrupts = None, None, 0

    @property
    def speaking(self):
        return self.playing is not None

    def play(self, u, on_heard):
        self.playing, self.cb = u, on_heard

    async def interrupt(self):
        self.interrupts += 1
        u, self.playing = self.playing, None
        if u:
            u.cancel()
            self.cb(u)
        return 0

    def heard_samples(self):
        return 0

    def finish(self):
        u, self.playing = self.playing, None
        self.cb(u)


def make_floor():
    events, gains, released = [], [], []
    player = FakePlayer()
    f = Floor(player, emit=events.append, set_human_gain=lambda g, r: gains.append(g),
              release_buffer=lambda: released.append(True))
    return f, player, events, gains, released


def test_a_reply_waits_for_a_gap():
    f, p, events, _, _ = make_floor()
    u = Utterance()
    f.propose(u, line_id="L1", slot=1, addressed=True, now=0.0)
    f.tick(0.1, RoomNow(speaking={2}, in_turn={2}, quiet_for=0.0))
    assert p.playing is None and f.state == "waiting"
    f.tick(0.3, RoomNow(quiet_for=0.2))
    assert p.playing is None
    f.tick(0.5, RoomNow(quiet_for=0.4))
    assert p.playing is u and f.state == "speaking"
    p.finish()
    assert f.state == "idle"


def test_fillers_keep_the_ai_talking_and_interruptions_stop_it():
    async def go():
        f, p, events, _, _ = make_floor()
        f.propose(Utterance(), line_id="L1", slot=1, addressed=True, now=0.0)
        f.tick(1.0, RoomNow())
        assert not f.bargein(2, "filler", "umm okay")
        assert not f.bargein(2, "unclear", "Friday")
        assert p.playing is not None and p.interrupts == 0
        assert f.bargein(2, "interrupt", "what about Friday")
        await asyncio.sleep(0)
        assert p.interrupts == 1 and f.state == "idle"
        assert any(e["type"] == "interrupted" for e in events)
    asyncio.run(go())


def test_a_correction_cuts_in_ducks_buffers_and_releases_after_it_is_heard():
    async def go():
        f, p, events, gains, released = make_floor()
        reply = Utterance()
        f.propose(reply, line_id="L1", slot=1, addressed=True, now=0.0)
        f.tick(1.0, RoomNow())
        assert p.playing is reply
        fix = Utterance("correction")
        f.correct(fix, line_id="L2", slot=2, now=1.1)
        f.tick(1.15, RoomNow(speaking={2}, in_turn={2}, quiet_for=0.0, paused={2: 0.0}))
        assert p.playing is reply                                  # the speaker hasn't paused: wait a moment
        f.tick(1.4, RoomNow(in_turn={2}, quiet_for=0.3, paused={2: 0.3}))   # a clause pause: cut in
        await asyncio.sleep(0)
        assert p.playing is fix and f.state == "correcting" and gains == [0.15] and reply.cancelled
        assert not f.bargein(3, "interrupt", "no no that's not right")       # nobody talks over a correction ...
        assert p.playing is fix
        p.finish()
        assert gains == [0.15, 1.0] and released == [True] and f.state == "idle"
    asyncio.run(go())


def test_a_correction_talks_over_someone_who_never_pauses():
    async def go():
        f, p, _, gains, _ = make_floor()
        fix = Utterance("correction")
        f.correct(fix, line_id="L2", slot=2, now=0.0)
        f.tick(0.5, RoomNow(speaking={2}, in_turn={2}, quiet_for=0.0, paused={2: 0.0}))
        assert p.playing is None
        f.tick(1.3, RoomNow(speaking={2}, in_turn={2}, quiet_for=0.0, paused={2: 0.0}))
        assert p.playing is fix and gains == [0.15]
    asyncio.run(go())


def test_the_host_can_stop_a_correction():
    async def go():
        f, p, _, gains, released = make_floor()
        f.correct(Utterance("correction"), line_id="L2", slot=2, now=0.0)
        f.tick(0.1, RoomNow())
        assert not f.bargein(3, "stop", "stop", host=False)
        assert f.bargein(1, "stop", "stop", host=True)
        await asyncio.sleep(0)
        assert p.playing is None and gains[-1] == 1.0 and released
    asyncio.run(go())


def test_stale_and_moved_on_replies_are_dropped():
    f, p, events, _, _ = make_floor()
    u = Utterance()
    f.propose(u, line_id="L1", slot=1, addressed=False, now=0.0)
    f.moved_on("L2", 2, 0.5)
    assert u.cancelled and f.pending is None
    u2 = Utterance()
    f.propose(u2, line_id="L3", slot=1, addressed=False, now=1.0)
    f.tick(4.6, RoomNow(speaking={2}, quiet_for=0.0))
    assert u2.cancelled and p.playing is None
    assert [e["reason"] for e in events if e["type"] == "reply_skipped"] == ["the conversation moved on", "no gap to speak in"]


def test_a_correction_talks_over_people_for_the_guard_then_yields_to_a_real_interruption():
    import time

    async def go():
        f, p, events, gains, released = make_floor()
        fix = Utterance("correction")
        f.correct(fix, line_id="L2", slot=2, now=0.0)
        f.tick(0.1, RoomNow())
        fix.started_at = time.monotonic() - 0.5                 # 0.5 s in: inside the 2 s guard
        assert not f.bargein(3, "interrupt", "no that's not what I said")
        assert not f.bargein(3, "stop", "wait")                  # not the host
        fix.started_at = time.monotonic() - 2.5                 # past the guard
        assert not f.bargein(3, "filler", "umm okay")            # fillers never stop it
        assert not f.bargein(3, "interrupt", "no that's not what I said")   # a line running past the guard ...
        f._insist[3] -= 0.7                                      # ... but still pressing 0.7 s later
        assert f.bargein(3, "interrupt", "no that's not what I said at all")
        await asyncio.sleep(0)
        assert p.playing is None and gains[-1] == 1.0 and released and f.state == "idle"
    asyncio.run(go())
