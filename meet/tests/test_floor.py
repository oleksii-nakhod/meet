"""Synthetic 20 ms sequences through the floor-timing state machines."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from floor import TICK, BargeIn, LineGate, Policy, TakeFloor, combine, truncate  # noqa: E402

P = Policy()


def run(machine, frames, t0=0.0, **kw):
    """frames: [(seconds, speaking, signal)]. Returns [(t, output)] for every tick."""
    out, t = [], t0
    for dur, speaking, sig in frames:
        for _ in range(round(dur / TICK)):
            out.append((round(t, 6), machine.update(round(t, 6), speaking, sig, **kw)))
            t += TICK
    return out


def commits(trace):
    return [t for t, acts in trace if "commit" in acts]


# --- LineGate -----------------------------------------------------------------------------------------------------
def test_mid_thought_pause_waits_past_max_wait():
    g = LineGate()
    tr = run(g, [(1.0, True, 0.9), (2.0, False, 0.9)])  # p_now stays high: they're mid-thought
    c = commits(tr)
    assert c and c[0] - 1.0 >= P.max_wait
    assert abs((c[0] - 0.98) - P.give_up) < 1e-6 and g.why == "timeout"


def test_finished_turn_commits_after_min_silence():
    g = LineGate()
    tr = run(g, [(1.0, True, 0.9), (0.6, False, 0.05)])
    c = commits(tr)
    assert len(c) == 1 and g.why == "gate"
    assert abs((c[0] - 0.98) - P.min_silence) < 1e-6  # 0.98 = their last voiced tick


def test_no_model_falls_back_to_max_wait():
    g = LineGate()
    c = commits(run(g, [(1.0, True, None), (1.5, False, None)]))
    assert len(c) == 1 and abs((c[0] - 0.98) - P.max_wait) < 1e-6 and g.why == "timeout"


def test_too_short_to_be_a_turn():
    g = LineGate()
    assert not commits(run(g, [(0.1, True, 0.9), (2.0, False, 0.01)]))


def test_resume_after_commit_makes_next_commit_a_continuation():
    g = LineGate()
    tr = run(g, [(1.0, True, 0.9), (0.3, False, 0.05), (0.5, True, 0.9), (0.6, False, 0.05)])
    acts = [(t, a) for t, acts in tr for a in acts]
    assert [a for _, a in acts] == ["commit", "resume", "commit"]
    t_resume = acts[1][0]
    assert abs(t_resume - (1.3 + P.resume_cancel - TICK)) < 1e-6  # resume_cancel of speech after they resumed
    assert g.merge  # the second commit continues the first piece
    assert g.last_end is not None and g.silence > 0


def test_a_click_after_commit_is_not_a_resume_and_settle_ends_pending():
    g = LineGate()
    tr = run(g, [(1.0, True, 0.9), (0.3, False, 0.05), (0.06, True, 0.9), (0.5, False, 0.05)])
    assert [a for _, acts in tr for a in acts] == ["commit"]
    g.settle()
    tr = run(g, [(0.5, True, 0.9)], t0=2.0)
    assert not any(acts for _, acts in tr)


# --- TakeFloor ----------------------------------------------------------------------------------------------------
class Who:  # a last speaker's state as LineGate exposes it
    def __init__(self, silence, p_now):
        self.silence, self.p_now = silence, p_now


def test_waits_while_someone_else_starts_talking():
    f = TakeFloor()
    assert f.may_play(10.0, Who(0.5, 0.05), True, "Let's ship it.", False) == ("wait", "someone speaking")
    # they stopped 40 ms ago: the room hasn't been quiet for min_silence yet
    assert f.may_play(10.04, Who(0.54, 0.05), False, "Let's ship it.", False) == ("wait", "silence")
    assert f.may_play(10.2, Who(0.7, 0.05), False, "Let's ship it.", False) == ("play", "p_now")


def test_drops_on_newer_line():
    f = TakeFloor()
    assert f.may_play(5.0, Who(1.0, 0.01), False, "Done.", True)[0] == "drop"


def test_syntactic_hold_until_give_up():
    f, text = TakeFloor(), "I think we should move it to the"
    for s in (0.2, 0.8, 1.2, P.give_up - TICK):
        assert f.may_play(s, Who(s, 0.01), False, text, False) == ("wait", "unfinished")
    assert f.may_play(P.give_up, Who(P.give_up, 0.01), False, text, False)[0] == "play"


def test_plays_when_p_now_low_and_holds_when_high():
    f = TakeFloor()
    assert f.may_play(1.0, Who(0.12, P.th_play - 0.01), False, "Sounds good.", False) == ("play", "p_now")
    f = TakeFloor()
    assert f.may_play(1.0, Who(1.0, 0.5), False, "Sounds good.", False) == ("wait", "hold")
    assert f.may_play(2.0, Who(P.give_up, 0.5), False, "Sounds good.", False) == ("play", "give_up")
    f = TakeFloor()
    assert f.may_play(1.0, Who(P.max_wait, None), False, "Sounds good.", False) == ("play", "max_wait")


# --- BargeIn ------------------------------------------------------------------------------------------------------
def events(trace):
    return [(t, o["event"]) for t, o in trace if o["event"]]


def test_mmhm_is_a_backchannel():
    b = BargeIn()
    tr = run(b, [(0.3, True, 0.8), (0.5, False, 0.1)])
    assert tr[0][1]["gain"] == P.duck_gain  # ducked at onset
    assert [e for _, e in events(tr)] == ["backchannel"]
    assert tr[-1][1]["gain"] == 1.0


def test_sustained_speech_is_an_interrupt():
    b = BargeIn()
    tr = run(b, [(1.2, True, 0.1)])
    ev = events(tr)
    assert [e for _, e in ev] == ["interrupt"] and abs(ev[0][0] - P.interrupt_max) < 1e-6
    # with VAP saying they take the floor it comes sooner, after interrupt_after
    b = BargeIn()
    ev = events(run(b, [(1.2, True, 0.1)], p_now=0.8))
    assert [e for _, e in ev] == ["interrupt"] and abs(ev[0][0] - P.interrupt_after) < 1e-6
    # Higgs is stopped: nothing more until its next reply
    assert not events(run(b, [(0.06, True, 0.0), (0.5, False, 0.0)], t0=2.0))
    b.reset()
    assert [e for _, e in events(run(b, [(0.06, True, 0.0), (0.5, False, 0.0)], t0=3.0))] == ["noise"]


def test_blip_is_noise():
    b = BargeIn()
    tr = run(b, [(0.06, True, 0.0), (0.5, False, 0.0)])
    assert [e for _, e in events(tr)] == ["noise"]
    assert tr[-1][1]["gain"] == 1.0


def test_fragmented_hesitant_speech_is_an_interrupt():
    b = BargeIn()
    frames = [(0.2, True, 0.1), (0.3, False, 0.1)] * 6  # "yeah, I think we, um, it..."
    ev = [e for _, e in events(run(b, frames))]
    assert "interrupt" in ev and ev[-1] == "interrupt" and ev.count("interrupt") == 1
    assert all(e == "noise" for e in ev[:-1])  # each fragment alone looked like a noise


def test_combine():
    assert combine([1.0, P.duck_gain, 1.0]) == P.duck_gain
    assert combine([]) == 1.0


# --- truncate -----------------------------------------------------------------------------------------------------
SEGS = [(0.0, 2.0, "First point."), (2.0, 4.0, "Second point."), (4.0, 6.0, "Third point.")]
TEXT = "First point. Second point. Third point."


def test_truncate_cuts_at_the_right_sentence():
    assert truncate(TEXT, SEGS, 3.0) == "First point. —"       # second sentence only half heard
    assert truncate(TEXT, SEGS, 3.3) == "First point. Second point. —"  # 65 % of it heard
    assert truncate(TEXT, SEGS, 0.5) == "—"
    assert truncate(TEXT, SEGS, 9.0) == TEXT + " —"
