"""The improved brain's text rules (no API): name repair, pre-filters, the v5 decision rule."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent  # noqa: E402


def r(people, text, ctx=()):
    return {"target_speaker": "Higgs", "people": people, "context_turns": [{"speaker": s, "text": t} for s, t in ctx],
            "current_turn": {"speaker": people[0], "text": text}}


def test_fix_name():
    assert agent.fix_name("Pigs, how should we split the work?", "Higgs") == "Higgs, how should we split the work?"
    assert agent.fix_name("Hey Hicks, are you there?", "Higgs") == "Hey Higgs, are you there?"
    assert agent.fix_name("thanks, Hicks.", "Higgs") == "thanks, Higgs."
    assert agent.fix_name("Pigs are cute, right?", "Higgs") == "Pigs are cute, right?"  # not set off by a comma
    assert agent.fix_name("Okay, thanks.", "Higgs") == "Okay, thanks."
    assert agent.fix_name("Eva, can you check it?", "Ava", ["Eva", "Tom"]) == "Eva, can you check it?"  # a person
    assert agent.fix_name("Eva, can you check it?", "Ava", ["Tom"]) == "Ava, can you check it?"
    assert agent.fix_name("Pigs. And what now?", "Higgs") == "Higgs. And what now?"


def test_prefilter():
    pf = lambda t, h=(): agent.prefilter(t, "Higgs", [{"speaker": s, "text": x} for s, x in h])[1]  # noqa: E731
    assert pf("...") == "skip" and pf("嗯。") == "skip"
    assert pf("Mm-hm.") == "backchannel" and pf("Yeah, yeah.") == "backchannel" and pf("Got it.") == "backchannel"
    assert pf("Yeah.", [("Higgs", "Want me to go on?")]) is None  # answers the AI's question
    assert pf("Yeah?") is None
    assert pf("um") == "filler" and pf("so the, uh, the") == "filler"
    assert pf("so basically what i'm trying to") == "unfinished"
    assert pf("Higgs, stop.") == "stop" and pf("Pigs stop.") == "stop" and pf("Hold on.") == "stop"
    assert pf("Stop sharing your screen, Priya.") is None
    for t in ("Go on.", "Tell me more about that.", "what do you mean by that", "Hey, Higgs.", "Okay, thanks."):
        assert pf(t) is None, t


def test_join_pieces():
    assert agent.join_pieces("I think the judges will.", "Want to see it running live?") == \
        "I think the judges will want to see it running live?"
    assert agent.join_pieces("Higgs, I was wondering whether.", "We should show the switch.") == \
        "Higgs, I was wondering whether we should show the switch."
    assert agent.join_pieces("Sorry, I was away.", "Higgs, when do we close?") == "Sorry, I was away. Higgs, when do we close?"


def test_facts_relevant():
    facts = ["Submissions close Saturday at 3 pm.", "The project is a video call where Higgs joins."]
    assert agent.facts_relevant("When do submissions close?", [], facts, ["Higgs", "Alex"])
    assert not agent.facts_relevant("How's it going?", [], facts, ["Higgs", "Alex"])
    assert not agent.facts_relevant("Higgs, are you there?", [], facts, ["Higgs", "Alex"])


def test_volunteers():
    assert agent.VOLUNTEERS.search("I can handle the backend integration.")
    assert not agent.VOLUNTEERS.search("You could handle the backend; I can remind you of the deadline.")


def test_rule_v5():
    rule = lambda d, x: agent.judge_rule(d, x, "v5")  # noqa: E731
    me = {"to": "me", "ask": False, "reply": True}
    assert rule(me, r(["Alex", "Priya"], "Hey, Higgs."))  # greeted: v4b said no
    assert not agent.judge_rule(me, r(["Alex", "Priya"], "Hey, Higgs."), "v4b")
    assert not rule({"to": "me", "ask": False, "reply": False}, r(["Alex", "Priya"], "Thanks, Higgs."))
    assert rule({"to": "me", "ask": False, "reply": False}, r(["Alex"], "I think we should cut the intro."))  # 1:1
    assert not rule({"to": "me", "ask": False, "reply": False}, r(["Alex"], "Okay, thanks."))
    assert not rule({"to": "self", "ask": False, "reply": False}, r(["Alex"], "where did I put that file"))
    follow = r(["Alex", "Priya"], "Why?", [("Alex", "Higgs, ideas?"), ("Higgs", "Open with a live call.")])
    assert rule({"to": "group", "ask": True, "reply": True}, follow)  # short follow-up right after the AI
    assert not rule({"to": "group", "ask": True, "reply": True, "can": False},
                    r(["Alex", "Priya"], "Priya, why?", [("Higgs", "Open with a live call.")]))
    assert rule({"to": "group", "ask": True, "answered": False, "can": True}, r(["Alex", "Priya"], "When's the deadline?"))
