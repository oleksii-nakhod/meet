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


def test_fix_name_calls():
    """Mishearings from the 2026-10-23 calls (out/rooms), and words that only sound alike."""
    p = ["Alex", "Priya", "Sam", "Jordan", "Mei"]
    fixed = [("Higgs", "Hi everyone. Hi X. Can you hear us?", "Hi everyone. Hi Higgs. Can you hear us?"),
             ("Higgs", "Ex, how would you structure the video?", "Higgs, how would you structure the video?"),
             ("Higgs", "Hello, hex. How are you?", "Hello, Higgs. How are you?"),
             ("Higgs", "Eggs in a few sentences.", "Higgs in a few sentences."),
             ("Nova", "Hey, hey Noah, how are you?", "Hey, hey Nova, how are you?"),
             ("Higgs", "Ahex, so are we getting a cat or not?", "Higgs, so are we getting a cat or not?"),
             ("Higgs", "Okay, house meeting. Hi, Hicks. How are you?", "Okay, house meeting. Hi, Higgs. How are you?"),
             ("Higgs", "As long as we ask first, Hicks, what do you think?",
              "As long as we ask first, Higgs, what do you think?"),
             ("Nova", "Noah walk us through the rollout plan.", "Nova, walk us through the rollout plan."),
             ("Assistant", "Okay, enough. Assistant just set up his snack.", "Okay, enough. Assistant, just set up his snack."),
             ("Nova", "What do you think Noah?", "What do you think Nova?"),
             ("Max", "Mac, what do you think?", "Max, what do you think?")]
    for ai, said, want in fixed:
        assert agent.fix_name(said, ai, p) == want, said
    for ai, said in [("Nova", "No, I don't think so."), ("Nova", "Now, what do we do?"), ("Higgs", "We need eggs and milk."),
                     ("Higgs", "My ex lives in Toronto."), ("Nova", "Noah said he'd be late."),
                     ("Higgs", "Hicks Street is closed today."), ("Nova", "Novel idea, let's try it."),
                     ("Max", "Maps are on the shared drive."), ("Higgs", "Can you hear us?"), ("Higgs", "Hey, how are you?"),
                     ("Max", "Max capacity is fifty people."), ("Higgs", "What do you think, Alex?")]:
        assert agent.fix_name(said, ai, p) == said, said


def test_name_from_local():
    p = ["Alex", "Priya", "Sam"]
    f = agent.name_from_local
    assert f("Okay, hey, do you know anything about Alex?", "Okay, Higgs, do you know anything about Alex?", "Higgs",
             p) == "Okay, Higgs, do you know anything about Alex?"
    assert f("Can you hear us?", "Nova, can you hear us?", "Nova", p) == "Nova, can you hear us?"
    assert f("What do you think?", "What do you think, Nova?", "Nova", p) == "What do you think, Nova?"
    assert f("We talked to Noah yesterday.", "We talked to Nova yesterday.", "Nova", p) == "We talked to Noah yesterday."
    assert f("I think it's a cool name.", "I think Nova is a cool name.", "Nova", p) == "I think it's a cool name."
    assert f("Nova, hi.", "Nova, hi.", "Nova", p) == "Nova, hi."
    assert f("Okay, hey, what do you think?", "Okay, Higgs's, what do you think?", "Higgs", p).startswith("Okay")  # no raise


def test_fix_terms():
    facts = ["The landlord is Mrs. Okafor; she said no pets.", "Ticket PAY-512 is owned by Jordan.",
             "The launch is Thursday at 10 am."]
    t = agent.fact_terms(facts)
    assert t == ({"Okafor": "Mrs. Okafor", "Jordan": "Jordan"}, {"PAY-512"})  # not "The", not the day
    f = lambda s: agent.fix_terms(s, t, ["Alex", "Priya"], "Higgs")  # noqa: E731
    assert f("Should we email Mr. O'Keffe about the cat?") == "Should we email Mrs. Okafor about the cat?"
    assert f("Where are we on K512?") == "Where are we on PAY-512?"
    assert f("So the launch is Tuesday.") == "So the launch is Tuesday."  # what people get wrong stays
    assert f("I like Oscar.") == "I like Oscar."
    assert f("Alex, PAY-512 is done.") == "Alex, PAY-512 is done."


def test_refusal():
    for t in ["I'm sorry, Priya. I can't continue this conversation.", "I’m unable to continue this conversation.",
              "I cannot continue the discussion.", "We can’t continue this chat."]:
        assert agent.REFUSAL.search(t), t
    for t in ["I can't continue the demo without the video.", "Let's continue this conversation tomorrow."]:
        assert not agent.REFUSAL.search(t), t


def test_prefilter():
    pf = lambda t, h=(): agent.prefilter(t, "Higgs", [{"speaker": s, "text": x} for s, x in h])[1]  # noqa: E731
    assert pf("...") == "skip" and pf("嗯。") == "skip"
    assert pf("Mm-hm.") == "backchannel" and pf("Yeah, yeah.") == "backchannel" and pf("Got it.") == "backchannel"
    assert pf("Yeah.", [("Higgs", "Want me to go on?")]) is None  # answers the AI's question
    assert pf("Yeah?") is None
    grp = lambda t: agent.prefilter(t, "Higgs", [], ["Alex", "Priya"])[1]  # noqa: E731
    assert grp("I think.") == "fragment" and grp("Sunday.") is None and grp("Five minutes.") is None
    assert grp("Thanks, Higgs.") is None and agent.prefilter("I think.", "Higgs", [], ["Alex"])[1] is None
    assert pf("Yes.", [("Higgs", "I can list them all. Let me know if you want that.")]) is None  # accepts an offer
    assert pf("Okay.", [("Higgs", "The deadline is Saturday at 3 pm.")]) == "backchannel"
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


def test_first_sentence():
    assert agent.first_sentence("Sure. The deadline is Saturday at 3 pm. Plan ahead.") == "Sure. The deadline is Saturday at 3 pm."
    assert agent.first_sentence("The video must be 2 to 3 minutes long. Keep it tight.") == "The video must be 2 to 3 minutes long."
    assert agent.first_sentence("The video must be", complete_only=True) is None
    assert agent.first_sentence("Yes.") == "Yes."


def test_vocative_only():
    assert agent.vocative_only("Hey, Higgs.", "Higgs") and agent.vocative_only("Higgs?", "Higgs")
    assert not agent.vocative_only("Hey Higgs, how long can the video be?", "Higgs")
    assert not agent.vocative_only("Hey there.", "Higgs")
    assert agent.vocative_only("Assistant, you caught all that, right?", "Assistant")
    assert not agent.vocative_only("Higgs, the deadline is Saturday, right? Because Sam said Sunday.", "Higgs")


def test_vocative_person():
    vp = agent.vocative_person
    assert vp("Alex, I can barely hear you.", ["Alex", "Sam"], "Higgs") == "Alex"
    assert vp("Hey Sam, can you check it?", ["Alex", "Sam"], "Higgs") == "Sam"
    assert vp("Can you check it, Sam?", ["Alex", "Sam"], "Higgs") == "Sam"
    assert vp("Alex, ask Higgs about it.", ["Alex"], "Higgs") is None  # the AI is named too
    assert vp("Alexander the Great was fast.", ["Alex"], "Higgs") is None


def test_person_called_before():
    pc = agent.person_called_before
    ctx = [{"speaker": "Priya", "text": "Alex."}]
    assert pc(ctx, "Priya", ["Alex"], "Higgs", "Your mic is a bit noisy.") == "Alex"
    assert pc(ctx, "Sam", ["Alex"], "Higgs", "Your mic is a bit noisy.") is None  # someone else's call
    assert pc(ctx, "Priya", ["Alex"], "Higgs", "Higgs, is my mic noisy?") is None
    assert pc([{"speaker": "Priya", "text": "Alex said the mic was noisy yesterday."}], "Priya", ["Alex"], "Higgs",
              "Is it?") is None


def test_echoes():
    line = "I'm not asking for the moon here. It's one stop. Assistant, please, just add the stop for me."
    assert agent.echoes("Marcus, I'm not asking for the moon here. It's one stop.", line)
    assert not agent.echoes("Done: I added the record shop as a stop on your route.", line)
    # an answer reusing the question's words isn't speaking as them (it was dropped as 'no draft', pause_1to1)
    assert not agent.echoes("Using the avatar in the intro is a great idea.",
                            "What do you think about using the avatar in the intro.")


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
    piece = r(["Priya", "Alex"], "Anything else?", [("Higgs", "Rent is due Sunday."), ("Priya", "Higgs. Thanks.")])
    assert rule({"to": "group", "ask": False, "reply": False}, piece)  # the same person just addressed the AI
