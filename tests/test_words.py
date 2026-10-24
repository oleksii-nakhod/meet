"""F2 word rules: fillers never stop the AI, real interruptions and stop requests do."""
import pytest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from words import bargein, content_words, fix_name, lexicon, prefilter  # noqa: E402

LEX = lexicon(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "fillers.en.yaml"))

FILLERS = ["Umm.", "Uh.", "Ahh.", "Ummmm...", "Hmm.", "Mm-hm.", "Uh-huh.", "Okay.", "Okay, okay.", "Yeah.",
           "Yeah, yeah.", "Yep.", "Right.", "Right, right.", "Sure.", "I see.", "Got it.", "Oh, okay.", "Oh wow.",
           "Makes sense.", "Mhm, yeah.", "Umm, okay, yeah.", "Ah, I see.", "Cool.", "Nice.", "Exactly.", "True.",
           "Uh, okay.", "Er.", "Oh.", "Interesting.", "Sounds good.", "Go on.", "Fair enough.", "Hmm, right.",
           "Yeah, totally.", "Okay, cool.", "Thanks.", "Ha ha.", "Absolutely."]
INTERRUPTS = ["What about Friday?", "Can we talk about the budget instead", "I think that's wrong",
              "No, the rent is due on the first", "Hang on, what did you say about the lease", "Higgs, when is the exam?",
              "Okay but what about the flights", "Yeah but that doesn't cover the hotel"]
STOPS = ["Stop.", "Wait.", "Wait, wait.", "Hold on.", "Higgs, stop.", "Hang on.", "One second.", "Sorry, sorry.",
         "Um, wait.", "Actually, no.", "Excuse me."]


@pytest.mark.parametrize("text", FILLERS)
def test_fillers_and_acknowledgements_never_stop_the_ai(text):
    assert bargein(text, 0.8, LEX, "Higgs") == "filler"


@pytest.mark.parametrize("text", INTERRUPTS)
def test_real_interruptions_stop_it(text):
    assert bargein(text, 1.0, LEX, "Higgs") in ("interrupt", "stop")


@pytest.mark.parametrize("text", STOPS)
def test_stop_requests_stop_it_at_once(text):
    assert bargein(text, 0.4, LEX, "Higgs") == "stop"


def test_too_little_to_tell_waits():
    assert bargein("Friday", 0.5, LEX) == "unclear"     # one content word: listen on
    assert bargein("", 0.5, LEX) == "unclear"           # nothing transcribed yet
    assert bargein("", 2.0, LEX) == "interrupt"         # long speech STT can't make out: let them talk


def test_content_words_drop_the_ai_name_and_phrases():
    assert content_words("Higgs, I see, okay", LEX, "Higgs") == []
    assert content_words("Okay so what about Friday", LEX) == ["what", "about", "friday"]


@pytest.mark.parametrize("text,kind", [
    ("Mm-hm.", "backchannel"), ("Um.", "filler"), ("So the, uh,", "unfinished"), ("Higgs, stop.", "stop"),
    ("What time is it?", None), ("I think.", "fragment"), ("Five minutes.", None), ("...", "skip"),
    ("Hey Higgs.", None), ("Wait, what did you say about the rent?", None),
])
def test_prefilter_in_a_group(text, kind):
    assert prefilter(text, "Higgs", "", LEX, ["Alex", "Mei"])[1] == kind


def test_a_yes_that_answers_the_ai_is_a_line():
    assert prefilter("Yeah.", "Higgs", "Want me to go on?", LEX, ["Alex"])[1] is None
    assert prefilter("Yeah.", "Higgs", "", LEX, ["Alex"])[1] == "backchannel"


def test_fix_name():
    assert fix_name("Pigs, how should we split it?", "Higgs") == "Higgs, how should we split it?"
    assert fix_name("Thanks, Hicks.", "Higgs") == "Thanks, Higgs."
    assert fix_name("Thanks, Mei.", "Higgs", ["Mei"]) == "Thanks, Mei."


def test_participant_names_are_repaired_too():
    assert prefilter("May. Did you clean the bathroom?", "Higgs", "", LEX, ["Alex", "Mei"])[0] == \
        "Mei. Did you clean the bathroom?"
    assert prefilter("Thanks, Alex.", "Higgs", "", LEX, ["Alex", "Mei"])[0] == "Thanks, Alex."
