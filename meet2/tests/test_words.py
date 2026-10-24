"""F2 word rules: fillers never stop the AI, real interruptions and stop requests do."""
import pytest

from meet.config import ROOT
from meet.listen.words import bargein, content_words, fix_name, lexicon, prefilter

LEX = lexicon(str(ROOT / "config" / "fillers.en.yaml"))

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


def test_pieces_cut_mid_sentence_join_back():
    from meet.listen.listener import join_pieces
    assert join_pieces("What does the lease say about having a?", "Dog.") == "What does the lease say about having a dog."
    assert join_pieces("Rent is due on the first.", "Okay?") == "Rent is due on the first. Okay?"
    assert join_pieces("I gave it to the.", "I mean the landlord.") == "I gave it to the. I mean the landlord."


def test_sources_are_attributed_from_the_words_a_reply_shares():
    from meet.agent import attribute
    sources = [{"doc": "lease.pdf", "page": 3, "text": "Section 7. No pets without the landlord's written approval."},
               {"doc": "lease.pdf", "page": 2, "text": "Rent is $2,400 a month, due on the 1st."}]
    assert attribute("The lease says no pets without the landlord's written approval.", sources, []) == "lease.pdf p.3"
    assert attribute("Sure, happy to help!", sources, []) is None
    facts = [{"text": "Rent is due on the 1st to Mrs. Okafor.", "doc": "house.docx", "page": None}]
    assert attribute("Rent is due on the 1st, to Mrs. Okafor.", [], facts) == "house.docx"


def test_participant_names_are_repaired_too():
    assert prefilter("May. Did you clean the bathroom?", "Higgs", "", LEX, ["Alex", "Mei"])[0] == \
        "Mei. Did you clean the bathroom?"
    assert prefilter("Thanks, Alex.", "Higgs", "", LEX, ["Alex", "Mei"])[0] == "Thanks, Alex."
