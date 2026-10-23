"""Unit tests for the engine's pure logic (no API, no models): run with `python -m pytest tests/unit`."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # noqa: E402
from mmhm.engine import N, VAD, unfinished  # noqa: E402
from mmhm.expressive import SentenceSpeaker, spoken, voiced  # noqa: E402


def test_unfinished_catches_mid_sentence_stops():
    for t in ["I've been learning to play the.", "My sister just moved to.", "The weird part is that, um,",
              "So I was thinking that maybe we could", "and we're.", "Higgs, I was wondering whether.",
              "So I was thinking that"]:
        assert unfinished(t), t


def test_unfinished_lets_finished_turns_through():
    for t in ["What's the best place to get coffee around here?", "Which team are you on?", "Thank you.",
              "I love it.", "Yeah, good point.", "", "Go on.", "Hold on", "Tell me more about that.",
              "what do you mean by that", "what are you working on", "Let's do this"]:
        assert not unfinished(t), t


def tone(db, n=N, f=200.0):
    t = np.arange(n) / 24000
    return (np.sin(2 * np.pi * f * t) * 32767 * 10 ** (db / 20) * np.sqrt(2)).astype(np.int16)


def test_vad_detects_speech_over_a_steady_noise_floor():
    v = VAD()
    rng = np.random.default_rng(0)
    for _ in range(150):  # 3 s of room noise at about -45 dBFS: the floor adapts, nothing is "speech"
        speaking, _ = v((rng.standard_normal(N) * 32767 * 10 ** (-45 / 20)).astype(np.int16))
    assert not speaking
    speaking, _ = v(tone(-20) + (rng.standard_normal(N) * 32767 * 10 ** (-45 / 20)).astype(np.int16))
    assert speaking


def test_vad_hangover_then_silence():
    v = VAD()
    v(tone(-20))
    assert v(np.zeros(N, np.int16))[0]          # within the 60 ms hangover
    for _ in range(5):
        speaking, _ = v(np.zeros(N, np.int16))
    assert not speaking


def test_sound_effect_tags_get_their_written_sound():
    assert voiced("<|sfx:laughter|> Oh no.").startswith("<|sfx:laughter|>Haha,")
    assert voiced("<|sfx:laughter|>Haha, nice.") == "<|sfx:laughter|>Haha, nice."
    assert voiced("<|emotion:surprise|> No way!") == "<|emotion:surprise|> No way!"
    assert spoken("<|sfx:sigh|>Uh, <|emotion:sadness|>I'm sorry.") == "Uh, I'm sorry."


def test_sentence_speaker_splits_and_carries_the_lead_tag():
    sp = SentenceSpeaker.__new__(SentenceSpeaker)  # no TTS thread: test the text handling only
    sp.buf, sp.lead, sp.segments, sp.queue = "", "", [], []
    import threading
    sp.cv = threading.Condition()
    for d in ["<|emotion:enthusiasm|>No way! That's ", "amazing. What was ", "the project?"]:
        sp.feed(d)
    sp._push(sp.buf)
    assert sp.queue[0].startswith("<|emotion:enthusiasm|>No way!")
    assert all(q.startswith("<|emotion:enthusiasm|>") for q in sp.queue)
    assert [spoken(q) for q in sp.queue] == ["No way!", "That's amazing.", "What was the project?"]


def test_rule_break_detection():
    from app.personas import rule_check
    for t in ["Are you an AI?", "honestly, are you a robot?", "r u human", "Is this a bot?", "are you real?"]:
        assert rule_check(t), t
    for t in ["Are you working on the AI track?", "What are you working on?", "I think the robot demo was cool."]:
        assert not rule_check(t), t
