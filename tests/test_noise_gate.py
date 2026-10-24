"""The noise gate (noise_gate.py): speech is found, noise is not."""
import wave
from pathlib import Path

import numpy as np
import pytest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from noise_gate import SpeakerFrontEnd  # noqa: E402

SR = 24000
DATA = Path(__file__).parent / "data"
rng = np.random.default_rng(7)


def speech() -> np.ndarray:
    with wave.open(str(DATA / "question.wav")) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def run(sig, fe=None):
    fe = fe or SpeakerFrontEnd()
    events = []
    for i in range(0, len(sig) - 479, 480):
        events += [(e.kind, e.t, e.duration) for e in fe.push(sig[i:i + 480])]
    return events, fe


def at_level(x, rms):
    x = np.asarray(x, float)
    return x / np.sqrt(np.mean(x ** 2)) * rms


def mix(*parts):
    return np.clip(sum(parts), -32768, 32767).astype(np.int16)


def brown(n):
    b = np.cumsum(rng.normal(0, 1, n))
    return b - np.convolve(b, np.ones(2400) / 2400, "same")


NOISES = {
    "white": lambda n: rng.normal(0, 1, n),
    "fan (brown)": brown,
    "hum 60 Hz": lambda n: np.sin(2 * np.pi * 60 * np.arange(n) / SR),
}


def test_speech_is_one_segment_with_its_start_and_end():
    sp = speech()
    events, _ = run(np.concatenate([np.zeros(2 * SR, np.int16), sp, np.zeros(SR, np.int16)]))
    kinds = [k for k, _, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "end" and kinds.count("start") == kinds.count("end") == 1
    start, end = events[0][1], events[-1][1]
    assert 1.8 < start < 2.6 and 5.6 < end < 6.6     # speech is ~2.2-5.9 s; the end comes after 0.45 s of silence
    assert "clause" in kinds                          # the pause after "Actually, wait." is a clause boundary


@pytest.mark.parametrize("name", list(NOISES))
def test_noise_alone_never_starts_a_segment(name):
    events, _ = run(at_level(NOISES[name](10 * SR), 3000).astype(np.int16))
    assert events == []


def test_keyboard_clicks_never_start_a_segment():
    sig = rng.normal(0, 30, 6 * SR)
    for k in range(10):
        s = int((0.5 + k * 0.5) * SR)
        sig[s:s + 240] += rng.normal(0, 9000, 240)
    assert run(sig.astype(np.int16))[0] == []


@pytest.mark.parametrize("name,snr", [("white", 10), ("white", 5), ("fan (brown)", 5), ("fan (brown)", 0)])
def test_speech_is_found_in_steady_noise(name, snr):
    sp = speech().astype(float)
    rms = np.sqrt(np.mean(sp ** 2))
    noise = at_level(NOISES[name](len(sp) + 3 * SR), rms / 10 ** (snr / 20))
    events, _ = run(mix(np.concatenate([np.zeros(2 * SR), sp, np.zeros(SR)]), noise))
    starts = [t for k, t, _ in events if k == "start"]
    ends = [t for k, t, _ in events if k == "end"]
    assert starts and ends and starts[0] < 3.1 and ends[-1] > 5.8, events   # the segment covers the speech


def test_background_talkers_far_below_the_speaker_are_ignored_once_the_voice_level_is_known():
    sp = speech().astype(float)
    babble = at_level(np.resize(sum(np.roll(sp, k * 7919) for k in range(6)), 30 * SR), np.sqrt(np.mean(sp ** 2)) / 10 ** (15 / 20))
    sig = babble.copy()
    sig[2 * SR:2 * SR + len(sp)] += sp
    sig[16 * SR:16 * SR + len(sp)] += sp
    events, fe = run(np.clip(sig, -32768, 32767).astype(np.int16))
    later = [t for k, t, _ in events if k == "start" and t > 9]
    assert later and all(15.5 < t < 17 for t in later), events   # only the speaker's second line, nothing between


def test_echo_of_the_ai_is_not_speech():
    sp = speech()
    fe = SpeakerFrontEnd()
    run(np.zeros(2 * SR, np.int16), fe)
    fe.set_ai_level(-6.0)                              # the AI is playing loudly ...
    quiet = (sp.astype(float) * 10 ** (-32 / 20)).astype(np.int16)   # ... and leaks back ~30 dB down
    events, _ = run(np.concatenate([quiet, np.zeros(SR, np.int16)]), fe)
    assert not any(k == "start" for k, _, _ in events)
    fe.set_ai_level(None)
    events, _ = run(np.concatenate([sp, np.zeros(SR, np.int16)]), fe)
    assert any(k == "start" for k, _, _ in events)


def test_noisy_flag():
    _, fe = run(at_level(rng.normal(0, 1, 3 * SR), 300).astype(np.int16))
    assert fe.noisy
    _, fe = run(at_level(rng.normal(0, 1, 3 * SR), 10).astype(np.int16))
    assert not fe.noisy
