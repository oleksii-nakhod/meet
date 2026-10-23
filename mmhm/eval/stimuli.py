"""Synthesized 'user' speech for the turn-taking and barge-in evaluations (Higgs TTS 3 preset voices).

COMPLETE: finished turns. HOLD: an unfinished clause (A), a mid-thought pause, then the rest (B).
BACKCHANNELS: what a listener says while the other person talks. INTERRUPTS: real attempts to take the floor.
Files are cached in out/stim/<slug>_<voice>.wav, so re-running costs nothing."""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mmhm.higgs import load_wav, outdir, save_wav, trim, tts  # noqa: E402

STIM = outdir("stim")
VOICES = ["jake", "nora", "marcus", "eleanor"]

COMPLETE = [
    "What's the best place to get coffee around here?",
    "Did you see the keynote this morning?",
    "I think the demo should be under three minutes.",
    "Honestly I'm kind of exhausted already.",
    "Have you ever been to Montreal?",
    "My laptop battery is basically dead.",
    "Which team are you on?",
    "That talk about robots was really cool.",
    "I can't believe it's already October.",
    "Do you want to grab lunch later?",
    "We should probably start coding soon.",
    "What did you study in school?",
]
HOLD = [  # (A, B): A is syntactically or prosodically unfinished
    ("So I was thinking that maybe we could", "just order more pizza for everyone."),
    ("The thing about the Wi-Fi on this floor is", "that it dies every time I push."),
    ("Okay so my plan for tomorrow is basically", "to sleep until noon and then code."),
    ("I really want to try the", "new ramen place on Baldwin street."),
    ("My sister just moved to", "Vancouver for a new job."),
    ("The weird part is that, um,", "nobody told us the deadline changed."),
    ("If we finish the backend tonight, then", "tomorrow we can focus on the slides."),
    ("I used to work at a bank, and", "honestly I hated every minute of it."),
    ("What I'd love to know is", "how they make the voices sound so real."),
    ("We tried three different", "approaches before anything worked."),
    ("The best part of the trip was when we", "got lost in the old town at night."),
    ("Last year my team built a", "game where you argue with a robot judge."),
]
# Held-out set: never used for tuning thresholds. Different sentences, and two voices not in the tuning set.
TEST_VOICES = ["oliver", "chloe", "jake", "nora"]
COMPLETE_TEST = [
    "Where did you park this morning?",
    "I just finished my first cup of tea.",
    "Is the Wi-Fi working for you right now?",
    "My flight home leaves on Sunday night.",
    "Have you tried the new dumpling place?",
    "I think our backend is finally stable.",
    "What time do the presentations start?",
    "I forgot my charger at home again.",
    "Do you know anyone else on this floor?",
    "This chair is surprisingly comfortable.",
    "Who's on your team this year?",
    "I'd love a nap right about now.",
]
HOLD_TEST = [
    ("The reason I came to this event is", "that my roommate dared me to sign up."),
    ("When I was a kid I always wanted to", "build robots that could talk back."),
    ("Honestly the hardest part of the project was", "getting the audio to stop echoing."),
    ("If you ever go to Lisbon, make sure you", "try the custard tarts near the river."),
    ("My manager keeps telling me that", "I should take more vacation days."),
    ("We could, um,", "maybe split the work by feature instead."),
    ("The first time I gave a talk, I", "forgot every single word of my intro."),
    ("So the plan is that tomorrow morning we", "meet at nine and test everything together."),
    ("I've been learning to play the", "cello for about two years now."),
    ("What really surprised me about Toronto was", "how friendly people are on the subway."),
    ("Before this job I spent three years", "teaching math at a high school."),
    ("The coffee machine on this floor is", "the only reason I'm still awake."),
]
BACKCHANNELS = {
    "mmhm": "Mm-hm.",
    "yeah": "Yeah.",
    "uhhuh": "Uh-huh.",
    "right": "Right.",
    "okay": "Okay.",
    "wow": "Wow.",
    "laugh": "<|sfx:laughter|>Haha.",
    "mhm2": "Mhm, mhm.",
}
INTERRUPTS = {
    "where": "Wait, sorry, where was that?",
    "askq": "Hold on, can I ask you something?",
    "notmeant": "No no, that's not what I meant at all.",
    "time": "Sorry to cut in, but what time is it?",
    "switch": "Actually, let's talk about something else.",
    "again": "Wait what? Say that again?",
}
# Held-out barge-in cues (new words, new voices oliver/chloe); never used for tuning.
BACKCHANNELS_TEST = {
    "mm": "Mm.",
    "sure": "Sure.",
    "ohwow": "Oh wow.",
    "exactly": "Exactly.",
    "yeahyeah": "Yeah, yeah.",
    "uhhuh": "Uh-huh.",
    "chuckle": "<|sfx:laughter|>Heh.",
    "nice": "Nice.",
}
INTERRUPTS_TEST = {
    "name": "Hang on, what was the name of that place?",
    "goback": "Sorry, can you go back a bit?",
    "wrong": "Wait, that's not right, it was last year.",
    "quickq": "Excuse me, quick question about that.",
    "stop": "Okay, okay, let me stop you there.",
    "meaning": "Hold on, what do you mean by that?",
}


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48]


def get(text, voice="jake"):
    """int16 24 kHz PCM of `text` in `voice`, trimmed of leading/trailing silence."""
    path = os.path.join(STIM, f"{slug(text)}_{voice}.wav")
    if not os.path.exists(path):
        pcm, _, _ = tts(text, voice=voice)
        save_wav(path, trim(pcm))
    return load_wav(path)


if __name__ == "__main__":
    n = 0
    for i, t in enumerate(COMPLETE):
        get(t, VOICES[i % 4]); n += 1
    for i, (a, b) in enumerate(HOLD):
        get(a, VOICES[i % 4]); get(b, VOICES[i % 4]); n += 2
    for v in VOICES:
        for t in list(BACKCHANNELS.values()) + list(INTERRUPTS.values()):
            get(t, v); n += 1
    for i, t in enumerate(COMPLETE_TEST):
        get(t, TEST_VOICES[i % 4]); n += 1
    for i, (a, b) in enumerate(HOLD_TEST):
        get(a, TEST_VOICES[i % 4]); get(b, TEST_VOICES[i % 4]); n += 2
    for v in ("oliver", "chloe"):
        for t in list(BACKCHANNELS_TEST.values()) + list(INTERRUPTS_TEST.values()):
            get(t, v); n += 1
    print(f"{n} stimuli in {STIM}")
