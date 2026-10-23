"""A scripted hackathon team meeting with an AI teammate ("Higgs"), annotated with when the AI *should* speak.

Each human turn is a decision point: after it there is a pause, and the AI must SPEAK or stay SILENT.
`gold` is our annotation: SPEAK when the AI is addressed by name, or when the group asks a question nobody in the
room can answer and the AI knows the answer; SILENT when people talk to each other, think aloud, or trail off.
`ai` is the line the AI says at SPEAK points (used as context for later turns, so every policy sees the same
history). The facts the AI knows are in FACTS."""

FACTS = ("Hackathon facts you know: the demo video must be 2 to 3 minutes and show the project running; judges review "
         "the public GitHub repo at a pinned commit plus the video; a live URL is optional; judging is Use of Higgs "
         "Audio 30%, User Experience 25%, Innovation 25%, Technical Execution 20%; submissions close Saturday at 3 pm.")

VOICES = {"Ana": "eleanor", "Ben": "marcus", "Chen": "oliver", "Higgs": "chloe"}

TURNS = [
    ("Ana", "Okay, so we have about three hours left before submissions close.", "SILENT", None),
    ("Ben", "Right, and the backend is still kind of flaky.", "SILENT", None),
    ("Chen", "I can look at the backend after lunch, if someone else takes the slides.", "SILENT", None),
    ("Ana", "I'll do the slides. Ben, can you record the demo video?", "SILENT", None),
    ("Ben", "Sure. How long is the video supposed to be again?", "SPEAK",
     "Two to three minutes, and it has to show the thing actually running, not slides."),
    ("Ana", "Okay, two to three minutes. That's tight.", "SILENT", None),
    ("Chen", "We should probably open with the turn-taking stuff, because", "SILENT", None),
    ("Chen", "that's the part people remember.", "SILENT", None),
    ("Ben", "Higgs, what do you think we should open with?", "SPEAK",
     "I'd open with the before-and-after: plain Higgs stopping on a mm-hm, then ours carrying on."),
    ("Ana", "I like that. Chen, do you agree?", "SILENT", None),
    ("Chen", "Yeah, I agree, as long as the audio doesn't echo.", "SILENT", None),
    ("Ben", "Does anyone remember if the judges want a live URL or just the repo?", "SPEAK",
     "The repo at a pinned commit plus the video. A live URL is optional, but it helps."),
    ("Ana", "Good, then let's deploy it tonight anyway.", "SILENT", None),
    ("Chen", "Ugh, I'm so hungry. Ben, do you want to grab pizza?", "SILENT", None),
    ("Ben", "Yes, absolutely, pizza first.", "SILENT", None),
    ("Ana", "Wait, before you go. Higgs, can you remind us how the judging is weighted?", "SPEAK",
     "Use of Higgs is thirty percent, user experience and innovation twenty-five each, technical execution twenty."),
    ("Ben", "Thirty percent on using Higgs, nice.", "SILENT", None),
    ("Chen", "So we really need to show the interruptions working.", "SILENT", None),
    ("Ana", "I think that's everything. Are we forgetting anything?", "SPEAK",
     "Just one thing: remove the API key from the repo before you tag the commit."),
    ("Ben", "Oh, good call. Okay, pizza.", "SILENT", None),
]


def records():
    """Decision points in the training format of our speak/silent model (speaker-labelled context + current)."""
    history, out = [], []
    for i, (spk, text, gold, ai) in enumerate(TURNS):
        out.append({"id": i, "target_speaker": "Higgs", "context_turns": list(history),
                    "current_turn": {"speaker": spk, "text": text}, "gold": gold})
        history.append({"speaker": spk, "text": text})
        if ai:
            history.append({"speaker": "Higgs", "text": ai})
    return out


if __name__ == "__main__":
    import json
    import os
    os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "out", "decision_points.jsonl"), "w") as f:
        for r in records():
            f.write(json.dumps(r) + "\n")
    print(sum(t[2] == "SPEAK" for t in TURNS), "SPEAK of", len(TURNS))
