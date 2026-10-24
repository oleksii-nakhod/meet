"""The whole agent against the scripted fake Boson (tests/fakes.py), in real time.

Covers UC2 (1:1 answer; fillers don't stop it; a real interruption does), UC3 (people talking to each other: the
AI stays quiet) and UC4 (a contradiction: the AI cuts in, ducks the humans, buffers what is said meanwhile and
handles it afterwards).
"""
import asyncio

import numpy as np

from meet.agent import HiggsAgent
from meet.kb import Library
from meet.rooms import Brief, Person, Room

from .fakes import FakeModels, Phrases

SR = 24000
PH = Phrases([
    "Higgs, when is the rent due?",            # 0
    "Umm, okay.",                              # 1
    "Yeah, yeah.",                             # 2
    "Wait, what about the internet bill?",     # 3
    "Mei, did you take the trash out?",        # 4
    "Yes, I did it last night.",               # 5
    "I'm pretty sure rent is due on the fifth.",  # 6
    "Hold on a second.",                       # 7
    "Can we talk about chores next?",          # 8
    "Higgs, what day is trash night?",         # 9
])


class FakeClient:
    def __init__(self):
        self.events = []

    def send_json(self, m):
        self.events.append(m)

    def send_audio(self, slot, pcm):
        pass

    def send_video(self, slot, jpeg):
        pass


class FakeTransport:
    name = "fake"

    def __init__(self):
        self.frames, self.cleared = 0, 0

    async def send_ai_audio(self, frame):
        self.frames += 1

    async def end_ai_segment(self):
        pass

    async def clear_ai_audio(self):
        self.cleared += 1

    def playout_delay(self):
        return 0.05

    def on_playback_finished(self, cb):
        pass


def make(tmp_path, names, models):
    room = Room("t1", Brief(title="House", ai="Higgs", setting="house meeting",
                            facts=["Rent is $2,400 a month, due on the 1st."]), log=False)
    room.transport = FakeTransport()
    for i, n in enumerate(names):
        room.people[i + 1] = Person(slot=i + 1, pid=n, name=n, client=FakeClient(), host=i == 0)
    lib = Library(tmp_path / "kb")
    lib.get(room.collection).set_facts(room.brief.facts)
    agent = HiggsAgent(room, models, lib)
    room.agent = agent
    return room, agent


async def speak(agent, slot, audio):
    """Feed one person's audio in real time, other people silent."""
    for k in range(0, len(audio) - 479, 480):
        for s in agent.listeners:
            frame = audio[k:k + 480] if s == slot else np.zeros(480, np.int16)
            agent.feed(s, frame.tobytes())
        await asyncio.sleep(0.02)


async def silence(agent, seconds):
    await speak(agent, -1, np.zeros(int(SR * seconds), np.int16))


def events(room, kind):
    return [e for e in room.people[1].client.events if e.get("type") == kind]


def test_one_to_one_answer_fillers_and_interruption(tmp_path):
    async def go():
        models = FakeModels(PH, reply=("Sure. The rent is due on the first of every month, to Mrs. Okafor. "
                                       "The internet is separate. Anything else you need for the house meeting today?"),
                            tts_seconds_per_word=0.3)
        room, agent = make(tmp_path, ["Alex"], models)
        await agent.start()
        await silence(agent, 2.0)                                # the front-end learns the room
        await speak(agent, 1, PH.audio(0))
        await silence(agent, 1.6)
        said = events(room, "ai_say")
        assert said and "first" in said[-1]["text"], said
        assert agent.output.speaking and room.transport.cleared == 0
        frames_before = room.transport.frames
        await speak(agent, 1, PH.audio(1))                       # "Umm, okay." while it talks
        await silence(agent, 0.4)
        await speak(agent, 1, PH.audio(2))                       # "Yeah, yeah."
        await silence(agent, 0.6)
        assert room.transport.cleared == 0 and agent.output.speaking, "a filler stopped the AI"
        assert room.transport.frames > frames_before
        await speak(agent, 1, PH.audio(3, seconds=1.3))          # a real interruption
        await silence(agent, 0.5)
        assert room.transport.cleared == 1 and not agent.output.speaking
        ai_lines = [line for line in agent.ledger if line["slot"] == 0]
        assert ai_lines and ai_lines[-1]["text"].endswith("[interrupted]")
        lines = [e for e in events(room, "line") if e.get("final")]
        assert any(e["kind"] in ("filler", "backchannel") for e in lines)
        await silence(agent, 1.5)                                # it answers the interruption
        assert len([e for e in events(room, "ai_say") if not e["done"]]) >= 2
        await agent.stop()
    asyncio.run(go())


def test_people_talking_to_each_other_get_no_reply(tmp_path):
    async def go():
        models = FakeModels(PH)
        room, agent = make(tmp_path, ["Alex", "Mei"], models)
        await agent.start()
        await silence(agent, 2.0)
        await speak(agent, 1, PH.audio(4))
        await silence(agent, 0.6)
        await speak(agent, 2, PH.audio(5))
        await silence(agent, 1.5)
        assert models.calls["judge"] >= 2
        assert room.transport.frames == 0 and not events(room, "ai_say")
        finals = [e["text"] for e in events(room, "line") if e.get("final")]
        assert PH.phrases[4] in finals and PH.phrases[5] in finals
        await agent.stop()
    asyncio.run(go())


def test_contradiction_cuts_in_ducks_buffers_and_releases(tmp_path):
    async def go():
        models = FakeModels(PH, contradictions={"fifth": {
            "contradicts": True, "evidence_id": "F1", "evidence": "due on the 1st", "confidence": 0.95,
            "correction": "Sorry to jump in, Mei. The fact sheet says rent is due on the 1st, not the fifth."}},
            tts_seconds_per_word=0.15)
        room, agent = make(tmp_path, ["Alex", "Mei"], models)
        await agent.start()
        await silence(agent, 2.0)
        await speak(agent, 2, PH.audio(6))
        await silence(agent, 0.8)
        fixes = events(room, "correction")
        assert fixes and "1st" in fixes[0]["text"] and fixes[0]["source"] == "fact sheet"
        gains = [e["gain"] for e in events(room, "human_gain")]
        assert gains == [0.15]
        assert agent.floor.state == "correcting"
        await speak(agent, 1, PH.audio(8))                       # Alex talks during the correction ...
        await silence(agent, 0.6)
        assert room.transport.cleared == 0                       # ... it doesn't stop the correction
        buffered = [e for e in events(room, "line") if e.get("kind") == "buffered"]
        assert buffered and buffered[0]["text"] == PH.phrases[8]
        for _ in range(60):
            if agent.floor.state != "correcting":
                break
            await silence(agent, 0.1)
        assert [e["gain"] for e in events(room, "human_gain")] == [0.15, 1.0]
        released = [e for e in events(room, "line") if e.get("final") and e["text"] == PH.phrases[8]
                    and e.get("kind") == "line"]
        assert released, "the buffered line was not handled after the correction"
        await agent.stop()
    asyncio.run(go())


def test_unverified_contradiction_never_interrupts(tmp_path):
    async def go():
        models = FakeModels(PH, contradictions={"fifth": {                 # quotes evidence that isn't there
            "contradicts": True, "evidence_id": "F1", "evidence": "due every Tuesday at noon", "confidence": 0.99,
            "correction": "Actually it's Tuesday."}})
        room, agent = make(tmp_path, ["Alex", "Mei"], models)
        await agent.start()
        await silence(agent, 2.0)
        await speak(agent, 2, PH.audio(6))
        await silence(agent, 1.0)
        assert not events(room, "correction") and not events(room, "human_gain")
        await agent.stop()
    asyncio.run(go())


def test_plain_mode_is_the_naive_baseline(tmp_path):
    """Plain answers every line and stops at any sound, so a filler does stop it there (the contrast for demos)."""
    async def go():
        models = FakeModels(PH, reply="Sure. The rent is due on the first of every month, to Mrs. Okafor, and it is "
                                      "split equally between the three of you.", tts_seconds_per_word=0.3)
        room, agent = make(tmp_path, ["Alex", "Mei"], models)
        await agent.start()
        agent.set_mode("plain")
        await silence(agent, 2.0)
        await speak(agent, 1, PH.audio(4))                       # said to Mei: smart would stay quiet
        await silence(agent, 1.2)
        assert models.calls["judge"] == 0 and agent.output.speaking
        await speak(agent, 2, PH.audio(1))                       # "Umm, okay." stops plain mode
        await silence(agent, 0.3)
        assert room.transport.cleared == 1 and not agent.output.speaking
        await agent.stop()
    asyncio.run(go())


def test_what_is_said_over_a_correction_is_kept_in_context(tmp_path):
    async def go():
        models = FakeModels(PH, contradictions={"fifth": {
            "contradicts": True, "evidence_id": "F1", "evidence": "due on the 1st", "confidence": 0.95,
            "correction": "Sorry to jump in, Mei. The fact sheet says rent is due on the 1st, not the fifth."}},
            tts_seconds_per_word=0.15)
        room, agent = make(tmp_path, ["Alex", "Mei"], models)
        await agent.start()
        await silence(agent, 2.0)
        await speak(agent, 2, PH.audio(6))
        await silence(agent, 0.8)
        assert agent.floor.state == "correcting"
        await speak(agent, 1, PH.audio(8))                       # talked over inside the guard: not stopped
        assert room.transport.cleared == 0
        for _ in range(60):
            if agent.floor.state != "correcting":
                break
            await silence(agent, 0.1)
        await silence(agent, 0.5)
        held = [line for line in agent.ledger if line["speaker"] == "Alex"]
        assert held and held[-1]["text"] == PH.phrases[8] + " [said while you were correcting]"
        await agent.stop()
    asyncio.run(go())
