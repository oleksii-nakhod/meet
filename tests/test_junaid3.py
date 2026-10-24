"""junaid3: the audit fixes and the new features, without API calls."""
import asyncio
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BOSON_API_KEY", "unused")
import agent  # noqa: E402
import fact_monitor  # noqa: E402
import meet_config  # noqa: E402


# ------------------------------------------------------------------ decide(): audit defects 4 and 5
def point(facts=("Rent is due on the 1st.",)):
    return {"target_speaker": "Higgs", "setting": "house meeting", "people": ["Sam", "Mei"], "facts": list(facts),
            "context_turns": [], "current_turn": {"speaker": "Sam", "text": "Rent is due on the 5th."}}


def test_a_judges_guess_is_not_a_verified_correction():
    async def ask(kind, _sys, _user, _i):
        return '{"to": "group", "ask": false, "wrong": true}' if kind == "judge" else '{"contradicts": 0}'

    speak, d = asyncio.run(agent.decide(point(), ask, samples=1, prompt="v5"))
    assert d["reading"]["wrong"] is False and d["reading"]["judge_wrong"] is True


def test_a_slow_fact_check_does_not_hold_up_the_answer():
    async def ask(kind, _sys, _user, _i):
        if kind == "check":
            await asyncio.sleep(30)                   # an upstream that never answers
        return '{"to": "group", "ask": true, "answered": false, "can": true}'

    t0 = time.perf_counter()
    speak, _ = asyncio.run(agent.decide(point(), ask, samples=1, prompt="v5"))
    assert speak and time.perf_counter() - t0 < agent.FACT_WAIT + 0.5


def test_check_false_skips_the_old_fact_checks():
    calls = []

    async def ask(kind, _sys, _user, _i):
        calls.append(kind)
        return '{"to": "me", "ask": true}'

    asyncio.run(agent.decide(point(), ask, samples=1, prompt="v5", check=False))
    assert calls == ["judge"]


# ------------------------------------------------------------------ voice fallback never rewinds (defect 3)
def test_a_realtime_voice_failure_after_audio_ends_cleanly_without_rewinding():
    class Pool:
        async def speak(self, text, on_audio, cancelled):
            on_audio(b"\x01\x00" * 4800)              # 0.2 s of speech is out ...
            return False                              # ... then the session fails

    async def go():
        audio, done = bytearray(), asyncio.Event()
        sp = agent.RealtimeSpeaker(Pool(), audio, done.set)
        sp.feed("Rent is due on the 1st.")
        sp.close()
        await asyncio.wait_for(done.wait(), 3)
        assert len(audio) == 9600 and sp.fallback is None   # kept, not replaced by a TTS restart
    asyncio.run(go())


# ------------------------------------------------------------------ heard, not sent (defect 2)
def test_a_reply_is_heard_only_after_playback():
    r = SimpleNamespace(drain_until=None, drain_ev=None)
    heard = agent.Reply.heard
    assert not heard(r, time.perf_counter())
    ev = asyncio.Event()
    r.drain_until, r.drain_ev = time.perf_counter() + 4, ev
    assert not heard(r, time.perf_counter())
    ev.set()                                          # the avatar reported playback finished
    assert heard(r, time.perf_counter())
    r.drain_until, r.drain_ev = time.perf_counter() - 0.01, None   # WebSocket: the estimate passed
    assert heard(r, time.perf_counter())


# ------------------------------------------------------------------ barge-in probe and the correction guard
def brain():
    room = SimpleNamespace(event=AsyncMock(), log=Mock())
    a = SimpleNamespace(ai="Higgs", history=[], names=lambda: ["Sam", "Mei"], line=AsyncMock(), room=room,
                        setting="house meeting", status=AsyncMock())
    b = agent.ImprovedBrain(a)
    b._interrupt = AsyncMock()
    return b


def listener(slot=2, stream_t=None):
    return SimpleNamespace(slot=slot, name="Mei", cur=[np.zeros(480, np.int16)], voiced=1.0,
                           stream_t=time.perf_counter() if stream_t is None else stream_t)


def probe_says(b, text):
    b.probe = SimpleNamespace(transcribe=AsyncMock(return_value=text))


def test_fillers_never_stop_a_reply_and_a_real_interruption_does():
    async def go():
        b = brain()
        r = SimpleNamespace(intervention=False, marks={"play_started": time.perf_counter() - 3})
        b.playing = r
        probe_says(b, "Umm, okay, yeah.")
        await b._probe(listener(), r)
        b._interrupt.assert_not_awaited()
        probe_says(b, "Wait, what about the internet bill?")
        await b._probe(listener(), r)
        b._interrupt.assert_awaited_once()
    asyncio.run(go())


def test_a_correction_talks_over_people_for_the_guard_then_yields_to_insistence():
    async def go():
        b = brain()
        r = SimpleNamespace(intervention=True, marks={"play_started": time.perf_counter() - 0.5})
        b.playing = r
        probe_says(b, "No no, that's not what the lease says")
        await b._probe(listener(), r)                 # inside the 2 s guard
        b._interrupt.assert_not_awaited()
        r.marks["play_started"] = time.perf_counter() - 2.5
        await b._probe(listener(), r)                 # just past the guard: not yet
        b._interrupt.assert_not_awaited()
        b.insist[2] -= 0.7                            # still pressing 0.7 s later
        await b._probe(listener(), r)
        b._interrupt.assert_awaited_once()
    asyncio.run(go())


def test_a_verified_contradiction_becomes_a_protected_correction_with_its_source():
    async def go():
        b = brain()
        b.a.rendered_avatar = False
        b.voice = lambda: None
        li = SimpleNamespace(slot=2, name="Mei")
        v = fact_monitor.Verdict(True, "F1", "due on the 1st", 0.95, "Sorry to jump in, Mei: rent is due on the 1st.",
                                 "fact sheet", "fact", "verified")
        await b._correct(li, 1, "Rent is due on the 5th.", time.perf_counter(), v)
        p = b.pending
        assert p.intervention and p.correction and p.source == "fact sheet" and "1st" in p.text
        ev = b.a.room.event.await_args_list[0].args[0]
        assert ev["type"] == "correction" and ev["source"] == "fact sheet"
        p.speaker.cancel()
    asyncio.run(go())


def test_an_unverified_contradiction_never_takes_the_floor():
    m = fact_monitor.Monitor(min_confidence=0.8, cooldown_s=0)
    items = [("F1", "Rent is due on the 1st.", "fact sheet", "fact")]
    v = m.judge({"contradicts": True, "evidence_id": "F1", "evidence": "rent is due every Tuesday", "confidence": 0.99,
                 "correction": "Actually it's Tuesday."}, items, speaker="Mei", text="Rent is due on the 5th.")
    assert not v.contradicts and "not in the cited source" in v.reason
    v = m.judge({"contradicts": True, "evidence_id": "F9", "evidence": "x", "confidence": 0.99, "correction": "x"},
                items, speaker="Mei", text="Rent is due on the 5th.")
    assert not v.contradicts


def test_person_vs_person_claims_are_evidence():
    m = fact_monitor.Monitor()
    m.remember(fact_monitor.Claim("1", "Jordan", "The plumber is coming on Friday.", time.monotonic()))
    items = m.evidence(None, "Mei", "The plumber is coming on Monday.")
    assert [(i, kind) for i, _, _, kind in items] == [("S1", "speaker"), ("G1", "general")]


# ------------------------------------------------------------------ sources, config, avatar plumbing
def test_replies_are_attributed_to_the_document_they_rest_on():
    src = [{"doc": "lease.pdf", "page": 3, "text": "Section 7. No pets without the landlord's written approval."}]
    assert agent.attribute("No pets without the landlord's written approval.", src, []) == "lease.pdf p.3"
    assert agent.attribute("Sure, happy to help!", src, []) is None


def test_meet_settings_in_dot_env_are_no_longer_dropped(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("MEET_PROACTIVE_FACTS=0\nMEET_FACT_DUCK_GAIN=0.2\n")
    monkeypatch.setenv("MEET_ENV_FILE", str(f))
    monkeypatch.delenv("MEET_PROACTIVE_FACTS", raising=False)
    monkeypatch.delenv("MEET_FACT_DUCK_GAIN", raising=False)
    meet_config.load_environment()
    assert os.environ["MEET_PROACTIVE_FACTS"] == "0" and os.environ["MEET_FACT_DUCK_GAIN"] == "0.2"


def test_each_room_gets_its_own_avatar_identity():
    import livekit_transport as lt
    assert lt.identities("roomA") != lt.identities("roomB")
    assert lt.identities("roomA") == ("agent-roomA", "avatar-roomA")


def test_the_live_avatar_does_not_switch_on_the_legacy_rendered_face():
    a = agent.HiggsAgent.__new__(agent.HiggsAgent)
    a.avatar = True
    a.room = SimpleNamespace(livekit=False, avatar_livekit=True)
    assert not a.rendered_avatar
    a.room = SimpleNamespace(livekit=False, avatar_livekit=False)
    assert a.rendered_avatar


def test_optional_models_switch_off_instead_of_falling_back_to_the_stub():
    # agent imported above without torch/maai/faster-whisper installed; those features are simply off
    assert agent.SIGNALS == agent._installed("torch", "maai") or os.environ.get("MEET_SIGNALS") == "0"
    assert agent.STT_FALLBACK in (False, True) and (agent.LOCAL_STT is None) == (not agent.STT_FALLBACK)


# ------------------------------------------------------------------ immediate corrections, any source
class ChatModels:
    """check -> the scripted checker verdict; confirm -> the scripted second opinion."""

    def __init__(self, verdict, confirm):
        self.verdict, self.confirm, self.calls = verdict, confirm, []

    async def chat(self, system, user, **kw):
        self.calls.append("confirm" if "uncontroversial knowledge" in system and "Statement:" in user else "check")
        return self.confirm if self.calls[-1] == "confirm" else self.verdict


FLAT = ('{"contradicts": true, "evidence_id": "G1", "evidence": "The Earth is round.", "confidence": 0.97, '
        '"correction": "Sorry to jump in, Sam, the Earth is round, not flat."}')


def test_a_general_knowledge_error_needs_a_second_opinion():
    m = fact_monitor.Monitor(cooldown_s=0)
    models = ChatModels(FLAT, '{"false": true, "confidence": 0.98}')
    v = asyncio.run(m.check(models, None, speaker="Sam", text="Since the Earth is flat, ships fall off.", setting="", history=""))
    assert v.contradicts and v.kind == "general" and v.source == "general knowledge" and models.calls == ["check", "confirm"]
    m2 = fact_monitor.Monitor(cooldown_s=0)
    v = asyncio.run(m2.check(ChatModels(FLAT, '{"false": false, "confidence": 0.6}'), None, speaker="Sam",
                             text="Since the Earth is flat, ships fall off.", setting="", history=""))
    assert not v.contradicts and "not confirmed" in v.reason


def test_general_knowledge_can_be_switched_off():
    m = fact_monitor.Monitor(general=False)
    assert all(kind != "general" for *_, kind in m.evidence(None, "Sam", "The Earth is flat."))


def midbrain(verdict):
    b = brain()
    b.a.history = []
    b.voice = lambda: None
    b.a.rendered_avatar = False
    b.probe = SimpleNamespace(transcribe=AsyncMock(return_value="So the lease says pets are totally fine, so we"))
    b.monitor = SimpleNamespace(check=AsyncMock(return_value=verdict))
    li = SimpleNamespace(slot=2, name="Mei", cur=[np.zeros(480, np.int16)], voiced=2.0, line_seq=4, corrected_seq=-1,
                         mid_text="")
    return b, li


def test_a_wrong_fact_is_corrected_mid_sentence_while_they_keep_talking():
    async def go():
        v = fact_monitor.Verdict(True, "D1", "no pets are allowed without the landlord's written approval", 0.93,
                                 "Sorry to jump in, Mei: the lease says no pets without written approval.",
                                 "lease.pdf p.3", "document", "verified")
        b, li = midbrain(v)
        await b._midcheck(li, 4)                       # line 4 is still being said
        assert b.pending is not None and b.pending.intervention and b.pending.source == "lease.pdf p.3"
        assert li.corrected_seq == 4
        b.pending.speaker.cancel()
        b.pending = None
        await b._midcheck(li, 4)                       # the same line is never corrected twice
        assert b.pending is None
    asyncio.run(go())


def test_a_mid_sentence_check_for_a_line_that_already_ended_is_dropped():
    async def go():
        b, li = midbrain(fact_monitor.Verdict(False, reason="no contradiction"))
        li.line_seq = 5                                # they finished; the line was committed meanwhile
        await b._midcheck(li, 4)
        b.monitor.check.assert_not_awaited()
    asyncio.run(go())


def test_the_corrected_speaker_carrying_on_does_not_cut_the_correction_short():
    async def go():
        b = brain()
        r = SimpleNamespace(intervention=True, marks={"play_started": time.perf_counter() - 3})
        b.playing = r
        probe_says(b, "and that's why old maps just stop at the edge of the world")
        monologue = listener(stream_t=time.perf_counter() - 8)    # began before the correction did
        for _ in range(3):
            await b._probe(monologue, r)
            b.insist[2] = b.insist.get(2, time.perf_counter()) - 1
        b._interrupt.assert_not_awaited()
    asyncio.run(go())
