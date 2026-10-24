"""Meeting memory (meeting_memory.py): running notes beside the conversation, used by replies after 14 lines."""
import asyncio
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent  # noqa: E402
import meeting_memory  # noqa: E402


def lines(n, start=0):
    return [{"speaker": "Sam" if i % 2 else "Mei", "text": f"line {i}"} for i in range(start, start + n)]


class Models:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    async def chat(self, system, user, **kw):
        self.calls.append(user)
        a = self.answers.pop(0) if self.answers else "- noted"
        if isinstance(a, Exception):
            raise a
        return a


# ------------------------------------------------------------------ the notes
def test_clean_accepts_notes_and_rejects_rambling():
    assert meeting_memory.clean("none") == ""
    assert meeting_memory.clean("None.") == ""
    assert meeting_memory.clean("Updated notes:\n- Venue: the library (Mei)\n\n- Budget $400") == \
        "- Venue: the library (Mei)\n- Budget $400"
    assert meeting_memory.clean("") is None
    assert meeting_memory.clean("word " * 500) is None


def test_clean_drops_transcript_lines_copied_into_the_notes():
    said = ["Six dollars for a latte.", "Fine. I'll book the room by Thursday."]
    out = "- Sam: Six dollars for a latte.\n- six dollars for a latte\n- Mei will book the room by Thursday"
    assert meeting_memory.clean(out, said) == "- Mei will book the room by Thursday"
    assert meeting_memory.clean("- Sam: Six dollars for a latte.", said) is None   # all copies: keep the old notes


def test_update_folds_in_all_but_the_newest_lines_once_enough_are_waiting():
    m = meeting_memory.Memory()
    assert not m.due(11)            # 11 - 4 margin = 7 lines ready: not yet
    assert m.due(12)
    h = lines(12)
    models = Models("- Venue is the library (Mei)")
    asyncio.run(m.update(models, h, ai="Higgs", setting="a meeting"))
    assert m.notes == "- Venue is the library (Mei)" and m.upto == 8 and not m.running
    assert "Mei: line 0" in models.calls[0] and "line 7" in models.calls[0] and "line 8" not in models.calls[0]
    assert not m.due(12)
    h += lines(8, 12)
    asyncio.run(m.update(models, h, ai="Higgs", setting="a meeting"))
    assert m.upto == 16 and "Notes so far:\n- Venue is the library (Mei)" in models.calls[1]
    assert "line 8" in models.calls[1] and "line 7" not in models.calls[1]


def test_catching_up_on_a_long_history_goes_in_batches():
    m = meeting_memory.Memory()
    models = Models("- a", "- b", "- c")
    asyncio.run(m.update(models, lines(100), ai="Higgs", setting="a meeting"))
    assert len(models.calls) == 3 and m.upto == 96 and m.notes == "- c"


def test_a_failed_or_unusable_update_keeps_the_old_notes_and_retries_from_the_same_place():
    m = meeting_memory.Memory(notes="- Budget $400", upto=8)
    log = Mock()
    asyncio.run(m.update(Models(RuntimeError("chat 503")), lines(20), ai="Higgs", setting="a meeting", log=log))
    assert m.notes == "- Budget $400" and m.upto == 8 and not m.running and m.failures == 1
    asyncio.run(m.update(Models(""), lines(20), ai="Higgs", setting="a meeting", log=log))
    assert m.notes == "- Budget $400" and m.upto == 8 and m.failures == 2
    models = Models("- Budget $400\n- Venue: library")
    asyncio.run(m.update(models, lines(20), ai="Higgs", setting="a meeting", log=log))
    assert m.upto == 16 and "line 8" in models.calls[0]
    assert log.call_count == 3


def test_a_cancelled_update_releases_the_lock():
    class Slow:
        async def chat(self, *a, **k):
            await asyncio.sleep(10)

    async def go():
        m = meeting_memory.Memory()
        t = asyncio.ensure_future(m.update(Slow(), lines(20), ai="Higgs", setting="a meeting"))
        await asyncio.sleep(0.01)
        assert m.running
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
        assert not m.running and m.upto == 0
    asyncio.run(go())


def test_context_short_meetings_are_unchanged_and_no_line_falls_in_a_gap():
    m = meeting_memory.Memory()
    assert m.context(14) == ("", 14)
    m.notes, m.upto = "- Venue: library", 30
    notes, recent = m.context(40)
    assert "- Venue: library" in notes and recent == 14   # notes cover everything before the window
    m.upto = 20
    assert m.context(40)[1] == 20                        # notes lag: the window starts at the first uncovered line
    m.upto = 0
    assert m.context(100)[1] == 30                       # never more than 30 raw lines
    m.notes = ""
    assert m.context(40)[0] == ""


# ------------------------------------------------------------------ in the reply path (ImprovedBrain.on_line)
class Pool:
    def __init__(self):
        self.seen = []

    def ask(self, convo, **kw):  # records synchronously: the drafts are futures that a silent line never awaits
        self.seen.append(convo)

        async def nothing():
            return ""
        return nothing()


def brain(history, monkeypatch):
    monkeypatch.setattr(agent, "PROACTIVE_FACTS", False)
    monkeypatch.setattr(agent, "decide", AsyncMock(return_value=(False, {"reading": {}, "why": "test", "votes": [],
                                                                          "checks": []})))
    room = SimpleNamespace(event=AsyncMock(), log=Mock(), id="r1")
    a = SimpleNamespace(ai="Higgs", history=history, names=lambda: ["Sam", "Mei"], line=AsyncMock(), room=room,
                        setting="house meeting", status=AsyncMock(), facts=[], avatar=False, rendered_avatar=False)
    b = agent.ImprovedBrain(a)
    b.pools = {k: Pool() for k in ("judge", "check", "write", "write0")}
    b.pool_key = (tuple(a.names()), tuple(a.facts))
    li = SimpleNamespace(slot=1, name="Sam", gate=SimpleNamespace(settle=Mock()), barge=SimpleNamespace(reset=Mock()),
                         in_flight=0, decided_early=False)
    b.listeners = {1: li, 2: SimpleNamespace(slot=2, name="Mei")}
    return b, li


def test_a_reply_after_a_long_stretch_gets_the_notes_and_the_recent_lines(monkeypatch):
    async def go():
        b, li = brain(lines(30), monkeypatch)
        b.memory.notes, b.memory.upto = "- The venue is the library (Mei)", 26
        b.models = SimpleNamespace(chat=AsyncMock(return_value="- noted"), close=AsyncMock())
        await b.on_line(li, "Remind me what we settled about the venue", time.perf_counter(), False)
        convo = b.pools["write0"].seen[0]
        assert convo.startswith("[Notes on earlier parts of this meeting")
        assert "- The venue is the library (Mei)" in convo
        assert "line 16" in convo and "line 15" not in convo     # the usual 14 lines
        assert convo.endswith("Sam: Remind me what we settled about the venue\n\nReply as Higgs:")
    asyncio.run(go())


def test_short_meetings_and_memory_off_give_the_exact_old_prompt(monkeypatch):
    async def go():
        b, li = brain(lines(10), monkeypatch)
        await b.on_line(li, "Remind me what we settled about the venue", time.perf_counter(), False)
        ctx = "\n".join(f"{t['speaker']}: {t['text']}" for t in lines(10))
        old = f"Conversation so far:\n{ctx}\nSam: Remind me what we settled about the venue\n\nReply as Higgs:"
        assert b.pools["write0"].seen[0] == old
        b, li = brain(lines(30), monkeypatch)
        b.memory = None                                          # MEET_MEMORY=0
        await b.on_line(li, "Remind me what we settled about the venue", time.perf_counter(), False)
        ctx = "\n".join(f"{t['speaker']}: {t['text']}" for t in lines(30)[-14:])
        assert b.pools["write0"].seen[0] == old.replace(old.split("\nSam: Remind")[0], f"Conversation so far:\n{ctx}")
    asyncio.run(go())


def test_new_lines_start_a_background_update_that_survives_a_mode_switch(monkeypatch):
    async def go():
        b, li = brain(lines(11), monkeypatch)
        b.models = SimpleNamespace(chat=AsyncMock(return_value="- The venue is the library (Mei)"), close=AsyncMock())
        await b.on_line(li, "Remind me what we settled about the venue", time.perf_counter(), False)  # 12 lines
        await asyncio.gather(*b.tasks)
        assert b.memory.upto == 8 and b.memory.notes == "- The venue is the library (Mei)"
        logged = [c.args[0] for c in b.a.room.log.call_args_list if c.args[0].get("type") == "memory"]
        assert [(d["upto"], d["lines"], d["notes"]) for d in logged] == [(8, 8, "- The venue is the library (Mei)")]
        assert agent.ImprovedBrain(b.a).memory is b.memory       # switched back from default mode: same notes
    asyncio.run(go())
