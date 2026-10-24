"""Regressions for acknowledgements arriving while a reply is playing or waiting."""
import asyncio
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent


def test_conversational_noises_but_not_requests():
    for text in ("umm", "ummmmm", "ah ahh", "okay", "mm-hm", "uh-huh", "okay okay yeah ahh umm"):
        assert agent.conversational_ack(text), text
    for text in ("okay but stop", "umm what is the deadline?", "ah, that is wrong", "wait", "stop talking"):
        assert not agent.conversational_ack(text), text


def brain():
    room = SimpleNamespace(event=AsyncMock())
    a = SimpleNamespace(ai="Higgs", history=[], names=lambda: ["Sam"], line=AsyncMock(), room=room)
    b = agent.ImprovedBrain(a)
    li = SimpleNamespace(slot=1, name="Sam", gate=SimpleNamespace(settle=Mock()),
                         barge=SimpleNamespace(reset=Mock()))
    b._interrupt = AsyncMock()
    return b, li


def test_ack_neither_interrupts_nor_replaces_pending_question():
    async def check():
        b, li = brain()
        reply = SimpleNamespace(intervention=False, marks={"play_started": time.perf_counter() - 3})
        b.playing = reply
        pending = object()
        b.pending = pending
        for text in ("umm ahh okay yeah okay", "okay", "mm-hm"):
            await b.on_line(li, text, time.perf_counter(), True)
        assert b.playing is reply and b.pending is pending
        b._interrupt.assert_not_awaited()
        assert b.line_no == 0
        assert len(b.a.line.await_args_list) == 3
    asyncio.run(check())


def test_stop_interrupts_only_current_ordinary_reply():
    async def check():
        b, li = brain()
        b.playing = SimpleNamespace(intervention=False, marks={"play_started": time.perf_counter() - 3})
        await b.on_line(li, "Higgs, stop talking.", time.perf_counter(), False)
        b._interrupt.assert_awaited_once()
        b._interrupt.reset_mock()
        b.playing.intervention = True
        await b.on_line(li, "Higgs, stop talking.", time.perf_counter(), False)
        b._interrupt.assert_not_awaited()
        b.playing.intervention = False
        await b.on_line(li, "Higgs, stop talking.", time.perf_counter() - 10, False)
        b._interrupt.assert_not_awaited()
    asyncio.run(check())


def test_long_ack_transcribed_after_playback_ends_is_not_a_new_request():
    async def check():
        b, li = brain()
        b.a.history.append({"speaker": "Higgs", "text": "Earth's tilt causes the seasons."})
        await b.on_line(li, "Ah, okay, okay, yeah.", time.perf_counter(), False)
        assert b.a.line.await_args.args[3]["why"] == "listener acknowledgement"
        assert b.line_no == 0 and b.pending is None
    asyncio.run(check())
