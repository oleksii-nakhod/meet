"""Voice-only media over the room's WebSockets: the default, no extra service needed for 1-5 people.

Humans' frames are relayed by the server (Room.relay_audio); this class only carries the AI's voice. The browser
cannot report when audio was heard, so playback-finished is never called and the agent uses sent time plus the
browser's jitter buffer (playout_delay).
"""
from __future__ import annotations

from typing import Callable

from .. import config


class WSTransport:
    name = "websocket"

    def __init__(self, room):
        self.room = room

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send_ai_audio(self, frame: bytes) -> None:
        self.room.ai_audio(frame)

    async def end_ai_segment(self) -> None:
        pass

    async def clear_ai_audio(self) -> None:
        self.room.broadcast({"type": "ai_flush"})

    def playout_delay(self) -> float:
        return config.settings().playout_delay_s

    def on_playback_finished(self, callback: Callable[[], None] | None) -> None:
        pass
