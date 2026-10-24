"""The one interface between the AI and the room's media, whichever transport carries it.

Voice-only rooms use the WebSocket transport (ws.py); a room with the live avatar on upgrades to LiveKit
(livekit.py). The agent never knows which one is in use.
"""
from __future__ import annotations

from typing import Callable, Protocol

FRAME_SAMPLES = 480           # 20 ms at 24 kHz
FRAME_BYTES = FRAME_SAMPLES * 2


class MediaTransport(Protocol):
    name: str

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def send_ai_audio(self, frame: bytes) -> None:
        """One 20 ms frame of the AI's voice (int16 mono 24 kHz). The caller paces in real time."""

    async def end_ai_segment(self) -> None:
        """The current utterance has no more audio (lets an avatar finish and report playback)."""

    async def clear_ai_audio(self) -> None:
        """Drop everything queued for playback (an interruption)."""

    def playout_delay(self) -> float:
        """Seconds between a frame being sent and being heard, for floor bookkeeping."""

    def on_playback_finished(self, callback: Callable[[], None] | None) -> None:
        """Called when the far end reports the last segment was heard; transports that cannot report it never
        call it, and the agent falls back to sent time + playout_delay()."""
