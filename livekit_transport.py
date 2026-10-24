"""LiveKit media transport and Boson's realtime LiveKit avatar for Meet.

The existing control WebSocket remains responsible for roster, transcript, chat,
and mode switches. LiveKit carries microphone, camera, AI audio, and avatar media.
"""
from __future__ import annotations

import asyncio
import json
import os
import ssl
import uuid
from contextlib import suppress
from datetime import timedelta
from typing import Any

import aiohttp
import certifi
import numpy as np
from livekit import api, rtc
from livekit.agents.voice.avatar import DataStreamAudioOutput

from meet_config import load_environment


load_environment()

SR = 24000
FRAME_SAMPLES = 480
AGENT_IDENTITY = "meet-agent"
AVATAR_IDENTITY = "boson-avatar"
BOSON_BASE = os.environ.get("BOSON_BASE_URL", "https://api.boson.ai/v1").rstrip("/")
DEFAULT_AVATAR = os.environ.get("BOSON_AVATAR_ID", "builtin_claire_v1")
MAX_DURATION = int(os.environ.get("MEET_AVATAR_MAX_DURATION", "1800"))
TLS = ssl.create_default_context(cafile=certifi.where())


def human_identity(pid: str) -> str:
    return f"human-{pid}"


def token(
    room_name: str,
    identity: str,
    name: str,
    *,
    publish: bool = True,
    subscribe: bool = True,
    publish_on_behalf: str | None = None,
) -> str:
    load_environment()
    grant = api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=publish,
        can_subscribe=subscribe,
        can_publish_data=True,
        can_publish_sources=["camera", "microphone"] if publish else None,
    )
    access = (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(identity)
        .with_name(name)
        .with_grants(grant)
        .with_ttl(timedelta(seconds=MAX_DURATION + 600))
    )
    if publish_on_behalf:
        access = access.with_kind("agent").with_attributes({"lk.publish_on_behalf": publish_on_behalf})
    return access.to_jwt()


def browser_credentials(room_name: str, pid: str, name: str) -> dict[str, str]:
    return {
        "url": os.environ["LIVEKIT_URL"],
        "token": token(room_name, human_identity(pid), name),
        "identity": human_identity(pid),
        "agent_identity": AGENT_IDENTITY,
        "avatar_identity": AVATAR_IDENTITY,
    }


class LiveKitBridge:
    """One server-side LiveKit participant for one Meet room."""

    def __init__(self, meet_room: Any):
        self.meet_room = meet_room
        self.room_name = f"meet-{meet_room.id}"
        self.rtc_room = rtc.Room()
        self.source: rtc.AudioSource | None = None
        self.to_avatar: DataStreamAudioOutput | None = None
        self.avatar_session_id: str | None = None
        self.tasks: set[asyncio.Task[Any]] = set()
        self.connected = False
        self.gain = 1.0
        self.avatar_segment = False
        self._http: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        if self.connected:
            return
        agent_token = token(self.room_name, AGENT_IDENTITY, self.meet_room.brief["ai"])
        await self.rtc_room.connect(os.environ["LIVEKIT_URL"], agent_token)
        self.connected = True

        self.source = rtc.AudioSource(SR, 1, queue_size_ms=200)
        track = rtc.LocalAudioTrack.create_audio_track("meet-agent-audio", self.source)
        opts = rtc.TrackPublishOptions()
        opts.source = rtc.TrackSource.SOURCE_MICROPHONE
        await self.rtc_room.local_participant.publish_track(track, opts)

        @self.rtc_room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            _publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            self._start_audio(track, participant)

        for participant in self.rtc_room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track is not None:
                    self._start_audio(publication.track, participant)

    def _start_audio(self, track: rtc.Track, participant: rtc.RemoteParticipant) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO or participant.identity == AVATAR_IDENTITY:
            return
        task = asyncio.create_task(self._read_audio(track, participant.identity))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _read_audio(self, track: rtc.Track, identity: str) -> None:
        """Decode a participant track into the 20 ms frames expected by HiggsAgent.feed."""
        pending = bytearray()
        try:
            async for event in rtc.AudioStream(track, sample_rate=SR, num_channels=1, frame_size_ms=20):
                pending.extend(bytes(event.frame.data))
                while len(pending) >= FRAME_SAMPLES * 2:
                    frame = bytes(pending[: FRAME_SAMPLES * 2])
                    del pending[: FRAME_SAMPLES * 2]
                    human = self.meet_room.by_livekit_identity(identity)
                    if human is not None and human.mic:
                        self.meet_room.agent.feed(human.slot, frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a replacement track or disconnect normally ends this reader
            self.meet_room.log({"type": "livekit_audio_end", "identity": identity, "error": repr(exc)[:200]})

    async def send_ai_audio(self, pcm: bytes) -> None:
        if not self.connected or not pcm:
            return
        if self.gain < 0.999:
            values = np.frombuffer(pcm, np.int16).astype(np.float32)
            pcm = np.clip(values * self.gain, -32768, 32767).astype(np.int16).tobytes()
        frame = rtc.AudioFrame(pcm, SR, 1, len(pcm) // 2)
        if self.to_avatar is not None:
            await self.to_avatar.capture_frame(frame)
            self.avatar_segment = True
        elif self.source is not None:
            await self.source.capture_frame(frame)

    async def flush_audio(self) -> None:
        # DataStreamAudioOutput expects one playback-finished notification per captured segment. Sending the avatar
        # a redundant clear after a segment already flushed makes it report playback_finished twice and can race the
        # next segment, which sounds like a restart. Only clear when this bridge actually has an open segment.
        if self.to_avatar is not None and self.avatar_segment:
            self.to_avatar.clear_buffer()
            self.to_avatar.flush()
            self.avatar_segment = False
        if self.source is not None:
            self.source.clear_queue()

    async def on_ai_status(self, status: str) -> None:
        if status == "listening" and self.to_avatar is not None and self.avatar_segment:
            self.to_avatar.flush()
            self.avatar_segment = False

    async def set_avatar(self, enabled: bool) -> None:
        if enabled:
            await self._start_avatar()
        else:
            await self._stop_avatar()

    async def _start_avatar(self) -> None:
        if self.avatar_session_id:
            return
        if not os.environ.get("BOSON_API_KEY"):
            raise RuntimeError("BOSON_API_KEY is required for the live avatar")
        avatar_token = token(
            self.room_name,
            AVATAR_IDENTITY,
            AVATAR_IDENTITY,
            publish=True,
            subscribe=False,
            publish_on_behalf=AGENT_IDENTITY,
        )
        payload = {
            "avatar_id": DEFAULT_AVATAR,
            "transport": {
                "type": "livekit",
                "url": os.environ["LIVEKIT_URL"],
                "room_name": self.room_name,
                "participant_token": avatar_token,
                "participant_identity": AVATAR_IDENTITY,
                "publisher_identity": AGENT_IDENTITY,
                "audio_source": "data_stream",
            },
            "output": {"width": 480, "height": 640},
            "max_duration_seconds": MAX_DURATION,
        }
        if self._http is None:
            connector = aiohttp.TCPConnector(ssl=TLS)
            self._http = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {os.environ['BOSON_API_KEY']}"},
                timeout=aiohttp.ClientTimeout(total=60),
                connector=connector,
            )
        # A session that was just deleted (avatar toggled off and on, a rejoin) still counts against the key for a few
        # seconds and the next create gets HTTP 429, so wait and retry as avatar.render does.
        for attempt in range(8):
            async with self._http.post(
                f"{BOSON_BASE}/avatar/sessions",
                json=payload,
                headers={"Idempotency-Key": str(uuid.uuid4())},
            ) as response:
                status, text = response.status, await response.text()
            if status != 429:
                break
            await asyncio.sleep(0.5 + 0.5 * attempt)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"error": text[:300]}
        if status not in (200, 201) or data.get("status") != "active" or not data.get("id"):
            raise RuntimeError(f"Boson avatar HTTP {status}: {data.get('error', data)}")
        self.avatar_session_id = str(data["id"])
        self.to_avatar = DataStreamAudioOutput(
            room=self.rtc_room,
            destination_identity=AVATAR_IDENTITY,
            sample_rate=SR,
            wait_remote_track=rtc.TrackKind.KIND_AUDIO,
        )

    async def _stop_avatar(self) -> None:
        self.to_avatar = None
        self.avatar_segment = False
        sid, self.avatar_session_id = self.avatar_session_id, None
        if sid and self._http:
            with suppress(Exception):
                async with self._http.delete(f"{BOSON_BASE}/avatar/sessions/{sid}") as response:
                    await response.read()

    async def stop(self) -> None:
        await self._stop_avatar()
        for task in list(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        if self.connected:
            with suppress(Exception):
                await self.rtc_room.disconnect()
        self.connected = False
        if self._http:
            await self._http.close()
            self._http = None

