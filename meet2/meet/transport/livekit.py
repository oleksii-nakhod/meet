"""The live avatar (F1 add-on): the AI's voice goes to a Boson avatar in a LiveKit room, which publishes synchronized
audio + video that browsers subscribe to. LiveKit is used only while the avatar is on; people's own audio and video
stay on the room WebSocket.

    agent-<room>  (this process)  --data stream audio-->  avatar-<room>  (Boson)  --A/V tracks-->  browsers

Identities are unique per room. The SDK keeps playback-finished handlers in a class-wide dict keyed by identity, so a
shared identity made two rooms' callbacks cross (Meet 1 audit, defect 6). The avatar's playback-finished report is
what tells the floor an utterance was heard.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from contextlib import suppress
from datetime import timedelta

import aiohttp

from .. import config
from ..boson import TLS

SR = 24000


def identities(room_id: str) -> tuple[str, str]:
    return f"agent-{room_id}", f"avatar-{room_id}"


def token(identity: str, name: str, room_name: str, *, publish: bool, subscribe: bool,
          publish_on_behalf: str | None = None) -> str:
    from livekit import api
    s = config.settings()
    grant = api.VideoGrants(room_join=True, room=room_name, can_publish=publish, can_subscribe=subscribe,
                            can_publish_data=True)
    access = (api.AccessToken(s.livekit_api_key, s.livekit_api_secret).with_identity(identity).with_name(name)
              .with_grants(grant).with_ttl(timedelta(seconds=s.avatar_max_duration + 600)))
    if publish_on_behalf:
        access = access.with_kind("agent").with_attributes({"lk.publish_on_behalf": publish_on_behalf})
    return access.to_jwt()


def browser_credentials(room, person) -> dict:
    """A subscribe-only token: browsers only watch the avatar here; their own media stays on the WebSocket."""
    s = config.settings()
    agent_id, avatar_id = identities(room.id)
    return {"url": s.livekit_url, "token": token(f"human-{room.id}-{person.slot}", person.name, f"meet-{room.id}",
                                                 publish=False, subscribe=True),
            "agent_identity": agent_id, "avatar_identity": avatar_id}


class AvatarBridge:
    """MediaTransport for the AI's voice while the avatar is on."""

    name = "livekit-avatar"

    def __init__(self, room):
        self.room = room
        self.room_name = f"meet-{room.id}"
        self.agent_id, self.avatar_id = identities(room.id)
        self.rtc_room = None
        self.out = None
        self.session_id: str | None = None
        self.segment_open = False
        self._finished_cb: Callable[[], None] | None = None
        self._http: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        from livekit import rtc
        from livekit.agents.voice.avatar import DataStreamAudioOutput
        s = config.settings()
        if not s.livekit_enabled:
            raise RuntimeError("LiveKit is not configured")
        self.rtc_room = rtc.Room()
        await self.rtc_room.connect(s.livekit_url, token(self.agent_id, self.room.brief.ai, self.room_name,
                                                         publish=True, subscribe=False))
        await self._start_avatar_session()
        self.out = DataStreamAudioOutput(self.rtc_room, destination_identity=self.avatar_id, sample_rate=SR,
                                         wait_remote_track=rtc.TrackKind.KIND_AUDIO)
        self.out.on("playback_finished", self._on_finished)
        self.room.log.write({"type": "avatar_started", "session": self.session_id})

    async def _start_avatar_session(self) -> None:
        s = config.settings()
        payload = {
            "avatar_id": s.boson_avatar_id,
            "transport": {
                "type": "livekit", "url": s.livekit_url, "room_name": self.room_name,
                "participant_token": token(self.avatar_id, self.avatar_id, self.room_name, publish=True,
                                           subscribe=False, publish_on_behalf=self.agent_id),
                "participant_identity": self.avatar_id, "publisher_identity": self.agent_id,
                "audio_source": "data_stream",
            },
            "output": {"width": 480, "height": 640},
            "max_duration_seconds": s.avatar_max_duration,
        }
        self._http = aiohttp.ClientSession(headers={"Authorization": f"Bearer {s.boson_api_key}"},
                                           connector=aiohttp.TCPConnector(ssl=TLS),
                                           timeout=aiohttp.ClientTimeout(total=60))
        async with self._http.post(f"{s.boson_base_url}/avatar/sessions", json=payload,
                                   headers={"Idempotency-Key": str(uuid.uuid4())}) as r:
            text = await r.text()
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = {"error": text[:300]}
            if r.status not in (200, 201) or not data.get("id"):
                raise RuntimeError(f"Boson avatar HTTP {r.status}: {data.get('error', data)}"[:300])
            self.session_id = str(data["id"])

    def _on_finished(self, ev) -> None:
        self.segment_open = False
        if self._finished_cb:
            self._finished_cb()

    # ------------------------------------------------------------ MediaTransport
    async def send_ai_audio(self, frame: bytes) -> None:
        from livekit import rtc
        if self.out is None:
            return
        await self.out.capture_frame(rtc.AudioFrame(frame, SR, 1, len(frame) // 2))
        self.segment_open = True

    async def end_ai_segment(self) -> None:
        if self.out is not None and self.segment_open:
            self.out.flush()

    async def clear_ai_audio(self) -> None:
        if self.out is not None and self.segment_open:
            self.out.clear_buffer()
            self.out.flush()
            self.segment_open = False

    def playout_delay(self) -> float:
        return 0.6        # the avatar renders before it plays; "heard" normally comes from playback-finished

    def on_playback_finished(self, callback: Callable[[], None] | None) -> None:
        self._finished_cb = callback

    async def stop(self) -> None:
        sid, self.session_id = self.session_id, None
        if sid and self._http:
            with suppress(Exception):
                s = config.settings()
                async with self._http.delete(f"{s.boson_base_url}/avatar/sessions/{sid}") as r:
                    await r.read()
        if self._http:
            await self._http.close()
            self._http = None
        if self.rtc_room is not None:
            with suppress(Exception):
                await self.rtc_room.disconnect()
            self.rtc_room = None
        self.out = None
        self.room.log.write({"type": "avatar_stopped"})
