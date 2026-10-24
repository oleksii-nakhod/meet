"""Rooms, participants and the control plane.

Wire format on /ws/room (one socket per participant):
  client -> server  text: JSON control messages
                    binary: 1 byte kind + payload; kind 1 = 20 ms audio (960 bytes, int16 mono 24 kHz), 2 = JPEG frame
  server -> client  text: JSON events
                    binary: 1 byte kind + 1 byte slot + payload; slot 0 is the AI, 1..5 are people
Each socket has its own sender task with bounded queues: a slow client loses audio, never stalls the room.
"""
from __future__ import annotations

import asyncio
import json
import secrets
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

from fastapi import WebSocket

from . import config
from .obs import NullLog, RoomLog, now

MAX_PEOPLE = 5
KIND_AUDIO, KIND_VIDEO = 1, 2


class Agent(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def feed(self, slot: int, pcm: bytes) -> None: ...
    def on_roster(self, people: list[dict]) -> None: ...
    def on_chat(self, slot: int, name: str, text: str) -> None: ...
    async def set_avatar(self, mode: str) -> None: ...
    def set_mode(self, mode: str) -> None: ...
    def on_kb_changed(self) -> None: ...


def new_id(n: int = 8) -> str:
    return secrets.token_urlsafe(n)[:11]


class Client:
    """One participant's socket and its sender."""

    AUDIO_MAX = 40   # frames queued across all slots (~0.27 s per slot with 3 speakers) before the oldest go

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.control: asyncio.Queue[str | None] = asyncio.Queue(maxsize=1000)
        self.audio: deque[bytes] = deque()
        self.video: dict[int, bytes] = {}
        self.wake = asyncio.Event()
        self.dropped = 0
        self.task = asyncio.ensure_future(self._send_loop())

    def send_json(self, msg: dict) -> None:
        try:
            self.control.put_nowait(json.dumps(msg))
        except asyncio.QueueFull:
            self.dropped += 1
        self.wake.set()

    def send_audio(self, slot: int, pcm: bytes) -> None:
        self.audio.append(bytes((KIND_AUDIO, slot)) + pcm)
        while len(self.audio) > self.AUDIO_MAX:
            self.audio.popleft()
            self.dropped += 1
        self.wake.set()

    def send_video(self, slot: int, jpeg: bytes) -> None:
        self.video[slot] = bytes((KIND_VIDEO, slot)) + jpeg   # newest frame per sender wins
        self.wake.set()

    async def _send_loop(self) -> None:
        try:
            while True:
                await self.wake.wait()
                self.wake.clear()
                while not self.control.empty():
                    msg = self.control.get_nowait()
                    if msg is None:
                        return
                    await self.ws.send_text(msg)
                while self.audio:
                    await self.ws.send_bytes(self.audio.popleft())
                for slot in list(self.video):
                    await self.ws.send_bytes(self.video.pop(slot))
        except Exception:  # noqa: BLE001  (socket gone: the receive loop cleans up)
            pass

    async def close(self) -> None:
        try:
            self.control.put_nowait(None)
        except asyncio.QueueFull:
            pass
        self.wake.set()
        try:
            await asyncio.wait_for(self.task, 1.0)
        except Exception:  # noqa: BLE001
            self.task.cancel()


@dataclass
class Person:
    slot: int
    pid: str
    name: str
    client: Client
    host: bool = False
    mic: bool = True
    cam: bool = True
    noisy: bool = False

    def public(self) -> dict:
        return {"slot": self.slot, "name": self.name, "host": self.host, "mic": self.mic, "cam": self.cam,
                "noisy": self.noisy}


@dataclass
class Brief:
    title: str = "Meeting"
    ai: str = "Higgs"
    setting: str = "online meeting"
    facts: list[str] = field(default_factory=list)
    agenda: str = ""


class Room:
    def __init__(self, rid: str, brief: Brief, *, passcode: str = "", avatar: str = "off", collection: str = "",
                 log: bool = True):
        self.id, self.brief, self.passcode = rid, brief, passcode
        self.host_key = new_id(12)
        self.avatar = avatar            # "off" | "live"
        self.mode = "smart"             # "smart" | "plain" (a baseline for demos)
        self.collection = collection or rid
        self.people: dict[int, Person] = {}
        s = config.settings()
        self.log = RoomLog(s.path(s.log_dir), rid) if log else NullLog()
        self.agent: Agent | None = None
        self.transport: Any = None      # MediaTransport, set by the server
        self.created = now()
        self.ai_state = "idle"
        self.human_gain = 1.0
        self.lines: list[dict] = []     # transcript, for late joiners
        self.notes: dict = {}
        self.log.write({"type": "created", "room": rid, "title": brief.title, "ai": brief.ai, "avatar": avatar})

    # ------------------------------------------------------------ membership
    def free_slot(self) -> int | None:
        return next((s for s in range(1, MAX_PEOPLE + 1) if s not in self.people), None)

    def by_pid(self, pid: str) -> Person | None:
        return next((p for p in self.people.values() if p.pid == pid), None)

    def roster(self) -> list[dict]:
        return [p.public() for p in sorted(self.people.values(), key=lambda p: p.slot)]

    def names(self) -> list[str]:
        return [p.name for p in sorted(self.people.values(), key=lambda p: p.slot)]

    def info(self) -> dict:
        return {"id": self.id, "title": self.brief.title, "ai": self.brief.ai, "setting": self.brief.setting,
                "agenda": self.brief.agenda, "passcode": bool(self.passcode), "people": len(self.people),
                "max": MAX_PEOPLE, "avatar": self.avatar, "mode": self.mode, "livekit": config.settings().livekit_enabled}

    # ------------------------------------------------------------ fan-out
    def broadcast(self, msg: dict, *, except_slot: int | None = None) -> None:
        for p in list(self.people.values()):
            if p.slot != except_slot:
                p.client.send_json(msg)

    def event(self, msg: dict, *, log: bool = True) -> None:
        """Broadcast and log an event."""
        if msg.get("type") == "line":
            self._keep_line(msg)
        elif msg.get("type") == "notes":
            self.notes = msg
        elif msg.get("type") == "ai_state":
            self.ai_state = msg.get("state", self.ai_state)
        elif msg.get("type") == "human_gain":
            self.human_gain = msg.get("gain", 1.0)
        self.broadcast(msg)
        if log:
            self.log.write(msg)

    def _keep_line(self, msg: dict) -> None:
        for i, line in enumerate(self.lines):
            if line.get("id") == msg.get("id"):
                self.lines[i] = msg
                return
        self.lines.append(msg)
        del self.lines[:-200]

    def relay_audio(self, slot: int, pcm: bytes) -> None:
        for p in list(self.people.values()):
            if p.slot != slot:
                p.client.send_audio(slot, pcm)

    def relay_video(self, slot: int, jpeg: bytes) -> None:
        for p in list(self.people.values()):
            if p.slot != slot:
                p.client.send_video(slot, jpeg)

    def ai_audio(self, pcm: bytes) -> None:
        for p in list(self.people.values()):
            p.client.send_audio(0, pcm)

    def welcome(self) -> dict:
        return {"type": "welcome", "room": self.info(), "roster": self.roster(), "lines": self.lines[-100:],
                "notes": self.notes, "ai_state": self.ai_state, "human_gain": self.human_gain}


class Rooms:
    def __init__(self):
        self.rooms: dict[str, Room] = {}

    def create(self, brief: Brief, **kw) -> Room:
        rid = new_id()
        while rid in self.rooms:
            rid = new_id()
        room = Room(rid, brief, **kw)
        self.rooms[rid] = room
        return room

    def get(self, rid: str) -> Room | None:
        return self.rooms.get(rid)
