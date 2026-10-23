"""Rooms: who is in the call, the per-client send queues, and the interface the AI participant codes against.

Each human has a slot 1..5 (slot 0 is the AI). Audio is relayed on the control socket as 1 byte slot + 960 bytes
(20 ms, mono int16, 24 kHz); camera JPEGs on a separate video socket as 1 byte slot + JPEG. Every socket has its
own sender task, so one slow client never delays the others: audio for a client whose queue is full is dropped,
video keeps only the newest pending frame per sender.

The agent (agent.HiggsAgent, or stub_agent.HiggsAgent when that is missing) gets:
  room.brief, room.people(), room.mode, room.avatar, room.status  (plain attributes / sync)
  await room.send_ai_audio(pcm), ai_flush(), ai_gain(g), event(d), ai_status(s),
        send_avatar_chunk(b), avatar_start(mime), avatar_stop(), avatar_idle(url)   (enqueue only, never block)"""
import asyncio
import json
import os
import secrets
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "out", "rooms")
MAX_HUMANS = 5
FRAME = 960  # 20 ms of int16 at 24 kHz

if os.environ.get("MEET_AGENT") == "stub":  # the call-layer tests: no Higgs calls
    from stub_agent import HiggsAgent
else:
    try:
        from agent import HiggsAgent
    except ImportError as e:
        print(f"agent.py not usable ({e}); using stub_agent", flush=True)
        from stub_agent import HiggsAgent


class Out:
    """One socket's outgoing traffic, sent by its own task. `send` is ordered and always delivered (control JSON,
    avatar fMP4); `lossy` is ordered but dropped past `max_lossy` queued (audio); `latest` keeps one item per key,
    newest wins (camera frames). A client that never drains its ordered queue gets disconnected."""

    def __init__(self, ws, max_lossy=15, max_queue=600):
        self.ws, self.max_lossy, self.max_queue = ws, max_lossy, max_queue
        self.q, self.latest_q, self.n_lossy, self.dropped = deque(), {}, 0, 0
        self.ev = asyncio.Event()
        self.closed, self.code = False, 1000
        self.task = asyncio.ensure_future(self._run())

    def send(self, item):
        if self.closed:
            return
        if len(self.q) >= self.max_queue:
            self.close()
            return
        self.q.append((False, item))
        self.ev.set()

    def lossy(self, item):
        if self.closed:
            return
        if self.n_lossy >= self.max_lossy:
            self.dropped += 1
            return
        self.n_lossy += 1
        self.q.append((True, item))
        self.ev.set()

    def latest(self, key, item):
        if not self.closed:
            self.latest_q[key] = item
            self.ev.set()

    def json(self, obj):
        self.send(json.dumps(obj))

    def close(self, code=1000):
        if not self.closed:
            self.closed, self.code = True, code
        self.ev.set()

    async def _run(self):
        try:
            while not self.closed:
                await self.ev.wait()
                self.ev.clear()
                while not self.closed and (self.q or self.latest_q):
                    if self.q:
                        is_lossy, item = self.q.popleft()
                        self.n_lossy -= is_lossy
                    else:
                        item = self.latest_q.pop(next(iter(self.latest_q)))
                    if isinstance(item, str):
                        await self.ws.send_text(item)
                    else:
                        await self.ws.send_bytes(item)
        except Exception:
            pass
        finally:
            self.closed = True
            try:
                await self.ws.close(self.code)
            except Exception:
                pass


class Human:
    def __init__(self, pid, name, slot, ws):
        self.pid, self.name, self.slot = pid, name, slot
        self.out = Out(ws)
        self.cam, self.mic = True, True
        self.video = None  # Out of this person's /ws/video socket


class Room:
    def __init__(self, title, ai="Higgs", setting="", facts=(), passcode=""):
        self.id = secrets.token_urlsafe(8)
        self.passcode = passcode or ""
        self.brief = {"title": title or "Meeting", "ai": ai or "Higgs", "setting": setting or "",
                      "facts": [f for f in facts if f]}
        self.mode, self.avatar, self.status = "default", False, "listening"
        self.humans = {}  # slot -> Human
        self.last_slot = {}  # pid -> slot it had, so a reconnect gets the same tile back
        self.idle_url = None
        self.created = time.time()
        os.makedirs(LOG_DIR, exist_ok=True)
        self.log_path = os.path.join(LOG_DIR, f"{self.id}.jsonl")
        self._started, self._life, self.errors = False, asyncio.Lock(), 0
        self.agent = HiggsAgent(self)
        self.log({"type": "created", **self.brief, "passcode": bool(self.passcode)})

    # ---- for the agent ----
    def people(self):
        return [{"slot": h.slot, "name": h.name} for h in sorted(self.humans.values(), key=lambda h: h.slot)]

    async def send_ai_audio(self, pcm: bytes):
        b = b"\x00" + bytes(pcm)
        for h in list(self.humans.values()):
            h.out.lossy(b)

    async def ai_flush(self):
        self._all({"type": "ai_flush"})

    async def ai_gain(self, g):
        self._all({"type": "ai_gain", "gain": max(0.0, min(1.0, float(g)))})

    async def event(self, d: dict):
        """Broadcast a JSON event to every human (and log it). An ai_state event also updates room.mode/avatar/status."""
        if d.get("type") == "ai_state":
            self.mode, self.avatar = d.get("mode", self.mode), bool(d.get("avatar", self.avatar))
            self.status = d.get("status", self.status)
            d = {**d, **self.ai_state()}
        if d.get("type") != "lab":
            self.log(d)
        self._all(d)

    async def ai_status(self, status):
        await self.event({"type": "ai_state", "status": status})

    async def send_avatar_chunk(self, b: bytes):
        self._video(b"\x00" + bytes(b))

    async def avatar_start(self, mime='video/mp4; codecs="avc1.42E01E"'):
        self._video(json.dumps({"type": "avatar_start", "mime": mime}))

    async def avatar_stop(self):
        self._video(json.dumps({"type": "avatar_stop"}))

    async def avatar_idle(self, url):
        self.idle_url = url
        self._video(json.dumps({"type": "avatar_idle", "url": url}))

    # ---- internals ----
    def ai_state(self):
        return {"type": "ai_state", "mode": self.mode, "avatar": self.avatar, "status": self.status}

    def log(self, d):
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"t": round(time.time(), 3), **d}) + "\n")
        except OSError:
            pass

    def _all(self, d):
        s = json.dumps(d)
        for h in list(self.humans.values()):
            h.out.send(s)

    def _video(self, item):
        for h in list(self.humans.values()):
            if h.video:
                h.video.send(item)

    def roster(self):
        people = [{"slot": h.slot, "name": h.name, "cam": h.cam, "mic": h.mic}
                  for h in sorted(self.humans.values(), key=lambda h: h.slot)]
        ai = {"name": self.brief["ai"], "mode": self.mode, "avatar": self.avatar, "status": self.status}
        for h in list(self.humans.values()):
            h.out.json({"type": "roster", "you": h.slot, "people": people, "ai": ai, "title": self.brief["title"]})
        self._safe(self.agent.on_roster, self.people())

    def _safe(self, fn, *a):
        try:
            fn(*a)
        except Exception as e:
            self.errors += 1
            if self.errors <= 50:  # feed() runs 50 times a second per person: don't flood the log
                self.log({"type": "agent_error", "where": fn.__name__, "error": repr(e)})

    def by_pid(self, pid):
        return next((h for h in self.humans.values() if h.pid == pid), None)

    def join(self, pid, name, ws):
        """Add a human (or replace their old connection on reconnect). Returns the Human, or None if the room is full."""
        old = self.by_pid(pid)
        if old:  # 4000 = replaced: the old page must not reconnect. The video socket (same pid) carries over.
            old.out.close(4000)
            slot = old.slot
        else:
            free = [s for s in range(1, MAX_HUMANS + 1) if s not in self.humans]
            if not free:
                return None
            slot = self.last_slot.get(pid) if self.last_slot.get(pid) in free else free[0]
        h = Human(pid, name, slot, ws)
        if old:
            h.video, h.cam, h.mic = old.video, old.cam, old.mic
        self.humans[slot] = h
        self.last_slot[pid] = slot
        self.log({"type": "join", "slot": slot, "name": name, "rejoin": bool(old)})
        h.out.json(self.ai_state())
        self.roster()
        asyncio.ensure_future(self._lifecycle())
        return h

    def leave(self, h):
        if self.humans.get(h.slot) is not h:
            return  # already replaced by a reconnect
        del self.humans[h.slot]
        h.out.close()
        if h.video:
            h.video.close()
        self.log({"type": "leave", "slot": h.slot, "name": h.name})
        self.roster()
        asyncio.ensure_future(self._lifecycle())

    async def _lifecycle(self):
        """agent.start() when the room goes from empty to occupied, agent.stop() when it empties (serialized)."""
        async with self._life:
            try:
                if self.humans and not self._started:
                    self._started = True
                    await self.agent.start()
                elif not self.humans and self._started:
                    self._started = False
                    await self.agent.stop()
            except Exception as e:
                self.log({"type": "agent_error", "where": "lifecycle", "error": repr(e)})

    def on_audio(self, h, pcm):
        if len(pcm) != FRAME:
            return
        b = bytes([h.slot]) + pcm
        for o in list(self.humans.values()):
            if o is not h:
                o.out.lossy(b)
        self._safe(self.agent.feed, h.slot, pcm)

    async def on_text(self, h, m):
        t = m.get("type")
        if t == "chat":
            text = str(m.get("text", "")).strip()[:2000]
            if text:
                await self.event({"type": "chat", "from": h.name, "slot": h.slot, "text": text})
                self._safe(self.agent.on_chat, h.name, text)
        elif t == "set_mode" and m.get("mode") in ("default", "improved"):
            self.mode = m["mode"]
            await self.event({**self.ai_state(), "by": h.name})
            self._safe(self.agent.set_mode, self.mode)
        elif t == "set_avatar":
            self.avatar = bool(m.get("on"))
            await self.event({**self.ai_state(), "by": h.name})
            self._safe(self.agent.set_avatar, self.avatar)
        elif t in ("cam", "mic"):
            setattr(h, t, bool(m.get("on")))
            self.roster()

    def on_video(self, h, jpeg):
        if not jpeg or len(jpeg) > 256_000:
            return
        b = bytes([h.slot]) + jpeg
        for o in list(self.humans.values()):
            if o is not h and o.video:
                o.video.latest(h.slot, b)

    def attach_video(self, h, ws):
        if h.video:
            h.video.close()
        h.video = Out(ws, max_queue=400)
        if self.idle_url:
            h.video.json({"type": "avatar_idle", "url": self.idle_url})
        return h.video
