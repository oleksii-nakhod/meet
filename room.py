"""Rooms: who is in the call, the per-client send queues, and the interface the AI participant codes against.

Each human has a slot 1..5 (slot 0 is the AI). Audio is relayed on the control socket as 1 byte slot + 960 bytes
(20 ms, mono int16, 24 kHz); camera JPEGs on a separate video socket as 1 byte slot + JPEG. Every socket has its
own sender task, so one slow client never delays the others: audio for a client whose queue is full is dropped,
video keeps only the newest pending frame per sender.

The agent (agent.HiggsAgent, or stub_agent.HiggsAgent when that is missing) gets:
  room.brief, room.people(), room.mode, room.avatar, room.status  (plain attributes / sync)
  room.live, room.live_state  (the live face, below)
  await room.send_ai_audio(pcm), ai_flush(), ai_gain(g), event(d), ai_status(s),
        send_avatar_chunk(b), avatar_start(mime), avatar_stop(), avatar_idle(url)   (enqueue only, never block)

Live face (live_avatar.py, when LiveKit is configured): while the avatar is on and someone is in the room, the room
keeps a realtime Boson avatar session in a LiveKit room. Everything sent to slot 0 also goes to it (ai_flush clears
it), and each page gets a subscribe-only token (`live_avatar` message) to play the avatar's video and audio on the
AI tile instead of slot 0's audio. live_state: None (off / not configured), "starting", "on", "failed" (the agent
then falls back to rendered video); agent.set_avatar() is called again whenever it changes."""
import asyncio
import json
import os
import queue
import secrets
import threading
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "out", "rooms")
MAX_HUMANS = 5
FRAME = 960  # 20 ms of int16 at 24 kHz

try:
    import live_avatar
    LIVE = live_avatar.settings()
except ImportError as e:  # livekit-agents not installed: rendered faces only
    live_avatar, LIVE = None, None
    print(f"live avatar off: {e}", flush=True)

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


class _Logger:
    """One background thread appending queued lines to their files (kept open)."""

    def __init__(self):
        self.q, self.files = queue.SimpleQueue(), {}
        threading.Thread(target=self._run, daemon=True).start()

    def put(self, path, line):
        self.q.put((path, line))

    def _run(self):
        while True:
            path, line = self.q.get()
            try:
                f = self.files.get(path)
                if f is None:
                    if len(self.files) > 64:  # rooms come and go
                        for old in list(self.files.values()):
                            old.close()
                        self.files.clear()
                    f = self.files[path] = open(path, "a", buffering=1)
                f.write(line)
            except OSError:
                pass


LOGGER = _Logger()


class Human:
    def __init__(self, pid, name, slot, ws):
        self.pid, self.name, self.slot = pid, name, slot
        self.out = Out(ws)
        self.cam, self.mic = True, True
        self.video = None  # Out of this person's /ws/video socket


class Room:
    def __init__(self, title, ai="Higgs", setting="", facts=(), passcode="", agenda="", rid=None):
        self.id = rid or secrets.token_urlsafe(8)
        self.passcode = passcode or ""
        # a script from web/agendas.js for the people to read from; the agent never sees it (only the brief), and
        # every move of the shared cursor is logged, so a run's lines and decisions can be matched to script steps
        self.agenda, self.script_step = agenda or "", 0
        self.brief = {"title": title or "Meeting", "ai": ai or "Higgs", "setting": setting or "",
                      "facts": [f for f in facts if f]}
        self.mode, self.avatar, self.status = "default", False, "listening"
        self.humans = {}  # slot -> Human
        self.last_slot = {}  # pid -> slot it had, so a reconnect gets the same tile back
        self.idle_url = None
        self.live, self.live_state, self._live_task = None, None, None
        self.created = time.time()
        os.makedirs(LOG_DIR, exist_ok=True)
        self.log_path = os.path.join(LOG_DIR, f"{self.id}.jsonl")
        self._started, self._life, self.errors = False, asyncio.Lock(), 0
        self.agent = HiggsAgent(self)
        self.log({"type": "created", **self.brief, "passcode": bool(self.passcode), "agenda": self.agenda})

    # ---- for the agent ----
    def people(self):
        return [{"slot": h.slot, "name": h.name} for h in sorted(self.humans.values(), key=lambda h: h.slot)]

    async def send_ai_audio(self, pcm: bytes):
        b = b"\x00" + bytes(pcm)
        if self.live:
            self.live.push(b[1:])
        for h in list(self.humans.values()):
            h.out.lossy(b)

    async def ai_flush(self):
        if self.live:
            self.live.clear()
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

    # ---- the live face ----
    def _live_sync(self):
        """Start or end the live avatar to match: avatar on, someone here, LiveKit configured."""
        want = bool(LIVE and self.avatar and self.humans)
        if want and self.live_state is None and not self._live_task:
            self.live_state = "starting"  # now: the agent mustn't set up a rendered face meanwhile
            self._live_task = asyncio.ensure_future(self._live_run())
        elif not want:
            if self._live_task:
                self._live_task.cancel()  # its finally closes the session
            elif self.live_state == "failed":
                self._set_live(None, None)

    async def _live_run(self):
        """Hold a live avatar while it's wanted; a session that ends (expired, Boson, network) is replaced."""
        la, tries = None, 0
        try:
            while True:
                self._set_live(None, "starting")
                la = live_avatar.LiveAvatar(self.id, LIVE)
                try:
                    took = await la.start()
                except Exception as e:  # noqa: BLE001  (no LiveKit / no capacity: rendered faces instead)
                    await la.close()
                    la = None
                    tries += 1
                    await self.event({"type": "ai_error", "error": f"live avatar: {e!r}"[:300]})
                    if tries >= 3:
                        self._set_live(None, "failed")
                        return
                    await asyncio.sleep(2.0 * tries)
                    continue
                tries = 0
                self.log({"type": "live_avatar", "on": True, "start_s": round(took, 2), "room": la.room_name})
                self._set_live(la, "on")
                await la.dead.wait()
                self.log({"type": "live_avatar", "on": False, "why": "ended", "frames": la.frames})
                self._set_live(None, "starting")
                await la.close()
                la = None
        except asyncio.CancelledError:
            if self.live_state != "failed":
                self._set_live(None, None)
        finally:
            self._live_task = None
            if la is not None:
                self.log({"type": "live_avatar", "on": False, "why": "off", "frames": la.frames})
                asyncio.ensure_future(la.close())

    def _set_live(self, la, state):
        was = self.live
        self.live, self.live_state = la, state
        if la is not None:
            for h in list(self.humans.values()):
                h.out.json(self._live_msg(h))
        elif was is not None:
            self._all({"type": "live_avatar", "on": False})
        self._safe(self.agent.set_avatar, self.avatar)

    def _live_msg(self, h):
        return {"type": "live_avatar", "on": True, "drain": round(self.live.drain, 2), **self.live.viewer(h.slot)}

    # ---- internals ----
    def ai_state(self):
        return {"type": "ai_state", "mode": self.mode, "avatar": self.avatar, "status": self.status}

    def log(self, d):
        """Queue a line for the room log. Written by a background thread: out/ is on NFS here, where an append took
        2-12 ms and sometimes far more, on the event loop that clocks everyone's audio (a 339 ms tick)."""
        LOGGER.put(self.log_path, json.dumps({"t": round(time.time(), 3), **d}) + "\n")

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
        if self.live:
            h.out.json(self._live_msg(h))
        if self.agenda:
            h.out.json({"type": "script", "step": self.script_step})
        self.roster()
        asyncio.ensure_future(self._lifecycle())
        return h

    def close(self, code=4005):
        """Disconnect everyone (4005: a new meeting replaced this one at the same link; pages reload into it)."""
        self.log({"type": "replaced"})
        for h in list(self.humans.values()):
            h.out.close(code)
            if h.video:
                h.video.close(code)

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
            self._live_sync()

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
            self._live_sync()
            self._safe(self.agent.set_avatar, self.avatar)
        elif t == "script" and isinstance(m.get("step"), int) and 0 <= m["step"] < 1000:
            self.script_step = m["step"]
            await self.event({"type": "script", "step": self.script_step, "by": h.name})
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
