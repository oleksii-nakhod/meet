"""Higgs Avatar live: a realtime avatar session in a LiveKit room, the face for the meeting's Higgs tile.

The rendered path (avatar.py) can only draw a face once a reply's whole speech exists (~1.5 s to the first frame),
so a reply waits for it. A realtime session draws the face while the speech plays: Meet's server streams Higgs's
20 ms output frames, as it sends them to the call, to a Boson avatar participant over a LiveKit data stream
(`DataStreamAudioOutput` from livekit-agents, the protocol Boson's avatar speaks), and the avatar publishes
lip-synced video plus that audio into the LiveKit room. Each page joins the room as a viewer only (subscribe, no
publish) and plays the avatar's tracks on the AI tile; the call itself stays on Meet's own sockets.

One session per Meet room while the face is on (`MEET_LIVE_AVATAR_ID`, Boson presets: builtin_claire_v1, _emma_v1,
_james_v1; custom faces aren't offered). Needs a LiveKit Cloud project (Boson refuses self-hosted servers):
LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET in the environment or the nearest .env up the tree.
MEET_LIVE_AVATAR=0 goes back to the rendered face."""
import asyncio
import os
import secrets
import time
import uuid
from collections import deque
from datetime import timedelta

import aiohttp
from livekit import api, rtc
from livekit.agents.utils import wait_for_track_publication
from livekit.agents.voice.avatar import DataStreamAudioOutput

from mmhm.higgs import BASE, HDR, SR

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
AVATAR_ID = os.environ.get("MEET_LIVE_AVATAR_ID", "builtin_claire_v1")
MAX_S = int(os.environ.get("MEET_LIVE_AVATAR_MAX", "3600"))  # the account cap is 3600 s by default; then a new one
# The first frames of an utterance go out together: the avatar plays what it has, and Meet's tick loop sends one frame
# every 20 ms with some jitter, so a cushion keeps its audio from running dry mid-word (tests/live_avatar_probe.py).
LEAD_FRAMES = int(os.environ.get("MEET_LIVE_AVATAR_LEAD", "5"))
END_GAP = 0.3  # no frame for this long: the utterance is over (the avatar goes back to listening)
# The avatar is heard ~1.1-1.7 s after a frame is pushed (median ~1.3 s; the same when the audio is pushed seconds
# ahead, so it is the avatar's pipeline, not pacing), and after clear() what is in that pipeline still plays for ~1.2 s:
# pages mute it themselves on ai_flush for DRAIN s. Its playback_finished RPC comes within ~0.1 s of the last sound,
# which keeps `lag` (push -> heard) measured (2026-10-23, tests/live_avatar_probe.py).
LAG0, DRAIN = 1.3, 1.25


def settings():
    """{LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET} from the environment, else the nearest .env up the tree;
    None if any is missing or MEET_LIVE_AVATAR=0."""
    if os.environ.get("MEET_LIVE_AVATAR", "1") == "0":
        return None
    d = HERE
    while not all(os.environ.get(k) for k in KEYS) and d != os.path.dirname(d):
        env = os.path.join(d, ".env")
        if os.path.exists(env):
            for line in open(env):
                k, _, v = line.strip().partition("=")
                if k in KEYS and v and not os.environ.get(k):
                    os.environ[k] = v.strip().strip('"').strip("'")
        d = os.path.dirname(d)
    return {k: os.environ[k] for k in KEYS} if all(os.environ.get(k) for k in KEYS) else None


def token(cfg, room, identity, *, publish, subscribe, ttl, on_behalf=None):
    t = (api.AccessToken(cfg["LIVEKIT_API_KEY"], cfg["LIVEKIT_API_SECRET"]).with_identity(identity).with_name(identity)
         .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=publish, can_subscribe=subscribe,
                                      can_publish_data=publish,
                                      can_publish_sources=["camera", "microphone"] if publish else None))
         .with_ttl(timedelta(seconds=ttl)))
    if on_behalf:  # LiveKit UIs then show the avatar as part of the agent
        t = t.with_kind("agent").with_attributes({"lk.publish_on_behalf": on_behalf})
    return t.to_jwt()


class LiveAvatar:
    """One realtime avatar session for one Meet room. start() returns once the avatar is in the LiveKit room and
    publishing; push() / clear() never block (a sender task streams the frames). `dead` is set when the avatar leaves
    (expired, ended by Boson, network)."""

    def __init__(self, meet_id, cfg=None):
        self.cfg = cfg or settings()
        tag = f"{meet_id}-{secrets.token_hex(3)}"
        self.room_name = f"meet-{tag}"
        # unique per Meet room: DataStreamAudioOutput keys its playback RPC handlers by this identity, process-wide
        self.avatar_identity, self.agent_identity = f"avatar-{tag}", f"higgs-{tag}"
        self.room, self.out, self.session_id = None, None, None
        self.q, self.ev = deque(maxlen=50 * 60), asyncio.Event()  # up to a minute of speech waits for the sender
        self.dead, self.closed, self.task = asyncio.Event(), False, None
        self.seg = False          # an utterance (byte stream) is open
        self.frames = 0           # frames sent (stats)
        self.t_start = None
        self.lag, self.drain = LAG0, DRAIN
        self.t_last = 0.0         # perf_counter of the last frame sent
        self.unheard = False      # frames sent that the avatar hasn't finished playing

    async def start(self, timeout=40.0):
        t0 = time.perf_counter()
        ttl = MAX_S + 600
        body = {"avatar_id": AVATAR_ID, "max_duration_seconds": MAX_S, "output": {"width": 480, "height": 640},
                "transport": {"type": "livekit", "url": self.cfg["LIVEKIT_URL"], "room_name": self.room_name,
                              "participant_token": token(self.cfg, self.room_name, self.avatar_identity, publish=True,
                                                         subscribe=False, ttl=ttl, on_behalf=self.agent_identity),
                              "participant_identity": self.avatar_identity, "publisher_identity": self.agent_identity,
                              "audio_source": "data_stream"}}
        # the agent joins first, so the room exists and the avatar's arrival can be seen
        self.room = rtc.Room()
        self.room.on("participant_disconnected", lambda p: p.identity == self.avatar_identity and self.dead.set())
        self.room.on("disconnected", lambda *_: self.dead.set())
        await self.room.connect(self.cfg["LIVEKIT_URL"], token(self.cfg, self.room_name, self.agent_identity, publish=True,
                                                                subscribe=True, ttl=ttl))
        async with aiohttp.ClientSession(headers=HDR, timeout=aiohttp.ClientTimeout(total=timeout)) as http:
            for attempt in range(4):
                async with http.post(f"{BASE}/avatar/sessions", json=body,
                                     headers={"Idempotency-Key": str(uuid.uuid4())}) as r:
                    data = await r.json(content_type=None)
                    if r.status == 429 and data.get("error", {}).get("code") in ("capacity_exhausted", "rate_limit_exceeded"):
                        await asyncio.sleep(min(10.0, float(r.headers.get("Retry-After") or 2)))
                        continue
                    if r.status not in (200, 201) or not data.get("id"):
                        raise RuntimeError(f"avatar session: HTTP {r.status} {str(data.get('error', data))[:200]}")
                    self.session_id = data["id"]
                    break
            else:
                raise RuntimeError("avatar session: no capacity (429)")
        self.out = DataStreamAudioOutput(self.room, destination_identity=self.avatar_identity, sample_rate=SR,
                                         wait_remote_track=rtc.TrackKind.KIND_AUDIO)
        self.out.on("playback_finished", self._finished)
        # ready = the avatar publishes its audio track (capture_frame waits for the same thing)
        await asyncio.wait_for(wait_for_track_publication(room=self.room, identity=self.avatar_identity,
                                                          kind=rtc.TrackKind.KIND_AUDIO), timeout)
        self.task = asyncio.ensure_future(self._send())
        self.t_start = time.perf_counter() - t0
        return self.t_start

    def push(self, pcm: bytes):
        """One 20 ms frame (int16 at 24 kHz) of what Higgs says, as it is said."""
        if not self.closed:
            self.q.append(bytes(pcm))
            self.ev.set()

    def playing(self):
        """Is the avatar still saying something already sent? (It plays ~`lag` s behind the push.)"""
        return bool(self.q) or (self.unheard and time.perf_counter() - self.t_last < self.lag + 2.0)

    def _finished(self, ev):
        if ev.interrupted:  # after clear_buffer the avatar answers at once, before the drain: not a timing
            return
        if not self.q and not self.seg:
            self.unheard = False
            lag = time.perf_counter() - self.t_last
            if 0.3 < lag < 4.0:
                self.lag = 0.7 * self.lag + 0.3 * lag

    def clear(self):
        """Higgs stopped (interrupted, mode switch): drop what the avatar hasn't played yet."""
        self.q.clear()
        self.unheard = False
        if self.out is not None:  # also after the utterance's stream closed: the avatar plays well behind the push
            self.out.clear_buffer()
            if self.seg:
                self.seg = False
                self.out.flush()

    async def _send(self):
        last = 0.0
        while not self.closed:
            try:
                await asyncio.wait_for(self.ev.wait(), END_GAP)
            except asyncio.TimeoutError:
                pass
            self.ev.clear()
            if not self.q:
                if self.seg and time.perf_counter() - last >= END_GAP:
                    self.seg = False
                    self.out.flush()  # end of the utterance
                continue
            if not self.seg and len(self.q) < LEAD_FRAMES:  # an utterance starts with a cushion (LEAD_FRAMES)
                await asyncio.sleep(0.02 * (LEAD_FRAMES - len(self.q)))
            self.seg = True
            while self.q:
                f = self.q.popleft()
                try:
                    await self.out.capture_frame(rtc.AudioFrame(f, SR, 1, len(f) // 2))
                except Exception:  # noqa: BLE001  (the room went away: `dead` says so)
                    self.dead.set()
                    return
                self.frames += 1
                self.unheard = True
            last = self.t_last = time.perf_counter()

    def viewer(self, who):
        """What a page needs to watch the avatar: a subscribe-only token for this room (`who` names it)."""
        return {"url": self.cfg["LIVEKIT_URL"], "room": self.room_name, "identity": self.avatar_identity,
                "token": token(self.cfg, self.room_name, f"viewer-{who}-{secrets.token_hex(3)}", publish=False,
                               subscribe=True, ttl=MAX_S + 600)}

    async def close(self):
        self.closed = True
        if self.task:
            self.task.cancel()
        if self.session_id:  # best effort: Boson also ends it when the room empties or it expires
            try:
                async with aiohttp.ClientSession(headers=HDR, timeout=aiohttp.ClientTimeout(total=15)) as http:
                    async with http.delete(f"{BASE}/avatar/sessions/{self.session_id}") as r:
                        await r.read()
            except Exception:  # noqa: BLE001
                pass
        if self.room is not None:
            try:
                await self.room.disconnect()
            except Exception:  # noqa: BLE001
                pass
