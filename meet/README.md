# Meet

A small video call (2–5 people, joined by link) with an AI participant, Higgs, on slot 0. This folder's call layer
is the server relay (`server.py`, `room.py`) and the browser UI (`web/`). The AI itself is `agent.py` (`HiggsAgent`);
if that is missing or can't be imported, `stub_agent.py` is used (beep, chat echo, test commands).

    ./run.sh meet                               # http://localhost:8600/
    ./run.sh meet --https --host 0.0.0.0        # others on the LAN (self-signed cert, accept the warning)
    MEET_ADMIN=secret ./run.sh meet             # public (e.g. behind a tunnel): creating rooms needs /?admin=secret
    MEET_AGENT=stub ./run.sh meet               # force the stub agent (no Higgs calls)

    ./run.sh meet-public                        # public https://<random>.trycloudflare.com link (prints the admin link)

## Higgs in the call (`agent.py`, `floor.py`, `avatar.py`)

A switch on the call, visible to everyone, flips Higgs between two brains:

- **default:** plain Higgs Realtime as Boson's docs prescribe (`mmhm.engine.Engine(mode="server_vad")`,
  unchanged). It hears everyone mixed together, answers after every turn, and stops on any sound.
- **improved:** Third Voice decides *whether* to speak and Mm-hm decides *when*.
  - **Hearing:** each person has their own transcription session, so names are known.
  - **Line end:** a line ends after 0.45 s of silence when the per-person turn-taking model (VAP) agrees the person
    is done; mid-thought pauses get held.
  - **Deciding:** v4b judges every line (3 judge samples + 3 fact-checks, `tv_eval.DEC_SYS4`). A correction is
    rewritten to say what's wrong.
  - **Taking the floor:** a reply starts only when the room has been quiet for 0.35 s and nobody who just spoke
    still holds the floor. It is dropped if someone spoke after its line, or if it's older than 3.5 s.
  - **While Higgs talks:** a listener's "mm-hm" only ducks it. A real interruption stops it, and Higgs remembers
    only what was heard (`floor.py`, ported from `mmhm/engine.py`; 16 unit tests).
- **Avatar** (improved mode, off by default): the reply is rendered by Higgs Avatar once its speech is ready, and
  the audio waits for the first video chunk. This adds ~2.5 s. A pre-rendered idle loop plays between replies.

Scripted test (`tests/meeting_run.py`): roommates meeting B2 from Third Voice's test set, 3 people, 32 lines, 6
where the AI should speak. Each person's lines are their own fake microphone in headless Chromium, so the
participants follow the script instead of waiting for Higgs.

| B2, 3 min | Higgs spoke | spoke when it should | stayed silent when it should | cut off by people | reply starts after the line | Higgs requests |
|---|---|---|---|---|---|---|
| default | 28 times | – (it starts a reply after nearly every line) | – | 37 | not measured | 40 |
| improved | 4 | 4/6 | 26/26 | 2 | 1.3–1.7 s | 298 |
| improved + Avatar | 2 (2 dropped while rendering) | 4/6 | 26/26 | 2 | 3.8–4.6 s | 297 |

Default Higgs also speaks as one of the roommates ("Yeah, that's me", "I'll do tonight's dishes") and agreed with
a wrong amount ("Yeah, that covers everything"). Improved Higgs corrected it: "Actually, Sam, you still owe Jordan
twenty-five…". Improved misses: what "E24" means (a question only the fact sheet answers) and one implicit correction.

    python tests/make_tracks.py B2                      # fake-mic tracks from third_voice's TTS lines (no API)
    python tests/meeting_run.py --mode improved [--avatar] [--switch-at 90]   # real Higgs calls (~300 per run)
    python -m pytest tests/test_floor.py                # timing logic, no API

Env: `MEET_SAMPLES` (judge samples per line, default 3), `MEET_BUDGET` (Higgs requests per server run, default
5000), `MEET_SIGNALS=0` (no per-person turn-taking models: timing falls back to silence rules), `MEET_VOICE`.

## Rooms

`/` creates a room (title, AI name, setting, facts, optional passcode) and gives a link `/r/<id>?k=<passcode>`; the
server also prints it. Rooms live in memory; each room logs to `out/rooms/<id>.jsonl`. `/r/<id>?lab` shows the
agent's turn-taking values on the tiles.

## Wire format

- `/ws/room?room=&pid=&name=&k=`: JSON control + binary audio. Client → server: 960 bytes = 20 ms mono int16
  at 24 kHz. Server → client: 1 byte slot + 960 bytes (slot 0 = Higgs, 1..5 = people), forwarded at once to everyone
  else. Close codes: 4001 room full, 4003 wrong passcode, 4004 no such room, 4000 replaced by a newer tab.
- `/ws/video?room=&pid=` (open after joining): client → server JPEG (320x240, ~8 fps, skipped while
  `bufferedAmount` > 64 KB); server → client 1 byte slot + JPEG, newest frame per sender wins. Slot 0 = Higgs avatar:
  fMP4 chunks for a MediaSource, between `avatar_start {mime}` and `avatar_stop`; `avatar_idle {url}` loops a file.
- Server → client JSON: `roster`, `chat`, `ai_flush`, `ai_gain`, `ai_state` (+ `by` when a person changed it),
  `line`, `ai_say`, `lab`. Client → server JSON: `chat`, `set_mode`, `set_avatar`, plus `mic`/`cam` `{on}` (roster
  shows muted/camera-off).

## Agent interface (see `room.py`)

    class HiggsAgent:
        def __init__(self, room)                 # per room, at room creation
        async def start(self) / stop(self)       # room goes occupied / empty (may happen again later)
        def feed(self, slot, pcm)                # every 20 ms frame of each unmuted human; must not block
        def on_roster(self, people)              # [{"slot", "name"}]
        def on_chat(self, name, text); def set_mode(self, mode); def set_avatar(self, on)

The room gives it `room.brief`, `room.people()`, `room.mode`, `room.avatar`, `room.status`, `room.id` (sync) and
`await room.send_ai_audio(pcm)` (one 960-byte frame; the agent paces at 20 ms), `ai_flush()`, `ai_gain(g)`,
`event(dict)`, `ai_status(s)`, `send_avatar_chunk(b)`, `avatar_start(mime)`, `avatar_stop()`, `avatar_idle(url)`.
These only enqueue: each socket has its own sender, audio for a client that falls behind is dropped.

## Test

    python tests/fake_room.py      # 3 headless Chromium pages with fake voices and cameras, then HTTPS; no Higgs calls

It checks roster, audio and video between all pages, chat, mode switch, the stub's slot-0 beep, ai_gain/ai_flush,
transcript lines, captions, lab bars, avatar fMP4 playback, reload/leave/slot reuse, the 5-person cap, passcode,
admin token, bounded queues, and saves `tests/shot_room.png`.
