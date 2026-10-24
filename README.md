# Meet

A small video call (2–5 people, joined by link) with an AI participant, Higgs, built on Boson AI's Higgs Audio
(Realtime, STT, TTS, Avatar). Plain Higgs Realtime answers after every turn and stops at any sound; Meet's "improved"
Higgs decides *whether* to speak (it stays out of conversations between people, answers questions put to it, corrects
wrong facts) and *when* (it waits out thinking pauses, keeps talking through a listener's "mm-hm", stops for a real
interruption). A switch on the call flips between the two so the difference can be heard live.

## Setup

    python3.12 -m venv .venv && . .venv/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: CPU-only torch, a much smaller download
    pip install -r requirements.txt
    pip install --no-deps maai==0.2.25      # turn-taking models (MaAI); its GUI/audio-device deps are stubbed
    cp .env.example .env                    # and put your BOSON_API_KEY in it (or export BOSON_API_KEY)
                                            # + LIVEKIT_URL/_API_KEY/_API_SECRET for the live face (optional, below)
    playwright install chromium             # tests that use a browser only

CPU is enough: each person in an improved-mode room runs the turn-taking models in their own process (~0.7 core).
The models download from Hugging Face on first use. `ffmpeg` is needed only to re-render the avatar's idle loop.

## Run

    ./run.sh                                    # http://localhost:8600/  (create a room, open its link)
    ./run.sh serve --https --host 0.0.0.0       # others on the LAN (self-signed cert in certs/, accept the warning)
    MEET_ADMIN=secret ./run.sh                  # public (e.g. behind a tunnel): creating rooms needs /?admin=secret
    MEET_AGENT=stub ./run.sh                    # the stub agent (beep, chat echo; no Higgs calls)
    ./run.sh public                             # public https://<random>.trycloudflare.com link (needs cloudflared)
    ./run.sh tests                              # unit tests, no API calls
    ./run.sh check                              # 6 interactive scenes against real Higgs (~3 min)

`./run.sh` is `python server.py --port 8600` with the venv's Python.

## Layout

- `server.py`, `room.py`: the call: rooms, the WebSocket audio/video relay, the agent interface. `web/`: the browser
  UI (create page with meeting presets, call page, avatar assets).
- `agent.py` (`HiggsAgent`): Higgs on slot 0, both brains. `floor.py`: when to take, keep and give up the floor.
  `avatar.py`: Higgs Avatar rendering. `live_avatar.py`: the live face (a realtime Boson avatar in a LiveKit room;
  `web/vendor/` has the LiveKit browser SDK it plays with). `stub_agent.py`: used if `agent.py` can't be imported or `MEET_AGENT=stub`.
- `mmhm/`: Mm-hm, the turn-taking layer from the team's earlier prototype: the Higgs Realtime/TTS client
  (`higgs.py`), the default-mode engine (`engine.py`), MaAI turn-taking and backchannel models (`models.py`),
  expressive speech (`expressive.py`).
- `third_voice.py`: Third Voice, the speak/stay-silent judge and fact-check prompts (v4b) that `agent.py` builds on.
- `tests/`: unit tests, browser and bot-driven call tests, judge and writer checks; `tests/data/`: their inputs (below).
- `out/` (not in git): room logs, test results, cached TTS clips.

## Higgs in the call (`agent.py`, `floor.py`, `avatar.py`)

A switch on the call, visible to everyone, flips Higgs between two brains:

- **default:** plain Higgs Realtime as Boson's docs prescribe (`mmhm.engine.Engine(mode="server_vad")`,
  unchanged). It hears everyone mixed together, answers after every turn, and stops on any sound.
- **improved:** Third Voice decides *whether* to speak and Mm-hm decides *when*.
  - **Hearing:** each person has their own transcription session (`language: en` hint), so names are known. Sessions
    reconnect on their own: Higgs closes a Realtime session after 5 minutes without speech, and opens can fail
    under load. Higgs's own voice leaking back through someone's speakers (more than 25 dB below what it is playing)
    is ignored, so it neither interrupts Higgs nor becomes a line.
  - **Line end:** a line ends after 0.30 s of silence when the per-person turn-taking model (VAP) agrees the person
    is done; mid-thought pauses get held. If the person goes on, the pieces merge into one line (the transcript
    panel replaces the fragment).
  - **Before judging** (`prefilter`): a misheard name is repaired ("Pigs, how should we…" → "Higgs, …"); backchannels
    ("mm-hm", "yeah", "got it"), fillers ("um", "so the, uh,"), unfinished fragments and "stop / hold on" are logged
    but not judged or answered, and don't make a reply someone else is waiting for stale.
  - **Deciding:** judge v5 (`agent.DEC_SYS5`: Third Voice's v4b plus greetings, the AI's own name, follow-ups and an
    explicit one-to-one mode) on every line, 3 judge samples + 3 fact-checks. It decides as soon as 2 of 3 judges say
    speak; a failed Higgs request is an abstention, not a "no".
  - **Writing:** the reply is drafted while the judge decides, and when the line looks like it wants a reply (one-to-one,
    the AI's name, a question) its speech starts streaming then too. The writer gets the fact sheet only when the line
    is about it (it recited it in small talk otherwise); a correction has its own draft; a slow draft gets a second
    request after 0.5 s; an interrupted turn is marked `[interrupted]` in the history so the next reply answers the
    interruption.
  - **Taking the floor:** a reply starts only when the room has been quiet for 0.35 s and nobody who just spoke
    still holds the floor. It is dropped if someone spoke after its line, or if it couldn't start within 3.5 s while
    people kept talking (8 s in a quiet room). Every reply that was decided but not spoken logs `reply_skipped`.
  - **While Higgs talks:** a listener's "mm-hm" only ducks it. A real interruption stops it, and Higgs remembers
    only what was heard (`floor.py`, ported from `mmhm/engine.py`; 16 unit tests).
- **Avatar** (both modes, off by default; the face is the uncropped Boson docs portrait). Improved: a reply is voiced
  by a Realtime session reading the writer's text (its whole speech exists sooner than with TTS 3: ~1.5 s vs ~2.0 s
  for one sentence, ~2.7 s vs ~5.7 s for three; TTS 3 takes over if Realtime hasn't started within 2 s), Higgs Avatar
  renders it into a buffer, and video and audio are released together when the floor opens. A long reply renders in
  two parts: its first >= 5.5 s as soon as that much speech exists, then the rest (as in the Read My Lips prototype; Chromium
  and Firefox play both renders as one stream). Replies are normal length again (they were one sentence with TTS 3;
  `MEET_AVATAR_VOICE=tts` goes back to that). A pre-rendered "Hmm." in the same face and voice plays ~1.5–2.4 s after
  the line while it renders (`MEET_AVATAR_FILLER=0` turns it off); the answer's face starts 3.7–5.0 s after the line
  (2026-10-23, `--set 3`, 12/12; 9–15 s answers ~5–6.6 s). Only someone taking the floor (0.8 s of speech) cancels a
  reply while its face renders; a "mm" doesn't. Default: each whole Realtime reply is held until its face renders
  (~2.9–3.7 s, no filler, no early part: the wait is what plain Higgs plus a face looks like), and speech stops it as
  before. A pre-rendered idle loop plays between replies. With LiveKit configured the face is live instead (next).

### Live face (LiveKit, `live_avatar.py`; 2026-10-23)

The rendered face can only start once a reply's whole speech exists, so the reply waits for it. With `LIVEKIT_URL`,
`LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` set (a [LiveKit Cloud](https://cloud.livekit.io) project; Boson refuses
self-hosted LiveKit), turning the avatar on starts a **realtime Boson avatar session** instead, one per room, released
when the avatar is turned off or the room empties (a session that ends is replaced; 3 failed starts fall back to the
rendered face). Higgs then talks exactly as with the face off (no render wait, no "Hmm.", normal-length replies), and
`room.py` streams every 20 ms frame it sends to slot 0 to the avatar over a LiveKit data stream (`DataStreamAudioOutput`,
the protocol Boson's avatar speaks). The avatar publishes lip-synced video + that audio into a LiveKit room; each page
joins it subscribe-only (a token in the `live_avatar` message) and plays both on the AI tile, instead of slot 0's
socket audio. The call itself (people's audio and cameras) stays on Meet's own sockets.

- **Faster replies:** with real Higgs (`tests/live_avatar_e2e.py`, one-to-one, 4 questions per mode) the face is
  heard 2.5–3.4 s after the question in improved (median 3.1 s; the rendered face: 3.7–5.0 s, 5–6.6 s for long
  answers) and 2.4–2.9 s in default (median 2.7 s). Session start: ~1.2–1.5 s after the switch.
- **The lag:** the avatar says a frame ~1.1–1.7 s after it is sent (median ~1.3 s; the same when audio is pushed
  seconds ahead, so it is the avatar's pipeline). Improved Higgs accounts for it: a reply counts as playing until the
  avatar reports `playback_finished` (it comes within ~0.1 s of the last sound), so an interruption in that tail still
  stops it; "what was heard" is the sent position minus the lag (the measured lag is in each reply's `timing.live_lag`);
  the echo gate compares against Higgs's output over the last 2.6 s instead of 0.6 s.
- **Interruptions:** Meet stops Higgs ~0.5 s into the interruption as before; the avatar's pipeline would still play
  ~1.2 s after that, so pages mute the face's audio at once on `ai_flush` for `drain` (1.25 s) and the lips run on
  silently. `ai_gain` (ducking for a "mm-hm") sets its volume.
- **Faces:** Boson's presets only (`MEET_LIVE_AVATAR_ID`: `builtin_claire_v1` (default), `builtin_emma_v1`,
  `builtin_james_v1`); a custom portrait isn't offered for live sessions. Video 480x640 at 24 fps (a new viewer starts
  on LiveKit's 240x320 layer and steps up within ~10 s).
- Bots and `tests/scenarios.py` still hear slot 0 on the socket (when it is sent, not when the face says it).
- Env: `MEET_LIVE_AVATAR=0` (rendered face even with LiveKit set), `MEET_LIVE_AVATAR_ID`, `MEET_LIVE_AVATAR_MAX`
  (session length, 3600 s, the account cap; then a new session).

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

    python tests/make_tracks.py B2                      # fake-mic tracks from the TTS lines in tests/data/tts (no API)
    python tests/meeting_run.py --mode improved [--avatar] [--switch-at 90]   # real Higgs calls (~300 per run)
    python -m pytest tests/test_floor.py                # timing logic, no API

(This table is from 2026-10-22. Rerun 2026-10-23 with everything below: improved 5/6, 25/26, all 32 lines heard,
replies 1.1–2.7 s, all five things it said right (both amounts, what E24 means, the lease section, the rent date); the
extra "speak" was a garbled "Wren-five." and was dropped unheard. Default the same morning: spoke 24 times, cut off
by people 36 times.)

### Interactive scenarios (2026-10-23, `tests/scenarios.py`)

37 scenes / 98 scored steps: one-to-one and 2–3 person calls; greetings, follow-ups, statements, thinking pauses,
fillers, humans talking to each other, group questions, wrong facts, someone answering first, "stop", "stay quiet",
"say that again", a late joiner, typed chat, a noisy mic, Higgs's voice leaking back at −30 dB, real (AMI) and TTS
backchannels and interruptions, quiet ones at −6/−10 dB, other AI names and settings.

| | steps right | reply heard after the line (median, p90) |
|---|---|---|
| improved, start of the night (set 1 only) | 29/49 | 1.85 s, 2.44 s |
| improved, end of the night (`final3`, sets 1+2 incl. quiet mics) | **101/102** | **1.46 s, 2.01 s** |
| default (plain Higgs), same scenes | 64/104 | 1.26 s, 1.42 s |

Also at the end of the night: mode switching 6/6, accents (voice-library Indian English / Canadian voices) 9/9,
past the 5-minute idle timeout 6/6 (default 6/6), Avatar 8/8, three rooms on one server 25/25 with 0 dropped frames.
MSI-Bench through the call (`tests/msi_meeting.py`, 24 cases, rendered speech with distance, reverb, ambience, a
bystander's voice): cues answered 10/18 → 21/24 and 22/24 in two final runs, premature replies 0 and 2/24, median
1.0 s after the cue. The B2 browser meeting: 5/6 speak, 25/26 silent.

- Backchannels while Higgs talks: improved kept talking 7/7; default stopped 5/7 and then answered the "mm-hm" as a
  new turn ~1.2 s later. Whether a backchannel stops default Higgs depends on the sound: clear "Mm-hm.", "Uh-huh.",
  "Yeah." and 2 real AMI ones stopped it every time; 3 other real AMI ones (soft hums) never did. Through a real
  Chrome mic path (`tests/browser_bc.py`): default 5/8 and 7/8 (4/8 at −10 dB), improved 0/8 with all 8 answers
  delivered.
- Real interruptions: improved stopped every time at normal level (~0.5–0.9 s); quiet real AMI ones at −6 dB (about
  −31 dBFS, as from a mic without auto gain) stop it most of the time (one miss in the last run): the echo gate treats
  sound 25 dB below what Higgs plays as its own voice coming back.
- Past Higgs's 5-minute idle timeout (`--set 4`): improved 6/6, default 6/6 (was 2/6: the session never came back).
- Mode switch mid-call (`--set 5`): answers after both switches, and the history carries over.
- With the Avatar (`--set 3`): 8/8; the face says "Hmm." ~1.5 s after the line, the answer starts ~4.0 s after it
  (a long list of numbers: ~8 s). B2 in 3 Chromium pages with the Avatar: 5/6 + 24/25, and every page played all 10
  face streams ("Hmm." + answer for 5 replies), no player errors.
- Real meetings (`tests/ami_meeting.py`, AMI, 4 real voices on 4 bots, nobody talks to the AI): ES2009c 10 min,
  142 lines, Higgs spoke once (unnecessary; that kind of line is now silent, 146 lines / 0 on the final code);
  ES2014c 10 min, 81 lines, never; ES2009c minutes 10–30, 342 lines, never (6 budget corrections it would have made,
  e.g. a "17.50 euro" estimate against the 12.50 cap, found no gap in the talk and were dropped). With a fifth person
  asking the brief's questions in the lulls (`--ask 5`, ES2014c minutes 10–25): the 3 put to Higgs by name answered
  correctly, 1.4–5.9 s after the question; the unaddressed group question and wrong fact lost to the chatter.

Env: `MEET_SAMPLES` (judge samples per line; default 1 in a one-to-one call, 3 with more people), `MEET_BUDGET`
(Higgs requests per server run, default 30000; the room is told when it runs out), `MEET_PROMPT` (judge, default
`v5`), `MEET_LINE_SILENCE` (0.30), `MEET_ECHO_DB` (25), `MEET_SPECULATE` (1), `MEET_LANG` (en), `MEET_SPARE_SIGNALS`
(2 preloaded turn-taking model processes), `MEET_SIGNALS=0` (no per-person turn-taking models: timing falls back to
silence rules), `MEET_VOICE`, `MEET_STT_FALLBACK=0` (no local transcription when Higgs STT fails: on 2026-10-23 it answered most lines with "the transcription is not available right now" after ~3 s, so faster-whisper base.en transcribes a line Higgs hasn't within `MEET_STT_WAIT` = 1.0 s; logged as `stt_fallback`).

## Running it for real (2026-10-23 notes)

- **Restart the server** after pulling changes (`./run.sh` / `./run.sh public`): a running server keeps the
  code it started with.
- **Machine:** each person in an improved-mode room has a turn-taking model process (~0.7 CPU core, ~620 MB RAM);
  the server keeps 2 preloaded (`MEET_SPARE_SIGNALS`). Three 3-person rooms on one server ran at 2–3 ms per room tick.
- **Higgs sessions:** an improved room holds ~4 standing sessions (one-to-one) or ~9 (group) plus one per person;
  the API key took 120 concurrent sessions in a test without refusal. Every request is counted against
  `MEET_BUDGET` (30000 per server run; the room is told when it runs out).
- **Echo:** headphones are best. Without them, Higgs ignores its own voice coming back up to 25 dB below what it
  plays (a typical echo canceller leaves less than that).
- **Showing default vs improved live:** default Higgs is reliably derailed by a clear "yeah" or "uh-huh" while it
  talks (it stops and answers the backchannel), by a mid-sentence thinking pause (it talks over it), and by people
  talking to each other (it answers them). A soft hummed "mm" often doesn't trigger its voice detector at all.
- **Avatar:** with LiveKit set, the live face (above): replies come ~1.3–1.7 s later than without the face, and
  pages need to reach `*.livekit.cloud` (WebRTC). Rendered face: in improved the face says "Hmm." ~1.5–2.4 s after the
  line and the answer follows ~2–3 s later (it waits for a quiet moment if someone talks into that pause); in default
  the reply just comes ~3 s later than without the face.

## Rooms

`/` creates a room (title, AI name, setting, facts, optional passcode) and gives a link `/r/<id>?k=<passcode>`; the
server also prints it. Rooms live in memory; each room logs to `out/rooms/<id>.jsonl`. `/r/<id>?lab` shows the
agent's turn-taking values on the tiles.

**Meeting scripts** (`web/agendas.js`): the create page offers 9 presets (hackathon planning, the default-vs-improved
video beats, a house meeting, a launch go/no-go with an AI named Nova, a fast standup with Max, trip planning and a mock
interview one-to-one, a study group, edge cases). A preset fills the form (facts + the agenda) and gives the call a
script: who says what, pauses, lines said over the AI, typed lines, and what the AI should do (answer, stay quiet,
correct, keep talking, stop) with what a right answer contains. The AI never sees the script. The join screen offers
the cast's names; the cursor is shared (click a line, ↑/↓ or j/k) and each move is logged as a `script` event, so a
run's lines and decisions can be matched to script steps.

## Wire format

- `/ws/room?room=&pid=&name=&k=`: JSON control + binary audio. Client → server: 960 bytes = 20 ms mono int16
  at 24 kHz. Server → client: 1 byte slot + 960 bytes (slot 0 = Higgs, 1..5 = people), forwarded at once to everyone
  else. Close codes: 4001 room full, 4003 wrong passcode, 4004 no such room, 4000 replaced by a newer tab.
- `/ws/video?room=&pid=` (open after joining): client → server JPEG (320x240, ~8 fps, skipped while
  `bufferedAmount` > 64 KB); server → client 1 byte slot + JPEG, newest frame per sender wins. Slot 0 = Higgs avatar:
  fMP4 chunks for a MediaSource, between `avatar_start {mime}` and `avatar_stop`; `avatar_idle {url}` loops a file.
- Server → client JSON: `roster`, `chat`, `ai_flush`, `ai_gain`, `ai_state` (+ `by` when a person changed it),
  `line`, `ai_say`, `lab`, `live_avatar` (`{on: true, url, token, room, identity, drain}`: join this LiveKit room
  subscribe-only and play `identity`'s tracks as Higgs; `{on: false}`: leave it). Client → server JSON: `chat`, `set_mode`, `set_avatar`, plus `mic`/`cam` `{on}` (roster
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
    python tests/live_avatar_browser.py   # the live face in 2 Chromium pages, stub agent (a real avatar session, ~1 min)
    python tests/live_avatar_probe.py     # the avatar alone: session start, lag, dropouts, clear(), fps
    python -m pytest tests/test_floor.py tests/test_agent_rules.py   # timing + text rules, no API (./run.sh tests)

With Higgs (each makes real API calls; keep to ~3 servers at once, Higgs caps concurrent sessions per key):

    python tests/scenarios.py --set all --modes improved default --par 3   # 37 interactive scenes, bots over /ws/room
    python tests/scenarios.py --set 3     # with the Avatar     --set 4: past Higgs's 5-minute idle timeout
    python tests/live_avatar_e2e.py       # live face: when it is heard after a question, and an interruption
    python tests/judge_cases.py [--file judge_heldout.jsonl]   # the speak/silent decision on 63 / 120 written lines
    python tests/tv_regress.py --set test --prompts v4b v5     # Third Voice's group test set through agent.decide
    python tests/writer_cases.py          # what the reply writer says, two prompt versions side by side
    python tests/browser_bc.py --mode default                  # backchannels through a real Chrome mic path
    python tests/ami_meeting.py --meeting ES2009c --minutes 10 # a real AMI meeting, 4 people on 4 bots (AMI_DIR)
    python tests/msi_meeting.py --n 24                         # MSI-Bench cases through the call (MSI_DIR)

`scenarios.py` scores each step from what a participant hears (Higgs's slot-0 audio) and the room's events, and
prints where a failure came from (what STT heard, the judge's reading, a floor drop, `reply_skipped`).

It checks roster, audio and video between all pages, chat, mode switch, the stub's slot-0 beep, ai_gain/ai_flush,
transcript lines, captions, lab bars, avatar fMP4 playback, reload/leave/slot reuse, the 5-person cap, passcode,
admin token, bounded queues, and saves `tests/shot_room.png`.

### Test data (`tests/data/`)

- `testset/`: Third Voice's hand-written meetings (`meetings_B/C.jsonl`) and its speak/silent points (`points_dev`,
  `points_dev2`: used to design the prompts; `points_test`: held out), for `tv_regress.py` and `make_tracks.py`.
- `tts/B2/`: meeting B2's lines rendered by Higgs TTS (one preset voice per person), for `make_tracks.py`.
- `ami_cues/`: 8 real listener backchannels and interruptions cut from the AMI Meeting Corpus (CC BY 4.0), for
  `scenarios.py` and `browser_bc.py`. `mic/`: fake-microphone speech for `fake_room.py`.

Not included (large, external): the AMI corpus for `ami_meeting.py` (`AMI_DIR`, default `data/AMI`: `amicorpus/`
headset audio + `ami_public_manual_1.5/`) and MSI-Bench for `msi_meeting.py` (`MSI_DIR`, default `data/msi_bench`).
