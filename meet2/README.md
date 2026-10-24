# Meet 2.0

A browser meeting for up to 5 people with an AI participant built on Boson AI's Higgs models. The AI listens to
everyone under their own name, speaks when it should, keeps talking through "umm" and "okay", stops for a real
interruption, and cuts in at once when someone states something that contradicts the meeting's documents or what
someone else said.

## Quick start

    make install            # .venv + core packages; creates .env from .env.example
    # put your BOSON_API_KEY in .env
    make doctor             # checks Python, packages, the key (one cheap call each), LiveKit
    make run                # http://localhost:8600

Create a meeting (or start from a preset), add documents to its knowledge base, copy the invite link, join.

Optional live avatar: `make install-avatar`, then set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` in
`.env` (LiveKit Cloud, or `livekit-server --dev` locally). Docker: `docker compose up --build`.

Other ways to run: `make stub` (the UI with a stand-in AI, no API calls), `make run-lan` (https for other devices on
your network), `make public` (a public link through cloudflared; room creation then needs `MEET_ADMIN`).

## What it does

| | Feature | How |
|---|---|---|
| F1 | **Voice only by default; avatar optional** | Rooms start voice only (an animated voice tile and captions). The host can turn on Boson's live avatar mid-call; only then does the server use LiveKit, for the avatar's synchronized audio and video. Turning it off returns to voice only without anyone leaving. |
| F2 | **Fillers never stop it** | While the AI talks, a listener's speech is transcribed every ~0.3 s on a separate session and classified: fillers and acknowledgements ("umm", "ahh", "okay", "yeah", "mm-hm", "I see", any mix) change nothing, not even the volume; "wait", "stop", "hold on" stop it at once; two or more content words are a real interruption. Lists and thresholds live in `config/fillers.en.yaml`. |
| F3 | **Follows a group conversation live** | Every person has their own transcription stream, so names are exact. Speech is cut at clause pauses, so each clause has words ~0.3 s after it ends. A notes panel keeps the topic, decisions, action items and open questions. A judge (Meet 1's tuned prompts) decides whether the AI should answer at all; it stays out of people talking to each other. |
| F4 | **Interrupts contradictions** | Each claim-like clause is checked, beside the conversation, against the fact sheet, the documents and what people said earlier. A confirmed contradiction (cited evidence that really exists, confidence ≥ 0.8, not already corrected, cooldown) takes the floor at the speaker's next pause, ducks everyone's playback of the other humans to 15%, and holds whatever is said meanwhile. For its first 2 s (`MEET_CORRECTION_GUARD`) nothing but the host's "stop" can cut it; after that, someone who keeps interrupting for 0.6 s more gets the floor, while a line that merely runs past the 2 s doesn't. When the correction has been heard, the volume returns and the held lines are handled in order, marked as said during the correction, so the AI answers them ("Yes, it's definitely the 1st…"). |
| F5 | **Knowledge base** | Upload PDF, DOCX, MD, TXT, CSV or HTML, add a link, or paste text, before or during the meeting. Documents are chunked and indexed (SQLite FTS5); key facts are extracted into a fact sheet the host can edit. Answers and corrections name their source. |
| F6 | **Background noise** | The browser's noise suppression, then a per-person front-end: speech must stand out from that person's own noise spectrum, be in the speech band, not be spectrally flat, and last 100 ms. Fans, hum, hiss and keyboard clicks never start a line; once someone's voice level is known, chatter far below it is ignored; the AI's own voice leaking back is ignored. Push-to-talk for very noisy places. |

Also: presets, passcodes, host key, text chat (the AI reads and answers it), camera tiles, a Smart/Plain switch
(Plain answers everything and stops at any sound, for demos), per-room JSONL logs with stage timings.

## Measured (2026-10-23/24, real Boson, `make live`)

| Scenario | Result |
|---|---|
| One-to-one question, voice only | answered with the right facts; first audio **1.6–1.8 s** after the question ends |
| Four fillers while the AI talks | **0** stops, playback continued |
| A real interruption | stopped **0.88–0.91 s** after the person started; then answered the interruption |
| Two people talking to each other | the AI stayed silent; every line under the right name |
| "Rent is due on the fifteenth" (fact sheet: the 1st) | corrected, ducked to 0.15, "are you sure?" said over it was held, then answered with the evidence |
| White noise, fan rumble, keyboard clicks | no lines, no replies |
| Question answered from an uploaded lease | right answer, source shown ("Lease p.1") |
| Live avatar | avatar audio + video over LiveKit; first avatar audio 3.4 s; switching back to voice only works |

## How it is built

    browser --WebSocket (control + people's audio/video)--> server
      Listener per person: front-end (noise, VAD, echo) -> own Higgs STT stream -> pieces (clauses) and turns
        pieces -> Monitor (fact check)       -> Floor.correct
        turns  -> prefilter -> judge + writer -> Floor.propose
      Floor: the one owner of speech (idle / waiting / speaking / correcting)
      OutputWorker: supervised, sentence-by-sentence speech, never rewinds, "heard" != "sent"
        -> WebSocket (voice only) or the Boson avatar via LiveKit

| Path | What |
|---|---|
| `meet/server.py`, `meet/rooms.py` | HTTP routes, the room socket, rooms and participants |
| `meet/agent.py` | the AI participant: wiring, replies, corrections, buffering |
| `meet/floor.py` | when the AI may start, keep and give up the floor |
| `meet/speech.py` | utterances, sentence synthesis, the output worker |
| `meet/listen/` | per-person listener (STT, turns, barge-in probe) and word rules (fillers, prefilter) |
| `meet/audio/frontend.py` | noise floor, voice activity, echo guard |
| `meet/brain.py` | judge and writer prompts |
| `meet/monitor.py` | contradiction checks with evidence verification |
| `meet/kb/` | parsers, index, fact sheet |
| `meet/transport/` | the media interface; WebSocket and LiveKit-avatar implementations |
| `meet/boson.py` | the only module that talks to Boson |
| `web/` | home + setup page, call page, worklets (no build step) |
| `config/` | presets, filler lists |
| `tests/` | unit and integration tests (fake Boson); `tests/live/` real scenarios |

## Tests

    make test           # 124 tests, no API calls, ~35 s (a scripted fake Boson drives the whole agent)
    make live           # 6 scenarios through the real server with bot participants (real API calls)
    make live-avatar    # the avatar scenario (needs LiveKit)

## Decisions made while building (vs. the plan)

* **Boson's server-side noise reduction is off.** Measured: it turned "Hey Higgs, when is the rent due" into "What do
  we do?" at every chunk size, and lost words in 5 dB white noise where plain STT was word-perfect.
  `MEET_DENOISE=boson` turns it back on.
* **No embeddings.** This Boson key has no embedding model, so retrieval is SQLite FTS5 (BM25) plus the fact sheet.
* **Text pipeline with speech synthesis, not speech-to-speech.** Chat completions answer in ~0.25 s and TTS starts in
  ~0.5 s; owning each step gives exact sentence boundaries for "[interrupted]" and full control of barge-in.
* **LiveKit carries only the avatar.** People's media stays on the WebSocket in every mode, so turning the avatar on
  or off never moves anyone's audio mid-call.
* **No local turn-taking model (MaAI) in v1.** Turns end on 0.45 s of silence; mid-sentence stops ("having a?") are
  joined back; an unfinished turn waits 1.5 s for the speaker to go on.

## Known limits

* Rooms live in memory: a restart ends meetings (knowledge bases are on disk and survive).
* Chatter as loud as the speaker can't be told apart from them by sound; a headset helps.
* The fact sheet and documents are the only external evidence: no web search.
* English filler lists only (add `config/fillers.<lang>.yaml` and set `MEET_FILLERS_FILE`).
