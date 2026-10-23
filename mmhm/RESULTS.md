# Mm-hm: results (2026-10-21, live against the Higgs API)

Simulated users are Higgs TTS 3 preset voices streamed in real time into the same engine that runs plain Higgs
and Mm-hm. **Thresholds were tuned on one stimulus set; every number below is from a separate held-out set**
(different sentences, cue words and voices: oliver/chloe/jake/nora for turns, oliver/chloe for barge-in).
Tuning-set numbers are in brackets for reference. Logs, session recordings and per-item rows are in `out/`.

## 1. Thinking pauses: does it answer half a sentence?

24 turns per condition: 12 complete turns, and 12 "hold" turns, i.e. an unfinished clause, a pause, then the rest
("I've been learning to play the … cello for about two years now"). A cut-in means agent audio starts after the
unfinished clause and before the user has finished.

| | plain Higgs `server_vad` | plain Higgs `semantic_vad` | **Higgs + Mm-hm** |
|---|---|---|---|
| Cut-ins, 1.4 s pauses | 12/12 | 12/12 | **4/12** [3/12]; with the STT syntax hold **1/12** [0/12] |
| Cut-ins, 0.8 s pauses | 3/12 | 2/12 | **0/12** [0/12] |
| Median gap after a finished turn (1.4 / 0.8 runs) | 1170 / 1080 ms | 820 / 770 ms | **780 / 800 ms** |
| p90 gap | 1240 / 1160 ms | 876 / 1116 ms | 1102 / 1154 ms |

- Over both pause lengths: **4/24 cut-ins vs 14/24 (semantic_vad) and 15/24 (server_vad)**, at semantic_vad's
  speed and ~300 ms faster than the default server_vad.
- What a cut-in sounds like: plain Higgs finishes your sentence for you, wrongly. "I've been learning to play
  the…" → "guitar?" (it was the cello). "When I was a kid I always wanted to…" → "be an astronaut." (build
  robots). "Before this job I spent three years…" → "working in retail, which was a wild ride." (teaching math).
  Mm-hm waits for all three. A/B audio: `../demo_clips/pause_*`.
- Correction to the 2026-10-12 probe (RESULTS.md, t3): with 0.8 s pauses plain Higgs *generates* a reply to the
  half-sentence every time, but it is usually cancelled before it's heard (2–3/12 audible). Long thinking
  pauses (1.4 s) are where it audibly talks over people: 12/12.
- Why it works: MaAI's VAP model separates "done" from "mid-thought" at 100 ms of silence (offline, 24 TTS
  stimuli: AUROC 0.97; in a live dialogue p(user holds the floor) is 0.08 median after finished turns vs 0.47
  inside unfinished ones). The engine starts Higgs generating speculatively at p < 0.30 and only lets it be
  heard at p < 0.12 (or after 1.6 s of silence). Speculative replies that turn out wrong are cancelled and
  deleted from the conversation (7 of 7 during a 30 s story, `eval/listen_demo.py`).
- Remaining failures (VAP only): 4 of 12 hold turns at 1.4 s, where VAP's confidence decays below the play
  threshold during a long pause.
- **Adding a syntactic hold from Higgs's own STT fixes most of them.** After a speculative commit, the input
  transcription of the words so far arrives ~0.3 s later; if it ends on a word that can't end a sentence ("…play
  the", "…moved to", "um,"), Mm-hm keeps the reply back until the user resumes or 1.6 s pass. Cut-ins at 1.4 s
  pauses: **tuning 3/12 → 0/12, held-out 4/12 → 1/12**, median gap unchanged (740 / 780 ms). The one held-out
  miss ends in "you" ("make sure you…"), which is deliberately not on the list ("thank you" is complete). One
  tuning-set false hold ("Which team are you on?" → 1.6 s) led to the rule that a transcript ending in "?" is
  complete. Word list: `mmhm/engine.py` `INCOMPLETE`.

### 1b. Real human speech (AMI meeting headsets)

The TTS stimuli are tidy. Real hesitations are not, so we cut real mid-sentence thinking pauses (0.8–1.6 s, no
sentence-final punctuation, nobody else speaking, same speaker continues) and real finished turns (someone else
takes over) from AMI headset recordings (`eval/ami_stimuli.py`, CC BY 4.0), with other speakers' bleed muted.
Set 1 (6 meetings) was used to extend the word list; set 2 (6 *different* meetings) was not touched.

| | `server_vad` | `semantic_vad` | **Mm-hm** |
|---|---|---|---|
| Cut-ins on real thinking pauses, set 2 (untouched) | 12/12 | 12/12 | **6/12** |
| Median / p90 gap after real finished turns, set 2 | 1140 / 1200 ms | 720 / 1160 ms | 840 / 1510 ms |
| Cut-ins, set 1 | 12/12 | 12/12 | 4/12 |

- Real speech is harder than TTS: Mm-hm halves the cut-ins rather than nearly eliminating them, and its p90
  gap is longer. At real thinking pauses VAP is often confidently wrong (p_now 0.05–0.13, as low as after a
  finished turn), and the words before the pause look like fragments to a person but not to a word list
  ("take over the entire", "so on the back", "with one-sided").
- Why: offline, MaAI's VAP separates real thinking pauses from real turn ends with AUROC ≈ 0.80 at every
  silence length (0.97 on the TTS stimuli). The other English variants (10 Hz, noise-robust, Kyoto) are no better
  (0.76–0.80; `eval/vap_variants.py`). Closing this gap means fine-tuning the turn-taking model on real
  conversation, not tuning thresholds.
- Tried: a turn-end classifier trained on real conversation (`eval/turnend_features.py`, `eval/turnend_train.py`):
  5,449 pauses from 24 AMI meetings in series never used for testing, tested on 2,580 pauses from the 12 test
  meetings, decided 200 ms into the silence. AUROC: VAP p_now alone 0.73 → VAP's four outputs 0.76 → plus prosody
  (final lengthening, energy drop, pitch slope) and the last word 0.79 (0.91 on mid-sentence holds). At ~10% of
  finished turns wrongly held, it catches 51% of real holds, about what the live engine already does (6/12), so
  it was **not** wired in. Most of the gain is VAP's own trend (p_now falling vs rising, p_future).
- Tried live and **not** adopted: a small logistic "hold" model on VAP's trend (p_now and p_future 100 ms apart,
  trained on the 24 training meetings; offline it catches 45% of real holds at 10% of finished turns held, vs
  32% for p_now alone). On real-speech set 2: still 6/12 cut-ins, and the median gap grew from 840 to 970 ms.
  It stays in the code as `Policy(hold_model=True)`, off by default.
- Tried and **not** adopted: asking Higgs (text mode, 92 ms) whether the transcript so far is a finished thought.
  It catches 20–22 of 24 real holds but also calls 9–13 of 24 finished turns unfinished, which would delay those
  replies to ~2.2 s. (`eval/completeness_offline.py`.)

### 1c. A noisy room (babble at 15 dB SNR, held-out TTS stimuli, 1.4 s pauses)

| | `server_vad` | Mm-hm, first version | **Mm-hm, final** |
|---|---|---|---|
| Cut-ins | 12/12 | 1/12 | **1–3/12** (two runs) |
| Median gap | 1180 ms | 2240 ms | **760 ms** |

The first version's energy VAD only updated its noise floor on frames it judged silent, which babble never is,
and the turn-taking model, hearing voices in the user's channel, kept believing the user was still talking.
Fixes: a minimum-statistics noise floor (10th percentile of the last 3 s) and a noise gate on the model's input
between the user's words. Clean held-out results after both changes: 0/12 cut-ins, 770 ms.

## 2. Backchannels: does "mm-hm" kill the story?

Higgs tells a story; 2.5 s into playback the listener makes a sound: 8 backchannels × 2 voices, or 6 real
interruptions.

| | plain Higgs (`server_vad`, client stops on `speech_started` as documented) | **Higgs + Mm-hm** |
|---|---|---|
| Backchannel stopped the story | 10/16 | **3/16** final engine (5/16 in an earlier run of the same cues) [1/16] |
| Real interruption stopped Higgs | 6/6 | 6/6 [6/6] |
| Median time to stop on a real interruption | 290 ms | 420 ms (volume ducks at once: median 40 ms) |

- When plain Higgs stops on a backchannel and the client truncates, Higgs does pick the thread back up
  ("…and had to spend the night in a hotel in the wrong city"), but only after **1.0–2.5 s of dead air**
  (median ~1.4 s). Without truncation (Oct 12 probe) it answered the "mm-hm" as a new turn, 5/5.
- Mm-hm's detector: MaAI `bc_det_en` scores backchannels 0.8–0.94 within 80–160 ms and real interruptions
  ≤ 0.15; VAP p(user takes the floor) is 0.79–0.94 at the decision point for real interruptions vs 0.04–0.06
  for sounds we let pass.
- Held-out failures: both chuckles (laughter isn't a backchannel class in `bc_det_en`), and "nice", "sure",
  "yeah, yeah" in one voice. Two interruptions that *start* like backchannels ("Okay, okay, let me stop you
  there", "Excuse me, quick question") were caught late (1.0–1.5 s).
- The cost: real interruptions stop ~130 ms later than plain Higgs, because Mm-hm listens for ~400 ms before
  deciding. The volume drops immediately, so the agent audibly yields at once (like a person who trails off).

### 2b. Real human sounds (AMI headsets, offline replay of the engine's rule with the real models)

From the 12 test meetings: 410 real listener backchannels ("yeah", "mm", "mm-hmm", "okay", "right", … while
someone else talks, and the speaker carries on) and 370 real interruptions (someone starts a turn over a speaker,
who stops within 1.5 s). `eval/bargein_ami.py`.

| | real backchannels | real interruptions |
|---|---|---|
| Mm-hm keeps talking | **396/410 (97%)** (300 recognized as backchannels, 96 ignored as short noises) | 3/370 |
| Mm-hm stops | 14/410 (3%) | **367/370 (99%)** |

Interruptions that start like a backchannel ("yeah, I think…") are still caught, at the 1 s cap.

**Live, with real cues** (40 real backchannels and 20 real interruptions cut from those meetings, played 2.5 s into
Higgs's story; `eval/ami_cues.py`, `eval/bargein.py --ami`):

| | plain Higgs (`server_vad`) | **Mm-hm** |
|---|---|---|
| real backchannel stopped the story | 14/40 | **0/40** |
| real interruption stopped Higgs | 20/20 (median 360 ms) | 17/20 → **19/20** with the fragment rule (median 500 ms; volume ducks at 80 ms) |

The first version missed three hesitant interruptions that start like a backchannel and pause ("yeah, I think
we, um, it…", "sorry, we'll get, um, warnings for…"): each fragment was too short to count as taking the floor.
Final rule: user speech that adds up to 1 s across fragments (reset after 1 s of quiet) is an interruption. Re-run:
19/20 caught, still 0/40 backchannels stopped. Real
backchannels also stop plain Higgs less often than TTS ones (35% vs 62%): many real "mm"s are quiet.

## 3. Listening noises

**On real conversation (AMI, offline replay of the trigger with the real model, `eval/listen_ami.py`).** In
monologue stretches (≥ 8 s, others only backchannelling), a trigger counts as a hit if it lands within 0.6 s of a
real listener's backchannel. Rules were compared on 24 training meetings (560 min) and the chosen one confirmed on
the 12 test meetings (227 min; real listeners: 4.6 backchannels/min; random timing would score precision 0.09):

| rule | rate | precision | recall |
|---|---|---|---|
| pause-only ("nod at every short pause") | 3.8/min | 0.09 (chance) | 0.09 |
| first version (model, only in 80–350 ms pauses) | 1.1/min | 0.13 | 0.04 |
| **final (model, any time, p_bc > 0.6)** | **5.4/min** | **0.17** | **0.24** |

Real listeners mostly say "mm-hm" *over* the speaker, not in pauses; letting the model fire any time gives a
human-like rate at 1.9× chance precision. (On the tidy TTS story below it fires less, ~1 per 30 s: TTS speech
rarely pushes the model above 0.6.)

**On a TTS story (first version):**
During a 33 s story told in 7 sentences, Mm-hm said "mm-hm" twice, both in mid-sentence phrase-boundary pauses
("…four hours, [mm-hm] and the last part was…"), never on top of words, then answered after the story
("Oh no, that's the worst! Did you have to call someone to let you in?"). `../demo_clips/listening_noises_story.wav`.
Two in 33 s is on the low side for a human listener; the trigger threshold is a dial.

## 4. Systems numbers

- CPU cost per 20 ms tick: VAP 3.2 ms, bc_det 10.7 ms, bc 6.6 ms, each in its own thread; 0 late ticks in all runs.
- Higgs commit → first audio: ~0.6–0.7 s; with a 100 ms silence trigger, the end-to-end gap is ~0.78 s.
- Opening ~6+ sessions at once with a preset voice: HTTP 429 from voice validation, session terminated, and the
  limit lingers for tens of seconds. Serializing opens (`mmhm/higgs.py`) fixes it.

## 5. What these numbers are not

- Real human speech (1b) is the most realistic number here: 6/12 vs 12/12, not 1/12.
- Not human judgements. The Turing Clock (`app/`) and its party mode exist to collect those: time until a person
  catches the AI, plain vs Mm-hm, and how often people judge each (and a real human) to be human.
- Small n (24 turns and 22 barge-in trials per condition). Treat differences like 12/12 vs 4/12 as real and
  1–2 trial differences as noise.
