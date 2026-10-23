# Third Voice: when should an AI teammate speak in a group?

Test bed: a hand-written 20-turn hackathon team meeting (Ana, Ben, Chen) with an AI teammate called "Higgs",
annotated with when the AI should speak (`meeting.py`): 5 SPEAK points (2 where Higgs is addressed by name,
3 group questions nobody answers that Higgs knows the answer to) and 15 SILENT points (people talking to each
other, trailing off, small talk).

## Status (2026-10-22, after the night session)

- **Fixed:** a decomposed Higgs judge (describe the utterance as JSON, then a fixed rule) + a fact-check scores
  **0.92** balanced accuracy on 255 held-out decisions from independently written meetings (heuristic 0.73, trained
  LoRAs ≤ 0.70, asking Higgs directly 0.56); 0.89 on TTS → Higgs STT transcripts; no speaker identification needed.
- **Real meetings:** on 60 min of real AMI meetings (manual transcripts, and the far-field room mic through Higgs
  STT) it speaks 0.12–0.15 times a minute, mostly about the project brief it was given.
- **Says the right thing:** when it speaks to correct someone, the reply names the mistake (21/21 corrections, was
  12/21); with a big fact sheet (~47 facts) per-line retrieval keeps accuracy at 0.90.
- **Live:** `./run.sh tv` (one shared mic in the browser); the AI starts talking 1.0–1.6 s after the line ends.
- **MSI-Bench (Boson AI's multi-speaker benchmark, 2026-10-22, `msi_eval.py`):** on the paper's premature-response
  probes (dialogue cut before anyone addresses the assistant) Third Voice answers 3/192 (text) and 3/96 (Higgs STT of
  background-speech scenes, no names); Higgs asked "SPEAK or SILENT?" answers 56% / 77%, the paper's hosted models
  21.5-97.9%. The shipped v4 missed commands ("Assistant, put that through": 410/579 named lines answered); v4b
  (`DEC_SYS4`: "ask" includes instructions, designed after seeing those errors) answers 572/576 cues, balanced accuracy
  0.925 (text) / 0.942 (speech) vs 0.797 / 0.895, and 0.897 vs 0.917 on our own test (within noise). A keyword rule
  ("assistant") ties it on text (+0.002 [-0.015, 0.020]); 87% of addressed lines there say "assistant".
- **Still missing:** a recorded, labelled real meeting with an AI in it; try it with three people at the venue.

## The fix (night of 2026-10-21/22): ask Higgs five small questions, not "should I speak?"

Asked "SPEAK or SILENT?", every LLM we tried says SPEAK nearly always. Asked instead to *describe* the latest
utterance as JSON (who it's aimed at: me / a person / the group / thinking aloud; does it ask for something; was
it already answered; can I answer it from my facts or general knowledge; does it contradict my facts), Higgs
Realtime (text mode) is accurate, and a fixed rule turns the description into the decision:
speak if it contradicts the facts, or it's aimed at me with a request, or it's a request to the group that nobody
answered and I can answer. Three samples, majority vote; ~0.35 s per decision (`tv_eval.py`, `DEC_SYS2`).

**New test set, written independently** (`testset/`): three writer agents that never saw any policy code wrote
12 meetings (380 utterances, 66 SPEAK) in 12 different settings with an AI participant and its fact sheet, labelled
per a written guideline (`testset/GUIDELINE.md`: addressed, group question it can answer, correction → SPEAK;
aimed at a person, mention, unfinished, question only humans can answer, already answered, chat → SILENT). A
blind second annotator agreed on 379/380 (κ = 0.99; the writers were told to rewrite ambiguous lines, and both are
the same model family, so this says the labels are clear, not that the task is easy). Split fixed before any
tuning: meetings A1–A4 = dev (prompt design), B1–B4 + C1–C4 = held-out test.

| policy (balanced accuracy) | old 20-turn meeting | dev (125) | **test (255)**, 95% CI over meetings | test, speech → Higgs STT |
|---|---|---|---|---|
| heuristic: "?" or the AI's name | 0.87 | 0.76 | 0.73 [0.68, 0.79] | 0.74 |
| heuristic2: name, or "?" not naming a person | 0.97 | 0.80 | 0.78 [0.72, 0.83] | 0.77 |
| zero-shot Higgs, SPEAK/SILENT | 0.63 | 0.56 | 0.56 | – |
| LoRA, AMI meetings | 0.63 | 0.51 | 0.44 | – |
| LoRA, synthetic meetings | 0.63 | 0.65 | 0.70 | – |
| **decomposed Higgs, v2, 3-vote** | 0.90 | 0.96 | **0.90 [0.87, 0.93]** | **0.86 [0.79, 0.93]** |
| **v4 = v2 + fact-check** (3 samples, unanimous; designed on dev + dev2, test is a second look) | 0.90 | 0.95 | **0.92 [0.88, 0.96]** | **0.89 [0.83, 0.95]** |

- Paired over meetings, decomposed − heuristic: **+0.15 [+0.09, +0.22]** on text, **+0.11 [+0.04, +0.19]** on
  speech; vs heuristic2: +0.11 [+0.05, +0.16] and +0.08 [+0.02, +0.14] (`paired.py`).
- On test it stays quiet at 205/210 silent points (heuristic: 145/210) and speaks at 37/45 (heuristic 35/45).
  By category: addressed 12/12, aimed at a person 29/30, mention 8/8, unfinished 10/10, questions only humans can
  answer 17/17; weakest: **corrections 5/9** and group questions it can answer 20/24.
- Speech round trip (`stt_meetings.py`): every line spoken by TTS 3 (preset voices), transcribed by
  higgs-stt-3.1. The transcripts are nearly verbatim (punctuation, "Tobi" → "Toby"), so most of the drop from 0.90
  to 0.86 is the judge's own sampling noise (single samples range 0.89–0.98 on dev), not recognition errors.
- v1 → v2 (designed on dev only): accept the AI's name for "me"; being talked to without a request ("okay,
  thanks") needs no reply; "can answer" means a *fact*, not a decision or preference that is up to the humans.
- **Corrections (v4).** v2 caught 5/9 corrections on test. On dev the extra fact-check call looked useless (3/4
  either way), so a second dev set was written by another independent agent, rich in corrections and near-misses
  (`testset/meetings_D.jsonl`, "dev2": 128 utterances, 12 corrections, 8 near-misses). It showed the problem:
  implicit contradictions ("peanut butter blossoms" vs a nut ban, "a couple of Advil" vs an ibuprofen-on-warfarin
  rule) pass the one-shot judge (v2: 7/12) but not a dedicated check (11/12). v4 = v2 OR all three fact-check
  samples agree: dev2 0.914 → 0.974, dev 0.957 → 0.947; on test (second look) 0.90 → 0.92, corrections 5/9 → 6/9;
  on speech 0.86 → 0.89, corrections 3/9 → 5/9. Asking the judge itself to cite the contradicted fact (v3) was
  worse (dev2 0.877). v4 runs 6 parallel requests per line; the live app uses it.
- **Cheaper is almost as good** (recomputed from cached answers on test, text / speech without names): 1 judge
  sample 0.888 / 0.875; **1 judge + 1 fact-check (2 requests) 0.914 / 0.898**; 3 judges 0.886 / 0.890; v4 (3 + 3)
  0.917 / 0.908. The fact-check is what helps; voting adds little. **Replication:** v4 re-run on test with three
  fresh samples of every call: 0.914 [0.86, 0.97] (first run 0.917). (A fresh set of 3 judge samples scored 0.886
  where the first set scored 0.899: run-to-run noise is about ±0.01.)

**Live, end to end on Higgs** (`live_demo.py` → `out/demo/<meeting>.wav/.json`, `demo_clips/third_voice_live_*.mp3`):
meeting audio streamed at speaking pace into a transcription-only Realtime session; after each line v4 decides from
the STT transcript while Higgs (text) drafts the reply speculatively; on SPEAK, TTS 3 speaks it (pre-opened sessions
keep the handshakes off the critical path). Decisions match the annotation 81/83: old meeting 19/20, held-out B3
(theatre group, AI "Higgs") 31/31, C2 (restaurant) 31/32; every reply in B3 and C2 is correct. The AI starts
talking **1.2–1.5 s (median) after the line ends**.

**Saying the right thing, not just speaking at the right time.** The reply is drafted in parallel with the decision,
so the writer doesn't know *why* it speaks; on corrections it often confirmed the mistake or talked about something
else ("I'll send a reminder about the fittings for Sunday", when they are on Saturday). One sentence in the writer's
instructions ("if the latest utterance contradicts what you know, start by correcting it plainly") fixed it: on the
21 correction points of test + dev2, correctly worded corrections went from 12/21 to 21/21 (one with a persona slip:
the AI spoke as if it were one of the roommates); 16 sampled non-correction replies stayed correct, none
"corrected" anything spuriously (`out/reply_corrections.json`).

**No diarisation needed.** With every human line labelled "Someone" (one shared mic, nobody identified), the
judge scores 0.89 [0.84, 0.94] on the held-out speech transcripts (0.86 with names): it reads who is being
addressed from the words, not from speaker labels (`testset/points_*_anon.jsonl`). v4 in exactly the live app's
setting (speech → Higgs STT, no names): **0.91 [0.84, 0.97]** (speaks at 38/45, stays quiet at 204/210).

**Live app** (`app.py` + `web/`, `./run.sh tv` → http://localhost:8500): one shared microphone in the browser →
Higgs Realtime (server VAD segments the talk, higgs-stt-3.1 transcribes) → judge + speculative reply → TTS 3
**streamed** to the page. Editable AI name, people and fact sheet; every line shows the decision and the judge's
reading of it ("to: person", "asks for something · already answered"…). The mic is muted while the AI talks.
Headless-Chromium test with the old meeting as the microphone (`browser_test.py`): every line it heard was
decided correctly in 0.6–1.0 s, and the AI started talking **1.0–1.6 s after the line ended** (2.2–5.5 s before
streaming the TTS; v2 and v4 runs). If people talk over the AI, the muted mic drops their words (the scripted test mic does this
on purpose; people usually wait).

**When the AI knows a lot more** (`testset/points_test_bigfacts.jsonl`): each test meeting's 6–8 facts padded with
40 facts from the other meetings. v4 drops from 0.917 to 0.856 [0.80, 0.91] (more false alarms, more misses).
Showing the judge only the 8 facts with the most word overlap with the last three lines (`retrieve.py`, no API
calls) brings it back to 0.903 [0.84, 0.96]. The live app does this when the fact sheet has more than 12 facts:
each line's 6 requests get the top 8 facts, in freshly opened sessions (decisions in ~0.8 s; browser test with 44
facts: every decision right except one line flagged against a padding fact that conflicts with the real deadline).
Moving the facts into each request instead, so pre-opened sessions could serve any fact sheet, cost accuracy
(0.886 on test, 0.872 with the big fact sheet), so the facts stay in the session instructions.

**Real meetings** (`ami_chatty.py`): two real AMI design meetings (manual transcripts, first 30 min each), with an AI
"Nova" that knows the project brief all AMI teams were given (sell at 25 €, cost ≤ 12.50 €, …). Nobody in AMI talks
to an AI, so the right behaviour is to stay quiet except when the group asks about or contradicts the brief. v4
speaks 0 times in ES2014c and 9 times in ES2009c (0.15 per minute overall). The best of the 9 is a real group
question it can answer ("our goal was to be under twelve fifty or we have to be under twelve fifty? Do we
remember?"); three flag cost or price statements that break the brief ("upwards of seventeen fifty Euro per
remote", "sell 'em for thirty", "if we run a bit over budget"); about three are unnecessary ("We can have a duck
tape casing.").
The same 60 minutes from the room's **far-field microphone** (Array1-01, one shared mic), transcribed by Higgs STT,
nobody identified (`ami_farfield.py`): real-room STT is rough ("Did you?" → "对。"), yet the judge stays quiet:
5 and 2 interventions (0.12 per minute), nearly all about the brief: the "upwards of 1750 euro to produce each one"
estimate, "our goal was to be under 1250, or we have to be under 1250", a competitor remote that controls four
devices (the brief says TV only); one ("is there an option that we could have that off or on") is unnecessary.
**Through the live app itself** (`ami_live.py`: minutes 10–25 of ES2009c's far-field mic streamed in real time into
`./run.sh tv`; Higgs's VAD segments it, Higgs STT transcribes, v4 decides): 163 lines, the AI spoke 4 times
(0.27/min), all about the brief: "That's way over our twelve fifty budget" (after "seventeen fifty euro"), "The
selling price is fixed at 25 euros" (twice in a row, once for "multiply 17.50 by two, that's 35" and once for "sell
them for thirty": correct but repetitive), and "drop the voice recognition to stay within the twelve fifty limit".
The "do we have to be under 1250?" question arrived merged with the next speaker's answer in one VAD segment (one
shared mic joins turns), so it looked answered and the AI stayed quiet. Same test on ES2014c: 176 lines, 1
intervention (a competitor remote "can control four devices", against the brief's TV-only rule), but the reply
drifted to the budget instead of making that point. A few segments were transcribed twice. The
app now skips a reply that is ≥ 80% the same as one of its last three lines (`repeats` in `app.py`; the line still
shows "SPEAK").

**Limits.** The labelled test meetings are written dialogue (by an LLM, then spoken by TTS), not recorded meetings;
the real AMI check has no labels and uses manual transcripts, not ASR. The facts the AI needs are given in a short fact
sheet. Each decision costs 3 Realtime text requests (~3 × 1k input tokens).

## Earlier results (2026-10-21, day)


| policy | spoke when it should | stayed silent when it should | balanced acc | AUROC |
|---|---|---|---|---|
| heuristic: "?" in the utterance or "Higgs" said | 5/5 | 11/15 | **0.87** | – |
| Qwen3-4B LoRA trained on AMI meetings (ours, Oct 10) | 5/5 | 5/15 | 0.67 | ~0.6 |
| same recipe on **synthetic meetings with an AI participant** + 1/3 AMI (`gen_data.py`, `train_synth.py`) | 2/5 | 13/15 | 0.63 | **0.78** |
| zero-shot Higgs Realtime (text mode), asked SPEAK/SILENT | 5/5 | 2/15 | 0.57 | – |
| zero-shot Qwen3-4B | 5/5 | 0/15 | 0.50 | – |

- The AMI-trained model doesn't transfer: nobody in AMI is an AI addressed by name, and its p(speak) drifts
  upward as the meeting gets longer (0.44 → 0.84).
- Synthetic data (239 meetings written line by line by Higgs text mode, AI names drawn from a list that excludes
  "Higgs") teaches *addressing* perfectly: p = 1.000 at "Higgs, what do you think…" and "Higgs, can you remind
  us…", and ≤ 0.04 at all 15 silent points. But it misses every **implicit group question** ("How long is the
  video supposed to be again?" 0.002, "Does anyone remember if the judges want a live URL?" 0.014). On held-out
  synthetic meetings it scores AUROC 0.99, so the synthetic data is easier than real talk.
- Zero-shot LLMs can't stay quiet (Higgs says SPEAK at 18/20 pauses).

### What this meant (before the fix)

"Should the AI speak?" is two decisions:
1. **Am I being addressed?** Learnable, and the synthetic-data LoRA does it well (75 ms/decision on an RTX 4090).
2. **Is this a question to the group that I can usefully answer?** Needs to know what the AI knows; an LLM with the
   facts in context, asked only about utterances that are questions not aimed at a named person, is the obvious
   next step.

A composite of the two would likely beat the heuristic here, but this 20-turn meeting was written knowing these
failure modes, so it can't be the test for that. It needs a fresh, independently written test set, which makes
Third Voice the least ready of the four ideas for this weekend.

## Files

Night additions: `testset/` (guideline, meetings, blind annotation, `build_points.py`), `tv_eval.py` (all
policies, cached Higgs answers in `out/cache_*.jsonl`), `paired.py`, `lora_eval.py` (+ `_job.sh`), `stt_meetings.py`, `live_demo.py`.

Day: 
`meeting.py` (test meeting + gold), `policy_gpu.py` / `policy_job.sh` (AMI LoRA + zero-shot on Slurm),
`policy_higgs.py` (Higgs zero-shot + heuristic), `gen_data.py` (synthetic meetings), `train_synth.py` /
`train_synth_job.sh` (fine-tune + evaluation), `out/` (decisions, synthetic data, adapter).
