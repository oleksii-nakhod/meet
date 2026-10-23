"""Third Voice, end to end on Higgs: a meeting's audio (the TTS-rendered lines from stt_meetings.py) is streamed
line by line into a Higgs Realtime session that only transcribes (higgs-stt-3.1). After each line, the decomposed
judge (v4: tv_eval.DEC_SYS2, 3 samples, majority, OR a unanimous 3-sample fact-check) decides SPEAK/SILENT from
the *transcript*; on SPEAK,
Higgs (text mode) writes a one-or-two-sentence reply from the facts and the conversation, and TTS 3 speaks it.
The measured time from the end of the line to the reply's first TTS audio is kept as the silence before the AI
speaks, so the output WAV has the real timing.

-> out/demo/<meeting>.wav (16 kHz), out/demo/<meeting>.json (lines, transcripts, decisions, latencies, gold)
Usage: python live_demo.py M0 [B3 ...]"""
import asyncio
import base64
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mmhm"))
from mmhm.higgs import Session, tts  # noqa: E402

import tv_eval  # noqa: E402
from stt_meetings import meetings  # noqa: E402

AI_VOICE = "chloe"
REPLY_SYS = ("You are {ai}, an AI participant with a voice in a group conversation ({setting}) with {people}. "
             "What you know: {facts} You decided to speak up after the latest utterance. If the latest utterance "
             "states something that contradicts what you know (a wrong day, time, number, place, dose or rule), start by "
             "correcting it plainly. Otherwise say what a helpful teammate would say: one or two short spoken sentences, "
             "no lists, no preamble.")
GAP = 0.45  # pause between human lines (s)


class Pool:
    """Pre-opened Realtime text sessions with fixed instructions (one meeting's facts), each used for one request and
    replaced in the background: opening a session (~0.15 s, serialized by mmhm.higgs) is off the critical path."""

    def __init__(self, instructions, size):
        self.instructions, self.q = instructions, asyncio.Queue()
        for _ in range(size):
            self._refill()

    def _refill(self):
        async def open_one():
            s = Session({"instructions": self.instructions, "output_modalities": ["text"]})
            await s.__aenter__()
            await self.q.put(s)
        asyncio.ensure_future(open_one())

    async def ask(self, user):
        s = await self.q.get()
        self._refill()
        try:
            since = s.now()
            await s.say_text(user)
            _, ev = await s.wait("response.done", timeout=45, since=since)
            return s.transcript(ev["response"]["id"]).strip()
        finally:
            asyncio.ensure_future(s.__aexit__(None, None, None))


async def judge(pool, r, checkers=None):
    """v4: majority of 3 judge samples, or all 3 fact-check samples agree on a contradiction (tv_eval.py)."""
    user = tv_eval.dec_prompt(r, tv_eval.DEC_SYS2)[1]
    asks = [pool.ask(user) for _ in range(3)]
    if checkers:
        asks += [checkers.ask(tv_eval.fact_prompt(r)[1]) for _ in range(3)]
    votes = await asyncio.gather(*asks)
    speak = [tv_eval.dec_rule(tv_eval.parse_json(v), r["target_speaker"], 2) for v in votes[:3]]
    facts = [bool((tv_eval.parse_json(v) or {}).get("contradicts")) for v in votes[3:]]
    return sum(speak) >= 2 or (bool(facts) and all(facts)), votes


async def reply(pool, history, cur):
    ctx = "\n".join(f"{t['speaker']}: {t['text']}" for t in history[-14:])
    return await pool.ask(f"Conversation so far:\n{ctx}\n{cur['speaker']}: {cur['text']}\n\nYour reply:")


async def run(mid):
    m = next(x for x in meetings() if x["meeting_id"] == mid)
    out_dir = os.path.join(HERE, "out", "demo")
    os.makedirs(out_dir, exist_ok=True)
    mix, log, history = [], [], []
    dummy = {"target_speaker": m["ai_name"], "setting": m["setting"], "people": m["people"], "facts": m["facts"],
             "context_turns": [], "current_turn": {"speaker": "", "text": ""}}
    judges = Pool(tv_eval.dec_prompt(dummy, tv_eval.DEC_SYS2)[0], 3)
    checkers = Pool(tv_eval.fact_prompt(dummy)[0], 3)
    writers = Pool(REPLY_SYS.format(ai=m["ai_name"], setting=m["setting"], people=", ".join(m["people"]),
                                    facts=" ".join(m["facts"])), 1)
    await asyncio.sleep(2)  # let the pools fill
    async with Session({"instructions": "Transcription only.", "output_modalities": ["text"],
                        "audio": {"input": {"turn_detection": None, "transcription": {"model": "higgs-stt-3.1"}}}}) as s:
        for i, d in enumerate(m["lines"]):
            pcm16 = sf.read(os.path.join(HERE, "out", "stt", mid, f"{i:02d}.flac"), dtype="int16")[0]
            mix.append(pcm16.astype(np.float32) / 32768)
            pcm24 = resample_poly(pcm16.astype(np.float32), 3, 2).clip(-32768, 32767).astype(np.int16)
            since = s.now()
            for k in range(0, len(pcm24), 4800):
                await s.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm24[k:k + 4800].tobytes()).decode()})
            await asyncio.sleep(len(pcm16) / 16000)  # a line takes as long to say as it lasts (lets the pools refill)
            t0 = time.perf_counter()  # the line has just ended (audio fully sent)
            await s.send({"type": "input_audio_buffer.commit"})
            _, ev = await s.wait("conversation.item.input_audio_transcription.completed", timeout=30, since=since)
            text = (ev.get("transcript") or "").strip()
            t_stt = time.perf_counter() - t0
            cur = {"speaker": d["speaker"], "text": text}
            r = {"meeting": mid, "line": i, "target_speaker": m["ai_name"], "facts": m["facts"], "people": m["people"],
                 "setting": m["setting"], "context_turns": list(history), "current_turn": cur}
            # the reply is written speculatively, in parallel with the decision, and dropped if the AI stays silent
            draft = asyncio.ensure_future(reply(writers, list(history), cur))
            speak, votes = await judge(judges, r, checkers)
            t_judge = time.perf_counter() - t0
            row = {"line": i, "speaker": d["speaker"], "said": d["text"], "stt": text, "gold": d["label"],
                   "decision": "SPEAK" if speak else "SILENT", "stt_s": round(t_stt, 2), "decided_s": round(t_judge, 2)}
            history.append(cur)
            if speak:
                ans = (await draft).strip().strip('"')
                t_text = time.perf_counter() - t0
                pcm, ttfa, _ = await asyncio.to_thread(tts, ans, AI_VOICE)
                gap = t_text + (ttfa or 0)  # streamed TTS: the AI starts talking at its first audio
                mix.append(np.zeros(int(gap * 16000), np.float32))
                mix.append(resample_poly(pcm.astype(np.float32) / 32768, 2, 3))
                row.update(ai_reply=ans, speaks_after_s=round(gap, 2))
                history.append({"speaker": m["ai_name"], "text": ans})
            else:
                draft.add_done_callback(lambda f: f.exception())  # discard the unused draft quietly
            mix.append(np.zeros(int(GAP * 16000), np.float32))
            await asyncio.sleep(0.3)  # real meetings have gaps: lets the session pools refill between lines
            log.append(row)
            print(json.dumps(row), flush=True)
    sf.write(os.path.join(out_dir, f"{mid}.wav"), np.concatenate(mix) * 0.9, 16000)
    json.dump(log, open(os.path.join(out_dir, f"{mid}.json"), "w"), indent=1)
    agree = sum((x["decision"] == x["gold"]) for x in log)
    lat = [x["speaks_after_s"] for x in log if "speaks_after_s" in x]
    print(f"{mid}: decisions match the annotation {agree}/{len(log)}; AI spoke {len(lat)}x, "
          f"median {np.median(lat) if lat else float('nan'):.2f}s after the line ended")


async def main():
    for mid in sys.argv[1:] or ["M0"]:
        await run(mid)


if __name__ == "__main__":
    asyncio.run(main())
