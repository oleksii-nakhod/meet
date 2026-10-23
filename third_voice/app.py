"""Third Voice, live: an AI that sits in on a group conversation through one shared microphone and speaks only
when it should (addressed, a group question it can answer, a correction of its facts).

Browser mic (20 ms Int16 frames, 24 kHz) -> Higgs Realtime (server_vad segments the talk, higgs-stt-3.1
transcribes it) -> after every line, the decomposed judge (tv_eval.DEC_SYS2, 3 samples, majority) plus a fact-check (3 samples,
unanimous; together "v4", tv_eval.py) decides from the
transcript, while a reply is drafted speculatively -> on SPEAK, TTS 3 speaks the reply through the page.
Nobody is diarised: every human line is "Someone" (on our test set this costs nothing: 0.89 vs 0.86 balanced
accuracy with names). The page mutes the mic while the AI talks, so it doesn't hear itself.

Run: python app.py [--port 8500] [--https]   then open http://localhost:8500/"""
import argparse
import asyncio
import base64
import json
import os
import sys
import time

import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mmhm"))
from mmhm.higgs import BASE, HDR, Session  # noqa: E402

import retrieve  # noqa: E402
import tv_eval  # noqa: E402
from live_demo import REPLY_SYS, Pool  # noqa: E402

VOICE = os.environ.get("TV_VOICE", "chloe")
RETRIEVE_OVER = 12  # more facts than this: retrieve the top 8 per line (test set: 0.856 -> 0.903 with 46-48 facts)
app = FastAPI()


def tts_stream(text, voice):
    """Higgs TTS 3, streamed: yields Int16 PCM chunks (24 kHz) as they are synthesized. Blocking; run in a thread."""
    body = {"model": "higgs-tts-3", "input": text, "response_format": "pcm", "stream": True, "voice": voice}
    with requests.post(f"{BASE}/audio/speech", headers=HDR, json=body, stream=True, timeout=120) as r:
        r.raise_for_status()
        carry = b""
        for chunk in r.iter_content(9600):
            chunk = carry + chunk
            n = len(chunk) - len(chunk) % 2
            carry = chunk[n:]
            if n:
                yield chunk[:n]


def repeats(reply, previous, threshold=0.8):
    """True if the reply is nearly the same as one of the AI's recent lines."""
    from difflib import SequenceMatcher
    norm = lambda x: " ".join(x.lower().split())  # noqa: E731
    return any(SequenceMatcher(None, norm(reply), norm(p)).ratio() >= threshold for p in previous)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    cfg = json.loads(await ws.receive_text())  # {"ai", "setting", "people": [...], "facts": [...]}
    ai, people, facts, setting = cfg["ai"], cfg.get("people") or [], cfg.get("facts") or [], cfg.get("setting") or "a meeting"
    point = {"target_speaker": ai, "setting": setting, "people": people, "facts": facts}
    judges = Pool(tv_eval.dec_prompt({**point, "context_turns": [], "current_turn": {"speaker": "", "text": ""}},
                                     tv_eval.DEC_SYS2)[0], 3)
    dummy = {**point, "context_turns": [], "current_turn": {"speaker": "", "text": ""}}
    checkers = Pool(tv_eval.fact_prompt(dummy)[0], 3)  # v4: a dedicated fact-check catches implicit corrections
    writers = Pool(REPLY_SYS.format(ai=ai, setting=setting, people=", ".join(people) or "the group", facts=" ".join(facts)), 1)
    history = []
    lock = asyncio.Lock()

    async def send_json(obj):
        async with lock:
            await ws.send_text(json.dumps(obj))

    async def on_line(text, t_end):
        cur = {"speaker": "Someone", "text": text}
        r = {**point, "meeting": "live", "line": len(history), "context_turns": list(history), "current_turn": cur}
        draft = asyncio.ensure_future(writers.ask(
            "Conversation so far:\n" + "\n".join(f"{t['speaker']}: {t['text']}" for t in history[-14:])
            + f"\nSomeone: {text}\n\nYour reply:"))
        if len(facts) > RETRIEVE_OVER:  # big fact sheet: judge with the 8 most relevant facts (fresh sessions)
            r8 = {**r, "facts": retrieve.top_facts(r, 8)}
            votes = await asyncio.gather(*[tv_eval.ask_higgs(*tv_eval.dec_prompt(r8, tv_eval.DEC_SYS2)) for _ in range(3)],
                                         *[tv_eval.ask_higgs(*tv_eval.fact_prompt(r8)) for _ in range(3)])
        else:
            user, fuser = tv_eval.dec_prompt(r, tv_eval.DEC_SYS2)[1], tv_eval.fact_prompt(r)[1]
            votes = await asyncio.gather(*[judges.ask(user) for _ in range(3)], *[checkers.ask(fuser) for _ in range(3)])
        parsed = [tv_eval.parse_json(v) for v in votes[:3]]
        facts_hit = [bool((tv_eval.parse_json(v) or {}).get("contradicts")) for v in votes[3:]]
        # v4: majority of the judge, or a contradiction all three fact-checks agree on
        speak = sum(tv_eval.dec_rule(p, ai, 2) for p in parsed) >= 2 or all(facts_hit)
        if all(facts_hit) and parsed[0] is not None:
            parsed[0]["wrong"] = True
        history.append(cur)
        await send_json({"type": "line", "text": text, "speak": speak, "why": parsed[0],
                         "decided_s": round(time.perf_counter() - t_end, 2)})
        if not speak:
            draft.add_done_callback(lambda f: f.exception())
            return
        ans = (await draft).strip().strip('"')
        if repeats(ans, [t["text"] for t in history if t["speaker"] == ai][-3:]):
            return  # it already said this (e.g. the same correction to a restated mistake): don't say it again
        history.append({"speaker": ai, "text": ans})
        loop, q = asyncio.get_running_loop(), asyncio.Queue()

        def worker():  # stream TTS chunks to the page as they arrive: the AI starts talking at the first chunk
            try:
                for ch in tts_stream(ans, VOICE):
                    loop.call_soon_threadsafe(q.put_nowait, ch)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        loop.run_in_executor(None, worker)
        first = True
        while (ch := await q.get()) is not None:
            if first:
                first = False
                await send_json({"type": "ai", "text": ans, "speaks_s": round(time.perf_counter() - t_end, 2)})
            async with lock:
                await ws.send_bytes(ch)

    scfg = {"instructions": "You only listen. Answer every turn with a single period.", "output_modalities": ["text"],
            "audio": {"input": {"turn_detection": {"type": "server_vad", "silence_duration_ms": 500},
                                "transcription": {"model": "higgs-stt-3.1"}}}}
    async with Session(scfg) as s:
        state = {"ev": 0, "t_end": time.perf_counter()}

        async def pump():
            while True:
                await asyncio.sleep(0.02)
                while state["ev"] < len(s.events):
                    _, ev = s.events[state["ev"]]
                    state["ev"] += 1
                    typ = ev.get("type", "")
                    if typ == "input_audio_buffer.speech_started":
                        await send_json({"type": "hearing"})
                    elif typ == "input_audio_buffer.speech_stopped":
                        state["t_end"] = time.perf_counter()
                    elif typ == "conversation.item.input_audio_transcription.completed":
                        text = (ev.get("transcript") or "").strip()
                        if text:
                            asyncio.ensure_future(on_line(text, state["t_end"]))

        task = asyncio.ensure_future(pump())
        await send_json({"type": "ready"})
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await s.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(msg["bytes"]).decode()})
        except WebSocketDisconnect:
            pass
        finally:
            task.cancel()


app.mount("/", StaticFiles(directory=os.path.join(HERE, "web"), html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--https", action="store_true")
    a = ap.parse_args()
    ssl = {}
    if a.https:
        from mmhm.local_https import uvicorn_ssl
        ssl = uvicorn_ssl()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", **ssl)
