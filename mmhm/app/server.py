"""Turing Clock web app: talk to Higgs in the browser; press the button when it gives itself away.

Each round is blind-assigned to `plain` (Higgs Realtime with its own server_vad turn detection, plain persona)
or `mmhm` (our turn-taking layer + humanized persona + listening noises). The browser streams 20 ms mic
frames over a WebSocket; the server runs one Engine per player and streams 20 ms speaker frames back, plus
telemetry for the on-screen meters.

Run: python app/server.py [--port 8000]   then open http://localhost:8000 (headphones recommended)."""
import argparse
import asyncio
import json
import os
import random
import sys
import time
import uuid

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, ".."))
from app.personas import HUMANIZED, PLAIN, rule_check  # noqa: E402
from mmhm.engine import N, Engine, Policy  # noqa: E402
from mmhm.higgs import load_wav, outdir, save_wav, trim, tts  # noqa: E402

DATA = os.environ.get("GAME_DATA") or outdir("game")  # override for demos/tests
from mmhm import voices  # noqa: E402
RECORD = os.environ.get("RECORD") == "1"  # save each round's audio (mic + speaker) for demo videos
EXPRESSIVE = os.environ.get("MMHM_EXPRESSIVE", "1") == "1"  # humanized arm: Realtime writes, TTS 3 speaks (laughs, sighs)
# Maya's voice: "The Soft Canadian Companion" from the hackathon voice library, cloned via the Voices API
# (set MAYA_VOICE=chloe to use a preset instead)
VOICE = os.environ.get("MAYA_VOICE") or voices.get("maya")["voice"]
BC_TEXTS = ["Mm-hm.", "Yeah.", "Mm.", "Uh-huh."]
app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "static")), name="static")
_bc_clips = None


def bc_clips():
    """Listening noises in the agent's own voice (Higgs TTS 3, same preset as Realtime), cached on disk."""
    global _bc_clips
    if _bc_clips is None:
        clips = []
        for i, text in enumerate(BC_TEXTS):
            path = os.path.join(DATA, f"bc_{VOICE[:24]}_{i}.wav")
            if not os.path.exists(path):
                pcm, _, _ = tts(text, voice=VOICE)
                save_wav(path, trim(pcm))
            clips.append((load_wav(path).astype(np.float32) * 0.7).astype(np.int16))
        _bc_clips = clips
    return _bc_clips


@app.on_event("startup")
async def warm_up():
    """Make the listening-noise clips and load the turn-taking models before the first player arrives."""
    await asyncio.to_thread(bc_clips)
    from mmhm.models import Signals
    Signals(vap=True, bcdet=True, bc=True).close()


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))


@app.post("/api/round")
async def save_round(body: dict):
    body["saved_at"] = time.time()
    with open(os.path.join(DATA, "rounds.jsonl"), "a") as f:
        f.write(json.dumps(body) + "\n")
    return {"ok": True}


@app.get("/api/rounds")
def rounds():
    path = os.path.join(DATA, "rounds.jsonl")
    rs = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    return JSONResponse([r for r in rs if not r.get("forced")])


@app.get("/dashboard")
def dashboard():
    return FileResponse(os.path.join(ROOT, "static", "dashboard.html"))


@app.get("/api/stats")
def stats():
    rounds = []
    path = os.path.join(DATA, "rounds.jsonl")
    if os.path.exists(path):
        rounds = [json.loads(line) for line in open(path)]
    out = {}
    for mode in ("plain", "mmhm", "human"):
        rs = [r for r in rounds if r.get("mode") == mode and r.get("seconds") and not r.get("forced")]
        secs = sorted(r["seconds"] for r in rs)
        tells = {}
        for r in rs:
            if r.get("tell"):
                tells[r["tell"]] = tells.get(r["tell"], 0) + 1
        guessed = [r for r in rs if r.get("guess") in ("human", "ai")]
        out[mode] = {"n": len(rs), "median_s": secs[len(secs) // 2] if secs else None,
                     "survived": sum(not r.get("pressed") for r in rs), "tells": tells,
                     "judged_human": f"{sum(r['guess'] == 'human' for r in guessed)}/{len(guessed)}"}
    return JSONResponse(out)


# Party mode: humans waiting on /operator.html who may be put through to a player instead of Higgs.
OPERATORS = []  # list of (websocket, done_event)
HUMAN_P = float(os.environ.get("PARTY_HUMAN_P", 0.5))  # chance a party round goes to a waiting human


@app.websocket("/ws_operator")
async def operator_endpoint(ws: WebSocket):
    await ws.accept()
    done = asyncio.Event()
    OPERATORS.append((ws, done))
    await ws.send_text(json.dumps({"type": "waiting", "queue": len(OPERATORS)}))
    await done.wait()  # the player's handler relays audio and sets this when the round ends


async def human_round(ws, op_ws, round_id):
    """Relay 20 ms frames both ways between a player and a human operator until either side leaves."""
    await op_ws.send_text(json.dumps({"type": "connected", "round": round_id}))
    await ws.send_text(json.dumps({"type": "ready"}))

    async def op_to_player():
        while True:
            m = await op_ws.receive()
            if m.get("type") == "websocket.disconnect":
                return
            if m.get("bytes") is not None:
                await ws.send_bytes(m["bytes"])

    pump = asyncio.ensure_future(op_to_player())
    try:
        while True:
            m = await ws.receive()
            if m.get("type") == "websocket.disconnect" or pump.done():
                break
            if m.get("bytes") is not None:
                await op_ws.send_bytes(m["bytes"])
            elif m.get("text") and json.loads(m["text"]).get("type") == "verdict":
                await ws.send_text(json.dumps({"type": "reveal", "mode": "human", "round": round_id}))
    finally:
        pump.cancel()
        try:
            await op_ws.send_text(json.dumps({"type": "ended"}))
        except Exception:
            pass


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    mode = ws.query_params.get("mode") or random.choice(["plain", "mmhm"])
    round_id = uuid.uuid4().hex[:10]
    if ws.query_params.get("party") is not None and not ws.query_params.get("mode") and random.random() < HUMAN_P:
        while OPERATORS:  # skip operators who left while waiting
            op_ws, done = OPERATORS.pop(0)
            try:
                await op_ws.send_text(json.dumps({"type": "ping"}))
            except Exception:
                done.set()
                continue
            await ws.send_text(json.dumps({"type": "hello", "round": round_id, "mode_hidden": True}))
            try:
                await human_round(ws, op_ws, round_id)
            finally:
                done.set()
            return
    log = []
    if mode == "mmhm":
        from mmhm.models import ProcSignals
        signals = ProcSignals(vap=True, bcdet=True, bc=True, wait=False)  # one process per player (GIL)
        await asyncio.to_thread(signals.ready.wait, 120)
        engine = Engine(mode="mmhm", policy=Policy(backchannels=True), instructions=HUMANIZED, voice=VOICE,
                        signals=signals, bc_clips=bc_clips(), log=log, expressive=EXPRESSIVE)
        engine.on_transcript = rule_check  # "are you an AI?" -> "hey, that's the one rule!"
    else:
        signals = None
        engine = Engine(mode="server_vad", instructions=PLAIN, voice=VOICE, log=log)
    await ws.send_text(json.dumps({"type": "hello", "round": round_id, "mode_hidden": True}))
    sent_log = 0
    rec_mic, rec_spk = [], []  # per-round recording, only with RECORD=1 (demo videos; tell players first)
    try:
        async with engine:
            await ws.send_text(json.dumps({"type": "ready"}))  # the page starts the mic only now (no backlog)
            last_tele = 0.0
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    mic = np.frombuffer(msg["bytes"], np.int16)
                    if len(mic) != N:
                        mic = np.resize(mic, N)
                    stopped_before = sum(r["kind"] == "stop" for r in log)
                    out = await engine.tick(mic)
                    if RECORD:
                        rec_mic.append(mic.copy()); rec_spk.append(out.copy())
                    if sum(r["kind"] == "stop" for r in log) > stopped_before:
                        await ws.send_text(json.dumps({"type": "flush"}))
                    await ws.send_bytes(out.tobytes())
                    if engine.t - last_tele >= 0.1:
                        last_tele = engine.t
                        await ws.send_text(json.dumps({
                            "type": "tele", "t": round(engine.t, 2), "state": engine.state,
                            "p_now": round(engine.sig.get("p_now", 0), 3),
                            "p_bcdet": round(engine.sig.get("p_bcdet", 0), 3),
                            "gain": round(engine.gain, 2)}))
                    while sent_log < len(log):
                        await ws.send_text(json.dumps({"type": "event", **log[sent_log]}))
                        sent_log += 1
                elif msg.get("text"):
                    m = json.loads(msg["text"])
                    if m.get("type") == "verdict":
                        await ws.send_text(json.dumps({"type": "reveal", "mode": mode, "round": round_id}))
    except WebSocketDisconnect:
        pass
    finally:
        if signals is not None:
            signals.close()
        with open(os.path.join(DATA, f"log_{round_id}_{mode}.jsonl"), "w") as f:
            for r in log:
                f.write(json.dumps(r) + "\n")
        if RECORD and rec_mic:
            import soundfile as sf
            sf.write(os.path.join(DATA, f"rec_{round_id}_{mode}.wav"),
                     np.stack([np.concatenate(rec_mic), np.concatenate(rec_spk)], 1), 24000, subtype="PCM_16")


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to let phones on the same network join")
    ap.add_argument("--https", action="store_true", help="self-signed HTTPS (browsers need it for the mic off localhost)")
    a = ap.parse_args()
    ssl = {}
    if a.https:
        from mmhm.local_https import uvicorn_ssl
        ssl = uvicorn_ssl()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", **ssl)
