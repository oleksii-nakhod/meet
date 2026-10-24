"""Meet: a small video call (2-5 people) joined by link, with an AI participant ("Higgs") on slot 0.

  /                         join a room by its id or link, or create one
  POST /api/rooms           -> {"id", "link": "/r/<id>"}; "id" is optional (letters, digits, - and _; taken -> 409);
                            needs the MEET_ADMIN token (field "admin" or X-Admin) only if that is set
  /r/<id>[?k=passcode]      join screen, then the call; a room empty for MEET_ROOM_IDLE s (600) is freed: the link
                            stops working and the id can be used again
  /ws/room?room=&pid=&name=&k=   control JSON + 20 ms int16 24 kHz audio (server -> client: 1 byte slot + PCM)
  /ws/video?room=&pid=      camera JPEGs (1 byte slot + JPEG) and the AI's avatar video (slot 0: fMP4 chunks)

Run: python server.py [--port 8600] [--host 127.0.0.1] [--https]"""
import argparse
import asyncio
import json
import os
import re
import secrets
import sys
import time

from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from meet_config import livekit_configured, load_environment  # noqa: E402

load_environment()  # must happen before room.py imports the Boson client
import room as room_module  # noqa: E402
from room import Room  # noqa: E402

try:
    import boson_http  # noqa: E402
    from agent import spend  # noqa: E402
    MODELS = boson_http.Models(spend=spend)  # the knowledge base extracts facts with it
except Exception as e:  # noqa: BLE001  (no key / stub agent: documents are still indexed, just no fact extraction)
    print(f"knowledge-base fact extraction off: {e!r}", flush=True)
    MODELS = None

WEB = os.path.join(HERE, "web")
ADMIN = os.environ.get("MEET_ADMIN", "")
MAX_ROOMS = int(os.environ.get("MEET_MAX_ROOMS", "100"))  # anyone can create rooms, so there's a cap
IDLE = float(os.environ.get("MEET_ROOM_IDLE", "600"))  # a room empty this long (s) is freed
ROOM_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
ROOMS = {}

app = FastAPI()


@app.middleware("http")
async def revalidate(request, call_next):
    """Pages and scripts are checked for a newer version on every load (a 304 if unchanged). Without this,
    Cloudflare kept serving a 4-hour-old room.js after an update."""
    response = await call_next(request)
    response.headers.setdefault("cache-control", "no-cache")
    return response


def free_idle_rooms():
    """Forget rooms nobody has been in for IDLE seconds (counted from creation if nobody ever joined). The agent and
    avatar already stopped when the room emptied; this frees the id and the link."""
    now = time.time()
    for rid, room in list(ROOMS.items()):
        if room.empty_since is not None and now - room.empty_since >= IDLE:
            del ROOMS[rid]
            room.log({"type": "freed"})
            room_module.LIBRARY.drop(rid)
            print(f"room freed after {IDLE:g} s empty: {rid}", flush=True)


def get_room(rid):
    free_idle_rooms()
    return ROOMS.get(rid)


def taken(rid):
    """A room has this id, ignoring case (so /r/standup and /r/Standup can't be two different rooms)."""
    return any(rid.lower() == r.lower() for r in ROOMS)


@app.get("/api/config")
def config():
    return {"admin_required": bool(ADMIN), "agent": room_module.HiggsAgent.__module__,
            "avatar_livekit": livekit_configured()}


@app.post("/api/rooms")
async def create_room(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if ADMIN and not secrets.compare_digest(str(body.get("admin") or request.headers.get("x-admin") or ""), ADMIN):
        return JSONResponse({"error": "admin token required"}, status_code=403)
    free_idle_rooms()
    rid = str(body.get("id") or "").strip()
    if rid and not ROOM_ID.fullmatch(rid):
        return JSONResponse({"error": "A room ID is 1-64 letters, digits, - or _."}, status_code=400)
    if rid and taken(rid):
        return JSONResponse({"error": f"There's already a room called {rid}; pick another ID or join it."}, status_code=409)
    if len(ROOMS) >= MAX_ROOMS:
        return JSONResponse({"error": "This server has too many rooms to make another."}, status_code=503)
    while not rid or taken(rid):
        rid = secrets.token_urlsafe(8)
    facts = body.get("facts") or []
    if isinstance(facts, str):
        facts = facts.split("\n")
    room = Room(str(body.get("title") or "")[:120], str(body.get("ai") or "Higgs")[:40], str(body.get("setting") or "")[:300],
                [str(f).strip()[:500] for f in facts][:100], str(body.get("passcode") or "")[:64],
                str(body.get("agenda") or "")[:40], str(body.get("mode") or "default"), rid=rid)
    for d in room.kb.docs():  # documents stay on disk by room id: a new room doesn't inherit an old one's
        room.kb.delete(d["id"])
    room.kb.set_facts(room.brief["facts"])
    ROOMS[rid] = room
    link = f"/r/{room.id}"
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    full = f"{proto}://{request.headers.get('host', request.url.netloc)}{link}"
    print(f"room created: {room.brief['title']!r} -> {full}" + (f"?k={room.passcode}" if room.passcode else ""), flush=True)
    return {"id": room.id, "link": link}


@app.get("/api/rooms/{rid}")
def room_info(rid: str):
    room = get_room(rid)
    if not room:
        return JSONResponse({"error": "no such room"}, status_code=404)
    return {"id": rid, "title": room.brief["title"], "ai": room.brief["ai"], "passcode": bool(room.passcode),
            "people": len(room.humans), "max": 5, "agenda": room.agenda, "livekit": room.livekit,
            "avatar_livekit": room.avatar_livekit}


@app.post("/api/rooms/{rid}/livekit-token")
async def livekit_token(rid: str, request: Request):
    """Mint one short-lived participant token after the normal Meet join succeeds."""
    room = ROOMS.get(rid)
    if not room or not room.avatar_livekit or not livekit_configured():
        return JSONResponse({"error": "LiveKit is not available for this room"}, status_code=404)
    try:
        body = await request.json()
    except Exception:
        body = {}
    pid = str(body.get("pid") or "")[:64]
    name = str(body.get("name") or "Guest").strip()[:40] or "Guest"
    supplied = str(body.get("passcode") or "")[:64]
    if room.passcode and not secrets.compare_digest(supplied, room.passcode):
        return JSONResponse({"error": "wrong passcode"}, status_code=403)
    human = room.by_pid(pid)
    if not pid or human is None:
        return JSONResponse({"error": "join the Meet room first"}, status_code=403)
    from livekit_transport import browser_credentials
    return browser_credentials(room.id, pid, name)


# ---------- knowledge base (documents, links, pasted text; the fact sheet) ----------
async def _ingest(room, doc_id):
    await room.kb.ingest(doc_id, MODELS)
    await room.event({"type": "kb", **room.kb.state()})


@app.get("/api/rooms/{rid}/kb")
def kb_state(rid: str):
    room = get_room(rid)
    if not room:
        return JSONResponse({"error": "no such room"}, status_code=404)
    return room.kb.state()


@app.post("/api/rooms/{rid}/kb/files")
async def kb_upload(rid: str, file: UploadFile = File(...)):
    room = get_room(rid)
    if not room:
        return JSONResponse({"error": "no such room"}, status_code=404)
    try:
        doc = room.kb.add_file(file.filename or "upload", await file.read())
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    asyncio.ensure_future(_ingest(room, doc["id"]))
    return doc


@app.post("/api/rooms/{rid}/kb/sources")
async def kb_source(rid: str, request: Request):
    room = get_room(rid)
    if not room:
        return JSONResponse({"error": "no such room"}, status_code=404)
    body = await request.json()
    try:
        if body.get("url"):
            doc = room.kb.add_url(str(body["url"]))
        elif str(body.get("text") or "").strip():
            doc = room.kb.add_text(str(body.get("title") or "Pasted text")[:200], str(body["text"])[:500_000])
        else:
            raise ValueError("give a url or some text")
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    asyncio.ensure_future(_ingest(room, doc["id"]))
    return doc


@app.delete("/api/rooms/{rid}/kb/docs/{doc_id}")
async def kb_delete(rid: str, doc_id: str):
    room = get_room(rid)
    if not room:
        return JSONResponse({"error": "no such room"}, status_code=404)
    room.kb.delete(doc_id)
    return {"ok": True}


@app.put("/api/rooms/{rid}/kb/facts")
async def kb_facts(rid: str, request: Request):
    room = get_room(rid)
    if not room:
        return JSONResponse({"error": "no such room"}, status_code=404)
    body = await request.json()
    room.kb.set_facts([str(f).strip()[:500] for f in body.get("facts", []) if str(f).strip()][:200])
    return room.kb.state()


@app.get("/r/{rid}")
def room_page(rid: str):
    return FileResponse(os.path.join(WEB, "room.html"), headers={"Cache-Control": "no-store"})


@app.websocket("/ws/room")
async def ws_room(ws: WebSocket):
    q = ws.query_params
    await ws.accept()
    room = get_room(q.get("room", ""))
    pid, name = q.get("pid", "")[:64], (q.get("name") or "Guest").strip()[:40] or "Guest"
    if not room or not pid:
        await ws.close(4004, "no such room")
        return
    if room.passcode and not secrets.compare_digest(q.get("k", ""), room.passcode):
        await ws.close(4003, "wrong passcode")
        return
    h = room.join(pid, name, ws)
    if not h:
        await ws.close(4001, "room is full")
        return
    try:
        while True:
            m = await ws.receive()
            if m.get("type") == "websocket.disconnect" or h.out.closed:
                break
            if m.get("bytes") is not None:
                room.on_audio(h, m["bytes"])
            elif m.get("text"):
                try:
                    msg = json.loads(m["text"])
                except ValueError:
                    continue
                if isinstance(msg, dict):
                    await room.on_text(h, msg)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        room.leave(h)


@app.websocket("/ws/video")
async def ws_video(ws: WebSocket):
    q = ws.query_params
    await ws.accept()
    room, pid = ROOMS.get(q.get("room", "")), q.get("pid", "")
    h = room and room.by_pid(pid)
    if not h:  # join /ws/room first
        await ws.close(4004, "not in the room")
        return
    out = room.attach_video(h, ws)
    try:
        while True:
            m = await ws.receive()
            h = room.by_pid(pid)  # a control-socket reconnect makes a new Human that keeps this socket
            if m.get("type") == "websocket.disconnect" or out.closed or not h or h.video is not out:
                break
            if m.get("bytes") is not None:
                room.on_video(h, m["bytes"])
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        out.close()
        h = room.by_pid(pid)
        if h and h.video is out:
            h.video = None


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to let others on the network join")
    ap.add_argument("--https", action="store_true", help="self-signed HTTPS (browsers need it for camera/mic off localhost)")
    a = ap.parse_args()
    ssl = {}
    if a.https:
        from mmhm.local_https import uvicorn_ssl
        ssl = uvicorn_ssl()
    print(f"agent: {sys.modules[Room.__module__].HiggsAgent.__module__}", flush=True)
    print(f"meet on {'https' if a.https else 'http'}://{a.host}:{a.port}/" + ("  (MEET_ADMIN set: creating rooms needs ?admin=<token>)" if ADMIN else ""), flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", **ssl)
