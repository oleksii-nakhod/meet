"""Meet: a small video call (2-5 people) joined by link, with an AI participant ("Higgs") on slot 0.

  /                         create a room (title, AI name, setting, facts, optional passcode)
  POST /api/rooms           -> {"id", "link": "/r/<id>"}; needs the MEET_ADMIN token (field "admin" or X-Admin) if set
  /r/<id>[?k=passcode]      join screen, then the call
  /ws/room?room=&pid=&name=&k=   control JSON + 20 ms int16 24 kHz audio (server -> client: 1 byte slot + PCM)
  /ws/video?room=&pid=      camera JPEGs (1 byte slot + JPEG) and the AI's avatar video (slot 0: fMP4 chunks)

Run: python server.py [--port 8600] [--host 127.0.0.1] [--https]"""
import argparse
import json
import os
import secrets
import sys

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from room import Room  # noqa: E402

WEB = os.path.join(HERE, "web")
ADMIN = os.environ.get("MEET_ADMIN", "")
ROOMS = {}
app = FastAPI()


@app.get("/api/config")
def config():
    return {"admin_required": bool(ADMIN)}


@app.post("/api/rooms")
async def create_room(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if ADMIN and not secrets.compare_digest(str(body.get("admin") or request.headers.get("x-admin") or ""), ADMIN):
        return JSONResponse({"error": "admin token required"}, status_code=403)
    facts = body.get("facts") or []
    if isinstance(facts, str):
        facts = facts.split("\n")
    room = Room(str(body.get("title") or "")[:120], str(body.get("ai") or "Higgs")[:40], str(body.get("setting") or "")[:300],
                [str(f).strip()[:500] for f in facts][:100], str(body.get("passcode") or "")[:64],
                str(body.get("agenda") or "")[:40])
    ROOMS[room.id] = room
    link = f"/r/{room.id}"
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    full = f"{proto}://{request.headers.get('host', request.url.netloc)}{link}"
    print(f"room created: {room.brief['title']!r} -> {full}" + (f"?k={room.passcode}" if room.passcode else ""), flush=True)
    return {"id": room.id, "link": link}


@app.get("/api/rooms/{rid}")
def room_info(rid: str):
    room = ROOMS.get(rid)
    if not room:
        return JSONResponse({"error": "no such room"}, status_code=404)
    return {"id": rid, "title": room.brief["title"], "ai": room.brief["ai"], "passcode": bool(room.passcode),
            "people": len(room.humans), "max": 5, "agenda": room.agenda}


@app.middleware("http")
async def revalidate(request: Request, call_next):
    """Pages and scripts are checked with the server on every load. Behind Cloudflare (public.sh) they were cached for
    4 hours, so after a restart pages ran the old room.js against the new server (2026-10-23: no live face)."""
    r = await call_next(request)
    p = request.url.path
    if p == "/" or p.endswith((".js", ".html", ".css")):
        r.headers.setdefault("Cache-Control", "no-cache")
    return r


@app.get("/r/{rid}")
def room_page(rid: str):
    html = open(os.path.join(WEB, "room.html")).read()
    for js in ("agendas.js", "room.js"):  # a changed file gets a new URL, so no cache can hand out the old one
        html = html.replace(f'src="/{js}"', f'src="/{js}?v={int(os.path.getmtime(os.path.join(WEB, js)))}"')
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.websocket("/ws/room")
async def ws_room(ws: WebSocket):
    q = ws.query_params
    await ws.accept()
    room = ROOMS.get(q.get("room", ""))
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
