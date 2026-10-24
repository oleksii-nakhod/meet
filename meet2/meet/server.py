"""HTTP routes, the room WebSocket, and start-up.

    python -m meet.server [--host 0.0.0.0] [--port 8600] [--https]
"""
from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .kb import KnowledgeBase, Library
from .rooms import KIND_AUDIO, KIND_VIDEO, Brief, Client, Person, Room, Rooms
from .transport.base import FRAME_BYTES
from .transport.ws import WSTransport

ROOT = config.ROOT
WEB = ROOT / "web"


def make_agent(app: FastAPI, room: Room):
    s = config.settings()
    if s.agent == "stub" or not s.boson_api_key:
        from .stub import StubAgent
        return StubAgent(room)
    from .agent import HiggsAgent
    return HiggsAgent(room, app.state.models, app.state.library)


class RoomIn(BaseModel):
    title: str = Field("Meeting", max_length=120)
    ai: str = Field("", max_length=40)
    setting: str = Field("online meeting", max_length=300)
    facts: list[str] = Field(default_factory=list, max_length=200)
    agenda: str | list[str] = ""
    passcode: str = Field("", max_length=64)
    avatar: str = "off"
    collection: str = Field("", max_length=64)
    admin: str = ""


class FactsIn(BaseModel):
    facts: list[str] = Field(default_factory=list, max_length=200)


class SourceIn(BaseModel):
    url: str = ""
    text: str = Field("", max_length=500_000)
    title: str = Field("", max_length=200)


def create_app() -> FastAPI:
    s = config.settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rooms = Rooms()
        app.state.library = Library(s.path(s.kb_dir))
        app.state.models = None
        if s.agent != "stub" and s.boson_api_key:
            from .boson import Models
            from .obs import Budget
            app.state.models = Models(s.boson_api_key, s.boson_base_url, llm=s.llm_model, stt=s.stt_model,
                                      tts=s.tts_model, voice=s.voice, language=s.language,
                                      noise_reduction=s.denoise == "boson", budget=Budget(s.budget))
        yield
        for room in list(app.state.rooms.rooms.values()):
            if room.agent:
                await room.agent.stop()
        if app.state.models:
            await app.state.models.close()

    app = FastAPI(title="Meet 2.0", lifespan=lifespan)

    def room_or_404(rid: str) -> Room:
        room = app.state.rooms.get(rid)
        if room is None:
            raise HTTPException(404, "no such room")
        return room

    def kb_for(room: Room) -> KnowledgeBase:
        return app.state.library.get(room.collection)

    # ------------------------------------------------------------------ pages
    @app.get("/")
    async def home():
        return FileResponse(WEB / "index.html")

    @app.get("/r/{rid}")
    async def room_page(rid: str):
        room_or_404(rid)
        return FileResponse(WEB / "room.html")

    # ------------------------------------------------------------------ API
    @app.get("/api/config")
    async def get_config():
        return {"admin_required": bool(s.admin_token), "livekit": s.livekit_enabled, "ai_name": s.ai_name,
                "agent": "stub" if (s.agent == "stub" or not s.boson_api_key) else "higgs",
                "avatar_default": s.avatar_default}

    @app.get("/api/presets")
    async def presets():
        return json.loads((ROOT / "config" / "presets.json").read_text())

    @app.get("/api/collections")
    async def collections():
        return app.state.library.list()

    @app.post("/api/rooms")
    async def create_room(body: RoomIn):
        if s.admin_token and body.admin != s.admin_token:
            raise HTTPException(403, "admin token required to create rooms")
        if body.avatar == "live" and not s.livekit_enabled:
            raise HTTPException(400, "the live avatar needs LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET")
        agenda = " · ".join(body.agenda) if isinstance(body.agenda, list) else body.agenda
        brief = Brief(title=body.title.strip() or "Meeting", ai=(body.ai or s.ai_name).strip(),
                      setting=body.setting.strip() or "online meeting",
                      facts=[f.strip() for f in body.facts if f.strip()], agenda=agenda)
        room = app.state.rooms.create(brief, passcode=body.passcode, avatar=body.avatar if body.avatar in (
            "off", "live") else "off", collection=body.collection)
        room.transport = WSTransport(room)
        kb = kb_for(room)
        if brief.facts and not kb.facts():
            kb.set_facts(brief.facts)
        room.agent = make_agent(app, room)
        return {"id": room.id, "url": f"/r/{room.id}", "host_key": room.host_key, "collection": room.collection}

    @app.get("/api/rooms/{rid}")
    async def room_info(rid: str):
        return room_or_404(rid).info()

    # knowledge base -----------------------------------------------------
    @app.get("/api/rooms/{rid}/kb")
    async def kb_state(rid: str):
        return kb_for(room_or_404(rid)).state()

    @app.post("/api/rooms/{rid}/kb/files")
    async def kb_upload(rid: str, file: UploadFile = File(...), by: str = Form("")):
        room = room_or_404(rid)
        data = await file.read()
        try:
            doc = kb_for(room).add_file(file.filename or "upload", data, added_by=by)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        asyncio.ensure_future(_ingest(room, doc["id"]))
        return doc

    @app.post("/api/rooms/{rid}/kb/sources")
    async def kb_source(rid: str, body: SourceIn):
        room = room_or_404(rid)
        try:
            if body.url:
                doc = kb_for(room).add_url(body.url)
            elif body.text.strip():
                doc = kb_for(room).add_text(body.title or "Pasted text", body.text)
            else:
                raise ValueError("give a url or some text")
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        asyncio.ensure_future(_ingest(room, doc["id"]))
        return doc

    @app.delete("/api/rooms/{rid}/kb/docs/{doc_id}")
    async def kb_delete(rid: str, doc_id: str):
        room = room_or_404(rid)
        kb_for(room).delete(doc_id)
        _kb_changed(room)
        return {"ok": True}

    @app.put("/api/rooms/{rid}/kb/facts")
    async def kb_facts(rid: str, body: FactsIn):
        room = room_or_404(rid)
        kb_for(room).set_facts([f.strip() for f in body.facts if f.strip()])
        _kb_changed(room)
        return kb_for(room).state()

    async def _ingest(room: Room, doc_id: str) -> None:
        kb = kb_for(room)
        room.event({"type": "kb", **kb.state()}, log=False)
        await kb.ingest(doc_id, app.state.models)
        _kb_changed(room)

    def _kb_changed(room: Room) -> None:
        room.event({"type": "kb", **kb_for(room).state()}, log=False)
        if room.agent:
            room.agent.on_kb_changed()

    # LiveKit ------------------------------------------------------------
    @app.post("/api/rooms/{rid}/livekit-token")
    async def livekit_token(rid: str, request: Request):
        room = room_or_404(rid)
        if not s.livekit_enabled:
            raise HTTPException(400, "LiveKit is not configured")
        body = await request.json()
        person = room.by_pid(str(body.get("pid", "")))
        if person is None:
            raise HTTPException(403, "join the room first")
        from .transport.livekit import browser_credentials
        return browser_credentials(room, person)

    # ------------------------------------------------------------------ the room socket
    @app.websocket("/ws/room")
    async def ws_room(ws: WebSocket, room: str = "", pid: str = "", name: str = "", k: str = "", h: str = ""):
        r = app.state.rooms.get(room)
        await ws.accept()
        if r is None:
            return await ws.close(4004, "no such room")
        if r.passcode and k != r.passcode:
            return await ws.close(4003, "wrong passcode")
        name = (name.strip() or "Guest")[:40]
        old = r.by_pid(pid) if pid else None
        if old is not None:                       # a reload or a second tab: the newer one wins the slot
            del r.people[old.slot]
            await old.client.close()
            try:
                await old.client.ws.close(4000, "replaced by a newer tab")
            except Exception:  # noqa: BLE001
                pass
            slot = old.slot
        else:
            slot = r.free_slot()
        if slot is None:
            return await ws.close(4001, "room full")
        person = Person(slot=slot, pid=pid or f"anon{slot}", name=name, client=Client(ws),
                        host=bool(h) and h == r.host_key)
        first = not r.people
        r.people[slot] = person
        person.client.send_json({**r.welcome(), "you": person.public()})
        r.event({"type": "join", "slot": slot, "name": name}, log=True)
        r.broadcast({"type": "roster", "people": r.roster()})
        if r.agent:
            if first:
                await r.agent.start()
            r.agent.on_roster(r.roster())
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data:
                    kind, payload = data[0], data[1:]
                    if kind == KIND_AUDIO and len(payload) == FRAME_BYTES and person.mic:
                        r.relay_audio(slot, payload)
                        if r.agent:
                            r.agent.feed(slot, payload)
                    elif kind == KIND_VIDEO and person.cam and len(payload) < 200_000:
                        r.relay_video(slot, payload)
                    continue
                if msg.get("text"):
                    await _control(r, person, json.loads(msg["text"]))
        except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
            pass
        finally:
            if r.people.get(slot) is person:
                del r.people[slot]
                await person.client.close()
                r.event({"type": "leave", "slot": slot, "name": name})
                r.broadcast({"type": "roster", "people": r.roster()})
                if r.agent:
                    r.agent.on_roster(r.roster())
                    if not r.people:
                        await r.agent.stop()

    async def _control(r: Room, p: Person, m: dict) -> None:
        t = m.get("type")
        if t == "chat":
            text = str(m.get("text", "")).strip()[:2000]
            if text:
                r.event({"type": "chat", "slot": p.slot, "name": p.name, "text": text})
                if r.agent:
                    r.agent.on_chat(p.slot, p.name, text)
        elif t in ("mic", "cam"):
            setattr(p, t, bool(m.get("on")))
            r.broadcast({"type": "roster", "people": r.roster()})
            if t == "mic" and r.agent:
                r.agent.on_roster(r.roster())
        elif t == "set_avatar":
            mode = "live" if m.get("mode") == "live" else "off"
            if mode == "live" and not s.livekit_enabled:
                p.client.send_json({"type": "toast", "level": "error",
                                    "text": "The live avatar needs LiveKit settings on the server."})
                return
            if r.agent:
                try:
                    await r.agent.set_avatar(mode)
                except Exception as e:  # noqa: BLE001
                    r.event({"type": "toast", "level": "error", "text": f"Avatar failed: {e}"[:200]})
                    return
            r.avatar = mode
            r.event({"type": "avatar", "mode": mode, "by": p.name})
        elif t == "set_mode":
            mode = "plain" if m.get("mode") == "plain" else "smart"
            r.mode = mode
            if r.agent:
                r.agent.set_mode(mode)
            r.event({"type": "mode", "mode": mode, "by": p.name})
        elif t == "ping":
            p.client.send_json({"type": "pong", "t": m.get("t")})

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Meet 2.0 server")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--https", action="store_true", help="self-signed certificate (for other devices on the LAN)")
    args = ap.parse_args()
    s = config.settings()
    if args.host or args.port:
        s = config.override(**{k: v for k, v in (("host", args.host), ("port", args.port)) if v})
    print("Meet 2.0 settings:\n" + config.describe(s))
    if s.agent != "stub" and not s.boson_api_key:
        print("\nNo BOSON_API_KEY: running the stub agent (no AI). Put the key in .env.")
    import uvicorn
    kw = {}
    if args.https:
        from .tls import self_signed
        kw["ssl_certfile"], kw["ssl_keyfile"] = self_signed(ROOT / "certs")
    scheme = "https" if args.https else "http"
    print(f"\nOpen {scheme}://{'localhost' if s.host in ('127.0.0.1', '0.0.0.0') else s.host}:{s.port}/")
    uvicorn.run(create_app(), host=s.host, port=s.port, log_level="warning", ws_max_size=1 << 20, **kw)


if __name__ == "__main__":
    main()
