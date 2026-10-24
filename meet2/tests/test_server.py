"""Rooms and the room socket, with the stub agent (no API calls)."""
import json
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

DATA = Path(__file__).parent / "data"


def speech() -> np.ndarray:
    with wave.open(str(DATA / "question.wav")) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


@pytest.fixture
def client():
    from meet.server import create_app
    with TestClient(create_app()) as c:
        yield c


def make_room(client, **kw):
    r = client.post("/api/rooms", json={"title": "Test", "facts": ["Rent is due on the 1st."], **kw})
    assert r.status_code == 200, r.text
    return r.json()


def recv_until(ws, pred, limit=600):
    """The first message matching pred: JSON events are passed as dicts, binary frames as bytes."""
    for _ in range(limit):
        m = ws.receive()
        if m.get("type") == "websocket.close":
            raise WebSocketDisconnect(m.get("code", 1000))
        msg = json.loads(m["text"]) if m.get("text") else m.get("bytes")
        if msg is not None and pred(msg):
            return msg
    raise AssertionError("expected message never arrived")


def event(kind):
    return lambda m: isinstance(m, dict) and m.get("type") == kind


def test_create_and_info(client):
    room = make_room(client)
    info = client.get(f"/api/rooms/{room['id']}").json()
    assert info["title"] == "Test" and info["people"] == 0 and info["passcode"] is False
    assert client.get(f"/r/{room['id']}").status_code == 200
    assert client.get("/r/nope").status_code == 404
    kb = client.get(f"/api/rooms/{room['id']}/kb").json()
    assert [f["text"] for f in kb["facts"]] == ["Rent is due on the 1st."]


def test_admin_token_and_live_avatar_need_settings():
    from meet import config
    from meet.server import create_app
    config.override(admin_token="s3cret")
    with TestClient(create_app()) as client:
        _admin_checks(client)


def _admin_checks(client):
    assert client.post("/api/rooms", json={"title": "x"}).status_code == 403
    assert client.post("/api/rooms", json={"title": "x", "admin": "s3cret"}).status_code == 200
    r = client.post("/api/rooms", json={"title": "x", "admin": "s3cret", "avatar": "live"})
    assert r.status_code == 400 and "LIVEKIT_URL" in r.json()["detail"]


def test_join_relay_chat_and_stub(client):
    room = make_room(client)
    rid = room["id"]
    with client.websocket_connect(f"/ws/room?room={rid}&pid=a&name=Alex&h={room['host_key']}") as a:
        hello = recv_until(a, lambda e: isinstance(e, dict) and e.get("type") == "welcome")
        assert hello["you"]["slot"] == 1 and hello["you"]["host"] is True
        with client.websocket_connect(f"/ws/room?room={rid}&pid=b&name=Bo") as b:
            assert recv_until(b, lambda e: isinstance(e, dict) and e.get("type") == "welcome")["you"]["host"] is False
            # the stub beeps on slot 0 when someone joins
            beep = recv_until(b, lambda m: isinstance(m, bytes) and m[0] == 1 and m[1] == 0)
            assert len(beep) == 962
            # Alex talks: Bo gets Alex's audio on slot 1, and the stub reports a line
            pcm = np.concatenate([np.zeros(48000, np.int16), speech(), np.zeros(24000, np.int16)])
            for i in range(0, len(pcm) - 479, 480):
                a.send_bytes(bytes([1]) + pcm[i:i + 480].tobytes())
            relayed = recv_until(b, lambda m: isinstance(m, bytes) and m[0] == 1 and m[1] == 1)
            assert len(relayed) == 962
            line = recv_until(b, lambda e: isinstance(e, dict) and e.get("type") == "line")
            assert line["speaker"] == "Alex" and line["text"].startswith("[speech")
            # chat reaches everyone, and the stub answers
            b.send_text(json.dumps({"type": "chat", "text": "hello"}))
            assert recv_until(a, lambda e: isinstance(e, dict) and e.get("type") == "chat" and e["name"] == "Bo")["text"] == "hello"
            assert "(stub)" in recv_until(a, lambda e: isinstance(e, dict) and e.get("type") == "chat" and e["slot"] == 0)["text"]
            # mute shows in the roster
            b.send_text(json.dumps({"type": "mic", "on": False}))
            roster = recv_until(a, lambda e: isinstance(e, dict) and e.get("type") == "roster" and any(not p["mic"] for p in e["people"]))
            assert [p["name"] for p in roster["people"] if not p["mic"]] == ["Bo"]
        assert recv_until(a, lambda e: isinstance(e, dict) and e.get("type") == "leave")["name"] == "Bo"


def test_passcode_full_and_missing(client):
    room = make_room(client, passcode="1234")
    rid = room["id"]
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect(f"/ws/room?room={rid}&pid=x&name=X&k=0000") as ws:
            ws.receive_text()
    assert e.value.code == 4003
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect("/ws/room?room=nope&pid=x&name=X") as ws:
            ws.receive_text()
    assert e.value.code == 4004
    socks = [client.websocket_connect(f"/ws/room?room={rid}&pid=p{i}&name=P{i}&k=1234") for i in range(5)]
    for s in socks:
        s.__enter__()
        recv_until(s, lambda e: isinstance(e, dict) and e.get("type") == "welcome")
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect(f"/ws/room?room={rid}&pid=late&name=Late&k=1234") as ws:
            ws.receive_text()
    assert e.value.code == 4001
    for s in socks:
        s.__exit__(None, None, None)


def test_rejoin_takes_over_the_slot(client):
    rid = make_room(client)["id"]
    with client.websocket_connect(f"/ws/room?room={rid}&pid=same&name=Alex") as first:
        assert recv_until(first, lambda e: isinstance(e, dict) and e.get("type") == "welcome")["you"]["slot"] == 1
        with client.websocket_connect(f"/ws/room?room={rid}&pid=same&name=Alex") as second:
            assert recv_until(second, lambda e: isinstance(e, dict) and e.get("type") == "welcome")["you"]["slot"] == 1
            with pytest.raises(WebSocketDisconnect) as e:
                recv_until(first, lambda m: False, limit=50)
            assert e.value.code == 4000
            assert client.get(f"/api/rooms/{rid}").json()["people"] == 1
