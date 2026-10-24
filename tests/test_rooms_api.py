"""Creating rooms with a chosen ID (stub agent, no API calls)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MEET_AGENT"] = "stub"
os.environ.setdefault("BOSON_API_KEY", "unused")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import room  # noqa: E402
import server  # noqa: E402
from kb import Library  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(room, "LIBRARY", Library(tmp_path / "kb"))
    monkeypatch.setattr(server, "MODELS", None)
    monkeypatch.setattr(server, "ROOMS", {})
    monkeypatch.setattr(server, "ADMIN", "")
    with TestClient(server.app) as c:
        yield c


def test_chosen_id_and_conflicts(client):
    r = client.post("/api/rooms", json={"id": "team-sync", "title": "Sync"})
    assert r.status_code == 200 and r.json() == {"id": "team-sync", "link": "/r/team-sync"}
    assert client.get("/api/rooms/team-sync").json()["title"] == "Sync"
    for taken in ("team-sync", "Team-Sync"):
        r = client.post("/api/rooms", json={"id": taken, "title": "Other"})
        assert r.status_code == 409 and "already" in r.json()["error"]
    assert server.ROOMS["team-sync"].brief["title"] == "Sync"  # not replaced
    for bad in ("a b", "x/y", "é", "x" * 65):
        assert client.post("/api/rooms", json={"id": bad}).status_code == 400
    rid = client.post("/api/rooms", json={"title": "Random"}).json()["id"]  # no ID: a random one
    assert rid != "team-sync" and rid in server.ROOMS


def test_room_nobody_joined_is_freed_after_idle(client):
    client.post("/api/rooms", json={"id": "standup", "title": "First"})
    server.ROOMS["standup"].empty_since -= server.IDLE - 5  # created 5 s short of the limit: still there
    assert client.get("/api/rooms/standup").status_code == 200
    server.ROOMS["standup"].empty_since -= 10
    assert client.get("/api/rooms/standup").status_code == 404 and "standup" not in server.ROOMS
    r = client.post("/api/rooms", json={"id": "Standup", "title": "Second"})  # the ID is free again
    assert r.status_code == 200 and client.get("/api/rooms/Standup").json()["title"] == "Second"


def test_room_is_freed_only_once_empty_that_long(client):
    client.post("/api/rooms", json={"id": "sync"})
    rm = server.ROOMS["sync"]
    with client.websocket_connect("/ws/room?room=sync&pid=p1&name=Ann"):
        for _ in range(100):
            if rm.humans:
                break
            time.sleep(0.01)
        assert rm.empty_since is None
        rm.created -= 3600  # however old the room is, it isn't freed while someone is in it
        assert client.get("/api/rooms/sync").status_code == 200
    for _ in range(100):
        if rm.empty_since is not None:
            break
        time.sleep(0.01)
    assert time.time() - rm.empty_since < 5  # the clock starts when the last person leaves
    assert client.get("/api/rooms/sync").status_code == 200
    rm.empty_since -= server.IDLE
    assert client.get("/api/rooms/sync").status_code == 404


def test_no_listing_and_admin_still_honored(client, monkeypatch):
    assert client.get("/api/rooms").status_code != 200  # rooms aren't listed: you join one you were given
    monkeypatch.setattr(server, "ADMIN", "tok")
    assert client.post("/api/rooms", json={"id": "x"}).status_code == 403
    assert client.post("/api/rooms", json={"id": "x", "admin": "tok"}).status_code == 200


def test_cap(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_ROOMS", 1)
    assert client.post("/api/rooms", json={}).status_code == 200
    assert client.post("/api/rooms", json={"id": "two"}).status_code == 503


def test_reused_id_starts_with_its_own_documents(client):
    """Documents stay on disk by room ID; a room created later (another server run) with that ID doesn't get them."""
    client.post("/api/rooms", json={"id": "house", "facts": ["Rent is due on the 1st."]})
    room.LIBRARY.get("house").add_text("Lease", "No pets allowed.")
    server.ROOMS.clear()  # a restart
    client.post("/api/rooms", json={"id": "house", "facts": ["Bins go out on Tuesday."]})
    st = client.get("/api/rooms/house/kb").json()
    assert st["docs"] == [] and [f["text"] for f in st["facts"]] == ["Bins go out on Tuesday."]
