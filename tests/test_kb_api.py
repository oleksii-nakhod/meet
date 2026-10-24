"""The knowledge-base routes on the room server (stub agent, no API calls)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MEET_AGENT"] = "stub"
os.environ.setdefault("BOSON_API_KEY", "unused")

from fastapi.testclient import TestClient  # noqa: E402

import room  # noqa: E402
import server  # noqa: E402
from kb import Library  # noqa: E402
from test_kb import LEASE_PAGES, make_pdf  # noqa: E402


def test_upload_index_and_edit_facts(tmp_path, monkeypatch):
    monkeypatch.setattr(room, "LIBRARY", Library(tmp_path / "kb"))
    monkeypatch.setattr(server, "MODELS", None)              # index only, no fact extraction calls
    with TestClient(server.app) as c:
        rid = c.post("/api/rooms", json={"title": "House", "facts": ["Rent is due on the 1st."]}).json()["id"]
        assert [f["text"] for f in c.get(f"/api/rooms/{rid}/kb").json()["facts"]] == ["Rent is due on the 1st."]
        r = c.post(f"/api/rooms/{rid}/kb/files", files={"file": ("lease.pdf", make_pdf(LEASE_PAGES), "application/pdf")})
        assert r.status_code == 200
        for _ in range(100):
            docs = c.get(f"/api/rooms/{rid}/kb").json()["docs"]
            if docs[0]["status"] in ("ready", "error"):
                break
            time.sleep(0.05)
        assert docs[0]["status"] == "ready" and docs[0]["pages"] == 3
        assert c.post(f"/api/rooms/{rid}/kb/files", files={"file": ("x.exe", b"MZ", "application/x")}).status_code == 400
        st = c.put(f"/api/rooms/{rid}/kb/facts", json={"facts": ["Rent is due on the 1st.", "No pets."]}).json()
        assert [f["text"] for f in st["facts"]] == ["Rent is due on the 1st.", "No pets."]
        assert server.ROOMS[rid].kb.search("pets allowed")[0]["page"] == 3
        assert c.get("/api/config").json()["agent"] == "stub_agent"
