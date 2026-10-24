import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every test: stub agent, no API key, logs and knowledge bases in a temp dir."""
    from meet import config
    monkeypatch.setattr(config, "_settings", None)
    for k in list(os.environ):
        if k.startswith(("MEET_", "BOSON_", "LIVEKIT_")):
            monkeypatch.delenv(k)
    monkeypatch.setenv("MEET_ENV_FILE", str(tmp_path / "none.env"))
    monkeypatch.setenv("MEET_AGENT", "stub")
    monkeypatch.setenv("MEET_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("MEET_KB_DIR", str(tmp_path / "kb"))
    yield tmp_path
    config._settings = None
