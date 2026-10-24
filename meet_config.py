"""Small, dependency-free environment loader shared by the server and LiveKit transport."""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ALLOWED = {
    "BOSON_API_KEY",
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "BOSON_AVATAR_ID",
}


def load_environment() -> None:
    """Load local development credentials without ever printing their values.

    Normal deployments should use process environment variables or ``meet/.env``.
    The two ``.api`` locations keep this workspace's existing demo credentials usable.
    """
    configured = os.environ.get("MEET_ENV_FILE")
    paths = [Path(configured)] if configured else []
    paths += [ROOT / ".env", ROOT.parent / ".api", ROOT.parent / "old tuff" / ".api"]
    for path in paths:
        if not path.is_file():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if (key in ALLOWED or key.startswith("MEET_")) and value:   # MEET_* settings too (they were dropped)
                    os.environ.setdefault(key, value)
            elif line.startswith("bai-"):
                os.environ.setdefault("BOSON_API_KEY", line)


def livekit_configured() -> bool:
    load_environment()
    return all(os.environ.get(k) for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"))

