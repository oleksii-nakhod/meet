"""Per-room JSONL logs on a monotonic clock, plus a process-wide request budget."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

T0 = time.monotonic()


def now() -> float:
    """Seconds since the process started (monotonic)."""
    return time.monotonic() - T0


class RoomLog:
    def __init__(self, directory: Path, room_id: str):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{room_id}.jsonl"
        self._f = self.path.open("a", buffering=1)

    def write(self, event: dict[str, Any]) -> None:
        if self._f.closed:
            return
        self._f.write(json.dumps({"t": round(now(), 3), "wall": round(time.time(), 3), **event}, default=str) + "\n")

    def close(self) -> None:
        self._f.close()


class NullLog:
    def write(self, event: dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


class Budget:
    """Counts model requests; once the limit is hit, callers get False and the agent goes quiet."""

    def __init__(self, limit: int):
        self.limit, self.used, self.over = limit, 0, False

    def spend(self, n: int = 1) -> bool:
        if self.used + n > self.limit:
            self.over = True
            return False
        self.used += n
        return True
