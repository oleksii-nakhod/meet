"""One typed configuration for the whole app.

Values come from the process environment, then from `.env` (or MEET_ENV_FILE). Every key in `.env.example` maps to a
field here; unknown MEET_* keys produce a warning instead of being silently dropped. `describe()` prints the
effective settings with secrets masked.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parent.parent

# env var -> field name. Keys outside this map (and outside LIVEKIT_/BOSON_ credentials) are reported as unknown.
ENV_KEYS = {
    "BOSON_API_KEY": "boson_api_key",
    "BOSON_BASE_URL": "boson_base_url",
    "BOSON_AVATAR_ID": "boson_avatar_id",
    "LIVEKIT_URL": "livekit_url",
    "LIVEKIT_API_KEY": "livekit_api_key",
    "LIVEKIT_API_SECRET": "livekit_api_secret",
    "MEET_HOST": "host",
    "MEET_PORT": "port",
    "MEET_ADMIN": "admin_token",
    "MEET_AGENT": "agent",
    "MEET_AI_NAME": "ai_name",
    "MEET_VOICE": "voice",
    "MEET_LANG": "language",
    "MEET_LLM_MODEL": "llm_model",
    "MEET_STT_MODEL": "stt_model",
    "MEET_TTS_MODEL": "tts_model",
    "MEET_AVATAR_DEFAULT": "avatar_default",
    "MEET_AVATAR_MAX_DURATION": "avatar_max_duration",
    "MEET_PROACTIVE_FACTS": "proactive_facts",
    "MEET_FACT_DUCK_GAIN": "fact_duck_gain",
    "MEET_FACT_MIN_CONFIDENCE": "fact_min_confidence",
    "MEET_FACT_COOLDOWN": "fact_cooldown_s",
    "MEET_CORRECTION_GUARD": "correction_guard_s",
    "MEET_FILLERS_FILE": "fillers_file",
    "MEET_DENOISE": "denoise",
    "MEET_LINE_SILENCE": "line_silence_s",
    "MEET_CLAUSE_SILENCE": "clause_silence_s",
    "MEET_FLOOR_SILENCE": "floor_silence_s",
    "MEET_PLAYOUT_DELAY": "playout_delay_s",
    "MEET_KB_DIR": "kb_dir",
    "MEET_LOG_DIR": "log_dir",
    "MEET_BUDGET": "budget",
    "MEET_SAMPLES": "judge_samples",
}
SECRETS = {"boson_api_key", "livekit_api_key", "livekit_api_secret", "admin_token"}


class Settings(BaseModel):
    boson_api_key: str = ""
    boson_base_url: str = "https://api.boson.ai/v1"
    boson_avatar_id: str = "builtin_claire_v1"
    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""

    host: str = "127.0.0.1"
    port: int = 8600
    admin_token: str = ""
    agent: Literal["higgs", "stub"] = "higgs"
    ai_name: str = "Higgs"
    voice: str = "chloe"
    language: str = "en"
    llm_model: str = "higgs-realtime"
    stt_model: str = "higgs-stt-3.1"
    tts_model: str = "higgs-tts-3"

    avatar_default: Literal["off", "live"] = "off"
    avatar_max_duration: int = 1800

    proactive_facts: bool = True
    fact_duck_gain: float = Field(0.15, ge=0.0, le=1.0)
    fact_min_confidence: float = Field(0.8, ge=0.0, le=1.0)
    fact_cooldown_s: float = Field(20.0, ge=0.0)
    correction_guard_s: float = Field(2.0, ge=0.0)   # a correction can't be talked down for this long

    fillers_file: str = "config/fillers.en.yaml"
    # Boson's server-side noise reduction ("boson") was measured 2026-10-23 to garble transcripts: "Hey Higgs, when
    # is the rent due" -> "What do we do?" at every chunk size, and it lost words in 5 dB white noise where plain STT
    # was word-perfect. Off by default; the browser's suppression + the front-end gate handle noise.
    denoise: Literal["off", "boson"] = "off"
    line_silence_s: float = Field(0.45, gt=0)   # silence that ends a person's turn
    clause_silence_s: float = Field(0.28, gt=0)  # shorter pause: a clause boundary (fact checks run per clause)
    floor_silence_s: float = Field(0.35, gt=0)  # room quiet this long before the AI may start talking
    playout_delay_s: float = Field(0.25, ge=0)  # browser jitter buffer: audio sent -> audio heard

    kb_dir: str = "data/kb"
    log_dir: str = "out/rooms"
    budget: int = 30000          # model requests per server run
    judge_samples: int = 0       # 0 = automatic: 1 in a one-to-one call, 3 in a group

    @property
    def livekit_enabled(self) -> bool:
        return bool(self.livekit_url and self.livekit_api_key and self.livekit_api_secret)

    def path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else ROOT / p


def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def load(env_file: str | os.PathLike | None = None, environ: dict[str, str] | None = None) -> tuple[Settings, list[str]]:
    """(settings, warnings). The process environment wins over the file."""
    environ = dict(os.environ if environ is None else environ)
    path = Path(env_file or environ.get("MEET_ENV_FILE") or ROOT / ".env")
    merged: dict[str, str] = _read_env_file(path) if path.is_file() else {}
    merged.update({k: v for k, v in environ.items() if k in ENV_KEYS})
    warnings = [f"unknown setting {k} in {path.name} (ignored)" for k in merged
                if k.startswith("MEET_") and k not in ENV_KEYS and k != "MEET_ENV_FILE"]
    values = {ENV_KEYS[k]: v for k, v in merged.items() if k in ENV_KEYS and v != ""}
    for k, v in list(values.items()):
        if Settings.model_fields[k].annotation is bool:
            values[k] = v.strip().lower() in ("1", "true", "yes", "on")
    return Settings(**values), warnings


def describe(s: Settings) -> str:
    rows = []
    for name in Settings.model_fields:
        v = getattr(s, name)
        if name in SECRETS:
            v = "set" if v else "(empty)"
        rows.append(f"  {name:22} {v}")
    rows.append(f"  {'livekit_enabled':22} {s.livekit_enabled}")
    return "\n".join(rows)


_settings: Settings | None = None


def settings() -> Settings:
    global _settings
    if _settings is None:
        try:
            _settings, warnings = load()
        except ValidationError as e:
            sys.exit(f"invalid configuration:\n{e}")
        for w in warnings:
            print(f"config: {w}", file=sys.stderr)
    return _settings


def override(**kw) -> Settings:
    """Tests and tools: replace settings for this process."""
    global _settings
    _settings = settings().model_copy(update=kw)
    return _settings
