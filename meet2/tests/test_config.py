"""Configuration: every advertised key is honoured, unknown ones warn, secrets never print."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from meet import config

ROOT = Path(__file__).resolve().parent.parent


def test_env_file_values_are_loaded_and_typed(tmp_path):
    f = tmp_path / ".env"
    f.write_text("BOSON_API_KEY=bai-xyz\nMEET_PROACTIVE_FACTS=0\nMEET_FACT_DUCK_GAIN=0.2\nMEET_PORT=9000\n"
                 "MEET_TYPO_SETTING=1\n# comment\n")
    s, warnings = config.load(f, environ={})
    assert s.boson_api_key == "bai-xyz" and s.proactive_facts is False and s.fact_duck_gain == 0.2 and s.port == 9000
    assert warnings == ["unknown setting MEET_TYPO_SETTING in .env (ignored)"]


def test_environment_wins_over_the_file(tmp_path):
    f = tmp_path / ".env"
    f.write_text("MEET_PORT=9000\n")
    s, _ = config.load(f, environ={"MEET_PORT": "9100"})
    assert s.port == 9100


def test_invalid_values_are_rejected(tmp_path):
    f = tmp_path / ".env"
    f.write_text("MEET_FACT_DUCK_GAIN=3\n")
    with pytest.raises(ValidationError):
        config.load(f, environ={})


def test_every_key_in_env_example_is_known():
    keys = [line.split("=", 1)[0].strip().lstrip("# ").strip() for line in (ROOT / ".env.example").read_text().splitlines()
            if "=" in line]
    assert keys and all(k in config.ENV_KEYS for k in keys), [k for k in keys if k not in config.ENV_KEYS]


def test_describe_masks_secrets(tmp_path):
    s, _ = config.load(tmp_path / "none", environ={"BOSON_API_KEY": "bai-secret", "LIVEKIT_API_SECRET": "lk"})
    text = config.describe(s)
    assert "bai-secret" not in text and "lk\n" not in text and "set" in text
