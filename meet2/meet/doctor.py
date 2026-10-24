"""Check an installation: python -m meet.doctor (make doctor). Prints what is on, off or broken, and why."""
from __future__ import annotations

import asyncio
import importlib
import shutil
import sys

from . import config

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "


def line(status: str, what: str, detail: str = "") -> bool:
    print(f"[{status}] {what}{('  — ' + detail) if detail else ''}")
    return status != FAIL


async def check_boson(s) -> bool:
    from .boson import Models
    m = Models(s.boson_api_key, s.boson_base_url, llm=s.llm_model, stt=s.stt_model, tts=s.tts_model, voice=s.voice,
               language=s.language)
    ok = True
    try:
        reply = await m.chat("Reply with the single word: ready", "Are you ready?", max_tokens=5, timeout=15)
        ok &= line(OK if reply else WARN, f"Boson chat ({s.llm_model})", repr(reply[:20]))
        n = 0
        async for chunk in m.tts_stream("Ready."):
            n += len(chunk)
        ok &= line(OK if n else FAIL, f"Boson speech ({s.tts_model}, voice {s.voice})", f"{n / 48000:.1f} s of audio")
        stt = m.open_stt()
        await asyncio.wait_for(stt.ready.wait(), 15)
        await stt.close()
        line(OK, f"Boson transcription session ({s.stt_model})")
    except Exception as e:  # noqa: BLE001
        ok = line(FAIL, "Boson API", str(e)[:200])
    finally:
        await m.close()
    return ok


def main() -> None:
    print("Meet 2.0 doctor\n")
    good = line(OK if sys.version_info >= (3, 11) else FAIL, "Python", sys.version.split()[0] + " (3.11+ needed)")
    for mod in ("fastapi", "uvicorn", "aiohttp", "numpy", "pydantic", "yaml", "multipart", "pypdf", "docx"):
        try:
            importlib.import_module(mod)
        except ImportError:
            good = line(FAIL, f"package {mod}", "run: pip install -r requirements.txt")
    line(OK, "core packages")
    try:
        s, warnings = config.load()
    except Exception as e:  # noqa: BLE001
        line(FAIL, "configuration", str(e)[:300])
        sys.exit(1)
    for w in warnings:
        line(WARN, "configuration", w)
    if not s.boson_api_key:
        good = line(FAIL, "BOSON_API_KEY", "not set: copy .env.example to .env and add your key (the server would "
                                            "run the stub agent)")
    else:
        good &= asyncio.run(check_boson(s))
    line(OK if shutil.which("ffmpeg") else WARN, "ffmpeg", "found" if shutil.which("ffmpeg") else
         "not found (only needed for media tooling)")
    if s.livekit_enabled:
        try:
            importlib.import_module("livekit.agents.voice.avatar")
            line(OK, "live avatar", f"LiveKit at {s.livekit_url}, avatar {s.boson_avatar_id}")
        except ImportError:
            good = line(FAIL, "live avatar", "LiveKit is configured but its packages are missing: "
                                             "pip install -r requirements-avatar.txt")
    else:
        line(WARN, "live avatar", "off (set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET to enable)")
    line(OK if s.admin_token or s.host in ("127.0.0.1", "localhost") else WARN, "room creation",
         "open to anyone who can reach the server" if not s.admin_token else "needs the admin token")
    print("\nAll good: run `make run`." if good else "\nFix the FAIL lines above.")
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
