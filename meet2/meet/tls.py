"""A self-signed certificate for --https (browsers on other LAN devices need https for the microphone)."""
from __future__ import annotations

import subprocess
from pathlib import Path


def self_signed(directory: Path) -> tuple[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    cert, key = directory / "cert.pem", directory / "key.pem"
    if not (cert.exists() and key.exists()):
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "365", "-subj",
                        "/CN=meet.local", "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)
    return str(cert), str(key)
