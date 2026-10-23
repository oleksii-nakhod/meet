"""HTTPS for the local servers: browsers only allow the microphone on https:// pages (or localhost), so players on
phones over the venue Wi-Fi need it. Makes a self-signed certificate once (players accept a warning) and returns
the uvicorn kwargs. For a certificate without warnings, put the server behind a tunnel or reverse proxy instead."""
import os
import subprocess

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "certs")


def uvicorn_ssl():
    os.makedirs(DIR, exist_ok=True)
    key, crt = os.path.join(DIR, "key.pem"), os.path.join(DIR, "cert.pem")
    if not os.path.exists(crt):
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", crt,
                        "-days", "30", "-subj", "/CN=higgs-hackathon"], check=True, capture_output=True)
    return {"ssl_keyfile": key, "ssl_certfile": crt}
