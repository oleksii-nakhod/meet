"""Higgs Avatar on the meeting's Higgs tile. The face is Read My Lips' most lip-readable setup: the Boson docs sample
face cropped to fill a 480x640 frame (web/avatar/face.jpg, made once with read_my_lips/portrait.py), voiced by the
same TTS preset (chloe); web/avatar/idle.mp4 is that face listening (read_my_lips/out/fillers, Avatar on room tone).

render() turns one reply's TTS audio into fragmented MP4 chunks as Higgs Avatar streams them (~1.5 s to the first
chunk, then about real time). One render at a time per API key: a second one gets HTTP 429, so wait and retry
(read_my_lips/agent_server.py). Blocking; run it in a thread."""
import base64
import io
import os
import sys
import time

import requests
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mmhm"))
from mmhm.higgs import BASE, HDR, SR  # noqa: E402

MIME = 'video/mp4; codecs="avc1.64001e, mp4a.40.2"'  # what Higgs Avatar returns (video + its own copy of the audio)
SIZE = "480x640"
IDLE_URL = "/avatar/idle.mp4"
FACE = "data:image/jpeg;base64," + base64.b64encode(open(os.path.join(HERE, "web", "avatar", "face.jpg"), "rb").read()).decode()


def render(pcm, on_chunk, cancelled):
    """pcm: int16 at 24 kHz. Calls on_chunk(bytes) for every fMP4 chunk until done or cancelled()."""
    buf = io.BytesIO()
    sf.write(buf, pcm, SR, format="WAV", subtype="PCM_16")
    body = {"model": "higgs-avatar", "ref_image": FACE, "size": SIZE,
            "input": "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()}
    for attempt in range(8):
        if cancelled():
            return
        r = requests.post(f"{BASE}/videos/stream", headers=HDR, json=body, stream=True, timeout=300)
        if r.status_code != 429:
            break
        r.close()
        time.sleep(0.5 + 0.5 * attempt)
    with r:
        r.raise_for_status()
        for chunk in r.iter_content(32768):
            if cancelled():
                return
            if chunk:
                on_chunk(chunk)


FILLER_TEXTS = ["Hmm.", "Let's see."]
FILLER_DIR = os.path.join(HERE, "out", "avatar_fillers")


def filler_assets(voice):
    """[(text, pcm int16, fMP4 bytes)] for the thinking sounds that cover the Avatar's render wait, rendered once
    (TTS in `voice` + this face) and cached in out/avatar_fillers/. Blocking (a few seconds per clip the first time)."""
    import numpy as np
    from mmhm.higgs import trim, tts
    os.makedirs(FILLER_DIR, exist_ok=True)
    out = []
    for i, text in enumerate(FILLER_TEXTS):
        pcm_path, mp4_path = os.path.join(FILLER_DIR, f"{voice}_{i}.pcm"), os.path.join(FILLER_DIR, f"{voice}_{i}.mp4")
        if not (os.path.exists(pcm_path) and os.path.exists(mp4_path)):
            pcm = trim(tts(text, voice=voice)[0])
            pcm = np.concatenate([pcm, np.zeros(SR // 10, np.int16)])
            chunks = []
            render(pcm, chunks.append, lambda: False)
            open(mp4_path + ".tmp", "wb").write(b"".join(chunks))
            open(pcm_path + ".tmp", "wb").write(pcm.tobytes())
            os.replace(mp4_path + ".tmp", mp4_path)
            os.replace(pcm_path + ".tmp", pcm_path)
        out.append((text, np.frombuffer(open(pcm_path, "rb").read(), np.int16), open(mp4_path, "rb").read()))
    return out
