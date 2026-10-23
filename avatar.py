"""Higgs Avatar on the meeting's Higgs tile. The face is the Boson docs sample portrait, uncropped, at 640x640
(web/avatar/face.jpg), voiced by the TTS preset chloe; web/avatar/idle.mp4 is that face listening (Avatar on room
tone, `python avatar.py idle`). The face-filling 480x640 crop from Read My Lips, which lip-reads better but looked too
close on the tile, is kept as face_zoom.jpg / idle_zoom.mp4.

render() turns one reply's TTS audio into fragmented MP4 chunks as Higgs Avatar streams them (~1.5 s to the first
chunk, then about real time). One render at a time per API key: a second one gets HTTP 429, so wait and retry
(the Read My Lips prototype's agent_server.py). Blocking; run it in a thread."""
import base64
import io
import os
import sys
import time

import requests
import soundfile as sf

from mmhm.higgs import BASE, HDR, SR

HERE = os.path.dirname(os.path.abspath(__file__))

MIME = 'video/mp4; codecs="avc1.64001e, mp4a.40.2"'  # what Higgs Avatar returns (video + its own copy of the audio)
SIZE = "640x640"
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
FACE_TAG = "docs"  # in the cached filler names: change it with face.jpg so clips of the old face aren't reused


def filler_assets(voice):
    """[(text, pcm int16, fMP4 bytes)] for the thinking sounds that cover the Avatar's render wait, rendered once
    (TTS in `voice` + this face) and cached in out/avatar_fillers/ (named by FACE_TAG and voice). Blocking (a few seconds per clip the first time)."""
    import numpy as np
    from mmhm.higgs import trim, tts
    os.makedirs(FILLER_DIR, exist_ok=True)
    out = []
    for i, text in enumerate(FILLER_TEXTS):
        pcm_path, mp4_path = os.path.join(FILLER_DIR, f"{FACE_TAG}-{voice}_{i}.pcm"), os.path.join(FILLER_DIR, f"{FACE_TAG}-{voice}_{i}.mp4")
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


def make_idle():
    """web/avatar/idle.mp4: the face listening. Avatar on 4 s of room tone, first 2.8 s (the mouth stays closed), as in
    the Read My Lips prototype's agent_server.py. Run after changing face.jpg."""
    import subprocess
    import tempfile
    import numpy as np
    tone = np.random.default_rng(0).normal(0, 0.0015, SR * 4)
    chunks = []
    render((tone * 32768).astype(np.int16), chunks.append, lambda: False)
    with tempfile.NamedTemporaryFile(suffix=".mp4") as raw:
        raw.write(b"".join(chunks))
        raw.flush()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw.name, "-t", "2.8", "-an", "-c:v", "libx264", "-crf",
                        "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", os.path.join(HERE, "web", "avatar", "idle.mp4")],
                       check=True)


if __name__ == "__main__" and sys.argv[1:] == ["idle"]:
    make_idle()
