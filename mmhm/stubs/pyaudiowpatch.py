"""Stub: MaAI imports pyaudio for mic input; we feed audio ourselves, so it is never used."""
paInt16 = paFloat32 = 8
class PyAudio:
    def __init__(self, *a, **k):
        raise RuntimeError("pyaudio stub: no audio devices on this machine")
