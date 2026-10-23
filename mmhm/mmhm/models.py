"""Streaming wrappers around MaAI (Kyoto, MIT licence) for the three signals the engine needs.

- vap    (20 Hz): p_now = probability that the *user* (channel 1) holds the floor right now. Low after a
                  pause means "they're done"; high means "they're only pausing".
- bc_det (12.5 Hz): probability that the user's current sound is a backchannel ("mm-hm", "yeah", "right")
                  rather than an attempt to take the floor. Channel 2 is the agent's audio.
- bc     (20 Hz): probability that a listener would backchannel now (used to say "mm-hm" at the right time).

All models take 16 kHz float32 audio for the user (ch1) and the agent's playback (ch2)."""
import os
import queue
import sys
import threading

import numpy as np

_STUBS = os.path.join(os.path.dirname(__file__), "..", "stubs")
if os.path.isdir(_STUBS) and _STUBS not in sys.path:
    sys.path.append(_STUBS)  # maai imports pyaudio/pygame for devices we never open

import torch  # noqa: E402
from maai import Maai, MaaiInput  # noqa: E402
import maai.models.vap_bc as _vap_bc  # noqa: E402

_vap_bc.print = lambda *a, **k: None  # the bc model prints every frame

torch.set_num_threads(2)

SR16 = 16000


class _Stream:
    def __init__(self, mode, frame_rate, model_type="normal", key=None):
        self.m = Maai(mode=mode, lang="en", frame_rate=frame_rate, model_type=model_type,
                      audio_ch1=MaaiInput.Zero(), audio_ch2=MaaiInput.Zero(), device="cpu")
        self.hop = int(round(SR16 / frame_rate))
        self.key = key
        self.all = {}  # every p_* output of the latest frame (VAP also gives p_future)
        self.b1 = np.zeros(0, np.float32)
        self.b2 = np.zeros(0, np.float32)
        self.value = 0.0

    def push(self, u, s):
        self.b1 = np.concatenate([self.b1, u])
        self.b2 = np.concatenate([self.b2, s])
        while len(self.b1) >= self.hop:
            self.m.process(self.b1[:self.hop], self.b2[:self.hop])
            self.b1, self.b2 = self.b1[self.hop:], self.b2[self.hop:]
            while not self.m.result_dict_queue.empty():
                r = self.m.result_dict_queue.get()
                if self.key in r:
                    self.value = float(np.ravel(r[self.key])[0])
                    self.all = {k: float(np.ravel(v)[0]) for k, v in r.items() if k.startswith("p_") and np.size(v)}
        return self.value


class Signals:
    """Feed matching 16 kHz chunks of user and agent audio; read the latest p_now / p_bcdet / p_bc.

    Each model runs in its own background thread (torch releases the GIL), so `push` never blocks the
    20 ms audio clock; `latest` is at most one model hop (50-80 ms) old."""

    def __init__(self, vap=True, bcdet=True, bc=True):
        self.latest = {}
        self._qs = []
        for name, make in (("p_now", lambda: _Stream("vap", 20, key="p_now")),
                           ("p_bcdet", lambda: _Stream("bc_det", 12.5, model_type="normal-ver2", key="p_bc_det")),
                           ("p_bc", lambda: _Stream("bc", 20, key="p_bc"))):
            if {"p_now": vap, "p_bcdet": bcdet, "p_bc": bc}[name]:
                q = queue.Queue()
                self._qs.append(q)
                self.latest[name] = 0.0
                threading.Thread(target=self._run, args=(name, make(), q), daemon=True).start()

    def _run(self, name, stream, q):
        while True:
            item = q.get()
            if item is None:
                return
            u, s = [item[0]], [item[1]]
            while not q.empty():  # catch up in one go if we fell behind
                nxt = q.get()
                if nxt is None:
                    return
                u.append(nxt[0]); s.append(nxt[1])
            self.latest[name] = stream.push(np.concatenate(u), np.concatenate(s))
            if name == "p_now":
                self.latest["p_future"] = stream.all.get("p_future", 0.0)

    def push(self, u16, s16):
        for q in self._qs:
            q.put((u16, s16))
        return dict(self.latest)

    def close(self):
        for q in self._qs:
            q.put(None)


def _signals_worker(flags, q, shared, ready, parent):
    torch.set_num_threads(1)
    sig = Signals(**flags)
    ready.set()
    keys = ("p_now", "p_bcdet", "p_bc", "p_future")
    while True:
        try:
            item = q.get(timeout=2.0)
        except queue.Empty:
            # A parent killed by a signal (a test harness's terminate(), a crash) never runs multiprocessing's exit
            # handler, so its daemon children live on: 167 orphans held 103 GB on 2026-10-23. Leave with the parent.
            if os.getppid() != parent:
                break
            continue
        if item is None:
            break
        out = sig.push(np.frombuffer(item[0], np.float32), np.frombuffer(item[1], np.float32))
        for i, k in enumerate(keys):
            if k in out:
                shared[i] = out[k]
        shared[4] += 1  # frames handed to the models: the parent's pushes minus this = how far behind we are
    sig.close()


class ProcSignals:
    """Signals in a separate process: one game server can serve many players at once. (All players' model threads
    in one process contend for the GIL: with 2 players the models fell 6.6 s behind live audio, with 8 players 15 s.)
    Same interface as Signals; `latest` values are at most a model hop plus a queue hop old."""

    def __init__(self, vap=True, bcdet=True, bc=True, wait=True):
        import multiprocessing as mp
        # MaAI asks the Hugging Face Hub about its files every time a model is built; with a process per player that
        # is 3 requests per round, and the Hub rate-limits (429). The server's warm-up has already cached the models.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        ctx = mp.get_context("spawn")
        self.flags = {"vap": vap, "bcdet": bcdet, "bc": bc}
        self.q = ctx.Queue()
        self.shared = ctx.Array("d", 5, lock=False)
        self.pushed = 0
        self.ready = ctx.Event()
        self.proc = ctx.Process(target=_signals_worker, args=(self.flags, self.q, self.shared, self.ready, os.getpid()),
                                daemon=True)
        self.proc.start()
        if wait:
            self.ready.wait(120)

    def lag(self):
        """Frames pushed but not yet handed to the models (each model thread may add one hop on top)."""
        return int(self.pushed - self.shared[4])

    def push(self, u16, s16):
        self.pushed += 1
        self.q.put((np.asarray(u16, np.float32).tobytes(), np.asarray(s16, np.float32).tobytes()))
        return {k: self.shared[i] for i, k in enumerate(("p_now", "p_bcdet", "p_bc", "p_future"))
                if self.flags[{"p_now": "vap", "p_bcdet": "bcdet", "p_bc": "bc", "p_future": "vap"}[k]]}

    def close(self):
        self.q.put(None)
