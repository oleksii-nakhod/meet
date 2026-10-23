"""How many simultaneous Turing Clock players can one machine serve? Runs N players' worth of turn-taking models
(VAP + bc_det + bc, each player's own threads) on synthetic audio for 20 s at real-time pace and reports how far
the models lag behind real time (no API calls). Usage: python tests/capacity.py 1 4 8 16 [--threads T]"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mmhm.models import ProcSignals, Signals  # noqa: E402

USE_PROC = "--proc" in sys.argv


def run(n, secs=20.0):
    players = [(ProcSignals if USE_PROC else Signals)(vap=True, bcdet=True, bc=True) for _ in range(n)]
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    k = 0
    while time.perf_counter() - t0 < secs:
        u = (rng.standard_normal(320) * 0.05).astype(np.float32)
        for p in players:
            p.push(u, np.zeros(320, np.float32))
        k += 1
        d = t0 + k * 0.02 - time.perf_counter()
        if d > 0:
            time.sleep(d)
    # backlog: chunks still queued for the models = how far they lag behind the live audio
    lag = max((p.q.qsize() if USE_PROC else max(q.qsize() for q in p._qs)) for p in players) * 0.02
    for p in players:
        p.close()
    return lag


if __name__ == "__main__":
    import torch  # noqa: F401
    print(f"{os.cpu_count()} CPUs, torch threads per model {torch.get_num_threads()}")
    for n in [int(a) for a in sys.argv[1:] if a.isdigit()] or [1, 4, 8]:
        print(f"{n:3d} players: models lag {run(n):.2f} s behind live audio after 20 s", flush=True)
