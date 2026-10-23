"""Offline: VAP p_now in the silence after COMPLETE turns vs after the unfinished part of HOLD turns.
Picks gate thresholds without spending API calls. Prints p_now at several silence durations per class."""
import os
import sys

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.stimuli import COMPLETE, HOLD, VOICES, get  # noqa: E402
from mmhm.models import _Stream  # noqa: E402

T = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6]


def trace(pcm24):
    st = _Stream("vap", 20, key="p_now")
    x = resample_poly(np.concatenate([np.zeros(12000, np.int16), pcm24, np.zeros(int(1.7 * 24000), np.int16)])
                      .astype(np.float32) / 32768, 2, 3).astype(np.float32)
    z = np.zeros_like(x)
    out = []
    for i in range(0, len(x) - 320, 320):  # 20 ms ticks like the engine
        out.append(st.push(x[i:i + 320], z[i:i + 320]))
    end = (12000 + len(pcm24)) / 24000
    return np.array(out), end


rows = {"complete": [], "hold": []}
for i, c in enumerate(COMPLETE):
    rows["complete"].append(trace(get(c, VOICES[i % 4])))
for i, (a, _) in enumerate(HOLD):
    rows["hold"].append(trace(get(a, VOICES[i % 4])))
print("p_now after the end of speech (20 ms ticks); median [min..max]")
print("silence  " + "  ".join(f"{t:>14}" for t in T))
for k, rs in rows.items():
    vals = np.array([[p[min(len(p) - 1, int((end + t) / 0.02))] for t in T] for p, end in rs])
    print(f"{k:8s} " + "  ".join(f"{np.median(v):.2f} [{v.min():.2f}..{v.max():.2f}]" for v in vals.T))
    np.save(f"out/pnow_{k}.npy", vals)
