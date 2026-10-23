"""Re-run VAP offline on a recorded live session (stereo: user, speaker) and report p_now in the pauses
after COMPLETE turns vs inside HOLD turns, using item timings from the results file.
Variants: real agent channel vs zeroed agent channel (to see what dialogue context does to p_now)."""
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mmhm.models import _Stream  # noqa: E402

wav, res = sys.argv[1], sys.argv[2]
x, sr = sf.read(wav, dtype="float32")
u = resample_poly(x[:, 0], 2, 3).astype(np.float32)
s = resample_poly(x[:, 1], 2, 3).astype(np.float32)
rows = [json.loads(line) for line in open(res)]
T = [0.1, 0.2, 0.5, 0.8, 1.2]
for name, ch2 in (("with agent channel", s), ("agent channel zeroed", np.zeros_like(s))):
    st = _Stream("vap", 20, key="p_now")
    p = np.array([st.push(u[i:i + 320], ch2[i:i + 320]) for i in range(0, len(u) - 320, 320)])
    print(name)
    for kind, key in (("complete", "t_end"), ("hold", "t_a_end")):
        vals = np.array([[p[min(len(p) - 1, int((r[key] + t) / 0.02))] for t in T] for r in rows if r["kind"] == kind])
        print(f"  {kind:8s} " + "  ".join(f"{t}s:{np.median(v):.2f}[{v.min():.2f}-{v.max():.2f}]" for t, v in zip(T, vals.T)))
