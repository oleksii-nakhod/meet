"""Offline: can the shape of a mic's level over time tell Higgs's own voice coming back (echo) from a quiet person
talking over it? (No API calls.) Higgs's output is TTS speech; the mic hears it back delayed (0.15 s, or 1.5 s with
the live face), band-limited, 25-35 dB down, with noise; or a real AMI interruption at -6 dB on top of that echo.
For every 20 ms frame the level-only rule (agent.ECHO_DB) would call echo, the envelope rule correlates the last
0.6 s of mic levels with Higgs's output levels at each lag up to 2.4 s.

    python tests/echo_offline.py

2026-10-24: no. The level rule suppresses 29-40% of a -6 dB interruption's frames (why soft_barge's int_07 sometimes
doesn't stop Higgs); letting a frame through on a sustained envelope mismatch (corr < 0.2-0.4 for 5-10 frames)
recovers 0% of them while keeping echo out: those frames are the interruption's quietest, where the mic is mostly
Higgs's echo, so their contour does follow Higgs. The level rule stays."""
import glob
import os
import sys

import numpy as np
from scipy.signal import butter, lfilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bots  # noqa: E402

N, SR = 480, 24000
rng = np.random.default_rng(0)


def db(x):
    x = x.astype(np.float32) / 32768
    return 20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9)


def frames_db(pcm):
    return np.array([db(pcm[i:i + N]) for i in range(0, len(pcm) - N + 1, N)])


def room(pcm, delay_s, gain_db):
    b, a = butter(2, [200 / 12000, 3800 / 12000], btype="band")
    x = lfilter(b, a, pcm.astype(np.float32))
    x = np.concatenate([np.zeros(int(delay_s * SR)), x])[:len(pcm)] * 10 ** (gain_db / 20)
    return x


def best_corr(mic_db, out_db, i, win=30, max_lag=120):
    """The best correlation of the mic's last `win` frame levels with the output's, over lags up to max_lag frames."""
    if i < win:
        return None
    m = mic_db[i - win:i]
    if m.std() < 1.0:
        return None
    best = -1.0
    for lag in range(0, max_lag, 2):
        j = i - lag
        if j - win < 0:
            break
        o = out_db[j - win:j]
        if o.std() < 1.0:
            continue
        best = max(best, float(np.corrcoef(m, o)[0, 1]))
    return best


def env_echo(mic_db, out_db, i, thr=0.7):
    c = best_corr(mic_db, out_db, i)
    return None if c is None else c >= thr


def unsuppressed(mic_db, out_db, frames, thr, run):
    """Frames the level rule calls echo that a sustained clear mismatch (corr < thr for `run` frames) lets through."""
    corr = {i: best_corr(mic_db, out_db, i) for i in range(len(mic_db))}
    out, k = set(), 0
    for i in range(len(mic_db)):
        c = corr[i]
        k = k + 1 if (c is not None and c < thr) else 0
        if k >= run:
            out.add(i)
    return np.mean([i in out for i in frames]) if frames else float("nan")


def main():
    clips = sorted(glob.glob(os.path.join(bots.CLIPS, "marcus_*.wav")))[:12]
    outs = np.concatenate([bots.load_wav(c) for c in clips])[: 40 * SR]  # Higgs talking for 40 s
    out_db = frames_db(outs)
    ints = [bots.ami("int_03.wav", -6.0), bots.ami("int_07.wav", -6.0)]
    for delay in (0.15, 1.5):
        for gain in (-25.0, -32.0):
            echo = room(outs, delay, gain) + rng.normal(0, 30, len(outs))
            mic = echo.copy()
            # real interruptions on top, at 10 s and 25 s
            spans = []
            for k, x in zip((10, 25), ints):
                a = k * SR
                mic[a:a + len(x)] += x[: len(mic) - a]
                spans.append((a // N, (a + len(x)) // N))
            mic = np.clip(mic, -32768, 32767).astype(np.int16)
            mdb = frames_db(mic)
            loud = np.array([max(out_db[max(0, i - 130):i + 1]) for i in range(len(out_db))])
            echo_frames = [i for i in range(len(mdb)) if mdb[i] > -50 and not any(a <= i < b for a, b in spans)]
            int_frames = [i for a, b in spans for i in range(a, min(b, len(mdb))) if mdb[i] > -45]
            lvl = lambda i: mdb[i] < loud[i] - 25  # noqa: E731  (the level rule: "echo")
            env = lambda i: env_echo(mdb, out_db, i)  # noqa: E731
            e_lvl = np.mean([lvl(i) for i in echo_frames]) if echo_frames else float("nan")
            i_lvl = np.mean([lvl(i) for i in int_frames]) if int_frames else float("nan")
            e_env = np.mean([bool(env(i)) for i in echo_frames]) if echo_frames else float("nan")
            i_env = np.mean([bool(env(i)) for i in int_frames]) if int_frames else float("nan")
            print(f"delay {delay:.2f} s, echo {gain:+.0f} dB: called echo by level: echo {e_lvl:.2f}, interruption {i_lvl:.2f};"
                  f" by envelope: echo {e_env:.2f}, interruption {i_env:.2f}", flush=True)
            supp_i = [i for i in int_frames if lvl(i)]  # what the level rule wrongly suppresses
            for thr, run in ((0.3, 5), (0.3, 10), (0.2, 5), (0.4, 10)):
                print(f"    let through when corr < {thr} for {run} frames: echo {unsuppressed(mdb, out_db, echo_frames, thr, run):.3f},"
                      f" suppressed interruption frames recovered {unsuppressed(mdb, out_db, supp_i, thr, run):.2f}", flush=True)


if __name__ == "__main__":
    main()
