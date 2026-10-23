"""Live turn-taking evaluation: response gap and mid-thought cut-ins, plain Higgs vs Mm-hm.

A simulated user (TTS voices from stimuli.py) talks to the engine in real time, one session per condition:
12 COMPLETE turns and 12 HOLD turns (unfinished clause + pause + the rest).
- gap: first audible agent audio minus the end of the user's speech (COMPLETE turns, and HOLD turns that
  weren't cut into).
- cut-in: agent audio started after the unfinished clause and before the user finished (HOLD turns).
Every session is also saved as a stereo WAV (left: user, right: what the speaker played).

Usage: python eval/turns.py [--pause 0.8] [--conds server_vad semantic_vad mmhm] [--n 12]"""
import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval import stimuli  # noqa: E402
from eval.stimuli import get  # noqa: E402
from mmhm.engine import N, Engine, Policy  # noqa: E402
from mmhm.higgs import SR, load_wav, outdir, silence  # noqa: E402

OUT = outdir("turns")
PERSONA = ("You are a friendly person chatting at a hackathon. Reply in one short casual sentence. "
           "Never use lists.")


class Clock:
    """Drives engine.tick() every 20 ms in real time, recording both channels. `noise` (int16 array) is added to
    every mic frame, silences included, like a room that never goes quiet."""

    def __init__(self, engine, noise=None):
        self.e = engine
        self.mic, self.spk = [], []
        self.late = 0
        self.t0 = None
        self.noise, self.npos = noise, 0

    async def feed(self, pcm):
        if self.t0 is None:
            self.t0 = time.perf_counter() - self.e.t
        for i in range(0, len(pcm), N):
            chunk = pcm[i:i + N]
            if len(chunk) < N:
                chunk = np.concatenate([chunk, np.zeros(N - len(chunk), np.int16)])
            if self.noise is not None:
                n = np.take(self.noise, range(self.npos, self.npos + N), mode="wrap")
                self.npos += N
                chunk = np.clip(chunk.astype(np.int32) + n, -32768, 32767).astype(np.int16)
            out = await self.e.tick(chunk)
            self.mic.append(chunk)
            self.spk.append(out)
            d = self.t0 + self.e.t - time.perf_counter()
            if d > 0:
                await asyncio.sleep(d)
            else:
                self.late += 1

    async def idle_until(self, pred, max_s):
        end = self.e.t + max_s
        while self.e.t < end and not pred():
            await self.feed(np.zeros(N, np.int16))

    def save(self, path):
        sf.write(path, np.stack([np.concatenate(self.mic), np.concatenate(self.spk)], 1), SR, subtype="PCM_16")


def ami_items(set2=False):
    """Real speech from AMI headsets (eval/ami_stimuli.py): (kind, file, pause_at, None)."""
    d = os.path.join(os.path.dirname(OUT), "ami_stim2" if set2 else "ami_stim")
    return [(it["kind"], os.path.join(d, it["file"]), it["pause_at"], None)
            for it in json.load(open(os.path.join(d, "items.json")))]


def room_noise(snr_db, level=0.08):
    """Multi-talker babble at `snr_db` below the stimuli's speech level (~0.08 RMS)."""
    import soundfile as sf_
    from scipy.signal import resample_poly
    b, sr = sf_.read(os.path.join(os.path.dirname(__file__), "..", "..", "read_my_lips", "web", "babble.wav"), dtype="float32")
    b = resample_poly(b, SR, sr)
    b = b / (np.sqrt(np.mean(b ** 2)) + 1e-9) * level * 10 ** (-snr_db / 20)
    return (b * 32767).astype(np.int16)


async def run(cond, pause, n, policy, test=False, ami=False, noise_snr=None):
    signals = None
    if cond == "mmhm":
        from mmhm.models import Signals
        signals = Signals(vap=True, bcdet=True, bc=False)
    log = []
    rows = []
    comp, hold, voices = ((stimuli.COMPLETE_TEST, stimuli.HOLD_TEST, stimuli.TEST_VOICES) if test
                          else (stimuli.COMPLETE, stimuli.HOLD, stimuli.VOICES))
    items = [("complete", c, None, voices[i % 4]) for i, c in enumerate(comp[:n])]
    items += [("hold", a, b, voices[i % 4]) for i, (a, b) in enumerate(hold[:n])]
    if ami:
        items = ami_items(set2=(ami == 2))
    items = [items[i] for i in np.random.default_rng(0).permutation(len(items))]
    async with Engine(mode=cond, policy=policy, instructions=PERSONA, signals=signals, log=log) as e:
        clk = Clock(e, room_noise(noise_snr) if noise_snr is not None else None)
        await clk.feed(silence(0.5))
        for kind, a, b, voice in items:
            k0 = len(log)
            t_start = e.t
            if ami:  # one real clip; a HOLD's pause is inside it
                clip = load_wav(a)
                await clk.feed(clip)
                t_end = e.t
                t_a_end = t_start + b if kind == "hold" else t_end
                t_b = t_a_end + 0.8 if kind == "hold" else None
                await clk.idle_until(lambda: any(r["kind"] in ("play_end", "stop") and r["t"] > t_end for r in log[k0:])
                                     and e.state == "listening" and e.current is None, 30)
                await clk.feed(silence(0.6))
                ev = log[k0:]
                plays = [r["t"] for r in ev if r["kind"] == "play"]
                cut = bool(kind == "hold" and any(t_a_end <= p < t_end for p in plays))
                after = [p for p in plays if p >= t_end]
                row = {"cond": cond, "kind": kind, "text": os.path.basename(a), "cut_in": cut if kind == "hold" else None,
                       "gap": round(after[0] - t_end, 3) if after else None, "t_a_end": round(t_a_end, 3),
                       "t_end": round(t_end, 3), "plays": [round(p, 3) for p in plays],
                       "aborts": sum(r["kind"] == "abort" for r in ev),
                       "replies": [r.get("text") for r in ev if r["kind"] == "response_done"]}
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                continue
            pa = get(a, voice)
            await clk.feed(pa)
            t_a_end = e.t
            t_b = None
            if b:
                await clk.feed(silence(pause))
                t_b = e.t
                await clk.feed(get(b, voice))
            t_end = e.t
            # wait for the reply to finish playing (or 10 s)
            await clk.idle_until(lambda: any(r["kind"] in ("play_end", "stop") and r["t"] > t_end for r in log[k0:])
                                 and e.state == "listening" and e.current is None, 30)
            await clk.feed(silence(0.6))
            ev = log[k0:]
            plays = [r["t"] for r in ev if r["kind"] == "play"]
            # cut-in: the agent starts talking while the user's thought is unfinished (after A, before B ends)
            cut = bool(t_b is not None and any(t_a_end <= p < t_end for p in plays))
            after = [p for p in plays if p >= t_end]
            row = {"cond": cond, "kind": kind, "text": a + (" ... " + b if b else ""), "voice": voice,
                   "cut_in": cut if b else None, "gap": round(after[0] - t_end, 3) if after else None,
                   "aborts": sum(r["kind"] == "abort" for r in ev),
                   "t_a_end": round(t_a_end, 3), "t_b": t_b and round(t_b, 3), "t_end": round(t_end, 3),
                   "plays": [round(p, 3) for p in plays],
                   "commits": [r.get("why") for r in ev if r["kind"] == "commit"],
                   "replies": [r.get("text") for r in ev if r["kind"] == "response_done"]}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        tag = f"{cond}_{f'ami{ami}' if ami else pause}" + ("_test" if test else "") + ("_hm" if policy.hold_model and cond == "mmhm" else "") + (f"_snr{noise_snr:g}" if noise_snr is not None else "") + ("" if policy.syntax_hold or cond != "mmhm" else "_nosyntax")
        clk.save(f"{OUT}/session_{tag}.wav")
        with open(f"{OUT}/log_{tag}.jsonl", "w") as f:
            for r in log:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{cond}] late ticks: {clk.late}/{len(clk.mic)}", flush=True)
    if signals is not None:
        signals.close()
    return rows


def summarize(rows):
    out = {}
    for cond in dict.fromkeys(r["cond"] for r in rows):
        rs = [r for r in rows if r["cond"] == cond]
        holds = [r for r in rs if r["kind"] == "hold"]
        gaps = [r["gap"] for r in rs if r["gap"] is not None and not r.get("cut_in")]
        out[cond] = {"median_gap_ms": int(np.median(gaps) * 1000) if gaps else None,
                     "p90_gap_ms": int(np.percentile(gaps, 90) * 1000) if gaps else None,
                     "cut_ins": f"{sum(r['cut_in'] for r in holds)}/{len(holds)}",
                     "no_reply": sum(r["gap"] is None for r in rs)}
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", type=float, default=0.8)
    ap.add_argument("--conds", nargs="+", default=["server_vad", "semantic_vad", "mmhm"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--test", action="store_true", help="use the held-out stimuli")
    ap.add_argument("--no_syntax", action="store_true", help="disable the STT-based mid-sentence hold")
    ap.add_argument("--ami", type=int, default=0, help="1/2: real speech from AMI meetings, set 1 or 2 (eval/ami_stimuli.py)")
    ap.add_argument("--noise_snr", type=float, default=None, help="add room babble at this SNR (dB)")
    ap.add_argument("--hold_model", action="store_true", help="experimental VAP-trend hold model")
    ap.add_argument("--th_start", type=float, default=Policy.th_start)
    ap.add_argument("--th_play", type=float, default=Policy.th_play)
    ap.add_argument("--hold_p", type=float, default=Policy.hold_p)
    ap.add_argument("--give_up", type=float, default=Policy.give_up)
    args = ap.parse_args()
    pol = Policy(th_start=args.th_start, th_play=args.th_play, hold_p=args.hold_p, give_up=args.give_up,
                 syntax_hold=not args.no_syntax, hold_model=args.hold_model)
    rows = []
    for c in args.conds:
        rows += await run(c, args.pause, args.n, pol, args.test, args.ami, args.noise_snr)
    tag = (f"{f'ami{args.ami}' if args.ami else args.pause}{'_test' if args.test else ''}"
           f"{f'_snr{args.noise_snr:g}' if args.noise_snr is not None else ''}_{int(time.time())}")
    with open(f"{OUT}/results_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = summarize(rows)
    print(json.dumps(s, indent=1))
    json.dump({"pause": args.pause, "test": args.test, "policy": vars(pol), "summary": s}, open(f"{OUT}/summary_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
