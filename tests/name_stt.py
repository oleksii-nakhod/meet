"""How often the AI's name survives transcription, and how often a line without it gains it (real API calls: Higgs TTS
for the clips, Higgs STT as each person's Listener uses it; local faster-whisper with and without the name as a
hotword). Lines that call the AI (several shapes: first word, after a greeting, mid-line, last word, glued to the next
word) and lines without it that have sound-alike words ("Now,", "Noah said", "hex code", "my ex"), in 7 voices, clean
and through a rough laptop mic (band-limited, noise, a little reverb, quieter).

    python tests/name_stt.py collect [--par 6]     # -> out/name_stt/rows.jsonl (cached clips; ~10 min first time)
    python tests/name_stt.py score                  # fix_name on the collected transcripts, no API calls

`score` counts, per AI name: positives where the name is in the repaired Higgs transcript (also with the Whisper second
opinion, agent.name_from_local), and negatives where a name appeared that wasn't said."""
import argparse
import asyncio
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bots  # noqa: E402

MEET = bots.MEET
OUT = os.path.join(MEET, "out", "name_stt")
ELLA = "voice_aca024debbb9ce3f32a478cfd4ba594cac716ec2f4b8479730e7584f9a15dae7"  # voice library: Indian English
MAYA = "voice_35bb36e156fdc13c7de4358d9d16dfc9c8a9b70d9ac81fec03a3ffed8f78b445"  # voice library: Canadian
VOICES = ["marcus", "nora", "jake", "oliver", "eleanor", ELLA, MAYA]
NAMES = ["Nova", "Higgs", "Max"]
POS = ["{n}, can you hear us?", "Hey {n}, how are you?", "Hey, hey {n}, how are you?",
       "{n}, say latency is still at 340 on Wednesday. What do you recommend?", "{n} walk us through the rollout plan.",
       "Okay, {n}. So, can you sum up the action items?", "As long as we ask first, {n}, what do you think?",
       "Who's primary on-call that week, {n}?", "Hi {n}.", "{n} help", "Morning, all. {n}, are you with us?",
       "So {n}, are we go or no go for Thursday?", "{n}, in a few sentences, how would you structure the video?",
       "Okay hey {n} do you know anything about the budget?", "Thanks, {n}.", "What do you think {n}?",
       "I have a question for {n}. How long can the video be?", "{n} what's our plan for the hackathon?"]
NEG = ["Now, what do we do about the budget?", "No, I don't think so.", "Next, let's talk about the venue.",
       "Noah said he'd be late today.", "Can you send me the hex code for the logo?", "We need eggs and milk.",
       "My ex lives in Toronto.", "Hey, how are you?", "Okay, so what's next?", "Nah, that's fine.",
       "Max capacity is fifty people.", "Never mind, let's move on.", "Hicks Street is closed today.",
       "Knock knock, anyone there?", "Novel idea, let's try it.", "Maps are on the shared drive.",
       "Hugs to everyone who helped.", "Honestly, I'm not sure.", "Mexico is lovely in March.", "Next time, then."]


def lines():
    out = [{"ai": n, "text": t.format(n=n), "said": True, "tpl": t} for n in NAMES for t in POS]
    return out + [{"ai": None, "text": t, "said": False, "tpl": t} for t in NEG]


def rough(pcm, seed=0):
    """A laptop mic across a room: 300-3400 Hz, a short reverb tail, -8 dB, noise at ~18 dB SNR."""
    from scipy.signal import butter, lfilter
    rng = np.random.default_rng(seed)
    x = pcm.astype(np.float32) / 32768
    b, a = butter(4, [300 / 12000, 3400 / 12000], btype="band")
    x = lfilter(b, a, x)
    ir = np.zeros(int(0.25 * bots.SR), np.float32)
    ir[0] = 1.0
    ir[1:] = rng.normal(0, 1, len(ir) - 1) * np.exp(-np.arange(1, len(ir)) / (0.05 * bots.SR)) * 0.08
    x = np.convolve(x, ir)[:len(x)] * 10 ** (-8 / 20)
    p = float(np.sqrt(np.mean(x ** 2))) + 1e-9
    x = x + rng.normal(0, p / 10 ** (18 / 20), len(x))
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


async def higgs_stt(s, pcm):
    """One line through a Listener-style transcription session: 300 ms pre-roll, the line, a commit."""
    since = s.now()
    pad = np.zeros(int(0.3 * bots.SR), np.int16)
    x = np.concatenate([pad, pcm, pad])
    for i in range(0, len(x), 4800):
        await s.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(x[i:i + 4800].tobytes()).decode()})
    await s.send({"type": "input_audio_buffer.commit"})
    _, ev = await s.wait("input_audio_buffer.committed", timeout=10, since=since)
    item = ev.get("item_id")
    _, ev = await s.wait(lambda e: e.get("type") == "conversation.item.input_audio_transcription.completed"
                         and e.get("item_id") == item, timeout=20, since=since)
    return (ev.get("transcript") or "").strip()


def whisper_model():
    from faster_whisper import WhisperModel
    return WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=4, num_workers=4)


def whisper(model, pcm, hot=None):
    from scipy.signal import resample_poly
    x = resample_poly(pcm.astype(np.float32) / 32768, 2, 3)
    kw = {"hotwords": hot} if hot else {}
    segs, _ = model.transcribe(x, language="en", beam_size=1, condition_on_previous_text=False,
                               without_timestamps=True, **kw)
    return " ".join(seg.text.strip() for seg in segs).strip()


async def collect(par):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from mmhm.higgs import Session
    os.makedirs(OUT, exist_ok=True)
    jobs = [(ln, v) for ln in lines() for v in VOICES]
    print(f"{len(jobs)} clips", flush=True)
    with ThreadPoolExecutor(6) as ex:  # TTS, cached in out/clips
        pcms = list(ex.map(lambda j: bots.clip(j[0]["text"], j[1]), jobs))
    print("clips ready", flush=True)
    q = asyncio.Queue()
    for k, (j, pcm) in enumerate(zip(jobs, pcms)):
        for cond in ("clean", "rough"):
            q.put_nowait((k, cond, j, pcm if cond == "clean" else rough(pcm, k)))
    rows = []
    model = whisper_model()
    wex = ThreadPoolExecutor(4)

    async def worker():
        s = None
        while not q.empty():
            k, cond, (ln, v), pcm = q.get_nowait()
            for attempt in range(3):
                try:
                    if s is None or s.close_code is not None:
                        s = Session({"instructions": "Transcription only.", "output_modalities": ["text"],
                                     "audio": {"input": {"turn_detection": None,
                                                         "transcription": {"model": "higgs-stt-3.1", "language": "en"}}}})
                        await s.__aenter__()
                    text = await higgs_stt(s, pcm)
                    break
                except Exception as e:  # noqa: BLE001
                    text, s = f"<error {e!r}>", None
                    await asyncio.sleep(2)
            loop = asyncio.get_running_loop()
            names = [ln["ai"]] if ln["ai"] else NAMES
            w_plain = await loop.run_in_executor(wex, whisper, model, pcm, None)
            w_hot = {n: await loop.run_in_executor(wex, whisper, model, pcm, n) for n in names}
            rows.append({**ln, "voice": v[:12], "cond": cond, "higgs": text, "whisper": w_plain, "whisper_hot": w_hot})
            if len(rows) % 50 == 0:
                print(len(rows), flush=True)
    t = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(par)])
    with open(os.path.join(OUT, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows in {time.perf_counter() - t:.0f} s -> {OUT}/rows.jsonl")


def has_name(text, n):
    import re
    return re.search(rf"\b{re.escape(n)}\b", text, re.I) is not None


def score(fixers):
    """fixers: {label: f(row, ai) -> text}. Prints recovery on positives and false names on negatives."""
    rows = [json.loads(x) for x in open(os.path.join(OUT, "rows.jsonl"))]
    for label, fx in fixers.items():
        print(f"== {label}")
        for n in NAMES:
            pos = [r for r in rows if r["ai"] == n]
            neg = [r for r in rows if not r["said"]]
            for cond in ("clean", "rough"):
                p = [r for r in pos if r["cond"] == cond]
                q = [r for r in neg if r["cond"] == cond]
                raw = sum(has_name(r["higgs"], n) for r in p)
                got = sum(has_name(fx(r, n), n) for r in p)
                fp = [r for r in q if has_name(fx(r, n), n) and not has_name(r["higgs"], n)]
                print(f"  {n:6s} {cond:5s}: heard {raw}/{len(p)}, repaired {got}/{len(p)}; false names {len(fp)}/{len(q)}"
                      + (f"  e.g. {[r['higgs'] for r in fp[:3]]}" if fp else ""))


def misses(fx, n=None, cond=None, limit=40):
    rows = [json.loads(x) for x in open(os.path.join(OUT, "rows.jsonl"))]
    for r in rows:
        if r["said"] and (n is None or r["ai"] == n) and (cond is None or r["cond"] == cond) and not has_name(fx(r, r["ai"]), r["ai"]):
            print(f"{r['ai']:6s} {r['cond']:5s} {r['voice']:12s} {r['higgs']!r}  | whisper {r['whisper']!r} hot {r['whisper_hot'].get(r['ai'])!r}")
            limit -= 1
            if not limit:
                return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "score", "misses"])
    ap.add_argument("--par", type=int, default=6)
    ap.add_argument("--name")
    a = ap.parse_args()
    if a.cmd == "collect":
        asyncio.run(collect(a.par))
    else:
        os.environ.setdefault("MEET_SIGNALS", "0")
        os.environ.setdefault("MEET_STT_FALLBACK", "0")
        import agent
        people = ["Alex", "Priya", "Sam"]
        fixers = {"none": lambda r, n: r["higgs"], "fix_name": lambda r, n: agent.fix_name(r["higgs"], n, people)}
        if hasattr(agent, "name_from_local"):
            fixers["fix_name + whisper"] = lambda r, n: agent.name_from_local(
                agent.fix_name(r["higgs"], n, people), r["whisper_hot"].get(n, r["whisper"]), n, people)
        if a.cmd == "score":
            score(fixers)
        else:
            misses(fixers["fix_name"], a.name)
