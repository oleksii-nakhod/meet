"""Write-then-read (today: a text writer drafts the reply, a Realtime audio session reads it aloud) against one hop (the
writer's own session answers in audio, its transcript is the text). Warm sessions, real Higgs, ~2 requests per
case per round. Prints, per case: time to first audio after the request for both, when one hop's first words and
whole transcript arrive, and both texts.

    python tests/onehop_bench.py [--rounds 2]"""
import argparse
import asyncio
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MEET_SIGNALS", "0")
import agent  # noqa: E402
from mmhm.higgs import SR  # noqa: E402

SETTING = "a product launch go/no-go meeting"
FACTS = "Launch is planned for Thursday at 10 am. Go criteria: zero P0 bugs, p95 latency under 300 ms."
CASES = [
    ("Sam", "Nova, can you hear us?"),
    ("Alex", "Nova, which go criteria are we failing right now? Latency is 340 on Wednesday."),
    ("Sam", "Nova, what's the capital of Australia?"),
    ("Alex", "Nova, can you sum up the action items so far?"),
    ("Sam", "Nova, can you explain in detail how a bicycle's gears work?"),
]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    sys_ = agent.WRITER.format(ai="Nova", setting=SETTING, people="Sam, Alex", facts=FACTS)
    write = agent.Pool(sys_, 1)
    read = agent.Pool(agent.READER, 1, cfg={"output_modalities": ["audio"], "audio": {"output": {"voice": agent.VOICE}}})
    hop = agent.Pool(sys_, 1, cfg={"output_modalities": ["audio"], "audio": {"output": {"voice": agent.VOICE}}})
    await asyncio.sleep(4.0)  # let the pools open
    rows = []
    for rnd in range(a.rounds):
        for who, line in CASES:
            convo = f"Conversation so far:\n\n{who}: {line}\n\nReply as Nova:"
            # two hops
            t0 = time.perf_counter()
            text = await write.ask(convo)
            t_text = time.perf_counter() - t0
            first = {}
            await read.speak(text, lambda b: first.setdefault("t", time.perf_counter() - t0), lambda: False)
            two = first.get("t")
            # one hop
            s = await hop._take()
            since = s.now()
            t0 = time.perf_counter()
            await s.say_text(convo)
            _, ev = await s.wait("response.created", timeout=10, since=since)
            rid = ev["response"]["id"]
            _, done = await s.wait(lambda e: e.get("type") == "response.done" and e["response"]["id"] == rid, 30, since)
            t_of = lambda typ: next((t for t, e in s.events if t >= since and e.get("type") == typ  # noqa: E731
                                     and e.get("response_id") == rid), None)
            base = t0 - s.t0  # session clock at t0
            ta, tw, td = (t_of("response.output_audio.delta"), t_of("response.output_audio_transcript.delta"),
                          t_of("response.output_audio_transcript.done"))
            hop_text = s.transcript(rid).strip()
            dur = len(s.audio.get(rid, b"")) / 2 / SR
            asyncio.ensure_future(hop._recycle(s, since))
            r = {"line": line, "two_first_audio": round(two, 2) if two else None, "text_done": round(t_text, 2),
                 "one_first_audio": round(ta - base, 2) if ta else None, "one_first_word": round(tw - base, 2) if tw else None,
                 "one_transcript_done": round(td - base, 2) if td else None, "one_audio_s": round(dur, 1),
                 "two_text": text, "one_text": hop_text}
            rows.append(r)
            print(r, flush=True)
            await asyncio.sleep(1.0)
    for k in ("two_first_audio", "one_first_audio", "one_first_word", "one_transcript_done", "text_done"):
        v = [r[k] for r in rows if r[k] is not None]
        print(k, "median", statistics.median(v) if v else None, "all", v)
    for p in (write, read, hop):
        await p.close()


if __name__ == "__main__":
    asyncio.run(main())
