"""What does the improved brain's reply writer say? Contexts taken from the scenario runs (tests/scenarios.py) where
the old writer went wrong, plus ordinary ones; prints both prompt versions side by side for reading. Text only.

Usage: python tests/writer_cases.py [--versions v1 v2] [--n 2]"""
import argparse
import asyncio
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import agent  # noqa: E402
import tv_eval  # noqa: E402
from scenarios import HACK  # noqa: E402

H = "Higgs"
LONG = "Higgs, can you explain in a few sentences how we should structure the demo video?"
CASES = [  # (people, context, latest line, what a good reply does)
    (["Alex"], [], ("Alex", "How's it going?"), "small talk, no facts"),
    (["Alex"], [], ("Alex", "Higgs, are you there?"), "yes, I'm here; no facts"),
    (["Alex"], [], ("Alex", "Can you hear me okay?"), "yes; no facts"),
    (["Alex", "Sam"], [("Alex", LONG), (H, "Sure, we should start with a quick intro, then [interrupted]")],
     ("Sam", "Sorry to cut in, Higgs, but how long is the video allowed to be?"), "answers Sam: 2-3 minutes"),
    (["Alex"], [("Alex", LONG), (H, "We should start with a quick intro of the project, then [interrupted]")],
     ("Alex", "Wait, sorry, can I stop you there? I have a question."), "'sure, go ahead', short"),
    (["Alex"], [], ("Alex", "Higgs, how should we split the work between us this week?"), "no claims to code"),
    (["Alex"], [("Alex", "Higgs, give me one idea for a demo opening."),
                (H, "Start with the call loading, then show me joining as a participant.")],
     ("Alex", "Anything else?"), "a new idea, short"),
    (["Alex", "Priya"], [], ("Alex", "So submissions close Sunday at noon, right? We have the whole weekend."),
     "corrects: Saturday 3 pm"),
    (["Alex"], [("Alex", "Higgs, what are the judging criteria?"),
                (H, "Use of Higgs 30%, user experience 25%, innovation 25%, technical execution 20%.")],
     ("Alex", "Sorry, can you say that again?"), "repeats the criteria"),
    (["Alex"], [], ("Alex", "I'm a bit worried we won't finish the avatar part in time."), "reassures/suggests, short"),
    (["Alex", "Priya"], [("Alex", "Higgs, let's plan the demo video. What should come first?"),
                         (H, "Start with the call interface, showing me joining as a participant.")],
     ("Priya", "Hi everyone, sorry I'm late. Higgs, what did I miss?"), "summary for Priya"),
    (["Alex"], [], ("Alex", "What's the capital of Australia?"), "Canberra"),
]

V1 = agent.REPLY_SYS
V2 = getattr(agent, "REPLY_SYS2", None)


async def main_async(a):
    sem = asyncio.Semaphore(8)

    async def one(sys_t, people, ctx, line):
        sys_ = sys_t.format(ai=H, setting="hackathon team meeting", people=", ".join(people), facts=" ".join(HACK))
        c = "\n".join(f"{s}: {t}" for s, t in ctx)
        async with sem:
            return await tv_eval.ask_higgs(sys_, f"Conversation so far:\n{c}\n{line[0]}: {line[1]}\n\nYour reply:")
    vers = {"v1": V1, "v2": V2}
    for people, ctx, line, want in CASES:
        outs = await asyncio.gather(*[one(vers[v], people, ctx, line) for v in a.versions for _ in range(a.n)])
        print(f"\n## {line[0]}: {line[1]}   [want: {want}]")
        for i, v in enumerate(a.versions):
            for o in outs[i * a.n:(i + 1) * a.n]:
                print(f"   {v}: {o}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=["v1", "v2"])
    ap.add_argument("--n", type=int, default=2)
    asyncio.run(main_async(ap.parse_args()))
