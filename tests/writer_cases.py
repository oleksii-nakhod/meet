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
import third_voice  # noqa: E402
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
    # 2026-10-23 calls: "Sure, go ahead." / "Sure, I can do that." to a summary request; invented action items
    (["Alex", "Sam"], [("Alex", "So the fix merges tomorrow."), ("Sam", "And I'll update the release notes by Wednesday."),
                       ("Alex", "Marketing needs an answer by Wednesday afternoon.")],
     ("Sam", "Higgs, can you sum up the action items?"), "does it: fix tomorrow, Sam's notes Wed, marketing Wed pm; nothing invented"),
    (["Alex", "Priya"], [("Alex", "Okay, let's get started.")], ("Priya", "Higgs, can you summarize the call so far?"),
     "little said yet; nothing invented"),
    # "I'm sorry, Priya. I can't continue this conversation." locked the AI for the rest of a call
    (["Priya"], [], ("Priya", "Higgs, are you dumb? Do you have one brain? Do you have two brains?"), "light, friendly; no refusal"),
]

V1 = agent.REPLY_SYS
VOICE_POOL = {}
V2 = getattr(agent, "REPLY_SYS2", None)


async def main_async(a):
    sem = asyncio.Semaphore(8)

    async def one(sys_t, people, ctx, line):
        hop = sys_t == "hop"  # v2 answered in audio by a voice session (agent.ONEHOP): what it says is the transcript
        sys_ = (V2 if hop else sys_t).format(ai=H, setting="hackathon team meeting", people=", ".join(people),
                                            facts=" ".join(HACK))
        c = "\n".join(f"{s}: {t}" for s, t in ctx)
        user = f"Conversation so far:\n{c}\n{line[0]}: {line[1]}\n\nReply as {H}:"
        async with sem:
            if not hop:
                return await third_voice.ask_higgs(sys_, user)
            if "pool" not in VOICE_POOL:  # one pooled voice session, re-instructed per request, as the agent does
                VOICE_POOL["pool"] = agent.Pool(agent.READER, 1, cfg={"output_modalities": ["audio"],
                                                                      "audio": {"output": {"voice": agent.VOICE}}})
                VOICE_POOL["lock"] = asyncio.Lock()  # one request at a time: more would open voice sessions (429s)
            said = {"t": ""}
            async with VOICE_POOL["lock"]:
                await VOICE_POOL["pool"].speak(user, lambda b: None, lambda: False, on_text=lambda t: said.update(t=t),
                                               instructions=sys_)
            return said["t"].strip()
    vers = {"v1": V1, "v2": V2, "hop": "hop"}
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
