"""Text-only check of the improved brain's speak/stay-silent decision (agent.decide) on hand-written lines.

Each case: who is in the call, the conversation so far, the latest line, and what a good participant would do:
"speak", "silent" or "any" (either is fine). Runs the same decision code and prompts as the live agent (3 judge +
3 fact-check samples), without audio. Higgs answers are cached by prompt in out/judge_cache.jsonl.

Usage: python tests/judge_cases.py [--prompt v4b|v5] [--only 1to1|group] [--fresh]"""
import argparse
import asyncio
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MEET = os.path.dirname(HERE)
sys.path.insert(0, MEET)

AI = "Higgs"
ONE = ["Alex"]
GROUP = ["Alex", "Priya", "Sam"]
H = "Higgs"
IDEA = (H, "We could open the video with a live call where I join as a participant.")

CASES = [
    # ---- 1:1: Alex and Higgs only. A conversation partner answers anything that isn't a filler or self-talk.
    (ONE, [], ("Alex", "Hey, Higgs."), "speak"),
    (ONE, [], ("Alex", "Hi!"), "speak"),
    (ONE, [], ("Alex", "Hello? Can you hear me?"), "speak"),
    (ONE, [], ("Alex", "Higgs?"), "speak"),
    (ONE, [], ("Alex", "Are you there?"), "speak"),
    (ONE, [], ("Alex", "Good morning."), "speak"),
    (ONE, [("Alex", "Hey, Higgs."), (H, "Hey Alex! What's up?")], ("Alex", "How are you doing?"), "speak"),
    (ONE, [], ("Alex", "When do submissions close?"), "speak"),
    (ONE, [], ("Alex", "What's fifteen percent of eighty?"), "speak"),
    (ONE, [], ("Alex", "What's the capital of Australia?"), "speak"),
    (ONE, [("Alex", "Give me one idea for a demo opening."), IDEA], ("Alex", "Anything else?"), "speak"),
    (ONE, [("Alex", "Give me one idea for a demo opening."), IDEA], ("Alex", "Go on."), "speak"),
    (ONE, [("Alex", "Give me one idea for a demo opening."), IDEA], ("Alex", "Tell me more about that."), "speak"),
    (ONE, [("Alex", "Give me one idea for a demo opening."), IDEA], ("Alex", "Why?"), "speak"),
    (ONE, [("Alex", "Give me one idea for a demo opening."), IDEA], ("Alex", "Hmm, I'm not sure about that."), "speak"),
    (ONE, [("Alex", "Give me one idea for a demo opening."), IDEA], ("Alex", "I don't think that works for us."), "speak"),
    (ONE, [("Alex", "Give me one idea for a demo opening."), IDEA], ("Alex", "Sorry, what?"), "speak"),
    (ONE, [("Alex", "Give me one idea for a demo opening."), IDEA], ("Alex", "Can you repeat that?"), "speak"),
    (ONE, [("Alex", "Give me one idea for a demo opening."), IDEA], ("Alex", "What do you mean by participant?"), "speak"),
    (ONE, [("Alex", "Give me one idea for a demo opening."), IDEA], ("Alex", "That's a great idea."), "any"),
    (ONE, [("Alex", "Give me one idea for a demo opening."), IDEA], ("Alex", "Okay, thanks."), "any"),
    (ONE, [("Alex", "Give me one idea for a demo opening."), IDEA], ("Alex", "Okay."), "any"),
    (ONE, [], ("Alex", "I think we should open the video with the problem, not the product."), "speak"),
    (ONE, [], ("Alex", "I'm a bit worried we won't finish the avatar part in time."), "speak"),
    (ONE, [], ("Alex", "Let's talk about the demo video."), "speak"),
    (ONE, [], ("Alex", "Give me your thoughts."), "speak"),
    (ONE, [("Alex", "I think we should cut the intro.")], ("Alex", "What do you think?"), "speak"),
    (ONE, [], ("Alex", "So yeah, that's where we are. Your turn."), "speak"),
    (ONE, [], ("Alex", "Um."), "silent"),
    (ONE, [], ("Alex", "Hmm."), "silent"),
    (ONE, [], ("Alex", "Hmm, where did I put that file."), "silent"),
    (ONE, [], ("Alex", "So the, uh,"), "silent"),
    (ONE, [], ("Alex", "Mom, I'm on a call, give me a minute!"), "silent"),
    (ONE, [("Alex", "Can you look something up?"), (H, "Sure, what do you need?")], ("Alex", "Hold on, let me find it."), "any"),
    (ONE, [("Alex", "Give me one idea."), IDEA], ("Alex", "Mm-hm."), "any"),
    # ---- group: Alex, Priya, Sam and Higgs. Speak when addressed, asked something it knows, or a fact is wrong.
    (GROUP, [], ("Alex", "Hey Higgs."), "speak"),
    (GROUP, [], ("Priya", "Higgs, are you there?"), "speak"),
    (GROUP, [], ("Sam", "Hi Higgs, can you hear us?"), "speak"),
    (GROUP, [], ("Alex", "Higgs, what do you think we should call the project?"), "speak"),
    (GROUP, [], ("Priya", "Hey Higgs, quick one: how much is the user experience worth in judging?"), "speak"),
    (GROUP, [("Alex", "Higgs, how should we open the video?"), IDEA], ("Priya", "Why a live call?"), "speak"),
    (GROUP, [("Alex", "Higgs, how should we open the video?"), IDEA], ("Sam", "Can you say more about that?"), "speak"),
    (GROUP, [("Alex", "Higgs, how should we open the video?"), IDEA], ("Alex", "Higgs, anything else?"), "speak"),
    (GROUP, [], ("Alex", "Wait, when do submissions close again?"), "speak"),
    (GROUP, [], ("Priya", "Does anyone remember how long the video can be?"), "speak"),
    (GROUP, [], ("Sam", "What are the judging criteria again?"), "speak"),
    (GROUP, [], ("Alex", "So submissions close Sunday at noon, right? We have the whole weekend."), "speak"),
    (GROUP, [], ("Priya", "And the video can be up to five minutes, so let's show everything."), "speak"),
    (GROUP, [("Alex", "What do we still need?")], ("Sam", "Can Higgs take notes for us?"), "speak"),
    (GROUP, [], ("Alex", "Higgs, can you summarize what we decided so far?"), "speak"),
    (GROUP, [], ("Alex", "Priya, did you push the new layout?"), "silent"),
    (GROUP, [("Alex", "Priya, did you push the new layout?")], ("Priya", "Yeah, I pushed it this morning."), "silent"),
    (GROUP, [("Alex", "Priya, did you push the new layout?"), ("Priya", "Yeah, I pushed it this morning.")],
     ("Sam", "Cool, I'll check it on my phone after lunch."), "silent"),
    (GROUP, [], ("Alex", "Sam, what do you think?"), "silent"),
    (GROUP, [], ("Priya", "I like the new colors."), "silent"),
    (GROUP, [], ("Sam", "What should we name it?"), "any"),
    (GROUP, [IDEA, ("Priya", "Hmm.")], ("Alex", "Higgs had a good point earlier, actually."), "silent"),
    (GROUP, [("Alex", "Higgs, how should we open the video?"), IDEA], ("Alex", "Thanks, Higgs."), "any"),
    (GROUP, [], ("Alex", "Let me share my screen."), "silent"),
    (GROUP, [], ("Priya", "Can everyone see my screen?"), "silent"),
    (GROUP, [("Alex", "When's the deadline?"), ("Priya", "Saturday at three.")], ("Alex", "Okay, great."), "silent"),
    (GROUP, [], ("Alex", "Priya, when's your flight tomorrow?"), "silent"),
    (GROUP, [("Alex", "Priya, when's your flight tomorrow?")], ("Priya", "At nine, so I'll join from the airport."), "silent"),
    (GROUP, [], ("Sam", "Um, so, I was thinking"), "silent"),
    (GROUP, [], ("Alex", "Mm-hm."), "silent"),
    (GROUP, [], ("Priya", "Ha, yeah."), "silent"),
    (GROUP, [("Sam", "I can record the voice-over tonight.")], ("Alex", "Great, thanks Sam."), "silent"),
    (GROUP, [], ("Priya", "Sam, can you ask Higgs about the deadline?"), "any"),
    (GROUP, [], ("Sam", "I'll grab a coffee, back in five."), "silent"),
    (GROUP, [IDEA], ("Priya", "Mm, okay."), "silent"),
    (GROUP, [IDEA], ("Alex", "I agree with Higgs, let's do that."), "silent"),
]


def point(people, ctx, line, facts):
    return {"target_speaker": AI, "setting": "hackathon team meeting", "people": people, "facts": facts,
            "context_turns": [{"speaker": s, "text": t} for s, t in ctx],
            "current_turn": {"speaker": line[0], "text": line[1]}}


async def main_async(a):
    import agent  # noqa: E402
    from scenarios import HACK
    import tv_eval
    cache_path = os.path.join(MEET, "out", "judge_cache.jsonl")
    cache = {}
    if os.path.exists(cache_path) and not a.fresh:
        for x in open(cache_path):
            d = json.loads(x)
            cache[d["k"]] = d["v"]
    sem = asyncio.Semaphore(12)

    async def ask(sys_, user, i):
        k = hashlib.sha1(json.dumps([sys_, user, i]).encode()).hexdigest()
        if k not in cache:
            async with sem:
                v = await tv_eval.ask_higgs(sys_, user)
            cache[k] = v
            if not v.strip():  # a failed request: don't keep it
                return v
            with open(cache_path, "a") as f:
                f.write(json.dumps({"k": k, "v": v}) + "\n")
        return cache[k]

    cases = CASES
    if a.file:
        cases = [(d["people"], [tuple(x) for x in d["ctx"]], tuple(d["line"]), d["want"])
                 for d in map(json.loads, open(os.path.join(HERE, a.file)))]
    cases = [c for c in cases if not a.only or (a.only == "1to1") == (len(c[0]) == 1)]

    async def one(c):
        people, ctx, line, want = c
        hist = [{"speaker": s_, "text": t} for s_, t in ctx]
        text, kind = agent.prefilter(line[1], AI, hist, people)
        if kind:  # skipped or a backchannel: the live agent doesn't judge it
            return c, False, {"reading": {"prefilter": kind}}
        r = point(people, ctx, (line[0], text), HACK)
        speak, why = await agent.decide(r, lambda _k, s, u, i: ask(s, u, i), samples=3, prompt=a.prompt)
        return c, speak, why

    res = await asyncio.gather(*[one(c) for c in cases])
    n = {"speak": [0, 0], "silent": [0, 0], "any": [0, 0]}
    for (people, ctx, line, want), speak, why in res:
        ok = want == "any" or (want == "speak") == speak
        n[want][0] += ok
        n[want][1] += 1
        if not ok or a.verbose:
            tag = "1:1 " if len(people) == 1 else "grp "
            print(f"{'OK' if ok else 'XX'} {tag}{line[0]}: {line[1]!r:70s} want {want:6s} got {'speak' if speak else 'silent'}"
                  f"  {json.dumps(why.get('reading'))}")
    print("speak when it should", f"{n['speak'][0]}/{n['speak'][1]}", "| silent when it should",
          f"{n['silent'][0]}/{n['silent'][1]}", "| either", f"{n['any'][1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="v4b")
    ap.add_argument("--only")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--file", help="cases from a jsonl file in tests/ (judge_heldout.jsonl)")
    ap.add_argument("-v", "--verbose", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
