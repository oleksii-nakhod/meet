"""The fact-check on lines where speech recognition misheard a name or code from the fact sheet (2026-10-23 calls:
Higgs "corrected" Mrs. Okafor heard as O'Keffe, PAY-512 heard as K512) and on lines that really are wrong. Real
Higgs calls (3 checks per line, as in a group).

    python tests/fact_cases.py [--n 3]"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import third_voice  # noqa: E402
os.environ.setdefault("MEET_SIGNALS", "0")
os.environ.setdefault("MEET_STT_FALLBACK", "0")
import agent  # noqa: E402

FACTS = ["The landlord is Mrs. Okafor; she said no pets without her written OK.",
         "Rent is $2,400 a month, due on the 1st.", "Ticket PAY-512 (refund bug) is owned by Jordan.",
         "The launch is Thursday at 10 am."]
CASES = [  # (line, should the check flag it)
    ("Should we write an email to Mr. O'Keffe about the cat?", False),
    ("Who is Mrs. Okafor again?", False),
    ("Jordan, where are we on K512, the refund bug?", False),
    ("Is pay five twelve still open?", False),
    ("So rent is twenty-four hundred, right?", False),
    ("So the launch is Friday at ten, great.", True),
    ("Rent is due on the fifteenth, so we have time.", True),
    ("Sam owns the refund bug, PAY-512.", True),
    ("Mrs. Okafor said pets are fine, so let's get the cat.", True),
]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    ok = 0
    for line, want in CASES:
        r = {"target_speaker": "Higgs", "setting": "house meeting", "people": ["Alex", "Priya", "Jordan"], "facts": FACTS,
             "context_turns": [{"speaker": "Alex", "text": "Okay, let's go through the list."}],
             "current_turn": {"speaker": "Priya", "text": agent.fix_terms(line, agent.fact_terms(FACTS), ["Alex", "Priya", "Jordan"], "Higgs")}}
        sys_, user = third_voice.fact_prompt(r)
        outs = await asyncio.gather(*[third_voice.ask_higgs(sys_, user) for _ in range(a.n)])
        hits = [bool((third_voice.parse_json(o) or {}).get("contradicts")) for o in outs]
        flagged = all(hits) and len(hits) == a.n  # decide(): every check has to agree
        ok += flagged == want
        print(f"{'OK' if flagged == want else 'XX'} want {'flag' if want else 'pass'}: {line!r} -> {hits}", flush=True)
    print(f"{ok}/{len(CASES)}")


if __name__ == "__main__":
    asyncio.run(main())
