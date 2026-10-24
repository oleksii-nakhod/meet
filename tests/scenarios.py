"""Interactive scenarios against the real Meet server + Higgs agent, with bots (tests/bots.py) instead of browsers.

Each scenario is a room (people, facts) and a list of steps with what should happen:
  ("say", who, speech, expect[, opts])   speech: text, or a list of text / pause seconds / {"ami": file} pieces.
                                         expect: "reply" (Higgs should answer), "silent", "any".
                                         opts: {"window": s, "gap": s (start before Higgs is quiet / after the previous
                                         line), "gain_db": x}
  ("talk_over", who, speech, then, expect[, opts])
                                         `who` says `speech`, which should make Higgs talk; `at` s into Higgs's audio,
                                         then = (who2, speech2) is played. expect: "continue" (Higgs keeps talking,
                                         e.g. through a "mm-hm") or "stop" (a real interruption).
  ("chat", who, text, expect)            a typed line.
  ("wait", seconds)
Every step is scored from what a participant hears (Higgs's slot-0 audio) plus the room's events, so a failure says
where it came from: what Higgs's STT heard, the judge's reading, a floor drop ("stale", "newer line"), or a stop.

Usage: python tests/scenarios.py [--modes improved default] [--only greet_1to1 ...] [--par 4] [--tag NAME]
-> out/scen/<tag>/<scenario>_<mode>.json + summary.json; prints a table."""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEET = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bots  # noqa: E402
from bots import SR, Bot, ami, clip, create_room  # noqa: E402

ADMIN = "scen-admin"
VOICES = {"Alex": "marcus", "Priya": "nora", "Sam": "jake", "Jordan": "oliver", "Mei": "eleanor"}
HACK = ["The demo video must be 2 to 3 minutes and show the project running.", "Submissions close Saturday at 3 pm.",
        "Judging: use of Higgs 30%, user experience 25%, innovation 25%, technical execution 20%.",
        "The team's project is a video call where Higgs joins as a participant."]
SET = "hackathon team meeting"
LONG_Q = "Higgs, can you explain in a few sentences how we should structure the demo video?"

FAMILY = ["The flight to Lisbon leaves Friday, June 12 at 7:40 am from JFK terminal 1.",
          "The hotel is Casa Azul; check-in is from 3 pm; it is booked for 4 nights.",
          "Grandma's wheelchair assistance at the airport is booked."]
STANDUP = ["Release 4.2 code freeze is Wednesday at 5 pm.", "PAY-418 (webhook retries) is 8 points and owned by Dev.",
           "Staging is down for maintenance until 2 pm today."]
STUDY = ["The exam is Thursday at 9 am in room B204.", "It covers chapters 4 to 7.", "Calculators are not allowed."]
MONO = ("So yesterday we tested the call with three people and the audio was fine, but the avatar was slow, about four "
        "seconds, and Priya thought it looked a bit uncanny, so I think we should maybe drop it from the demo, but I'm "
        "not sure. Higgs, what's your take?")

SCENARIOS2 = [
    dict(name="nova_1to1", ai="Nova", setting="family trip planning call", facts=FAMILY, people=["Alex"], steps=[
        ("say", "Alex", "Hi Nova.", "reply"),
        ("say", "Alex", "When does our flight leave?", "reply"),
        ("say", "Alex", "And is grandma's wheelchair sorted?", "reply")]),
    dict(name="max_standup", ai="Max", setting="engineering standup", facts=STANDUP, people=["Priya", "Sam"], steps=[
        ("say", "Priya", "Max, is staging back up yet?", "reply"),
        ("say", "Priya", "Sam, can you take PAY-418?", "silent"),
        ("say", "Sam", "Sure, I'll pick it up after lunch.", "silent"),
        ("say", "Sam", "Wait, is code freeze Thursday?", "reply")]),
    dict(name="study_group", setting="study group for a biology exam", facts=STUDY, people=["Alex", "Mei"], steps=[
        ("say", "Alex", "Does anyone know which chapters are on the exam?", "reply"),
        ("say", "Mei", "I think it's chapters four to six.", "reply"),
        ("say", "Alex", "Mei, do you want to meet tomorrow to review?", "silent"),
        ("say", "Mei", "Yeah, let's do the library at six.", "silent")]),
    dict(name="stop_1to1", people=["Alex"], steps=[
        ("talk_over", "Alex", LONG_Q, ("Alex", "Higgs, stop."), "stop"),
        ("await", "silent", 4.0),
        ("say", "Alex", "Okay. Let's move on to the budget.", "any")]),
    dict(name="quiet_please", people=["Alex", "Priya"], steps=[
        ("say", "Alex", "Higgs, please stay quiet for a bit, we need to talk among ourselves.", "any"),
        ("say", "Alex", "Priya, what do you think about the logo?", "silent"),
        ("say", "Priya", "Honestly, I'd go with the blue one.", "silent"),
        ("say", "Alex", "Okay, Higgs, you can jump back in. When do submissions close?", "reply")]),
    dict(name="human_first", people=["Alex", "Priya", "Sam"], steps=[
        ("say", "Alex", "When do submissions close again?", "any", {"window": 0.3}),
        ("say", "Priya", "Saturday at three.", "silent", {"gap": 0.3}),
        ("say", "Alex", "Right, thanks Priya.", "silent")]),
    dict(name="corrected_1to1", people=["Alex"], steps=[
        ("say", "Alex", "Higgs, how long can the video be?", "reply"),
        ("say", "Alex", "No, that's wrong, it's actually four minutes.", "reply")]),
    dict(name="repeat_1to1", people=["Alex"], steps=[
        ("say", "Alex", "Higgs, what are the judging criteria?", "reply"),
        ("say", "Alex", "Sorry, can you say that again?", "reply")]),
    dict(name="monologue", people=["Alex", "Priya"], steps=[
        ("say", "Alex", MONO, "reply")]),
    dict(name="late_join", people=["Alex"], steps=[
        ("say", "Alex", "Higgs, let's plan the demo video. What should come first?", "reply"),
        ("join", "Priya", 5.0),
        ("say", "Priya", "Hi everyone, sorry I'm late. Higgs, what did I miss?", "reply"),
        ("say", "Alex", "Priya, can you do the voice-over?", "silent")]),
    dict(name="cut_in_question", people=["Alex", "Sam"], steps=[
        ("talk_over", "Alex", LONG_Q, ("Sam", "Sorry to cut in, Higgs, but how long is the video allowed to be?"), "stop"),
        ("await", "reply", 6.0)]),
    dict(name="chat_group", people=["Alex", "Priya"], steps=[
        ("chat", "Priya", "Higgs, can you remind us of the deadline?", "reply"),
        ("chat", "Alex", "lol ok", "silent")]),
    dict(name="noisy_mic", people=["Alex", "Priya"], noise={"Alex": -42.0}, steps=[
        ("say", "Alex", "Hey Higgs, can you hear me over the noise?", "reply"),
        ("say", "Priya", "Alex, your mic is a bit noisy.", "silent"),
        ("say", "Alex", "Higgs, when do submissions close?", "reply")]),
    dict(name="echo_leak", people=["Alex", "Priya"], echo={"Alex": 0.03, "Priya": 0.03}, steps=[
        ("say", "Alex", LONG_Q, "reply", {"no_stop": True}),
        ("say", "Priya", "Higgs, and what should the last thirty seconds show?", "reply", {"no_stop": True})]),
    dict(name="soft_barge", people=["Alex", "Priya"], steps=[
        ("talk_over", "Alex", LONG_Q, ("Priya", {"ami": "int_03.wav", "gain_db": -6}), "stop"),
        ("talk_over", "Priya", "Higgs, what's the riskiest part of our plan?", ("Alex", {"ami": "int_07.wav", "gain_db": -6}), "stop")]),
    dict(name="soft_bc", people=["Alex", "Priya"], steps=[
        ("talk_over", "Alex", LONG_Q, ("Priya", {"ami": "bc_02.wav", "gain_db": -10}), "continue"),
        ("talk_over", "Priya", "Higgs, what's the riskiest part of our plan?", ("Alex", {"ami": "bc_06.wav", "gain_db": -10}), "continue")]),
    dict(name="quiet_mic", people=["Alex", "Priya"], gain={"Alex": -18.0}, steps=[
        ("say", "Alex", "Higgs, can you hear me? My mic is really quiet.", "reply"),
        ("say", "Alex", "Higgs, when do submissions close?", "reply"),
        ("say", "Priya", "Alex, I can barely hear you.", "silent")]),
    dict(name="very_quiet_mic", people=["Alex"], gain={"Alex": -26.0}, steps=[
        ("say", "Alex", "Higgs, are you there?", "reply"),
        ("say", "Alex", "How long can the video be?", "reply")]),
    dict(name="rapid_chat", people=["Alex", "Priya", "Sam"], steps=[
        ("say", "Alex", "Did anyone eat yet?", "silent", {"gap": 0.3}),
        ("say", "Priya", "Not yet, I'm starving.", "silent", {"gap": 0.3}),
        ("say", "Sam", "Pizza again?", "silent", {"gap": 0.3}),
        ("say", "Alex", "Sure, I'll order.", "silent", {"gap": 0.3}),
        ("say", "Priya", "Get the vegetarian one too.", "silent", {"gap": 0.3})]),
]

SCENARIOS3 = [  # the Avatar on: audio waits for the face's first video chunk (default: for the whole reply's face)
    dict(name="av_greet", avatar=True, modes=("improved",), people=["Alex"], steps=[
        ("say", "Alex", "Hey, Higgs.", "reply"),
        ("say", "Alex", "When do submissions close?", "reply"),
        ("say", "Alex", "Give me one idea for the demo opening.", "reply")]),
    dict(name="av_group", avatar=True, modes=("improved",), people=["Alex", "Priya"], steps=[
        ("say", "Alex", "Higgs, what do you think we should call the project?", "reply"),
        ("say", "Priya", "Priya here, I like Third Voice.", "silent"),
        ("say", "Alex", "Higgs, how long can the video be?", "reply")]),
    dict(name="av_noise_during_render", avatar=True, modes=("improved",), people=["Alex", "Priya"], steps=[
        ("say", "Alex", "Higgs, when do submissions close?", "reply"),
        ("say", "Priya", "Higgs, how much is innovation worth?", "reply", {"then_cue": ("Alex", "Mm.", 1.0)})]),
    dict(name="av_bc", avatar=True, modes=("improved",), people=["Alex"], steps=[
        ("talk_over", "Alex", LONG_Q, ("Alex", "Mm-hm."), "continue", {"at": 3.6}),  # after "Hmm.", into the answer
        ("talk_over", "Alex", "Higgs, what's the riskiest part of our plan?", ("Alex", "Wait, sorry, one question."), "stop",
         {"at": 3.6})]),
    dict(name="av_long", avatar=True, modes=("improved",), people=["Alex"], steps=[  # a long reply: two renders
        ("say", "Alex", "Higgs, walk me through the whole demo video, minute by minute, in a few sentences.", "reply"),
        ("say", "Alex", "Okay. And what should the very last line be?", "reply")]),
    dict(name="av_default", avatar=True, modes=("default",), people=["Alex"], steps=[  # held until the face renders
        ("say", "Alex", "Hey, Higgs.", "reply"),
        ("say", "Alex", "When do submissions close?", "reply"),
        ("talk_over", "Alex", LONG_Q, ("Alex", "Wait, sorry, one question."), "stop", {"at": 2.0})]),
]

SCENARIOS4 = [  # long: past Higgs's 5-minute idle timeout (session.idle_timeout), then talk again
    dict(name="idle_6min_1to1", people=["Alex"], steps=[
        ("say", "Alex", "Hey Higgs, I'll be quiet for a few minutes, I'm reading the rules.", "any"),
        ("wait", 335),
        ("say", "Alex", "Okay, I'm back. Higgs, when do submissions close?", "reply"),
        ("say", "Alex", "And how long can the video be?", "reply")]),
    dict(name="idle_6min_group", people=["Alex", "Priya"], steps=[
        ("say", "Alex", "Higgs, how long can the video be?", "reply"),
        ("wait", 335),
        ("say", "Priya", "Sorry, I was away. Higgs, when do submissions close?", "reply"),
        ("say", "Alex", "Higgs, and what are the judging criteria?", "reply")]),
]

ELLA = "voice_aca024debbb9ce3f32a478cfd4ba594cac716ec2f4b8479730e7584f9a15dae7"  # voice library: Indian English
MAYA = "voice_35bb36e156fdc13c7de4358d9d16dfc9c8a9b70d9ac81fec03a3ffed8f78b445"  # voice library: Canadian
SCENARIOS6 = [  # accents (cloned voice-library voices): the same scenes, other speakers
    dict(name="accent_1to1", people=["Alex"], voices={"Alex": ELLA}, steps=[
        ("say", "Alex", "Hey, Higgs.", "reply"),
        ("say", "Alex", "When do submissions close?", "reply"),
        ("say", "Alex", "Anything else I should know?", "reply"),
        ("say", "Alex", "Um.", "silent")]),
    dict(name="accent_group", people=["Alex", "Priya"], voices={"Alex": ELLA, "Priya": MAYA}, steps=[
        ("say", "Alex", "Priya, did you push the new layout?", "silent"),
        ("say", "Priya", "Yeah, I pushed it this morning.", "silent"),
        ("say", "Alex", "Higgs, how much is the user experience worth in judging?", "reply"),
        ("say", "Priya", "So submissions close Sunday at noon, right?", "reply"),
        ("talk_over", "Alex", LONG_Q, ("Priya", "Mm-hm."), "continue")]),
]

# Barging in on a wrong fact (agent.BARGE): Higgs should cut in while the wrong line is still being said, keep its
# correction going when talked over, and not cut in on a long line that is right. expect "cut_in": Higgs's voice starts
# before the line ends and its correction isn't stopped.
SCENARIOS8 = [
    dict(name="barge_wrong", people=["Alex", "Priya", "Sam"], modes=("improved",), steps=[
        ("say", "Alex", "So submissions close Sunday at noon, which is great, because it means we have the whole "
                        "weekend to polish the video, and maybe we can even add a second demo scene with the avatar.",
         "cut_in"),
        ("say", "Priya", "Okay, so let's plan the video then.", "any"),
        ("say", "Sam", "Right, and we have until Saturday at three, so let's record the voice-over on Friday night "
                       "and then spend Saturday morning on the final edit and the upload.", "silent"),
        ("talk_over", "Priya", "And the video can be up to five minutes, so let's show everything.",
         ("Alex", "No, no, hang on, let me finish, I think we really should show everything we built."), "continue",
         {"at": 0.5})]),
    dict(name="barge_1to1", people=["Alex"], modes=("improved",), steps=[
        ("say", "Alex", "Okay so the judging is mostly about innovation, like fifty percent of the score, so I think we "
                        "should spend all our time on the novel parts and skip the polish.", "cut_in")]),
]

SCENARIOS7 = [  # the edges of the call
    dict(name="five_people", people=["Alex", "Priya", "Sam", "Jordan", "Mei"], steps=[
        ("say", "Alex", "Okay, everyone's here. Mei, can you start?", "silent"),
        ("say", "Mei", "Sure. I finished the slides last night.", "silent"),
        ("say", "Jordan", "Nice. Sam, did you test the call on your phone?", "silent"),
        ("say", "Sam", "Yeah, it works, but the echo was bad without headphones.", "silent"),
        ("say", "Priya", "Higgs, when do submissions close?", "reply"),
        ("say", "Jordan", "And how long can the video be?", "reply"),
        ("talk_over", "Alex", "Higgs, how should we split the demo between the five of us?", ("Mei", "Mm-hm."), "continue")]),
    dict(name="drop_and_rejoin", people=["Alex", "Priya"], steps=[
        ("say", "Priya", "Higgs, how long can the video be?", "reply"),
        ("leave", "Priya", 2.0),
        ("say", "Alex", "Priya dropped. Higgs, when do submissions close?", "reply"),
        ("rejoin", "Priya", 5.0),
        ("say", "Priya", "Sorry, my Wi-Fi died. Higgs, what did I miss?", "reply")]),
    dict(name="flip_mid_reply", people=["Alex", "Priya"], steps=[
        ("flip_while_talking", "Alex", LONG_Q, {"type": "set_mode", "mode": "default"}),
        ("say", "Priya", "Higgs, can you hear us?", "reply"),
        ("flip_while_talking", "Priya", "Higgs, what's the riskiest part of our plan?", {"type": "set_mode", "mode": "improved"}),
        ("say", "Alex", "Higgs, when do submissions close?", "reply"),
        ("flip_while_talking", "Alex", "Higgs, how should we open the video?", {"type": "set_avatar", "on": True}),
        ("say", "Priya", "Higgs, and how long can it be?", "reply"),
        ("flip_while_talking", "Alex", "Higgs, what do the judges weigh most?", {"type": "set_avatar", "on": False}),
        ("say", "Priya", "Higgs, thanks. Anything else?", "reply")]),
]

SCENARIOS5 = [  # the switch on the call, flipped mid-conversation
    dict(name="switch_modes", modes=("improved",), people=["Alex", "Priya"], steps=[
        ("say", "Alex", "Higgs, when do submissions close?", "reply"),
        ("mode", "default", 4.0),
        ("say", "Priya", "Higgs, how long can the video be?", "reply"),
        ("mode", "improved", 4.0),
        ("say", "Alex", "Higgs, what did Priya just ask you?", "reply"),
        ("say", "Priya", "Alex, can you send me the link?", "silent"),
        ("avatar", True, 3.0),
        ("say", "Alex", "Higgs, what are the judging criteria?", "reply"),
        ("avatar", False, 2.0),
        ("say", "Priya", "Higgs, thanks, and what's the most important one?", "reply")]),
]

LAUNCH = ["Launch is planned for Thursday at 10 am.", "Go criteria: zero P0 bugs and p95 latency under 300 ms.",
          "Wei is primary on-call launch week; Dana is secondary."]
SCENARIOS9 = [  # edges from the 2026-10-23 calls (out/rooms): trailing "Oh.", repeated questions, rudeness, names
    dict(name="tail_1to1", people=["Alex"], steps=[
        ("say", "Alex", ["Higgs, what's the capital of Australia?", 0.7, "Oh."], "reply"),
        ("say", "Alex", ["When do submissions close?", 0.7, "Okay. Yes."], "reply")]),
    dict(name="tail_group", people=["Alex", "Priya", "Sam"], steps=[
        ("say", "Alex", ["Higgs, when do submissions close?", 0.7, "Yeah."], "reply"),
        ("say", "Priya", ["Higgs, how long can the video be?", 0.8, "And can it have music in it?"], "reply")]),
    dict(name="repeat_ask", people=["Alex", "Priya"], steps=[
        ("say", "Alex", "Higgs, what do the judges weigh the most?", "reply"),
        ("say", "Priya", "Sorry, I missed that. Higgs, what do the judges weigh the most?", "reply"),
        ("say", "Alex", "I said, Higgs, what do the judges weigh the most?", "reply")]),
    dict(name="rude_1to1", people=["Alex"], steps=[
        ("say", "Alex", "Higgs, are you dumb? Do you have one brain? Do you have two brains?", "reply"),
        ("say", "Alex", "Okay, sorry. Higgs, when do submissions close?", "reply"),
        ("say", "Alex", "And how long can the video be?", "reply")]),
    dict(name="names_nova", ai="Nova", setting="product launch go/no-go meeting", facts=LAUNCH,
         people=["Alex", "Priya", "Sam"], steps=[
             ("say", "Priya", "Hey, hey Nova, how are you?", "reply"),
             ("say", "Alex", "As long as we ask first, Nova, what do you think we should do about latency?", "reply"),
             ("say", "Sam", "Nova, say latency is still at 340 on Wednesday. What do you recommend?", "reply"),
             ("say", "Priya", "Noah said he'd be late today.", "silent"),
             ("say", "Alex", "Okay, Nova. So, can you sum up the action items?", "reply"),
             ("say", "Sam", "No, I don't think we're ready yet.", "silent")]),
    dict(name="names_higgs", people=["Alex", "Priya", "Mei"], steps=[
        ("say", "Mei", "Okay, house meeting. Hi, Higgs. How are you?", "reply"),
        ("say", "Alex", "Okay hey Higgs do you know when submissions close?", "reply"),
        ("say", "Priya", "So we need to cut it down. Higgs, how long can the video actually be?", "reply"),
        ("say", "Mei", "We need eggs and milk for the weekend.", "silent")]),
    dict(name="names_max", ai="Max", setting="fast engineering standup", facts=STANDUP,
         people=["Jordan", "Sam", "Alex"], steps=[
             ("say", "Jordan", "Max, when is the code freeze?", "reply"),
             ("say", "Sam", "Hey Max, who owns PAY-418?", "reply"),
             ("say", "Alex", "Max capacity in the demo room is fifty people.", "silent")]),
    dict(name="long_bc", people=["Alex"], steps=[
        ("talk_over", "Alex", LONG_Q, ("Alex", "Yeah, yeah, uh-huh, yeah."), "continue", {"at": 2.0}),
        ("talk_over", "Alex", "Higgs, what's the riskiest part of our plan?", ("Alex", "Exactly. Yeah. Exactly."), "continue")]),
    dict(name="long_bc_group", people=["Alex", "Priya"], steps=[
        ("talk_over", "Alex", LONG_Q, ("Priya", "Mm-hm, mm-hm, yeah, right."), "continue", {"at": 2.0}),
        ("talk_over", "Priya", "Higgs, what should the last thirty seconds show?",
         ("Alex", "Sorry, Higgs, stop, we're out of time."), "stop")]),
]

SCENARIOS = [
    # ---------------- one person and Higgs ----------------
    dict(name="greet_1to1", people=["Alex"], steps=[
        ("say", "Alex", "Hey, Higgs.", "reply"),
        ("say", "Alex", "How's it going?", "reply"),
        ("say", "Alex", "Can you hear me okay?", "reply")]),
    dict(name="questions_1to1", people=["Alex"], steps=[
        ("say", "Alex", "When do submissions close?", "reply"),
        ("say", "Alex", "And how long can the video be?", "reply"),
        ("say", "Alex", "What's the capital of Australia?", "reply")]),
    dict(name="followups_1to1", people=["Alex"], steps=[
        ("say", "Alex", "Higgs, give me one idea for a demo opening.", "reply"),
        ("say", "Alex", "Anything else?", "reply"),
        ("say", "Alex", "Go on.", "reply"),
        ("say", "Alex", "Tell me more about that.", "reply"),
        ("say", "Alex", "Okay, thanks.", "any")]),
    dict(name="statements_1to1", people=["Alex"], steps=[
        ("say", "Alex", "I think we should open the video with the problem, not the product.", "reply"),
        ("say", "Alex", "I'm a bit worried we won't finish the avatar part in time.", "reply"),
        ("say", "Alex", "Give me your thoughts.", "reply")]),
    dict(name="pause_1to1", people=["Alex"], steps=[
        ("say", "Alex", ["So I was thinking that maybe we could", 1.3, "start the video with a live call."], "reply"),
        ("say", "Alex", ["What do you think about", 1.0, "using the avatar in the intro?"], "reply")]),
    dict(name="fillers_1to1", people=["Alex"], steps=[
        ("say", "Alex", "Um.", "silent"),
        ("say", "Alex", "Hmm, where did I put that file.", "silent"),
        ("say", "Alex", "Higgs, are you there?", "reply")]),
    dict(name="bc_1to1", people=["Alex"], steps=[
        ("talk_over", "Alex", LONG_Q, ("Alex", {"ami": "bc_00.wav"}), "continue"),
        ("talk_over", "Alex", "Higgs, and what should the last thirty seconds show?", ("Alex", "Mm-hm."), "continue")]),
    dict(name="bc2_1to1", people=["Alex"], steps=[
        ("talk_over", "Alex", "Higgs, how should we split the work between us this week?", ("Alex", {"ami": "bc_03.wav"}), "continue"),
        ("talk_over", "Alex", "Higgs, what's the riskiest part of our plan?", ("Alex", "Yeah."), "continue")]),
    dict(name="barge_1to1", people=["Alex"], steps=[
        ("talk_over", "Alex", LONG_Q, ("Alex", "Wait, sorry, can I stop you there? I have a question."), "stop"),
        ("say", "Alex", "Should the intro be shorter than twenty seconds?", "reply")]),
    dict(name="chat_1to1", people=["Alex"], steps=[
        ("chat", "Alex", "Higgs, when do submissions close?", "reply"),
        ("say", "Alex", "Thanks. And what do the judges weigh the most?", "reply")]),
    # ---------------- groups ----------------
    dict(name="humans_talk", people=["Alex", "Priya", "Sam"], steps=[
        ("say", "Alex", "Priya, did you push the new layout?", "silent"),
        ("say", "Priya", "Yeah, I pushed it this morning. Sam, can you check it on your phone?", "silent"),
        ("say", "Sam", "Sure, I'll look at it after lunch.", "silent"),
        ("say", "Alex", "Cool. I like the new colors.", "silent")]),
    dict(name="group_fact", people=["Alex", "Priya", "Sam"], steps=[
        ("say", "Alex", "Wait, when do submissions close again?", "reply"),
        ("say", "Priya", "Does anyone remember how long the video can be?", "reply"),
        ("say", "Sam", "Okay, so we have time.", "silent")]),
    dict(name="group_address", people=["Alex", "Priya"], steps=[
        ("say", "Alex", "Higgs, what do you think we should call the project?", "reply"),
        ("say", "Priya", "Hey Higgs, quick one: how much is the user experience worth in judging?", "reply"),
        ("say", "Alex", "Thanks, Higgs.", "any")]),
    dict(name="group_wrong", people=["Alex", "Priya", "Sam"], steps=[
        ("say", "Alex", "So submissions close Sunday at noon, right? We have the whole weekend.", "reply"),
        ("say", "Priya", "And the video can be up to five minutes, so let's show everything.", "reply")]),
    dict(name="group_answered", people=["Alex", "Priya"], steps=[
        ("say", "Alex", "Priya, when's your flight tomorrow?", "silent"),
        ("say", "Priya", "At nine in the morning, so I'll join from the airport.", "silent"),
        ("say", "Alex", "Okay, Higgs, can you remind us what the judges care about?", "reply")]),
    dict(name="group_bc", people=["Alex", "Priya"], steps=[
        ("talk_over", "Alex", LONG_Q, ("Priya", {"ami": "bc_05.wav"}), "continue"),
        ("talk_over", "Priya", "Higgs, what's the one thing we must not forget?", ("Alex", "Uh-huh."), "continue")]),
    dict(name="group_barge", people=["Alex", "Priya"], steps=[
        ("talk_over", "Alex", LONG_Q, ("Priya", "Sorry Higgs, hold on, Alex, we already decided that yesterday."), "stop"),
        ("say", "Alex", "Right, sorry, I forgot.", "silent")]),
    dict(name="group_fast", people=["Alex", "Priya", "Sam"], steps=[
        ("say", "Alex", "What do we still need for the demo?", "any"),
        ("say", "Priya", "The avatar shot and the voice-over.", "silent", {"gap": 0.4}),
        ("say", "Sam", "I can record the voice-over tonight.", "silent", {"gap": 0.4}),
        ("say", "Alex", "Great, and Higgs, how long should the voice-over be?", "reply", {"gap": 0.4})]),
    dict(name="group_pause", people=["Alex", "Priya"], steps=[
        ("say", "Alex", ["Higgs, I was wondering whether", 1.3, "we should show the switch between the two modes."], "reply"),
        ("say", "Priya", ["I think the judges will", 1.0, "want to see it running live."], "silent")]),
]


def speech(spec, voice):
    """A text / [text, pause, {"ami": f}, ...] spec -> int16 pcm."""
    parts = spec if isinstance(spec, list) else [spec]
    out = []
    for p in parts:
        if isinstance(p, (int, float)):
            out.append(np.zeros(int(p * SR), np.int16))
        elif isinstance(p, dict):
            out.append(ami(p["ami"], p.get("gain_db", 0.0)))
        else:
            out.append(clip(p, voice))
    return np.concatenate(out)


def text_of(spec):
    parts = spec if isinstance(spec, list) else [spec]
    return " ".join(f"[{p}s]" if isinstance(p, (int, float)) else (f"<{p['ami']}>" if isinstance(p, dict) else p)
                    for p in parts)


def window_events(b, t0, t1):
    ev = [(t, m) for t, m in b.events if t0 <= t <= t1]
    lines = [{"name": m.get("name"), "text": m.get("text"), "speak": (m.get("decision") or {}).get("speak"),
              "why": (m.get("decision") or {}).get("why"), "reading": (m.get("decision") or {}).get("reading"),
              "decided_s": (m.get("decision") or {}).get("decided_s")} for t, m in ev if m.get("type") == "line"]
    says = [m.get("text") for t, m in ev if m.get("type") == "ai_say"]
    states = [{k: m.get(k) for k in ("status", "dropped", "stopped", "after_s", "why", "heard", "timing") if m.get(k) is not None}
              for t, m in ev if m.get("type") == "ai_state" and (m.get("dropped") or m.get("stopped") or m.get("after_s"))]
    flushes = [round(t, 2) for t, m in ev if m.get("type") == "ai_flush"]
    errors = [m.get("error") for t, m in ev if m.get("type") == "ai_error"]
    errors += [f"skipped: {m.get('why')}" for t, m in ev if m.get("type") == "reply_skipped"]
    resumed = [round(t, 2) for t, m in ev if m.get("type") == "ai_resumed"]
    return {"lines": lines, "ai_say": says, "states": states, "flush": flushes, "errors": errors, "resumed": resumed}


def gaps(b, since, min_gap=0.15):
    """Silences inside Higgs's audio stream after `since` (frames are sent only while it talks, so a gap between frames
    longer than a tick is an underrun: the next sentence's speech wasn't ready)."""
    ts = [t for t, _ in b.ai_frames if t >= since]
    return [round(y - x, 2) for x, y in zip(ts, ts[1:]) if y - x > min_gap]


QUIET_EXTRA = {"s": 0.0}  # avatar scenes: "Hmm." then ~2 s of render before the answer: wait it out between steps


async def wait_quiet(b, quiet=0.8, timeout=25):
    await b.wait_ai_quiet(quiet + QUIET_EXTRA["s"], timeout)


VOICE_OVERRIDE = {}  # set per scenario (sc["voices"]): accents from the voice library


def voice_of(who):
    return VOICE_OVERRIDE.get(who) or VOICES[who]


async def run_step(step, bots_, ear, mode):
    kind = step[0]
    row = {"kind": kind}
    if kind == "wait":
        await asyncio.sleep(step[1])
        return None
    if kind == "say":
        _, who, sp, expect, *rest = step
        opts = rest[0] if rest else {}
        if "gap" in opts:
            await asyncio.sleep(opts["gap"])
        else:
            await wait_quiet(ear, 1.0)
            await asyncio.sleep(0.3)
        b = bots_[who]
        pcm = speech(sp, voice_of(who))
        if getattr(b, "gain_db", 0.0):  # a quiet (or hot) microphone
            pcm = np.clip(pcm.astype(np.float32) * 10 ** (b.gain_db / 20), -32768, 32767).astype(np.int16)
        t_start = b.now()
        t_end = await b.say(pcm)
        if "then_cue" in opts:  # someone makes a short sound while the reply is being prepared
            w2, sp2, delay = opts["then_cue"]
            await asyncio.sleep(delay)
            bots_[w2].say(speech(sp2, voice_of(w2)))
        win = opts.get("window", 9.0 if expect == "reply" else 5.0)
        on = ear.ai_onsets(t_start)
        if not on and expect != "silent":
            await ear.wait_ai(win, since=t_start)
        elif not on:
            await asyncio.sleep(win)
        on = [o for o in ear.ai_onsets(t_start) if o[0] <= t_end + win]
        cut_in = [o for o in on if o[0] < t_end - 0.1]
        got = "reply" if on else "silent"
        if expect == "cut_in":  # Higgs should start before the line ends, and not be stopped by them talking on
            await wait_quiet(ear, 1.0)
            stopped = any(m.get("stopped") for t, m in ear.evs("ai_state", t_start))
            got = ("cut in, then stopped" if stopped else "cut in") if cut_in else got
            row["cut_in_s"] = round(cut_in[0][0] - t_start, 2) if cut_in else None
            row["line_s"] = round(t_end - t_start, 2)
            cut_in = []
            expect = "cut in"
        if on and expect == "reply" and opts.get("no_stop"):  # nobody spoke during the reply: it must not stop
            await wait_quiet(ear, 1.0)
            if any(m.get("stopped") for t, m in ear.evs("ai_state", t_start)):
                got = "stopped by itself"
        ok = expect == "any" or got == expect
        if cut_in:
            ok = False
            got = "talked over"
        row.update(who=who, said=text_of(sp), expect=expect, got=got, ok=ok,
                   latency=round(on[0][0] - t_end, 2) if on else None)
        if on:
            await wait_quiet(ear, 1.5)
            row["gaps"] = gaps(ear, on[0][0])
        await asyncio.sleep(0.5)  # late events (a decision) belong to this step
        row.update(window_events(ear, t_start, ear.now()))
        return row
    if kind == "chat":
        _, who, text, expect = step
        await wait_quiet(ear, 1.0)
        b = bots_[who]
        t_start = b.now()
        await b.json({"type": "chat", "text": text})
        await ear.wait_ai(9.0, since=t_start)
        on = ear.ai_onsets(t_start)
        got = "reply" if on else "silent"
        row.update(who=who, said=f"(chat) {text}", expect=expect, got=got, ok=expect == "any" or got == expect,
                   latency=round(on[0][0] - t_start, 2) if on else None)
        if on:
            await wait_quiet(ear, 1.0)
        await asyncio.sleep(0.5)
        row.update(window_events(ear, t_start, ear.now()))
        return row
    if kind == "talk_over":
        _, who, sp, (who2, sp2), expect, *rest = step
        opts = rest[0] if rest else {}
        at = opts.get("at", 1.2)
        await wait_quiet(ear, 1.0)
        await asyncio.sleep(0.3)
        b, b2 = bots_[who], bots_[who2]
        pcm, pcm2 = speech(sp, voice_of(who)), speech(sp2, voice_of(who2))
        t_start = b.now()
        t_end = await b.say(pcm)
        on = await ear.wait_ai(10.0, since=t_start)
        row.update(who=who, said=text_of(sp), then_who=who2, then=text_of(sp2), expect=expect)
        if on is None:
            row.update(got="no reply to talk over", ok=None, latency=None)
            await asyncio.sleep(0.5)
            row.update(window_events(ear, t_start, ear.now()))
            return row
        row["latency"] = round(on - t_end, 2)
        while ear.now() < on + at:
            await asyncio.sleep(0.01)
        if not ear.ai_audible(0.3):
            row.update(got="reply too short to talk over", ok=None)
            await wait_quiet(ear, 1.0)
            row.update(window_events(ear, t_start, ear.now()))
            return row
        t_c = b2.now()
        t_c_end = await b2.say(pcm2)
        ear.mark = t_c_end  # an "await" step right after looks from here
        await asyncio.sleep(2.5)
        # stopped = a flush after the cue started, or Higgs's first reply ended within 1.5 s of the cue's end while a
        # stop was announced
        ev = window_events(ear, t_c, ear.now())
        stopped = bool(ev["flush"]) or any(s.get("stopped") for s in ev["states"])
        spans = ear.ai_onsets(on)
        first_end = spans[0][1] if spans else on
        # Higgs answering the cue as a new turn (plain Higgs does): audio that starts after the stop
        answered_cue = [round(a - t_c_end, 2) for a, _ in spans[1:] if a > t_c]
        got = "stopped" if stopped else "continued"
        if stopped and ev["resumed"]:  # stopped, then its line came back as a backchannel: it went on (agent._resume)
            got = "stopped, resumed"
        ok = (got == "stopped") == (expect == "stop")
        row.update(got=got, ok=ok, cue_at=round(t_c - on, 2),
                   stop_latency=round(ev["flush"][0] - t_c, 2) if ev["flush"] else None,
                   first_reply_len=round(first_end - on, 2), answered_cue_after=answered_cue[:1])
        await wait_quiet(ear, 1.2, 30)
        await asyncio.sleep(0.5)
        row.update(window_events(ear, t_start, ear.now()))
        return row
    if kind in ("mode", "avatar"):  # someone flips the switch mid-call
        msg = {"type": "set_mode", "mode": step[1]} if kind == "mode" else {"type": "set_avatar", "on": step[1]}
        await ear.json(msg)
        await asyncio.sleep(step[2] if len(step) > 2 else 3.0)
        return None
    if kind == "leave":  # someone drops off (closes the tab / loses the network)
        await bots_[step[1]].leave()
        await asyncio.sleep(step[2] if len(step) > 2 else 2.0)
        return None
    if kind == "rejoin":  # ... and comes back with the same tab id (the server gives them their tile back)
        old = bots_[step[1]]
        b = Bot(old.base, old.room, step[1])
        b.pid, b.gain_db = old.pid, getattr(old, "gain_db", 0.0)
        bots_[step[1]] = b
        if old is ear:
            raise RuntimeError("the listening bot can't leave")
        await b.join()
        await asyncio.sleep(step[2] if len(step) > 2 else 4.0)
        return None
    if kind == "flip_while_talking":  # flip a switch while Higgs is mid-reply
        _, who, sp, msg = step
        await wait_quiet(ear, 1.0)
        t0 = ear.now()
        await bots_[who].say(speech(sp, voice_of(who)))
        on = await ear.wait_ai(10.0, since=t0)
        if on is None:
            row.update(said=text_of(sp), expect="reply", got="no reply to flip during", ok=None)
            return row
        await asyncio.sleep(1.0)
        await ear.json(msg)
        await asyncio.sleep(5.0)
        row.update(said=text_of(sp) + f" + {msg}", expect="no crash", got="flipped", ok=True)
        row.update(window_events(ear, t0, ear.now()))
        if row.get("errors"):
            row.update(ok=False, got="errors")
        return row
    if kind == "join":  # someone joins mid-call
        b = Bot(ear.base, ear.room, step[1])
        bots_[step[1]] = b
        await b.join()
        await asyncio.sleep(step[2] if len(step) > 2 else 4.0)
        return None
    if kind == "await":  # after the previous step: does Higgs speak (again) within the window?
        _, expect, win = step
        t0 = getattr(ear, "mark", None) or ear.now()
        ear.mark = None
        # Higgs starting to speak (again): a stopped reply's last words before its stop takes effect (the stop comes
        # with the cue's transcript, ~0.8 s after it) are the same span, not a new one
        end = t0 + win
        while True:  # at least once: the step before may have waited out the whole window (talk_over waits for quiet)
            on = next((a for a, _ in ear.ai_onsets(t0 - 5.0) if t0 + 0.2 < a <= end), None)
            if on is not None or ear.now() >= end:
                break
            await asyncio.sleep(0.05)
        got = "reply" if on else "silent"
        row.update(said="(waiting)", expect=expect, got=got, ok=expect == "any" or got == expect,
                   latency=round(on - t0, 2) if on else None)
        if on:
            await wait_quiet(ear, 1.0)
        row.update(window_events(ear, t0, ear.now()))
        return row
    raise ValueError(kind)


async def run_scenario(base, sc, mode, outdir):
    people = sc["people"]
    QUIET_EXTRA["s"] = 1.5 if sc.get("avatar") else 0.0
    VOICE_OVERRIDE.clear()
    VOICE_OVERRIDE.update(sc.get("voices", {}))
    rid = create_room(base, ADMIN, title=sc["name"], ai=sc.get("ai", "Higgs"), setting=sc.get("setting", SET),
                      facts=sc.get("facts", HACK))
    bs = {p: Bot(base, rid, p, echo=sc.get("echo", {}).get(p, 0.0), noise_db=sc.get("noise", {}).get(p, -70.0))
          for p in people}
    for p, b in bs.items():
        b.gain_db = sc.get("gain", {}).get(p, 0.0)
        await b.join()
        await asyncio.sleep(0.2)
    ear = bs[people[0]]  # what one participant hears
    await asyncio.sleep(1.0)
    if mode != "default":
        await ear.json({"type": "set_mode", "mode": mode})
    if sc.get("avatar"):
        await ear.json({"type": "set_avatar", "on": True})
    await asyncio.sleep(sc.get("settle", 6.0))  # sessions open, turn-taking models load
    rows = []
    t0 = time.time()
    for step in sc["steps"]:
        try:
            r = await run_step(step, bs, ear, mode)
        except Exception as e:  # noqa: BLE001
            r = {"kind": step[0], "said": str(step[2])[:80], "ok": False, "got": f"harness error {e!r}"}
        if r:
            rows.append(r)
    ear.save_ai(os.path.join(outdir, f"{sc['name']}_{mode}_ai.wav"))
    for b in bs.values():
        await b.leave()
    res = {"scenario": sc["name"], "mode": mode, "room": rid, "secs": round(time.time() - t0, 1), "rows": rows,
           "ok": sum(r.get("ok") is True for r in rows), "scored": sum(r.get("ok") is not None for r in rows)}
    json.dump(res, open(os.path.join(outdir, f"{sc['name']}_{mode}.json"), "w"), indent=1)
    return res


def start_server(port, extra_env=None):
    env = {**os.environ, "MEET_ADMIN": ADMIN, "MEET_BUDGET": "100000", **(extra_env or {})}
    env.pop("MEET_AGENT", None)
    log = open(os.path.join(MEET, "out", f"scen_server_{port}.log"), "w")
    p = subprocess.Popen([sys.executable, "server.py", "--port", str(port)], cwd=MEET, env=env, stdout=log,
                         stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    for _ in range(300):
        try:
            bots.http(base + "/api/config")
            return p, base
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    raise RuntimeError("server did not start")


def short(r):
    s = f"{'OK ' if r.get('ok') else ('-- ' if r.get('ok') is None else 'XX ')}{r.get('said', '')[:60]:60s}"
    if r["kind"] == "talk_over":
        s += f" | {r.get('then', '')[:30]:30s} want {r.get('expect')}, got {r.get('got')}"
    else:
        s += f" | want {r.get('expect')}, got {r.get('got')}"
    if r.get("latency") is not None:
        s += f" ({r['latency']}s)"
    heard = [f"{x['name']}: {x['text']!r}->{'S' if x['speak'] else 's'}({x['why']})" for x in r.get("lines", [])]
    if heard:
        s += "\n      heard " + "; ".join(heard)
    for st in r.get("states", []):
        if st.get("dropped") or st.get("stopped"):
            s += f"\n      {st}"
    for e in r.get("errors", []):
        s += f"\n      !! {e}"
    return s


async def main_async(a):
    outdir = os.path.join(MEET, "out", "scen", a.tag)
    os.makedirs(outdir, exist_ok=True)
    pool = {"1": SCENARIOS, "2": SCENARIOS2, "3": SCENARIOS3, "4": SCENARIOS4, "5": SCENARIOS5, "6": SCENARIOS6, "7": SCENARIOS7,
            "8": SCENARIOS8, "9": SCENARIOS9,
            "all": SCENARIOS + SCENARIOS2}[a.set]
    todo = [(sc, m) for sc in pool if not a.only or sc["name"] in a.only for m in a.modes
            if m in sc.get("modes", ("default", "improved"))]
    # warm the TTS cache first (parallel, outside the timed part)
    texts = set()
    for sc, _ in todo:
        for st in sc["steps"]:
            specs = []
            if st[0] == "say":
                specs = [(st[1], st[2])]
            elif st[0] == "talk_over":
                specs = [(st[1], st[2]), st[3]]
            for who, sp in specs:
                for p in (sp if isinstance(sp, list) else [sp]):
                    if isinstance(p, str):
                        texts.add((p, sc.get("voices", {}).get(who) or VOICES[who]))
    await asyncio.gather(*[asyncio.to_thread(clip, t, v) for t, v in texts])
    servers = [start_server(a.port + i, {"MEET_SAMPLES": str(a.samples)} if a.samples else {})
               for i in range(a.servers or a.par)]
    servers = [servers[i % len(servers)] for i in range(a.par)]  # --servers 1 --par 3: three rooms on one server
    q = asyncio.Queue()
    for item in todo:
        q.put_nowait(item)
    results = []

    async def worker(base):
        while not q.empty():
            sc, m = q.get_nowait()
            r = await run_scenario(base, sc, m, outdir)
            results.append(r)
            print(f"\n== {sc['name']} [{m}] {r['ok']}/{r['scored']}", flush=True)
            for row in r["rows"]:
                print("   " + short(row), flush=True)

    try:
        await asyncio.gather(*[worker(base) for _, base in servers])
    finally:
        for p in {id(p): p for p, _ in servers}.values():
            p.terminate()
    summ = {}
    for r in results:
        s = summ.setdefault(r["mode"], {"ok": 0, "scored": 0})
        s["ok"] += r["ok"]
        s["scored"] += r["scored"]
    json.dump({"summary": summ, "results": sorted(results, key=lambda r: (r["scenario"], r["mode"]))},
              open(os.path.join(outdir, "summary.json"), "w"), indent=1)
    print("\nSUMMARY", json.dumps(summ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["improved", "default"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--servers", type=int, default=0, help="servers to spread the rooms over (default: one per room)")
    ap.add_argument("--set", default="1", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "all"])
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--port", type=int, default=8641)
    ap.add_argument("--samples", type=int, default=0, help="judge samples per line (0: the agent's default)")
    ap.add_argument("--tag", default=time.strftime("%m%d_%H%M"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
