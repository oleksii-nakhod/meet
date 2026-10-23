"""Headless-Chromium test of the live Third Voice page: the old meeting's 20 lines (TTS, out/stt/M0) are stitched
with 1.6 s pauses into one WAV that Chromium uses as the microphone. Prints every line with the AI's decision and
the AI's replies, and saves a screenshot. Usage: python browser_test.py [port] [secs] [bigfacts 0/1]  (app.py must be running)"""
import asyncio
import os
import sys

import numpy as np
import soundfile as sf
from playwright.async_api import async_playwright
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome")


def fake_mic():
    path = os.path.join(HERE, "out", "fake_meeting_M0.wav")
    if not os.path.exists(path):
        parts = [np.zeros(16000, np.float32)]
        for i in range(20):
            x = sf.read(os.path.join(HERE, "out", "stt", "M0", f"{i:02d}.flac"), dtype="float32")[0]
            parts += [x, np.zeros(int(1.6 * 16000), np.float32)]
        parts.append(np.zeros(16000 * 30, np.float32))  # don't loop back to the start too soon
        sf.write(path, resample_poly(np.concatenate(parts), 3, 1), 48000, subtype="PCM_16")
    return path


async def main(port=8400, secs=110, bigfacts=0):
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=CHROME, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={fake_mic()}", "--autoplay-policy=no-user-gesture-required"])
        page = await b.new_page(viewport={"width": 1100, "height": 900})
        page.on("pageerror", lambda e: print("pageerror:", e))
        await page.goto(f"http://127.0.0.1:{port}/")
        if bigfacts:  # pad the fact sheet with 40 unrelated facts from the test meetings (exercises retrieval)
            import json
            other = [f for line in open(os.path.join(HERE, "testset", "meetings_B.jsonl")) for f in json.loads(line)["facts"]]
            other += [f for line in open(os.path.join(HERE, "testset", "meetings_C.jsonl")) for f in json.loads(line)["facts"]]
            cur = await page.input_value("#facts")
            await page.fill("#facts", cur + "\n" + "\n".join(other[:40]))
        await page.click("#start")
        await page.wait_for_timeout(secs * 1000)
        for t in await page.eval_on_selector_all("#log li", "els => els.map(e => e.innerText.replace(/\\n/g, ' | '))"):
            print(t)
        await page.screenshot(path=os.path.join(HERE, "out", "shot_third_voice.png"), full_page=True)
        await b.close()


asyncio.run(main(*map(int, sys.argv[1:4])))
