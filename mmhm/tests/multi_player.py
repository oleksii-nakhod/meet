"""N simultaneous players on one game server (headless Chromium, fake mics). Prints each round's turn timing.
Usage: python tests/multi_player.py [n] [port]"""
import asyncio
import glob
import json
import os
import sys
import time

from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome")
GAME = os.path.join(HERE, "..", "out", "game")


async def player(p, port, k):
    b = await p.chromium.launch(executable_path=CHROME, args=[
        "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-audio-capture={HERE}/fake_mic.wav", "--autoplay-policy=no-user-gesture-required"])
    page = await b.new_page()
    await page.goto(f"http://127.0.0.1:{port}/?mode=mmhm")
    await page.wait_for_timeout(300 * k)
    await page.click("#start")
    await page.wait_for_timeout(22000)
    await page.click("#giveup")
    await page.wait_for_timeout(1500)
    await b.close()


async def main(n, port):
    t0 = time.time()
    async with async_playwright() as p:
        await asyncio.gather(*[player(p, port, k) for k in range(n)])
    await asyncio.sleep(2)
    for f in sorted(glob.glob(os.path.join(GAME, "log_*_mmhm.jsonl")), key=os.path.getmtime):
        if os.path.getmtime(f) < t0:
            continue
        log = [json.loads(line) for line in open(f)]
        c = [r["t"] for r in log if r["kind"] == "commit"]
        r_ = [r["t"] for r in log if r["kind"] == "respond"]
        gaps = [round(min(x for x in r_ if x > cc) - cc, 2) for cc in c if any(x > cc for x in r_)]
        print(os.path.basename(f), "commit->respond", gaps[:4], "| events", len(log))


asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3, int(sys.argv[2]) if len(sys.argv) > 2 else 8123))
