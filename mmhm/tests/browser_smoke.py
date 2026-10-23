"""End-to-end smoke test of the Turing Clock in headless Chromium with a fake microphone (tests/fake_mic.wav).
Usage: python tests/browser_smoke.py [mode] [port]  (server must be running)."""
import asyncio
import os
import sys

from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome")


async def main(mode="mmhm", port=8123, secs=24):
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=CHROME, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={HERE}/fake_mic.wav", "--autoplay-policy=no-user-gesture-required"])
        page = await b.new_page()
        page.on("console", lambda m: print("console:", m.text))
        page.on("pageerror", lambda e: print("pageerror:", e))
        await page.goto(f"http://127.0.0.1:{port}/?lab&mode={mode}")
        await page.click("#start")
        await page.wait_for_timeout(secs * 1000)
        events = await page.eval_on_selector_all("#events li", "els => els.map(e => e.textContent)")
        print("\n".join(reversed(events)))
        print("state:", await page.text_content("#state"), "| clock:", await page.text_content("#clock"))
        await page.screenshot(path=f"{HERE}/shot_game_{mode}.png", full_page=True)
        await page.click("#gotcha")
        await page.click("[data-tell=slow]")
        await page.wait_for_selector("#reveal:not(.hidden)", timeout=5000)
        await page.wait_for_timeout(800)
        print("reveal:", await page.text_content("#reveal_title"), "|", await page.text_content("#reveal_text"))
        await page.screenshot(path=f"{HERE}/shot_reveal_{mode}.png", full_page=True)
        await b.close()


asyncio.run(main(*(sys.argv[1:2] or ["mmhm"]), *(map(int, sys.argv[2:3]) or [8123])))
