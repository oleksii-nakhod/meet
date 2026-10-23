"""Party mode smoke test: an operator page and a player page in two headless browsers with different fake mics.
Needs a server started with PARTY_HUMAN_P=1.0 (always route to the waiting human)."""
import asyncio
import os
import sys

from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome")


def args(wav):
    return ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={wav}", "--autoplay-policy=no-user-gesture-required"]


async def main(port):
    async with async_playwright() as p:
        bo = await p.chromium.launch(executable_path=CHROME, args=args(f"{HERE}/fake_operator.wav"))
        bp = await p.chromium.launch(executable_path=CHROME, args=args(f"{HERE}/fake_mic.wav"))
        op = await bo.new_page(); pl = await bp.new_page()
        for pg, name in ((op, "op"), (pl, "player")):
            pg.on("pageerror", lambda e, n=name: print(n, "pageerror:", e))
        await op.goto(f"http://127.0.0.1:{port}/static/operator.html")
        await op.click("#go")
        await op.wait_for_timeout(1500)
        print("operator:", await op.text_content("#title"))
        await pl.goto(f"http://127.0.0.1:{port}/?party")
        await pl.click("#start")
        await pl.wait_for_timeout(1500)
        print("operator:", await op.text_content("#title"))
        # measure audio arriving at each side via the player worklet level messages
        await pl.wait_for_timeout(8000)
        lvl = await pl.evaluate("Math.max(0, ...strip.her.map(x => x[1]))")
        print("max level of operator audio heard by player:", round(lvl, 3))
        await pl.click("#giveup")
        await pl.click("[data-guess=human]")
        await pl.wait_for_selector("#reveal:not(.hidden)", timeout=5000)
        print("player reveal:", await pl.text_content("#reveal_title"), "|", await pl.text_content("#reveal_text"))
        await pl.wait_for_timeout(1500)
        print("operator:", await op.text_content("#title"))
        await bo.close(); await bp.close()


asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8124))
