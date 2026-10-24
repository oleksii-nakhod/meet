"""Screenshots of the AI tile with the live face (stub agent, a real avatar session, ~40 s): idle, thinking (the glow),
and just after ai_flush (the still of it listening over its moving lips). -> out/shots/live_*.png

    python tests/live_face_shots.py [--port 8616]"""
import argparse
import asyncio
import os
import sys

from playwright.async_api import async_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fake_room import ADMIN, CHROME, MEET, WAVS, args, chat, http, join, start_server, wait  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8616)
    a = ap.parse_args()
    out = os.path.join(MEET, "out", "shots")
    os.makedirs(out, exist_ok=True)
    srv, base = start_server(a.port)
    try:
        _, d = http(base + "/api/rooms", {"title": "Live face shots", "ai": "Higgs"}, {"X-Admin": ADMIN})
        async with async_playwright() as pw:
            b = await pw.chromium.launch(executable_path=CHROME, args=args(WAVS[0]))
            pg = await (await b.new_context(viewport={"width": 1100, "height": 700})).new_page()
            await join(pg, f"{base}/r/{d['id']}", "Ana")
            await pg.click("#avatar")
            await wait(pg, "meet.live.state === 'on' && meet.live.video >= 1", 40000)
            await wait(pg, "document.querySelector('.tile.ai video.lk').videoWidth >= 480", 30000)
            await asyncio.sleep(4.0)  # a still is taken every second while it listens
            tile = pg.locator(".tile.ai")
            await tile.screenshot(path=os.path.join(out, "live_idle.png"))
            await pg.evaluate("onAiState({type: 'ai_state', status: 'thinking'})")
            await asyncio.sleep(0.6)
            await tile.screenshot(path=os.path.join(out, "live_thinking.png"))
            await pg.evaluate("onAiState({type: 'ai_state', status: 'listening'})")
            await chat(pg, "/beep")
            await asyncio.sleep(1.6)  # the face is saying the beep now
            await chat(pg, "/flush")
            await asyncio.sleep(0.3)
            on = await pg.evaluate("document.querySelector('.tile.ai').classList.contains('still-on')")
            await tile.screenshot(path=os.path.join(out, "live_flushed.png"))
            print("still-on after flush:", on, "->", out)
            await b.close()
    finally:
        srv.kill()


if __name__ == "__main__":
    asyncio.run(main())
