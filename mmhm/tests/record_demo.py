"""Record a demo clip of the Turing Clock lab view: headless Chromium with a scripted fake microphone, the page
recorded as video, the round's audio (mic + Maya) recorded by the server (start it with RECORD=1), muxed together.

  RECORD=1 python app/server.py --port 8126 &
  python tests/record_demo.py 8126 mmhm      -> ../demo_clips/turing_clock_lab_<mode>.mp4
The scripted user: a question, an "mm-hm" while Maya answers, a 1.4 s thinking pause mid-sentence, and a real
interruption."""
import asyncio
import glob
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from eval.stimuli import get  # noqa: E402
from mmhm.higgs import SR, save_wav, silence  # noqa: E402

CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome")
DEMO = os.path.join(HERE, "..", "..", "demo_clips")
GAME = os.path.join(HERE, "..", "out", "game")


def script():
    v = "jake"
    parts = [silence(1.5), get("Hey Maya! So what are you working on this weekend?", v), silence(2.2),
             get("Mm-hm.", v), silence(7.0),
             get("I've been learning to play the", v), silence(1.4), get("cello for about two years now.", v),
             silence(2.5), get("Wait, sorry, where was that?", v), silence(8.0)]
    pcm = np.concatenate(parts)
    path = os.path.join(HERE, "demo_mic.wav")
    save_wav(path, pcm)
    return path, len(pcm) / SR


async def main(port, mode):
    from playwright.async_api import async_playwright
    mic, dur = script()
    before = set(glob.glob(os.path.join(GAME, "rec_*.wav")))
    vid_dir = os.path.join(HERE, "video_tmp")
    os.makedirs(vid_dir, exist_ok=True)
    for f in glob.glob(os.path.join(vid_dir, "*")):
        os.remove(f)
    frames = []  # (timestamp, path) from the DevTools screencast
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=CHROME, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={mic}", "--autoplay-policy=no-user-gesture-required"])
        ctx = await b.new_context(viewport={"width": 1280, "height": 800})
        page = await ctx.new_page()
        cdp = await ctx.new_cdp_session(page)

        def on_frame(params):
            import base64
            path = os.path.join(vid_dir, f"f{len(frames):05d}.jpg")
            open(path, "wb").write(base64.b64decode(params["data"]))
            frames.append((params["metadata"]["timestamp"], path))
            asyncio.ensure_future(cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]}))

        cdp.on("Page.screencastFrame", on_frame)
        await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 85, "maxWidth": 1280, "maxHeight": 800})
        await page.goto(f"http://127.0.0.1:{port}/?lab&mode={mode}")
        await page.wait_for_timeout(800)
        t_click = time.time()
        await page.click("#start")
        await page.wait_for_timeout(int((dur - 1.0) * 1000))
        await page.click("#giveup")
        await page.wait_for_timeout(1500)
        await cdp.send("Page.stopScreencast")
        await ctx.close(); await b.close()
    # frames arrive only when the page changes: give each one its real duration
    with open(os.path.join(vid_dir, "list.txt"), "w") as f:
        for (t, path), nxt in zip(frames, frames[1:] + [(frames[-1][0] + 0.1, None)]):
            f.write(f"file '{path}'\nduration {max(0.001, nxt[0] - t):.3f}\n")
        f.write(f"file '{frames[-1][1]}'\n")
    video = os.path.join(vid_dir, "page.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(vid_dir, "list.txt"),
                    "-vf", "fps=25,format=yuv420p", "-c:v", "libx264", video], check=True)
    t_first = frames[0][0]
    for _ in range(20):
        new = set(glob.glob(os.path.join(GAME, "rec_*.wav"))) - before
        if new:
            break
        await asyncio.sleep(0.5)
    rec = sorted(new)[-1]
    offset = (t_click - t_first) + 0.4  # first video frame -> first mic frame at the server (getUserMedia + socket)
    out = os.path.join(DEMO, f"turing_clock_lab_{mode}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-itsoffset", f"{offset:.2f}", "-i", rec,
                    "-filter_complex", "[1:a]pan=stereo|c0=0.8*c0+0.4*c1|c1=0.4*c0+0.8*c1[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                    "-c:a", "aac", "-shortest", out], check=True)
    print(out)


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8126, sys.argv[2] if len(sys.argv) > 2 else "mmhm"))
