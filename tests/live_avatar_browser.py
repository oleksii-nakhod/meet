"""The live face in real browsers: 2 headless Chromium pages, the stub agent (no Higgs calls), a real Boson avatar
session and LiveKit room (~1 min; needs LIVEKIT_* and BOSON_API_KEY). Checks that turning the avatar on gives every
page the avatar's LiveKit video and audio on the AI tile, that slot 0's audio then plays only from the avatar (not
also from the socket), how long after slot 0's audio reaches a page the avatar says it, that ai_flush mutes it for
the drain, and that turning it off or leaving releases the session.

Usage: python tests/live_avatar_browser.py [--port 8613]. Exit code 1 if any check fails."""
import argparse
import asyncio
import json
import os
import sys

from playwright.async_api import async_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fake_room import ADMIN, CHROME, MEET, WAVS, args, chat, check, http, join, results, start_server, wait  # noqa: E402

# a level meter on the avatar's own audio track, as the page hears it
METER = """() => {
  if (window.lkm) return true;
  if (!live.audio || !live.audio.srcObject) return false;
  const ac = new AudioContext(), an = ac.createAnalyser();
  ac.createMediaStreamSource(live.audio.srcObject).connect(an);
  const buf = new Float32Array(an.fftSize);
  window.lkm = { lvl: [], ws: [] };
  let n0 = meet.audioRx[0] || 0;
  setInterval(() => {
    an.getFloatTimeDomainData(buf);
    let s = 0; for (const v of buf) s += v * v;
    const now = performance.now();
    lkm.lvl.push([now, Math.sqrt(s / buf.length), meet.levels[0] || 0]);
    const n = meet.audioRx[0] || 0;
    if (n > n0) lkm.ws.push([now, n - n0]);
    n0 = n;
    if (lkm.lvl.length > 3000) lkm.lvl.splice(0, 1000);
  }, 10);
  return true;
}"""


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8613)
    a = ap.parse_args()
    srv, base = start_server(a.port)
    try:
        st, d = http(base + "/api/rooms", {"title": "Live face", "ai": "Higgs"}, {"X-Admin": ADMIN})
        rid = d["id"]
        async with async_playwright() as pw:
            browsers = [await pw.chromium.launch(executable_path=CHROME, args=args(w)) for w in WAVS[:2]]
            pages = [await (await b.new_context(viewport={"width": 1280, "height": 800})).new_page() for b in browsers]
            errors = []
            for pg in pages:
                pg.on("pageerror", lambda e: errors.append(str(e)))
            for pg, name in zip(pages, ["Ana", "Ben"]):
                await join(pg, f"{base}/r/{rid}", name)
            A, B = pages
            await wait(A, "meet.roster && meet.roster.people.length === 2")

            await A.click("#avatar")
            ok = all([await wait(p, "meet.live.state === 'on' && meet.live.video >= 1 && meet.live.audio >= 1", 40000)
                      for p in pages])
            check("avatar on: every page joins the LiveKit room and gets the avatar's video + audio", ok,
                  [await p.evaluate("JSON.stringify(meet.live)") for p in pages])
            # LiveKit starts a new viewer on the 240x320 layer and steps up as its bandwidth estimate grows (~10 s)
            await asyncio.gather(*[wait(p, "document.querySelector('.tile.ai video.lk').videoWidth >= 480", 30000) for p in pages])
            # decoded frames per second (headless Chromium paints fewer than it decodes on a busy machine)
            vid = [await p.evaluate("""() => new Promise((done) => {
                const v = document.querySelector('.tile.ai video.lk'), q0 = v.getVideoPlaybackQuality();
                let n = 0; const t0 = performance.now();
                const f = () => { n++; if (performance.now() - t0 < 3000) v.requestVideoFrameCallback(f); };
                v.requestVideoFrameCallback(f);
                setTimeout(() => { const q = v.getVideoPlaybackQuality();
                  done([v.videoWidth, v.videoHeight, +((q.totalVideoFrames - q0.totalVideoFrames) / 3).toFixed(1), +(n / 3).toFixed(1),
                        document.querySelector('.tile.ai').classList.contains('lk-on')]); }, 3000); })""") for p in pages]
            check("the avatar's video plays on the AI tile at full size (480x640), >= 20 fps decoded",
                  all(w >= 480 and h >= 640 and fps >= 20 and on for w, h, fps, _, on in vid),
                  f"[w, h, decoded fps, painted fps, lk-on]: {vid}")

            ok = await wait(B, METER, 5000)
            await asyncio.sleep(1.5)
            lags, mixer0, heard_for = [], 0.0, []
            for _ in range(3):
                await B.evaluate("lkm.lvl.length = 0; lkm.ws.length = 0")
                await chat(A, "/beep")
                await asyncio.sleep(4.0)
                m = await B.evaluate("lkm")
                ws_t = m["ws"][0][0] if m["ws"] else None
                loud = [t for t, r, _ in m["lvl"] if r > 0.02]
                if ws_t and loud:
                    lags.append(round((loud[0] - ws_t) / 1000, 2))
                    heard_for.append(round((loud[-1] - loud[0]) / 1000, 2))
                mixer0 = max([mixer0] + [x for _, _, x in m["lvl"]])
            check("3 beeps sent to slot 0 are heard from the avatar's audio", len(lags) == 3, f"heard for {heard_for} s")
            if lags:
                check("…about a lag (~1.3 s) after the socket's copy arrives", 0.3 < sorted(lags)[len(lags) // 2] < 3.0,
                      f"{lags} s")
            check("slot 0's socket audio isn't also played (no double voice)", mixer0 < 0.01, f"mixer slot 0 peak {mixer0:.3f}")

            await chat(A, "/flush")
            ok = await wait(B, "meet.live.flushes >= 1 && live.audio.muted === true", 3000)
            check("ai_flush mutes the avatar's audio at once", ok)
            ok = await wait(B, "live.audio.muted === false", 3000)
            check("…and unmutes it after the drain", ok)

            await A.click("#avatar")
            ok = all([await wait(p, "meet.live.state === 'off' && !document.querySelector('.tile.ai').classList.contains('lk-on')", 10000)
                      for p in pages])
            check("avatar off: every page leaves the LiveKit room", ok)
            await A.click("#avatar")
            ok = all([await wait(p, "meet.live.state === 'on' && meet.live.video >= 2", 40000) for p in pages])
            check("avatar on again: a new session reaches every page", ok)
            for b in browsers:
                await b.close()
            await asyncio.sleep(3.0)
            check("no page errors", not errors, errors[:3])
        log = [json.loads(x) for x in open(os.path.join(MEET, "out", "rooms", f"{rid}.jsonl"))]
        lv = [x for x in log if x["type"] == "live_avatar"]
        check("room log: two sessions started and both ended", [x["on"] for x in lv] == [True, False, True, False],
              [(x["on"], x.get("start_s") or x.get("why")) for x in lv])
        errs = [x for x in log if x["type"] in ("ai_error", "agent_error")]
        check("room log: no errors", not errs, errs[:3])
    finally:
        srv.kill()
    print(f"\n{sum(ok for _, ok in results)}/{len(results)} passed")
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
