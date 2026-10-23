// Turing Clock client: mic -> 20 ms Int16 frames -> WebSocket -> server Engine -> 20 ms speaker frames back.
const $ = (id) => document.getElementById(id);
const LAB = new URLSearchParams(location.search).has("lab");
const FORCE_MODE = new URLSearchParams(location.search).get("mode"); // for recording demos: ?mode=plain|mmhm
const PARTY = new URLSearchParams(location.search).has("party"); // a coin flip may connect you to a human operator

let ws, ctx, micNode, playerNode, stream, t0, timer, round = null, mode = null, readyResolve = null;
const strip = { you: [], her: [], marks: [] }; // samples of [t, level]

async function start() {
  ctx = new AudioContext({ sampleRate: 24000 });
  await ctx.audioWorklet.addModule("/static/worklets.js");
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  });
  const src = ctx.createMediaStreamSource(stream);
  micNode = new AudioWorkletNode(ctx, "mic-processor");
  playerNode = new AudioWorkletNode(ctx, "player-processor", { outputChannelCount: [2] });
  src.connect(micNode);
  if (PARTY) {
    // same phone-line sound for the human and the AI, so the channel itself gives nothing away
    const hp = new BiquadFilterNode(ctx, { type: "highpass", frequency: 300 });
    const lp = new BiquadFilterNode(ctx, { type: "lowpass", frequency: 3400 });
    playerNode.connect(hp).connect(lp).connect(ctx.destination);
  } else playerNode.connect(ctx.destination);
  playerNode.port.onmessage = (e) => push("her", e.data);

  const q = FORCE_MODE ? `?mode=${FORCE_MODE}` : PARTY ? "?party" : "";
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws${q}`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = onMessage;
  ws.onclose = () => { if (!$("game").classList.contains("hidden")) end(false, null); };
  // until the server's models and Higgs session are up, drop mic frames (a MessagePort queues messages until a
  // handler is attached, and a burst of stale audio would throw the server's 20 ms clock off)
  let live = false;
  micNode.port.onmessage = (e) => {
    if (!live) return;
    const f = new Int16Array(e.data);
    let m = 0;
    for (let i = 0; i < f.length; i += 8) m = Math.max(m, Math.abs(f[i]));
    push("you", m / 32768);
    if (ws.readyState === 1) ws.send(e.data);
  };
  await new Promise((r) => (ws.onopen = r));
  $("start").textContent = "Connecting…";
  await new Promise((r) => (readyResolve = r));
  live = true;

  $("intro").classList.add("hidden");
  $("game").classList.remove("hidden");
  if (LAB) $("lab").classList.remove("hidden");
  t0 = performance.now();
  timer = setInterval(tickClock, 50);
  requestAnimationFrame(draw);
}

function onMessage(e) {
  if (typeof e.data !== "string") { playerNode.port.postMessage(e.data, [e.data]); return; }
  const m = JSON.parse(e.data);
  if (m.type === "hello") round = m.round;
  else if (m.type === "ready") { if (readyResolve) readyResolve(); }
  else if (m.type === "flush") playerNode.port.postMessage("flush");
  else if (m.type === "tele" && LAB) {
    $("m_pnow").style.width = `${m.p_now * 100}%`;
    $("m_bc").style.width = `${m.p_bcdet * 100}%`;
    $("state").textContent = `state: ${m.state}  gain: ${m.gain}`;
  } else if (m.type === "event") {
    if (["commit", "abort", "backchannel", "backchannel_heard", "stop", "duck"].includes(m.kind)) strip.marks.push([now(), m.kind]);
    if (LAB) {
      const li = document.createElement("li");
      li.textContent = `${m.t.toFixed(2)}  ${m.kind}${m.text ? "  " + m.text : ""}${m.why ? "  (" + m.why + ")" : ""}`;
      $("events").prepend(li);
    }
  } else if (m.type === "reveal") { mode = m.mode; showReveal(); }
}

const now = () => (performance.now() - t0) / 1000;
function push(who, level) { strip[who].push([now(), level]); }

function tickClock() {
  const s = now();
  const mm = String(Math.floor(s / 60)).padStart(2, "0");
  const ss = (s % 60).toFixed(1).padStart(4, "0");
  $("clock").textContent = `${mm}:${ss}`;
}

function draw() {
  if ($("game").classList.contains("hidden")) return;
  const c = $("strip"), g = c.getContext("2d"), W = c.width, H = c.height, T = now(), span = 10;
  g.clearRect(0, 0, W, H);
  const x = (t) => W - ((T - t) / span) * W;
  const css = getComputedStyle(document.documentElement);
  for (const [who, y0, col] of [["you", H * 0.3, "--you"], ["her", H * 0.72, "--her"]]) {
    g.fillStyle = css.getPropertyValue(col);
    const arr = strip[who];
    while (arr.length && arr[0][0] < T - span) arr.shift();
    for (const [t, l] of arr) {
      const h = Math.min(1, l * 3) * H * 0.22;
      g.fillRect(x(t), y0 - h, 2, Math.max(1, 2 * h));
    }
  }
  g.fillStyle = css.getPropertyValue("--mark");
  g.font = "11px ui-monospace, monospace";
  strip.marks = strip.marks.filter(([t]) => t > T - span);
  for (const [t, k] of strip.marks) { g.fillRect(x(t), 0, 1, H); g.fillText(k, x(t) + 3, 12); }
  requestAnimationFrame(draw);
}

let result = {};
function end(pressed, tell) {
  clearInterval(timer);
  result = { round, pressed, seconds: Math.round(now() * 10) / 10, tell, guess: result.guess, party: PARTY };
  try { stream.getTracks().forEach((t) => t.stop()); ctx.close(); } catch (_) {}
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "verdict", ...result }));
  else showReveal();
}

async function showReveal() {
  $("game").classList.add("hidden"); $("tell").classList.add("hidden"); $("guess").classList.add("hidden");
  $("reveal").classList.remove("hidden");
  result.mode = mode;
  result.forced = !!FORCE_MODE; // demo/test rounds with a chosen mode are kept out of the stats
  await fetch("/api/round", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(result) });
  if (ws) ws.close();
  const name = mode === "human" ? "a real person (a volunteer down the hall)" : mode === "mmhm" ? "Higgs + Mm-hm (our turn-taking layer)" : "plain Higgs Realtime";
  $("reveal_title").textContent = result.pressed ? `Caught at ${result.seconds} s` : `You couldn't catch it in ${result.seconds} s`;
  $("reveal_text").textContent = `This round was ${name}.`;
  const st = await (await fetch("/api/stats")).json();
  const max = Math.max(1, ...["plain", "mmhm"].map((k) => st[k].median_s || 0));
  $("bars").innerHTML = ["plain", "mmhm"].map((k) => `
    <div class="row"><span>${k === "mmhm" ? "Higgs + Mm-hm" : "plain Higgs"}</span>
    <div><div class="bar ${k}" style="width:${((st[k].median_s || 0) / max) * 100}%"></div></div>
    <span>${st[k].median_s ?? "–"} s <small>(n=${st[k].n})</small></span></div>`).join("") +
    `<p class="hint">Median time until players caught it. Higher means more human.</p>`;
}

if (PARTY) {
  $("who").innerHTML = "You're about to talk to <b>Maya</b>. Maya is either an AI running on Higgs or a real person down the hall. A coin flip decided.";
  $("hint").textContent = "Use headphones. At the end, tell us who you think it was.";
}
$("start").onclick = () => start().catch((e) => alert("Could not start: " + e.message));
$("gotcha").onclick = () => {
  clearInterval(timer);
  result.seconds = Math.round(now() * 10) / 10;
  $("game").classList.add("hidden");
  $("tell").classList.remove("hidden");
};
document.querySelectorAll("#tell [data-tell]").forEach((b) => (b.onclick = () => {
  const s = result.seconds;
  result.guess = "ai";
  end(true, b.dataset.tell);
  result.seconds = s;
}));
$("giveup").onclick = () => {
  if (!PARTY) return end(false, null);
  clearInterval(timer);
  result.seconds = Math.round(now() * 10) / 10;
  $("game").classList.add("hidden");
  $("guess").classList.remove("hidden");
};
document.querySelectorAll("#guess [data-guess]").forEach((b) => (b.onclick = () => {
  const s = result.seconds;
  result.guess = b.dataset.guess;
  end(false, null);
  result.seconds = s;
}));
$("again").onclick = () => location.reload();
