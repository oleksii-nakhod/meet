// Third Voice page: mic -> server (Higgs Realtime VAD + STT -> speak/silent judge -> TTS reply) -> speakers.
// The mic is muted while the AI's reply plays (and 300 ms after), so the AI doesn't transcribe itself.
const $ = (id) => document.getElementById(id);
let ws, ctx, muteUntil = 0, playAt = 0;

function why(w) {
  if (!w) return "";
  const parts = [`to: ${w.to}`];
  if (w.ask) parts.push("asks for something");
  if (w.answered) parts.push("already answered");
  if (w.ask && !w.can) parts.push("only people can answer");
  if (w.wrong) parts.push("contradicts the facts");
  return parts.join(" · ");
}

function addLine(cls, html) {
  const log = $("log");
  log.querySelector(".empty")?.remove();
  const li = document.createElement("li");
  li.className = cls; li.innerHTML = html;
  log.appendChild(li); log.scrollTop = 1e9;
}
const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

async function start() {
  $("start").disabled = true;
  ctx = new AudioContext({ sampleRate: 24000 });
  await ctx.audioWorklet.addModule("worklets.js");
  const mic = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
  const node = new AudioWorkletNode(ctx, "mic-processor");
  ctx.createMediaStreamSource(mic).connect(node);
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  await new Promise((r) => (ws.onopen = r));
  ws.send(JSON.stringify({
    ai: $("ai").value.trim() || "Higgs", setting: $("setting").value.trim(),
    people: $("people").value.split(",").map((s) => s.trim()).filter(Boolean),
    facts: $("facts").value.split("\n").map((s) => s.trim()).filter(Boolean),
  }));
  ws.onmessage = onMessage;
  node.port.onmessage = (e) => {
    if (ws.readyState !== 1) return;
    // while the AI talks, send silence instead of the mic (keeps the server's audio clock running)
    ws.send(performance.now() < muteUntil ? new Int16Array(480).buffer : e.data);
  };
}

async function onMessage(e) {
  if (typeof e.data !== "string") { // streamed TTS: Int16 PCM at 24 kHz, scheduled back to back
    const pcm = new Int16Array(e.data);
    const buf = ctx.createBuffer(1, pcm.length, 24000);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
    const src = ctx.createBufferSource();
    src.buffer = buf; src.connect(ctx.destination);
    playAt = Math.max(playAt, ctx.currentTime + 0.05);
    src.start(playAt);
    playAt += buf.duration;
    muteUntil = performance.now() + (playAt - ctx.currentTime) * 1000 + 300;
    $("status").textContent = `${$("ai").value} is speaking`;
    src.onended = () => { if (ctx.currentTime >= playAt - 0.01) $("status").textContent = "listening"; };
    return;
  }
  const m = JSON.parse(e.data);
  if (m.type === "ready") $("status").textContent = "listening";
  else if (m.type === "hearing") $("status").textContent = "hearing someone…";
  else if (m.type === "line") {
    addLine("", `<div class="txt">${esc(m.text)}</div><div class="meta"><span class="chip ${m.speak ? "speak" : ""}">${m.speak ? "SPEAK" : "silent"}</span>${esc(why(m.why))} · decided in ${m.decided_s} s</div>`);
    $("status").textContent = "listening";
  } else if (m.type === "ai") {
    addLine("ai", `<div class="txt">${esc($("ai").value)}: ${esc(m.text)}</div><div class="meta">starts ${m.speaks_s} s after the line ended</div>`);
  }
}

$("start").onclick = () => start().catch((err) => { alert(err.message); $("start").disabled = false; });
