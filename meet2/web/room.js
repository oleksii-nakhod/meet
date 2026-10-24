// The call page: join, microphone/camera, the per-slot mixer, transcript, notes, knowledge base, chat, avatar.
// People's audio and video always travel over the room WebSocket. With the live avatar on, the AI's synchronized
// audio + video come from LiveKit instead (the page subscribes to the avatar participant only).
import { mountKB } from "/static/kb.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const rid = location.pathname.split("/").pop();
const qs = new URLSearchParams(location.search);
const KIND_AUDIO = 1, KIND_VIDEO = 2;
const S = {
  room: null, you: null, roster: [], ws: null, ctx: null, mixer: null, mic: null, stream: null,
  micOn: true, camOn: true, ptt: false, pttDown: false, levels: [], aiState: "idle", avatar: "off",
  lines: new Map(), captions: "", frames: new Map(), lk: null, kb: null, humanGain: 1,
};
let pid = sessionStorage.getItem(`pid:${rid}`);
if (!pid) { pid = Math.random().toString(36).slice(2, 12); sessionStorage.setItem(`pid:${rid}`, pid); }

function toast(text, level = "info") {
  const el = document.createElement("div");
  el.className = `toast ${level}`;
  el.textContent = text;
  $("toasts").append(el);
  setTimeout(() => el.remove(), level === "error" ? 8000 : 4500);
}

// ------------------------------------------------------------------ join screen
const info = await fetch(`/api/rooms/${rid}`).then((r) => (r.ok ? r.json() : null));
if (!info) {
  $("jtitle").textContent = "This meeting doesn't exist";
  $("jsub").textContent = "The link may be wrong, or the server restarted (rooms live in memory).";
} else {
  S.room = info;
  $("jtitle").textContent = info.title;
  $("jsub").textContent = `${info.people} of ${info.max} people here · AI: ${info.ai}`;
  $("jform").classList.remove("hidden");
  $("jname").value = localStorage.getItem("name") || "";
  if (info.passcode) { $("jkrow").classList.remove("hidden"); $("jk").value = qs.get("k") || ""; }
  $("jname").focus();
}
$("jgo").onclick = join;
$("jname").addEventListener("keydown", (e) => { if (e.key === "Enter") join(); });

async function join() {
  const name = $("jname").value.trim();
  if (!name) { $("jname").focus(); return; }
  localStorage.setItem("name", name);
  $("jgo").disabled = true; $("jerr").textContent = "";
  S.camOn = $("jcam").checked;
  try { await startMedia(); } catch (e) {
    $("jerr").textContent = `Microphone access is needed to join (${e.message}).`; $("jgo").disabled = false; return;
  }
  connect(name);
}

// ------------------------------------------------------------------ media
async function startMedia() {
  const audio = { channelCount: 1, noiseSuppression: true, echoCancellation: true, autoGainControl: true };
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({ audio, video: S.camOn ? { width: 320, height: 240, frameRate: 15 } : false });
  } catch (e) {
    if (!S.camOn) throw e;
    S.camOn = false;
    S.stream = await navigator.mediaDevices.getUserMedia({ audio });
  }
  S.ctx = new AudioContext({ sampleRate: 24000 });
  await S.ctx.audioWorklet.addModule("/static/worklets.js");
  const src = S.ctx.createMediaStreamSource(S.stream);
  S.mic = new AudioWorkletNode(S.ctx, "mic-processor");
  src.connect(S.mic);
  S.mic.port.onmessage = (e) => {
    const { pcm, peak } = e.data;
    S.levels[S.you?.slot ?? 1] = peak;
    if (!S.ws || S.ws.readyState !== 1 || !S.micOn || (S.ptt && !S.pttDown)) return;
    const out = new Uint8Array(961);
    out[0] = KIND_AUDIO;
    out.set(new Uint8Array(pcm), 1);
    S.ws.send(out);
  };
  S.mixer = new AudioWorkletNode(S.ctx, "mixer-processor", { outputChannelCount: [2] });
  S.mixer.connect(S.ctx.destination);
  S.mixer.port.onmessage = (e) => { e.data.levels.forEach((v, i) => { if (i !== S.you?.slot) S.levels[i] = v; }); };
  if (S.camOn) startCamera();
}

let camTimer = null;
function startCamera() {
  const track = S.stream.getVideoTracks()[0];
  if (!track) return;
  const video = document.createElement("video");
  video.muted = true; video.playsInline = true; video.srcObject = new MediaStream([track]); video.play();
  S.selfVideo = video;
  const canvas = document.createElement("canvas");
  canvas.width = 320; canvas.height = 240;
  const g = canvas.getContext("2d");
  clearInterval(camTimer);
  camTimer = setInterval(() => {
    if (!S.camOn || !S.ws || S.ws.readyState !== 1 || S.ws.bufferedAmount > 65536 || video.readyState < 2) return;
    g.drawImage(video, 0, 0, 320, 240);
    canvas.toBlob(async (b) => {
      if (!b || !S.ws || S.ws.readyState !== 1) return;
      const buf = new Uint8Array(await b.arrayBuffer());
      const out = new Uint8Array(buf.length + 1);
      out[0] = KIND_VIDEO; out.set(buf, 1);
      S.ws.send(out);
    }, "image/jpeg", 0.6);
  }, 160);
}

// ------------------------------------------------------------------ socket
function connect(name) {
  const host = localStorage.getItem(`host:${rid}`) || "";
  const params = new URLSearchParams({ room: rid, pid, name, k: $("jk").value || qs.get("k") || "", h: host });
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/room?${params}`);
  ws.binaryType = "arraybuffer";
  S.ws = ws;
  ws.onmessage = (e) => (typeof e.data === "string" ? onEvent(JSON.parse(e.data)) : onBinary(e.data));
  ws.onclose = (e) => {
    const why = { 4001: "The meeting is full (5 people).", 4003: "Wrong passcode.", 4004: "This meeting doesn't exist any more.", 4000: "You joined from another tab." }[e.code];
    if (!S.you) { $("jerr").textContent = why || "Couldn't connect. Try again."; $("jgo").disabled = false; return; }
    if (why) { toast(why, "error"); return; }
    toast("Disconnected. Reconnecting…", "error");
    setTimeout(() => connect(name), 1500);
  };
}

function onBinary(buf) {
  const head = new Uint8Array(buf, 0, 2);
  if (head[0] === KIND_AUDIO) {
    const slot = head[1];
    if (slot === 0 && S.avatar === "live" && S.lk) return; // the avatar's audio comes from LiveKit
    const out = new Uint8Array(buf.byteLength - 1);
    out.set(new Uint8Array(buf, 1));
    out[0] = slot;
    S.mixer?.port.postMessage(out.buffer, [out.buffer]);
  } else if (head[0] === KIND_VIDEO) {
    const slot = head[1];
    const url = URL.createObjectURL(new Blob([new Uint8Array(buf, 2)], { type: "image/jpeg" }));
    const old = S.frames.get(slot);
    S.frames.set(slot, url);
    const img = document.querySelector(`.tile[data-slot="${slot}"] img`);
    if (img) img.src = url;
    if (old) setTimeout(() => URL.revokeObjectURL(old), 1000);
  }
}

function onEvent(m) {
  switch (m.type) {
    case "welcome":
      S.you = m.you; S.room = m.room; S.roster = m.roster; S.avatar = m.room.avatar;
      $("join").classList.add("hidden"); $("call").classList.remove("hidden"); $("chatform").classList.remove("hidden");
      $("rtitle").textContent = m.room.title;
      $("lines").innerHTML = "";
      S.lines.clear();
      m.lines.forEach(addLine);
      if (!m.lines.length) $("lines").innerHTML = `<li class="empty">Lines appear here as people talk.</li>`;
      renderNotes(m.notes);
      setAiState(m.ai_state);
      setHumanGain(m.human_gain, "");
      if (!S.kb) S.kb = mountKB($("pane-kb"), rid, { by: S.you.name, live: true });
      renderRoster(); renderAvatarButton(); renderMode(m.room.mode);
      if (S.avatar === "live") startLiveKit();
      S.ctx?.resume();
      break;
    case "roster": S.roster = m.people; S.you = m.people.find((p) => p.slot === S.you?.slot) || S.you; renderRoster(); break;
    case "line": addLine(m); break;
    case "ai_say": aiSay(m); break;
    case "ai_state": setAiState(m.state); break;
    case "ai_flush": S.mixer?.port.postMessage({ flush: 0 }); break;
    case "human_gain": setHumanGain(m.gain, m.reason); break;
    case "correction": showCorrection(m); break;
    case "notes": renderNotes(m); break;
    case "kb": S.kb?.update(m); break;
    case "chat": addChat(m); break;
    case "avatar":
      S.avatar = m.mode; renderAvatarButton(); renderRoster();
      if (m.mode === "live") startLiveKit(); else stopLiveKit();
      toast(`${m.by ? m.by + " turned" : "Turned"} the avatar ${m.mode === "live" ? "on" : "off"}`);
      break;
    case "mode": renderMode(m.mode); toast(`${m.by} switched ${S.room.ai} to ${m.mode}`); break;
    case "toast": toast(m.text, m.level); break;
    case "join": if (m.slot !== S.you?.slot) toast(`${m.name} joined`); break;
    case "leave": toast(`${m.name} left`); break;
  }
}

// ------------------------------------------------------------------ tiles
function renderRoster() {
  $("count").textContent = `${S.roster.length} ${S.roster.length === 1 ? "person" : "people"} + ${S.room.ai}`;
  const ai = `<div class="tile ai ${S.aiState}" data-slot="0">
      <div id="aivideo" class="${S.avatar === "live" ? "" : "hidden"}" style="position:absolute;inset:0"></div>
      <div class="ring ${S.avatar === "live" ? "hidden" : ""}">${esc(S.room.ai[0] || "A")}</div>
      <div class="caption" id="caption">${esc(S.captions)}</div>
      <div class="name">${esc(S.room.ai)} <span class="badge">AI</span></div></div>`;
  const people = S.roster.map((p) => {
    const me = p.slot === S.you?.slot;
    const src = me ? "" : S.frames.get(p.slot) || "";
    const media = me && S.camOn && S.selfVideo ? `<video autoplay muted playsinline></video>`
      : p.cam && src ? `<img src="${src}" alt="">` : `<img alt="" class="${p.cam && !me ? "" : "hidden"}"><div class="initial">${esc(p.name[0] || "?")}</div>`;
    return `<div class="tile" data-slot="${p.slot}">${media}
      <div class="name">${esc(p.name)}${me ? " (you)" : ""}${p.host ? ' <span class="badge">host</span>' : ""}${!p.mic ? ' <span class="badge">muted</span>' : ""}${p.noisy ? ' <span class="badge">noisy mic</span>' : ""}</div></div>`;
  }).join("");
  $("grid").innerHTML = ai + people;
  const v = document.querySelector(`.tile[data-slot="${S.you?.slot}"] video`);
  if (v && S.selfVideo) v.srcObject = S.selfVideo.srcObject;
  if (S.lk?.videoEl) $("aivideo").append(S.lk.videoEl);
}

function setAiState(state) {
  S.aiState = state || "idle";
  $("aistate").textContent = S.aiState;
  const t = document.querySelector('.tile[data-slot="0"]');
  if (t) t.className = `tile ai ${S.aiState}`;
}

function animate() {
  for (const t of document.querySelectorAll(".tile")) {
    const slot = +t.dataset.slot, v = S.levels[slot] || 0;
    t.classList.toggle("speaking", v > 0.04 && slot !== 0);
    if (slot === 0) {
      const ring = t.querySelector(".ring");
      if (ring) ring.style.transform = `scale(${1 + Math.min(0.35, v * 1.4)})`;
    }
  }
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);

function setHumanGain(gain, reason) {
  S.humanGain = Math.max(0, Math.min(1, Number(gain ?? 1)));
  for (let slot = 1; slot <= 5; slot++) S.mixer?.port.postMessage({ gain: S.humanGain, slot, ramp: 0.12 });
  if (S.humanGain >= 1) $("banner").classList.add("hidden");
}

// ------------------------------------------------------------------ transcript
function addLine(m) {
  const list = $("lines");
  list.querySelector(".empty")?.remove();
  let li = S.lines.get(m.id);
  if (!li) {
    li = document.createElement("li");
    S.lines.set(m.id, li);
    list.append(li);
  }
  const kind = m.kind || "line";
  li.className = [kind === "ai" || kind === "correction" ? "ai" : "", kind, m.final === false ? "partial" : ""].join(" ");
  const tag = { filler: "filler", backchannel: "backchannel", buffered: "held during correction", stop: "asked to stop" }[kind];
  const meta = [m.why, tag, m.interrupted ? "interrupted" : "", m.source ? `source: ${m.source}` : ""].filter(Boolean).join(" · ");
  li.innerHTML = `<div class="who">${esc(m.speaker)}</div><div>${esc(m.text)}</div>${meta ? `<div class="meta">${esc(meta)}</div>` : ""}`;
  const pane = $("pane-lines");
  if (pane.scrollHeight - pane.scrollTop - pane.clientHeight < 120) pane.scrollTop = pane.scrollHeight;
}

function aiSay(m) {
  S.captions = m.text;
  const c = $("caption");
  if (c) c.textContent = m.text;
  addLine({ id: m.id, speaker: S.room.ai, text: m.text, kind: m.kind === "correction" ? "correction" : "ai",
    interrupted: m.interrupted, source: m.source, why: m.why });
  if (m.done) setTimeout(() => { if (S.captions === m.text) { S.captions = ""; const c2 = $("caption"); if (c2) c2.textContent = ""; } }, 2500);
}

function showCorrection(m) {
  const b = $("banner");
  b.innerHTML = `<b>Correction</b> · ${esc(m.text)}${m.source ? `<div class="sub">Source: ${esc(m.source)}${m.evidence ? ` — “${esc(m.evidence)}”` : ""}</div>` : ""}`;
  b.classList.remove("hidden");
  setTimeout(() => { if (S.humanGain >= 1) b.classList.add("hidden"); }, 12000);
}

function renderNotes(n) {
  if (!n || !(n.topic || n.decisions?.length || n.actions?.length || n.questions?.length)) return;
  const list = (t, xs) => (xs && xs.length ? `<h3>${t}</h3><ul>${xs.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "");
  $("pane-notes").innerHTML = `${n.topic ? `<h3>Topic</h3><div>${esc(n.topic)}</div>` : ""}${list("Decisions", n.decisions)}${list("Action items", n.actions)}${list("Open questions", n.questions)}${list("Claims to check", n.claims)}`;
}

function addChat(m) {
  $("chat").querySelector(".empty")?.remove();
  const li = document.createElement("li");
  if (m.slot === 0) li.className = "ai";
  li.innerHTML = `<div class="who">${esc(m.name)}</div><div>${esc(m.text)}</div>`;
  $("chat").append(li);
  $("pane-chat").scrollTop = $("pane-chat").scrollHeight;
  if (m.slot !== S.you?.slot && !document.querySelector('[data-tab="chat"]').classList.contains("on")) toast(`${m.name}: ${m.text}`);
}
$("chatform").onsubmit = (e) => {
  e.preventDefault();
  const text = $("chatin").value.trim();
  if (text && S.ws?.readyState === 1) { S.ws.send(JSON.stringify({ type: "chat", text })); $("chatin").value = ""; }
};

document.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
  for (const id of ["lines", "notes", "kb", "chat"]) $(`pane-${id}`).classList.toggle("hidden", id !== b.dataset.tab);
});

// ------------------------------------------------------------------ controls
$("mic").onclick = () => {
  S.micOn = !S.micOn;
  $("mic").textContent = S.micOn ? "Mute" : "Unmute"; $("mic").classList.toggle("on", !S.micOn);
  S.ws?.send(JSON.stringify({ type: "mic", on: S.micOn }));
};
$("cam").onclick = async () => {
  if (!S.stream.getVideoTracks().length) {
    try {
      const v = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240, frameRate: 15 } });
      S.stream.addTrack(v.getVideoTracks()[0]);
      S.camOn = true; startCamera();
    } catch { toast("No camera available", "error"); return; }
  } else {
    S.camOn = !S.camOn;
    S.stream.getVideoTracks().forEach((t) => (t.enabled = S.camOn));
  }
  $("cam").textContent = S.camOn ? "Camera off" : "Camera on";
  S.ws?.send(JSON.stringify({ type: "cam", on: S.camOn }));
  renderRoster();
};
$("ptt").onclick = () => {
  S.ptt = !S.ptt;
  $("ptt").textContent = `Push to talk: ${S.ptt ? "on (hold Space)" : "off"}`; $("ptt").classList.toggle("on", S.ptt);
};
$("ptt").addEventListener("pointerdown", () => { if (S.ptt) S.pttDown = true; });
$("ptt").addEventListener("pointerup", () => { S.pttDown = false; });
document.addEventListener("keydown", (e) => { if (S.ptt && e.code === "Space" && document.activeElement?.tagName !== "INPUT") { S.pttDown = true; e.preventDefault(); } });
document.addEventListener("keyup", (e) => { if (e.code === "Space") S.pttDown = false; });
function renderAvatarButton() {
  $("avatar").textContent = `Avatar: ${S.avatar === "live" ? "on" : "off"}`;
  $("avatar").classList.toggle("on", S.avatar === "live");
  $("transport").textContent = S.avatar === "live" ? "avatar via LiveKit" : "voice only";
}
$("avatar").onclick = () => {
  if (!S.room.livekit) { toast("The live avatar needs LiveKit settings on the server (see README).", "error"); return; }
  S.ws?.send(JSON.stringify({ type: "set_avatar", mode: S.avatar === "live" ? "off" : "live" }));
};
function renderMode(mode) {
  document.querySelectorAll("#modesw button").forEach((b) => b.classList.toggle("on", b.dataset.m === (mode || "smart")));
}
document.querySelectorAll("#modesw button").forEach((b) => b.onclick = () => S.ws?.send(JSON.stringify({ type: "set_mode", mode: b.dataset.m })));
$("invite").onclick = async () => {
  await navigator.clipboard.writeText(location.href.split("?")[0] + (qs.get("k") ? `?k=${encodeURIComponent(qs.get("k"))}` : ""));
  toast("Invite link copied");
};
$("leave").onclick = () => {
  S.ws.onclose = null; S.ws.close(); stopLiveKit();
  S.stream?.getTracks().forEach((t) => t.stop()); S.ctx?.close();
  document.body.innerHTML = `<div class="page"><h1>You left the meeting</h1><p><a href="${location.pathname}">Rejoin</a></p></div>`;
};

// ------------------------------------------------------------------ live avatar (LiveKit, subscribe only)
const LK_URL = "https://cdn.jsdelivr.net/npm/livekit-client@2.5.7/dist/livekit-client.umd.min.js";
function loadScript(src) {
  return new Promise((ok, fail) => { const s = document.createElement("script"); s.src = src; s.onload = ok; s.onerror = fail; document.head.append(s); });
}
async function startLiveKit() {
  if (S.lk) return;
  S.lk = { connecting: true };
  try {
    if (!window.LivekitClient) await loadScript(LK_URL);
    const cred = await fetch(`/api/rooms/${rid}/livekit-token`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ pid }) }).then((r) => r.json());
    if (!cred.token) throw new Error(cred.detail || "no token");
    const { Room, RoomEvent, Track } = window.LivekitClient;
    const room = new Room({ adaptiveStream: true });
    S.lk = { room, avatar: cred.avatar_identity };
    room.on(RoomEvent.TrackSubscribed, (track, _pub, participant) => {
      if (participant.identity !== cred.avatar_identity) return;
      const el = track.attach();
      if (track.kind === Track.Kind.Video) {
        el.style.cssText = "width:100%;height:100%;object-fit:cover"; S.lk.videoEl = el; $("aivideo")?.append(el);
      } else { el.style.display = "none"; document.body.append(el); S.lk.audioEl = el; }
    });
    room.on(RoomEvent.Disconnected, () => { if (S.avatar === "live") toast("Avatar connection lost", "error"); });
    await room.connect(cred.url, cred.token, { autoSubscribe: true });
  } catch (e) {
    toast(`Couldn't show the avatar: ${e.message}`, "error");
    S.lk = null;
  }
}
function stopLiveKit() {
  if (!S.lk) return;
  S.lk.room?.disconnect(); S.lk.audioEl?.remove(); S.lk.videoEl?.remove();
  S.lk = null;
}
