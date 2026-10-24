// Meet room page. Two sockets: /ws/room (control JSON + 20 ms audio frames, 1 byte slot + Int16 PCM at 24 kHz)
// and /ws/video (camera JPEGs, 1 byte slot + JPEG; slot 0 = the AI's avatar as fMP4 chunks for a MediaSource).
// Slot 0 is the AI, 1..5 are people. window.meet holds counters for tests.
const $ = (id) => document.getElementById(id);
const RID = decodeURIComponent(location.pathname.split("/").pop());
const QS = new URLSearchParams(location.search);
const LAB = QS.has("lab");
const MODE_LABEL = { default: "Realtime", improved: "Meeting mode" }; // "default"/"improved" stay the internal names
const W = (window.meet = { audioRx: {}, videoRx: {}, msgs: {}, errors: [], audioTx: 0, videoTx: 0, videoSkip: 0,
  peak: [], roster: null, slot: null, ai: { mode: "default", avatar: false, status: "listening" }, levels: [], gain0: 1,
  humanGain: 1, avatar: { starts: 0, chunks: 0, played: 0 } });
const store = (s, k, v) => { try { if (v === undefined) return s.getItem(k); s.setItem(k, v); } catch { return null; } };
const wsUrl = (p) => `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${p}`;
const bump = (o, k) => (o[k] = (o[k] || 0) + 1);
window.addEventListener("error", (e) => W.errors.push(String(e.message)));
window.addEventListener("unhandledrejection", (e) => W.errors.push(String(e.reason)));

let info, pid, myName, passcode = QS.get("k") || "";
let ws, vws, ctx, mixer, micNode, stream, camVideo, left = false, joined = false, retry = 0, vretry = 0;
let lkRoom = null, lkConnecting = null, lkCred = null; // LiveKit: only while the live avatar is on
const lkAudio = new Map();
const LK_MIC_OPTIONS = { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 };
let muted = false, camOn = true, selfPeak = 0, capBusy = false, capTimer;
const tiles = {}; // slot -> tile element
let aiName = "Higgs";
let agenda = null, step = 0; // the room's script (web/agendas.js) and the shared cursor in it
let kbPanel = null;          // the knowledge-base panel (web/kb.js), mounted on first join

function newPid() {
  const a = new Uint8Array(12); crypto.getRandomValues(a);
  return Array.from(a, (b) => b.toString(16).padStart(2, "0")).join("");
}

function toast(text, cls = "") {
  const t = document.createElement("div");
  t.className = "toast " + cls; t.textContent = text;
  $("toasts").appendChild(t);
  setTimeout(() => t.remove(), 3500);
}

// ---------- join screen ----------
async function init() {
  const r = await fetch(`/api/rooms/${encodeURIComponent(RID)}`);
  if (!r.ok) {
    $("jtitle").textContent = "Room not found"; $("jsub").textContent = "Check the link, or ask for a new one.";
    $("jhome").classList.remove("hidden");
    return;
  }
  info = await r.json();
  aiName = info.ai;
  document.title = `${info.title} · Meet`;
  $("jtitle").textContent = info.title;
  $("jsub").textContent = `${info.people} of ${info.max} people here · ${info.ai} (AI) is in the room`;
  $("jname").value = store(localStorage, "meet_name") || "";
  if (info.passcode) { $("jkrow").classList.remove("hidden"); $("jk").value = passcode; }
  agenda = (window.AGENDAS || []).find((a) => a.id === info.agenda) || null;
  castButtons();
  if (agenda) renderScript();
  $("jform").classList.remove("hidden");
  ($("jname").value ? $("jgo") : $("jname")).focus();
}

async function join() {
  myName = $("jname").value.trim();
  if (!myName) { $("jerr").textContent = "Enter your name."; $("jname").focus(); return; }
  if (info.passcode) passcode = $("jk").value.trim();
  store(localStorage, "meet_name", myName);
  $("jgo").disabled = true; $("jerr").textContent = "";
  try {
    if (!ctx) await setupMedia();
    pid = store(sessionStorage, `meet_pid_${RID}`) || newPid();
    store(sessionStorage, `meet_pid_${RID}`, pid);
    left = false;
    connect();
  } catch (e) {
    $("jerr").textContent = `Couldn't start: ${e.message}`;
    $("jgo").disabled = false;
  }
}

async function setupMedia() {
  if (info.livekit) {
    camVideo = document.createElement("video");
    camVideo.muted = true; camVideo.playsInline = true; camVideo.autoplay = true;
    return; // LiveKit acquires and publishes the devices after its room connection is authorized.
  }
  const audio = { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 };
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio, video: { width: 320, height: 240, frameRate: { ideal: 15 } } });
  } catch (e) { // no camera (or refused): audio only
    stream = await navigator.mediaDevices.getUserMedia({ audio });
    camOn = false;
  }
  stream.getVideoTracks().forEach((t) => (t.enabled = camOn)); // a rejoin keeps "Camera off"
  ctx = new AudioContext({ sampleRate: 24000 });
  await ctx.audioWorklet.addModule("/worklets.js");
  micNode = new AudioWorkletNode(ctx, "mic-processor");
  ctx.createMediaStreamSource(stream).connect(micNode);
  mixer = new AudioWorkletNode(ctx, "mixer-processor", { outputChannelCount: [2] });
  mixer.connect(ctx.destination);
  ctx.resume().catch(() => {}); // rejects if Leave closes the context first
  micNode.port.onmessage = (e) => {
    const pcm = new Int16Array(e.data);
    let p = 0;
    for (let i = 0; i < pcm.length; i += 4) p = Math.max(p, Math.abs(pcm[i]));
    if (!muted) selfPeak = Math.max(selfPeak, p / 32768);
    if (!muted && ws && ws.readyState === 1 && ws.bufferedAmount < 32768) { ws.send(e.data); W.audioTx++; }
  };
  mixer.port.onmessage = (e) => {
    const { levels, gain, drops } = e.data;
    W.levels = levels; W.gain0 = gain; W.drops = drops;
    W.peak = levels.map((v, i) => Math.max(W.peak[i] || 0, v)); // loudest level heard per slot (tests)
    for (const s in tiles) setLevel(+s, +s === W.slot ? selfPeak : levels[s] || 0);
    selfPeak = 0;
  };
  camVideo = document.createElement("video");
  camVideo.muted = true; camVideo.playsInline = true; camVideo.autoplay = true;
  if (stream.getVideoTracks().length) { camVideo.srcObject = stream; camVideo.play().catch(() => {}); }
  capTimer = setInterval(capture, 125); // ~8 fps
}

// ---------- control socket ----------
function connect() {
  const q = new URLSearchParams({ room: RID, pid, name: myName, k: passcode });
  const sock = (ws = new WebSocket(wsUrl(`/ws/room?${q}`)));
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    retry = 0;
    if (muted) ws.send(JSON.stringify({ type: "mic", on: false }));
    if (!camOn) ws.send(JSON.stringify({ type: "cam", on: false }));
  };
  ws.onmessage = onMessage;
  ws.onerror = () => {};
  ws.onclose = (e) => {
    if (left || sock !== ws) return;
    const fatal = { 4001: "The room is full (5 people).", 4003: "Wrong passcode.", 4004: "This room doesn't exist any more.",
      4000: "You joined from another tab, so this one was disconnected." }[e.code];
    if (fatal) { leaveCall(fatal); return; }
    if (!joined) { // never got in: back to the join screen
      $("jerr").textContent = "Couldn't connect. Try again."; $("jgo").disabled = false; return;
    }
    toast("Connection lost, reconnecting…");
    setTimeout(() => !left && connect(), Math.min(5000, 500 * 2 ** retry++));
  };
}

function onMessage(e) {
  if (typeof e.data !== "string") {
    if (info.livekit || !mixer) return;
    bump(W.audioRx, new Uint8Array(e.data, 0, 1)[0]);
    mixer.port.postMessage(e.data, [e.data]);
    return;
  }
  const m = JSON.parse(e.data);
  bump(W.msgs, m.type);
  if (m.type === "roster") onRoster(m);
  else if (m.type === "chat") addChat(m);
  else if (m.type === "ai_flush") { if (mixer) mixer.port.postMessage({ flush: 0 }); }
  else if (m.type === "ai_gain") { if (mixer) mixer.port.postMessage({ gain: Math.max(0, Math.min(1, +m.gain)), slot: 0, ramp: 0.03 }); }
  else if (m.type === "human_gain") setHumanGain(m.gain, m.reason);
  else if (m.type === "user_buffering" && m.active) toast(`Listening to ${m.name}; processing it after the correction`, "mode");
  else if (m.type === "ai_state") onAiState(m);
  else if (m.type === "line") addLine(m);
  else if (m.type === "ai_note") toast(m.text, "mode");
  else if (m.type === "ai_say") aiSay(m.text, m.source, m.correction);
  else if (m.type === "correction") toast(`${aiName} is correcting ${m.speaker}${m.source ? ` (source: ${m.source})` : ""}`, "mode");
  else if (m.type === "lab") lab(m);
  else if (m.type === "script") setStep(m.step, false);
  else if (m.type === "kb") kbPanel?.update(m);
}

function onRoster(m) {
  const first = !joined;
  if (first) {
    joined = true;
    $("join").classList.add("hidden"); $("call").classList.remove("hidden");
    $("rtitle").textContent = m.title || info.title;
    toast("Use headphones so nobody hears themselves echo.");
    if (agenda) { markMine(); showScript(store(localStorage, "meet_script") !== "0"); }
    if (!kbPanel) import("/kb.js").then((mod) => { kbPanel = mod.mountKB($("kb"), RID, { by: myName, live: true }); }).catch(() => {});
  } else if (W.roster) {
    const before = new Set(W.roster.people.map((p) => p.slot + p.name));
    const now = new Set(m.people.map((p) => p.slot + p.name));
    for (const p of m.people) if (!before.has(p.slot + p.name) && p.slot !== m.you) toast(`${p.name} joined`);
    for (const p of W.roster.people) if (!now.has(p.slot + p.name) && p.slot !== m.you) toast(`${p.name} left`);
  }
  W.roster = m; W.slot = m.you;
  aiName = m.ai.name; $("ainame").textContent = aiName + ":";
  $("side").querySelector("h3").textContent = `Transcript · what ${aiName} heard and decided`;
  $("count").textContent = `${m.people.length} ${m.people.length === 1 ? "person" : "people"} + ${aiName}`;
  renderTiles(m);
  onAiState({ ...m.ai, type: "ai_state" });
  if (!vws || vws.readyState > 1) connectVideo(); // people's video always uses the video socket
}

// ---------- tiles ----------
function makeTile(slot) {
  const t = document.createElement("div");
  t.className = "tile"; t.dataset.slot = slot;
  if (slot === 0) {
    t.classList.add("ai");
    t.innerHTML = `<video class="idle" muted playsinline loop></video><video class="live" muted playsinline></video>
      <div class="orb"><div class="ring"></div><span></span></div><div class="badge"></div><div class="status"></div>
      <div class="cap"></div><div class="name"></div>`;
  } else {
    t.innerHTML = `<div class="ph"><span></span></div><div class="name"><b></b><span class="off hidden">muted</span></div>`;
    if (slot === W.slot) { t.classList.add("self"); t.prepend(camVideo); camVideo.play().catch(() => {}); }
    else if (info.livekit) { const v = document.createElement("video"); v.className = "remote"; v.autoplay = true; v.playsInline = true; t.prepend(v); }
    else t.prepend(document.createElement("img"));
  }
  if (LAB && slot > 0) {
    const l = document.createElement("div");
    l.className = "lab"; l.innerHTML = `<div>now<b><i></i></b></div><div>bc<b><i></i></b></div>`;
    t.appendChild(l);
  }
  return t;
}

function renderTiles(m) {
  const want = [{ slot: 0, name: aiName }, ...m.people];
  const keep = new Set(want.map((p) => p.slot));
  for (const s in tiles) {
    const isSelf = tiles[s].classList.contains("self");
    if (!keep.has(+s) || isSelf !== (+s === m.you && +s > 0)) { // gone, or the slot changed hands
      const img = tiles[s].querySelector("img");
      if (img && img.src) URL.revokeObjectURL(img.src);
      tiles[s].remove(); delete tiles[s];
    }
  }
  for (const p of want) {
    let t = tiles[p.slot];
    if (!t) { t = tiles[p.slot] = makeTile(p.slot); }
    const grid = $("grid");
    // order: AI, me, then the others by slot
    const order = (s) => (s === 0 ? -2 : s === m.you ? -1 : s);
    t.style.order = order(p.slot);
    if (!t.parentNode) grid.appendChild(t);
    if (p.slot === 0) { t.querySelector(".name").textContent = `${aiName} · AI`; t.querySelector(".orb span").textContent = aiName[0] || "H"; continue; }
    t.querySelector(".name b").textContent = p.slot === m.you ? `${p.name} (you)` : p.name;
    t.querySelector(".ph span").textContent = (p.name[0] || "?").toUpperCase();
    t.querySelector(".off").classList.toggle("hidden", p.mic !== false);
    if (p.slot === m.you) t.classList.toggle("has-video", camOn && (info.livekit ? !!camVideo?.srcObject : !!stream?.getVideoTracks().length));
    else { t.dataset.cam = p.cam !== false ? "1" : ""; if (p.cam === false) t.classList.remove("has-video"); }
  }
  if (info.livekit && lkRoom) attachLiveKitTracks();
  fit();
}

function fit() { // largest 4:3 tiles that fit the stage
  const n = Object.keys(tiles).length, st = $("stage"), gap = 10;
  const w = st.clientWidth - 24, h = st.clientHeight - 24;
  let best = 0, bc = 1;
  if (matchMedia("(max-width: 860px)").matches) { // phone: two columns, the page scrolls
    bc = Math.min(2, n); best = (w - gap * (bc - 1)) / bc;
  } else for (let c = 1; c <= n; c++) {
    const r = Math.ceil(n / c);
    const tw = Math.min((w - gap * (c - 1)) / c, ((h - gap * (r - 1)) / r) * 4 / 3);
    if (tw > best) { best = tw; bc = c; }
  }
  $("grid").style.gridTemplateColumns = `repeat(${bc}, ${Math.max(120, Math.floor(best))}px)`;
}
window.addEventListener("resize", fit);

function setLevel(slot, v) {
  const t = tiles[slot];
  if (t) t.style.setProperty("--lvl", Math.min(1, v * 3).toFixed(2));
}

function lab(m) {
  if (!LAB || !tiles[m.slot]) return;
  const bars = tiles[m.slot].querySelectorAll(".lab i");
  if (bars.length) { bars[0].style.width = `${Math.round((m.p_now || 0) * 100)}%`; bars[1].style.width = `${Math.round((m.p_bc || 0) * 100)}%`; }
}

// ---------- the AI ----------
function onAiState(m) {
  const prev = W.ai;
  W.ai = { mode: m.mode ?? prev.mode, avatar: m.avatar ?? prev.avatar, status: m.status ?? prev.status };
  if (m.by && W.ai.mode !== prev.mode) toast(`${m.by} switched ${aiName} to ${MODE_LABEL[W.ai.mode]}`, "mode");
  if (W.ai.status === "connecting" && prev.status !== "connecting") toast(`${aiName} is reconnecting…`, "mode");
  if (m.by && W.ai.avatar !== prev.avatar) toast(`${m.by} turned ${aiName}'s avatar ${W.ai.avatar ? "on" : "off"}`, "mode");
  const sw = $("modesw");
  sw.setAttribute("aria-checked", String(W.ai.mode === "improved"));
  sw.querySelectorAll(".opt").forEach((o) => o.classList.toggle("cur", o.dataset.m === W.ai.mode));
  $("avatar").textContent = `Avatar: ${W.ai.avatar ? "on" : "off"}`;
  const t = tiles[0];
  if (!t) return;
  const b = t.querySelector(".badge");
  b.textContent = MODE_LABEL[W.ai.mode] || W.ai.mode; b.classList.toggle("improved", W.ai.mode === "improved");
  t.querySelector(".status").textContent = W.ai.status;
  for (const s of ["listening", "thinking", "speaking"]) t.classList.toggle(s, W.ai.status === s);
  t.classList.toggle("idle-on", W.ai.avatar && !!t.querySelector("video.idle").src);
  if (info.avatar_livekit && joined) { // the live avatar's audio + video come from LiveKit, only while it's on
    if (W.ai.avatar) connectLiveKit().catch((e) => { W.errors.push(`LiveKit: ${e.message}`); toast(`Avatar: ${e.message}`); });
    else disconnectLiveKit();
  }
  if (W.ai.status !== "speaking") setTimeout(() => W.ai.status !== "speaking" && (t.querySelector(".cap").textContent = ""), 4000);
}

function aiSay(text, source, correction) {
  const t = tiles[0];
  clearTimeout(aiSay.timer);
  aiSay.timer = setTimeout(() => W.ai.status !== "speaking" && tiles[0] && (tiles[0].querySelector(".cap").textContent = ""), 8000);
  if (t) { t.querySelector(".cap").innerHTML = ""; const s = document.createElement("span"); s.textContent = text; t.querySelector(".cap").appendChild(s); }
  addLine({ slot: 0, name: aiName, text, source, correction });
}

// ---------- side panel ----------
function addItem(list, cls, who, text, extra, slot) {
  const ul = $(list);
  ul.querySelector(".empty")?.remove();
  const li = document.createElement("li");
  li.className = cls;
  if (slot !== undefined && slot !== null) li.dataset.slot = String(slot);
  const w = document.createElement("div"); w.className = "who"; w.textContent = who;
  const x = document.createElement("div"); x.textContent = text;
  li.append(w, x);
  if (extra) li.appendChild(extra);
  const atEnd = ul.scrollHeight - ul.scrollTop - ul.clientHeight < 40;
  ul.appendChild(li);
  if (atEnd) ul.scrollTop = 1e9;
}

function addChat(m) {
  addItem("chat", m.slot === 0 ? "ai" : "", m.slot === W.slot ? `${m.from} (you)` : m.from, m.text);
}

function addLine(m) {
  let extra = null;
  if (m.decision) {
    const d = m.decision;
    extra = document.createElement("div"); extra.className = "sub";
    const c = document.createElement("span");
    c.className = "chip" + (d.speak ? " speak" : ""); c.textContent = d.speak ? "SPEAK" : "silent";
    const rest = Object.entries(d).filter(([k]) => k !== "speak" && k !== "why").map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`);
    extra.append(c, [d.why, ...rest].filter(Boolean).join(" · "));
  }
  if (m.source || m.correction) {
    extra = document.createElement("div"); extra.className = "sub";
    extra.textContent = [m.correction ? "correction" : "", m.source ? `source: ${m.source}` : ""].filter(Boolean).join(" · ");
  }
  if (m.merge) {  // the same person went on after a pause: their last line becomes the whole line
    const prev = [...$("lines").querySelectorAll("li")].reverse().find((x) => x.dataset.slot === String(m.slot));
    if (prev) prev.remove();
  }
  addItem("lines", m.slot === 0 ? "ai" : "", m.name || "", m.text, extra, m.slot);
}

function setHumanGain(value, reason = "") {
  const gain = Math.max(0, Math.min(1, Number(value)));
  W.humanGain = gain;
  for (const [key, el] of lkAudio) if (key.startsWith("human-")) el.volume = gain;
  if (mixer) for (let slot = 1; slot <= 5; slot++) mixer.port.postMessage({ gain, slot, ramp: 0.08 });
  if (gain < 1 && reason === "fact_correction") toast(`${aiName} is correcting a verified fact`, "mode");
}

// ---------- LiveKit media ----------
function disconnectLiveKit() {
  if (lkRoom) { const r = lkRoom; lkRoom = null; r.disconnect().catch(() => {}); }
  for (const [, el] of lkAudio) el.remove(); lkAudio.clear();
  tiles[0]?.classList.remove("av-on");
}

// Subscribe-only: the page watches the avatar. A failed or dropped connection is cleaned up and retried while the
// avatar is on (a half-connected room used to block every later attempt).
async function connectLiveKit() {
  if (!info.avatar_livekit || left || lkRoom) return;
  if (lkConnecting) return lkConnecting;
  let room = null;
  lkConnecting = (async () => {
    if (!window.LivekitClient) throw new Error("the LiveKit browser SDK did not load");
    const response = await fetch(`/api/rooms/${encodeURIComponent(RID)}/livekit-token`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ pid, name: myName, passcode }),
    });
    const credentials = await response.json();
    if (!response.ok) throw new Error(credentials.error || `token request failed (${response.status})`);
    lkCred = credentials;
    const { Room, RoomEvent, Track } = LivekitClient;
    room = new Room({ adaptiveStream: true, disconnectOnPageLeave: true });
    room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => attachLiveKitTrack(track, publication, participant));
    room.on(RoomEvent.TrackUnsubscribed, (track, publication, participant) => detachLiveKitTrack(track, publication, participant));
    room.on(RoomEvent.ParticipantConnected, () => setTimeout(attachLiveKitTracks, 0));
    room.on(RoomEvent.ParticipantDisconnected, (participant) => {
      for (const [key, el] of lkAudio) if (key.startsWith(`${participant.identity}:`)) { el.remove(); lkAudio.delete(key); }
    });
    room.on(RoomEvent.ActiveSpeakersChanged, (speakers) => {
      for (const slot in tiles) setLevel(+slot, 0);
      for (const participant of speakers) {
        const person = W.roster?.people.find((p) => p.identity === participant.identity);
        if (person) setLevel(person.slot, 0.65);
        else if (participant.identity === credentials.agent_identity || participant.identity === credentials.avatar_identity) setLevel(0, 0.65);
      }
    });
    room.on(RoomEvent.Disconnected, () => {
      if (lkRoom !== room) return; // we disconnected it on purpose
      lkRoom = null;
      for (const [, el] of lkAudio) el.remove(); lkAudio.clear();
      if (!left && W.ai.avatar) { toast("Avatar connection lost, reconnecting…", "mode"); setTimeout(() => connectLiveKit().catch(() => {}), 2000); }
    });
    room.on(RoomEvent.AudioPlaybackStatusChanged, () => {
      if (!room.canPlaybackAudio) toast("Click anywhere in the call to enable audio", "mode");
    });
    await room.connect(credentials.url, credentials.token, { autoSubscribe: true });
    if (left || !W.ai.avatar) { room.disconnect().catch(() => {}); return; } // turned off while connecting
    lkRoom = room;
    attachLiveKitTracks();
    await unlockLiveKitAudio();
  })();
  try { await lkConnecting; }
  catch (e) { if (room && lkRoom !== room) room.disconnect().catch(() => {}); throw e; }
  finally { lkConnecting = null; }
}

async function unlockLiveKitAudio() {
  if (!lkRoom) return;
  try { await lkRoom.startAudio(); } catch {}
  for (const [, el] of lkAudio) {
    el.muted = false;
    el.volume = el.dataset.livekitAudio?.startsWith("human-") ? W.humanGain : 1;
    try { await el.play(); } catch {}
  }
}

function attachLiveKitTracks() {
  if (!lkRoom) return;
  const { Track } = LivekitClient;
  for (const publication of lkRoom.localParticipant.trackPublications.values()) {
    if (publication.source === Track.Source.Camera && publication.track && camVideo) {
      publication.track.attach(camVideo);
      camVideo.muted = true;
      tiles[W.slot]?.classList.toggle("has-video", camOn);
    }
  }
  for (const participant of lkRoom.remoteParticipants.values()) {
    for (const publication of participant.trackPublications.values()) {
      if (publication.track) attachLiveKitTrack(publication.track, publication, participant);
    }
  }
}

function attachLiveKitTrack(track, publication, participant) {
  if (!W.roster) return;
  const { Track } = LivekitClient;
  const person = W.roster.people.find((p) => p.identity === participant.identity);
  const isAgent = participant.identity === lkCred?.agent_identity;
  const isAvatar = participant.identity === lkCred?.avatar_identity;
  if (isAgent) return; // only the avatar's tracks are for us
  if (track.kind === Track.Kind.Audio) {
    const key = `${participant.identity}:${publication.trackSid}`;
    if (!lkAudio.has(key)) {
      const el = track.attach();
      el.autoplay = true; el.muted = false;
      el.volume = participant.identity.startsWith("human-") ? W.humanGain : 1;
      el.dataset.livekitAudio = key; el.style.display = "none";
      $("call").appendChild(el); lkAudio.set(key, el);
      el.play?.().catch(() => toast("Click anywhere in the call to enable audio", "mode"));
    }
    return;
  }
  if (track.kind !== Track.Kind.Video) return;
  if (isAvatar) {
    const tile = tiles[0], video = tile?.querySelector("video.live");
    if (video) { track.attach(video); video.muted = true; video.play().catch(() => {}); tile.classList.add("av-on"); W.avatar.starts++; }
  } else if (person) {
    const tile = tiles[person.slot], video = tile?.querySelector("video.remote");
    if (video) { track.attach(video); video.play().catch(() => {}); tile.classList.add("has-video"); bump(W.videoRx, person.slot); }
  } else if (!isAgent) {
    W.errors.push(`Unknown LiveKit participant ${participant.identity}`);
  }
}

function detachLiveKitTrack(track, publication, participant) {
  const key = `${participant.identity}:${publication.trackSid}`;
  const audio = lkAudio.get(key);
  if (audio) { track.detach(audio); audio.remove(); lkAudio.delete(key); }
  if (participant.identity === lkCred?.avatar_identity) tiles[0]?.classList.remove("av-on");
  const person = W.roster?.people.find((p) => p.identity === participant.identity);
  if (person) tiles[person.slot]?.classList.remove("has-video");
}

// ---------- video socket ----------
function connectVideo() {
  const sock = (vws = new WebSocket(wsUrl(`/ws/video?${new URLSearchParams({ room: RID, pid })}`)));
  vws.binaryType = "arraybuffer";
  vws.onopen = () => (vretry = 0);
  vws.onmessage = (e) => {
    if (typeof e.data === "string") {
      const m = JSON.parse(e.data);
      bump(W.msgs, m.type);
      if (m.type === "avatar_start") avatarStart(m.mime);
      else if (m.type === "avatar_stop") { av.ended = true; pump(); }
      else if (m.type === "avatar_idle") avatarIdle(m.url);
      return;
    }
    const slot = new Uint8Array(e.data, 0, 1)[0];
    bump(W.videoRx, slot);
    if (slot === 0) { if (av.ms) { av.q.push(e.data.slice(1)); W.avatar.chunks++; pump(); } return; }
    const t = tiles[slot], img = t && t.querySelector("img");
    if (!img || !t.dataset.cam) return;
    const url = URL.createObjectURL(new Blob([new Uint8Array(e.data, 1)], { type: "image/jpeg" }));
    const old = img.src;
    img.onload = () => { if (old) URL.revokeObjectURL(old); t.classList.add("has-video"); };
    img.onerror = () => URL.revokeObjectURL(url);
    img.src = url;
  };
  vws.onerror = () => {};
  vws.onclose = (e) => {
    if (left || sock !== vws || e.code === 4000 || !ws || ws.readyState > 1) return; // the control socket's reconnect reopens this
    setTimeout(() => !left && ws.readyState === 1 && (!vws || vws.readyState > 1) && connectVideo(), Math.min(5000, 500 * 2 ** vretry++));
  };
}

function capture() {
  if (!camOn || capBusy || !vws || vws.readyState !== 1 || !camVideo.videoWidth) return;
  if (vws.bufferedAmount > 65536) { W.videoSkip++; return; } // slow link: skip this frame
  capBusy = true;
  const c = capture.c || (capture.c = Object.assign(document.createElement("canvas"), { width: 320, height: 240 }));
  c.getContext("2d").drawImage(camVideo, 0, 0, 320, 240);
  c.toBlob((b) => {
    if (!b) { capBusy = false; return; }
    b.arrayBuffer().then((ab) => { if (vws.readyState === 1) { vws.send(ab); W.videoTx++; } }).finally(() => (capBusy = false));
  }, "image/jpeg", 0.6);
}

// Avatar video on the AI tile: fMP4 chunks appended to a MediaSource (as in read_my_lips/web/agent.js).
const av = { ms: null, sb: null, q: [], ended: false };
function avatarStart(mime) {
  const v = tiles[0]?.querySelector("video.live");
  if (!v) return;
  W.avatar.starts++;
  Object.assign(av, { ms: new MediaSource(), sb: null, q: [], ended: false });
  if (v.src) URL.revokeObjectURL(v.src);
  v.src = URL.createObjectURL(av.ms);
  const ms = av.ms;
  ms.addEventListener("sourceopen", () => {
    if (av.ms !== ms) return;
    try { av.sb = ms.addSourceBuffer(mime); } catch (e) { W.errors.push(`avatar: ${e.message}`); return; }
    av.sb.mode = "sequence";
    av.sb.addEventListener("updateend", pump);
    pump();
  }, { once: true });
  v.onplaying = () => { W.avatar.played++; tiles[0].classList.add("av-on"); };
  v.onended = () => tiles[0]?.classList.remove("av-on");
  v.play().catch(() => {});
}
function pump() {
  if (!av.sb || av.sb.updating) return;
  try {
    if (av.q.length) av.sb.appendBuffer(av.q.shift());
    else if (av.ended && av.ms.readyState === "open") av.ms.endOfStream();
  } catch (e) { W.errors.push(`avatar: ${e.message}`); }
}
function avatarIdle(url) {
  const v = tiles[0]?.querySelector("video.idle");
  if (!v) return;
  v.src = url; v.play().catch(() => {});
  tiles[0].classList.toggle("idle-on", W.ai.avatar);
}

// ---------- script ----------
const EXPECT = { reply: "{ai} answers", silent: "{ai} stays quiet", correct: "{ai} corrects it", continue: "{ai} keeps talking",
  stop: "{ai} stops", any: "either is fine" };
const subAi = (t) => t.replaceAll("{ai}", aiName);

const EXTRA_CAST = ["Junaid"]; // offered in every room, with no lines in any script

function castButtons() {
  const box = $("jcast").querySelector(".cast");
  box.replaceChildren();
  const cast = agenda?.cast || [];
  if (!agenda) $("jcast").querySelector("label").textContent = "Join as";
  for (const name of [...cast, ...EXTRA_CAST.filter((n) => !cast.includes(n))]) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = name;
    b.classList.toggle("cur", $("jname").value.trim().toLowerCase() === name.toLowerCase());
    b.onclick = () => { $("jname").value = name; box.querySelectorAll("button").forEach((x) => x.classList.toggle("cur", x === b)); $("jgo").focus(); };
    box.appendChild(b);
  }
  $("jcast").classList.remove("hidden");
}

function sayNode(text) { // "[2s]" -> a pause chip
  const d = document.createElement("div");
  d.className = "say";
  for (const part of subAi(text).split(/\[(\d+(?:\.\d+)?)s\]/).map((x, i) => [x, i % 2])) {
    if (!part[1]) { d.append(part[0]); continue; }
    const p = document.createElement("span"); p.className = "pause"; p.textContent = `pause ${part[0]} s`;
    d.appendChild(p);
  }
  return d;
}

function renderScript() {
  $("stitle").textContent = `Script · ${agenda.name}`;
  $("sagenda").textContent = agenda.agenda.map((x, i) => `${i + 1}. ${x}`).join(" · ");
  const ol = $("steps");
  ol.replaceChildren();
  agenda.script.forEach((x, i) => {
    const li = document.createElement("li");
    if (x.h) { li.className = "h"; li.textContent = x.h; ol.appendChild(li); return; }
    li.className = "step" + (x.act ? " act" : ""); li.dataset.i = i; li.dataset.who = (x.who || "").toLowerCase();
    if (!x.act) {
      const w = document.createElement("div"); w.className = "who";
      const b = document.createElement("b"); b.textContent = x.who;
      w.append(b, x.over ? `while ${aiName} is talking` : x.chat ? "types in the chat" : "");
      const e = document.createElement("span"); e.className = `exp ${x.expect}`; e.textContent = subAi(EXPECT[x.expect] || x.expect);
      w.appendChild(e);
      if (x.chat) {
        const c = document.createElement("button"); c.className = "tochat"; c.textContent = "put in chat";
        c.onclick = (ev) => { ev.stopPropagation(); $("chatin").value = subAi(x.chat); $("chatin").focus(); };
        w.appendChild(c);
      }
      li.appendChild(w);
    }
    li.appendChild(sayNode(x.act || x.say || x.chat));
    if (x.probe) { const p = document.createElement("div"); p.className = "probe"; p.textContent = subAi(x.probe); li.appendChild(p); }
    li.onclick = () => setStep(i, true);
    ol.appendChild(li);
  });
  setStep(step, false);
}

function markMine() {
  const me = (myName || "").toLowerCase();
  $("steps").querySelectorAll(".step").forEach((li) => li.classList.toggle("mine", !!me && li.dataset.who === me));
}

function setStep(i, share) {
  step = i;
  if (!agenda) return;
  let cur = null;
  $("steps").querySelectorAll(".step").forEach((li) => {
    const n = +li.dataset.i;
    li.classList.toggle("cur", n === i); li.classList.toggle("done", n < i);
    if (n === i) cur = li;
  });
  if (cur && !$("script").classList.contains("hidden")) cur.scrollIntoView({ block: "center", behavior: "smooth" });
  if (share) send({ type: "script", step: i }); // everyone's cursor follows, and the room log marks the step
}

function moveStep(d) {
  const all = [...$("steps").querySelectorAll(".step")].map((li) => +li.dataset.i);
  const k = all.indexOf(step);
  const next = all[Math.max(0, Math.min(all.length - 1, (k < 0 ? 0 : k) + d))];
  if (next !== undefined) setStep(next, true);
}

function showScript(on) {
  $("script").classList.toggle("hidden", !on); $("body").classList.toggle("with-script", on);
  $("scriptbtn").classList.remove("hidden"); $("scriptbtn").classList.toggle("on", on);
  $("scriptbtn").textContent = on ? "Hide script" : "Script";
  store(localStorage, "meet_script", on ? "1" : "0");
  fit();
  if (on) setStep(step, false);
}

$("scriptbtn").onclick = () => showScript($("script").classList.contains("hidden"));
$("sprev").onclick = () => moveStep(-1);
$("snext").onclick = () => moveStep(1);
document.addEventListener("keydown", (e) => {
  if (!agenda || !joined || e.ctrlKey || e.metaKey || e.altKey || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
  if (e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); moveStep(1); }
  else if (e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); moveStep(-1); }
});

// ---------- controls ----------
const send = (o) => { if (!ws || ws.readyState !== 1) return false; ws.send(JSON.stringify(o)); return true; };
$("mic").onclick = async () => {
  muted = !muted;
  $("mic").textContent = muted ? "Unmute" : "Mute"; $("mic").classList.toggle("on", muted);
  if (info.livekit && lkRoom) {
    try { await lkRoom.localParticipant.setMicrophoneEnabled(!muted, LK_MIC_OPTIONS); }
    catch (e) { toast(`Microphone: ${e.message}`); }
  }
  send({ type: "mic", on: !muted });
};
$("cam").onclick = async () => {
  if (!info.livekit && !stream?.getVideoTracks().length) { toast("No camera"); return; }
  camOn = !camOn;
  if (info.livekit && lkRoom) {
    try { await lkRoom.localParticipant.setCameraEnabled(camOn); attachLiveKitTracks(); }
    catch (e) { camOn = false; toast(`Camera: ${e.message}`); }
  } else stream.getVideoTracks().forEach((t) => (t.enabled = camOn));
  $("cam").textContent = camOn ? "Camera off" : "Camera on"; $("cam").classList.toggle("on", !camOn);
  tiles[W.slot]?.classList.toggle("has-video", camOn);
  send({ type: "cam", on: camOn });
};
const flipMode = () => send({ type: "set_mode", mode: W.ai.mode === "improved" ? "default" : "improved" });
$("modesw").onclick = flipMode;
$("modesw").onkeydown = (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); flipMode(); } };
$("avatar").onclick = async () => {
  await unlockLiveKitAudio();
  send({ type: "set_avatar", on: !W.ai.avatar });
};
$("call").addEventListener("pointerdown", () => unlockLiveKitAudio(), { passive: true });
$("invite").onclick = () => {
  const url = `${location.origin}/r/${encodeURIComponent(RID)}${passcode ? `?k=${encodeURIComponent(passcode)}` : ""}`;
  const ok = () => toast("Invite link copied");
  if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(url).then(ok, () => prompt("Invite link", url));
  else prompt("Invite link", url);
};
$("chatform").onsubmit = (e) => {
  e.preventDefault();
  const text = $("chatin").value.trim();
  if (text && send({ type: "chat", text })) $("chatin").value = "";
};
$("leave").onclick = () => leaveCall("You left the call.");
$("jgo").onclick = join;
$("jname").onkeydown = $("jk").onkeydown = (e) => { if (e.key === "Enter") join(); }; // returning false would cancel every key

function leaveCall(msg) {
  left = true; joined = false;
  if (ws) ws.onmessage = null; // frames still in flight would reach the closed mixer
  try { ws && ws.close(1000); vws && vws.close(1000); } catch {}
  if (lkRoom) { lkRoom.disconnect().catch(() => {}); lkRoom = null; }
  for (const [, el] of lkAudio) el.remove(); lkAudio.clear();
  for (const s in tiles) { tiles[s].remove(); delete tiles[s]; }
  W.roster = null;
  stopMedia();
  $("call").classList.add("hidden"); $("join").classList.remove("hidden");
  $("jerr").textContent = msg; $("jgo").disabled = false; $("jgo").textContent = "Rejoin";
  $("jhome").classList.remove("hidden");
}

// Release the mic and camera (the browser's in-use indicator goes off); Rejoin asks for them again.
function stopMedia() {
  clearInterval(capTimer);
  stream?.getTracks().forEach((t) => t.stop());
  if (camVideo) camVideo.srcObject = null;
  ctx?.close().catch(() => {});
  stream = ctx = mixer = micNode = camVideo = null;
}

init().catch((e) => { $("jtitle").textContent = "Couldn't load the room"; $("jerr").textContent = e.message; $("jhome").classList.remove("hidden"); });
