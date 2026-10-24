// Knowledge base panel (F5), used on the setup page and in the room's Knowledge tab.
//   mountKB(el, roomId, { by }) -> { update(state) }  (update() takes a state pushed over the room socket)

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const BUSY = new Set(["queued", "parsing", "indexed", "extracting facts"]);

export function mountKB(el, roomId, opts = {}) {
  el.innerHTML = `
    <div class="drop" data-k="drop">Drop PDF, DOCX, MD, TXT, CSV or HTML files here, or
      <label style="display:inline;color:var(--accent);cursor:pointer">browse<input type="file" multiple hidden
      accept=".pdf,.docx,.md,.markdown,.txt,.csv,.html,.htm" data-k="file"></label></div>
    <div class="row2">
      <input class="grow" placeholder="https://… a web page or PDF link" data-k="url">
      <button data-k="addurl">Add link</button>
    </div>
    <details style="margin-top:10px"><summary class="sub" style="cursor:pointer">Paste text instead</summary>
      <input placeholder="Title" data-k="ttitle" style="margin-top:8px">
      <textarea placeholder="Paste notes, minutes, a policy…" data-k="text" style="margin-top:8px"></textarea>
      <button data-k="addtext" style="margin-top:8px">Add text</button>
    </details>
    <ul class="doclist" data-k="docs"></ul>
    <h2 style="margin-top:18px">Fact sheet</h2>
    <div class="sub">The AI answers from these and uses them to catch contradictions. Edit, delete or add facts, then save.</div>
    <ul class="factlist" data-k="facts"></ul>
    <div class="row2">
      <button data-k="addfact">Add a fact</button>
      <button class="primary" data-k="savefacts">Save fact sheet</button>
      <span class="sub" data-k="saved"></span>
    </div>`;
  const $ = (k) => el.querySelector(`[data-k="${k}"]`);
  let state = { docs: [], facts: [] }, dirty = false, timer = null;

  const api = async (path, init) => {
    const r = await fetch(`/api/rooms/${roomId}/kb${path}`, init);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || body.error || `HTTP ${r.status}`);
    return body;
  };
  const say = (t) => { $("saved").textContent = t; setTimeout(() => { if ($("saved").textContent === t) $("saved").textContent = ""; }, 3000); };

  function renderDocs() {
    $("docs").innerHTML = state.docs.length ? state.docs.map((d) => {
      const cls = d.status === "ready" ? "good" : d.status === "error" ? "bad" : "warn";
      const detail = d.status === "error" ? esc(d.error) : d.status === "ready" ? `${d.pages} page${d.pages === 1 ? "" : "s"} · ${d.facts} facts` : "";
      return `<li><span class="grow">${esc(d.name)}<br><span class="sub">${detail}</span></span>
        <span class="pill ${cls}">${esc(d.status)}</span><button class="danger" data-del="${d.id}" title="Remove">✕</button></li>`;
    }).join("") : `<li class="empty">No documents yet.</li>`;
  }
  function renderFacts() {
    if (dirty) return;
    $("facts").innerHTML = state.facts.map((f) => factRow(f.text, f.doc ? `${f.doc}${f.page ? " p." + f.page : ""}` : f.origin)).join("")
      || `<li class="empty">No facts yet: add some, or upload a document.</li>`;
  }
  const factRow = (text, src) => `<li><input value="${esc(text)}" data-fact><span class="src">${esc(src || "")}</span>
    <button class="danger" data-rmfact title="Delete">✕</button></li>`;

  function update(s) {
    state = s;
    renderDocs();
    renderFacts();
    clearTimeout(timer);
    if (!opts.live && state.docs.some((d) => BUSY.has(d.status))) timer = setTimeout(refresh, 1200);
  }
  async function refresh() { try { update(await api("")); } catch (e) { say(e.message); } }

  async function upload(files) {
    for (const f of files) {
      const fd = new FormData();
      fd.append("file", f);
      fd.append("by", opts.by || "");
      try { await api("/files", { method: "POST", body: fd }); } catch (e) { say(`${f.name}: ${e.message}`); }
    }
    refresh();
  }
  const drop = $("drop");
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); upload(e.dataTransfer.files); });
  $("file").addEventListener("change", (e) => upload(e.target.files));
  $("addurl").onclick = async () => {
    const url = $("url").value.trim();
    if (!url) return;
    try { await api("/sources", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ url }) }); $("url").value = ""; } catch (e) { say(e.message); }
    refresh();
  };
  $("addtext").onclick = async () => {
    const text = $("text").value;
    if (!text.trim()) return;
    try {
      await api("/sources", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text, title: $("ttitle").value }) });
      $("text").value = ""; $("ttitle").value = "";
    } catch (e) { say(e.message); }
    refresh();
  };
  $("docs").addEventListener("click", async (e) => {
    const id = e.target.dataset.del;
    if (id) { await api(`/docs/${id}`, { method: "DELETE" }); refresh(); }
  });
  $("facts").addEventListener("input", () => { dirty = true; });
  $("facts").addEventListener("click", (e) => { if (e.target.dataset.rmfact !== undefined) { e.target.closest("li").remove(); dirty = true; } });
  $("addfact").onclick = () => {
    const empty = $("facts").querySelector(".empty");
    if (empty) empty.remove();
    $("facts").insertAdjacentHTML("beforeend", factRow("", "host"));
    $("facts").lastElementChild.querySelector("input").focus();
    dirty = true;
  };
  $("savefacts").onclick = async () => {
    const facts = [...$("facts").querySelectorAll("[data-fact]")].map((i) => i.value.trim()).filter(Boolean);
    try {
      dirty = false;
      update(await api("/facts", { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ facts }) }));
      say("Saved");
    } catch (e) { say(e.message); }
  };
  refresh();
  return { update, refresh };
}
