/*
 * Minimal chat UI client (Issue #56).
 *
 * Drives the existing backend end-to-end with no build step:
 *  - JWT login (POST /api/auth/login/), token kept in localStorage.
 *  - Conversation CRUD (/api/conversations/), pinned to a ModelVersion.
 *  - Streaming replies over the WebSocket the backend already exposes
 *    (chat/consumers.py): open ws/chat/<id>/?token=<JWT>, POST the prompt to
 *    kick off generation, render {type:"token",delta} deltas live, finalize on
 *    {type:"done"}.
 *  - Per-message thinking / web-search toggles sent with the prompt.
 *  - Delete a conversation (backend now frees its on-disk state file too).
 *
 * Served as an external same-origin file because the CSP is script-src 'self'
 * (no inline scripts) — see web/middleware/security_headers.py.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const store = {
    get access() { return localStorage.getItem("hvn_access"); },
    set access(v) { v ? localStorage.setItem("hvn_access", v) : localStorage.removeItem("hvn_access"); },
    get refresh() { return localStorage.getItem("hvn_refresh"); },
    set refresh(v) { v ? localStorage.setItem("hvn_refresh", v) : localStorage.removeItem("hvn_refresh"); },
  };

  let activeId = null;       // selected conversation id
  let ws = null;             // current WebSocket
  let generating = false;    // a turn is currently streaming (Send→Stop state)
  let streamingBubble = null; // assistant bubble being filled by the stream
  let modelList = [];        // last-loaded ModelVersions (reused by the training form)
  const STAGES = ["html", "css", "js", "php", "python", "devops"];

  // ---- HTTP helpers --------------------------------------------------------

  async function api(path, { method = "GET", body, auth = true, retry = true } = {}) {
    const headers = { "Content-Type": "application/json" };
    if (auth && store.access) headers["Authorization"] = `Bearer ${store.access}`;
    const res = await fetch(`/api${path}`, {
      method, headers, body: body ? JSON.stringify(body) : undefined,
    });
    if (res.status === 401 && auth && retry && store.refresh) {
      // One transparent refresh-and-retry so an expired access token doesn't
      // bounce the user to the login screen mid-session.
      if (await tryRefresh()) return api(path, { method, body, auth, retry: false });
    }
    return res;
  }

  async function tryRefresh() {
    if (!store.refresh) return false;
    const res = await fetch("/api/auth/refresh/", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh: store.refresh }),
    });
    if (!res.ok) { logout(); return false; }
    const data = await res.json();
    store.access = data.access;
    return true;
  }

  // ---- Auth ----------------------------------------------------------------

  $("login-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("login-err").textContent = "";
    const body = {
      username: $("username").value.trim(),
      password: $("password").value,
      otp_token: $("otp").value.trim(),
    };
    const res = await fetch("/api/auth/login/", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      $("login-err").textContent = data.detail || data.otp_token || "Login failed. Check your credentials.";
      return;
    }
    const data = await res.json();
    store.access = data.access;
    store.refresh = data.refresh;
    showApp();
  });

  $("logout").addEventListener("click", logout);

  function logout() {
    if (store.refresh) {
      api("/auth/logout/", { method: "POST", body: { refresh: store.refresh } }).catch(() => {});
    }
    store.access = null; store.refresh = null;
    closeSocket();
    closeTrainingSocket();
    trainJobId = null;
    $("training").hidden = true;
    $("sidebar").hidden = true; $("main").hidden = true; $("login").hidden = false;
  }

  // ---- Bootstrap -----------------------------------------------------------

  async function showApp() {
    $("login").hidden = true; $("sidebar").hidden = false; $("main").hidden = false;
    await loadModels();
    await loadConversations();
    await initTraining();
  }

  async function loadModels() {
    const sel = $("model-select");
    sel.innerHTML = "";
    const res = await api("/models/");
    if (!res.ok) { logout(); return; }
    const models = await res.json();
    const list = models.results || models; // handle paginated or plain list
    if (!list.length) {
      const opt = document.createElement("option");
      opt.textContent = "No model registered"; opt.value = "";
      sel.appendChild(opt);
      $("create-convo").disabled = true;
      return;
    }
    $("create-convo").disabled = false;
    modelList = list;
    for (const m of list) {
      const opt = document.createElement("option");
      opt.value = m.id; opt.textContent = m.tag + (m.is_active ? " (active)" : "");
      if (m.is_active) opt.selected = true;
      sel.appendChild(opt);
    }
  }

  async function loadConversations() {
    const res = await api("/conversations/");
    if (!res.ok) { logout(); return; }
    const data = await res.json();
    const list = data.results || data;
    const box = $("convos");
    box.innerHTML = "";
    for (const c of list) {
      const row = document.createElement("div");
      row.className = "convo" + (c.id === activeId ? " active" : "");
      row.dataset.id = c.id;

      const title = document.createElement("span");
      title.className = "title";
      title.textContent = c.title || "Untitled chat";
      row.appendChild(title);

      const del = document.createElement("button");
      del.className = "del"; del.title = "Delete conversation"; del.textContent = "✕";
      del.addEventListener("click", (e) => { e.stopPropagation(); deleteConversation(c.id); });
      row.appendChild(del);

      row.addEventListener("click", () => selectConversation(c.id));
      box.appendChild(row);
    }
  }

  $("create-convo").addEventListener("click", async () => {
    const modelId = $("model-select").value;
    if (!modelId) return;
    const res = await api("/conversations/", {
      method: "POST", body: { title: "New chat", model_version: modelId },
    });
    if (!res.ok) { setStatus("Could not create conversation."); return; }
    const convo = await res.json();
    await loadConversations();
    selectConversation(convo.id);
  });

  async function deleteConversation(id) {
    if (!confirm("Delete this conversation? Its saved state is also freed.")) return;
    const res = await api(`/conversations/${id}/`, { method: "DELETE" });
    if (res.status !== 204 && !res.ok) { setStatus("Delete failed."); return; }
    if (id === activeId) {
      activeId = null; closeSocket();
      $("messages").innerHTML = '<div class="empty">Select or create a conversation to start chatting.</div>';
      setGenerating(false);
      $("send").disabled = true;
    }
    await loadConversations();
  }

  // ---- Conversation view + streaming --------------------------------------

  async function selectConversation(id) {
    activeId = id;
    streamingBubble = null;
    document.querySelectorAll(".convo").forEach((el) =>
      el.classList.toggle("active", el.dataset.id === id));

    const res = await api(`/conversations/${id}/messages/`);
    if (!res.ok) { setStatus("Could not load messages."); return; }
    const messages = await res.json();
    const box = $("messages");
    box.innerHTML = "";
    for (const m of messages) {
      if (m.role === "system") continue;
      addBubble(m.role, m.content);
    }
    if (!messages.length) box.innerHTML = '<div class="empty">Say hello to start the conversation.</div>';
    openSocket(id);
    setGenerating(false);
    $("send").disabled = false;
  }

  function openSocket(id) {
    closeSocket();
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${scheme}://${location.host}/ws/chat/${id}/?token=${encodeURIComponent(store.access)}`);
    ws.onmessage = (evt) => {
      const data = JSON.parse(evt.data);
      if (data.type === "token") {
        if (!streamingBubble) streamingBubble = addBubble("assistant", "");
        streamingBubble.dataset.raw = (streamingBubble.dataset.raw || "") + data.delta;
        renderAssistant(streamingBubble);
        scrollDown();
      } else if (data.type === "done") {
        // A stopped turn (Issue #61) is discarded server-side, so the partial
        // bubble we streamed is orphaned - drop it to match the saved history.
        if (data.stopped && streamingBubble) streamingBubble.remove();
        streamingBubble = null;
        setStatus(data.stopped ? "Stopped." : "");
        setGenerating(false);
      } else if (data.type === "error") {
        // Terminal failure frame (Issue #60) - un-stick the composer instead of
        // hanging on "Generating…" forever.
        streamingBubble = null;
        setStatus(data.detail || "Generation failed. Please try again.");
        setGenerating(false);
      }
    };
    ws.onclose = (evt) => {
      if (evt.code === 4401) { setStatus("Session expired — please log in again."); logout(); }
      else if (evt.code === 4403) setStatus("You do not have access to this conversation.");
      else if (generating) {
        // Socket dropped mid-generation - don't leave the UI stuck. send() will
        // transparently reopen it on the next attempt.
        setStatus("Connection lost. Send again to reconnect.");
        setGenerating(false);
      }
    };
  }

  // Toggle the composer between idle (Send) and generating (Stop) states.
  function setGenerating(on) {
    generating = on;
    $("send").hidden = on;
    $("send").disabled = on;
    $("stop").hidden = !on;
  }

  function closeSocket() {
    if (ws) { ws.onclose = null; ws.close(); ws = null; }
  }

  async function waitForSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return true;
    if (!ws) return false;
    return new Promise((resolve) => {
      const t = setTimeout(() => resolve(false), 3000);
      ws.addEventListener("open", () => { clearTimeout(t); resolve(true); }, { once: true });
    });
  }

  async function send() {
    const promptEl = $("prompt");
    const content = promptEl.value.trim();
    if (!content || !activeId || generating) return;

    // Transparently (re)open a dropped socket instead of dead-ending on
    // "Connecting…" (Issue #63): a chat whose socket closed (server restart,
    // idle drop) would otherwise never recover without reselecting it.
    if (!(await waitForSocket())) {
      openSocket(activeId);
      if (!(await waitForSocket())) { setStatus("Connecting… try again in a moment."); return; }
    }

    const emptyMsg = $("messages").querySelector(".empty");
    if (emptyMsg) emptyMsg.remove();

    addBubble("user", content);
    promptEl.value = "";
    setGenerating(true);
    streamingBubble = null;
    setStatus("Generating…");
    scrollDown();

    const res = await api(`/conversations/${activeId}/messages/`, {
      method: "POST",
      body: {
        content,
        thinking: $("tg-thinking").checked,
        web_search: $("tg-websearch").checked,
      },
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      setStatus(data.detail || "Message rejected.");
      setGenerating(false);
    }
    // Success: tokens arrive over the WebSocket; a done/error frame ends the turn.
  }

  async function stop() {
    if (!generating || !activeId) return;
    setStatus("Stopping…");
    // Best-effort: the server stops at the next token and sends a done frame
    // (which flips the UI back). Don't flip locally yet so a failed request
    // doesn't leave the composer enabled while the worker is still streaming.
    const res = await api(`/conversations/${activeId}/stop/`, { method: "POST" });
    if (!res.ok) setStatus("Could not stop — it should finish shortly.");
  }

  $("send").addEventListener("click", send);
  $("stop").addEventListener("click", stop);
  $("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  $("tg-showthink").addEventListener("change", () => {
    document.querySelectorAll(".msg.assistant").forEach(renderAssistant);
  });

  // ---- Rendering -----------------------------------------------------------

  const THINK_RE = /\[THINK\]([\s\S]*?)\[\/THINK\]/g;

  function addBubble(role, content) {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    el.dataset.raw = content || "";
    if (role === "assistant") renderAssistant(el);
    else el.textContent = content;
    $("messages").appendChild(el);
    return el;
  }

  function renderAssistant(el) {
    const raw = el.dataset.raw || "";
    const showThink = $("tg-showthink").checked;
    el.innerHTML = "";
    let last = 0, m;
    THINK_RE.lastIndex = 0;
    while ((m = THINK_RE.exec(raw)) !== null) {
      appendText(el, raw.slice(last, m.index));
      if (showThink) {
        const t = document.createElement("span");
        t.className = "think";
        t.textContent = m[1].trim();
        el.appendChild(t);
      }
      last = THINK_RE.lastIndex;
    }
    appendText(el, raw.slice(last));
    if (!el.textContent && !el.querySelector(".think")) appendText(el, "…");
  }

  function appendText(el, text) {
    if (text) el.appendChild(document.createTextNode(text));
  }

  function scrollDown() {
    const box = $("messages");
    box.scrollTop = box.scrollHeight;
  }

  function setStatus(text) { $("status").textContent = text; }

  // ---- Training (Issue #57, staff-only) -----------------------------------

  let trainWs = null;    // WebSocket for the active/attached training job
  let trainJobId = null; // id of the job being tracked

  async function initTraining() {
    // The training endpoints are staff-only (403 for everyone else) - probe
    // once and only reveal the launcher for staff, matching the backend guard.
    const res = await api("/training/");
    if (!res.ok) { $("open-training").hidden = true; return; }
    $("open-training").hidden = false;

    // Re-attach to a job still running from a previous page load.
    const data = await res.json();
    const list = data.results || data;
    const active = list.find((j) => ["queued", "running", "cancelling"].includes(j.status));
    if (active) { openTraining(); attachJob(active); }
  }

  function openTraining() {
    // Populate the model-version picker from the already-loaded list.
    const sel = $("tr-model");
    sel.innerHTML = "";
    for (const m of modelList) {
      const opt = document.createElement("option");
      opt.value = m.id; opt.textContent = m.tag + (m.is_active ? " (active)" : "");
      sel.appendChild(opt);
    }
    // Stage checkboxes (all checked by default).
    const box = $("tr-stages");
    if (!box.childElementCount) {
      for (const s of STAGES) {
        const lab = document.createElement("label");
        const cb = document.createElement("input");
        cb.type = "checkbox"; cb.value = s; cb.checked = true; cb.className = "tr-stage";
        lab.appendChild(cb); lab.appendChild(document.createTextNode(s));
        box.appendChild(lab);
      }
    }
    $("tr-err").textContent = "";
    $("training").hidden = false;
  }

  $("open-training").addEventListener("click", openTraining);
  $("tr-close").addEventListener("click", () => { $("training").hidden = true; });

  $("tr-start").addEventListener("click", async () => {
    $("tr-err").textContent = "";
    const stages = [...document.querySelectorAll(".tr-stage")].filter((c) => c.checked).map((c) => c.value);
    if (!$("tr-model").value) { $("tr-err").textContent = "Register a model version to continue from first."; return; }
    if (!stages.length) { $("tr-err").textContent = "Select at least one stage."; return; }

    const body = {
      from_model_version: $("tr-model").value,
      stages,
      epochs_per_stage: Number($("tr-epochs").value),
      seq_len: Number($("tr-seqlen").value),
      batch_size: Number($("tr-batch").value),
      lr: Number($("tr-lr").value),
      precision: $("tr-precision").value,
      smoke: $("tr-smoke").checked,
    };
    $("tr-start").disabled = true;
    const res = await api("/training/", { method: "POST", body });
    $("tr-start").disabled = false;
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      $("tr-err").textContent = data.detail || JSON.stringify(data);
      return;
    }
    attachJob(await res.json());
  });

  $("tr-cancel").addEventListener("click", async () => {
    if (!trainJobId) return;
    await api(`/training/${trainJobId}/cancel/`, { method: "POST" });
    trStatus("Cancelling after the current stage…");
  });

  function attachJob(job) {
    trainJobId = job.id;
    $("tr-progress").hidden = false;
    $("tr-start").disabled = true;
    $("tr-cancel").hidden = false;
    $("tr-log").textContent = "";
    renderJob(job);
    openTrainingSocket(job.id);
  }

  function renderJob(job) {
    trStatus(`${job.status}${job.current_stage ? " — " + job.current_stage : ""}${job.detail ? " · " + job.detail : ""}`);
    trBar(job.stages_done, job.stages_total);
    if (job.error) trLog("ERROR: " + job.error);
  }

  function openTrainingSocket(id) {
    closeTrainingSocket();
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    trainWs = new WebSocket(`${scheme}://${location.host}/ws/training/${id}/?token=${encodeURIComponent(store.access)}`);
    trainWs.onmessage = (evt) => {
      const data = JSON.parse(evt.data);
      if (data.type === "status") {
        trStatus(`${data.status}${data.detail ? " · " + data.detail : ""}`);
      } else if (data.type === "progress") {
        onTrainingEvent(data.event);
      } else if (data.type === "done") {
        finishJob(data);
      }
    };
    trainWs.onclose = (evt) => {
      if (evt.code === 4401) { logout(); }
      else if (evt.code === 4403) trStatus("You do not have access to training.");
    };
  }

  function onTrainingEvent(ev) {
    if (!ev) return;
    if (ev.event === "run_start") {
      trLog(`▶ run: ${ev.todo.join(", ") || "(nothing to do)"}`);
      trBar(ev.done.length, ev.stages.length);
    } else if (ev.event === "stage_start") {
      trStatus(`running — training '${ev.stage}' (${ev.index + 1}/${ev.total})`);
      trLog(`→ stage '${ev.stage}' started`);
    } else if (ev.event === "stage_complete") {
      const m = ev.metrics || {};
      trLog(`✓ stage '${ev.stage}' done in ${ev.seconds}s | PPL ${m.ppl} | ${m.bits_per_char} bits/char`);
      trBar(ev.index + 1, ev.total);
    } else if (ev.event === "run_complete") {
      trBar(ev.total, ev.total);
    }
  }

  async function finishJob(data) {
    trStatus(`${data.status}${data.detail ? " · " + data.detail : ""}`);
    if (data.error) trLog("ERROR: " + data.error);
    if (data.model_version_tag) trLog(`✓ registered model version: ${data.model_version_tag}`);
    $("tr-cancel").hidden = true;
    $("tr-start").disabled = false;
    trainJobId = null;
    closeTrainingSocket();
    if (data.status === "succeeded") await loadModels();  // the new checkpoint is now pickable
  }

  function closeTrainingSocket() {
    if (trainWs) { trainWs.onclose = null; trainWs.close(); trainWs = null; }
  }

  function trStatus(t) { $("tr-status").textContent = t; }
  function trLog(line) { $("tr-log").textContent += (line + "\n"); $("tr-log").scrollTop = $("tr-log").scrollHeight; }
  function trBar(done, total) { $("tr-bar").firstElementChild.style.width = total ? `${Math.round(100 * done / total)}%` : "0"; }

  // ---- Start ---------------------------------------------------------------

  if (store.access) showApp();
})();
