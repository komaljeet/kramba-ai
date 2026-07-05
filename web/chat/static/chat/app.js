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
  let streamingBubble = null; // assistant bubble being filled by the stream

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
    $("sidebar").hidden = true; $("main").hidden = true; $("login").hidden = false;
  }

  // ---- Bootstrap -----------------------------------------------------------

  async function showApp() {
    $("login").hidden = true; $("sidebar").hidden = false; $("main").hidden = false;
    await loadModels();
    await loadConversations();
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
        streamingBubble = null;
        setStatus("");
        $("send").disabled = false;
      }
    };
    ws.onclose = (evt) => {
      if (evt.code === 4401) { setStatus("Session expired — please log in again."); logout(); }
      else if (evt.code === 4403) setStatus("You do not have access to this conversation.");
    };
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
    if (!content || !activeId) return;

    if (!(await waitForSocket())) { setStatus("Connecting… try again in a moment."); return; }

    const emptyMsg = $("messages").querySelector(".empty");
    if (emptyMsg) emptyMsg.remove();

    addBubble("user", content);
    promptEl.value = "";
    $("send").disabled = true;
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
      $("send").disabled = false;
    }
    // Success: tokens arrive over the WebSocket; the "done" frame re-enables send.
  }

  $("send").addEventListener("click", send);
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

  // ---- Start ---------------------------------------------------------------

  if (store.access) showApp();
})();
