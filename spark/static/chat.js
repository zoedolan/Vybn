/* ========================================================
   Vybn App — client logic
   ======================================================== */

let ws = null;
let token = "";
let mediaRecorder = null;
let audioChunks = [];
let reconnectDelay = 1000;
const MAX_RECONNECT_DELAY = 30000;

// ---- DOM refs ----
const loginScreen = document.getElementById("login-screen");
const chatScreen  = document.getElementById("chat-screen");
const tokenInput  = document.getElementById("token-input");
const messages    = document.getElementById("messages");
const msgInput    = document.getElementById("msg-input");
const statusText  = document.getElementById("status-text");
const micBtn      = document.getElementById("mic-btn");
const recInd      = document.getElementById("recording-indicator");
const menuOverlay = document.getElementById("menu-overlay");

// ---- Haptic feedback ----
function haptic(style = "light") {
  if (navigator.vibrate) {
    const patterns = { light: [10], medium: [20], heavy: [30, 10, 30] };
    navigator.vibrate(patterns[style] || [10]);
  }
}

// ---- Token persistence (stays on device, never sent to server beyond auth) ----
function saveToken(t) {
  try { localStorage.setItem("vybn_token", t); } catch(e) {}
}
function loadToken() {
  try { return localStorage.getItem("vybn_token") || ""; } catch(e) { return ""; }
}
function clearToken() {
  try { localStorage.removeItem("vybn_token"); } catch(e) {}
}

// ---- Auto-login if token saved ----
(function autoLogin() {
  const saved = loadToken();
  if (saved) {
    token = saved;
    loginScreen.classList.remove("active");
    chatScreen.classList.add("active");
    connectWS();
  }
})();

// ---- Login ----
function doLogin() {
  token = tokenInput.value.trim();
  if (!token) return;
  saveToken(token);
  loginScreen.classList.remove("active");
  chatScreen.classList.add("active");
  haptic("medium");
  connectWS();
}
tokenInput.addEventListener("keydown", e => { if (e.key === "Enter") doLogin(); });

function doLogout() {
  if (ws) ws.close();
  clearToken();
  token = "";
  chatScreen.classList.remove("active");
  loginScreen.classList.add("active");
  tokenInput.value = "";
  messages.innerHTML = "";
  toggleMenu();
}

// ---- WebSocket ----
function connectWS() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws?token=${encodeURIComponent(token)}`);

  ws.onopen = () => {
    statusText.textContent = "connected";
    statusText.style.color = "#5cffb2";
    reconnectDelay = 1000; // reset on success
  };
  
  ws.onclose = (e) => {
    statusText.textContent = "reconnecting…";
    statusText.style.color = "#ff5c6c";
    // Exponential backoff with jitter
    const jitter = Math.random() * 1000;
    setTimeout(connectWS, reconnectDelay + jitter);
    reconnectDelay = Math.min(reconnectDelay * 1.5, MAX_RECONNECT_DELAY);
  };
  
  ws.onerror = () => {};
  
  ws.onmessage = (evt) => {
    const data = JSON.parse(evt.data);
    if (data.type === "message") {
      hideThinking();
      appendMessage(data);
      haptic("light"); // subtle buzz when Vybn responds
    } else if (data.type === "status") {
      if (data.status === "thinking") showThinking();
      else if (data.status === "ready") hideThinking();
    }
  };
}

// ---- Simple markdown rendering ----
function renderMarkdown(text) {
  // Escape HTML first
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  
  // Code blocks
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Line breaks (but not inside pre tags)
  html = html.replace(/\n/g, '<br>');
  // Fix pre tags — remove extra <br> inside them
  html = html.replace(/<pre><code>([\s\S]*?)<\/code><\/pre>/g, (match, code) => {
    return '<pre><code>' + code.replace(/<br>/g, '\n') + '</code></pre>';
  });
  
  return html;
}

// ---- Messages ----
function appendMessage(data) {
  // Remove typing indicator if present
  const thinking = messages.querySelector(".thinking-indicator");
  if (thinking) thinking.remove();

  const div = document.createElement("div");
  const isUser = data.role === "user";
  div.className = `msg ${isUser ? "user" : "vybn"}`;

  if (isUser) {
    div.textContent = data.content;
  } else {
    div.innerHTML = renderMarkdown(data.content);
  }

  if (data.ts) {
    const ts = document.createElement("span");
    ts.className = "timestamp";
    const d = new Date(data.ts);
    ts.textContent = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    div.appendChild(ts);
  }
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

function showThinking() {
  if (messages.querySelector(".thinking-indicator")) return;
  const div = document.createElement("div");
  div.className = "thinking-indicator";
  div.innerHTML = '<div class="thinking-content"><span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-label">thinking</span></div>';
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  statusText.textContent = "thinking…";
  statusText.style.color = "#c4a0ff";
}

function hideThinking() {
  const indicators = messages.querySelectorAll(".thinking-indicator");
  indicators.forEach(el => el.remove());
  statusText.textContent = "connected";
  statusText.style.color = "#5cffb2";
}

// ---- Send ----
function sendMessage() {
  const text = msgInput.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ message: text }));
  
  
  msgInput.value = "";
  msgInput.style.height = "auto";
  haptic("light");
  showThinking();
}

function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoGrow(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 120) + "px";
}

// ---- Voice ----
async function toggleVoice() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    micBtn.classList.remove("recording");
    recInd.classList.add("hidden");
    haptic("medium");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(audioChunks, { type: "audio/webm" });
      await transcribeAndSend(blob);
    };
    mediaRecorder.start();
    micBtn.classList.add("recording");
    recInd.classList.remove("hidden");
    haptic("heavy");
  } catch (err) {
    alert("Microphone access denied or unavailable.");
  }
}

async function transcribeAndSend(blob) {
  const form = new FormData();
  form.append("audio", blob, "voice.webm");
  try {
    const resp = await fetch("/voice", {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: form,
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      alert(err.detail || "Voice transcription unavailable");
      return;
    }
    const { text } = await resp.json();
    if (text) {
      msgInput.value = text;
      sendMessage();
    }
  } catch (e) {
    alert("Transcription request failed.");
  }
}

// ---- History ----
async function loadHistory() {
  toggleMenu();
  try {
    const resp = await fetch(`/history?token=${encodeURIComponent(token)}`);
    const data = await resp.json();
    messages.innerHTML = "";
    data.forEach(appendMessage);
  } catch (e) { /* ignore */ }
}

function clearChat() {
  messages.innerHTML = "";
  toggleMenu();
}

// ---- Menu ----
function toggleMenu() {
  menuOverlay.classList.toggle("hidden");
}

// ---- Keep connection alive ----
setInterval(() => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "ping" }));
  }
}, 30000);

// ---- Focus input when chat screen shown ----
const observer = new MutationObserver(() => {
  if (chatScreen.classList.contains("active")) {
    setTimeout(() => msgInput.focus(), 100);
  }
});
observer.observe(chatScreen, { attributes: true, attributeFilter: ['class'] });
