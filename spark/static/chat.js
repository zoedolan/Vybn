/* ========================================================
   Vybn Chat — client logic
   ======================================================== */

let ws = null;
let token = "";
let mediaRecorder = null;
let audioChunks = [];

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

// ---- Login ----
function doLogin() {
  token = tokenInput.value.trim();
  if (!token) return;
  loginScreen.classList.remove("active");
  chatScreen.classList.add("active");
  connectWS();
}
tokenInput.addEventListener("keydown", e => { if (e.key === "Enter") doLogin(); });

function doLogout() {
  if (ws) ws.close();
  chatScreen.classList.remove("active");
  loginScreen.classList.add("active");
  tokenInput.value = "";
  messages.innerHTML = "";
  toggleMenu();
}

// ---- WebSocket ----
function connectWS() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws?token=${encodeURIComponent(token)}`);

  ws.onopen = () => {
    statusText.textContent = "connected";
    statusText.style.color = "#5cffb2";
  };
  ws.onclose = () => {
    statusText.textContent = "disconnected";
    statusText.style.color = "#ff5c6c";
    setTimeout(connectWS, 3000);
  };
  ws.onerror = () => {};
  ws.onmessage = (evt) => {
    const data = JSON.parse(evt.data);
    if (data.type === "message") appendMessage(data);
  };
}

// ---- Messages ----
function appendMessage(data) {
  // Remove typing indicator if present
  const typing = messages.querySelector(".typing-indicator");
  if (typing) typing.remove();

  const div = document.createElement("div");
  div.className = `msg ${data.role === "user" ? "user" : "vybn"}`;

  const text = document.createTextNode(data.content);
  div.appendChild(text);

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

function showTyping() {
  if (messages.querySelector(".typing-indicator")) return;
  const div = document.createElement("div");
  div.className = "typing-indicator";
  div.innerHTML = "<span></span><span></span><span></span>";
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

// ---- Send ----
function sendMessage() {
  const text = msgInput.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ message: text }));
  msgInput.value = "";
  msgInput.style.height = "auto";
  showTyping();
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
      // Whisper not available — fall back gracefully
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
