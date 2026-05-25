// --- Main Application Logic ---

const statusDiv = document.getElementById("status");
const statusText = statusDiv.querySelector(".status-text");
const authSection = document.getElementById("auth-section");
const appSection = document.getElementById("app-section");
const sessionEndSection = document.getElementById("session-end-section");
const restartBtn = document.getElementById("restartBtn");
const micBtn = document.getElementById("micBtn");
const cameraBtn = document.getElementById("cameraBtn");
const screenBtn = document.getElementById("screenBtn");
const disconnectBtn = document.getElementById("disconnectBtn");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const videoPreview = document.getElementById("video-preview");
const videoPlaceholder = document.getElementById("video-placeholder");
const connectBtn = document.getElementById("connectBtn");
const chatLog = document.getElementById("chat-log");
const callTimerEl = document.getElementById("call-timer");
const languageIndicator = document.getElementById("language-indicator");

let currentGeminiMessageDiv = null;
let currentUserMessageDiv = null;

// New state
let callStartTime = null;
let callTimerInterval = null;
let currentLanguage = "EN";
let productData = null;
let leadData = null;
let demoData = null;
let audioVisualizerAnimationId = null;

// Call transcript & outcome tracking
let callTranscript = []; // {role: "user"|"gemini", text: "", time: ""}
let toolCallsLog = [];   // {name, args, result}

const mediaHandler = new MediaHandler();
const geminiClient = new GeminiClient({
  onOpen: () => {
    setStatus("connected", "Connected");
    authSection.classList.add("hidden");
    appSection.classList.remove("hidden");

    startCallTimer();

    // Send initial trigger to force agent to start talking
    geminiClient.sendText(
      `Hi, I'm interested in learning about SalesBot.`
    );

    // Auto-start mic
    startMic();
  },
  onMessage: (event) => {
    if (typeof event.data === "string") {
      try {
        const msg = JSON.parse(event.data);
        handleJsonMessage(msg);
      } catch (e) {
        console.error("Parse error:", e);
      }
    } else {
      mediaHandler.playAudio(event.data);
    }
  },
  onClose: (e) => {
    console.log("WS Closed:", e);
    setStatus("disconnected", "Disconnected");
    showSessionEnd();
  },
  onError: (e) => {
    console.error("WS Error:", e);
    setStatus("error", "Connection Error");
  },
});

// --- Status Helper ---
function setStatus(className, text) {
  statusDiv.className = "status-badge " + className;
  statusText.textContent = text;
}

// --- Call Timer ---
function startCallTimer() {
  callStartTime = Date.now();
  callTimerEl.textContent = "00:00";
  callTimerInterval = setInterval(() => {
    const elapsed = Date.now() - callStartTime;
    const mins = Math.floor(elapsed / 60000).toString().padStart(2, "0");
    const secs = Math.floor((elapsed % 60000) / 1000).toString().padStart(2, "0");
    callTimerEl.textContent = `${mins}:${secs}`;
  }, 1000);
}

function stopCallTimer() {
  if (callTimerInterval) {
    clearInterval(callTimerInterval);
    callTimerInterval = null;
  }
}

function getCallDuration() {
  if (!callStartTime) return "00:00";
  const elapsed = Date.now() - callStartTime;
  const mins = Math.floor(elapsed / 60000).toString().padStart(2, "0");
  const secs = Math.floor((elapsed % 60000) / 1000).toString().padStart(2, "0");
  return `${mins}:${secs}`;
}

// --- Language Detection ---
function detectLanguage(text) {
  if (!text) return;
  let gujarati = 0, devanagari = 0, latin = 0;
  for (const ch of text) {
    const code = ch.codePointAt(0);
    if (code >= 0x0a80 && code <= 0x0aff) gujarati++;
    else if (code >= 0x0900 && code <= 0x097f) devanagari++;
    else if (code >= 0x0041 && code <= 0x007a) latin++;
  }
  const total = gujarati + devanagari + latin;
  if (total === 0) return;

  let lang = currentLanguage;
  if (gujarati > devanagari && gujarati > latin) lang = "GU";
  else if (devanagari > gujarati && devanagari > latin) lang = "HI";
  else if (latin > 5) lang = "EN";

  if (lang !== currentLanguage) {
    currentLanguage = lang;
    languageIndicator.textContent = lang;
  }
}

// --- Audio Visualizer ---
function initAudioVisualizer() {
  const canvas = document.getElementById("audio-visualizer");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  function resizeCanvas() {
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
  }
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  function draw() {
    audioVisualizerAnimationId = requestAnimationFrame(draw);
    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;

    ctx.clearRect(0, 0, width, height);

    // Subtle grid lines
    ctx.strokeStyle = "rgba(0, 212, 255, 0.04)";
    ctx.lineWidth = 1;
    for (let y = 0; y < height; y += 20) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Output (Gemini speaking) - frequency bars
    const outputAnalyser = mediaHandler.getOutputAnalyser();
    if (outputAnalyser) {
      const bufferLength = outputAnalyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      outputAnalyser.getByteFrequencyData(dataArray);

      const barCount = Math.min(bufferLength, 64);
      const barWidth = width / barCount;
      for (let i = 0; i < barCount; i++) {
        const barHeight = (dataArray[i] / 255) * height * 0.85;
        if (barHeight < 1) continue;

        const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
        gradient.addColorStop(0, "rgba(0, 212, 255, 0.7)");
        gradient.addColorStop(0.5, "rgba(124, 58, 237, 0.5)");
        gradient.addColorStop(1, "rgba(124, 58, 237, 0.1)");
        ctx.fillStyle = gradient;
        ctx.fillRect(i * barWidth + 1, height - barHeight, barWidth - 2, barHeight);
      }
    }

    // Input (user speaking) - waveform line
    const inputAnalyser = mediaHandler.getInputAnalyser();
    if (inputAnalyser) {
      const bufferLength = inputAnalyser.fftSize;
      const dataArray = new Float32Array(bufferLength);
      inputAnalyser.getFloatTimeDomainData(dataArray);

      ctx.strokeStyle = "rgba(16, 185, 129, 0.5)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      const sliceWidth = width / bufferLength;
      for (let i = 0; i < bufferLength; i++) {
        const y = (dataArray[i] * 0.5 + 0.5) * height;
        if (i === 0) ctx.moveTo(0, y);
        else ctx.lineTo(i * sliceWidth, y);
      }
      ctx.stroke();
    }
  }

  draw();
}

function stopAudioVisualizer() {
  if (audioVisualizerAnimationId) {
    cancelAnimationFrame(audioVisualizerAnimationId);
    audioVisualizerAnimationId = null;
  }
}

// --- Tool Call Renderers ---
function renderProductInfoPanel(data) {
  const el = document.getElementById("product-info-content");
  if (!el || !data) return;

  let html = `<div class="kb-results">`;
  if (data.results && data.results.length > 0) {
    html += `<div class="kb-query"><strong>Query:</strong> ${escapeHtml(data.query || "")}</div>`;
    for (const result of data.results) {
      html += `<div class="kb-chunk">${escapeHtml(result)}</div>`;
    }
  } else {
    html += `<div class="info-placeholder">No results found</div>`;
  }
  html += `</div>`;
  el.innerHTML = html;
}

function renderLeadPanel(data) {
  const el = document.getElementById("lead-info-content");
  if (!el || !data) return;

  const details = data.details || data;
  el.innerHTML = `<div class="info-grid">
    <span class="info-label">Lead ID</span>
    <span class="info-value highlight">${data.lead_id || "-"}</span>
    <span class="info-label">Contact</span>
    <span class="info-value">${details.contact_name || "-"}</span>
    <span class="info-label">Company</span>
    <span class="info-value">${details.company_name || "-"}</span>
    <span class="info-label">Use Case</span>
    <span class="info-value">${details.use_case || "-"}</span>
    <span class="info-label">Team Size</span>
    <span class="info-value">${details.team_size || "-"}</span>
    <span class="info-label">Budget</span>
    <span class="info-value">${details.budget_range || "-"}</span>
    <span class="info-label">Timeline</span>
    <span class="info-value">${details.timeline || "-"}</span>
    <span class="info-label">Status</span>
    <span class="info-value success">${details.status || "Qualified"}</span>
  </div>`;
}

function renderDemoPanel(data) {
  const panel = document.getElementById("demo-panel");
  const el = document.getElementById("demo-content");
  if (!panel || !el || !data) return;

  panel.classList.remove("hidden");

  el.innerHTML = `<div class="info-grid">
    <span class="info-label">Demo ID</span>
    <span class="info-value highlight">${data.demo_id || "-"}</span>
    <span class="info-label">Contact</span>
    <span class="info-value">${data.contact_name || "-"}</span>
    <span class="info-label">Email</span>
    <span class="info-value">${data.email || "-"}</span>
    <span class="info-label">Date & Time</span>
    <span class="info-value">${data.preferred_date || "-"} at ${data.preferred_time || "-"}</span>
    <span class="info-label">Duration</span>
    <span class="info-value">${data.duration || "30 minutes"}</span>
    <span class="info-label">Host</span>
    <span class="info-value">${data.host || "-"}</span>
    <span class="info-label">Meeting Link</span>
    <span class="info-value highlight">${data.meeting_link || "-"}</span>
  </div>`;
}

// --- Message Handling ---
function handleJsonMessage(msg) {
  if (msg.type === "status") {
    const statusMsgDiv = document.createElement("div");
    statusMsgDiv.className = "message system";
    statusMsgDiv.innerHTML = `<span class="msg-text" style="color:#f59e0b;font-style:italic;">${escapeHtml(msg.text)}</span>`;
    chatLog.appendChild(statusMsgDiv);
    chatLog.scrollTop = chatLog.scrollHeight;
    return;
  } else if (msg.type === "interrupted") {
    mediaHandler.stopAudioPlayback();
    currentGeminiMessageDiv = null;
    currentUserMessageDiv = null;
  } else if (msg.type === "turn_complete") {
    currentGeminiMessageDiv = null;
    currentUserMessageDiv = null;
  } else if (msg.type === "user") {
    if (currentUserMessageDiv) {
      const textEl = currentUserMessageDiv.querySelector(".msg-text");
      if (textEl) textEl.textContent += msg.text;
      if (callTranscript.length && callTranscript[callTranscript.length - 1].role === "user") {
        callTranscript[callTranscript.length - 1].text += msg.text;
      }
      chatLog.scrollTop = chatLog.scrollHeight;
    } else {
      currentUserMessageDiv = appendMessage("user", msg.text);
      callTranscript.push({ role: "user", text: msg.text, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) });
    }
  } else if (msg.type === "gemini") {
    if (currentGeminiMessageDiv) {
      const textEl = currentGeminiMessageDiv.querySelector(".msg-text");
      if (textEl) textEl.textContent += msg.text;
      if (callTranscript.length && callTranscript[callTranscript.length - 1].role === "gemini") {
        callTranscript[callTranscript.length - 1].text += msg.text;
      }
      chatLog.scrollTop = chatLog.scrollHeight;
    } else {
      currentGeminiMessageDiv = appendMessage("gemini", msg.text);
      callTranscript.push({ role: "gemini", text: msg.text, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) });
    }
    detectLanguage(msg.text);
  } else if (msg.type === "tool_call") {
    toolCallsLog.push({ name: msg.name, args: msg.args, result: msg.result });
    if (msg.name === "search_knowledge_base") {
      productData = msg.result;
      renderProductInfoPanel(msg.result);
    } else if (msg.name === "qualify_lead") {
      leadData = msg.result;
      renderLeadPanel(msg.result);
    } else if (msg.name === "schedule_demo") {
      demoData = msg.result;
      renderDemoPanel(msg.result);
    }
  }
}

function appendMessage(type, text) {
  const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${type}`;
  msgDiv.innerHTML = `<span class="msg-text">${escapeHtml(text)}</span><span class="msg-time">${time}</span>`;
  chatLog.appendChild(msgDiv);
  chatLog.scrollTop = chatLog.scrollHeight;
  return msgDiv;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// --- Call Me (Twilio outbound) ---
const phoneInput = document.getElementById("phoneInput");
const callMeBtn = document.getElementById("callMeBtn");
const callMeStatus = document.getElementById("callMeStatus");

if (callMeBtn) {
  callMeBtn.onclick = async () => {
    let phone = phoneInput.value.trim();
    if (!phone) {
      callMeStatus.textContent = "Enter your phone number";
      callMeStatus.className = "call-me-status error";
      return;
    }
    // Auto-add +91 if user just typed digits
    phone = phone.replace(/[\s\-()]/g, "");
    if (!phone.startsWith("+")) {
      phone = "+91" + phone;
    }

    callMeBtn.disabled = true;
    callMeStatus.textContent = "Calling...";
    callMeStatus.className = "call-me-status loading";

    try {
      const res = await fetch("/call-me", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phone }),
      });
      const data = await res.json();
      if (data.success) {
        callMeStatus.textContent = "Calling " + phone + " - pick up your phone! Opening live transcript...";
        callMeStatus.className = "call-me-status success";
        // Open live transcript dashboard in new tab
        setTimeout(() => window.open("/live", "_blank"), 1000);
      } else {
        callMeStatus.textContent = data.error || "Failed to call";
        callMeStatus.className = "call-me-status error";
      }
    } catch (e) {
      callMeStatus.textContent = "Network error: " + e.message;
      callMeStatus.className = "call-me-status error";
    }
    callMeBtn.disabled = false;
  };

  // Enter key triggers call
  if (phoneInput) {
    phoneInput.onkeypress = (e) => {
      if (e.key === "Enter") callMeBtn.click();
    };
  }
}

// --- Connect ---
connectBtn.onclick = async () => {
  setStatus("disconnected", "Connecting...");
  connectBtn.disabled = true;

  try {
    await mediaHandler.initializeAudio();
    geminiClient.connect();
  } catch (error) {
    console.error("Connection error:", error);
    setStatus("error", "Failed: " + error.message);
    connectBtn.disabled = false;
  }
};

// --- Mic ---
async function startMic() {
  try {
    await mediaHandler.startAudio((data) => {
      if (geminiClient.isConnected()) {
        geminiClient.send(data);
      }
    });
    micBtn.classList.add("active");
    micBtn.dataset.active = "true";

    // Start visualizer after mic is active (analyser nodes ready)
    initAudioVisualizer();
  } catch (e) {
    console.error("Could not start audio capture", e);
  }
}

micBtn.onclick = async () => {
  if (mediaHandler.isRecording) {
    mediaHandler.stopAudio();
    micBtn.classList.remove("active");
    micBtn.dataset.active = "false";
  } else {
    await startMic();
  }
};

// --- Camera ---
cameraBtn.onclick = async () => {
  if (cameraBtn.dataset.active === "true") {
    mediaHandler.stopVideo(videoPreview);
    cameraBtn.classList.remove("active");
    cameraBtn.dataset.active = "false";
    screenBtn.dataset.active = "false";
    screenBtn.classList.remove("active");
    videoPlaceholder.classList.remove("hidden");
  } else {
    if (mediaHandler.videoStream) {
      mediaHandler.stopVideo(videoPreview);
      screenBtn.classList.remove("active");
      screenBtn.dataset.active = "false";
    }
    try {
      await mediaHandler.startVideo(videoPreview, (base64Data) => {
        if (geminiClient.isConnected()) {
          geminiClient.sendImage(base64Data);
        }
      });
      cameraBtn.classList.add("active");
      cameraBtn.dataset.active = "true";
      videoPlaceholder.classList.add("hidden");
    } catch (e) {
      console.error("Could not access camera", e);
    }
  }
};

// --- Screen Share ---
screenBtn.onclick = async () => {
  if (screenBtn.dataset.active === "true") {
    mediaHandler.stopVideo(videoPreview);
    screenBtn.classList.remove("active");
    screenBtn.dataset.active = "false";
    cameraBtn.dataset.active = "false";
    cameraBtn.classList.remove("active");
    videoPlaceholder.classList.remove("hidden");
  } else {
    if (mediaHandler.videoStream) {
      mediaHandler.stopVideo(videoPreview);
      cameraBtn.classList.remove("active");
      cameraBtn.dataset.active = "false";
    }
    try {
      await mediaHandler.startScreen(
        videoPreview,
        (base64Data) => {
          if (geminiClient.isConnected()) {
            geminiClient.sendImage(base64Data);
          }
        },
        () => {
          screenBtn.classList.remove("active");
          screenBtn.dataset.active = "false";
          videoPlaceholder.classList.remove("hidden");
        }
      );
      screenBtn.classList.add("active");
      screenBtn.dataset.active = "true";
      videoPlaceholder.classList.add("hidden");
    } catch (e) {
      console.error("Could not share screen", e);
    }
  }
};

// --- Text Input ---
sendBtn.onclick = sendText;
textInput.onkeypress = (e) => {
  if (e.key === "Enter") sendText();
};

function sendText() {
  const text = textInput.value;
  if (text && geminiClient.isConnected()) {
    geminiClient.sendText(text);
    appendMessage("user", text);
    textInput.value = "";
  }
}

// --- Disconnect ---
disconnectBtn.onclick = () => {
  geminiClient.disconnect();
};

// --- Session End ---
function showSessionEnd() {
  appSection.classList.add("hidden");
  sessionEndSection.classList.remove("hidden");
  mediaHandler.stopAudio();
  mediaHandler.stopVideo(videoPreview);
  stopCallTimer();
  stopAudioVisualizer();

  const summaryEl = document.getElementById("call-summary");
  if (!summaryEl) return;

  // Determine call outcome
  const outcome = demoData ? "Demo Scheduled" : (leadData ? "Lead Qualified" : "Information Provided");
  const outcomeClass = demoData ? "outcome-success" : "outcome-neutral";

  let html = "";

  // Outcome badge
  html += `<div class="outcome-badge ${outcomeClass}">${outcome}</div>`;

  // Call overview
  html += `<div class="summary-section">
    <div class="summary-section-title">Call Overview</div>`;
  html += summaryItem("Duration", getCallDuration());
  html += summaryItem("Language", langLabel(currentLanguage));
  html += summaryItem("Messages", `${callTranscript.length} exchanges`);
  html += summaryItem("Tool Calls", `${toolCallsLog.length}`);
  html += `</div>`;

  // Lead info
  if (leadData) {
    const details = leadData.details || leadData;
    html += `<div class="summary-section">
      <div class="summary-section-title">Lead Information</div>`;
    html += summaryItem("Lead ID", leadData.lead_id);
    html += summaryItem("Contact", details.contact_name);
    html += summaryItem("Company", details.company_name);
    html += summaryItem("Use Case", details.use_case);
    html += summaryItem("Team Size", details.team_size);
    html += summaryItem("Timeline", details.timeline);
    html += `</div>`;
  }

  // Demo details
  if (demoData) {
    html += `<div class="summary-section">
      <div class="summary-section-title">Demo Details</div>`;
    html += summaryItem("Demo ID", demoData.demo_id);
    html += summaryItem("Date", demoData.preferred_date);
    html += summaryItem("Time", demoData.preferred_time);
    html += summaryItem("Host", demoData.host);
    html += summaryItem("Meeting Link", demoData.meeting_link);
    html += summaryItem("Duration", demoData.duration);
    html += `</div>`;
  }

  // Full transcript (collapsible)
  if (callTranscript.length > 0) {
    html += `<div class="summary-section">
      <button class="transcript-toggle" onclick="this.parentElement.classList.toggle('expanded')">
        <span class="toggle-icon">&#9654;</span> View Full Transcript (${callTranscript.length} messages)
      </button>
      <div class="transcript-full">`;
    for (const entry of callTranscript) {
      const roleLabel = entry.role === "user" ? "You" : "Aria";
      const roleClass = entry.role === "user" ? "tr-user" : "tr-gemini";
      html += `<div class="tr-line ${roleClass}">
        <span class="tr-time">${entry.time}</span>
        <span class="tr-role">${roleLabel}:</span>
        <span class="tr-text">${escapeHtml(entry.text)}</span>
      </div>`;
    }
    html += `</div></div>`;
  }

  summaryEl.innerHTML = html;
}

function summaryItem(label, value) {
  return `<div class="summary-item"><span class="summary-label">${label}</span><span class="summary-value">${value || "-"}</span></div>`;
}

function langLabel(code) {
  const map = { HI: "Hindi", EN: "English", GU: "Gujarati", MR: "Marathi" };
  return map[code] || code;
}

// --- Reset ---
function resetUI() {
  authSection.classList.remove("hidden");
  appSection.classList.add("hidden");
  sessionEndSection.classList.add("hidden");

  mediaHandler.stopAudio();
  mediaHandler.stopVideo(videoPreview);
  videoPlaceholder.classList.remove("hidden");
  stopCallTimer();
  stopAudioVisualizer();

  micBtn.classList.remove("active");
  micBtn.dataset.active = "false";
  cameraBtn.classList.remove("active");
  cameraBtn.dataset.active = "false";
  screenBtn.classList.remove("active");
  screenBtn.dataset.active = "false";

  chatLog.innerHTML = "";
  connectBtn.disabled = false;
  callTimerEl.textContent = "00:00";
  languageIndicator.textContent = "EN";
  currentLanguage = "EN";
  productData = null;
  leadData = null;
  demoData = null;
  currentGeminiMessageDiv = null;
  currentUserMessageDiv = null;
  callTranscript = [];
  toolCallsLog = [];

  // Reset info panels
  const pic = document.getElementById("product-info-content");
  if (pic) pic.innerHTML = '<div class="info-placeholder">Ask about SalesBot features...</div>';
  const lic = document.getElementById("lead-info-content");
  if (lic) lic.innerHTML = '<div class="info-placeholder">No lead info yet</div>';
  const dp = document.getElementById("demo-panel");
  if (dp) dp.classList.add("hidden");

  // Reset mobile state
  const colRight = document.querySelector(".col-right");
  if (colRight) colRight.classList.remove("mobile-expanded");
  const videoPanel = document.querySelector(".video-panel");
  if (videoPanel) videoPanel.classList.remove("mobile-video-active");
  document.querySelectorAll(".mobile-tab").forEach((t, i) => {
    t.classList.toggle("active", i === 0);
  });
  // Show all panels again (desktop mode)
  Object.values({ p: "product-info-panel", l: "lead-info-panel", d: "demo-panel" }).forEach((id) => {
    const p = document.getElementById(id);
    if (p) p.style.display = "";
  });
}

restartBtn.onclick = () => {
  resetUI();
};

// --- Mobile: Tab switching for info panels ---
(function initMobileTabs() {
  const tabs = document.querySelectorAll(".mobile-tab");
  const colRight = document.querySelector(".col-right");
  const panelMap = {
    product: "product-info-panel",
    lead: "lead-info-panel",
    demo: "demo-panel",
  };

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      // Toggle expand/collapse
      if (tab.classList.contains("active") && colRight.classList.contains("mobile-expanded")) {
        colRight.classList.remove("mobile-expanded");
        return;
      }

      // Switch active tab
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      colRight.classList.add("mobile-expanded");

      // Show only selected panel
      const selectedId = panelMap[tab.dataset.tab];
      Object.values(panelMap).forEach((id) => {
        const panel = document.getElementById(id);
        if (!panel) return;
        if (id === selectedId) {
          panel.style.display = "";
        } else {
          panel.style.display = "none";
        }
      });
    });
  });
})();

// --- Mobile: Video PiP (picture-in-picture style) ---
function updateMobileVideoState(isActive) {
  const videoPanel = document.querySelector(".video-panel");
  if (!videoPanel) return;
  if (window.innerWidth <= 768) {
    if (isActive) {
      videoPanel.classList.add("mobile-video-active");
    } else {
      videoPanel.classList.remove("mobile-video-active");
    }
  }
}

// Patch camera/screen handlers to update mobile video state
const origCameraClick = cameraBtn.onclick;
cameraBtn.onclick = async () => {
  await origCameraClick();
  updateMobileVideoState(cameraBtn.dataset.active === "true");
};

const origScreenClick = screenBtn.onclick;
screenBtn.onclick = async () => {
  await origScreenClick();
  updateMobileVideoState(screenBtn.dataset.active === "true");
};
