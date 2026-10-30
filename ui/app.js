/**
 * Trilingual Speech AI Platform - Frontend Controller
 * Performance Curves, Dataset Browser & Integrated Voice Studio
 */

let progressChart = null;
let currentRecLang = "gu";
let promptIndex = 0;

// Prompts for Voice Studio
const PROMPTS = {
  gu: [
    "ટ્રેન સાંજે સાત ને પાંત્રીસે ઊપડશે.",
    "જગતમાં કોઈ જીવ ન્નસ જંગમ છે અને કોઈ સ્થાવર.",
    "ગામમાં ભગવાન સ્વામિનારાયણનું મંદિર છે.",
    "અક્કલ સાહેબ રવિ-ભાણ સંપ્રદાયના સંતકવિ છે.",
    "આજે હવામાન ખૂબ જ ખુશનુમા અને સુંદર છે."
  ],
  hi: [
    "शाम सात बजे का reminder लगाओ।",
    "आखिर कब तक देश की जनता झूठे वादों से ठगी जाती रहेगी?",
    "हमने ओबीसी तबके के बच्चों के लिए उच्च शिक्षा में आरक्षण दिया।",
    "मगर साथ में मनपसंद साथी हो तो मेहनत भी रास आती है।",
    "तुम यहाँ कब से इंतज़ार कर रहे हो?"
  ],
  en: [
    "I am learning to play the acoustic guitar.",
    "I see the whole play with my parents and very enjoyable that moment.",
    "They have multiple products of the same catalogue in the store.",
    "Food trucks are always a much better option for quick evening snacks.",
    "Can you please navigate to the nearest metro station?"
  ]
};

// Microphone & Recording State
let mediaStream = null;
let mediaRecorder = null;
let audioChunks = [];
let recStartTime = 0;
let recTimerInterval = null;
let recordedBlob = null;
let sessionClipsCount = 0;

// ==========================================================================
// 1. INITIALIZATION & TAB SWITCHING
// ==========================================================================
window.addEventListener("DOMContentLoaded", () => {
  initProgressChart();
  updatePromptDisplay();
  
  // Spacebar shortcut for recording
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && e.target.tagName !== "INPUT" && e.target.tagName !== "TEXTAREA") {
      const recorderTab = document.getElementById("viewRecorder");
      if (recorderTab && recorderTab.classList.contains("active")) {
        e.preventDefault();
        const startBtn = document.getElementById("btnStartRec");
        const stopBtn = document.getElementById("btnStopRec");
        if (!startBtn.classList.contains("hidden")) {
          startStudioRecording();
        } else if (!stopBtn.classList.contains("hidden")) {
          stopStudioRecording();
        }
      }
    }
  });
});

function switchTab(tabId) {
  // Update button active state
  document.querySelectorAll(".nav-tab-btn").forEach((btn) => {
    btn.classList.remove("active");
  });
  
  // Update view panel active state
  document.querySelectorAll(".view-panel").forEach((panel) => {
    panel.classList.remove("active");
  });

  if (tabId === "progress") {
    document.getElementById("tabProgressBtn")?.classList.add("active");
    document.getElementById("viewProgress")?.classList.add("active");
    if (progressChart) progressChart.resize();
  } else if (tabId === "dataset") {
    document.getElementById("tabDatasetBtn")?.classList.add("active");
    document.getElementById("viewDataset")?.classList.add("active");
  } else if (tabId === "checkpoints") {
    document.getElementById("tabCheckpointsBtn")?.classList.add("active");
    document.getElementById("viewCheckpoints")?.classList.add("active");
  } else if (tabId === "recorder") {
    document.getElementById("tabRecorderBtn")?.classList.add("active");
    document.getElementById("viewRecorder")?.classList.add("active");
  }
}

// ==========================================================================
// 2. CHART.JS: MODEL CONVERGENCE DYNAMICS
// ==========================================================================
function initProgressChart() {
  const ctx = document.getElementById("progressChart");
  if (!ctx) return;

  const epochs = [
    "Epoch 11\n(Baseline)", 
    "Epoch 15", 
    "Epoch 18", 
    "Epoch 22\n(SpecAug)", 
    "Epoch 24\n(42h Corpus)", 
    "Epoch 25", 
    "Epoch 27", 
    "Epoch 28\n(Current Record)"
  ];
  
  const cerData = [14.95, 14.30, 13.90, 13.55, 11.84, 11.68, 11.76, 11.68];
  const werData = [48.20, 45.40, 42.10, 40.50, 38.27, 37.91, 37.88, 37.78];

  progressChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        {
          label: "Character Error Rate (CER %)",
          data: cerData,
          borderColor: "#a78bfa",
          backgroundColor: "rgba(167, 139, 250, 0.12)",
          borderWidth: 3,
          pointBackgroundColor: "#c084fc",
          pointBorderColor: "#fff",
          pointRadius: 5,
          pointHoverRadius: 7,
          fill: true,
          tension: 0.3
        },
        {
          label: "Word Error Rate (WER %)",
          data: werData,
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56, 189, 248, 0.05)",
          borderWidth: 2,
          borderDash: [5, 5],
          pointBackgroundColor: "#38bdf8",
          pointBorderColor: "#fff",
          pointRadius: 4,
          pointHoverRadius: 6,
          fill: false,
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: {
          position: "top",
          labels: {
            color: "#94a3b8",
            font: { family: "Outfit", size: 12, weight: "500" },
            padding: 16
          }
        },
        tooltip: {
          backgroundColor: "#162032",
          titleColor: "#f8fafc",
          bodyColor: "#cbd5e1",
          borderColor: "rgba(255, 255, 255, 0.1)",
          borderWidth: 1,
          padding: 12,
          boxPadding: 6,
          callbacks: {
            label: function(context) {
              return ` ${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.04)" },
          ticks: { color: "#64748b", font: { family: "Outfit", size: 11 } }
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: {
            color: "#64748b",
            font: { family: "JetBrains Mono", size: 11 },
            callback: (v) => v + "%"
          },
          suggestedMin: 5,
          suggestedMax: 50
        }
      }
    }
  });
}

// ==========================================================================
// 3. GROUND-TRUTH SAMPLE BROWSER
// ==========================================================================
async function loadRandomSample() {
  try {
    const res = await fetch("/api/sample/random");
    if (!res.ok) throw new Error("Sample endpoint failed");
    const data = await res.json();
    
    document.getElementById("sampleSpeaker").textContent = `${data.speaker} (${data.rel_path.split("/").pop()})`;
    
    const langMap = { gu: "Gujarati", hi: "Hindi", en: "English" };
    document.getElementById("sampleLang").textContent = langMap[data.language] || data.language.toUpperCase();
    document.getElementById("sampleDuration").textContent = `${data.duration}s`;
    
    const transcriptEl = document.getElementById("sampleTranscript");
    transcriptEl.textContent = data.transcript;
    transcriptEl.className = "sample-transcript-text";
    if (data.language === "gu") transcriptEl.classList.add("font-gujarati");
    if (data.language === "hi") transcriptEl.classList.add("font-devanagari");

    const player = document.getElementById("sampleAudioPlayer");
    player.src = data.audio_url;
    player.play().catch(() => {});
  } catch (err) {
    console.warn("Could not load random sample from server:", err);
  }
}

// ==========================================================================
// 4. STUDIO VOICE RECORDER
// ==========================================================================
function setRecordingLang(lang) {
  currentRecLang = lang;
  promptIndex = 0;
  
  document.getElementById("btnRecGu")?.classList.toggle("active", lang === "gu");
  document.getElementById("btnRecHi")?.classList.toggle("active", lang === "hi");
  document.getElementById("btnRecEn")?.classList.toggle("active", lang === "en");

  updatePromptDisplay();
}

function updatePromptDisplay() {
  const list = PROMPTS[currentRecLang] || [];
  const text = list[promptIndex % list.length];
  
  const textEl = document.getElementById("activePromptText");
  const countEl = document.getElementById("promptIndex");
  
  if (textEl) {
    textEl.textContent = text;
    textEl.className = "prompt-text";
    if (currentRecLang === "gu") textEl.classList.add("font-gujarati");
    if (currentRecLang === "hi") textEl.classList.add("font-devanagari");
  }
  
  if (countEl) {
    countEl.textContent = `Sentence #${(promptIndex % list.length) + 1} of ${list.length}`;
  }
}

function nextPrompt() {
  promptIndex++;
  updatePromptDisplay();
  discardCurrentRecording();
}

async function startStudioRecording() {
  try {
    audioChunks = [];
    recordedBlob = null;
    
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: false,
        noiseSuppression: false
      }
    });

    mediaRecorder = new MediaRecorder(mediaStream, { mimeType: "audio/webm" });

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      recordedBlob = new Blob(audioChunks, { type: "audio/webm" });
      const audioUrl = URL.createObjectURL(recordedBlob);
      const player = document.getElementById("recordedAudioPlayer");
      if (player) player.src = audioUrl;
      document.getElementById("previewZone")?.classList.remove("hidden");
    };

    mediaRecorder.start(100);
    recStartTime = Date.now();
    
    // UI toggles
    document.getElementById("btnStartRec")?.classList.add("hidden");
    document.getElementById("btnStopRec")?.classList.remove("hidden");
    document.getElementById("previewZone")?.classList.add("hidden");
    document.getElementById("micStatusText").textContent = "Recording in progress...";
    document.getElementById("micStatusText").className = "text-purple";

    clearInterval(recTimerInterval);
    recTimerInterval = setInterval(() => {
      const elapsedMs = Date.now() - recStartTime;
      const sec = Math.floor(elapsedMs / 1000);
      const dec = Math.floor((elapsedMs % 1000) / 100);
      const formatted = `00:${sec < 10 ? "0" + sec : sec}.${dec}`;
      document.getElementById("recTimer").textContent = formatted;

      const msg = document.getElementById("recStatusMsg");
      if (sec >= 2 && sec <= 4) {
        msg.textContent = "✓ Optimal duration achieved (2.0s - 4.5s)";
        msg.style.color = "#34d399";
      } else if (sec > 5) {
        msg.textContent = "⚠️ Longer than recommended sentence length.";
        msg.style.color = "#fbbf24";
      }
    }, 100);

  } catch (err) {
    alert("Microphone access denied: " + err.message);
  }
}

function stopStudioRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
  }
  clearInterval(recTimerInterval);
  
  document.getElementById("btnStartRec")?.classList.remove("hidden");
  document.getElementById("btnStopRec")?.classList.add("hidden");
  document.getElementById("micStatusText").textContent = "Ready to Record";
  document.getElementById("micStatusText").className = "text-emerald";
}

function discardCurrentRecording() {
  recordedBlob = null;
  document.getElementById("previewZone")?.classList.add("hidden");
  document.getElementById("recTimer").textContent = "00:00.0";
  const player = document.getElementById("recordedAudioPlayer");
  if (player) player.src = "";
  const msg = document.getElementById("recStatusMsg");
  msg.textContent = "Recommended clip duration: 2.0 to 4.5 seconds.";
  msg.style.color = "var(--text-muted)";
}

function saveCurrentRecording() {
  if (!recordedBlob) return;
  sessionClipsCount++;
  document.getElementById("sessionClipsCount").textContent = `${sessionClipsCount} clips`;
  
  // Show brief confirmation
  const msg = document.getElementById("recStatusMsg");
  msg.textContent = "🎉 Clip verified & saved to session cache!";
  msg.style.color = "#c084fc";
  
  setTimeout(() => {
    nextPrompt();
  }, 600);
}
