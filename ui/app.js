/**
 * Trilingual Speech Studio — Frontend Engine
 * Handles real-time microphone DSP, live waveform visualization,
 * file drag-and-drop, sample loading, and API communication.
 */

// Application State
let currentLanguage = "auto";
let currentMode = "mic";

// Microphone & DSP State
let mediaStream = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let recordStartTime = 0;
let timerInterval = null;
let audioContext = null;
let analyserNode = null;
let animationFrameId = null;

// Output State
let currentTranscription = "";
let currentSrtSubtitles = "";

// ============================================================================
// 1. INITIALIZATION & SHORTCUTS
// ============================================================================
window.addEventListener("DOMContentLoaded", () => {
  initWaveformCanvas();

  // Spacebar shortcut to start/stop recording (when not in an input/textarea)
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && e.target.tagName !== "INPUT" && e.target.tagName !== "TEXTAREA") {
      if (currentMode === "mic") {
        e.preventDefault();
        toggleRecording();
      }
    }
  });

  // Setup Drag & Drop listeners
  const dropZone = document.getElementById("dropZone");
  if (dropZone) {
    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove("dragover");
      });
    });

    dropZone.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        processSelectedFile(files[0]);
      }
    });
  }
});

// ============================================================================
// 2. LANGUAGE & MODE CONTROLLERS
// ============================================================================
function setLanguage(lang) {
  currentLanguage = lang;
  document.querySelectorAll(".lang-pill").forEach((pill) => {
    if (pill.dataset.lang === lang) {
      pill.classList.add("active");
    } else {
      pill.classList.remove("active");
    }
  });
  showToast(`Language set to ${getLanguageName(lang)}`);
}

function getLanguageName(code) {
  const map = {
    auto: "Auto-Detect",
    gu: "Gujarati (ગુજરાતી)",
    hi: "Hindi (हिन्दी)",
    en: "English"
  };
  return map[code] || code;
}

function switchMode(mode) {
  currentMode = mode;
  document.querySelectorAll(".mode-tab-btn").forEach((btn) => btn.classList.remove("active"));
  document.querySelectorAll(".input-panel").forEach((panel) => panel.classList.remove("active"));

  if (mode === "mic") {
    document.getElementById("tabMicBtn").classList.add("active");
    document.getElementById("panelMic").classList.add("active");
  } else {
    document.getElementById("tabFileBtn").classList.add("active");
    document.getElementById("panelFile").classList.add("active");
  }
}

// ============================================================================
// 3. LIVE MICROPHONE & WEBAUDIO DSP
// ============================================================================
async function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}

async function startRecording() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true
      }
    });

    audioChunks = [];
    mediaRecorder = new MediaRecorder(mediaStream);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      const mime = mediaRecorder.mimeType || "audio/webm";
      const ext = mime.includes("mp4") ? "mp4" : (mime.includes("ogg") ? "ogg" : "webm");
      const audioBlob = new Blob(audioChunks, { type: mime });
      handleRecordedAudio(audioBlob, `mic_recording.${ext}`);
    };

    mediaRecorder.start(100);
    isRecording = true;
    recordStartTime = Date.now();

    // UI Updates
    document.getElementById("micBtn").classList.add("recording");
    document.getElementById("micIconIdle").classList.add("hidden");
    document.getElementById("micIconRec").classList.remove("hidden");
    document.getElementById("micPulseRing").classList.remove("hidden");
    document.getElementById("micLabel").textContent = "Listening... (Click or Space to Stop)";
    
    // Timer
    timerInterval = setInterval(updateRecordTimer, 50);

    // Audio Visualizer
    setupAudioVisualizer(mediaStream);

  } catch (err) {
    console.error("Microphone access error:", err);
    showToast("Microphone access denied or unavailable.");
  }
}

function stopRecording() {
  if (!isRecording) return;

  isRecording = false;
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }

  clearInterval(timerInterval);
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }

  // UI Updates
  document.getElementById("micBtn").classList.remove("recording");
  document.getElementById("micIconIdle").classList.remove("hidden");
  document.getElementById("micIconRec").classList.add("hidden");
  document.getElementById("micPulseRing").classList.add("hidden");
  document.getElementById("micLabel").textContent = "Transcribing audio...";

  // Reset visualizer canvas
  drawIdleWaveform();
}

function updateRecordTimer() {
  const elapsed = Date.now() - recordStartTime;
  const minutes = Math.floor(elapsed / 60000);
  const seconds = Math.floor((elapsed % 60000) / 1000);
  const tenths = Math.floor((elapsed % 1000) / 100);

  const timerEl = document.getElementById("micTimer");
  timerEl.textContent = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${tenths}`;
}

function handleRecordedAudio(blob, filename = "mic_recording.webm") {
  // Setup audio player preview
  const audioUrl = URL.createObjectURL(blob);
  const player = document.getElementById("audioElement");
  player.src = audioUrl;
  document.getElementById("playerAudioLabel").textContent = `Microphone Recording (${document.getElementById("micTimer").textContent})`;
  document.getElementById("audioPlayerStrip").classList.remove("hidden");

  // Send for transcription
  sendAudioForTranscription(blob, filename);
}

// ============================================================================
// 4. AUDIO VISUALIZER (Web Audio API)
// ============================================================================
function initWaveformCanvas() {
  drawIdleWaveform();
}

function drawIdleWaveform() {
  const canvas = document.getElementById("waveformCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(99, 102, 241, 0.25)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();
}

function setupAudioVisualizer(stream) {
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioContext.createMediaStreamSource(stream);
  analyserNode = audioContext.createAnalyser();
  analyserNode.fftSize = 256;
  source.connect(analyserNode);

  const canvas = document.getElementById("waveformCanvas");
  const ctx = canvas.getContext("2d");
  const bufferLength = analyserNode.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function draw() {
    if (!isRecording) return;
    animationFrameId = requestAnimationFrame(draw);

    analyserNode.getByteFrequencyData(dataArray);

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const barWidth = (w / bufferLength) * 2.2;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * h * 0.85;

      const gradient = ctx.createLinearGradient(0, h, 0, 0);
      gradient.addColorStop(0, "rgba(99, 102, 241, 0.4)");
      gradient.addColorStop(1, "rgba(6, 182, 212, 0.9)");

      ctx.fillStyle = gradient;
      ctx.fillRect(x, (h - barHeight) / 2, barWidth, Math.max(2, barHeight));

      x += barWidth + 1.5;
    }
  }

  draw();
}

// ============================================================================
// 5. SAVED AUDIO & SAMPLE PICKER
// ============================================================================
function triggerFileInput() {
  document.getElementById("audioFileInput").click();
}

function handleFileSelected(event) {
  const file = event.target.files[0];
  if (file) {
    processSelectedFile(file);
  }
}

function processSelectedFile(file) {
  const audioUrl = URL.createObjectURL(file);
  const player = document.getElementById("audioElement");
  player.src = audioUrl;
  document.getElementById("playerAudioLabel").textContent = file.name;
  document.getElementById("audioPlayerStrip").classList.remove("hidden");

  sendAudioForTranscription(file, file.name);
}

async function loadSample(lang, audioUrl, label) {
  setLanguage(lang);

  const player = document.getElementById("audioElement");
  player.src = audioUrl;
  document.getElementById("playerAudioLabel").textContent = `${getLanguageName(lang)} Sample`;
  document.getElementById("audioPlayerStrip").classList.remove("hidden");
  player.play();

  // Fetch sample as blob and run inference
  try {
    setLoadingState(true);
    const response = await fetch(audioUrl);
    const blob = await response.blob();
    sendAudioForTranscription(blob, `sample_${lang}.wav`);
  } catch (err) {
    console.error("Failed to load sample audio:", err);
    setLoadingState(false);
    showToast("Failed to load sample file.");
  }
}

// ============================================================================
// 6. BACKEND API TRANSCRIPTION
// ============================================================================
async function sendAudioForTranscription(audioBlob, filename) {
  setLoadingState(true);

  const formData = new FormData();
  formData.append("file", audioBlob, filename);
  formData.append("model_type", "production_kenlm");
  formData.append("language", currentLanguage);

  try {
    const response = await fetch("/api/infer/upload", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    if (data.error && !data.predicted_text) {
      displayError(data.error);
    } else {
      displayResults(data);
    }
  } catch (err) {
    console.error("Inference request failed:", err);
    showToast("Transcription failed. Please try again.");
    displayError("Transcription encountered an error. Check server logs.");
  } finally {
    setLoadingState(false);
    document.getElementById("micLabel").textContent = "Click to Start Speaking";
  }
}

// ============================================================================
// 7. RESULTS DISPLAY & TELEMETRY
// ============================================================================
function setLoadingState(isLoading) {
  const badge = document.getElementById("outputStateBadge");
  const skeleton = document.getElementById("loadingSkeleton");
  const textEl = document.getElementById("transcriptionText");

  if (isLoading) {
    badge.className = "output-badge processing";
    badge.textContent = "Transcribing...";
    skeleton.classList.remove("hidden");
    textEl.classList.add("hidden");
  } else {
    skeleton.classList.add("hidden");
    textEl.classList.remove("hidden");
  }
}

function displayResults(data) {
  const badge = document.getElementById("outputStateBadge");
  badge.className = "output-badge complete";
  badge.textContent = "Done";

  currentTranscription = data.predicted_text || data.text || "(No speech detected)";
  currentSrtSubtitles = data.srt_subtitles || "";

  const textEl = document.getElementById("transcriptionText");
  textEl.textContent = currentTranscription;
  textEl.classList.remove("empty");

  // Telemetry strip
  document.getElementById("metricsStrip").classList.remove("hidden");
  document.getElementById("metricDuration").textContent = `${data.duration_sec || 0}s`;
  document.getElementById("metricLatency").textContent = `${data.inference_time_ms || 0} ms`;
  document.getElementById("metricLM").textContent = "KenLM 3-Gram (5.4% CER)";

  // Language script detection
  const detected = detectScriptFromText(currentTranscription);
  document.getElementById("metricLang").textContent = detected;

  // Word timestamps
  renderWordTimestamps(data.word_timestamps || []);
}

function displayError(msg) {
  const badge = document.getElementById("outputStateBadge");
  badge.className = "output-badge idle";
  badge.textContent = "Error";

  const textEl = document.getElementById("transcriptionText");
  textEl.textContent = msg;
  textEl.classList.add("empty");
}

function detectScriptFromText(text) {
  if (/[\u0A80-\u0AFF]/.test(text)) return "Gujarati (ગુજરાતી)";
  if (/[\u0900-\u097F]/.test(text)) return "Hindi (हिन्दी)";
  if (/[a-zA-Z]/.test(text)) return "English";
  return "Multilingual";
}

function renderWordTimestamps(timestamps) {
  const drawer = document.getElementById("timestampsDrawer");
  const list = document.getElementById("timestampsList");

  if (!timestamps || timestamps.length === 0) {
    drawer.classList.add("hidden");
    return;
  }

  drawer.classList.remove("hidden");
  list.innerHTML = "";

  timestamps.forEach((t) => {
    const chip = document.createElement("div");
    chip.className = "timestamp-chip";
    chip.innerHTML = `
      <time>${t.start.toFixed(2)}s - ${t.end.toFixed(2)}s</time>
      <span>${escapeHtml(t.word)}</span>
    `;
    list.appendChild(chip);
  });
}

// ============================================================================
// 8. ACTIONS: COPY & EXPORT SRT
// ============================================================================
function copyTranscription() {
  if (!currentTranscription) {
    showToast("No transcription to copy.");
    return;
  }

  navigator.clipboard.writeText(currentTranscription).then(() => {
    const btnText = document.getElementById("copyBtnText");
    const originalText = btnText.textContent;
    btnText.textContent = "Copied!";
    showToast("✓ Copied to clipboard");
    setTimeout(() => {
      btnText.textContent = originalText;
    }, 2000);
  }).catch(() => {
    showToast("Failed to copy text.");
  });
}

function downloadSrt() {
  if (!currentSrtSubtitles) {
    // Generate basic SRT if not provided directly
    if (!currentTranscription) {
      showToast("No subtitles to export.");
      return;
    }
    currentSrtSubtitles = `1\n00:00:00,000 --> 00:00:05,000\n${currentTranscription}\n`;
  }

  const blob = new Blob([currentSrtSubtitles], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `transcription_${Date.now()}.srt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  showToast("✓ Subtitles (.srt) downloaded");
}

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => {
    toast.classList.remove("show");
  }, 2400);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
