/**
 * STT Voice Data Collection Platform
 * Pure Vanilla JavaScript | MediaRecorder API | IndexedDB | Web Audio API
 */

// Global State
let db = null;
let sentences = [];
let currentIndex = 0;
let uploadedFiles = [];

// MediaRecorder & Audio State
let mediaStream = null;
let mediaRecorder = null;
let audioChunks = [];
let currentRecordedBlob = null;
let currentRecordedDuration = 0;
let recordingStartTime = 0;
let timerInterval = null;

// Web Audio API Visualizer State
let audioContext = null;
let analyserNode = null;
let visualizerAnimationId = null;

// ==========================================================================
// 1. INDEXEDDB STORAGE (Persistent Browser Database)
// ==========================================================================
const DB_NAME = "STT_Voice_Collection_DB";
const DB_VERSION = 1;

function initIndexedDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (e) => {
      const database = e.target.result;
      if (!database.objectStoreNames.contains("sentences")) {
        database.createObjectStore("sentences", { keyPath: "id" });
      }
      if (!database.objectStoreNames.contains("files")) {
        database.createObjectStore("files", { keyPath: "name" });
      }
      if (!database.objectStoreNames.contains("appState")) {
        database.createObjectStore("appState", { keyPath: "key" });
      }
    };

    request.onsuccess = (e) => {
      db = e.target.result;
      resolve(db);
    };

    request.onerror = (e) => {
      console.error("IndexedDB error:", e);
      reject(e);
    };
  });
}

async function saveSentenceToDB(sentenceObj) {
  if (!db) return;
  return new Promise((resolve, reject) => {
    const tx = db.transaction("sentences", "readwrite");
    const store = tx.objectStore("sentences");
    store.put(sentenceObj);
    tx.oncomplete = () => resolve();
    tx.onerror = (e) => reject(e);
  });
}

async function loadAllSentencesFromDB() {
  if (!db) return [];
  return new Promise((resolve, reject) => {
    const tx = db.transaction("sentences", "readonly");
    const store = tx.objectStore("sentences");
    const request = store.getAll();
    request.onsuccess = () => resolve(request.result || []);
    request.onerror = (e) => reject(e);
  });
}

async function saveAppState(key, value) {
  if (!db) return;
  const tx = db.transaction("appState", "readwrite");
  tx.objectStore("appState").put({ key, value });
}

async function loadAppState(key) {
  if (!db) return null;
  return new Promise((resolve) => {
    const tx = db.transaction("appState", "readonly");
    const req = tx.objectStore("appState").get(key);
    req.onsuccess = () => resolve(req.result ? req.result.value : null);
    req.onerror = () => resolve(null);
  });
}

// ==========================================================================
// 2. TEXT EXTRACTION & UPLOAD LOGIC
// ==========================================================================
function detectLanguage(fileName) {
  const lower = fileName.toLowerCase();
  if (lower.includes("english") || lower.includes("eng")) return "English";
  if (lower.includes("hindi") || lower.includes("hin")) return "Hindi";
  if (lower.includes("gujarati") || lower.includes("gujrati") || lower.includes("guj")) return "Gujarati";
  return "English";
}

function extractSentencesFromContent(content) {
  const lines = content.split(/\r?\n/);
  const extracted = [];

  for (let line of lines) {
    line = line.trim();
    if (!line) continue;
    // Skip Markdown headers (#), separators (---) or prompt note lines
    if (line.startsWith("#") || line.startsWith("---") || line.startsWith("- ")) continue;
    
    // Remove leading numbering like "1. ", "1) " if present
    const cleaned = line.replace(/^\d+[\.\)]\s*/, "").trim();
    if (cleaned.length > 0) {
      extracted.push(cleaned);
    }
  }

  return extracted;
}

async function processUploadedFile(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = async (e) => {
      const content = e.target.result;
      const detectedLang = detectLanguage(file.name);
      const extractedLines = extractSentencesFromContent(content);

      const langCodeMap = { English: "ENG", Hindi: "HIN", Gujarati: "GUJ" };
      const langPrefix = langCodeMap[detectedLang] || "STT";

      const newSentences = extractedLines.map((text, idx) => {
        const idNumber = String(sentences.length + idx + 1).padStart(6, "0");
        return {
          id: `${langPrefix}-${idNumber}`,
          language: detectedLang,
          text: text,
          duration: 0.0,
          status: "pending",
          audioBlob: null,
          timestamp: null,
          fileName: file.name
        };
      });

      uploadedFiles.push({
        name: file.name,
        language: detectedLang,
        count: extractedLines.length,
        status: "Ready"
      });

      // Append and save to IndexedDB
      sentences.push(...newSentences);
      for (const s of newSentences) {
        await saveSentenceToDB(s);
      }

      resolve();
    };
    reader.readAsText(file, "UTF-8");
  });
}

async function handleFileSelect(fileList) {
  if (!fileList || fileList.length === 0) return;
  showToast("Processing uploaded text files...");

  for (let i = 0; i < fileList.length; i++) {
    await processUploadedFile(fileList[i]);
  }

  renderUploadedFilesTable();
  updateProgressAndStats();
  displayCurrentSentence();
  showToast(`${fileList.length} file(s) imported successfully!`);
}

function renderUploadedFilesTable() {
  const tbody = document.getElementById("uploadedFilesTableBody");
  if (!tbody) return;

  if (uploadedFiles.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="4" style="text-align: center; color: var(--text-muted); padding: 32px;">
          No files uploaded yet. Upload .txt files or click "Load All 6 Prompt Files" above.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = uploadedFiles.map((f, idx) => `
    <tr>
      <td><strong>${escapeHtml(f.name)}</strong></td>
      <td>
        <select class="lang-select" onchange="updateFileLanguage(${idx}, this.value)">
          <option value="English" ${f.language === "English" ? "selected" : ""}>English</option>
          <option value="Hindi" ${f.language === "Hindi" ? "selected" : ""}>Hindi</option>
          <option value="Gujarati" ${f.language === "Gujarati" ? "selected" : ""}>Gujarati</option>
        </select>
      </td>
      <td>${f.count} sentences</td>
      <td><span class="status-tag recorded">✓ ${f.status}</span></td>
    </tr>
  `).join("");
}

async function updateFileLanguage(fileIdx, newLang) {
  const fileObj = uploadedFiles[fileIdx];
  if (!fileObj) return;
  fileObj.language = newLang;

  // Update in sentences list
  sentences.forEach((s) => {
    if (s.fileName === fileObj.name) {
      s.language = newLang;
      saveSentenceToDB(s);
    }
  });

  displayCurrentSentence();
  updateProgressAndStats();
  showToast(`Updated language for ${fileObj.name} to ${newLang}`);
}

// ⚡ 1-Click Load 6 Project Prompt Files Preset
async function loadProjectPrompts() {
  const projectPromptFiles = [
    { name: "prompts_english1.md", lang: "English" },
    { name: "prompts_english2.md", lang: "English" },
    { name: "prompts_gujarati1.md", lang: "Gujarati" },
    { name: "prompts_gujarati2.md", lang: "Gujarati" },
    { name: "prompts_hindi1.md", lang: "Hindi" },
    { name: "prompts_hindi2.md", lang: "Hindi" }
  ];

  showToast("Loading 6 project prompt files...");

  try {
    for (const p of projectPromptFiles) {
      const res = await fetch(p.name);
      if (!res.ok) continue;
      const text = await res.text();
      const extractedLines = extractSentencesFromContent(text);

      const langCodeMap = { English: "ENG", Hindi: "HIN", Gujarati: "GUJ" };
      const langPrefix = langCodeMap[p.lang] || "STT";

      const newSentences = extractedLines.map((lineText, idx) => {
        const idNumber = String(sentences.length + idx + 1).padStart(6, "0");
        return {
          id: `${langPrefix}-${idNumber}`,
          language: p.lang,
          text: lineText,
          duration: 0.0,
          status: "pending",
          audioBlob: null,
          timestamp: null,
          fileName: p.name
        };
      });

      uploadedFiles.push({
        name: p.name,
        language: p.lang,
        count: extractedLines.length,
        status: "Ready"
      });

      sentences.push(...newSentences);
      for (const s of newSentences) {
        await saveSentenceToDB(s);
      }
    }

    renderUploadedFilesTable();
    updateProgressAndStats();
    displayCurrentSentence();
    showToast(`Loaded ${sentences.length} sentences from 6 prompt files!`);
  } catch (err) {
    console.error("Error loading project files:", err);
    showToast("Error loading project prompt files: " + err.message);
  }
}

// ==========================================================================
// 3. SENTENCE QUEUE & UI DISPLAY
// ==========================================================================
function displayCurrentSentence() {
  const sentenceDisplay = document.getElementById("sentenceDisplay");
  const sentenceIdLabel = document.getElementById("sentenceIdLabel");
  const currentLangBadge = document.getElementById("currentLangBadge");
  const counterText = document.getElementById("counterText");
  const playbackAudio = document.getElementById("playbackAudio");
  const audioPlayerBox = document.getElementById("audioPlayerBox");
  const reRecordBtn = document.getElementById("reRecordBtn");
  const validationNotice = document.getElementById("validationNotice");

  resetRecordingUI();

  if (sentences.length === 0) {
    sentenceDisplay.textContent = 'Please upload text files or click "Load All 6 Prompt Files" in the Upload tab.';
    sentenceDisplay.className = "sentence-display";
    sentenceIdLabel.textContent = "ID: ---";
    currentLangBadge.textContent = "Waiting";
    counterText.textContent = "0 / 0";
    return;
  }

  if (currentIndex < 0) currentIndex = 0;
  if (currentIndex >= sentences.length) currentIndex = sentences.length - 1;

  const currentItem = sentences[currentIndex];

  // Update UI Elements
  sentenceDisplay.textContent = currentItem.text;
  sentenceDisplay.className = `sentence-display lang-${currentItem.language}`;
  sentenceIdLabel.textContent = `ID: ${currentItem.id}`;
  currentLangBadge.textContent = currentItem.language;
  currentLangBadge.className = `language-badge pill-${currentItem.language}`;
  counterText.textContent = `${currentIndex + 1} / ${sentences.length}`;

  saveAppState("currentIndex", currentIndex);
  updateProgressAndStats();

  // If this sentence was already recorded, load its audio into the player
  if (currentItem.audioBlob) {
    const audioUrl = URL.createObjectURL(currentItem.audioBlob);
    playbackAudio.src = audioUrl;
    audioPlayerBox.classList.add("show");
    reRecordBtn.style.display = "inline-flex";

    // Show duration feedback
    updateDurationFeedback(currentItem.duration);
  } else {
    audioPlayerBox.classList.remove("show");
    reRecordBtn.style.display = "none";
    validationNotice.className = "validation-notice hide";
  }
}

function updateProgressAndStats() {
  const total = sentences.length;
  const recorded = sentences.filter((s) => s.status === "recorded").length;
  const remaining = total - recorded;
  const percent = total > 0 ? (recorded / total) * 100 : 0;

  const progressBarFill = document.getElementById("progressBarFill");
  const statRecordedCount = document.getElementById("statRecordedCount");
  const statRemainingCount = document.getElementById("statRemainingCount");

  if (progressBarFill) progressBarFill.style.width = `${percent}%`;
  if (statRecordedCount) statRecordedCount.textContent = recorded;
  if (statRemainingCount) statRemainingCount.textContent = remaining;
}

// ==========================================================================
// 4. MICROPHONE RECORDING SYSTEM (MediaRecorder & AudioContext)
// ==========================================================================
let autoAdvanceOnStop = false;

async function toggleRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    // Single step: Stop, Save & Next immediately!
    autoAdvanceOnStop = true;
    stopRecording();
  } else {
    autoAdvanceOnStop = false;
    await startRecording();
  }
}

async function startRecording() {
  if (sentences.length === 0) {
    showToast("Please upload or load sentences first!");
    return;
  }

  // Reuse existing mediaStream if already granted and active
  if (!mediaStream || !mediaStream.active || mediaStream.getAudioTracks().every(t => t.readyState === "ended")) {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      });
    } catch (err) {
      console.error("Microphone permission error:", err);
      showToast("Microphone access is required for recording. Please allow permission.");
      return;
    }
  }

  // Setup Web Audio Analyser for live visualizer (if not already created)
  try {
    if (!audioContext || audioContext.state === "closed") {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }
    if (!analyserNode) {
      const sourceNode = audioContext.createMediaStreamSource(mediaStream);
      analyserNode = audioContext.createAnalyser();
      analyserNode.fftSize = 128;
      sourceNode.connect(analyserNode);
    }
    startWaveformVisualizer();
  } catch (e) {
    console.warn("AudioContext visualizer not supported:", e);
  }

  audioChunks = [];
  mediaRecorder = new MediaRecorder(mediaStream);

  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) {
      audioChunks.push(e.data);
    }
  };

  mediaRecorder.onstop = async () => {
    stopWaveformVisualizer();
    currentRecordedBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });

    // Calculate duration in seconds
    currentRecordedDuration = Number(((Date.now() - recordingStartTime) / 1000).toFixed(2));

    // If Stop & Save Next was triggered, commit and move to next sentence immediately
    if (autoAdvanceOnStop) {
      autoAdvanceOnStop = false;
      await commitSaveAndNext();
    } else {
      onRecordingFinished();
    }
  };

  recordingStartTime = Date.now();
  mediaRecorder.start(100);

  // Update UI to Recording State
  const recordToggleBtn = document.getElementById("recordToggleBtn");
  const recordBtnLabel = document.getElementById("recordBtnLabel");
  const recIndicator = document.getElementById("recIndicator");
  const recIconMic = document.getElementById("recIconMic");
  const saveNextBtn = document.getElementById("saveNextBtn");

  recordToggleBtn.classList.add("recording-active");
  recordBtnLabel.textContent = "⏹️ Stop, Save & Next →";
  recIndicator.classList.add("active");
  recIconMic.style.display = "none";

  if (saveNextBtn) {
    saveNextBtn.classList.add("recording-pulse");
    saveNextBtn.innerHTML = "⏹️ Stop, Save & Next →";
  }

  // Hide old player while recording
  document.getElementById("audioPlayerBox").classList.remove("show");
  document.getElementById("validationNotice").className = "validation-notice hide";

  // Start live stopwatch timer
  startTimer();
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
  stopTimer();

  // Reset Button UI
  const recordToggleBtn = document.getElementById("recordToggleBtn");
  const recordBtnLabel = document.getElementById("recordBtnLabel");
  const recIndicator = document.getElementById("recIndicator");
  const recIconMic = document.getElementById("recIconMic");
  const saveNextBtn = document.getElementById("saveNextBtn");

  recordToggleBtn.classList.remove("recording-active");
  recordBtnLabel.textContent = "Start Recording";
  recIndicator.classList.remove("active");
  recIconMic.style.display = "block";

  if (saveNextBtn) {
    saveNextBtn.classList.remove("recording-pulse");
    saveNextBtn.innerHTML = "Save & Next →";
  }
}

function onRecordingFinished() {
  const playbackAudio = document.getElementById("playbackAudio");
  const audioPlayerBox = document.getElementById("audioPlayerBox");
  const reRecordBtn = document.getElementById("reRecordBtn");

  if (currentRecordedBlob) {
    const audioUrl = URL.createObjectURL(currentRecordedBlob);
    playbackAudio.src = audioUrl;
    audioPlayerBox.classList.add("show");
    reRecordBtn.style.display = "inline-flex";

    // Validate 2-4 seconds range
    updateDurationFeedback(currentRecordedDuration);
  }
}

// 5. DURATION VALIDATION (2-4 seconds)
function updateDurationFeedback(durationSec) {
  const notice = document.getElementById("validationNotice");
  if (!notice) return;

  if (durationSec < 2.0) {
    notice.className = "validation-notice warning";
    notice.innerHTML = `⚠️ Recording is short (${durationSec}s). Recommended: 2–4 seconds.`;
  } else if (durationSec > 4.0) {
    notice.className = "validation-notice warning";
    notice.innerHTML = `⚠️ Recording is longer than recommended (${durationSec}s). Recommended: 2–4 seconds.`;
  } else {
    notice.className = "validation-notice good";
    notice.innerHTML = `✓ Good duration (${durationSec}s)`;
  }
}

// Stopwatch Timer
function startTimer() {
  const timerText = document.getElementById("timerText");
  clearInterval(timerInterval);

  timerInterval = setInterval(() => {
    const elapsedMs = Date.now() - recordingStartTime;
    const mins = Math.floor(elapsedMs / 60000);
    const secs = Math.floor((elapsedMs % 60000) / 1000);
    const cs = Math.floor((elapsedMs % 1000) / 10);

    timerText.textContent = `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${String(cs).padStart(2, "0")}`;
  }, 30);
}

function stopTimer() {
  clearInterval(timerInterval);
}

function resetRecordingUI() {
  stopTimer();
  document.getElementById("timerText").textContent = "00:00.00";
  document.getElementById("validationNotice").className = "validation-notice hide";
  currentRecordedBlob = null;
  currentRecordedDuration = 0;
  clearWaveformCanvas();
}

function reRecordSentence() {
  resetRecordingUI();
  document.getElementById("audioPlayerBox").classList.remove("show");
  document.getElementById("reRecordBtn").style.display = "none";
  showToast("Ready to re-record. Click Start Recording.");
}

// ==========================================================================
// 5. SAVE & NAVIGATION (Save & Next, Previous, Skip)
// ==========================================================================
async function commitSaveAndNext() {
  if (sentences.length === 0) return;
  const currentItem = sentences[currentIndex];

  if (currentRecordedBlob) {
    currentItem.audioBlob = currentRecordedBlob;
    currentItem.duration = currentRecordedDuration;
    currentItem.status = "recorded";
    currentItem.timestamp = new Date().toISOString();
    await saveSentenceToDB(currentItem);
    showToast(`✓ Saved #${currentItem.id} (${currentItem.duration}s)`);
  }

  // Immediately advance to next sentence
  if (currentIndex < sentences.length - 1) {
    currentIndex++;
    displayCurrentSentence();
  } else {
    showToast("🎉 Congratulations! You have reached the end of the sentence queue!");
    displayCurrentSentence();
  }
}

async function saveAndNextSentence() {
  if (sentences.length === 0) return;

  // If user clicked Save & Next while still recording, stop and commit in one go!
  if (mediaRecorder && mediaRecorder.state === "recording") {
    autoAdvanceOnStop = true;
    stopRecording();
    return;
  }

  const currentItem = sentences[currentIndex];

  // If a new audio was recorded, save it
  if (currentRecordedBlob) {
    await commitSaveAndNext();
  } else if (currentItem.audioBlob) {
    // Already recorded previously, just move forward
    if (currentIndex < sentences.length - 1) {
      currentIndex++;
      displayCurrentSentence();
    } else {
      showToast("Reached the end of the queue.");
    }
  } else {
    // Not recorded yet
    showToast("Please record the sentence first, or click Skip.");
  }
}

function goToPreviousSentence() {
  if (currentIndex > 0) {
    currentIndex--;
    displayCurrentSentence();
  } else {
    showToast("Already at the first sentence.");
  }
}

async function skipCurrentSentence() {
  if (sentences.length === 0) return;

  const currentItem = sentences[currentIndex];
  if (currentItem.status !== "recorded") {
    currentItem.status = "skipped";
    await saveSentenceToDB(currentItem);
  }

  if (currentIndex < sentences.length - 1) {
    currentIndex++;
    displayCurrentSentence();
  } else {
    showToast("Reached the end of the queue.");
  }
}

// ==========================================================================
// 6. LIVE WAVEFORM VISUALIZER (Web Audio API on Canvas)
// ==========================================================================
function startWaveformVisualizer() {
  const canvas = document.getElementById("waveformCanvas");
  if (!canvas || !analyserNode) return;
  const ctx = canvas.getContext("2d");

  // Handle high-dpi displays
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.parentElement.clientWidth * dpr;
  canvas.height = canvas.parentElement.clientHeight * dpr;
  ctx.scale(dpr, dpr);

  const bufferLength = analyserNode.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function draw() {
    visualizerAnimationId = requestAnimationFrame(draw);
    analyserNode.getByteTimeDomainData(dataArray);

    const width = canvas.parentElement.clientWidth;
    const height = canvas.parentElement.clientHeight;

    ctx.clearRect(0, 0, width, height);
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = "#8b5cf6";
    ctx.beginPath();

    const sliceWidth = width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * height) / 2;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    ctx.lineTo(width, height / 2);
    ctx.stroke();
  }

  draw();
}

function stopWaveformVisualizer() {
  if (visualizerAnimationId) {
    cancelAnimationFrame(visualizerAnimationId);
    visualizerAnimationId = null;
  }
  clearWaveformCanvas();
}

function clearWaveformCanvas() {
  const canvas = document.getElementById("waveformCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.parentElement.clientWidth;
  const height = canvas.parentElement.clientHeight;
  ctx.clearRect(0, 0, width, height);

  // Draw flat idle center line
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "#cbd5e1";
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
}

// ==========================================================================
// 7. DATASET VIEW & SEARCH/FILTER
// ==========================================================================
let currentDatasetFilter = "all";

function setDatasetFilter(filter, btn) {
  currentDatasetFilter = filter;
  document.querySelectorAll(".filter-pill-btn").forEach((b) => b.classList.remove("active"));
  if (btn) btn.classList.add("active");
  renderDatasetTable();
}

function filterDatasetTable() {
  renderDatasetTable();
}

function renderDatasetTable() {
  const tbody = document.getElementById("datasetTableBody");
  const searchInput = document.getElementById("datasetSearchInput");
  const search = searchInput ? searchInput.value.toLowerCase().trim() : "";

  if (!tbody) return;

  const filtered = sentences.filter((s) => {
    // Filter by category
    if (currentDatasetFilter !== "all") {
      if (["English", "Hindi", "Gujarati"].includes(currentDatasetFilter)) {
        if (s.language !== currentDatasetFilter) return false;
      } else {
        if (s.status !== currentDatasetFilter) return false;
      }
    }
    // Search query
    if (search) {
      const matchText = s.text.toLowerCase().includes(search);
      const matchId = s.id.toLowerCase().includes(search);
      if (!matchText && !matchId) return false;
    }
    return true;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 40px;">
          No items match the current filter.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = filtered.slice(0, 200).map((s, idx) => {
    const audioBtn = s.audioBlob
      ? `<button class="btn btn-secondary" style="padding: 6px 12px; font-size: 12px;" onclick="playDatasetAudio('${s.id}')">▶ Play</button>`
      : `<span style="color: var(--text-muted); font-size: 12px;">No Audio</span>`;

    return `
      <tr>
        <td><code>${s.id}</code></td>
        <td><span class="language-badge pill-${s.language}" style="font-size: 11px; padding: 3px 10px;">${s.language}</span></td>
        <td style="max-width: 380px; word-break: break-word;">${escapeHtml(s.text)}</td>
        <td>${s.duration > 0 ? s.duration + "s" : "-"}</td>
        <td><span class="status-badge ${s.status}">${s.status}</span></td>
        <td>${audioBtn}</td>
        <td>
          <button class="btn btn-secondary" style="padding: 6px 12px; font-size: 12px;" onclick="jumpToSentence('${s.id}')">
            Record
          </button>
        </td>
      </tr>
    `;
  }).join("");
}

function playDatasetAudio(sentenceId) {
  const s = sentences.find((item) => item.id === sentenceId);
  if (!s || !s.audioBlob) return;
  const audio = new Audio(URL.createObjectURL(s.audioBlob));
  audio.play();
}

function jumpToSentence(sentenceId) {
  const idx = sentences.findIndex((item) => item.id === sentenceId);
  if (idx !== -1) {
    currentIndex = idx;
    switchView("recording");
    displayCurrentSentence();
  }
}

// ==========================================================================
// 8. EXPORT SYSTEM (Metadata CSV & JSZip Dataset)
// ==========================================================================
function exportMetadataCSV() {
  if (sentences.length === 0) {
    showToast("No dataset sentences to export.");
    return;
  }

  // Format: audio_path,language,text,duration,status
  let csvContent = "\uFEFFaudio_path,language,text,duration,status\n";

  sentences.forEach((s) => {
    const ext = s.audioBlob ? "webm" : "none";
    const audioPath = s.audioBlob ? `recordings/${s.id}.${ext}` : "";
    const escapedText = `"${s.text.replace(/"/g, '""')}"`;
    csvContent += `${audioPath},${s.language},${escapedText},${s.duration || 0},${s.status}\n`;
  });

  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "metadata.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  showToast("metadata.csv downloaded successfully!");
}

async function exportCompleteDatasetZIP() {
  const recordedList = sentences.filter((s) => s.audioBlob);
  if (recordedList.length === 0) {
    showToast("No audio recordings found to export. Please record some sentences first.");
    return;
  }

  if (typeof JSZip === "undefined") {
    showToast("JSZip library is loading. Please check your internet connection.");
    return;
  }

  showToast(`Packaging ${recordedList.length} recordings into ZIP...`);

  const zip = new JSZip();
  const audioFolder = zip.folder("dataset/audio");

  // 1. Add all audio files
  for (const item of recordedList) {
    audioFolder.file(`${item.id}.webm`, item.audioBlob);
  }

  // 2. Add metadata.csv
  let csvContent = "\uFEFFaudio_path,language,text,duration,status\n";
  sentences.forEach((s) => {
    const audioPath = s.audioBlob ? `dataset/audio/${s.id}.webm` : "";
    const escapedText = `"${s.text.replace(/"/g, '""')}"`;
    csvContent += `${audioPath},${s.language},${escapedText},${s.duration || 0},${s.status}\n`;
  });
  zip.file("dataset/metadata.csv", csvContent);

  // 3. Generate ZIP download
  const zipBlob = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(zipBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "stt_voice_dataset.zip";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  showToast("stt_voice_dataset.zip downloaded successfully!");
}

// ==========================================================================
// 9. NAVIGATION VIEWS & UTILITIES
// ==========================================================================
function switchView(viewName) {
  document.querySelectorAll(".view-section").forEach((sec) => sec.classList.remove("active-view"));
  document.querySelectorAll(".tab-btn").forEach((btn) => btn.classList.remove("active"));

  if (viewName === "recording") {
    document.getElementById("recordingView").classList.add("active-view");
    document.getElementById("tabRecordingBtn").classList.add("active");
    displayCurrentSentence();
  } else if (viewName === "upload") {
    document.getElementById("uploadView").classList.add("active-view");
    document.getElementById("tabUploadBtn").classList.add("active");
  } else if (viewName === "dataset") {
    document.getElementById("datasetView").classList.add("active-view");
    document.getElementById("tabDatasetBtn").classList.add("active");
    renderDatasetTable();
  } else if (viewName === "export") {
    document.getElementById("exportView").classList.add("active-view");
    document.getElementById("tabExportBtn").classList.add("active");
  }
}

function showToast(message) {
  const toast = document.getElementById("appToast");
  if (!toast) return;
  toast.textContent = message;
  toast.style.display = "block";
  setTimeout(() => {
    toast.style.display = "none";
  }, 3000);
}

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Keyboard Shortcuts: Space to record/stop
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.code === "Space") {
    e.preventDefault();
    toggleRecording();
  }
});

// Drag and drop setup for file dropzone
const dropzone = document.getElementById("fileDropzone");
if (dropzone) {
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-active");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-active");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-active");
    if (e.dataTransfer.files) {
      handleFileSelect(e.dataTransfer.files);
    }
  });
}

// ==========================================================================
// 10. INITIALIZATION
// ==========================================================================
window.addEventListener("DOMContentLoaded", async () => {
  await initIndexedDB();

  // Restore stored sentences from IndexedDB
  const restoredSentences = await loadAllSentencesFromDB();
  if (restoredSentences && restoredSentences.length > 0) {
    sentences = restoredSentences;
    const savedIndex = await loadAppState("currentIndex");
    if (savedIndex !== null && savedIndex < sentences.length) {
      currentIndex = savedIndex;
    }
  }

  displayCurrentSentence();
  clearWaveformCanvas();
});
