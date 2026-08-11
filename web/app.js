const form = document.getElementById("predict-form");
const uploadInput = document.getElementById("image-file");
const cameraInput = document.getElementById("camera-file");
const resultRaw = document.getElementById("result-raw");
const statusText = document.getElementById("status");
const preview = document.getElementById("preview");
const modeUpload = document.getElementById("mode-upload");
const modeCamera = document.getElementById("mode-camera");
const pickUpload = document.getElementById("pick-upload");
const pickCamera = document.getElementById("pick-camera");
const uploadName = document.getElementById("upload-name");
const cameraName = document.getElementById("camera-name");
const uploadPickerBox = document.getElementById("upload-picker-box");
const cameraPickerBox = document.getElementById("camera-picker-box");

const resultEmpty = document.getElementById("result-empty");
const resultCard = document.getElementById("result-card");
const resultPrediction = document.getElementById("result-prediction");
const resultConfidence = document.getElementById("result-confidence");
const resultLatency = document.getElementById("result-latency");
const resultMeter = document.getElementById("result-meter");
const resultNote = document.getElementById("result-note");

const resultDefectProb = document.getElementById("result-defect-prob");
const resultAe = document.getElementById("result-ae");
const verdict = document.getElementById("verdict");
const meterMark = document.getElementById("meter-mark");
const thresholdLabel = document.getElementById("threshold-label");
const attention = document.getElementById("attention");
const attentionImg = document.getElementById("attention-img");
const viewHeatmap = document.getElementById("view-heatmap");
const viewOriginal = document.getElementById("view-original");

let selectedFile = null;
let inputMode = "upload";
let heatmapUrl = null;
let originalUrl = null;

function renderGuide(prediction, confidence) {
  const c = Number(confidence || 0);
  if (c < 0.7) return "Low confidence: treat this as uncertain.";
  if (prediction === "defective") return "Strong defective signal: send for manual review.";
  return "Strong good signal: likely normal sample.";
}

function switchMode(mode) {
  inputMode = mode;
  selectedFile = null;

  if (preview) {
    preview.src = "";
    preview.style.display = "none";
  }

  if (uploadInput) {
    uploadInput.value = "";
    uploadInput.hidden = true;
    uploadInput.required = mode === "upload";
  }

  if (cameraInput) {
    cameraInput.value = "";
    cameraInput.hidden = true;
    cameraInput.required = mode === "camera";
  }

  if (uploadPickerBox) uploadPickerBox.hidden = mode !== "upload";
  if (cameraPickerBox) cameraPickerBox.hidden = mode !== "camera";
  if (uploadName) uploadName.textContent = "No file selected";
  if (cameraName) cameraName.textContent = "No photo selected";

  if (modeUpload && modeCamera) {
    modeUpload.classList.toggle("active", mode === "upload");
    modeCamera.classList.toggle("active", mode === "camera");
  }

  statusText.textContent = mode === "upload"
    ? "Upload mode selected. Pick a photo and click Analyze image."
    : "Camera mode selected. On phone it opens camera; on desktop it may open file picker.";
}

function setPreview(file) {
  if (!preview || !file) return;
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.style.display = "block";
}

function setSelectedFile(file, sourceLabel) {
  selectedFile = file;
  setPreview(file);
  if (statusText && file) {
    statusText.textContent = `${sourceLabel}: ${file.name}. Click Analyze image.`;
  }
}

if (uploadInput) {
  uploadInput.addEventListener("change", () => {
    const file = uploadInput.files && uploadInput.files[0] ? uploadInput.files[0] : null;
    if (file) {
      if (uploadName) uploadName.textContent = file.name;
      setSelectedFile(file, "Uploaded image");
    }
  });
}

if (cameraInput) {
  cameraInput.addEventListener("change", () => {
    const file = cameraInput.files && cameraInput.files[0] ? cameraInput.files[0] : null;
    if (file) {
      if (cameraName) cameraName.textContent = file.name;
      setSelectedFile(file, "Camera image");
    }
  });
}

if (pickUpload && uploadInput) {
  pickUpload.addEventListener("click", () => uploadInput.click());
}

if (pickCamera && cameraInput) {
  pickCamera.addEventListener("click", () => cameraInput.click());
}

if (modeUpload) {
  modeUpload.addEventListener("click", () => switchMode("upload"));
}

if (modeCamera) {
  modeCamera.addEventListener("click", () => switchMode("camera"));
}

switchMode("upload");

/* Sample strip — load a dataset image straight into the form so the demo can be
   tried without hunting for a bottle photo. */
document.querySelectorAll(".sample").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const src = btn.dataset.src;
    statusText.textContent = "Loading sample...";
    try {
      const blob = await (await fetch(src)).blob();
      const name = src.split("/").slice(-3).join("-");
      const file = new File([blob], name, { type: blob.type || "image/png" });

      document.querySelectorAll(".sample").forEach((s) => s.classList.remove("active"));
      btn.classList.add("active");

      setSelectedFile(file, "Sample");
      statusText.textContent = "Sample loaded. Click Analyze image.";
    } catch (err) {
      statusText.innerHTML = `<span class="bad">Could not load sample:</span> ${err.message}`;
    }
  });
});

if (viewHeatmap && viewOriginal) {
  viewHeatmap.addEventListener("click", () => {
    if (!heatmapUrl) return;
    attentionImg.src = heatmapUrl;
    viewHeatmap.classList.add("active");
    viewOriginal.classList.remove("active");
  });
  viewOriginal.addEventListener("click", () => {
    if (!originalUrl) return;
    attentionImg.src = originalUrl;
    viewOriginal.classList.add("active");
    viewHeatmap.classList.remove("active");
  });
}

if (form) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const activeInput = inputMode === "upload" ? uploadInput : cameraInput;
    const file = selectedFile || (activeInput && activeInput.files && activeInput.files[0]);

    if (!file) {
      statusText.innerHTML = inputMode === "upload"
        ? '<span class="bad">Select a file first.</span>'
        : '<span class="bad">Take a photo first.</span>';
      return;
    }

    const data = new FormData();
    data.append("file", file);

    resultRaw.textContent = "Running inference...";
    statusText.textContent = `Analyzing ${file.name}...`;

    try {
      const response = await fetch("/predict", { method: "POST", body: data });
      const payload = await response.json();
      resultRaw.textContent = JSON.stringify(payload, null, 2);

      if (!response.ok) {
        statusText.innerHTML = `<span class="bad">Request failed:</span> ${payload.detail || response.statusText}`;
        return;
      }

      const isDefective = payload.prediction === "defective";
      const colorClass = isDefective ? "bad" : "ok";
      const defectPct = Number(payload.defect_prob) * 100;
      const thresholdPct = Number(payload.classification_threshold) * 100;
      const confidence = Number(payload.confidence);

      // The threshold is what the verdict actually turns on, so say so plainly
      // rather than leaving the reader to infer it from two percentages.
      const guide =
        `Defect probability ${defectPct.toFixed(0)}% is ` +
        `${isDefective ? "at or above" : "below"} the ${thresholdPct.toFixed(0)}% threshold → ` +
        `${payload.prediction}.` +
        (confidence < 0.7 ? " Confidence is under 70% — treat as uncertain." : "") +
        (payload.needs_review ? " The autoencoder disagrees; flagged for review." : "") +
        " This is a screening pass — always confirm with manual QA.";

      statusText.innerHTML = `Prediction: <span class="${colorClass}">${payload.prediction}</span> · defect prob ${defectPct.toFixed(0)}%`;

      resultEmpty.hidden = true;
      resultCard.hidden = false;

      verdict.className = `verdict ${colorClass}`;
      resultPrediction.textContent = isDefective ? "✕ DEFECTIVE" : "✓ GOOD";

      resultDefectProb.textContent = `${defectPct.toFixed(0)}%`;
      resultMeter.style.width = `${Math.max(2, defectPct)}%`;

      meterMark.hidden = false;
      meterMark.style.left = `${thresholdPct}%`;
      thresholdLabel.textContent = `↑ ${thresholdPct.toFixed(0)}% threshold`;

      resultConfidence.textContent = `${(confidence * 100).toFixed(0)}%`;
      resultLatency.textContent = `${Number(payload.latency_ms).toFixed(1)} ms`;
      resultAe.textContent =
        payload.anomaly_score == null ? "—" : Number(payload.anomaly_score).toFixed(4);

      resultNote.textContent = guide;

      if (payload.heatmap_b64) {
        if (originalUrl) URL.revokeObjectURL(originalUrl);
        originalUrl = URL.createObjectURL(file);
        heatmapUrl = `data:image/png;base64,${payload.heatmap_b64}`;
        attentionImg.src = heatmapUrl;
        viewHeatmap.classList.add("active");
        viewOriginal.classList.remove("active");
        attention.hidden = false;
      } else {
        attention.hidden = true;
      }
    } catch (err) {
      resultRaw.textContent = JSON.stringify({ error: err.message }, null, 2);
      statusText.innerHTML = `<span class="bad">Network error:</span> ${err.message}`;
    }
  });
}
