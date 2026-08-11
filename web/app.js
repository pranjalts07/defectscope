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

let selectedFile = null;
let inputMode = "upload";

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

      const guide = renderGuide(payload.prediction, payload.confidence);
      const colorClass = payload.prediction === "defective" ? "bad" : "ok";
      statusText.innerHTML = `Prediction: <span class="${colorClass}">${payload.prediction}</span> | confidence ${payload.confidence}. ${guide}`;

      resultEmpty.hidden = true;
      resultCard.hidden = false;
      resultPrediction.textContent = payload.prediction;
      resultPrediction.className = `result-value ${colorClass}`;
      resultConfidence.textContent = `${(Number(payload.confidence) * 100).toFixed(1)}%`;
      resultLatency.textContent = `${Number(payload.latency_ms).toFixed(1)} ms`;
      resultMeter.style.width = `${Math.max(2, Number(payload.confidence) * 100)}%`;
      resultNote.textContent = guide;
    } catch (err) {
      resultRaw.textContent = JSON.stringify({ error: err.message }, null, 2);
      statusText.innerHTML = `<span class="bad">Network error:</span> ${err.message}`;
    }
  });
}
