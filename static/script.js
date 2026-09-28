const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const uploadPrompt = document.getElementById("uploadPrompt");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predictBtn");
const resultBox = document.getElementById("result");
const mainPrediction = document.getElementById("mainPrediction");
const mainConfidence = document.getElementById("mainConfidence");
const top3Box = document.getElementById("top3");
const errorBox = document.getElementById("error");

let selectedFile = null;

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) {
    handleFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    handleFile(fileInput.files[0]);
  }
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("Please select an image file.");
    return;
  }
  selectedFile = file;
  hideError();
  resultBox.hidden = true;

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.hidden = false;
    uploadPrompt.hidden = true;
  };
  reader.readAsDataURL(file);

  predictBtn.disabled = false;
}

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  predictBtn.disabled = true;
  predictBtn.textContent = "Analyzing...";
  hideError();

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Prediction failed.");
    }

    const data = await response.json();
    displayResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Analyze Leaf";
  }
});

function displayResult(data) {
  mainPrediction.textContent = data.prediction;
  mainConfidence.textContent = `Confidence: ${data.confidence}%`;

  top3Box.innerHTML = "";
  data.top_3.forEach((item) => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<span>${item.label}</span><span>${item.confidence}%</span>`;
    top3Box.appendChild(row);
  });

  resultBox.hidden = false;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
}
