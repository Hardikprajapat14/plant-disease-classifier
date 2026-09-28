# 🌿 Plant Disease Classifier

An end-to-end web application that classifies plant leaf diseases from photos,
using transfer learning (MobileNetV2) and a FastAPI backend.

## Project structure

```
plant-disease-classifier/
├── download_dataset.py    # Downloads the dataset from Kaggle automatically
├── train_model.py         # Trains the classifier on your dataset
├── app.py                 # FastAPI backend (serves API + frontend)
├── requirements.txt
├── static/
│   ├── index.html         # Upload UI
│   ├── style.css
│   └── script.js
└── model/                 # Created after training
    ├── plant_disease_model.keras
    ├── class_names.json
    └── training_history.png
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get a dataset

You need the **PlantVillage** dataset (54k+ labeled leaf images, 38 classes).
Two ways to get it:

### Option A — automatic download via `download_dataset.py` (recommended)

1. Log in to [kaggle.com](https://www.kaggle.com) → **Account → Settings → API
   → "Create New Token"**. This downloads `kaggle.json`.
2. Place it at:
   - Mac/Linux: `~/.kaggle/kaggle.json` (then `chmod 600 ~/.kaggle/kaggle.json`)
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
3. Run:
   ```bash
   python download_dataset.py --output_dir ./dataset
   ```
   This downloads and unzips the dataset straight into `./dataset` — no
   manual browser download needed. A plain Chrome address-bar URL won't work
   here because Kaggle requires an authenticated API call, not a public link.

### Option B — manual download

Download from https://www.kaggle.com/datasets/emmarex/plantdisease and unzip
it yourself so you have one folder per class, e.g.:

```
dataset/
    Tomato___Late_blight/
    Tomato___healthy/
    Potato___Early_blight/
    ...
```

(Any similarly-structured image folder dataset works — the script just needs
one subfolder per class.)

## 3. Train the model

```bash
python train_model.py --data_dir ./dataset --epochs 15 --fine_tune_epochs 5
```

This will:
- Fine-tune a MobileNetV2 backbone on your dataset (2-stage training: frozen
  base, then partial unfreeze for fine-tuning)
- Save the trained model to `model/plant_disease_model.keras`
- Save the class name mapping to `model/class_names.json`
- Save an accuracy/loss plot to `model/training_history.png`

Training on a GPU is strongly recommended. On the full PlantVillage dataset
with a GPU, expect roughly 15-30 minutes and 95%+ validation accuracy. On CPU
it will be much slower — consider using a subset of classes to start.

## 4. Run the web app

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser, upload a leaf photo, and
click "Analyze Leaf" to get the predicted disease with confidence scores.

## 5. API reference

- `GET /health` — check server + model status
- `POST /predict` — multipart form upload, field name `file` (an image).
  Returns:
  ```json
  {
    "prediction": "Tomato - Late blight",
    "confidence": 97.42,
    "top_3": [
      {"label": "Tomato - Late blight", "confidence": 97.42},
      {"label": "Tomato - Early blight", "confidence": 1.85},
      {"label": "Tomato - healthy", "confidence": 0.4}
    ]
  }
  ```

## Deployment options

- **Render / Railway**: push this repo, set the start command to
  `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Hugging Face Spaces**: use the "Docker" or "FastAPI" space type
- **Docker**: a minimal Dockerfile would be:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY . .
  RUN pip install -r requirements.txt
  CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

## Next steps / ideas to extend

- Add Grad-CAM visualization to highlight the diseased region on the leaf
- Add a treatment-recommendation lookup table keyed by predicted class
- Swap MobileNetV2 for EfficientNetB0 for higher accuracy (larger model)
- Add authentication + a history of past scans per user
- Convert the model to TensorFlow Lite for mobile/edge deployment
