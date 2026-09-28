"""
app.py
------
FastAPI backend that serves the trained plant disease classifier and the
static frontend.

Run with:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in your browser.
"""

import io
import json
import os

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
IMG_SIZE = (224, 224)

app = FastAPI(title="Plant Disease Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_names = {}


@app.on_event("startup")
def load_model():
    """Loads the trained model and class names into memory on startup."""
    global model, class_names

    if not os.path.exists(MODEL_PATH):
        print(
            f"WARNING: model not found at {MODEL_PATH}. "
            "Train it first with train_model.py, then restart the server."
        )
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH) as f:
        raw = json.load(f)
    class_names = {int(k): v for k, v in raw.items()}
    print(f"Loaded model with {len(class_names)} classes.")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Converts raw image bytes into a model-ready batch of shape (1, 224, 224, 3)."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    img = img.resize(IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)  # model applies its own preprocessing internally
    return arr


def format_label(raw_label: str) -> str:
    """Turns 'Tomato___Late_blight' into 'Tomato - Late blight'."""
    parts = raw_label.split("___")
    if len(parts) == 2:
        crop, disease = parts
        return f"{crop.replace('_', ' ')} - {disease.replace('_', ' ')}"
    return raw_label.replace("_", " ")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train it first with train_model.py.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    batch = preprocess_image(image_bytes)

    preds = model.predict(batch)[0]
    top_indices = preds.argsort()[-3:][::-1]  # top-3 predictions

    results = [
        {
            "label": format_label(class_names[int(idx)]),
            "confidence": round(float(preds[idx]) * 100, 2),
        }
        for idx in top_indices
    ]

    return JSONResponse(
        {
            "prediction": results[0]["label"],
            "confidence": results[0]["confidence"],
            "top_3": results,
        }
    )


# Serve the frontend (index.html, css, js) at "/"
app.mount("/", StaticFiles(directory="static", html=True), name="static")
