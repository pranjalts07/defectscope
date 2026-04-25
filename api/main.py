"""
DefectScope FastAPI application.

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles

from api.schemas import PredictionResponse, HealthResponse
from inference.predict import DefectPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DefectScope",
    description="Manufacturing defect detection API — CNN + anomaly detection",
    version="0.1.0",
)

# Serve static files (web UI)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
async def load_models():
    """Load models once at startup. Fail fast if weights are missing."""
    try:
        app.state.predictor = DefectPredictor()
        logger.info(f"Models loaded successfully on {app.state.predictor.device}")
    except FileNotFoundError as e:
        logger.error(f"Model checkpoint not found: {e}")
        app.state.predictor = None


@app.get("/")
async def root():
    """Serve the web UI."""
    from fastapi.responses import FileResponse
    static_dir = Path(__file__).parent / "static"
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "DefectScope API. POST /predict with an image file."}


@app.get("/health", response_model=HealthResponse)
async def health():
    models_loaded = app.state.predictor is not None
    return HealthResponse(
        status="ok" if models_loaded else "degraded",
        models_loaded=models_loaded,
        device=str(app.state.predictor.device) if models_loaded else "unknown",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if app.state.predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Check server logs.")

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail=f"Expected an image file, got: {content_type}")

    img_bytes = await file.read()
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = app.state.predictor.predict(img_rgb)

    logger.info(f"Prediction: {result['prediction']} (conf={result['confidence']:.3f})")
    return PredictionResponse(**result)
