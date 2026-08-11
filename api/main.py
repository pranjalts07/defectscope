"""
DefectScope FastAPI application.

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    PredictionResponse,
    HealthResponse,
    MetricsResponse,
    ThresholdUpdateRequest,
)
from inference.predict import DefectPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DefectScope",
    description=(
        "Upload a product or surface image and the app predicts whether it looks good or defective. "
        "This is a minimal manufacturing QA MVP with a browser demo, health checks, metrics, and a FastAPI backend."
    ),
    version="0.1.0",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "defaultModelExpandDepth": -1,
        "docExpansion": "list",
        "displayRequestDuration": True,
    },
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

# how-it-works.html illustrates good vs defective with real MVTec frames.
# The dataset itself is gitignored, so a few representative images are kept
# under samples/ and served from here.
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


LANDING_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>DefectScope</title>
    <style>
        :root {
            --bg: #f4f1ea;
            --panel: rgba(255, 255, 255, 0.78);
            --panel-strong: #ffffff;
            --text: #141414;
            --muted: #5d5a53;
            --line: rgba(20, 20, 20, 0.1);
            --accent: #0f766e;
            --accent-2: #b45309;
            --good: #1f7a4d;
            --bad: #b42318;
            --shadow: 0 24px 80px rgba(20, 20, 20, 0.12);
            --radius: 22px;
        }

        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(180, 83, 9, 0.15), transparent 24%),
                linear-gradient(180deg, #f8f5ef 0%, var(--bg) 58%, #efe9df 100%);
            min-height: 100vh;
        }

        .shell {
            max-width: 1180px;
            margin: 0 auto;
            padding: 28px 20px 48px;
        }

        .hero {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 22px;
            align-items: stretch;
        }

        .panel {
            background: var(--panel);
            backdrop-filter: blur(16px);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
        }

        .hero-main {
            padding: 34px;
            position: relative;
            overflow: hidden;
            isolation: isolate;
        }

        .hero-main::after {
            content: "";
            position: absolute;
            inset: auto -120px -120px auto;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(15, 118, 110, 0.22), transparent 68%);
            z-index: -1;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.08);
            color: var(--accent);
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.02em;
        }

        h1 {
            margin: 18px 0 12px;
            font-size: clamp(40px, 6vw, 68px);
            line-height: 0.95;
            letter-spacing: -0.06em;
            max-width: 9ch;
        }

        .lead {
            max-width: 62ch;
            font-size: 18px;
            line-height: 1.7;
            color: var(--muted);
            margin: 0 0 24px;
        }

        .metrics {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-top: 24px;
        }

        .metric {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 16px;
        }

        .metric .label {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 8px;
        }

        .metric .value {
            font-size: 24px;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        .hero-side {
            padding: 22px;
            display: grid;
            gap: 14px;
            align-content: start;
        }

        .card {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 18px;
        }

        .card h2, .card h3 {
            margin: 0 0 10px;
            font-size: 16px;
            letter-spacing: -0.02em;
        }

        .card p, .card li {
            margin: 0;
            color: var(--muted);
            line-height: 1.6;
            font-size: 14px;
        }

        .callout {
            border-left: 4px solid var(--accent);
            background: linear-gradient(180deg, rgba(15, 118, 110, 0.07), rgba(15, 118, 110, 0.03));
        }

        .list {
            display: grid;
            gap: 8px;
            margin-top: 12px;
            padding-left: 0;
            list-style: none;
        }

        .list li::before {
            content: "•";
            color: var(--accent);
            font-weight: 800;
            margin-right: 8px;
        }

        .flow {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin: 22px 0;
        }

        .step {
            padding: 18px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--line);
        }

        .step .num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 999px;
            margin-bottom: 12px;
            background: rgba(15, 118, 110, 0.12);
            color: var(--accent);
            font-weight: 800;
            font-size: 13px;
        }

        .demo {
            display: grid;
            grid-template-columns: 0.9fr 1.1fr;
            gap: 18px;
            align-items: start;
            margin-top: 22px;
        }

        .upload {
            padding: 22px;
        }

        .upload-form {
            display: grid;
            gap: 12px;
        }

        .file {
            padding: 14px;
            border-radius: 14px;
            border: 1px dashed rgba(20, 20, 20, 0.2);
            background: rgba(244, 241, 234, 0.55);
        }

        button {
            border: 0;
            border-radius: 14px;
            padding: 14px 16px;
            font-weight: 800;
            font-size: 14px;
            cursor: pointer;
            background: linear-gradient(135deg, var(--accent), #14b8a6);
            color: white;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
            box-shadow: 0 10px 20px rgba(15, 118, 110, 0.22);
        }

        button:hover { transform: translateY(-1px); }

        pre {
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
            background: #0c111d;
            color: #e5eefc;
            padding: 18px;
            border-radius: 18px;
            min-height: 260px;
            font-size: 13px;
            line-height: 1.55;
            overflow: auto;
        }

        .status {
            margin-top: 10px;
            font-size: 14px;
            color: var(--muted);
        }

        .status strong.good { color: var(--good); }
        .status strong.bad { color: var(--bad); }

        .links {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 18px;
        }

        .links a {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            text-decoration: none;
            color: var(--text);
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 10px 14px;
            font-size: 14px;
            font-weight: 700;
        }

        .section-title {
            margin: 0 0 12px;
            font-size: 22px;
            letter-spacing: -0.03em;
        }

        @media (max-width: 960px) {
            .hero, .demo, .flow { grid-template-columns: 1fr; }
            .metrics { grid-template-columns: 1fr; }
            h1 { max-width: none; }
        }
    </style>
</head>
<body>
    <div class="shell">
        <section class="hero">
            <div class="panel hero-main">
                <div class="eyebrow">DefectScope · manufacturing defect detection</div>
                <h1>Spot bad parts before they ship.</h1>
                <p class="lead">
                    DefectScope is a compact quality-control app for product images. Upload a JPG or PNG,
                    and it predicts whether the part looks good or defective. The goal is a clean MVP you
                    can explain in one minute and deploy in one command.
                </p>
                <div class="links">
                    <a href="/docs">Open API docs</a>
                    <a href="/health">Check health</a>
                    <a href="/metrics">View metrics</a>
                </div>
                <div class="metrics">
                    <div class="metric">
                        <div class="label">Primary job</div>
                        <div class="value">Good vs defective</div>
                    </div>
                    <div class="metric">
                        <div class="label">Model stack</div>
                        <div class="value">DenseNet + AE</div>
                    </div>
                    <div class="metric">
                        <div class="label">Deployment</div>
                        <div class="value">FastAPI + Docker</div>
                    </div>
                </div>
            </div>

            <div class="panel hero-side">
                <div class="card">
                    <h2>What this actually does</h2>
                    <p>
                        Upload a surface or part image. The app runs inference and returns the predicted class,
                        confidence, and latency. This page is intentionally minimal so the workflow is obvious.
                    </p>
                </div>
                <div class="card">
                    <h3>When to use it</h3>
                    <p>Portfolio demo, interview walkthrough, local QA tool, or a starter production API for anomaly detection.</p>
                </div>
                <div class="card">
                    <h3>Current limitation</h3>
                    <p>It is category-specific and best treated as an MVP, not a final manufacturing system.</p>
                </div>
                <div class="card callout">
                    <h3>What should I upload?</h3>
                    <p>Use any JPG or PNG of a product surface, part, or test image. If you want the cleanest demo, use a MVTec-style sample image from a category like bottle, cable, or tile.</p>
                    <ul class="list">
                        <li>Good for demo: close-up product photos</li>
                        <li>Also works: any valid image file</li>
                        <li>Do not upload code or documents here</li>
                    </ul>
                </div>
            </div>
        </section>

        <section class="flow">
            <div class="step"><div class="num">1</div><h3>Upload</h3><p>Choose one JPG or PNG of a product surface or part.</p></div>
            <div class="step"><div class="num">2</div><h3>Analyze</h3><p>Click the button and the backend sends the image to the model.</p></div>
            <div class="step"><div class="num">3</div><h3>Read</h3><p>You get good or defective, confidence, and latency in the response panel.</p></div>
            <div class="step"><div class="num">4</div><h3>Deploy</h3><p>Use /health and /metrics to show the project is live and working.</p></div>
        </section>

        <section class="demo">
            <div class="panel upload">
                <h2 class="section-title">Try the demo</h2>
                <form class="upload-form" id="predict-form">
                    <input class="file" id="image-file" type="file" accept="image/*" required />
                    <button type="submit">Analyze image</button>
                </form>
                <div class="status" id="status">Ready. Pick an image, then click Analyze image.</div>
                <div class="card" style="margin-top:14px;">
                    <h3>Useful endpoints</h3>
                    <p><strong>GET /health</strong> model readiness</p>
                    <p><strong>GET /metrics</strong> request counts and latency</p>
                    <p><strong>POST /threshold/update</strong> change anomaly threshold</p>
                </div>
            </div>
            <div class="panel" style="padding:22px;">
                <h2 class="section-title">Response</h2>
                <pre id="result">No prediction yet.</pre>
            </div>
        </section>
    </div>

    <script>
        const form = document.getElementById('predict-form');
        const input = document.getElementById('image-file');
        const result = document.getElementById('result');
        const status = document.getElementById('status');

        const pretty = (value) => JSON.stringify(value, null, 2);

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            if (!input.files || !input.files[0]) return;

            const file = input.files[0];
            const formData = new FormData();
            formData.append('file', file);

            status.innerHTML = 'Running inference on <strong>' + file.name + '</strong>...';
            result.textContent = 'Loading...';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData,
                });

                const payload = await response.json();
                result.textContent = pretty(payload);

                if (response.ok) {
                    status.innerHTML = 'Prediction complete: <strong class="good">' + payload.prediction + '</strong> · confidence ' + payload.confidence;
                } else {
                    status.innerHTML = 'Prediction failed: <strong class="bad">' + (payload.detail || response.statusText) + '</strong>';
                }
            } catch (error) {
                status.innerHTML = 'Prediction failed: <strong class="bad">' + error.message + '</strong>';
                result.textContent = pretty({ error: error.message });
            }
        });
    </script>
</body>
</html>"""


@app.on_event("startup")
async def load_models():
    """Load models once at startup. Fail fast if weights are missing."""
    app.state.request_count = 0
    app.state.total_latency_ms = 0.0
    app.state.recent_predictions = deque(maxlen=100)

    existing_predictor = getattr(app.state, "predictor", None)
    if existing_predictor is not None:
        logger.info("Using preconfigured predictor from app state")
        return

    try:
        app.state.predictor = DefectPredictor()
        logger.info(f"Models loaded successfully on {app.state.predictor.device}")
    except FileNotFoundError as e:
        logger.error(f"Model checkpoint not found: {e}")
        app.state.predictor = None
    except Exception as e:
        logger.exception(f"Unexpected model initialization error: {e}")
        app.state.predictor = None


@app.get("/", response_class=FileResponse, summary="Landing page", include_in_schema=False)
async def landing_page():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/image-guide", response_class=FileResponse, include_in_schema=False)
async def image_guide_page():
    return FileResponse(WEB_DIR / "image-guide.html")


@app.get("/how-it-works", response_class=FileResponse, include_in_schema=False)
async def how_it_works_page():
    return FileResponse(WEB_DIR / "how-it-works.html")


@app.get("/blog", response_class=FileResponse, include_in_schema=False)
async def blog_page():
    return FileResponse(WEB_DIR / "blog.html")


@app.get("/health", response_model=HealthResponse, summary="Check model readiness")
async def health():
    models_loaded = app.state.predictor is not None
    return HealthResponse(
        status="ok" if models_loaded else "degraded",
        models_loaded=models_loaded,
        device=str(app.state.predictor.device) if models_loaded else "unknown",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify an uploaded image",
    description="Upload a JPG or PNG of a product surface or part. The API returns good/defective, confidence, and latency.",
)
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

    app.state.request_count += 1
    app.state.total_latency_ms += float(result["latency_ms"])
    app.state.recent_predictions.append(result["prediction"])

    logger.info(f"Prediction: {result['prediction']} (conf={result['confidence']:.3f})")
    return PredictionResponse(**result)


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="View lightweight runtime metrics",
    description="Shows request count, average latency, and the recent defect rate from the last 100 requests.",
)
async def metrics():
    served = int(getattr(app.state, "request_count", 0))
    total_latency = float(getattr(app.state, "total_latency_ms", 0.0))
    recent = list(getattr(app.state, "recent_predictions", []))

    avg_latency = (total_latency / served) if served > 0 else 0.0
    defects = sum(1 for pred in recent if pred == "defective")
    defect_rate = (defects / len(recent)) if recent else 0.0

    return MetricsResponse(
        requests_served=served,
        avg_latency_ms=round(avg_latency, 2),
        defect_rate_last_100=round(defect_rate, 4),
    )


@app.post(
    "/threshold/update",
    response_model=HealthResponse,
    summary="Update the anomaly threshold",
    description="Updates the in-memory anomaly threshold used by the demo and review workflow.",
)
async def update_threshold(payload: ThresholdUpdateRequest):
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Cannot update threshold.")

    if payload.anomaly_threshold <= 0:
        raise HTTPException(status_code=422, detail="anomaly_threshold must be > 0")

    predictor.anomaly_threshold = float(payload.anomaly_threshold)
    logger.info(f"Updated anomaly threshold to {predictor.anomaly_threshold:.6f}")

    return HealthResponse(status="ok", models_loaded=True, device=str(predictor.device))
