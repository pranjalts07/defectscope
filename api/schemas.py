from pydantic import BaseModel
from typing import Optional


class PredictionResponse(BaseModel):
    prediction: str           # "good" or "defective"
    confidence: float         # CNN softmax probability for predicted class
    anomaly_score: float      # Mean reconstruction error from autoencoder
    anomaly_threshold: float  # Threshold used for anomaly decision
    needs_review: bool        # True when CNN and AE disagree — send to human review
    heatmap_b64: Optional[str]  # Base64-encoded PNG of Grad-CAM overlay
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str


class MetricsResponse(BaseModel):
    requests_served: int
    avg_latency_ms: float
    defect_rate_last_100: float
