from pydantic import BaseModel
from typing import Optional


class PredictionResponse(BaseModel):
    prediction: str           # "good" or "defective"
    confidence: float         # Softmax probability of the predicted class
    defect_prob: float        # Raw probability of being defective (0–1), always present
    classification_threshold: float     # Threshold used: defect_prob >= threshold → defective
    anomaly_score: Optional[float]      # Mean reconstruction error from autoencoder
    anomaly_threshold: Optional[float]  # Threshold used for anomaly decision
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


class ThresholdUpdateRequest(BaseModel):
    anomaly_threshold: float
