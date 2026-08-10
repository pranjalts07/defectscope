"""
Single-image inference — DenseNet classifier cross-verified by the autoencoder.

A unit passes only when both models agree it is good: the classifier must not
flag it, and its reconstruction error must stay under the trained threshold.
Disagreement is surfaced as needs_review rather than silently resolved.

Usage as CLI:
    python -m inference.predict --image path/to/image.png
"""

import argparse
import time

import cv2
import numpy as np
import torch
import yaml

from models.autoencoder import ConvAutoencoder
from models.cnn_classifier import get_densenet
from utils import get_device
from utils.transforms import preprocess_image


class DefectPredictor:
    """
    Wraps both models and handles inference for a single image.
    Load once at server startup; call .predict() per request.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = get_device()
        self._load_models()

    def _load_models(self):
        cnn_path = self.cfg["paths"]["cnn_checkpoint"]
        ae_path = self.cfg["paths"]["ae_checkpoint"]

        self.cnn = get_densenet(num_classes=self.cfg["cnn"]["num_classes"])
        self.cnn.load_state_dict(torch.load(cnn_path, map_location=self.device))
        self.cnn.eval().to(self.device)

        self.ae = ConvAutoencoder()
        self.ae.load_state_dict(torch.load(ae_path, map_location=self.device))
        self.ae.eval().to(self.device)

        # Trained on good images only; reconstruction error above this is anomalous.
        self.anomaly_threshold = float(self.cfg["autoencoder"]["anomaly_threshold"])

        self.include_heatmap = False

    def predict(self, img_np: np.ndarray) -> dict:
        """Run both models on a uint8 RGB image array and gate on agreement."""
        start = time.perf_counter()

        batch = preprocess_image(img_np).to(self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.cnn(batch)
            probs = torch.softmax(logits, dim=1)[0]

            reconstruction = self.ae(batch)
            anomaly_score = float(
                torch.nn.functional.mse_loss(reconstruction, batch).item()
            )

        cnn_confidence = float(probs.max().item())
        cnn_flags_defect = int(probs.argmax().item()) == 1
        ae_flags_defect = anomaly_score > self.anomaly_threshold

        result = {
            # Agreement gate: a unit passes only if neither model objects.
            "prediction": "good" if not (cnn_flags_defect or ae_flags_defect) else "defective",
            "confidence": round(cnn_confidence, 4),
            "anomaly_score": round(anomaly_score, 6),
            "anomaly_threshold": self.anomaly_threshold,
            # The models disagreed — worth a human look rather than a silent call.
            "needs_review": cnn_flags_defect != ae_flags_defect,
            "heatmap_b64": None,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    predictor = DefectPredictor(config_path=args.config)
    result = predictor.predict(img_rgb)

    print("\n--- DefectScope Prediction ---")
    print(f"Prediction:  {result['prediction'].upper()}")
    print(f"Confidence:  {result['confidence']:.1%}")
    print(f"Latency:     {result['latency_ms']} ms")


if __name__ == "__main__":
    main()
