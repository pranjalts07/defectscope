"""
Single-image inference — CNN classifier only.

Usage as CLI:
    python -m inference.predict --image path/to/image.png
"""

import argparse
import time

import cv2
import numpy as np
import torch
import yaml

from models.cnn_classifier import get_densenet
from utils import get_device
from utils.transforms import preprocess_image


class DefectPredictor:
    """
    Wraps the CNN model and handles inference for a single image.
    Load once at server startup; call .predict() per request.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = get_device()
        self._load_models()

    def _load_models(self):
        cnn_path = self.cfg["paths"]["cnn_checkpoint"]

        self.cnn = get_densenet(num_classes=self.cfg["cnn"]["num_classes"])
        self.cnn.load_state_dict(torch.load(cnn_path, map_location=self.device))
        self.cnn.eval().to(self.device)

        self.include_heatmap = False

    def predict(self, img_np: np.ndarray) -> dict:
        """Run CNN inference on a uint8 RGB image array."""
        start = time.perf_counter()

        cnn_tensor = preprocess_image(img_np).to(self.device)

        with torch.no_grad():
            logits = self.cnn(cnn_tensor.unsqueeze(0))
            probs  = torch.softmax(logits, dim=1)[0]

        cnn_pred_idx   = int(probs.argmax().item())
        cnn_confidence = float(probs.max().item())

        result = {
            "prediction":  "defective" if cnn_pred_idx == 1 else "good",
            "confidence":  round(cnn_confidence, 4),
            "anomaly_score":     None,
            "anomaly_threshold": None,
            "needs_review":      False,
            "heatmap_b64":       None,
            "latency_ms":  round((time.perf_counter() - start) * 1000, 2),
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
