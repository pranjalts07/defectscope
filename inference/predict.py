"""
Single-image inference — DenseNet classifier with autoencoder second opinion.

The classifier makes the call. The autoencoder's reconstruction error is
reported alongside it, and when the two disagree the result is flagged for
review rather than silently resolved.

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

        # Two tensors: the classifier wants ImageNet stats, the autoencoder
        # wants plain [0, 1] to match its Sigmoid output.
        cnn_batch = preprocess_image(img_np).to(self.device).unsqueeze(0)
        ae_batch = (
            preprocess_image(img_np, normalize=False).to(self.device).unsqueeze(0)
        )

        with torch.no_grad():
            probs = torch.softmax(self.cnn(cnn_batch), dim=1)[0]
            reconstruction = self.ae(ae_batch)
            anomaly_score = float(
                torch.nn.functional.mse_loss(reconstruction, ae_batch).item()
            )

        cnn_confidence = float(probs.max().item())
        cnn_flags_defect = int(probs.argmax().item()) == 1
        ae_flags_defect = anomaly_score > self.anomaly_threshold

        # The classifier decides. On this dataset the autoencoder's
        # reconstruction error barely separates good from defective (good mean
        # 0.00204 vs defect mean 0.00234, ranges overlapping), so using it as a
        # veto would reject nearly every good unit. It is kept as a second
        # opinion: when it disagrees with the classifier, flag for review.
        result = {
            "prediction": "defective" if cnn_flags_defect else "good",
            "confidence": round(cnn_confidence, 4),
            "anomaly_score": round(anomaly_score, 6),
            "anomaly_threshold": self.anomaly_threshold,
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
