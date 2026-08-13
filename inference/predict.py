"""
Single-image inference — CNN classifier + Grad-CAM heatmap + Autoencoder anomaly score.

Usage as CLI:
    python -m inference.predict --image path/to/image.png
"""

import argparse
import base64
import logging
import os
import time
from typing import Optional

import cv2
import numpy as np
import torch
import yaml

from models.autoencoder import ConvAutoencoder
from models.cnn_classifier import get_densenet
from utils import get_device
from utils.gradcam import GradCAM, overlay_gradcam
from utils.transforms import preprocess_image, preprocess_image_ae

logger = logging.getLogger(__name__)


class DefectPredictor:
    """
    Wraps both the CNN classifier and the Autoencoder anomaly detector.

    - CNN  → binary prediction (good / defective) + Grad-CAM spatial heatmap
    - AE   → reconstruction error as an independent anomaly signal
    - When CNN and AE disagree the result is flagged needs_review = True

    Load once at server startup; call .predict() per request.
    """

    def __init__(self, config_path: Optional[str] = None):
        config_path = config_path or os.getenv("CONFIG_PATH", "configs/config.yaml")
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = get_device()
        self._load_models()

    # ------------------------------------------------------------------ #
    #  Model loading                                                        #
    # ------------------------------------------------------------------ #

    def _load_models(self):
        cnn_path = self.cfg["paths"]["cnn_checkpoint"]
        if not os.path.exists(cnn_path):
            raise FileNotFoundError(f"CNN checkpoint not found at: {cnn_path}")

        # Thresholds
        self.classification_threshold = float(
            self.cfg["cnn"].get("classification_threshold", 0.5)
        )
        self.anomaly_threshold = float(
            self.cfg["autoencoder"].get("anomaly_threshold", 0.0)
        )

        # ── CNN ──
        self.cnn = get_densenet(num_classes=self.cfg["cnn"]["num_classes"])
        self.cnn.load_state_dict(torch.load(cnn_path, map_location=self.device))
        self.cnn.eval().to(self.device)

        # ── Grad-CAM ──
        # Target: norm5 is the final BN after all dense blocks (8×8 @ 1024ch for 256px input).
        # Always visualise class 1 (defective) so the heatmap shows "where the model
        # looked for a defect" regardless of the final label.
        self.gradcam = GradCAM(self.cnn, target_layer=self.cnn.features.norm5)
        # Process-wide master switch: is Grad-CAM available at all? Callers that never
        # want the overlay (evaluation/benchmark.py, timing runs) flip this off once.
        # Whether an individual request actually pays for it is decided per call —
        # see predict(include_heatmap=...).
        self.include_heatmap = True
        logger.info("Grad-CAM registered on features.norm5")

        # ── Autoencoder (optional — soft-fail if checkpoint missing) ──
        ae_path = self.cfg["paths"]["ae_checkpoint"]
        if os.path.exists(ae_path):
            self.ae = ConvAutoencoder()
            self.ae.load_state_dict(torch.load(ae_path, map_location=self.device))
            self.ae.eval().to(self.device)
            logger.info("Autoencoder loaded — dual-model mode active")
        else:
            self.ae = None
            logger.warning(f"AE checkpoint not found at {ae_path} — anomaly score disabled")

    # ------------------------------------------------------------------ #
    #  Inference                                                            #
    # ------------------------------------------------------------------ #

    def predict(self, img_np: np.ndarray, include_heatmap: bool = False) -> dict:
        """
        Run full inference on a uint8 RGB image array.

        Pipeline:
          1. CNN forward pass (no_grad) → defect probability + label
          2. Grad-CAM backward pass     → spatial heatmap (base64 PNG), opt-in
          3. Autoencoder forward pass   → reconstruction error / anomaly score
          4. Combine: flag needs_review when CNN and AE disagree

        include_heatmap defaults to False because step 2 is by far the most
        expensive part of the request: it needs a backward pass through DenseNet,
        which on a small shared CPU costs several times the forward pass alone.
        The verdict never depends on it, so the fast path skips it and returns
        heatmap_b64 = None; explainability is something the caller asks for.
        """
        start = time.perf_counter()

        # ── 1. CNN classification ──────────────────────────────────────
        cnn_tensor = preprocess_image(img_np).to(self.device)

        with torch.no_grad():
            logits = self.cnn(cnn_tensor.unsqueeze(0))
            probs  = torch.softmax(logits, dim=1)[0]

        defect_prob    = float(probs[1].item())
        is_defective   = defect_prob >= self.classification_threshold
        cnn_confidence = defect_prob if is_defective else float(probs[0].item())

        # ── 2. Grad-CAM heatmap ────────────────────────────────────────
        heatmap_b64: Optional[str] = None
        if include_heatmap and self.include_heatmap:
            try:
                # Always visualise class 1 (defect signal) — even on good images,
                # this shows which regions the model checked and found unremarkable.
                cam     = self.gradcam.generate(cnn_tensor, class_idx=1)
                # The overlay is drawn on the 256px view the model actually saw,
                # so resize here rather than on the fast path that never needs it.
                overlay = overlay_gradcam(cv2.resize(img_np, (256, 256)), cam, alpha=0.45)
                # overlay is RGB; imencode expects BGR
                _, buf      = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                heatmap_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            except Exception as exc:
                logger.warning(f"Grad-CAM failed (non-fatal): {exc}")

        # ── 3. Autoencoder anomaly score ───────────────────────────────
        anomaly_score: Optional[float] = None
        needs_review  = False

        if self.ae is not None:
            try:
                ae_tensor = preprocess_image_ae(img_np).to(self.device)
                with torch.no_grad():
                    recon         = self.ae(ae_tensor.unsqueeze(0))
                    anomaly_score = float(
                        ((ae_tensor.unsqueeze(0) - recon) ** 2).mean().item()
                    )
                # Flag for human review when the two models disagree
                ae_says_defective = anomaly_score >= self.anomaly_threshold
                needs_review      = is_defective != ae_says_defective
            except Exception as exc:
                logger.warning(f"Autoencoder inference failed (non-fatal): {exc}")

        return {
            "prediction":             "defective" if is_defective else "good",
            "confidence":             round(cnn_confidence, 4),
            "defect_prob":            round(defect_prob, 4),
            "classification_threshold": round(self.classification_threshold, 4),
            "anomaly_score":          round(anomaly_score, 6) if anomaly_score is not None else None,
            "anomaly_threshold":      self.anomaly_threshold,
            "needs_review":           needs_review,
            "heatmap_b64":            heatmap_b64,
            "latency_ms":             round((time.perf_counter() - start) * 1000, 2),
        }


# ------------------------------------------------------------------ #
#  CLI entry point                                                      #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Skip the Grad-CAM backward pass (matches the API's fast default).",
    )
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # The CLI is a local inspection tool, so it keeps the heatmap on by default —
    # the opposite of the API, where latency is what a visitor notices first.
    predictor = DefectPredictor(config_path=args.config)
    result    = predictor.predict(img_rgb, include_heatmap=not args.no_heatmap)

    print("\n─── DefectScope Prediction ───")
    print(f"Prediction:    {result['prediction'].upper()}")
    print(f"Defect prob:   {result['defect_prob']:.1%}  (threshold {result['classification_threshold']:.1%})")
    print(f"Confidence:    {result['confidence']:.1%}")
    if result["anomaly_score"] is not None:
        print(f"Anomaly score: {result['anomaly_score']:.6f}  (threshold {result['anomaly_threshold']:.6f})")
    print(f"Needs review:  {result['needs_review']}")
    print(f"Heatmap:       {'yes' if result['heatmap_b64'] else 'no'}")
    print(f"Latency:       {result['latency_ms']} ms")


if __name__ == "__main__":
    main()
