"""
Grid search over anomaly threshold values.

The autoencoder's anomaly_threshold in config.yaml is hand-tuned.
This script finds the optimal value for your test set by sweeping
over a range and measuring F1 at each point.

Run after training the autoencoder:
    python -m evaluation.threshold_search --category bottle

Then update configs/config.yaml with the printed optimal threshold.
"""

import argparse
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score

from data.dataset import MVTecDataset, get_ae_transforms
from inference.predict import DefectPredictor


def main(args):
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    category = args.category or cfg["data"]["category"]
    predictor = DefectPredictor()

    # Must use AE transforms here — the autoencoder was trained on [0,1] inputs
    test_ds = MVTecDataset(
        cfg["data"]["root_dir"], category, split="test",
        transform=get_ae_transforms(),
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Collect anomaly scores for all test images
    predictor.ae.eval()
    labels_all, scores_all = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(predictor.device)
            recons = predictor.ae(imgs)
            errors = ((imgs - recons) ** 2).mean(dim=(1, 2, 3))
            labels_all.extend(labels.numpy())
            scores_all.extend(errors.cpu().numpy())

    scores = np.array(scores_all)
    labels = np.array(labels_all)

    print(f"Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"AUC-ROC: {roc_auc_score(labels, scores):.4f}")
    print()

    # Sweep threshold
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    results = []

    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        results.append((thresh, f1))

    best_thresh, best_f1 = max(results, key=lambda x: x[1])

    print(f"Optimal threshold: {best_thresh:.6f}")
    print(f"Best F1 at threshold: {best_f1:.4f}")
    print()
    print(f"Update configs/config.yaml:")
    print(f"  autoencoder.anomaly_threshold: {best_thresh:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None)
    args = parser.parse_args()
    main(args)
