"""
Full evaluation of both models on the MVTec test set.

Usage:
    python -m evaluation.evaluate --category bottle
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from data.dataset import MVTecDataset, get_eval_transforms, get_ae_transforms
from models.cnn_classifier import get_densenet
from models.autoencoder import ConvAutoencoder
from utils import get_device
from utils.metrics import compute_classification_metrics, find_best_threshold


def evaluate_cnn(model, test_loader, device) -> tuple:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = compute_classification_metrics(all_labels, all_preds, all_probs)
    best_thresh, best_f1 = find_best_threshold(all_labels, all_probs, metric="f1")
    metrics["best_threshold"]      = round(best_thresh, 4)
    metrics["best_f1_at_threshold"] = round(best_f1, 4)

    return metrics, all_labels, all_probs


def evaluate_autoencoder(ae_model, test_loader, device) -> tuple:
    ae_model.eval()
    all_labels, all_scores = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            recons = ae_model(imgs)
            errors = ((imgs - recons) ** 2).mean(dim=(1, 2, 3))
            all_labels.extend(labels.numpy())
            all_scores.extend(errors.cpu().numpy())

    ae_auc = roc_auc_score(all_labels, all_scores)
    best_thresh, best_f1 = find_best_threshold(all_labels, all_scores, metric="f1")
    ae_preds = [1 if s >= best_thresh else 0 for s in all_scores]
    ae_metrics = compute_classification_metrics(all_labels, ae_preds, all_scores)
    ae_metrics["best_threshold"] = round(best_thresh, 6)

    return all_labels, all_scores, ae_auc, ae_metrics


def plot_roc_curves(cnn_labels, cnn_probs, ae_labels, ae_scores, output_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))

    fpr_cnn, tpr_cnn, _ = roc_curve(cnn_labels, cnn_probs)
    fpr_ae,  tpr_ae,  _ = roc_curve(ae_labels,  ae_scores)

    auc_cnn = roc_auc_score(cnn_labels, cnn_probs)
    auc_ae  = roc_auc_score(ae_labels,  ae_scores)

    ax.plot(fpr_cnn, tpr_cnn, label=f"DenseNet-121 (AUC = {auc_cnn:.3f})", lw=2)
    ax.plot(fpr_ae,  tpr_ae,  label=f"Autoencoder (AUC = {auc_ae:.3f})",   lw=2, linestyle="--")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — DefectScope")
    ax.legend()
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {output_path}")


def main(args):
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    category = args.category or cfg["data"]["category"]
    device   = get_device()

    cnn = get_densenet(num_classes=cfg["cnn"]["num_classes"])
    cnn.load_state_dict(torch.load(cfg["paths"]["cnn_checkpoint"], map_location=device))
    cnn.eval().to(device)

    ae = ConvAutoencoder()
    ae.load_state_dict(torch.load(cfg["paths"]["ae_checkpoint"], map_location=device))
    ae.eval().to(device)

    cnn_test_ds = MVTecDataset(cfg["data"]["root_dir"], category, split="test",
                               transform=get_eval_transforms())
    ae_test_ds  = MVTecDataset(cfg["data"]["root_dir"], category, split="test",
                               transform=get_ae_transforms())

    cnn_loader = DataLoader(cnn_test_ds, batch_size=32, shuffle=False, num_workers=2)
    ae_loader  = DataLoader(ae_test_ds,  batch_size=32, shuffle=False, num_workers=2)

    print(f"Evaluating on {category} — {len(cnn_test_ds)} test images")

    cnn_metrics, cnn_labels, cnn_probs = evaluate_cnn(cnn, cnn_loader, device)
    print("\nCNN Classifier:")
    for k, v in cnn_metrics.items():
        print(f"  {k}: {v}")

    ae_labels, ae_scores, ae_auc, ae_metrics = evaluate_autoencoder(ae, ae_loader, device)
    print(f"\nAutoencoder Anomaly Detection:")
    print(f"  AUC-ROC: {ae_auc:.4f}")
    for k, v in ae_metrics.items():
        print(f"  {k}: {v}")

    results = {
        "category": category,
        "cnn": cnn_metrics,
        "autoencoder": {"auc_roc": round(ae_auc, 4)},
        "test_samples": len(cnn_test_ds),
    }

    out_dir = Path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{category}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    plot_roc_curves(cnn_labels, cnn_probs, ae_labels, ae_scores,
                    output_path="docs/assets/roc_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None)
    args = parser.parse_args()
    main(args)
