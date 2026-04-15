"""
Evaluate the CNN classifier on the MVTec test set.

Usage:
    python -m evaluation.evaluate --category bottle
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import MVTecDataset, get_eval_transforms
from models.cnn_classifier import get_densenet
from utils import get_device
from utils.metrics import compute_classification_metrics


def evaluate_cnn(model, test_loader, device) -> dict:
    """CNN classifier evaluation — returns metrics dict."""
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

    return compute_classification_metrics(all_labels, all_preds, all_probs)


def main(args):
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    category = args.category or cfg["data"]["category"]
    device   = get_device()

    model = get_densenet(num_classes=cfg["cnn"]["num_classes"])
    model.load_state_dict(torch.load(cfg["paths"]["cnn_checkpoint"], map_location=device))
    model.eval().to(device)

    test_ds = MVTecDataset(
        cfg["data"]["root_dir"], category, split="test",
        transform=get_eval_transforms(),
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    print(f"Evaluating CNN on {category} — {len(test_ds)} test images")
    metrics = evaluate_cnn(model, test_loader, device)

    print("\nCNN Classifier results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    out_dir = Path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{category}_results.json"
    with open(results_path, "w") as f:
        json.dump({"category": category, "cnn": metrics}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None)
    args = parser.parse_args()
    main(args)
