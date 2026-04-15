"""
Train the CNN defect classifier (DenseNet-121 with transfer learning).

Usage:
    python -m training.train_cnn --category bottle --epochs 30

Logs to Weights & Biases. Set WANDB_API_KEY in .env before running.
"""

import argparse
import os

from dotenv import load_dotenv
load_dotenv()

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import wandb
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import build_classifier_dataloaders
from models.cnn_classifier import get_densenet
from utils import get_device


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def eval_epoch(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            correct += (outputs.argmax(1) == labels.to(device)).sum().item()
            total += len(labels)

    return {"accuracy": correct / total}


def main(args):
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    category = args.category or cfg["data"]["category"]
    epochs   = args.epochs   or cfg["cnn"]["epochs"]

    wandb.init(
        project=cfg["wandb"]["project"],
        name=f"densenet121-{category}",
        config={**cfg["cnn"], "category": category},
    )

    device = get_device()
    print(f"Training on: {device}")

    train_loader, val_loader, class_counts = build_classifier_dataloaders(
        root_dir=cfg["data"]["root_dir"],
        category=category,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    model = get_densenet(
        num_classes=cfg["cnn"]["num_classes"],
        freeze_layers=cfg["cnn"]["freeze_layers"],
    ).to(device)

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["cnn"]["learning_rate"],
        weight_decay=cfg["cnn"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    defective_weight = class_counts["train_good"] / max(class_counts["train_defective"], 1)
    class_weights = torch.tensor([1.0, defective_weight], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_acc = 0.0

    for epoch in range(epochs):
        loss, acc = train_epoch(model, train_loader, optimizer, criterion, device)
        metrics   = eval_epoch(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch + 1:02d}/{epochs} | Loss: {loss:.4f} | Train Acc: {acc:.3f} | Val Acc: {metrics['accuracy']:.3f}")
        wandb.log({"epoch": epoch + 1, "train_loss": loss, "train_acc": acc, **metrics})

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), cfg["paths"]["cnn_checkpoint"])

    print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args)
