"""
Train the convolutional autoencoder for unsupervised anomaly detection.

The autoencoder sees ONLY good images during training. At test time, defective
images produce higher reconstruction error because the decoder never learned to
reproduce anomalous patterns.

Usage:
    python -m training.train_autoencoder --category bottle --epochs 50
"""

import argparse
import os

import torch
import torch.nn as nn
import yaml

from data.dataset import build_ae_loader
from models.autoencoder import ConvAutoencoder
from utils import get_device
from torch.optim import Adam


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for imgs, _ in loader:
        imgs = imgs.to(device)
        recon = model(imgs)
        loss = criterion(recon, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main(args):
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    category = args.category or cfg["data"]["category"]
    epochs = args.epochs or cfg["autoencoder"]["epochs"]

    device = get_device()
    print(f"Training autoencoder on: {device}")

    train_loader = build_ae_loader(
        root_dir=cfg["data"]["root_dir"],
        category=category,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    model = ConvAutoencoder().to(device)
    optimizer = Adam(model.parameters(), lr=cfg["autoencoder"]["learning_rate"])
    criterion = nn.MSELoss()

    best_loss = float("inf")

    for epoch in range(epochs):
        loss = train_epoch(model, loader=train_loader, optimizer=optimizer,
                           criterion=criterion, device=device)
        print(f"AE Epoch {epoch + 1:03d}/{epochs} | Recon Loss: {loss:.6f}")

        if loss < best_loss:
            best_loss = loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/autoencoder.pth")

    print(f"\nBest reconstruction loss: {best_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args)
