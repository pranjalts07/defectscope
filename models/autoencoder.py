import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder for unsupervised anomaly detection.

    The core idea: train only on "good" images. The network learns to reconstruct
    normal product appearance. At test time, defective images produce higher
    reconstruction error (MSE) because the decoder hasn't learned to reproduce
    anomalous regions.

    Architecture: 4-layer encoder compresses 256x256 → 16x16x256 latent space,
    then 4-layer decoder reconstructs back to 256x256x3.
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 256 → 128
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 128 → 64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 64 → 32
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 32 → 16
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 16 → 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 32 → 64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 64 → 128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # 128 → 256
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation — useful for debugging and visualization."""
        return self.encoder(x)
