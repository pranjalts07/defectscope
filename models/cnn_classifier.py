import torch
import torch.nn as nn
from torchvision.models import densenet121


def get_densenet(num_classes: int = 2, dropout: float = 0.4) -> nn.Module:
    """
    DenseNet-121 with this project's classification head — the model that is
    actually shipped for inference (DefectCNN below is kept for comparison).

    The head must match models/densenet_defect.pth exactly:
        classifier.0 -> Linear(1024, 256)
        classifier.3 -> Linear(256, num_classes)
    Indices 1 and 2 are ReLU and Dropout, which hold no parameters and so do
    not appear in the checkpoint.

    weights=None on purpose: the caller loads the trained checkpoint, so there
    is no reason to pull ImageNet weights over the network at startup.
    """
    model = densenet121(weights=None)
    in_features = model.classifier.in_features  # 1024 for DenseNet-121
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )
    return model


class DefectCNN(nn.Module):
    """
    Custom CNN built from scratch. Useful for understanding architecture choices
    and for categories where the dataset is large enough not to need pretraining.

    In practice, DenseNet-121 outperforms this on MVTec because the training sets
    are small (200-400 images per category). Keeping this for comparison.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 — 256x256 → 128x128
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2 — 128x128 → 64x64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3 — 64x64 → 32x32
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
