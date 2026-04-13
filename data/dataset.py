import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from albumentations import (
    Compose,
    HorizontalFlip,
    Rotate,
    RandomBrightnessContrast,
    Normalize,
)
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 256) -> Compose:
    return Compose([
        HorizontalFlip(p=0.5),
        Rotate(limit=15, p=0.5),
        RandomBrightnessContrast(p=0.3),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transforms(image_size: int = 256) -> Compose:
    # No augmentation at eval time — just normalize
    return Compose([
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_ae_transforms() -> Compose:
    """
    Autoencoder-specific transform — pixel values normalized to [0, 1] only.

    The autoencoder decoder ends with Sigmoid() which outputs [0, 1].
    MSE loss only makes sense when input and output are in the same range.
    ImageNet normalization produces values roughly in [-2, 2] — completely
    incompatible with Sigmoid output.
    """
    return Compose([
        Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])


class MVTecDataset(Dataset):
    """
    Loads one category from the MVTec Anomaly Detection dataset.

    MVTec structure:
        <root>/<category>/train/good/*.png
        <root>/<category>/test/good/*.png
        <root>/<category>/test/<defect_type>/*.png

    Labels: 0 = good, 1 = defective (any defect type)
    """

    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = "train",
        transform: Compose | None = None,
        good_only: bool = False,
    ):
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.transform = transform
        self.good_only = good_only

        self.image_paths: list[str] = []
        self.labels: list[int] = []

        self._load_data()

    def _load_data(self):
        base = os.path.join(self.root_dir, self.category, self.split)

        if not os.path.exists(base):
            raise FileNotFoundError(
                f"Dataset directory not found: {base}\n"
                f"Download MVTec AD from https://www.mvtec.com/company/research/datasets/mvtec-ad"
            )

        for label_name in sorted(os.listdir(base)):
            label = 0 if label_name == "good" else 1

            if self.good_only and label != 0:
                continue

            folder = os.path.join(base, label_name)
            if not os.path.isdir(folder):
                continue

            for img_file in sorted(os.listdir(folder)):
                if not img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    continue
                self.image_paths.append(os.path.join(folder, img_file))
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, self.labels[idx]

    def class_counts(self) -> dict[str, int]:
        good = sum(1 for l in self.labels if l == 0)
        defective = sum(1 for l in self.labels if l == 1)
        return {"good": good, "defective": defective}


def build_ae_loader(
    root_dir: str,
    category: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """Autoencoder trains only on good images using [0,1] normalization."""
    good_ds = MVTecDataset(
        root_dir, category, split="train",
        transform=get_ae_transforms(),
        good_only=True,
    )
    return DataLoader(good_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
