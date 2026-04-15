import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from albumentations import (
    Compose,
    HorizontalFlip,
    Rotate,
    RandomBrightnessContrast,
    Normalize,
)
from albumentations.pytorch import ToTensorV2

CLASSIFIER_VAL_RATIO = 0.2
RANDOM_SEED = 42

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
    return Compose([
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_ae_transforms() -> Compose:
    """
    Autoencoder-specific transform — pixel values normalized to [0, 1] only.

    The autoencoder decoder ends with Sigmoid() which outputs [0, 1].
    MSE loss only makes sense when input and output are in the same range.
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


class AllImagesDataset(Dataset):
    """
    Collects every image from both train/ and test/ into one flat list.

    MVTec's train split has only good images — defective images live in test/.
    To train a supervised CNN properly we need both, so we pool them here
    and let build_classifier_dataloaders do a stratified split.
    """

    def __init__(self, root_dir: str, category: str, transform: Compose | None = None):
        self.transform = transform
        self.image_paths: list[str] = []
        self.labels: list[int] = []

        self._collect_split(root_dir, category, "train")
        self._collect_split(root_dir, category, "test")

    def _collect_split(self, root_dir: str, category: str, split: str):
        base = os.path.join(root_dir, category, split)
        if not os.path.exists(base):
            return

        for label_name in sorted(os.listdir(base)):
            folder = os.path.join(base, label_name)
            if not os.path.isdir(folder):
                continue

            label = 0 if label_name == "good" else 1

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


class TransformSubset(Dataset):
    """
    Wraps a Subset and applies a different transform than the parent dataset.

    PyTorch Subset doesn't let you override the transform after the fact.
    We need this so train and val subsets can have different augmentation pipelines.
    """

    def __init__(self, subset: Subset, transform: Compose):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        raw_img, label = self.subset[idx]
        img = self.transform(image=raw_img)["image"]
        return img, label


def build_classifier_dataloaders(
    root_dir: str,
    category: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Returns (train_loader, val_loader, class_counts) with a stratified split.

    Pools all images from train/ and test/, then splits 80/20 stratified by label.
    Augmentation is applied to training subset only.
    """
    full_dataset = AllImagesDataset(root_dir, category, transform=None)

    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=CLASSIFIER_VAL_RATIO,
        stratify=full_dataset.labels,
        random_state=RANDOM_SEED,
    )

    train_subset = Subset(full_dataset, train_indices)
    val_subset   = Subset(full_dataset, val_indices)

    train_ds = TransformSubset(train_subset, transform=get_train_transforms())
    val_ds   = TransformSubset(val_subset,   transform=get_eval_transforms())

    train_good      = sum(1 for i in train_indices if full_dataset.labels[i] == 0)
    train_defective = sum(1 for i in train_indices if full_dataset.labels[i] == 1)
    val_good        = sum(1 for i in val_indices   if full_dataset.labels[i] == 0)
    val_defective   = sum(1 for i in val_indices   if full_dataset.labels[i] == 1)

    class_counts = {
        "train_good": train_good,
        "train_defective": train_defective,
        "val_good": val_good,
        "val_defective": val_defective,
    }

    print(f"CNN train split — good: {train_good}, defective: {train_defective}")
    print(f"CNN val split   — good: {val_good},  defective: {val_defective}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, class_counts


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
