"""
Dataset tests — these run without the MVTec dataset by mocking the filesystem.
"""

import os
import tempfile
import numpy as np
import pytest
import cv2
from unittest.mock import patch

from data.dataset import MVTecDataset, get_train_transforms, get_eval_transforms


def create_fake_mvtec(tmpdir: str, category: str = "bottle"):
    """Create a minimal MVTec-like directory structure with random images."""
    splits = {
        "train": ["good"],
        "test": ["good", "broken_large"],
    }
    paths = []
    for split, labels in splits.items():
        for label in labels:
            folder = os.path.join(tmpdir, category, split, label)
            os.makedirs(folder, exist_ok=True)
            for i in range(3):
                img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                path = os.path.join(folder, f"{i:03d}.png")
                cv2.imwrite(path, img)
                paths.append(path)
    return paths


class TestMVTecDataset:
    def test_loads_train_split(self, tmp_path):
        create_fake_mvtec(str(tmp_path))
        ds = MVTecDataset(
            str(tmp_path), "bottle", split="train",
            transform=get_eval_transforms(),
        )
        assert len(ds) == 3  # 3 good images in train

    def test_loads_test_split(self, tmp_path):
        create_fake_mvtec(str(tmp_path))
        ds = MVTecDataset(
            str(tmp_path), "bottle", split="test",
            transform=get_eval_transforms(),
        )
        assert len(ds) == 6  # 3 good + 3 defective

    def test_good_only_flag(self, tmp_path):
        create_fake_mvtec(str(tmp_path))
        ds = MVTecDataset(
            str(tmp_path), "bottle", split="test",
            transform=get_eval_transforms(),
            good_only=True,
        )
        labels = [ds[i][1] for i in range(len(ds))]
        assert all(l == 0 for l in labels), "good_only=True returned defective samples"

    def test_getitem_returns_tensor_and_label(self, tmp_path):
        import torch
        create_fake_mvtec(str(tmp_path))
        ds = MVTecDataset(
            str(tmp_path), "bottle", split="train",
            transform=get_eval_transforms(),
        )
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 256, 256)
        assert label in (0, 1)

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ds = MVTecDataset(str(tmp_path), "nonexistent_category", split="train")
            ds._load_data()
