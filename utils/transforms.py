import cv2
import numpy as np
import torch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img_np: np.ndarray, image_size: int = 256) -> torch.Tensor:
    """
    Preprocess a raw uint8 RGB image for model inference.

    Returns a float32 tensor of shape (C, H, W), normalized with ImageNet stats.
    This is the single source of truth for inference preprocessing —
    training transforms (albumentations) should use the same stats.
    """
    img = cv2.resize(img_np, (image_size, image_size))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    # HWC → CHW
    return torch.from_numpy(img.transpose(2, 0, 1))
