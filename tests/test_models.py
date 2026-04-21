"""
Unit tests for model forward passes.

These don't require the MVTec dataset — they just check that the model
architectures are correct and produce the right output shapes.
"""

import torch
import pytest

from models.cnn_classifier import DefectCNN, get_densenet
from models.autoencoder import ConvAutoencoder


BATCH_SIZE = 4
IMG_SIZE = 256


def make_fake_batch():
    return torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)


class TestDefectCNN:
    def test_output_shape(self):
        model = DefectCNN(num_classes=2)
        model.eval()
        x = make_fake_batch()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, 2), f"Expected (B, 2), got {out.shape}"

    def test_no_nan_in_output(self):
        model = DefectCNN()
        x = make_fake_batch()
        out = model(x)
        assert not torch.isnan(out).any()


class TestDenseNet:
    def test_output_shape(self):
        model = get_densenet(num_classes=2, freeze_layers=False)
        model.eval()
        x = make_fake_batch()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, 2)

    def test_frozen_layers(self):
        model = get_densenet(num_classes=2, freeze_layers=True)
        frozen = [p for p in model.parameters() if not p.requires_grad]
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(frozen) > 0, "Expected some frozen layers"
        assert len(trainable) > 0, "Expected some trainable layers"


class TestAutoencoder:
    def test_reconstruction_shape(self):
        model = ConvAutoencoder()
        model.eval()
        x = make_fake_batch()
        with torch.no_grad():
            recon = model(x)
        assert recon.shape == x.shape, f"Reconstruction shape mismatch: {recon.shape} vs {x.shape}"

    def test_output_range(self):
        """Sigmoid output should be in [0, 1]"""
        model = ConvAutoencoder()
        x = make_fake_batch()
        with torch.no_grad():
            recon = model(x)
        assert recon.min() >= 0.0 and recon.max() <= 1.0

    def test_reconstruction_error_not_zero(self):
        """Random input should have nonzero reconstruction error (untrained model)"""
        model = ConvAutoencoder()
        x = make_fake_batch()
        with torch.no_grad():
            recon = model(x)
            err = ((x - recon) ** 2).mean()
        assert err.item() > 0.0
