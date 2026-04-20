"""
Export the trained CNN to ONNX format for faster CPU inference.

ONNX Runtime typically gives 1.5-2x speedup over PyTorch on CPU.
Not necessary for GPU deployments, but useful for edge deployment
on factory floor hardware where GPUs aren't available.

Usage:
    python scripts/export_onnx.py

Then benchmark with:
    python -m evaluation.benchmark  (set USE_ONNX=true in .env)
"""

import torch
import yaml

from models.cnn_classifier import get_densenet


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    checkpoint = cfg["paths"]["cnn_checkpoint"]
    output_path = checkpoint.replace(".pth", ".onnx")

    model = get_densenet(num_classes=2, freeze_layers=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    print(f"ONNX model exported to: {output_path}")
    print("Run with onnxruntime: pip install onnxruntime")
    print("Load with: ort.InferenceSession(output_path)")


if __name__ == "__main__":
    main()
