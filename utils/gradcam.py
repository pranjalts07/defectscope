import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Hooks into a target convolutional layer and uses the gradient of the
    target class score w.r.t. that layer's activations to weight the
    feature maps. The weighted sum gives a coarse heatmap showing which
    spatial regions most influenced the prediction.

    Usage:
        cam = GradCAM(model, target_layer=model.features[-1])
        heatmap = cam.generate(img_tensor, class_idx=1)  # 1 = defective
        overlay = overlay_gradcam(original_img_np, heatmap)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._register_hooks()

    def _save_gradient(self, grad: torch.Tensor):
        self.gradients = grad.detach()

    def _register_hooks(self):
        def save_activation(module, input, output):
            self.activations = output.detach()
            # Hook the activation tensor, not the module. torchvision's DenseNet
            # applies F.relu(..., inplace=True) to norm5's output, which makes a
            # module-level full_backward_hook raise "Output 0 of
            # BackwardHookFunctionBackward is a view and is being modified inplace".
            if output.requires_grad:
                output.register_hook(self._save_gradient)

        self.target_layer.register_forward_hook(save_activation)

    def generate(self, img_tensor: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """
        Generate a normalized CAM heatmap for the given class index.

        Args:
            img_tensor: Single preprocessed image (C, H, W) — no batch dim
            class_idx: Which class to visualize. 1 = defective is the usual choice.

        Returns:
            heatmap: float32 array in [0, 1], shape (256, 256)
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(img_tensor.unsqueeze(0))
        output[0, class_idx].backward()

        # Global average pooling over spatial dims to get per-channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (256, 256))

        # Normalize to [0, 1] — eps prevents divide-by-zero on all-zero maps
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.astype(np.float32)


def overlay_gradcam(
    img_np: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Blend a GradCAM heatmap onto the original image.

    Args:
        img_np: Original image as uint8 RGB array (H, W, 3)
        cam: Normalized heatmap from GradCAM.generate(), shape (H, W)
        alpha: Heatmap blend weight. 0.4 works well visually.

    Returns:
        overlay: uint8 RGB array (H, W, 3)
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # applyColorMap returns BGR
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    return overlay
