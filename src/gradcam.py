"""
gradcam.py
----------
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
for DRNet interpretability.

Grad-CAM (Selvaraju et al., 2017) produces a coarse heatmap highlighting
the regions of the input image that were most influential for a given
class prediction.

How Grad-CAM works:
  1. Forward pass → get logits
  2. Backpropagate gradients of the target class score w.r.t. the final
     convolutional feature maps (stage4 output)
  3. Global-average-pool the gradients over the spatial dimensions
     → get "importance weights" α_k for each channel k
  4. Compute weighted combination of feature maps:
     L = ReLU(Σ_k  α_k · A^k)
  5. Upsample L to input image size and overlay

Why Grad-CAM matters for DR:
  - Clinicians need to trust the model's reasoning
  - Grad-CAM reveals which retinal regions drove the prediction:
    optic disc, fovea, or peripheral lesion zones
  - If the model highlights noise instead of lesions, it signals
    a spurious correlation — critical to detect before deployment

Usage:
    python src/gradcam.py \
        --checkpoint outputs/checkpoints/best_fold1.pth \
        --image data/images/im001.jpg \
        --label 2 \
        [--target_class 2]
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import build_model, DRNet
from src.dataset import get_val_transforms, CLASS_NAMES, IMAGE_SIZE
from src.utils import get_device, load_checkpoint

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


# ─────────────────────────────────────────────
# GradCAM Class
# ─────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM implementation for DRNet.

    Hooks into the last convolutional stage (stage4) to capture:
      - Forward activations (feature maps)
      - Backward gradients (gradient of target class score w.r.t. activations)

    The hooks are registered on __init__ and removed on __del__ to avoid
    memory leaks during multi-image batch processing.

    Args:
        model       : DRNet model (should be in eval mode)
        target_layer: PyTorch module to attach hooks to
                      (defaults to model.stage4)
    """

    def __init__(self, model: DRNet, target_layer=None):
        self.model = model
        self.model.eval()

        # Use stage4 as the target layer: it has the richest semantic features
        # while still retaining spatial resolution (8×8 maps)
        self.target_layer = target_layer or model.stage4

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        """Forward hook: saves feature maps produced by target_layer."""
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        """Backward hook: saves gradients flowing back through target_layer."""
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> tuple:
        """
        Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor : Preprocessed image tensor, shape (1, 3, H, W)
            target_class : Class index to explain. If None, uses predicted class.

        Returns:
            (cam, predicted_class, probabilities)
              cam             : Normalized heatmap, shape (H, W), values in [0, 1]
              predicted_class : Integer class index
              probabilities   : Softmax probabilities, shape (num_classes,)
        """
        # Ensure gradient tracking
        input_tensor = input_tensor.clone().requires_grad_(True)

        # ── Forward pass ─────────────────────────────────────────────────────
        self.model.zero_grad()
        logits = self.model(input_tensor)                     # (1, 5)
        probs  = torch.softmax(logits, dim=1).squeeze(0)      # (5,)
        predicted_class = probs.argmax().item()

        # Default to predicted class if not specified
        if target_class is None:
            target_class = predicted_class

        # ── Backward pass ─────────────────────────────────────────────────────
        # Backpropagate the score of the target class only
        self.model.zero_grad()
        class_score = logits[0, target_class]
        class_score.backward()

        # ── Compute CAM ──────────────────────────────────────────────────────
        # α_k = global average pool of gradients over spatial dims (H, W)
        gradients   = self._gradients    # (1, C, h, w)
        activations = self._activations  # (1, C, h, w)

        # Channel importance weights: mean gradient magnitude per channel
        alpha = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (alpha * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)

        # ReLU: only keep positive contributions (negative = opposing evidence)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze()  # (h, w)
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cam.cpu().numpy()

        return cam, predicted_class, probs.detach().cpu().numpy()

    def __del__(self):
        """Remove hooks when object is garbage collected."""
        if hasattr(self, "_fwd_hook"):
            self._fwd_hook.remove()
        if hasattr(self, "_bwd_hook"):
            self._bwd_hook.remove()


# ─────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────

def overlay_heatmap(
    original_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_image : Original image as numpy array (H, W, 3), uint8, RGB
        cam            : Grad-CAM heatmap (h, w), values in [0, 1]
        alpha          : Heatmap opacity (0=invisible, 1=fully opaque)
        colormap       : OpenCV colormap for heatmap

    Returns:
        Overlay image, numpy array (H, W, 3), uint8, RGB
    """
    H, W = original_image.shape[:2]

    # Upsample CAM to original image resolution
    cam_resized = cv2.resize(cam, (W, H))
    cam_uint8   = (cam_resized * 255).astype(np.uint8)

    # Apply colormap (produces BGR)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend with original image
    overlay = (alpha * heatmap_rgb + (1 - alpha) * original_image).astype(np.uint8)
    return overlay


def save_gradcam_figure(
    original_image: np.ndarray,
    cam: np.ndarray,
    predicted_class: int,
    true_label: Optional[int],
    probabilities: np.ndarray,
    save_path: str,
    image_name: str = "",
) -> None:
    """
    Save a multi-panel Grad-CAM visualization figure.

    Panels:
        1. Original fundus image
        2. Grad-CAM heatmap (standalone)
        3. Heatmap overlaid on original
        4. Class probability bar chart

    Args:
        original_image  : Numpy RGB image (H, W, 3)
        cam             : Grad-CAM heatmap (h, w)
        predicted_class : Model's predicted class index
        true_label      : Ground truth label (if known)
        probabilities   : Per-class softmax probabilities
        save_path       : Output file path
        image_name      : Image filename (for title)
    """
    overlay = overlay_heatmap(original_image, cam, alpha=0.45)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    pred_name = CLASS_NAMES[predicted_class]
    true_name = CLASS_NAMES[true_label] if true_label is not None else "Unknown"
    correct   = "✓" if (true_label is not None and predicted_class == true_label) else "✗"

    fig.suptitle(
        f"Grad-CAM — {image_name}\n"
        f"Predicted: {pred_name} | True: {true_name} {correct}",
        fontsize=12, fontweight="bold"
    )

    # Panel 1: Original
    axes[0].imshow(original_image)
    axes[0].set_title("Original Fundus Image")
    axes[0].axis("off")

    # Panel 2: Heatmap
    H, W = original_image.shape[:2]
    cam_up = cv2.resize(cam, (W, H))
    im = axes[1].imshow(cam_up, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    # Panel 4: Probability bar chart
    class_names = [CLASS_NAMES[i] for i in range(len(probabilities))]
    colors = ["#2ca02c" if i == predicted_class else "#aec7e8"
              for i in range(len(probabilities))]
    bars = axes[3].barh(class_names, probabilities, color=colors, edgecolor="black", linewidth=0.5)
    axes[3].set_xlim(0, 1)
    axes[3].set_xlabel("Confidence")
    axes[3].set_title("Class Probabilities")
    axes[3].invert_yaxis()

    # Annotate bar values
    for bar, prob in zip(bars, probabilities):
        axes[3].text(
            min(prob + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
            f"{prob:.3f}", va="center", ha="left", fontsize=9
        )
    axes[3].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Grad-CAM saved → {save_path}")


# ─────────────────────────────────────────────
# High-Level API
# ─────────────────────────────────────────────

def explain_image(
    image_path: str,
    checkpoint_path: str,
    output_dir: str = "outputs/gradcam",
    true_label: Optional[int] = None,
    target_class: Optional[int] = None,
    image_size: int = IMAGE_SIZE,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Generate and save Grad-CAM for a single image.

    Args:
        image_path      : Path to the input .jpg image
        checkpoint_path : Path to trained .pth checkpoint
        output_dir      : Directory to save the visualization
        true_label      : Ground truth class (if known)
        target_class    : Class to explain (None = predicted class)
        image_size      : Model input resolution
        device          : Compute device (auto-detected if None)

    Returns:
        Dictionary with predicted_class, probabilities, cam
    """
    if device is None:
        device = get_device()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model().to(device)
    load_checkpoint(model, checkpoint_path, device=device)

    # ── Prepare image ─────────────────────────────────────────────────────────
    img_pil = Image.open(image_path).convert("RGB")
    original_np = np.array(img_pil.resize((image_size, image_size)))  # (H, W, 3) uint8

    transform = get_val_transforms(image_size)
    tensor = transform(img_pil).unsqueeze(0).to(device)  # (1, 3, H, W)

    # ── Generate Grad-CAM ─────────────────────────────────────────────────────
    gradcam = GradCAM(model)
    cam, predicted_class, probs = gradcam.generate(tensor, target_class=target_class)

    # ── Save visualization ────────────────────────────────────────────────────
    img_stem = Path(image_path).stem
    save_path = str(Path(output_dir) / f"gradcam_{img_stem}.png")
    save_gradcam_figure(
        original_image=original_np,
        cam=cam,
        predicted_class=predicted_class,
        true_label=true_label,
        probabilities=probs,
        save_path=save_path,
        image_name=Path(image_path).name,
    )

    logger.info(
        f"Image: {Path(image_path).name} | "
        f"Predicted: {CLASS_NAMES[predicted_class]} ({probs[predicted_class]:.3f})"
    )

    return {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class],
        "probabilities":   probs.tolist(),
        "cam":             cam,
        "save_path":       save_path,
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualization for a single retinal image"
    )
    parser.add_argument("--checkpoint",    required=True, help="Path to .pth checkpoint")
    parser.add_argument("--image",         required=True, help="Path to input image (.jpg)")
    parser.add_argument("--true_label",    type=int, default=None,
                        help="Ground truth label (0-4). Optional.")
    parser.add_argument("--target_class",  type=int, default=None,
                        help="Class to explain. Defaults to predicted class.")
    parser.add_argument("--output_dir",    default="outputs/gradcam")
    parser.add_argument("--image_size",    type=int, default=256)

    args = parser.parse_args()

    result = explain_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        true_label=args.true_label,
        target_class=args.target_class,
        image_size=args.image_size,
    )

    print(f"\nPrediction  : {result['predicted_label']}")
    print(f"Confidence  : {result['probabilities'][result['predicted_class']]:.4f}")
    print(f"All probs   : {[round(p, 4) for p in result['probabilities']]}")
    print(f"CAM saved to: {result['save_path']}")