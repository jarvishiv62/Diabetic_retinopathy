"""
losses.py
---------
Loss functions designed to maximize sensitivity (minimize false negatives)
in the context of imbalanced, multi-class DR classification.

Two loss functions are provided:

1. FocalLoss:
   - Down-weights easy, correctly classified examples
   - Up-weights hard, misclassified examples (especially subtle DR grades)
   - Naturally emphasizes rare, severe classes
   - Recommended for this project

2. WeightedCrossEntropyLoss:
   - Standard cross-entropy with manual class weights
   - Simple, interpretable, stable baseline
   - Use when Focal Loss causes instability

Why standard CrossEntropyLoss is insufficient for DR:
  - With balanced classes, CE treats all misclassifications equally.
  - A false negative on Severe DR is clinically far more dangerous than
    a false positive, but CE does not encode this asymmetry.
  - FocalLoss + class weighting addresses both the difficulty and the
    severity imbalance simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification (Lin et al., 2017).

    Standard cross-entropy:
        CE(p_t) = -log(p_t)

    Focal Loss:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Components:
        p_t   : Predicted probability of the true class
        α_t   : Per-class weight (handles class imbalance)
        γ     : Focusing parameter — reduces loss contribution of easy examples

    Effect on training:
        - When the model correctly classifies an example with high confidence
          (p_t → 1), the factor (1 - p_t)^γ → 0, so that sample contributes
          very little to the gradient update.
        - When the model is wrong (p_t → 0), the factor → 1 and the loss is
          equivalent to weighted CE — forcing the model to focus on hard cases.

    Clinical relevance for DR:
        - Early DR stages (Mild) are hard to distinguish from No DR.
        - Focal Loss forces the network to allocate gradient budget to these
          hard, clinically dangerous misclassifications rather than wasting
          capacity on already well-classified easy examples.

    Args:
        alpha   : Per-class weight tensor of shape (num_classes,).
                  If None, uniform weighting is used.
        gamma   : Focusing exponent (default: 2.0). Higher γ → harder focus.
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Register alpha as buffer (moves with .to(device) calls)
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : Raw model outputs, shape (B, C)
            targets : Ground truth class indices, shape (B,)

        Returns:
            Scalar loss value
        """
        # Compute standard CE loss (element-wise, no reduction)
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")

        # Compute p_t: probability of the correct class
        log_pt = F.log_softmax(logits, dim=1)                       # (B, C)
        pt = torch.exp(log_pt.gather(1, targets.unsqueeze(1)))       # (B, 1)
        pt = pt.squeeze(1)                                           # (B,)

        # Focal modulating factor
        focal_weight = (1.0 - pt) ** self.gamma

        # Focal loss = focal_weight × CE
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ─────────────────────────────────────────────
# Weighted Cross Entropy Loss
# ─────────────────────────────────────────────

class WeightedCrossEntropyLoss(nn.Module):
    """
    Class-weighted Cross Entropy Loss.

    Assigns higher penalty to misclassifications of rare or severe classes.
    This is a simpler alternative to Focal Loss with more predictable behavior.

    Args:
        weight : Per-class weight tensor of shape (num_classes,).
                 Typically set to inverse class frequency.
        reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(self, weight: torch.Tensor = None, reduction: str = "mean"):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.reduction = reduction

        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weight, reduction=self.reduction)


# ─────────────────────────────────────────────
# Loss Builder
# ─────────────────────────────────────────────

def build_loss(
    loss_type: str = "focal",
    class_weights: torch.Tensor = None,
    gamma: float = 2.0,
) -> nn.Module:
    """
    Factory function to build the loss function.

    Args:
        loss_type     : "focal" | "weighted_ce"
        class_weights : Optional per-class weight tensor (from dataset.get_class_weights())
        gamma         : Focal loss focusing parameter (only used if loss_type="focal")

    Returns:
        Loss function module
    """
    loss_type = loss_type.lower().strip()

    if loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma, reduction="mean")

    elif loss_type in ("weighted_ce", "weighted_cross_entropy"):
        return WeightedCrossEntropyLoss(weight=class_weights, reduction="mean")

    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. Choose from: 'focal', 'weighted_ce'"
        )


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    B, C = 8, 5  # 8 samples, 5 classes
    logits = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))
    weights = torch.tensor([1.0, 1.5, 1.2, 2.0, 2.5])  # Severe/Proliferative upweighted

    fl = FocalLoss(alpha=weights, gamma=2.0)
    wce = WeightedCrossEntropyLoss(weight=weights)

    print(f"Focal Loss        : {fl(logits, targets).item():.4f}")
    print(f"Weighted CE Loss  : {wce(logits, targets).item():.4f}")

    # Test via factory
    focal = build_loss("focal", weights, gamma=2.0)
    wce2  = build_loss("weighted_ce", weights)
    print(f"Factory Focal     : {focal(logits, targets).item():.4f}")
    print(f"Factory WCE       : {wce2(logits, targets).item():.4f}")