"""
model.py
--------
Custom CNN Architecture for Diabetic Retinopathy Classification.

Architecture: DRNet
  ┌─────────────────────────────────────────────────────────────┐
  │  Input (B, 3, 256, 256)                                     │
  │  → Stem Block (initial feature extraction)                  │
  │  → ConvBlock 1 + SE Attention  (64 channels)                │
  │  → ConvBlock 2 + SE Attention  (128 channels)               │
  │  → ConvBlock 3 + SE Attention  (256 channels)               │
  │  → ConvBlock 4 + SE Attention  (512 channels)               │
  │  → Global Average Pooling                                   │
  │  → Dropout (0.5)                                            │
  │  → FC Head → 5-class output                                 │
  └─────────────────────────────────────────────────────────────┘

Design Rationale (Research Perspective):
─────────────────────────────────────────
1. Custom CNN vs. Pretrained Backbone:
   Although pretrained models (ResNet, EfficientNet) offer strong features,
   training from scratch gives full architectural control and avoids the risk
   of ImageNet features being misaligned with retinal image statistics.
   With careful regularization, a custom CNN can match pretrained performance
   on small domain-specific datasets.

2. Squeeze-and-Excitation (SE) Block:
   Fundus images contain spatially distributed lesions (microaneurysms,
   hemorrhages, exudates) that have different diagnostic relevance per DR grade.
   SE blocks perform global channel recalibration — they learn "which feature
   maps (channels) matter most" for a given input, improving sensitivity to
   subtle pathology without adding significant parameters.

3. Batch Normalization:
   Stabilizes training of deep CNNs by reducing internal covariate shift.
   Especially important with small batch sizes (16) and a small dataset
   where gradient estimates can be noisy.

4. Dropout (0.5 in FC, 0.3 in transitions):
   Primary regularizer to combat overfitting on 400 images. Applied after
   attention-fused feature maps and in the classification head.

5. Global Average Pooling (GAP) instead of Flatten:
   GAP reduces spatial feature maps to a 1D vector by averaging, which:
   - Dramatically reduces parameters (no huge FC layer after conv)
   - Acts as structural regularization
   - Preserves spatial semantics used by Grad-CAM

6. Progressive Channel Expansion (64→128→256→512):
   Follows the principle of learning increasingly abstract representations —
   early layers capture edges/textures, deeper layers capture lesion patterns.

7. No MaxPool after every block (uses stride=2 in conv):
   Strided convolutions are preferred over max-pooling because they are
   learnable and preserve more spatial information — important for detecting
   small lesions in fundus images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Squeeze-and-Excitation Block
# ─────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Hu et al., 2018).

    Mechanism:
      1. SQUEEZE: Global Average Pool → compress spatial (H,W) to (1,1)
                  Each channel's global distribution is summarized into a scalar.
      2. EXCITATION: Two FC layers with ReLU + Sigmoid learn per-channel weights.
      3. SCALE: Original feature map is multiplied channel-wise by learned weights.

    In DR context:
      Different channels respond to different retinal structures (vessels, optic disc,
      fovea, lesions). SE lets the network upweight channels most diagnostic for each
      severity class and suppress irrelevant background channels.

    Args:
        channels  : Number of input/output channels
        reduction : Bottleneck ratio for the excitation FC layers (default: 16)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()

        # Ensure bottleneck has at least 1 unit
        bottleneck = max(channels // reduction, 1)

        self.squeeze = nn.AdaptiveAvgPool2d(1)  # (B, C, H, W) → (B, C, 1, 1)

        self.excitation = nn.Sequential(
            nn.Flatten(),                                    # (B, C, 1, 1) → (B, C)
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),                                    # Output ∈ (0, 1) per channel
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()

        # Squeeze: global spatial information per channel
        s = self.squeeze(x)            # (B, C, 1, 1)

        # Excitation: learn channel importance weights
        e = self.excitation(s)         # (B, C)
        e = e.view(b, c, 1, 1)        # (B, C, 1, 1) — broadcast over H,W

        # Scale: recalibrate feature map
        return x * e                   # (B, C, H, W)


# ─────────────────────────────────────────────
# Convolutional Block
# ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Fundamental building block of DRNet.

    Structure:
        Conv2d → BN → ReLU → Conv2d → BN → ReLU → SEBlock

    Two convolutional layers per block allow learning richer
    intra-block feature combinations before attention recalibration.

    Args:
        in_channels  : Input channels
        out_channels : Output channels
        stride       : Stride of the first conv (use 2 for downsampling)
        dropout_p    : Dropout probability applied after SE block
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_p: float = 0.0,
    ):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            # First conv: may downsample (stride=2)
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Second conv: same spatial size, refine features
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # SE attention after feature extraction
        self.se = SEBlock(out_channels, reduction=16)

        # Optional dropout for regularization
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

        # Residual shortcut connection (if dimensions change)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.block(x)
        out = self.se(out)         # Apply SE attention
        out = self.dropout(out)    # Apply dropout (Dropout2d drops entire channels)

        # Residual connection: helps gradient flow in deeper network
        out = out + residual
        out = F.relu(out, inplace=True)

        return out


# ─────────────────────────────────────────────
# Main DRNet Architecture
# ─────────────────────────────────────────────

class DRNet(nn.Module):
    """
    DRNet: Custom CNN for 5-class Diabetic Retinopathy Classification.

    Input  : (B, 3, 256, 256)
    Output : (B, 5) — raw logits (apply softmax for probabilities)

    Feature map size progression:
        Input        : 256×256
        After stem   : 128×128  (64 ch)
        After block1 : 64×64   (64 ch)
        After block2 : 32×32   (128 ch)
        After block3 : 16×16   (256 ch)
        After block4 : 8×8     (512 ch)
        After GAP    : 1×1     (512 ch)
        After FC     : 5

    Args:
        num_classes : Number of output classes (default: 5)
        dropout_p   : Dropout probability for FC head (default: 0.5)
    """

    def __init__(self, num_classes: int = 5, dropout_p: float = 0.5):
        super(DRNet, self).__init__()

        # ── Stem Block ───────────────────────────────────────────────────────
        # Larger 7×7 kernel captures low-level patterns (vessel edges) over a
        # wider receptive field in the first layer — standard in medical imaging.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 256→64 after stem
        )

        # ── Convolutional Stages ─────────────────────────────────────────────
        # Each block uses stride=2 in the first conv to halve spatial dimensions.
        # Dropout2d (spatial dropout) drops entire feature maps — more effective
        # than element-wise dropout for convolutional layers.
        self.stage1 = ConvBlock(64,  64,  stride=1, dropout_p=0.1)   # 64→64
        self.stage2 = ConvBlock(64,  128, stride=2, dropout_p=0.2)   # 64→32
        self.stage3 = ConvBlock(128, 256, stride=2, dropout_p=0.3)   # 32→16
        self.stage4 = ConvBlock(256, 512, stride=2, dropout_p=0.3)   # 16→8

        # ── Global Average Pooling ───────────────────────────────────────────
        # Aggregates spatial feature maps to a fixed-length vector regardless
        # of input size. Also creates the class activation maps used by Grad-CAM.
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B, 512, 8, 8) → (B, 512, 1, 1)

        # ── Classification Head ──────────────────────────────────────────────
        # Two FC layers with dropout: transforms features to class logits.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

        # ── Weight Initialization ────────────────────────────────────────────
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Kaiming (He) initialization for conv layers.
        This accounts for ReLU non-linearities and helps avoid vanishing/
        exploding gradients when training from scratch.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass. Returns raw logits."""
        x = self.stem(x)      # Initial feature extraction
        x = self.stage1(x)    # Stage 1: local texture features
        x = self.stage2(x)    # Stage 2: intermediate lesion features
        x = self.stage3(x)    # Stage 3: complex pathological patterns
        x = self.stage4(x)    # Stage 4: high-level semantic features
        x = self.gap(x)       # Global spatial aggregation
        x = self.classifier(x)  # Map to class logits
        return x

    def get_feature_maps(self, x: torch.Tensor):
        """
        Forward pass that also returns the final conv feature maps.
        Used by Grad-CAM to compute class activation maps.

        Returns:
            (logits, feature_maps_before_gap)
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feature_maps = self.stage4(x)   # (B, 512, 8, 8) — used by Grad-CAM
        pooled = self.gap(feature_maps)
        logits = self.classifier(pooled)
        return logits, feature_maps

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────

def build_model(num_classes: int = 5, dropout_p: float = 0.5) -> DRNet:
    """
    Factory function to instantiate DRNet.

    Args:
        num_classes : Number of DR severity classes (default: 5)
        dropout_p   : FC head dropout rate (default: 0.5)

    Returns:
        Initialized DRNet model (weights on CPU by default)
    """
    model = DRNet(num_classes=num_classes, dropout_p=dropout_p)
    return model


# ─────────────────────────────────────────────
# Sanity Check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = build_model()
    print("=" * 60)
    print("DRNet Architecture Summary")
    print("=" * 60)
    print(model)
    print("=" * 60)
    print(f"Total trainable parameters: {model.count_parameters():,}")
    print("=" * 60)

    # Forward pass test
    dummy_input = torch.randn(4, 3, 256, 256)  # Batch of 4 images
    logits = model(dummy_input)
    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {logits.shape}")   # Expected: (4, 5)

    # Feature map test (for Grad-CAM)
    logits, fmaps = model.get_feature_maps(dummy_input)
    print(f"Feature maps : {fmaps.shape}")    # Expected: (4, 512, 8, 8)

    # Probabilities
    probs = torch.softmax(logits, dim=1)
    print(f"Probabilities: {probs[0].detach().numpy().round(3)}")