"""
evaluate.py
-----------
Standalone evaluation script for a trained DRNet model.

Generates:
  1. Confusion Matrix (normalized + raw)
  2. Full Classification Report (precision, recall, F1 per class)
  3. Per-class sensitivity breakdown
  4. ROC curves (one-vs-rest per class)
  5. AUC summary table
  6. JSON evaluation report

Usage:
    python src/evaluate.py \
        --checkpoint outputs/checkpoints/best_fold1.pth \
        --image_dir data/images \
        --csv_path data/labels.csv \
        [--split val|all]
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import DRDataset, get_val_transforms, CLASS_NAMES, NUM_CLASSES, safe_collate
from src.model import build_model
from src.utils import create_output_dirs, get_device, load_checkpoint, save_metrics_json

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


# ─────────────────────────────────────────────
# Inference Loop
# ─────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """
    Run inference over a DataLoader and collect predictions and probabilities.

    Returns:
        all_labels : Ground truth class indices (numpy array)
        all_preds  : Predicted class indices (numpy array)
        all_probs  : Softmax probability matrix, shape (N, 5)
    """
    model.eval()

    all_labels = []
    all_preds  = []
    all_probs  = []

    for images, labels in loader:
        if images is None:
            continue

        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# ─────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: str,
    normalize: bool = True,
) -> np.ndarray:
    """
    Plot and save a styled confusion matrix.

    Normalized CM shows recall per row (what fraction of each true class
    was correctly identified). This is the medically relevant view since
    we care most about rows: "of all true Severe DR cases, how many did
    the model correctly classify?"

    Args:
        labels     : Ground truth labels
        preds      : Predicted labels
        save_path  : Output file path
        normalize  : If True, normalize by true class totals (row normalization)

    Returns:
        The confusion matrix (raw counts)
    """
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    cm_raw = cm.copy()

    if normalize:
        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm_norm, row_sums, where=row_sums != 0)
    else:
        cm_display = cm.astype(float)

    class_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("DRNet — Confusion Matrix", fontsize=14, fontweight="bold")

    for ax, matrix, title, fmt in zip(
        axes,
        [cm_display, cm_raw],
        ["Normalized (Row = True Class)", "Raw Counts"],
        [".2f", "d"]
    ):
        sns.heatmap(
            matrix,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            vmin=0, vmax=1 if fmt == ".2f" else None,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {save_path}")
    return cm_raw


# ─────────────────────────────────────────────
# ROC Curves
# ─────────────────────────────────────────────

def plot_roc_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: str,
) -> dict:
    """
    Plot one-vs-rest ROC curves for all 5 DR classes.

    ROC curves visualize the tradeoff between sensitivity (TPR) and
    1-specificity (FPR) across all classification thresholds.
    For DR, we want each class curve to be high and toward the upper-left.

    Returns:
        Dictionary of {class_name: auc_score}
    """
    # Binarize labels for one-vs-rest
    labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    auc_scores = {}

    fig, ax = plt.subplots(figsize=(9, 7))

    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        auc_val = auc(fpr, tpr)
        auc_scores[CLASS_NAMES[i]] = auc_val
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{CLASS_NAMES[i]} (AUC = {auc_val:.3f})")

    # Diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("ROC Curves — DRNet (One-vs-Rest per Class)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curves saved → {save_path}")
    return auc_scores


# ─────────────────────────────────────────────
# Per-Class Sensitivity Report
# ─────────────────────────────────────────────

def print_sensitivity_report(labels: np.ndarray, preds: np.ndarray) -> dict:
    """
    Print detailed per-class sensitivity and specificity.

    For each class c:
        Sensitivity = TP_c / (TP_c + FN_c)  ← did we catch all true cases?
        Specificity = TN_c / (TN_c + FP_c)  ← did we avoid false alarms?

    In DR:
        - Low sensitivity for Severe/Proliferative = missed diagnoses = harm
        - We accept lower specificity to maximize sensitivity

    Returns:
        Dictionary of per-class metrics
    """
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    per_class_metrics = {}

    print("\n" + "=" * 65)
    print(f"{'Class':<22} {'Sensitivity':>12} {'Specificity':>12} {'Support':>8}")
    print("=" * 65)

    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        support = cm[i, :].sum()

        flag = "⚠ LOW" if sensitivity < 0.70 else ""
        print(
            f"  {CLASS_NAMES[i]:<20} {sensitivity:>12.4f} {specificity:>12.4f} "
            f"{support:>8}  {flag}"
        )

        per_class_metrics[CLASS_NAMES[i]] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "support": int(support),
        }

    print("=" * 65)
    return per_class_metrics


# ─────────────────────────────────────────────
# Main Evaluation Function
# ─────────────────────────────────────────────

def evaluate(
    checkpoint_path: str,
    image_dir: str,
    csv_path: str,
    output_dir: str = "outputs",
    batch_size: int = 16,
    num_workers: int = 2,
    image_size: int = 256,
) -> dict:
    """
    Full evaluation pipeline.

    Args:
        checkpoint_path : Path to saved .pth checkpoint
        image_dir       : Directory of images
        csv_path        : CSV with filename + label
        output_dir      : Base output directory
        batch_size      : Inference batch size
        num_workers     : DataLoader workers
        image_size      : Input resolution

    Returns:
        Evaluation results dictionary
    """
    output_dirs = create_output_dirs(output_dir)
    device = get_device()

    # ── Load Model ────────────────────────────────────────────────────────────
    model = build_model().to(device)
    ckpt  = load_checkpoint(model, checkpoint_path, device=device)
    fold  = ckpt.get("fold", "?")
    epoch = ckpt.get("epoch", "?")
    logger.info(f"Evaluating checkpoint from Fold {fold}, Epoch {epoch}")

    # ── Load Dataset ──────────────────────────────────────────────────────────
    dataset = DRDataset(
        image_dir, csv_path,
        transform=get_val_transforms(image_size)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=safe_collate,
        pin_memory=True,
    )

    logger.info(f"Evaluating on {len(dataset)} samples...")

    # ── Run Inference ─────────────────────────────────────────────────────────
    labels, preds, probs = run_inference(model, loader, device)

    # ── Classification Report ─────────────────────────────────────────────────
    class_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    report = classification_report(
        labels, preds,
        target_names=class_names,
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    # ── Sensitivity Report ────────────────────────────────────────────────────
    per_class = print_sensitivity_report(labels, preds)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm_path = str(output_dirs["plots"] / f"confusion_matrix_fold{fold}.png")
    cm_raw = plot_confusion_matrix(labels, preds, cm_path, normalize=True)

    # ── ROC Curves ────────────────────────────────────────────────────────────
    roc_path = str(output_dirs["plots"] / f"roc_curves_fold{fold}.png")
    auc_scores = plot_roc_curves(labels, probs, roc_path)

    macro_auc = np.mean(list(auc_scores.values()))
    logger.info(f"Macro AUC: {macro_auc:.4f}")
    for cls, auc_val in auc_scores.items():
        logger.info(f"  AUC {cls}: {auc_val:.4f}")

    # ── Compile Results ───────────────────────────────────────────────────────
    results = {
        "fold":          fold,
        "epoch":         epoch,
        "checkpoint":    checkpoint_path,
        "n_samples":     len(labels),
        "macro_auc":     macro_auc,
        "per_class_auc": auc_scores,
        "per_class":     per_class,
    }

    save_metrics_json(
        results,
        str(output_dirs["reports"] / f"eval_fold{fold}.json")
    )
    logger.info("Evaluation complete.")
    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DRNet checkpoint"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pth checkpoint file")
    parser.add_argument("--image_dir",  default="data/images",
                        help="Directory of images")
    parser.add_argument("--csv_path",   default="data/labels.csv",
                        help="CSV with filename + label columns")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size",  type=int, default=256)

    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        image_dir=args.image_dir,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )