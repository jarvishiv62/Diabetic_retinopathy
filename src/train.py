"""
train.py
--------
Full K-Fold cross-validation training loop for DRNet.

This module is the core training orchestrator. It handles:
  - K-Fold loop with stratified splits
  - Per-epoch forward + backward pass
  - Metric computation (accuracy, sensitivity, F1, AUC)
  - Learning rate scheduling
  - Early stopping
  - Best-model checkpoint saving
  - Training curve plotting
  - Final cross-fold metric aggregation

Usage:
    python src/train.py [--config path/to/config.json]
    python src/train.py  (uses defaults)
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ── Local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import get_kfold_dataloaders, DRDataset, CLASS_NAMES, NUM_CLASSES
from src.model import build_model
from src.losses import build_loss
from src.utils import (
    set_seed,
    setup_logging,
    create_output_dirs,
    get_device,
    save_checkpoint,
    plot_training_curves,
    plot_fold_comparison,
    log_metrics,
    save_metrics_json,
    EarlyStopping,
)


# ─────────────────────────────────────────────
# Default Configuration
# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Data
    "image_dir":    "data/images",
    "csv_path":     "data/labels.csv",
    "image_size":   256,

    # Training
    "n_folds":      5,
    "epochs":       80,
    "batch_size":   16,
    "num_workers":  2,
    "seed":         42,

    # Model
    "num_classes":  5,
    "dropout_p":    0.5,

    # Optimizer
    "optimizer":    "adamw",          # "adam" | "adamw"
    "lr":           1e-3,
    "weight_decay": 1e-4,

    # Scheduler
    "scheduler":    "cosine",         # "cosine" | "step" | "plateau"
    "lr_min":       1e-6,             # CosineAnnealingLR minimum LR

    # Loss
    "loss_type":    "focal",          # "focal" | "weighted_ce"
    "focal_gamma":  2.0,

    # Early Stopping
    "patience":     20,               # epochs without improvement
    "es_metric":    "val_f1",         # metric to monitor

    # Output
    "output_dir":   "outputs",
}


# ─────────────────────────────────────────────
# Single Epoch: Train
# ─────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """
    Run one full training epoch.

    Args:
        model     : DRNet model (in train mode)
        loader    : Training DataLoader
        criterion : Loss function
        optimizer : Optimizer
        device    : torch.device
        epoch     : Current epoch number (for logging)

    Returns:
        (avg_loss, accuracy) over the epoch
    """
    model.train()
    logger = logging.getLogger(__name__)

    total_loss = 0.0
    all_preds  = []
    all_labels = []
    n_batches  = 0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"Training Epoch {epoch}", leave=False)):
        # Skip corrupted batches returned by safe_collate
        if images is None:
            continue

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass + gradient clipping (prevents gradient explosion)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


# ─────────────────────────────────────────────
# Single Epoch: Validate
# ─────────────────────────────────────────────

@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    """
    Run one full validation epoch. No gradient computation.

    Computes:
      - Loss and accuracy
      - Per-class and macro sensitivity (recall) — primary metric for DR
      - Macro F1 score
      - Macro AUC (one-vs-rest)

    Returns:
        Dictionary with all validation metrics
    """
    model.eval()

    total_loss = 0.0
    all_preds   = []
    all_labels  = []
    all_probs   = []   # Softmax probabilities for AUC
    n_batches   = 0

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        if images is None:
            continue

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

        total_loss  += loss.item()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        n_batches   += 1

    # ── Compute Metrics ──────────────────────────────────────────────────────
    avg_loss = total_loss / max(n_batches, 1)
    accuracy = accuracy_score(all_labels, all_preds)

    # Macro F1: treats all classes equally (good for balanced datasets)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Per-class recall (sensitivity): how many true positives are caught
    # This is THE critical metric for DR — missing a Severe case is dangerous
    recall_per_class = recall_score(
        all_labels, all_preds, average=None,
        labels=list(range(NUM_CLASSES)), zero_division=0
    )
    macro_sensitivity = recall_per_class.mean()

    # AUC (one-vs-rest): measures discrimination ability across all thresholds
    try:
        all_probs_np = np.array(all_probs)
        auc = roc_auc_score(
            all_labels, all_probs_np,
            multi_class="ovr", average="macro",
            labels=list(range(NUM_CLASSES))
        )
    except ValueError:
        # Happens when a class has zero samples in validation split
        auc = 0.0

    metrics = {
        "val_loss":        avg_loss,
        "val_accuracy":    accuracy,
        "val_f1":          f1_macro,
        "val_sensitivity": macro_sensitivity,
        "val_auc":         auc,
    }

    # Add per-class sensitivity
    for i, s in enumerate(recall_per_class):
        metrics[f"val_sens_{CLASS_NAMES[i].replace(' ', '_')}"] = float(s)

    return metrics


# ─────────────────────────────────────────────
# Single Fold Training
# ─────────────────────────────────────────────

def train_fold(
    fold: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    output_dirs: Dict,
    device: torch.device,
    class_weights: torch.Tensor,
) -> Dict:
    """
    Train and evaluate a single fold.

    Args:
        fold          : Fold index (0-based)
        train_loader  : DataLoader for training split
        val_loader    : DataLoader for validation split
        config        : Training configuration dict
        output_dirs   : Dict of output directory Paths
        device        : Compute device
        class_weights : Inverse-frequency class weights tensor

    Returns:
        Dict with best validation metrics for this fold
    """
    logger = logging.getLogger(__name__)
    fold_display = fold + 1
    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold_display}/{config['n_folds']}")
    logger.info(f"{'='*60}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(
        num_classes=config["num_classes"],
        dropout_p=config["dropout_p"]
    ).to(device)

    logger.info(f"DRNet parameters: {model.count_parameters():,}")

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion = build_loss(
        loss_type=config["loss_type"],
        class_weights=class_weights.to(device),
        gamma=config["focal_gamma"],
    )

    # ── Optimizer ────────────────────────────────────────────────────────────
    if config["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

    # ── Scheduler ────────────────────────────────────────────────────────────
    # CosineAnnealingLR: smoothly decays LR from lr to lr_min over all epochs.
    # This avoids sharp LR drops (StepLR) which can destabilize training on
    # small datasets where each epoch carries high variance.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config["lr_min"],
    )

    # ── Early Stopping ───────────────────────────────────────────────────────
    early_stopper = EarlyStopping(
        patience=config["patience"],
        mode="max",        # monitoring val_f1 — higher is better
        min_delta=1e-4,
    )

    # ── Training Loop ────────────────────────────────────────────────────────
    train_losses, val_losses   = [], []
    train_accs,   val_accs     = [], []

    best_val_f1   = -1.0
    best_metrics  = {}
    best_ckpt_path = output_dirs["checkpoints"] / f"best_fold{fold_display}.pth"

    epoch_progress = tqdm(range(1, config["epochs"] + 1), 
                          desc=f"Fold {fold_display}/{config['n_folds']}", 
                          unit="epoch")
    for epoch in epoch_progress:
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_metrics["val_loss"])
        train_accs.append(train_acc)
        val_accs.append(val_metrics["val_accuracy"])

        # Update progress bar with metrics
        epoch_progress.set_postfix({
            'Loss': f"{train_loss:.3f}",
            'Acc': f"{train_acc:.3f}",
            'Val_F1': f"{val_metrics['val_f1']:.3f}",
            'Val_Acc': f"{val_metrics['val_accuracy']:.3f}"
        })

        # Log epoch summary
        logger.info(
            f"Fold {fold_display} | Epoch {epoch:03d}/{config['epochs']} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Acc: {val_metrics['val_accuracy']:.4f} | "
            f"Val F1: {val_metrics['val_f1']:.4f} | "
            f"Val Sens: {val_metrics['val_sensitivity']:.4f} | "
            f"Val AUC: {val_metrics['val_auc']:.4f}"
        )

        # Save best checkpoint (based on val_f1)
        if val_metrics["val_f1"] > best_val_f1:
            best_val_f1  = val_metrics["val_f1"]
            best_metrics = {**val_metrics, "epoch": epoch, "fold": fold_display}
            save_checkpoint(
                model, optimizer, epoch, fold_display,
                val_metrics, str(best_ckpt_path)
            )

        # Early stopping check
        if early_stopper.step(val_metrics["val_f1"]):
            logger.info(f"Early stopping at epoch {epoch} for fold {fold_display}.")
            break

    # ── Post-training: save plots ────────────────────────────────────────────
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        fold=fold_display,
        save_path=str(output_dirs["plots"] / f"curves_fold{fold_display}.png"),
    )

    logger.info(
        f"\nFold {fold_display} BEST → "
        f"Epoch: {best_metrics.get('epoch')} | "
        f"F1: {best_metrics.get('val_f1', 0):.4f} | "
        f"Sensitivity: {best_metrics.get('val_sensitivity', 0):.4f} | "
        f"AUC: {best_metrics.get('val_auc', 0):.4f}"
    )

    # Log per-class sensitivity for this fold's best checkpoint
    for i in range(NUM_CLASSES):
        key = f"val_sens_{CLASS_NAMES[i].replace(' ', '_')}"
        val = best_metrics.get(key, 0.0)
        logger.info(f"  {CLASS_NAMES[i]:20s} Sensitivity: {val:.4f}")

    return best_metrics


# ─────────────────────────────────────────────
# Main Training Entry Point
# ─────────────────────────────────────────────

def main(config: Dict) -> None:
    """
    Full K-Fold cross-validation training pipeline.

    Steps:
        1. Setup reproducibility, logging, output directories
        2. Load dataset, compute class weights
        3. Build K-Fold data loaders
        4. Train each fold independently
        5. Aggregate and report cross-fold metrics
        6. Save summary report
    """
    # ── Setup ────────────────────────────────────────────────────────────────
    set_seed(config["seed"])
    output_dirs = create_output_dirs(config["output_dir"])
    setup_logging(log_dir=str(output_dirs["logs"]))
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Diabetic Retinopathy Classification — Training")
    logger.info("=" * 60)
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    device = get_device()

    # ── Class Weights ────────────────────────────────────────────────────────
    # Compute from the full dataset (before splitting) to get true distribution
    full_ds = DRDataset(config["image_dir"], config["csv_path"])
    class_weights = full_ds.get_class_weights()
    logger.info(f"Class weights: {class_weights.numpy().round(3)}")
    logger.info(f"Total samples: {len(full_ds)}")

    # ── K-Fold DataLoaders ────────────────────────────────────────────────────
    folds = get_kfold_dataloaders(
        image_dir=config["image_dir"],
        csv_path=config["csv_path"],
        n_splits=config["n_folds"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        image_size=config["image_size"],
        seed=config["seed"],
    )

    # ── K-Fold Training Loop ─────────────────────────────────────────────────
    fold_results = []

    for fold_idx, (train_loader, val_loader) in enumerate(folds):
        set_seed(config["seed"] + fold_idx)  # Unique seed per fold for reproducibility

        fold_metrics = train_fold(
            fold=fold_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dirs=output_dirs,
            device=device,
            class_weights=class_weights,
        )
        fold_results.append(fold_metrics)

        # Save per-fold metrics
        save_metrics_json(
            fold_metrics,
            str(output_dirs["reports"] / f"metrics_fold{fold_idx + 1}.json")
        )

    # ── Cross-Fold Aggregation ────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("K-FOLD CROSS VALIDATION SUMMARY")
    logger.info("=" * 60)

    summary_metrics = {
        "val_accuracy":    [],
        "val_f1":          [],
        "val_sensitivity": [],
        "val_auc":         [],
    }

    for i, fold_metrics in enumerate(fold_results):
        logger.info(
            f"  Fold {i+1}: "
            f"Acc={fold_metrics.get('val_accuracy', 0):.4f} | "
            f"F1={fold_metrics.get('val_f1', 0):.4f} | "
            f"Sens={fold_metrics.get('val_sensitivity', 0):.4f} | "
            f"AUC={fold_metrics.get('val_auc', 0):.4f}"
        )
        for key in summary_metrics:
            summary_metrics[key].append(fold_metrics.get(key, 0.0))

    logger.info("\nMean ± Std across folds:")
    final_summary = {}
    for key, values in summary_metrics.items():
        mean_v = np.mean(values)
        std_v  = np.std(values)
        logger.info(f"  {key:20s}: {mean_v:.4f} ± {std_v:.4f}")
        final_summary[key] = {"mean": mean_v, "std": std_v, "per_fold": values}

    # ── Save Summary ─────────────────────────────────────────────────────────
    save_metrics_json(
        {"config": config, "fold_results": fold_results, "summary": final_summary},
        str(output_dirs["reports"] / "final_summary.json")
    )

    # ── Fold Comparison Plot ──────────────────────────────────────────────────
    plot_fold_comparison(
        fold_results,
        save_path=str(output_dirs["plots"] / "fold_comparison.png"),
    )

    logger.info("\nTraining complete.")
    logger.info(f"Best checkpoints saved in: {output_dirs['checkpoints']}")
    logger.info(f"Plots saved in            : {output_dirs['plots']}")
    logger.info(f"Reports saved in          : {output_dirs['reports']}")


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DRNet for Diabetic Retinopathy Classification"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file. If not provided, defaults are used."
    )
    # Allow overriding specific config keys via CLI
    parser.add_argument("--image_dir",  type=str, default=None)
    parser.add_argument("--csv_path",   type=str, default=None)
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_folds",    type=int, default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--loss_type",  type=str, default=None)
    parser.add_argument("--seed",       type=int, default=None)

    args = parser.parse_args()

    # Start with defaults
    config = DEFAULT_CONFIG.copy()

    # Load from JSON if provided
    if args.config:
        with open(args.config) as f:
            file_config = json.load(f)
        config.update(file_config)

    # CLI overrides take highest priority
    for key in ["image_dir", "csv_path", "epochs", "batch_size",
                "n_folds", "lr", "loss_type", "seed"]:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    main(config)