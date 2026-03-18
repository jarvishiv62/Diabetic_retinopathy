"""
utils.py
--------
Utility functions shared across the entire pipeline:

  - Reproducibility (seed setting, deterministic mode)
  - Logging setup
  - Output directory management
  - Training/validation curve plotting
  - Checkpoint saving and loading
  - Per-class metric display
"""

import os
import random
import logging
import json
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt

import torch


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Covers: Python random, NumPy, PyTorch (CPU + CUDA), cuDNN behavior.

    Note: deterministic=True may reduce throughput slightly but ensures
    bit-exact reproducibility across runs — critical for research comparisons.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU safety
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def setup_logging(
    log_dir: str = "outputs/logs",
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure root logger to write to both console and a timestamped log file.

    Args:
        log_dir   : Directory to save log files
        log_level : Minimum logging level (default: INFO)

    Returns:
        Configured root logger
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"train_{timestamp}.log"

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    root_logger.info(f"Logging initialized. Log file: {log_file}")
    return root_logger


# ─────────────────────────────────────────────
# Output Directory Setup
# ─────────────────────────────────────────────

def create_output_dirs(base_dir: str = "outputs") -> Dict[str, Path]:
    """
    Create standardized output directory structure.

    Structure:
        outputs/
          checkpoints/  ← saved model weights
          plots/        ← training curves, confusion matrix
          logs/         ← text log files
          reports/      ← JSON metric reports
          gradcam/      ← Grad-CAM visualization images

    Returns:
        Dictionary mapping directory names to Path objects
    """
    dirs = {
        "checkpoints": Path(base_dir) / "checkpoints",
        "plots":       Path(base_dir) / "plots",
        "logs":        Path(base_dir) / "logs",
        "reports":     Path(base_dir) / "reports",
        "gradcam":     Path(base_dir) / "gradcam",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ─────────────────────────────────────────────
# Device Utility
# ─────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Return the best available device: CUDA > CPU.
    Logs device name for reproducibility records.
    """
    logger = logging.getLogger(__name__)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available. Using CPU — training will be slow.")
    return device


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage (RAM and GPU if available).
    
    Returns:
        Dictionary with memory usage in GB
    """
    memory_info = {}
    
    # System RAM
    memory = psutil.virtual_memory()
    memory_info["ram_used_gb"] = memory.used / 1024**3
    memory_info["ram_total_gb"] = memory.total / 1024**3
    memory_info["ram_percent"] = memory.percent
    
    # GPU memory if available
    if torch.cuda.is_available():
        memory_info["gpu_used_gb"] = torch.cuda.memory_allocated() / 1024**3
        memory_info["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        memory_info["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory_info["gpu_percent"] = (memory_info["gpu_used_gb"] / memory_info["gpu_total_gb"]) * 100
    
    return memory_info


def log_memory_usage(prefix: str = "") -> None:
    """Log current memory usage with optional prefix."""
    memory = get_memory_usage()
    logger = logging.getLogger(__name__)
    
    msg = f"{prefix}Memory: RAM {memory['ram_used_gb']:.1f}/{memory['ram_total_gb']:.1f}GB ({memory['ram_percent']:.1f}%)"
    
    if "gpu_used_gb" in memory:
        msg += f" | GPU {memory['gpu_used_gb']:.1f}/{memory['gpu_total_gb']:.1f}GB ({memory['gpu_percent']:.1f}%)"
    
    logger.info(msg)


# ─────────────────────────────────────────────
# Checkpoint Management
# ─────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    fold: int,
    metrics: Dict,
    save_path: str,
) -> None:
    """
    Save model checkpoint with full metadata.

    Saves both model weights and optimizer state so training can be
    resumed exactly. Metrics are embedded in the checkpoint for quick
    inspection without re-running evaluation.
    """
    checkpoint = {
        "epoch":           epoch,
        "fold":            fold,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics":         metrics,
        "timestamp":       datetime.now().isoformat(),
    }
    torch.save(checkpoint, save_path)
    logging.getLogger(__name__).info(
        f"Checkpoint saved → {save_path} "
        f"(Fold {fold}, Epoch {epoch}, Val F1: {metrics.get('val_f1', 'N/A'):.4f})"
    )


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Load model weights (and optionally optimizer state) from a checkpoint.

    Args:
        model           : Model instance to load weights into
        checkpoint_path : Path to the saved .pth file
        optimizer       : If provided, also loads optimizer state
        device          : Device to map tensors to

    Returns:
        The full checkpoint dictionary (contains epoch, metrics, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    logger = logging.getLogger(__name__)
    logger.info(
        f"Checkpoint loaded from {checkpoint_path} "
        f"(Fold {checkpoint.get('fold')}, Epoch {checkpoint.get('epoch')})"
    )
    return checkpoint


# ─────────────────────────────────────────────
# Training Curve Plots
# ─────────────────────────────────────────────

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    fold: int,
    save_path: str,
) -> None:
    """
    Plot and save training vs. validation loss and accuracy curves.

    These curves are the primary diagnostic tool for:
      - Detecting overfitting (val loss diverges from train loss)
      - Detecting underfitting (both losses remain high)
      - Verifying learning rate schedule effectiveness

    Args:
        train_losses : List of per-epoch training losses
        val_losses   : List of per-epoch validation losses
        train_accs   : List of per-epoch training accuracies
        val_accs     : List of per-epoch validation accuracies
        fold         : Current fold number (for title)
        save_path    : Path to save the figure
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves — Fold {fold}", fontsize=14, fontweight="bold")

    # ── Loss Plot ────────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, val_losses,   "r-o", markersize=3, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Annotate minimum validation loss
    min_val_loss = min(val_losses)
    min_epoch = val_losses.index(min_val_loss) + 1
    ax1.axvline(x=min_epoch, color="red", linestyle="--", alpha=0.4,
                label=f"Best Val Loss @ epoch {min_epoch}")
    ax1.annotate(f"{min_val_loss:.4f}", xy=(min_epoch, min_val_loss),
                 xytext=(min_epoch + 1, min_val_loss),
                 fontsize=8, color="red")

    # ── Accuracy Plot ────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(epochs, train_accs, "b-o", markersize=3, label="Train Acc")
    ax2.plot(epochs, val_accs,   "r-o", markersize=3, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.getLogger(__name__).info(f"Training curves saved → {save_path}")


def plot_fold_comparison(
    fold_metrics: List[Dict],
    save_path: str,
) -> None:
    """
    Bar chart comparing key metrics across all K folds.

    Useful for assessing fold-to-fold variance — high variance indicates
    the model is sensitive to the train/val split (common with small datasets).

    Args:
        fold_metrics : List of metric dicts, one per fold
        save_path    : Path to save the figure
    """
    n_folds = len(fold_metrics)
    folds = [f"Fold {i+1}" for i in range(n_folds)]
    metrics_to_plot = ["val_accuracy", "val_sensitivity", "val_f1", "val_auc"]
    labels = ["Accuracy", "Sensitivity", "F1 Score", "AUC"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(max(10, n_folds * 2), 5))

    x = np.arange(n_folds)
    bar_width = 0.2

    for i, (metric, label, color) in enumerate(zip(metrics_to_plot, labels, colors)):
        values = [fm.get(metric, 0) for fm in fold_metrics]
        offset = (i - len(metrics_to_plot) / 2) * bar_width + bar_width / 2
        ax.bar(x + offset, values, bar_width, label=label, color=color, alpha=0.85)

    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("K-Fold Cross Validation — Metric Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.4, label="0.8 reference")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.getLogger(__name__).info(f"Fold comparison plot saved → {save_path}")


# ─────────────────────────────────────────────
# Metric Logging
# ─────────────────────────────────────────────

def log_metrics(metrics: Dict, prefix: str = "", logger_name: str = __name__) -> None:
    """
    Log all metrics in a formatted, readable table.

    Args:
        metrics     : Dictionary of metric_name → value
        prefix      : String prefix for log line (e.g., "Fold 1 | Epoch 10")
        logger_name : Logger to use
    """
    logger = logging.getLogger(logger_name)
    parts = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
             for k, v in metrics.items()]
    logger.info(f"{prefix} | {' | '.join(parts)}")


def save_metrics_json(metrics: Dict, save_path: str) -> None:
    """Save metrics dictionary to a JSON file for later analysis."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logging.getLogger(__name__).info(f"Metrics saved → {save_path}")


# ─────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────

class EarlyStopping:
    """
    Early stopping monitor to halt training when validation metric stops improving.

    Monitors a metric (e.g., val_loss or val_f1) and stops training if it
    does not improve for `patience` consecutive epochs. This prevents overfitting
    on the small dataset by avoiding unnecessary training epochs.

    Args:
        patience  : Number of epochs to wait without improvement before stopping
        mode      : "min" (lower is better, e.g., loss) or
                    "max" (higher is better, e.g., F1, AUC)
        min_delta : Minimum change to count as improvement
    """

    def __init__(self, patience: int = 15, mode: str = "min", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.should_stop = False
        self._logger = logging.getLogger(__name__)

    def step(self, value: float) -> bool:
        """
        Update the monitor with the latest metric value.

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = value
            return False

        improved = (
            (self.mode == "min" and value < self.best_value - self.min_delta) or
            (self.mode == "max" and value > self.best_value + self.min_delta)
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            self._logger.info(
                f"EarlyStopping: no improvement for {self.counter}/{self.patience} epochs "
                f"(best={self.best_value:.4f}, current={value:.4f})"
            )
            if self.counter >= self.patience:
                self._logger.info(
                    f"EarlyStopping triggered after {self.patience} epochs without improvement."
                )
                self.should_stop = True

        return self.should_stop

    def reset(self) -> None:
        """Reset state (call between folds)."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False