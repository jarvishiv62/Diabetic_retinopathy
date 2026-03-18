"""
dataset.py
----------
Custom PyTorch Dataset for Diabetic Retinopathy Classification.

Responsibilities:
  - Load images and labels from a CSV file
  - Apply training augmentation or validation transforms
  - Support K-Fold cross-validation splitting
  - Handle corrupted/unreadable images gracefully

Design rationale:
  - CSV-based label loading is flexible and reproducible
  - Augmentation is applied only during training to prevent data leakage
  - Separate transform pipelines for train/val ensure fair evaluation
  - K-Fold splitting is stratified to preserve class balance in each fold
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

# Configure module-level logger
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# DR severity class mapping (integer → human-readable)
CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

NUM_CLASSES = 5
IMAGE_SIZE = 256  # Input resolution (256×256)

# ImageNet statistics used for normalization.
# Retinal fundus images benefit from ImageNet normalization when using
# pretrained backbones; for training from scratch it still stabilizes gradients.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
# Transform Pipelines
# ─────────────────────────────────────────────

def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Augmentation pipeline for training.

    Augmentations chosen for medical fundus images:
      - RandomHorizontalFlip: retinas are anatomically symmetric
      - RandomRotation(±20°): camera angle variation
      - ColorJitter: simulates different imaging devices / lighting conditions
      - RandomAffine (scale + translate): zoom and slight displacement
      - Normalize: zero-mean, unit-variance per channel

    We intentionally avoid aggressive crops or elastic deformations that could
    destroy subtle lesion patterns (microaneurysms, exudates) critical for grading.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Deterministic pipeline for validation/test — no augmentation.
    Only resize and normalize to ensure fair, reproducible evaluation.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────
# Dataset Class
# ─────────────────────────────────────────────

class DRDataset(Dataset):
    """
    Diabetic Retinopathy Dataset.

    Expects:
      - image_dir : directory containing all .jpg images
      - csv_path  : CSV file with columns ['filename', 'label']
                    label ∈ {0, 1, 2, 3, 4}
      - transform : torchvision transform pipeline (train or val)

    Handles:
      - Corrupted/unreadable images → returns a zero-tensor with label -1
        (these are filtered out during DataLoader collation)
    """

    def __init__(
        self,
        image_dir: str,
        csv_path: str,
        transform: transforms.Compose = None,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform or get_val_transforms()

        # ── Load and validate CSV ────────────────────────────────────────────
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"labels.csv not found at: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Validate required columns
        required_cols = {"filename", "label"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(
                f"labels.csv must contain columns {required_cols}. "
                f"Found: {list(self.df.columns)}"
            )

        # Validate label range
        invalid = self.df[~self.df["label"].isin(range(NUM_CLASSES))]
        if len(invalid) > 0:
            logger.warning(
                f"{len(invalid)} rows have invalid labels and will be skipped:\n"
                f"{invalid}"
            )
            self.df = self.df[self.df["label"].isin(range(NUM_CLASSES))].reset_index(drop=True)

        # Check all files exist; warn about missing ones
        missing = [
            row["filename"]
            for _, row in self.df.iterrows()
            if not (self.image_dir / row["filename"]).exists()
        ]
        if missing:
            logger.warning(
                f"{len(missing)} image file(s) listed in CSV not found on disk. "
                f"They will return empty tensors. First 5: {missing[:5]}"
            )

        self.labels = self.df["label"].values.astype(np.int64)
        logger.info(f"Dataset loaded: {len(self.df)} samples from '{image_dir}'")

    # ── Dunder Methods ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row["filename"]
        label = int(row["label"])

        # ── Graceful handling for unreadable images ──────────────────────────
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            logger.warning(f"Could not read image '{img_path}': {e}. Returning zero tensor.")
            # Return zero image with sentinel label -1 so collate_fn can filter it
            dummy = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            return dummy, torch.tensor(-1, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    # ── Utility ──────────────────────────────────────────────────────────────

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for weighted loss functions.

        For a dataset of N samples across C classes:
            weight_c = N / (C * count_c)

        Higher weight → rarer class → penalized more when misclassified.
        Critical for DR: Severe and Proliferative classes must not be ignored.
        """
        counts = np.bincount(self.labels, minlength=NUM_CLASSES).astype(float)
        counts = np.maximum(counts, 1)  # avoid division by zero
        weights = len(self.labels) / (NUM_CLASSES * counts)
        return torch.tensor(weights, dtype=torch.float32)

    def get_labels(self) -> np.ndarray:
        """Return all labels as numpy array (needed for stratified k-fold)."""
        return self.labels


# ─────────────────────────────────────────────
# Collate Function
# ─────────────────────────────────────────────

def safe_collate(batch):
    """
    Custom collate function that filters out corrupted samples
    (those with label == -1) before forming a batch.

    This prevents a single bad image from crashing an entire training epoch.
    """
    # Filter out samples where label is -1 (corrupted images)
    batch = [(img, lbl) for img, lbl in batch if lbl.item() != -1]

    if len(batch) == 0:
        # Edge case: entire batch was corrupted — return None signals to skip
        return None, None

    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return images, labels


# ─────────────────────────────────────────────
# K-Fold DataLoader Factory
# ─────────────────────────────────────────────

def get_kfold_dataloaders(
    image_dir: str,
    csv_path: str,
    n_splits: int = 5,
    batch_size: int = 16,
    num_workers: int = 2,
    image_size: int = IMAGE_SIZE,
    seed: int = 42,
) -> list:
    """
    Build K stratified folds, each yielding (train_loader, val_loader).

    Why K-Fold?
      With only 400 images, a single train/val split wastes precious data.
      K-Fold ensures every sample is used for validation exactly once,
      giving a more reliable estimate of generalization performance.

    Why Stratified?
      Each fold maintains the same class proportions as the full dataset,
      preventing any fold from having zero samples of a rare class.

    Returns:
        List of (train_loader, val_loader) tuples, one per fold.
    """
    # Full dataset with val transforms initially (transforms assigned per-fold below)
    full_dataset = DRDataset(image_dir, csv_path, transform=None)
    labels = full_dataset.get_labels()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logger.info(
            f"Fold {fold_idx + 1}/{n_splits}: "
            f"{len(train_indices)} train | {len(val_indices)} val"
        )

        # Create separate dataset instances so transforms don't cross-contaminate
        train_dataset = DRDataset(image_dir, csv_path, transform=get_train_transforms(image_size))
        val_dataset   = DRDataset(image_dir, csv_path, transform=get_val_transforms(image_size))

        train_subset = Subset(train_dataset, train_indices)
        val_subset   = Subset(val_dataset,   val_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=safe_collate,
            pin_memory=True,
            drop_last=True,  # avoid batch-norm issues with size-1 batches
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=safe_collate,
            pin_memory=True,
        )

        folds.append((train_loader, val_loader))

    return folds


# ─────────────────────────────────────────────
# Quick Sanity Check (run as __main__)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    # Usage: python dataset.py <image_dir> <csv_path>
    image_dir = sys.argv[1] if len(sys.argv) > 1 else "data/images"
    csv_path  = sys.argv[2] if len(sys.argv) > 2 else "data/labels.csv"

    ds = DRDataset(image_dir, csv_path, transform=get_train_transforms())
    img, lbl = ds[0]
    print(f"Sample — image shape: {img.shape}, label: {lbl.item()} ({CLASS_NAMES[lbl.item()]})")
    print(f"Class weights: {ds.get_class_weights()}")

    folds = get_kfold_dataloaders(image_dir, csv_path, n_splits=5, batch_size=8)
    train_loader, val_loader = folds[0]
    imgs, lbls = next(iter(train_loader))
    print(f"Batch — images: {imgs.shape}, labels: {lbls}")