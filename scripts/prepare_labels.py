"""
scripts/prepare_labels.py
--------------------------
Helper script to generate labels.csv for the DRNet pipeline.

Since your images are named im001.jpg, im002.jpg, etc. (no class info
in the filename), you need to create a CSV that maps each filename to
its DR grade label.

This script provides two modes:

  Mode 1 — INTERACTIVE (default):
    Lists all .jpg files and prompts you to enter labels interactively.
    Useful for small datasets.

  Mode 2 — TEMPLATE:
    Generates a pre-filled CSV template with all filenames and empty
    label column — then you fill in the labels manually in Excel/LibreOffice.

  Mode 3 — IMPORT from APTOS/Kaggle format:
    If you have a Kaggle DR dataset CSV (with 'image' and 'diagnosis' columns),
    this converts it to the format expected by DRNet.

Label mapping:
    0 = No DR
    1 = Mild
    2 = Moderate
    3 = Severe
    4 = Proliferative DR

Usage:
    # Generate template CSV (recommended first step)
    python scripts/prepare_labels.py --mode template --image_dir data/images

    # Convert Kaggle/APTOS CSV
    python scripts/prepare_labels.py --mode kaggle \
        --kaggle_csv path/to/train.csv \
        --image_dir data/images

    # Validate an existing labels.csv
    python scripts/prepare_labels.py --mode validate \
        --csv_path data/labels.csv \
        --image_dir data/images
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from collections import Counter

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
VALID_LABELS = set(range(5))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_image_files(image_dir: str) -> list:
    """Return sorted list of image filenames in the directory."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    files = sorted([
        f.name for f in image_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])
    logger.info(f"Found {len(files)} image files in '{image_dir}'")
    return files


# ─────────────────────────────────────────────
# Mode 1: Template
# ─────────────────────────────────────────────

def generate_template(image_dir: str, output_csv: str = "data/labels.csv") -> None:
    """
    Create a CSV template with all image filenames and an empty label column.
    Open in Excel or any spreadsheet tool to fill in labels.
    """
    files = get_image_files(image_dir)

    df = pd.DataFrame({
        "filename": files,
        "label":    ["" for _ in files],  # empty — user fills in
    })

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    logger.info(f"Template CSV saved → {output_csv}")
    logger.info(
        f"  Please open '{output_csv}' and fill in the 'label' column.\n"
        f"  Label values:\n"
        + "\n".join(f"    {k} = {v}" for k, v in CLASS_NAMES.items())
    )


# ─────────────────────────────────────────────
# Mode 2: Interactive
# ─────────────────────────────────────────────

def interactive_labeling(image_dir: str, output_csv: str = "data/labels.csv") -> None:
    """
    Interactively prompt for labels for each image.
    Type the label (0-4) and press Enter.
    Press Ctrl+C to stop and save progress.
    """
    files = get_image_files(image_dir)

    # Check if partial progress exists
    rows = []
    if Path(output_csv).exists():
        existing = pd.read_csv(output_csv)
        done = set(existing["filename"].tolist())
        rows = existing.to_dict("records")
        files = [f for f in files if f not in done]
        logger.info(f"Resuming: {len(done)} already labeled, {len(files)} remaining.")

    label_str = " | ".join(f"{k}={v}" for k, v in CLASS_NAMES.items())
    print(f"\nLabels: {label_str}")
    print("Press Ctrl+C to save and exit at any time.\n")

    try:
        for i, fname in enumerate(files):
            while True:
                try:
                    val = input(f"[{i+1}/{len(files)}] {fname} → label (0-4): ").strip()
                    label = int(val)
                    if label not in VALID_LABELS:
                        print(f"  Invalid label '{val}'. Must be 0-4.")
                        continue
                    rows.append({"filename": fname, "label": label})
                    break
                except ValueError:
                    print(f"  Please enter an integer (0-4).")

    except KeyboardInterrupt:
        print("\nInterrupted — saving progress.")

    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(df)} labels → {output_csv}")


# ─────────────────────────────────────────────
# Mode 3: Kaggle/APTOS Import
# ─────────────────────────────────────────────

def import_from_kaggle(
    kaggle_csv: str,
    image_dir: str,
    output_csv: str = "data/labels.csv",
) -> None:
    """
    Convert a Kaggle DR dataset CSV to DRNet format.

    Kaggle format: columns 'image' (or 'id_code') and 'diagnosis' (0-4)
    DRNet format : columns 'filename' and 'label'
    """
    df_kaggle = pd.read_csv(kaggle_csv)
    logger.info(f"Kaggle CSV loaded: {len(df_kaggle)} rows, columns: {list(df_kaggle.columns)}")

    # Detect filename column
    name_col = None
    for candidate in ["image", "id_code", "filename", "Image name"]:
        if candidate in df_kaggle.columns:
            name_col = candidate
            break
    if name_col is None:
        raise ValueError(
            f"Could not find image filename column. Available: {list(df_kaggle.columns)}"
        )

    # Detect label column
    label_col = None
    for candidate in ["diagnosis", "label", "DR_grade", "level"]:
        if candidate in df_kaggle.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError(
            f"Could not find label column. Available: {list(df_kaggle.columns)}"
        )

    logger.info(f"Using: filename_col='{name_col}', label_col='{label_col}'")

    # Build DRNet CSV
    existing_files = set(get_image_files(image_dir))
    rows = []
    skipped = 0

    for _, row in df_kaggle.iterrows():
        fname = str(row[name_col])
        if not fname.endswith((".jpg", ".jpeg", ".png")):
            fname += ".jpg"  # APTOS uses no extension in CSV

        label = int(row[label_col])

        if fname not in existing_files:
            skipped += 1
            continue

        if label not in VALID_LABELS:
            logger.warning(f"Skipping '{fname}': invalid label {label}")
            skipped += 1
            continue

        rows.append({"filename": fname, "label": label})

    if skipped > 0:
        logger.warning(f"Skipped {skipped} rows (missing files or invalid labels)")

    df_out = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    logger.info(f"Converted {len(df_out)} samples → {output_csv}")


# ─────────────────────────────────────────────
# Mode 4: Validate
# ─────────────────────────────────────────────

def validate_csv(csv_path: str, image_dir: str) -> None:
    """
    Validate an existing labels.csv:
      - Check all files exist on disk
      - Check labels are in valid range
      - Report class distribution
      - Warn about imbalances
    """
    if not Path(csv_path).exists():
        logger.error(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    existing_files = set(get_image_files(image_dir))

    print(f"\n{'='*50}")
    print(f"Validating: {csv_path}")
    print(f"{'='*50}")
    print(f"Total rows      : {len(df)}")

    # Check columns
    missing_cols = {"filename", "label"} - set(df.columns)
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        return

    # Check missing files
    missing_files = [r["filename"] for _, r in df.iterrows()
                     if r["filename"] not in existing_files]
    print(f"Missing files   : {len(missing_files)}")
    if missing_files:
        for f in missing_files[:5]:
            print(f"  {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")

    # Check empty labels
    empty = df[df["label"].isna() | (df["label"].astype(str).str.strip() == "")]
    print(f"Empty labels    : {len(empty)}")

    # Check invalid labels
    valid_df = df.dropna(subset=["label"])
    valid_df = valid_df[valid_df["label"].astype(str).str.strip() != ""]
    try:
        valid_df = valid_df.copy()
        valid_df["label"] = valid_df["label"].astype(int)
        invalid = valid_df[~valid_df["label"].isin(VALID_LABELS)]
        print(f"Invalid labels  : {len(invalid)}")
    except ValueError:
        logger.error("Labels column contains non-integer values.")
        return

    # Class distribution
    counts = Counter(valid_df["label"].tolist())
    print(f"\nClass Distribution:")
    total = sum(counts.values())
    for label, name in CLASS_NAMES.items():
        c = counts.get(label, 0)
        bar = "█" * int(c / max(total, 1) * 30)
        pct = c / max(total, 1) * 100
        print(f"  {label} {name:<18} {c:>4} ({pct:5.1f}%)  {bar}")

    # Balance check
    max_c = max(counts.values()) if counts else 0
    min_c = min(counts.values()) if counts else 0
    if max_c > 0 and min_c / max_c < 0.5:
        print(
            f"\n⚠ Class imbalance detected (ratio {min_c}/{max_c} = {min_c/max_c:.2f}). "
            "Consider class weighting (already supported in DRNet)."
        )
    else:
        print("\n✓ Classes are reasonably balanced.")

    print(f"\n✓ Validation complete. Labels ready: {total - len(missing_files) - len(empty)}")
    print(f"{'='*50}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare labels.csv for DRNet training"
    )
    parser.add_argument(
        "--mode",
        choices=["template", "interactive", "kaggle", "validate"],
        default="template",
        help=(
            "template   : Generate blank CSV template to fill manually\n"
            "interactive: Enter labels one-by-one in terminal\n"
            "kaggle     : Convert Kaggle/APTOS CSV format\n"
            "validate   : Validate existing labels.csv"
        )
    )
    parser.add_argument("--image_dir",   default="data/images",  help="Directory of images")
    parser.add_argument("--csv_path",    default="data/labels.csv", help="Output/input CSV path")
    parser.add_argument("--kaggle_csv",  default=None, help="Path to Kaggle-format CSV (mode=kaggle)")

    args = parser.parse_args()

    if args.mode == "template":
        generate_template(args.image_dir, args.csv_path)

    elif args.mode == "interactive":
        interactive_labeling(args.image_dir, args.csv_path)

    elif args.mode == "kaggle":
        if not args.kaggle_csv:
            logger.error("--kaggle_csv required for mode=kaggle")
            sys.exit(1)
        import_from_kaggle(args.kaggle_csv, args.image_dir, args.csv_path)

    elif args.mode == "validate":
        validate_csv(args.csv_path, args.image_dir)