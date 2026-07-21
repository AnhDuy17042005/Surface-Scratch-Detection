"""
    Evaluate a binary U-Net checkpoint and export segmentation metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


"""Project Root"""
PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

"""Config Imports"""
from configs.data import IMAGE_EXTENSIONS, SCRATCH_TRAIN_DATASET
from configs.path import METRICS_DIR
from configs.unet import (
    UNET_DEVICE,
    UNET_INFERENCE_PATCH_BATCH_SIZE,
    UNET_INFERENCE_TILE_OVERLAP,
    UNET_INFERENCE_TILE_SIZE,
    UNET_MODEL,
    UNET_THRESHOLD,
)


"""U-Net Inference Imports"""
from src.unet.inference import get_device, load_model, preprocess
from src.unet.sliding_windows import predict_sliding_window_probability


"""Default Output Directory"""
DEFAULT_OUTPUT_DIR = METRICS_DIR


def parse_args() -> argparse.Namespace:
    """
        Parse command line arguments for U-Net evaluation.

        Args:
            --model      : path to U-Net checkpoint
            --data       : path to scratch tile dataset
            --split      : train / valid / test split
            --img-size   : optional inference image size
            --mode       : full / sliding inference mode
            --patch-size : sliding-window patch size
            --overlap    : sliding-window overlap ratio
            --threshold  : probability threshold for binary mask
            --device     : cpu / cuda / auto
            --output-dir : directory to save metrics
            --max-images : optional image limit for quick test
    """

    parser = argparse.ArgumentParser(
        description="Evaluate U-Net scratch segmentation on a dataset split."
    )

    """Input Arguments"""
    parser.add_argument("--model", type=Path, default=UNET_MODEL)
    parser.add_argument("--data", type=Path, default=SCRATCH_TRAIN_DATASET)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")

    """Inference Arguments"""
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=UNET_THRESHOLD)
    parser.add_argument("--device", type=str, default=UNET_DEVICE)
    parser.add_argument(
        "--mode",
        choices=("sliding", "full"),
        default="sliding",
        help="Match inference.py: sliding by default, full for resized baseline.",
    )
    parser.add_argument("--patch-size", type=int, default=UNET_INFERENCE_TILE_SIZE)
    parser.add_argument("--overlap", type=float, default=UNET_INFERENCE_TILE_OVERLAP)
    parser.add_argument(
        "--patch-batch-size",
        type=int,
        default=UNET_INFERENCE_PATCH_BATCH_SIZE,
    )

    """Output Arguments"""
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    """Smoke Test Argument"""
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for a quick smoke test.",
    )

    args = parser.parse_args()

    """Validate Image Size"""
    if args.img_size is not None and args.img_size < 32:
        parser.error("--img-size must be at least 32")

    """Validate Threshold"""
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    """Validate Sliding-window Arguments"""
    if args.patch_size < 1:
        parser.error("--patch-size must be positive")

    if not 0.0 <= args.overlap < 1.0:
        parser.error("--overlap must be in [0, 1)")

    if args.patch_batch_size < 1:
        parser.error("--patch-batch-size must be positive")

    """Validate Max Images"""
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be at least 1")

    return args


def find_image_mask_pairs(data_root: Path, split: str) -> list[tuple[Path, Path]]:
    """
        Find image-mask pairs from a dataset split.

        Dataset structure:
            data_root/
            ├── train/
            │   ├── images/
            │   └── masks/
            ├── valid/
            │   ├── images/
            │   └── masks/
            └── test/
                ├── images/
                └── masks/

        Mask rule:
            image name : abc.jpg
            mask name  : abc.png
    """

    """Dataset Directories"""
    image_dir = data_root / split / "images"
    mask_dir = data_root / split / "masks"

    """Check Image Directory"""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    """Check Mask Directory"""
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    """Collect Images"""
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    """Check Empty Dataset"""
    if not images:
        raise ValueError(f"No evaluation images found in: {image_dir}")

    pairs: list[tuple[Path, Path]] = []
    missing_masks: list[Path] = []

    """Match Each Image With Its Mask"""
    for image_path in images:
        mask_path = mask_dir / f"{image_path.stem}.png"

        if mask_path.is_file():
            pairs.append((image_path, mask_path))
        else:
            missing_masks.append(mask_path)

    """Report Missing Masks"""
    if missing_masks:
        examples = ", ".join(str(path) for path in missing_masks[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_masks)} masks, for example: {examples}"
        )

    return pairs


def confusion_counts(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    """
        Compute pixel-level confusion matrix.

        Args:
            prediction: predicted binary mask
            target    : ground truth binary mask

        Returns:
            tp: true positive pixels
            fp: false positive pixels
            fn: false negative pixels
            tn: true negative pixels
    """

    """Convert Masks To Boolean"""
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    """Compute Confusion Counts"""
    return {
        "tp": int(np.logical_and(prediction, target).sum()),
        "fp": int(np.logical_and(prediction, np.logical_not(target)).sum()),
        "fn": int(np.logical_and(np.logical_not(prediction), target).sum()),
        "tn": int(
            np.logical_and(
                np.logical_not(prediction),
                np.logical_not(target),
            ).sum()
        ),
    }


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    """
        Compute segmentation metrics from TP, FP, FN, TN.

        Metrics:
            precision      : TP / (TP + FP)
            recall         : TP / (TP + FN)
            specificity    : TN / (TN + FP)
            iou            : TP / (TP + FP + FN)
            dice           : 2TP / (2TP + FP + FN)
            pixel_accuracy : (TP + TN) / total
    """

    """Read Confusion Counts"""
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    """Metric Denominators"""
    predicted_positive = tp + fp
    actual_positive = tp + fn
    union = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn
    total = tp + fp + fn + tn

    """Compute Metrics"""
    precision = tp / predicted_positive if predicted_positive else float(fn == 0)
    recall = tp / actual_positive if actual_positive else 1.0
    specificity = tn / (tn + fp) if (tn + fp) else 1.0
    iou = tp / union if union else 1.0
    dice = 2 * tp / dice_denominator if dice_denominator else 1.0
    accuracy = (tp + tn) / total if total else 1.0

    """Return Counts And Metrics"""
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "iou": iou,
        "dice": dice,
        "pixel_accuracy": accuracy,
    }


def add_counts(total: dict[str, int], current: dict[str, int]) -> None:
    """
        Add current image confusion counts to total counts.

        Used for global metrics.
    """

    """Accumulate TP, FP, FN, TN"""
    for key in total:
        total[key] += current[key]


def mean_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    """
        Compute mean per-image metrics.

        Args:
            rows  : list of per-image metric rows
            prefix: raw / post_processed
    """

    """Metric Names"""
    names = (
        "precision",
        "recall",
        "specificity",
        "iou",
        "dice",
        "pixel_accuracy",
    )

    """Average Each Metric"""
    return {
        name: float(np.mean([row[f"{prefix}_{name}"] for row in rows]))
        for name in names
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """
        Write per-image metrics to CSV file.

        Args:
            path: output CSV path
            rows: per-image metric rows
    """

    """Write CSV"""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predict_probability(
    model: torch.nn.Module,
    image: np.ndarray,
    img_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    """
        Predict probability map using the same modes exposed by inference.py.
    """

    """Use sliding-window inference for patch-trained production evaluation"""
    if args.mode == "sliding":
        return predict_sliding_window_probability(
            model=model,
            image=image,
            patch_size=args.patch_size,
            overlap=args.overlap,
            batch_size=args.patch_batch_size,
            device=device,
        )

    """Use resized full-image inference as a fast baseline"""
    tensor = preprocess(image, img_size, device)

    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

    return cv2.resize(
        probability,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


def main() -> None:
    """
        Evaluate U-Net model on one dataset split.

        Pipeline:
            1. Parse arguments
            2. Load image-mask pairs
            3. Load U-Net checkpoint
            4. Predict binary mask
            5. Compute raw prediction metrics
            6. Compute metrics
            7. Save JSON and CSV reports
    """

    """Parse Arguments"""
    args = parse_args()

    """Resolve Paths"""
    model_path = args.model.resolve()
    data_root = args.data.resolve()
    output_dir = args.output_dir.resolve()

    """Check Model Checkpoint"""
    if not model_path.is_file():
        raise FileNotFoundError(f"U-Net checkpoint not found: {model_path}")

    """Find Evaluation Pairs"""
    pairs = find_image_mask_pairs(data_root, args.split)

    """Limit Images For Smoke Test"""
    if args.max_images is not None:
        pairs = pairs[: args.max_images]

    """Load Device And Model"""
    device = get_device(args.device)
    model, img_size = load_model(model_path, device, args.img_size)

    """Create Output Directory"""
    output_dir.mkdir(parents=True, exist_ok=True)

    """Global Raw Counts"""
    raw_total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    """Global Post-processed Counts"""
    processed_total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    """Per-image Metric Rows"""
    per_image: list[dict[str, Any]] = []

    """Start Timer"""
    started = time.perf_counter()

    """Evaluate Each Image"""
    for image_path, mask_path in pairs:
        """Read Image And Mask"""
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        target_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        """Check Image"""
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        """Check Mask"""
        if target_image is None:
            raise ValueError(f"Cannot read mask: {mask_path}")

        """Check Image-Mask Size"""
        if image.shape[:2] != target_image.shape[:2]:
            raise ValueError(
                f"Image and mask size differ for {image_path.name}: "
                f"{image.shape[:2]} != {target_image.shape[:2]}"
            )

        """Predict Probability Map"""
        probability = predict_probability(
            model=model,
            image=image,
            img_size=img_size,
            args=args,
            device=device,
        )

        """Prepare Target Mask"""
        target = target_image > 0

        """Raw Prediction"""
        raw_prediction = probability >= args.threshold

        """Compute Raw Metrics"""
        raw_counts = confusion_counts(raw_prediction, target)
        raw_metrics = metrics_from_counts(raw_counts)

        """Update Global Counts"""
        add_counts(raw_total, raw_counts)

        """Post-processing is disabled for the first full-image baseline"""
        processed_prediction = raw_prediction

        """Compute Processed Metrics For Report Compatibility"""
        processed_counts = confusion_counts(processed_prediction, target)
        processed_metrics = metrics_from_counts(processed_counts)

        """Update Global Processed Counts"""
        add_counts(processed_total, processed_counts)

        """Base Row Information"""
        row: dict[str, Any] = {
            "image": image_path.name,
            "width": image.shape[1],
            "height": image.shape[0],
            "positive_pixels": int(target.sum()),
        }

        """Add Raw Metrics To Row"""
        row.update({f"raw_{key}": value for key, value in raw_metrics.items()})

        """Add Processed Metrics To Row"""
        row.update({
            f"post_processed_{key}": value
            for key, value in processed_metrics.items()
        })

        """Save Per-image Row"""
        per_image.append(row)

    """Elapsed Time"""
    elapsed_seconds = time.perf_counter() - started

    """Build Evaluation Report"""
    report = {
        "model_type": "U-Net binary segmentation",
        "model": str(model_path),
        "data": str(data_root),
        "split": args.split,
        "image_count": len(per_image),

        # Evaluation Settings
        "settings": {
            "img_size": img_size,
            "mode": args.mode,
            "threshold": args.threshold,
            "device": str(device),
            "patch_size": args.patch_size if args.mode == "sliding" else None,
            "overlap": args.overlap if args.mode == "sliding" else None,
            "patch_batch_size": (
                args.patch_batch_size if args.mode == "sliding" else None
            ),
            "post_processing": False,
            "max_images": args.max_images,
        },

        # Global Metrics
        "global": {
            "raw": metrics_from_counts(raw_total),
            "post_processed": metrics_from_counts(processed_total),
        },

        # Mean Per-image Metrics
        "mean_per_image": {
            "raw": mean_metrics(per_image, "raw"),
            "post_processed": mean_metrics(per_image, "post_processed"),
        },

        # Speed Metrics
        "elapsed_seconds": round(elapsed_seconds, 3),
        "milliseconds_per_image": round(
            elapsed_seconds * 1000 / len(per_image),
            3,
        ),
    }

    """Save JSON Report"""
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    """Save CSV Report"""
    write_csv(output_dir / "per_image_metrics.csv", per_image)

    """Print Summary"""
    processed = report["global"]["post_processed"]

    print("\nU-Net evaluation complete")
    print(f"Model: {model_path}")
    print(f"Split: {args.split} ({len(per_image)} images)")
    print(f"Mode: {args.mode}")
    if args.mode == "sliding":
        print(
            f"Patch size: {args.patch_size} | "
            f"Overlap: {args.overlap} | "
            f"Patch batch size: {args.patch_batch_size}"
        )
    print(f"Metrics: {metrics_path}")
    print(
        "Post-processed: "
        f"IoU={processed['iou']:.4f} Dice={processed['dice']:.4f} "
        f"P={processed['precision']:.4f} R={processed['recall']:.4f}"
    )


"""Run Main"""
if __name__ == "__main__":
    main()
