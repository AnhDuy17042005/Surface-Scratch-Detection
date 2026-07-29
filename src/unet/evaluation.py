"""
    Evaluate a binary U-Net checkpoint and export segmentation metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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

"""Matplotlib cache directory for Linux/server environments."""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
from src.unet.inference import get_device, load_model, make_overlay, preprocess
from src.unet.sliding_windows import predict_sliding_window_probability


DEFAULT_OUTPUT_DIR = METRICS_DIR
METRIC_VERSIONS = (1, 2, 3, 4, 5)
VERSION_PATTERN = re.compile(r"(?:unet_v|train_v|train_ver|version_)([1-5])\b")


def parse_args() -> argparse.Namespace:
    """
        Parse command line arguments for U-Net evaluation v2.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate U-Net scratch segmentation and save metric plots."
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
    parser.add_argument(
        "--model-version",
        type=int,
        choices=METRIC_VERSIONS,
        default=None,
        help=(
            "Metric output version. If omitted, infer from model path such as "
            "models/unet/unet_v5/best.pth."
        ),
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help=(
            "Optional training history JSON. If omitted, evaluation looks for "
            "results.json next to the checkpoint."
        ),
    )
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save side-by-side image / GT / prediction / overlay examples.",
    )
    parser.add_argument("--sample-count", type=int, default=20)

    """Threshold Curve Arguments"""
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)

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

    """Validate Plot Arguments"""
    if args.sample_count < 0:
        parser.error("--sample-count cannot be negative")
    if not 0.0 <= args.threshold_min <= 1.0:
        parser.error("--threshold-min must be between 0 and 1")
    if not 0.0 <= args.threshold_max <= 1.0:
        parser.error("--threshold-max must be between 0 and 1")
    if args.threshold_min > args.threshold_max:
        parser.error("--threshold-min must be <= --threshold-max")
    if args.threshold_step <= 0.0:
        parser.error("--threshold-step must be positive")

    """Validate Max Images"""
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be at least 1")

    return args


def load_pyplot():
    """
        Import matplotlib lazily so metric-only runs still work without plots.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def infer_model_version(model_path: Path) -> int | None:
    """
        Infer model version from path parts such as unet_v5 or train_v5.
    """

    for part in reversed(model_path.parts):
        match = VERSION_PATTERN.search(part)

        if match:
            return int(match.group(1))

    return None


def prepare_version_output_dir(
    output_root: Path,
    model_path: Path,
    model_version: int | None,
) -> tuple[Path, int]:
    """
        Create metrics/version_1..version_5 and return this run's output folder.
    """

    for version in METRIC_VERSIONS:
        (output_root / f"version_{version}").mkdir(parents=True, exist_ok=True)

    version = model_version or infer_model_version(model_path)

    if version is None:
        raise ValueError(
            "Cannot infer model version from model path. "
            "Use --model-version with one of: 1, 2, 3, 4, 5."
        )

    return output_root / f"version_{version}", version


def resolve_history_path(model_path: Path, history_path: Path | None) -> Path | None:
    """
        Resolve the training history path for a checkpoint.
    """

    if history_path is not None:
        return history_path.resolve()

    default_history = model_path.parent / "results.json"

    if default_history.is_file():
        return default_history

    return None


def load_training_history(path: Path) -> dict[str, list[float | int]]:
    """
        Load training history generated by src.unet.train.save_history().
    """

    history = json.loads(path.read_text(encoding="utf-8"))
    required_keys = (
        "epoch",
        "train_loss",
        "train_iou",
        "train_dice",
        "val_loss",
        "val_iou",
        "val_dice",
        "lr",
        "epoch_time",
    )

    missing_keys = [key for key in required_keys if key not in history]

    if missing_keys:
        raise ValueError(
            f"Training history missing keys: {', '.join(missing_keys)}"
        )

    epoch_count = len(history["epoch"])

    for key in required_keys:
        if len(history[key]) != epoch_count:
            raise ValueError(
                f"Training history key '{key}' has length {len(history[key])}; "
                f"expected {epoch_count}"
            )

    return {
        key: [float(value) if key != "epoch" else int(value) for value in history[key]]
        for key in required_keys
    }


def smooth_values(values: list[float], window: int = 5) -> list[float]:
    """
        Smooth plot values with a centered moving average.
    """

    if len(values) < 3:
        return values

    window = max(1, min(window, len(values)))
    half_window = window // 2
    smoothed: list[float] = []

    for index in range(len(values)):
        start = max(0, index - half_window)
        end = min(len(values), index + half_window + 1)
        smoothed.append(float(np.mean(values[start:end])))

    return smoothed


def plot_training_history(
    history: dict[str, list[float | int]],
    output_path: Path,
) -> None:
    """
        Save YOLO-style per-epoch U-Net training plots.
    """

    plt = load_pyplot()
    epochs = [int(value) for value in history["epoch"]]

    plot_specs = (
        ("train/loss", "train_loss"),
        ("val/loss", "val_loss"),
        ("train/iou", "train_iou"),
        ("val/iou", "val_iou"),
        ("train/dice", "train_dice"),
        ("val/dice", "val_dice"),
        ("lr", "lr"),
        ("epoch_time", "epoch_time"),
    )

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), dpi=150)
    axes = axes.ravel()

    for axis, (title, key) in zip(axes, plot_specs):
        values = [float(value) for value in history[key]]
        axis.plot(
            epochs,
            values,
            marker="o",
            markersize=3,
            linewidth=1.6,
            label="results",
        )
        axis.plot(
            epochs,
            smooth_values(values),
            linestyle=":",
            linewidth=2.0,
            color="#ff7f0e",
            label="smooth",
        )
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)

    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def find_image_mask_pairs(data_root: Path, split: str) -> list[tuple[Path, Path]]:
    """
        Find image-mask pairs from a dataset split.
    """

    image_dir = data_root / split / "images"
    mask_dir = data_root / split / "masks"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise ValueError(f"No evaluation images found in: {image_dir}")

    pairs: list[tuple[Path, Path]] = []
    missing_masks: list[Path] = []

    for image_path in images:
        mask_path = mask_dir / f"{image_path.stem}.png"

        if mask_path.is_file():
            pairs.append((image_path, mask_path))
        else:
            missing_masks.append(mask_path)

    if missing_masks:
        examples = ", ".join(str(path) for path in missing_masks[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_masks)} masks, for example: {examples}"
        )

    return pairs


def confusion_counts(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    """
        Compute pixel-level TP, FP, FN, TN counts.
    """

    prediction = prediction.astype(bool)
    target = target.astype(bool)

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
    """

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    predicted_positive = tp + fp
    actual_positive = tp + fn
    union = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn
    total = tp + fp + fn + tn

    precision = tp / predicted_positive if predicted_positive else float(fn == 0)
    recall = tp / actual_positive if actual_positive else 1.0
    specificity = tn / (tn + fp) if (tn + fp) else 1.0
    iou = tp / union if union else 1.0
    dice = 2 * tp / dice_denominator if dice_denominator else 1.0
    accuracy = (tp + tn) / total if total else 1.0

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
        Add current image confusion counts to global totals.
    """

    for key in total:
        total[key] += current[key]


def mean_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    """
        Compute mean per-image metrics.
    """

    names = (
        "precision",
        "recall",
        "specificity",
        "iou",
        "dice",
        "pixel_accuracy",
    )

    return {
        name: float(np.mean([row[f"{prefix}_{name}"] for row in rows]))
        for name in names
    }


def predict_probability(
    model: torch.nn.Module,
    image: np.ndarray,
    img_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    """
        Predict probability map using sliding-window or full-image mode.
    """

    if args.mode == "sliding":
        return predict_sliding_window_probability(
            model=model,
            image=image,
            patch_size=args.patch_size,
            overlap=args.overlap,
            batch_size=args.patch_batch_size,
            device=device,
        )

    tensor = preprocess(image, img_size, device)

    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

    return cv2.resize(
        probability,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


def threshold_values(args: argparse.Namespace) -> list[float]:
    """
        Build stable threshold values for threshold-curve metrics.
    """

    values = []
    current = args.threshold_min

    while current <= args.threshold_max + 1e-9:
        values.append(round(float(current), 4))
        current += args.threshold_step

    if args.threshold not in values:
        values.append(round(float(args.threshold), 4))

    return sorted(set(values))


def write_threshold_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """
        Write threshold sweep metrics to CSV.
    """

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_threshold_rows(
    threshold_totals: dict[float, dict[str, int]]
) -> list[dict[str, Any]]:
    """
        Convert threshold confusion counts into metric rows.
    """

    rows = []

    for threshold, counts in threshold_totals.items():
        metrics = metrics_from_counts(counts)
        rows.append({
            "threshold": threshold,
            **metrics,
        })

    return sorted(rows, key=lambda row: row["threshold"])


def save_confusion_matrix_plot(
    counts: dict[str, int],
    output_path: Path,
    normalized: bool,
) -> None:
    """
        Save pixel-level binary confusion matrix.
    """

    plt = load_pyplot()
    matrix = np.array(
        [
            [counts["tn"], counts["fp"]],
            [counts["fn"], counts["tp"]],
        ],
        dtype=np.float64,
    )

    if normalized:
        row_sums = matrix.sum(axis=1, keepdims=True)
        display = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix),
            where=row_sums > 0,
        )
        title = "Pixel Confusion Matrix (Row Normalized)"
        color_label = "Ratio"
        text_format = "{:.3f}"
    else:
        display = np.log10(matrix + 1.0)
        title = "Pixel Confusion Matrix (log10 count + 1)"
        color_label = "log10(count + 1)"
        text_format = "{:,.0f}"

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    image = ax.imshow(display, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=color_label)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["Background", "Scratch"])
    ax.set_yticks([0, 1], labels=["Background", "Scratch"])

    for row_index in range(2):
        for col_index in range(2):
            value = display[row_index, col_index] if normalized else matrix[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                text_format.format(value),
                ha="center",
                va="center",
                color="#111827",
                fontsize=10,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_global_metrics_bar(report: dict[str, Any], output_path: Path) -> None:
    """
        Save a bar chart for global segmentation metrics.
    """

    plt = load_pyplot()
    metrics = report["global"]["raw"]
    names = ["precision", "recall", "specificity", "iou", "dice", "pixel_accuracy"]
    values = [float(metrics[name]) for name in names]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    bars = ax.bar(
        [name.replace("_", "\n").title() for name in names],
        values,
        color=["#2563eb", "#16a34a", "#7c3aed", "#f59e0b", "#dc2626", "#0891b2"],
    )
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Global Segmentation Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.margins(x=0.04)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.025, 1.08),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_per_image_line_plot(
    rows: list[dict[str, Any]],
    metric_name: str,
    output_path: Path,
) -> None:
    """
        Save a per-image metric line plot.
    """

    plt = load_pyplot()
    values = [float(row[f"raw_{metric_name}"]) for row in rows]

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)
    ax.plot(range(1, len(values) + 1), values, color="#2563eb", linewidth=1.6)
    ax.axhline(float(np.mean(values)), color="#dc2626", linestyle="--", linewidth=1.2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Image Index")
    ax.set_ylabel(metric_name.title())
    ax.set_title(f"Per-image {metric_name.title()}")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_precision_recall_scatter(rows: list[dict[str, Any]], output_path: Path) -> None:
    """
        Save a per-image precision/recall scatter plot.
    """

    plt = load_pyplot()
    positives = np.array([int(row["positive_pixels"]) > 0 for row in rows])
    precision = np.array([float(row["raw_precision"]) for row in rows])
    recall = np.array([float(row["raw_recall"]) for row in rows])

    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=150)
    ax.scatter(
        precision[~positives],
        recall[~positives],
        s=18,
        alpha=0.65,
        color="#94a3b8",
        label="No GT scratch",
    )
    ax.scatter(
        precision[positives],
        recall[positives],
        s=24,
        alpha=0.75,
        color="#7c3aed",
        label="GT scratch",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title("Per-image Precision vs Recall")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_positive_pixels_histogram(rows: list[dict[str, Any]], output_path: Path) -> None:
    """
        Save a histogram showing foreground mask pixel distribution.
    """

    plt = load_pyplot()
    positives = np.array([int(row["positive_pixels"]) for row in rows])

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.hist(positives, bins=30, color="#16a34a", alpha=0.8)
    ax.set_xlabel("GT Scratch Pixels")
    ax.set_ylabel("Image Count")
    ax.set_title("Ground-truth Foreground Pixel Distribution")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_error_pixels_scatter(rows: list[dict[str, Any]], output_path: Path) -> None:
    """
        Save a false-positive / false-negative scatter plot.
    """

    plt = load_pyplot()
    fp = np.array([int(row["raw_fp"]) for row in rows])
    fn = np.array([int(row["raw_fn"]) for row in rows])
    positives = np.array([int(row["positive_pixels"]) > 0 for row in rows])

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.scatter(
        fp[~positives],
        fn[~positives],
        s=18,
        alpha=0.65,
        color="#94a3b8",
        label="No GT scratch",
    )
    ax.scatter(
        fp[positives],
        fn[positives],
        s=24,
        alpha=0.75,
        color="#dc2626",
        label="GT scratch",
    )
    ax.set_xlabel("False Positive Pixels")
    ax.set_ylabel("False Negative Pixels")
    ax.set_title("Per-image Error Pixels")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_threshold_curves(rows: list[dict[str, Any]], output_path: Path) -> None:
    """
        Save IoU/Dice/Precision/Recall curves across thresholds.
    """

    plt = load_pyplot()
    thresholds = [float(row["threshold"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for metric_name, color in (
        ("iou", "#f59e0b"),
        ("dice", "#dc2626"),
        ("precision", "#2563eb"),
        ("recall", "#16a34a"),
    ):
        values = [float(row[metric_name]) for row in rows]
        ax.plot(
            thresholds,
            values,
            marker="o",
            markersize=3,
            linewidth=1.6,
            color=color,
            label=metric_name.title(),
        )

    ax.set_xlim(min(thresholds), max(thresholds))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep Curves")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_sample_panel(
    sample_dir: Path,
    image_path: Path,
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    metrics: dict[str, float | int],
) -> None:
    """
        Save one side-by-side visual sample.
    """

    plt = load_pyplot()
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_rgb = np.zeros_like(image_rgb)
    gt_rgb[target] = (255, 255, 0)
    pred_rgb = np.zeros_like(image_rgb)
    pred_rgb[prediction] = (255, 0, 0)
    overlay_rgb = cv2.cvtColor(
        make_overlay(image, prediction.astype(np.uint8) * 255),
        cv2.COLOR_BGR2RGB,
    )

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=150)
    panels = (
        ("Image", image_rgb),
        ("GT Mask", gt_rgb),
        ("Prediction", pred_rgb),
        ("Overlay", overlay_rgb),
    )

    for axis, (title, panel) in zip(axes, panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(
        f"{image_path.name} | "
        f"IoU={float(metrics['iou']):.3f} "
        f"Dice={float(metrics['dice']):.3f} "
        f"P={float(metrics['precision']):.3f} "
        f"R={float(metrics['recall']):.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(sample_dir / f"{image_path.stem}_sample.jpg")
    plt.close(fig)


def save_all_plots(
    output_dir: Path,
    report: dict[str, Any],
    per_image: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    training_history: dict[str, list[float | int]] | None,
) -> list[str]:
    """
        Save all evaluation plots and return their relative paths.
    """

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    """Remove stale plot images from older evaluation_v2 runs."""
    for old_plot in plots_dir.glob("*.png"):
        old_plot.unlink()

    plot_paths = {
        "confusion_matrix_normalized": plots_dir / "confusion_matrix_normalized.png",
        "global_metrics_bar": plots_dir / "global_metrics_bar.png",
        "threshold_curves": plots_dir / "threshold_curves.png",
    }

    if training_history is not None:
        plot_paths["training_history"] = plots_dir / "training_history.png"

    counts = report["global"]["raw"]
    save_confusion_matrix_plot(
        counts=counts,
        output_path=plot_paths["confusion_matrix_normalized"],
        normalized=True,
    )
    save_global_metrics_bar(report, plot_paths["global_metrics_bar"])
    save_threshold_curves(threshold_rows, plot_paths["threshold_curves"])

    if training_history is not None:
        plot_training_history(
            history=training_history,
            output_path=plot_paths["training_history"],
        )

    return [str(path.relative_to(output_dir)) for path in plot_paths.values()]


def main() -> None:
    """
        Evaluate U-Net model on one dataset split and save visual reports.
    """

    """Parse Arguments"""
    args = parse_args()

    """Resolve Paths"""
    model_path = args.model.resolve()
    data_root = args.data.resolve()
    output_root = args.output_dir.resolve()
    output_dir, model_version = prepare_version_output_dir(
        output_root=output_root,
        model_path=model_path,
        model_version=args.model_version,
    )

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

    """Remove stale table reports; evaluation.py now keeps plot outputs only."""
    for stale_report in (
        "metrics.json",
        "per_image_metrics.csv",
        "threshold_metrics.csv",
    ):
        stale_path = output_dir / stale_report

        if stale_path.is_file():
            stale_path.unlink()

    """Remove stale visual samples so saved samples remain scratch-only."""
    if args.save_samples:
        sample_dir = output_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        for old_sample in sample_dir.glob("*_sample.jpg"):
            old_sample.unlink()

    """Global Counts"""
    raw_total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    processed_total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    """Threshold Curve Counts"""
    thresholds = threshold_values(args)
    threshold_totals = {
        threshold: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for threshold in thresholds
    }

    """Per-image Metric Rows"""
    per_image: list[dict[str, Any]] = []
    saved_sample_count = 0

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

        """Post-processing is disabled for first v2 visual baseline"""
        processed_prediction = raw_prediction
        processed_counts = confusion_counts(processed_prediction, target)
        processed_metrics = metrics_from_counts(processed_counts)
        add_counts(processed_total, processed_counts)

        """Accumulate Threshold Curve Counts"""
        for threshold in thresholds:
            threshold_prediction = probability >= threshold
            threshold_counts = confusion_counts(threshold_prediction, target)
            add_counts(threshold_totals[threshold], threshold_counts)

        """Base Row Information"""
        row: dict[str, Any] = {
            "image": image_path.name,
            "width": image.shape[1],
            "height": image.shape[0],
            "positive_pixels": int(target.sum()),
        }

        """Add Raw And Processed Metrics To Row"""
        row.update({f"raw_{key}": value for key, value in raw_metrics.items()})
        row.update({
            f"post_processed_{key}": value
            for key, value in processed_metrics.items()
        })
        per_image.append(row)

        """Save Visual Samples"""
        if (
            args.save_samples
            and int(target.sum()) > 0
            and saved_sample_count < args.sample_count
        ):
            save_sample_panel(
                sample_dir=output_dir / "samples",
                image_path=image_path,
                image=image,
                target=target,
                prediction=raw_prediction,
                metrics=raw_metrics,
            )
            saved_sample_count += 1

    """Elapsed Time"""
    elapsed_seconds = time.perf_counter() - started

    """Build Threshold Metrics"""
    threshold_rows = build_threshold_rows(threshold_totals)

    """Load Training History If Available"""
    history_path = resolve_history_path(model_path, args.history)
    training_history = (
        load_training_history(history_path)
        if history_path is not None
        else None
    )

    """Build Evaluation Report"""
    report = {
        "model_type": "U-Net binary segmentation",
        "evaluation_version": 2,
        "model": str(model_path),
        "model_version": model_version,
        "data": str(data_root),
        "split": args.split,
        "image_count": len(per_image),
        "metrics_root": str(output_root),

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
            "plots": args.plots,
            "save_samples": args.save_samples,
            "sample_count": args.sample_count,
            "saved_sample_count": saved_sample_count,
            "history": str(history_path) if history_path is not None else None,
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_step": args.threshold_step,
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

        # Best Thresholds
        "best_thresholds": {
            "iou": max(threshold_rows, key=lambda row: row["iou"]),
            "dice": max(threshold_rows, key=lambda row: row["dice"]),
        },

        # Speed Metrics
        "elapsed_seconds": round(elapsed_seconds, 3),
        "milliseconds_per_image": round(
            elapsed_seconds * 1000 / len(per_image),
            3,
        ),
    }

    if training_history is not None and history_path is not None:
        report["training_history"] = {
            "source": str(history_path),
            "epochs": len(training_history["epoch"]),
            "best_val_iou": max(float(value) for value in training_history["val_iou"]),
            "best_val_dice": max(float(value) for value in training_history["val_dice"]),
        }

    """Save Plots"""
    if args.plots:
        report["plots"] = save_all_plots(
            output_dir=output_dir,
            report=report,
            per_image=per_image,
            threshold_rows=threshold_rows,
            training_history=training_history,
        )
        if "training_history" in report:
            report["training_history"]["plot"] = "plots/training_history.png"

    """Print Summary"""
    processed = report["global"]["post_processed"]
    best_iou = report["best_thresholds"]["iou"]
    best_dice = report["best_thresholds"]["dice"]

    print("\nU-Net evaluation v2 complete")
    print(f"Model: {model_path}")
    print(f"Version: {model_version}")
    print(f"Split: {args.split} ({len(per_image)} images)")
    print(f"Mode: {args.mode}")
    if args.mode == "sliding":
        print(
            f"Patch size: {args.patch_size} | "
            f"Overlap: {args.overlap} | "
            f"Patch batch size: {args.patch_batch_size}"
        )
    if args.plots:
        print(f"Plots: {output_dir / 'plots'}")
        if "training_history" in report:
            print(f"Training history plot: {output_dir / 'plots' / 'training_history.png'}")
    if args.save_samples:
        print(f"Samples: {output_dir / 'samples'}")
    print(
        "Post-processed: "
        f"IoU={processed['iou']:.4f} Dice={processed['dice']:.4f} "
        f"P={processed['precision']:.4f} R={processed['recall']:.4f}"
    )
    print(
        "Best threshold: "
        f"IoU@{best_iou['threshold']:.2f}={best_iou['iou']:.4f} | "
        f"Dice@{best_dice['threshold']:.2f}={best_dice['dice']:.4f}"
    )


"""Run Main"""
if __name__ == "__main__":
    main()
