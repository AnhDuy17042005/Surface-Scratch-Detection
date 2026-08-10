"""
    Pixel-level evaluation for YOLO scratch segmentation.

    This evaluates YOLO segmentation masks against binary scratch masks using
    pixel-level segmentation metrics.
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.data import IMAGE_EXTENSIONS
from configs.yolo import (
    YOLO_SCRATCH_CONFIDENCE_THRESHOLD,
    YOLO_SCRATCH_EVAL_DATASET,
    YOLO_SCRATCH_IMAGE_SIZE,
    YOLO_SCRATCH_IOU_THRESHOLD,
    YOLO_SCRATCH_MASK_THRESHOLD,
    YOLO_SCRATCH_METRICS_OUTPUT,
    YOLO_SCRATCH_MODEL,
)
from src.evaluation.plot_utils import (
    load_yolo_results_csv,
    resolve_yolo_history_path,
    save_binary_confusion_matrix_plot,
    save_global_metrics_bar,
    save_threshold_curves,
    save_yolo_training_history,
)
from src.yolo.scratch.inference import normalize_ultralytics_model_path
from ultralytics import YOLO


DEFAULT_MODEL = YOLO_SCRATCH_MODEL
DEFAULT_OUTPUT_DIR = YOLO_SCRATCH_METRICS_OUTPUT


def parse_args() -> argparse.Namespace:
    """
        Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate YOLO-seg against binary scratch masks."
    )
    parser.add_argument("--data", type=Path, default=YOLO_SCRATCH_EVAL_DATASET)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=YOLO_SCRATCH_IMAGE_SIZE)
    parser.add_argument("--conf", type=float, default=YOLO_SCRATCH_CONFIDENCE_THRESHOLD)
    parser.add_argument("--iou", type=float, default=YOLO_SCRATCH_IOU_THRESHOLD)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mask-threshold", type=float, default=YOLO_SCRATCH_MASK_THRESHOLD)
    parser.add_argument(
        "--confidence-min",
        "--threshold-min",
        dest="confidence_min",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--confidence-max",
        "--threshold-max",
        dest="confidence_max",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--confidence-step",
        "--threshold-step",
        dest="confidence_step",
        type=float,
        default=0.05,
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be in [0, 1]")
    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou must be in [0, 1]")
    if not 0.0 <= args.mask_threshold <= 1.0:
        parser.error("--mask-threshold must be in [0, 1]")
    if not 0.0 <= args.confidence_min <= 1.0:
        parser.error("--confidence-min must be in [0, 1]")
    if not 0.0 <= args.confidence_max <= 1.0:
        parser.error("--confidence-max must be in [0, 1]")
    if args.confidence_min > args.confidence_max:
        parser.error("--confidence-min must be <= --confidence-max")
    if args.confidence_step <= 0.0:
        parser.error("--confidence-step must be positive")
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be at least 1")

    return args


def resolve_project_path(path: Path) -> Path:
    """
        Resolve a path relative to the project root.
    """

    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def find_pairs(data_root: Path, split: str) -> list[tuple[Path, Path]]:
    """
        Find image/mask pairs in a split dataset.
    """

    image_dir = data_root / split / "images"
    mask_dir = data_root / split / "masks"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found for image: {image_path}")

        pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in: {image_dir}")

    return pairs


def confusion_counts(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    """
        Compute binary pixel-level confusion counts.
    """

    pred = prediction.astype(bool)
    gt = target.astype(bool)

    return {
        "tp": int(np.logical_and(pred, gt).sum()),
        "fp": int(np.logical_and(pred, ~gt).sum()),
        "fn": int(np.logical_and(~pred, gt).sum()),
        "tn": int(np.logical_and(~pred, ~gt).sum()),
    }


def add_counts(total: dict[str, int], counts: dict[str, int]) -> None:
    """
        Accumulate confusion counts in-place.
    """

    for key, value in counts.items():
        total[key] += value


def safe_divide(numerator: float, denominator: float) -> float:
    """
        Return 0 when denominator is zero.
    """

    return numerator / denominator if denominator else 0.0


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """
        Convert confusion counts to segmentation metrics.
    """

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    iou = safe_divide(tp, tp + fp + fn)
    dice = safe_divide(2 * tp, 2 * tp + fp + fn)
    pixel_accuracy = safe_divide(tp + tn, tp + fp + fn + tn)

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "iou": iou,
        "dice": dice,
        "pixel_accuracy": pixel_accuracy,
    }


def confidence_values(args: argparse.Namespace) -> list[float]:
    """
        Build stable confidence values for curve metrics.
    """

    values = []
    current = args.confidence_min

    while current <= args.confidence_max + 1e-9:
        values.append(round(float(current), 4))
        current += args.confidence_step

    values.append(round(float(args.conf), 4))
    return sorted(set(values))


def build_confidence_rows(
    confidence_totals: dict[float, dict[str, int]]
) -> list[dict[str, Any]]:
    """
        Convert confidence confusion counts into metric rows.
    """

    rows = []

    for confidence, counts in confidence_totals.items():
        rows.append({
            "confidence": confidence,
            **counts,
            **metrics_from_counts(counts),
        })

    return sorted(rows, key=lambda row: row["confidence"])


def write_confidence_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """
        Save confidence sweep metrics.
    """

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_plot_outputs(
    output_dir: Path,
    report: dict[str, Any],
    confidence_rows: list[dict[str, Any]],
    history_rows: list[dict[str, float]],
) -> list[str]:
    """
        Save YOLO scratch plots in the standard evaluation folder layout.
    """

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for old_plot in plots_dir.glob("*.png"):
        old_plot.unlink()

    plot_paths = {
        "confusion_matrix_normalized": plots_dir / "confusion_matrix_normalized.png",
        "global_metrics_bar": plots_dir / "global_metrics_bar.png",
        "threshold_curves": plots_dir / "threshold_curves.png",
    }

    save_binary_confusion_matrix_plot(
        counts=report["counts"],
        output_path=plot_paths["confusion_matrix_normalized"],
        class_names=("scratch", "background"),
    )
    save_global_metrics_bar(
        metrics=report["global"],
        output_path=plot_paths["global_metrics_bar"],
        metric_names=[
            "precision",
            "recall",
            "specificity",
            "iou",
            "dice",
            "pixel_accuracy",
        ],
        title="YOLO Scratch Global Segmentation Metrics",
    )
    save_threshold_curves(
        rows=confidence_rows,
        output_path=plot_paths["threshold_curves"],
        x_name="confidence",
        metric_names=["iou", "dice", "precision", "recall"],
        title="YOLO Scratch Confidence Curves",
    )

    if history_rows:
        plot_paths["training_history"] = plots_dir / "training_history.png"
        save_yolo_training_history(
            rows=history_rows,
            output_path=plot_paths["training_history"],
            task="segment",
        )

    return [str(path.relative_to(output_dir)) for path in plot_paths.values()]


def result_to_confidence_mask(
    result: Any,
    width: int,
    height: int,
    mask_threshold: float,
    confidence_threshold: float,
) -> tuple[np.ndarray, int]:
    """
        Convert YOLO result to mask after filtering instances by confidence.
    """

    mask = np.zeros((height, width), dtype=np.uint8)
    if result.masks is None or result.masks.data is None:
        return mask, 0

    mask_data = result.masks.data.detach().cpu().numpy()
    if mask_data.size == 0:
        return mask, 0

    if result.boxes is None or result.boxes.conf is None:
        confidences = np.ones(mask_data.shape[0], dtype=np.float32)
    else:
        confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32)

    instance_count = min(mask_data.shape[0], confidences.shape[0])
    if instance_count == 0:
        return mask, 0

    keep = confidences[:instance_count] >= confidence_threshold
    if not np.any(keep):
        return mask, 0

    combined = np.max(mask_data[:instance_count][keep], axis=0)

    if combined.shape != (height, width):
        combined = cv2.resize(
            combined.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    mask[combined >= mask_threshold] = 255
    return mask, int(np.count_nonzero(keep))


def evaluate_pair(
    model: YOLO,
    image_path: Path,
    mask_path: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], float, dict[float, dict[str, int]]]:
    """
        Evaluate one image/mask pair.
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    target_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    if target_image is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    if image.shape[:2] != target_image.shape[:2]:
        raise ValueError(
            f"Image and mask size mismatch: {image_path.name} "
            f"{image.shape[:2]} != {target_image.shape[:2]}"
        )

    start = time.perf_counter()
    result = model.predict(
        source=image,
        imgsz=args.imgsz,
        conf=min(args.conf, args.confidence_min),
        iou=args.iou,
        device=args.device,
        task="segment",
        verbose=False,
    )[0]
    inference_seconds = time.perf_counter() - start

    h, w = image.shape[:2]
    mask, instances = result_to_confidence_mask(
        result=result,
        width=w,
        height=h,
        mask_threshold=args.mask_threshold,
        confidence_threshold=args.conf,
    )

    target = target_image > 0
    prediction = mask > 0
    counts = confusion_counts(prediction, target)
    metrics = metrics_from_counts(counts)
    confidence_counts: dict[float, dict[str, int]] = {}

    for confidence in confidence_values(args):
        confidence_mask, _confidence_instances = result_to_confidence_mask(
            result=result,
            width=w,
            height=h,
            mask_threshold=args.mask_threshold,
            confidence_threshold=confidence,
        )
        confidence_counts[confidence] = confusion_counts(confidence_mask > 0, target)

    row = {
        "image": image_path.name,
        "instances": instances,
        "positive_pixels": int(target.sum()),
        "predicted_pixels": int(prediction.sum()),
        "inference_seconds": inference_seconds,
        **counts,
        **metrics,
    }
    return row, inference_seconds, confidence_counts


def write_outputs(
    output_dir: Path,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    confidence_rows: list[dict[str, Any]],
) -> None:
    """
        Save JSON and CSV evaluation outputs.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not rows:
        return

    with (output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    stale_threshold_csv = output_dir / "threshold_sweep.csv"
    if stale_threshold_csv.exists():
        stale_threshold_csv.unlink()

    write_confidence_csv(output_dir / "confidence_sweep.csv", confidence_rows)


def main() -> None:
    """
        Run YOLO-seg evaluation.
    """

    args = parse_args()
    model_path = normalize_ultralytics_model_path(resolve_project_path(args.model))
    data_root = resolve_project_path(args.data)
    output_dir = resolve_project_path(args.output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    pairs = find_pairs(data_root, args.split)
    if args.max_images is not None:
        pairs = pairs[: args.max_images]

    model = YOLO(str(model_path), task="segment")

    total_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    confidences = confidence_values(args)
    confidence_totals = {
        confidence: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for confidence in confidences
    }
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for image_path, mask_path in pairs:
        row, _seconds, pair_confidence_counts = evaluate_pair(
            model=model,
            image_path=image_path,
            mask_path=mask_path,
            args=args,
        )
        rows.append(row)
        add_counts(total_counts, {key: int(row[key]) for key in ("tp", "fp", "fn", "tn")})
        for confidence, counts in pair_confidence_counts.items():
            add_counts(confidence_totals[confidence], counts)

    elapsed_seconds = time.perf_counter() - started
    global_metrics = metrics_from_counts(total_counts)
    confidence_rows = build_confidence_rows(confidence_totals)
    history_path = resolve_yolo_history_path(model_path, args.history)
    history_rows = load_yolo_results_csv(history_path) if history_path is not None else []
    report = {
        "model_type": "YOLO segmentation",
        "evaluation_version": 2,
        "model": str(model_path),
        "data": str(data_root),
        "split": args.split,
        "image_count": len(rows),
        "settings": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "mask_threshold": args.mask_threshold,
            "max_images": args.max_images,
            "plots": args.plots,
            "history": str(history_path) if history_path is not None else None,
            "confidence_min": args.confidence_min,
            "confidence_max": args.confidence_max,
            "confidence_step": args.confidence_step,
        },
        "counts": total_counts,
        "global": global_metrics,
        "best_confidences": {
            "iou": max(confidence_rows, key=lambda row: row["iou"]),
            "dice": max(confidence_rows, key=lambda row: row["dice"]),
        },
        "elapsed_seconds": elapsed_seconds,
        "milliseconds_per_image": elapsed_seconds * 1000 / len(rows),
        "mean_inference_seconds": float(np.mean([row["inference_seconds"] for row in rows])),
    }

    if history_rows and history_path is not None:
        report["training_history"] = {
            "source": str(history_path),
            "epochs": len(history_rows),
        }

    if args.plots:
        report["plots"] = add_plot_outputs(
            output_dir=output_dir,
            report=report,
            confidence_rows=confidence_rows,
            history_rows=history_rows,
        )
        if "training_history" in report:
            report["training_history"]["plot"] = "plots/training_history.png"

    write_outputs(
        output_dir=output_dir,
        report=report,
        rows=rows,
        confidence_rows=confidence_rows,
    )

    print("YOLO-seg evaluation complete")
    print(f"Model: {model_path}")
    print(f"Split: {args.split} ({len(rows)} images)")
    print(
        "Global: "
        f"precision={global_metrics['precision']:.4f} "
        f"recall={global_metrics['recall']:.4f} "
        f"iou={global_metrics['iou']:.4f} "
        f"dice={global_metrics['dice']:.4f} "
        f"pixel_accuracy={global_metrics['pixel_accuracy']:.4f}"
    )
    print(f"Mean inference: {report['mean_inference_seconds']:.4f}s/image")
    print(f"Total elapsed: {elapsed_seconds:.2f}s")
    print(f"Output: {output_dir}")
    if args.plots:
        print(f"Plots: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
