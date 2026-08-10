"""
    Prompt-based evaluation for a fine-tuned SAM2 scratch model.

    The SAM2 training dataset stores each scratch instance as a one-frame video:
        JPEGImages/<split>/<sample>/00000.png
        Annotations/<split>/<sample>/00000.png

    This script evaluates how well SAM2 can recover one scratch instance when
    given an ideal point, box, or box+point prompt derived from the ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.path import DATA_DIR, METRICS_DIR, MODELS_DIR
from src.evaluation.plot_utils import (
    load_pyplot,
    safe_float,
    save_binary_confusion_matrix_plot,
    save_global_metrics_bar,
)


DEFAULT_MODEL_DIR = MODELS_DIR / "sam2"
DEFAULT_CHECKPOINT = DEFAULT_MODEL_DIR / "checkpoint.pt"
DEFAULT_DATA = DATA_DIR / "scratch_sam2_format"
DEFAULT_OUTPUT_DIR = METRICS_DIR / "sam2"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


@dataclass(frozen=True)
class Sample:
    """
        One SAM2 one-frame sample.
    """

    name: str
    image_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    """
        Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned SAM2 on scratch instance masks."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sam2-repo", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--prompt",
        choices=("point", "box", "box_point"),
        default="box_point",
        help="Prompt style used for evaluation.",
    )
    parser.add_argument("--box-pad", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-overlays", type=int, default=20)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.box_pad < 0:
        parser.error("--box-pad must be >= 0")
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be >= 1")
    if args.save_overlays < 0:
        parser.error("--save-overlays must be >= 0")

    return args


def resolve_project_path(path: Path) -> Path:
    """
        Resolve a path relative to the project root.
    """

    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def setup_sam2_import(sam2_repo: Path | None) -> None:
    """
        Optionally import SAM2 from a local checkout.
    """

    if sam2_repo is None:
        return

    sam2_repo = resolve_project_path(sam2_repo)
    if not sam2_repo.is_dir():
        raise FileNotFoundError(f"SAM2 repo not found: {sam2_repo}")
    sys.path.insert(0, str(sam2_repo))


def select_device(name: str) -> str:
    """
        Select CPU/CUDA for evaluation.
    """

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return name


def load_samples(data_root: Path, split: str, max_images: int | None) -> list[Sample]:
    """
        Load SAM2 one-frame samples from ImageSets.
    """

    list_path = data_root / "ImageSets" / f"{split}.txt"
    image_root = data_root / "JPEGImages"
    mask_root = data_root / "Annotations"

    if not list_path.is_file():
        raise FileNotFoundError(f"Split list not found: {list_path}")

    samples: list[Sample] = []
    with list_path.open("r", encoding="utf-8") as file:
        names = [line.strip() for line in file if line.strip()]

    if max_images is not None:
        names = names[:max_images]

    for name in names:
        image_path = resolve_frame_image(image_root / name)
        mask_path = mask_root / name / "00000.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        samples.append(Sample(name=name, image_path=image_path, mask_path=mask_path))

    if not samples:
        raise RuntimeError(f"No samples found for split: {split}")

    return samples


def resolve_frame_image(frame_dir: Path) -> Path:
    """
        Find one SAM2 frame image, preferring PNG but accepting older JPG data.
    """

    candidates = [
        frame_dir / "00000.png",
        frame_dir / "00000.jpg",
        frame_dir / "00000.jpeg",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    names = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"Image not found in {frame_dir}; expected one of: {names}")


def load_image_rgb(path: Path) -> np.ndarray:
    """
        Load one image as RGB uint8.
    """

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    """
        Load one binary mask.
    """

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return mask > 0


def mask_box(mask: np.ndarray, pad: int) -> np.ndarray:
    """
        Build an XYXY box around the foreground mask.
    """

    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Cannot build a prompt from an empty mask.")

    height, width = mask.shape[:2]
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(width - 1, int(xs.max()) + pad)
    y2 = min(height - 1, int(ys.max()) + pad)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def positive_point(mask: np.ndarray) -> np.ndarray:
    """
        Pick a foreground point near the center of the scratch.
    """

    binary = mask.astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)

    if not mask[y, x]:
        ys, xs = np.where(mask)
        mid = len(xs) // 2
        x = int(xs[mid])
        y = int(ys[mid])

    return np.array([[x, y]], dtype=np.float32)


def predict_mask(
    predictor: Any,
    image: np.ndarray,
    target: np.ndarray,
    prompt: str,
    box_pad: int,
) -> tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None]:
    """
        Predict one binary mask from a point/box prompt.
    """

    predictor.set_image(image)

    point_coords = None
    point_labels = None
    box = None

    if prompt in {"point", "box_point"}:
        point_coords = positive_point(target)
        point_labels = np.array([1], dtype=np.int32)
    if prompt in {"box", "box_point"}:
        box = mask_box(target, box_pad)

    masks, scores, _logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
        return_logits=False,
        normalize_coords=True,
    )

    best_index = int(np.argmax(scores))
    prediction = masks[best_index].astype(bool)
    return prediction, float(scores[best_index]), point_coords, box


def confusion_counts(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    """
        Compute binary pixel-level confusion counts.
    """

    return {
        "tp": int(np.logical_and(prediction, target).sum()),
        "fp": int(np.logical_and(prediction, ~target).sum()),
        "fn": int(np.logical_and(~prediction, target).sum()),
        "tn": int(np.logical_and(~prediction, ~target).sum()),
    }


def safe_divide(numerator: float, denominator: float) -> float:
    """
        Divide while handling empty denominators.
    """

    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """
        Compute segmentation metrics from confusion counts.
    """

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    return {
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "specificity": safe_divide(tn, tn + fp),
        "iou": safe_divide(tp, tp + fp + fn),
        "dice": safe_divide(2 * tp, 2 * tp + fp + fn),
        "pixel_accuracy": safe_divide(tp + tn, tp + fp + fn + tn),
    }


def add_counts(total: dict[str, int], counts: dict[str, int]) -> None:
    """
        Accumulate confusion counts in-place.
    """

    for key, value in counts.items():
        total[key] += value


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """
        Compute mean per-sample metrics.
    """

    names = ("precision", "recall", "specificity", "iou", "dice", "pixel_accuracy")
    return {
        name: float(np.mean([safe_float(row[name]) for row in rows]))
        for name in names
    }


def save_overlay(
    image_rgb: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    point: np.ndarray | None,
    box: np.ndarray | None,
    output_path: Path,
) -> None:
    """
        Save an RGB overlay showing GT in green, prediction in red, and overlap in yellow.
    """

    overlay = image_rgb.copy()
    gt_only = np.logical_and(target, ~prediction)
    pred_only = np.logical_and(prediction, ~target)
    both = np.logical_and(target, prediction)

    overlay[gt_only] = (0, 255, 0)
    overlay[pred_only] = (255, 0, 0)
    overlay[both] = (255, 255, 0)
    blended = cv2.addWeighted(image_rgb, 0.65, overlay, 0.35, 0)

    bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    if box is not None:
        x1, y1, x2, y2 = box.astype(int).tolist()
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 255), 1)
    if point is not None:
        x, y = point[0].astype(int).tolist()
        cv2.circle(bgr, (x, y), 4, (255, 255, 255), -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), bgr):
        raise RuntimeError(f"Cannot write overlay: {output_path}")


def save_per_sample_metrics_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    """
        Save IoU/Dice/Precision/Recall curves across evaluated samples.
    """

    if not rows:
        return

    plt = load_pyplot()
    xs = list(range(1, len(rows) + 1))

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=150)
    for metric_name, color in (
        ("iou", "#f59e0b"),
        ("dice", "#dc2626"),
        ("precision", "#2563eb"),
        ("recall", "#16a34a"),
    ):
        values = [safe_float(row[metric_name]) for row in rows]
        ax.plot(
            xs,
            values,
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=color,
            label=metric_name.title(),
        )

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Score")
    ax.set_title("SAM2 Per-sample Prompt Metrics")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_score_iou_scatter(rows: list[dict[str, Any]], output_path: Path) -> None:
    """
        Save a scatter plot comparing SAM score with IoU.
    """

    if not rows:
        return

    plt = load_pyplot()
    sam_scores = [safe_float(row["sam_score"]) for row in rows]
    ious = [safe_float(row["iou"]) for row in rows]
    dices = [safe_float(row["dice"]) for row in rows]

    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=150)
    scatter = ax.scatter(
        sam_scores,
        ious,
        c=dices,
        s=38,
        alpha=0.8,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Dice")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("SAM Score")
    ax.set_ylabel("IoU")
    ax.set_title("SAM2 Score vs IoU")
    ax.grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_mask_ratio_histogram(rows: list[dict[str, Any]], output_path: Path) -> None:
    """
        Save predicted/ground-truth mask size ratio histogram.
    """

    if not rows:
        return

    plt = load_pyplot()
    ratios = [safe_float(row["pred_gt_ratio"]) for row in rows]
    clipped = np.clip(np.array(ratios, dtype=np.float64), 0.0, 5.0)

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    ax.hist(clipped, bins=30, color="#7c3aed", alpha=0.82)
    ax.axvline(1.0, color="#dc2626", linestyle="--", linewidth=1.3)
    ax.set_xlabel("Predicted Pixels / Ground-truth Pixels")
    ax.set_ylabel("Sample Count")
    ax.set_title("SAM2 Mask Size Ratio")
    ax.grid(axis="y", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_all_plots(
    output_dir: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[str]:
    """
        Save visual evaluation plots and return relative paths.
    """

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for old_plot in plots_dir.glob("*.png"):
        old_plot.unlink()

    plot_paths = {
        "confusion_matrix_normalized": plots_dir / "confusion_matrix_normalized.png",
        "global_metrics_bar": plots_dir / "global_metrics_bar.png",
        "per_sample_metrics": plots_dir / "per_sample_metrics.png",
        "sam_score_vs_iou": plots_dir / "sam_score_vs_iou.png",
        "mask_size_ratio": plots_dir / "mask_size_ratio.png",
    }

    save_binary_confusion_matrix_plot(
        counts=summary["counts"],
        output_path=plot_paths["confusion_matrix_normalized"],
        class_names=("scratch", "background"),
    )
    save_global_metrics_bar(
        metrics=summary["metrics"],
        output_path=plot_paths["global_metrics_bar"],
        metric_names=[
            "precision",
            "recall",
            "specificity",
            "iou",
            "dice",
            "pixel_accuracy",
        ],
        title="SAM2 Global Segmentation Metrics",
    )
    save_per_sample_metrics_plot(rows, plot_paths["per_sample_metrics"])
    save_score_iou_scatter(rows, plot_paths["sam_score_vs_iou"])
    save_mask_ratio_histogram(rows, plot_paths["mask_size_ratio"])

    return [str(path.relative_to(output_dir)) for path in plot_paths.values()]


def remove_stale_csv(output_dir: Path) -> None:
    """
        Remove CSV output from previous evaluator versions.
    """

    stale_csv = output_dir / "per_sample_metrics.csv"
    if stale_csv.is_file():
        stale_csv.unlink()


def build_predictor(config: str, checkpoint: Path, device: str) -> Any:
    """
        Build a SAM2 image predictor from a fine-tuned checkpoint.
    """

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(config, str(checkpoint), device=device)
    return SAM2ImagePredictor(model)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """
        Run prompt-based SAM2 evaluation.
    """

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    setup_sam2_import(args.sam2_repo)

    data_root = resolve_project_path(args.data)
    checkpoint = resolve_project_path(args.checkpoint)
    output_dir = resolve_project_path(args.output_dir) / args.prompt

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = select_device(args.device)
    samples = load_samples(data_root, args.split, args.max_images)
    predictor = build_predictor(args.config, checkpoint, device)

    total_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, sample in enumerate(samples, start=1):
        sample_start = time.perf_counter()
        image = load_image_rgb(sample.image_path)
        target = load_mask(sample.mask_path)

        prediction, sam_score, point, box = predict_mask(
            predictor=predictor,
            image=image,
            target=target,
            prompt=args.prompt,
            box_pad=args.box_pad,
        )

        counts = confusion_counts(prediction, target)
        add_counts(total_counts, counts)
        metrics = metrics_from_counts(counts)
        seconds = time.perf_counter() - sample_start

        gt_pixels = int(target.sum())
        pred_pixels = int(prediction.sum())
        row = {
            "sample": sample.name,
            "image": str(sample.image_path),
            "mask": str(sample.mask_path),
            "sam_score": sam_score,
            "gt_pixels": gt_pixels,
            "pred_pixels": pred_pixels,
            "pred_gt_ratio": safe_divide(pred_pixels, gt_pixels),
            "seconds": seconds,
            **counts,
            **metrics,
        }
        rows.append(row)

        if index <= args.save_overlays:
            overlay_path = output_dir / "overlays" / f"{index:04d}_{Path(sample.name).name}.png"
            save_overlay(image, target, prediction, point, box, overlay_path)

        print(
            f"[{index}/{len(samples)}] {sample.name} "
            f"IoU={metrics['iou']:.4f} Dice={metrics['dice']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
            f"score={sam_score:.4f} {seconds:.2f}s",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    global_metrics = metrics_from_counts(total_counts)
    summary = {
        "checkpoint": str(checkpoint),
        "config": args.config,
        "data": str(data_root),
        "split": args.split,
        "prompt": args.prompt,
        "box_pad": args.box_pad,
        "device": device,
        "samples": len(samples),
        "seconds": elapsed,
        "average_seconds_per_sample": safe_divide(elapsed, len(samples)),
        "counts": total_counts,
        "metrics": global_metrics,
        "mean_per_sample": mean_metrics(rows),
        "per_sample": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_csv(output_dir)

    if args.plots:
        summary["plots"] = save_all_plots(
            output_dir=output_dir,
            summary=summary,
            rows=rows,
        )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nSAM2 evaluation summary")
    print(f"Samples: {len(samples)}")
    print(f"Output: {output_dir}")
    if args.plots:
        print(f"Plots: {output_dir / 'plots'}")
    print(
        "Global: "
        f"IoU={global_metrics['iou']:.4f} "
        f"Dice={global_metrics['dice']:.4f} "
        f"P={global_metrics['precision']:.4f} "
        f"R={global_metrics['recall']:.4f}"
    )
    print(
        "Mean/sample: "
        f"IoU={summary['mean_per_sample']['iou']:.4f} "
        f"Dice={summary['mean_per_sample']['dice']:.4f}"
    )

    return summary


def main() -> None:
    """
        Entrypoint.
    """

    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
