"""Evaluate YOLO component bounding-box detector."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

"""Keep matplotlib/Ultralytics cache away from the user's home folder."""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ultralytics import YOLO

from configs.yolo import (
    YOLO_COMPONENT_BATCH_SIZE,
    YOLO_COMPONENT_IMAGE_SIZE,
    YOLO_COMPONENT_IOU_THRESHOLD,
    YOLO_COMPONENT_METRICS_OUTPUT,
    YOLO_COMPONENT_MODEL,
    YOLO_COMPONENT_TASK,
    YOLO_COMPONENT_TRAIN_DATA,
    YOLO_COMPONENT_WORKERS,
)
from src.evaluation.plot_utils import (
    load_yolo_results_csv,
    resolve_yolo_history_path,
    save_global_metrics_bar,
    save_image_collage,
    save_yolo_training_history,
)


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate YOLO component bounding-box detector."
    )

    parser.add_argument("--model", type=Path, default=YOLO_COMPONENT_MODEL)
    parser.add_argument("--data", type=Path, default=YOLO_COMPONENT_TRAIN_DATA)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=YOLO_COMPONENT_IMAGE_SIZE)
    parser.add_argument("--batch", type=int, default=YOLO_COMPONENT_BATCH_SIZE)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=YOLO_COMPONENT_IOU_THRESHOLD)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=YOLO_COMPONENT_WORKERS)
    parser.add_argument("--output-dir", type=Path, default=YOLO_COMPONENT_METRICS_OUTPUT)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--history", type=Path, default=None)

    args = parser.parse_args()

    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")
    if args.batch < 1:
        parser.error("--batch must be at least 1")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou must be between 0 and 1")

    return args


def to_builtin(value: Any) -> Any:
    """Convert numpy/path values to JSON-compatible Python values."""

    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def box_summary(metrics: Any) -> dict[str, float]:
    """Return aggregate bbox metrics."""

    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "map50_95": float(metrics.box.map),
    }


def add_plot_outputs(
    output_dir: Path,
    ultralytics_dir: Path | None,
    report: dict[str, Any],
    history_path: Path | None,
) -> list[str]:
    """
        Save YOLO component plots in the standard evaluation folder layout.
    """

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for old_plot in plots_dir.glob("*.png"):
        old_plot.unlink()

    plot_paths: list[Path] = []
    source_dir = ultralytics_dir if ultralytics_dir is not None else output_dir

    confusion_src = source_dir / "confusion_matrix_normalized.png"
    confusion_dst = plots_dir / "confusion_matrix_normalized.png"
    if confusion_src.is_file():
        shutil.copy2(confusion_src, confusion_dst)
        plot_paths.append(confusion_dst)

    global_metrics_path = plots_dir / "global_metrics_bar.png"
    save_global_metrics_bar(
        metrics=report["bbox"],
        output_path=global_metrics_path,
        metric_names=["precision", "recall", "map50", "map75", "map50_95"],
        title="YOLO Component Global Box Metrics",
    )
    plot_paths.append(global_metrics_path)

    threshold_curves_path = plots_dir / "threshold_curves.png"
    if save_image_collage(
        source_paths=[
            source_dir / "BoxF1_curve.png",
            source_dir / "BoxP_curve.png",
            source_dir / "BoxR_curve.png",
            source_dir / "BoxPR_curve.png",
        ],
        output_path=threshold_curves_path,
        title="YOLO Component Confidence and PR Curves",
    ):
        plot_paths.append(threshold_curves_path)

    history_rows = (
        load_yolo_results_csv(history_path)
        if history_path is not None
        else []
    )
    if history_rows:
        training_history_path = plots_dir / "training_history.png"
        save_yolo_training_history(
            rows=history_rows,
            output_path=training_history_path,
            task="detect",
        )
        plot_paths.append(training_history_path)
        report["training_history"] = {
            "source": str(history_path),
            "epochs": len(history_rows),
            "plot": "plots/training_history.png",
        }

    return [str(path.relative_to(output_dir)) for path in plot_paths]


def remove_stale_ultralytics_plots(output_dir: Path) -> None:
    """
        Remove old Ultralytics root-level plot artifacts from previous runs.
    """

    patterns = [
        "Box*_curve.png",
        "confusion_matrix*.png",
        "val_batch*_labels.jpg",
        "val_batch*_pred.jpg",
        "val_batch*_labels.png",
        "val_batch*_pred.png",
    ]

    for pattern in patterns:
        for path in output_dir.glob(pattern):
            path.unlink()


def main() -> None:
    """Evaluate YOLO component detector and save a compact metrics report."""

    args = parse_args()
    model_path = args.model.resolve()
    data_path = args.data.resolve()
    output_dir = args.output_dir.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"YOLO data YAML not found: {data_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_ultralytics_plots(output_dir)
    model = YOLO(str(model_path), task=YOLO_COMPONENT_TASK)
    history_path = resolve_yolo_history_path(model_path, args.history)

    val_args: dict[str, Any] = {
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "workers": args.workers,
        "plots": args.plots,
        "verbose": True,
    }

    if args.device:
        val_args["device"] = args.device

    with tempfile.TemporaryDirectory(prefix="yolo_component_eval_") as temp_dir:
        temp_root = Path(temp_dir)
        ultralytics_dir = temp_root / "val"
        val_args.update({
            "project": str(temp_root),
            "name": ultralytics_dir.name,
            "exist_ok": True,
        })

        started = time.perf_counter()
        metrics = model.val(**val_args)
        elapsed_seconds = time.perf_counter() - started

        report = {
            "model_type": "YOLO component bbox detector",
            "evaluation_version": 2,
            "model": str(model_path),
            "data": str(data_path),
            "split": args.split,
            "settings": {
                "imgsz": args.imgsz,
                "batch": args.batch,
                "conf": args.conf,
                "iou": args.iou,
                "device": args.device or "auto",
                "plots": args.plots,
                "history": str(history_path) if history_path is not None else None,
            },
            "bbox": box_summary(metrics),
            "fitness": float(metrics.fitness),
            "speed_ms_per_image": to_builtin(metrics.speed),
            "elapsed_seconds": round(elapsed_seconds, 3),
        }

        if args.plots:
            report["plots"] = add_plot_outputs(
                output_dir=output_dir,
                ultralytics_dir=ultralytics_dir,
                report=report,
                history_path=history_path,
            )

        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(to_builtin(report), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        bbox = report["bbox"]

    print("\nYOLO component evaluation complete")
    print(f"Model: {model_path}")
    print(f"Split: {args.split}")
    print(f"Metrics: {metrics_path}")
    if args.plots:
        print(f"Plots: {output_dir / 'plots'}")
    print(
        "Box: "
        f"P={bbox['precision']:.4f} R={bbox['recall']:.4f} "
        f"mAP50={bbox['map50']:.4f} mAP50-95={bbox['map50_95']:.4f}"
    )


if __name__ == "__main__":
    main()
