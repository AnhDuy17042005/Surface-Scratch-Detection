"""
    Benchmark YOLO PyTorch and OpenVINO inference latency.

    Metric:
        Average latency in milliseconds per image.

    Run:
        python -m src.openvino.evaluation --max-images 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Callable

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

"""Matplotlib cache directory for Linux/server environments."""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from configs.data import IMAGE_EXTENSIONS
from configs.path import METRICS_DIR
from configs.yolo import (
    YOLO_COMPONENT_IMAGE_SIZE,
    YOLO_COMPONENT_MODEL,
    YOLO_COMPONENT_TASK,
    YOLO_COMPONENT_TRAIN_DATA,
    YOLO_SCRATCH_IMAGE_SIZE,
    YOLO_SCRATCH_MODEL,
    YOLO_SCRATCH_TASK,
    YOLO_SCRATCH_TRAIN_DATA,
)


DEFAULT_OUTPUT_DIR = METRICS_DIR / "openvino"


def default_openvino_model(pytorch_model: Path | None = None) -> Path:
    """
        Build default YOLO OpenVINO model directory from the PyTorch checkpoint.
    """

    base_model = pytorch_model if pytorch_model is not None else YOLO_SCRATCH_MODEL
    return base_model.with_name("best_openvino_model")


def default_image_dir(task: str) -> Path:
    """
        Return the default YOLO validation image directory.
    """

    if task == "detect":
        return YOLO_COMPONENT_TRAIN_DATA.parent / "valid" / "images"
    return YOLO_SCRATCH_TRAIN_DATA.parent / "test" / "images"


def parse_args() -> argparse.Namespace:
    """
        Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Benchmark YOLO PyTorch vs OpenVINO average inference latency."
    )

    """Input arguments"""
    parser.add_argument("--task", choices=("detect", "segment"), default="segment")
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None)

    """Runtime arguments"""
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)

    """Model arguments"""
    parser.add_argument("--pytorch-model", type=Path, default=None)
    parser.add_argument("--openvino-model", type=Path, default=None)

    """Output arguments"""
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()

    if args.imgsz is None:
        args.imgsz = YOLO_COMPONENT_IMAGE_SIZE if args.task == "detect" else YOLO_SCRATCH_IMAGE_SIZE
    if args.image_dir is None:
        args.image_dir = default_image_dir(args.task)
    if args.pytorch_model is None:
        args.pytorch_model = YOLO_COMPONENT_MODEL if args.task == "detect" else YOLO_SCRATCH_MODEL
    if args.openvino_model is None:
        args.openvino_model = default_openvino_model(args.pytorch_model)

    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be at least 1")
    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    return args


def check_model_path(path: Path, label: str) -> None:
    """
        Validate model checkpoint or model directory.
    """

    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def find_images(image_dir: Path, max_images: int | None) -> list[Path]:
    """
        Find benchmark images in a directory.
    """

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if max_images is not None:
        images = images[:max_images]

    if not images:
        raise ValueError(f"No benchmark images found in: {image_dir}")

    return images


def load_images(image_paths: list[Path]) -> list[tuple[Path, object]]:
    """
        Load images once so benchmark time does not include disk I/O.
    """

    loaded = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        loaded.append((image_path, image))

    return loaded


def benchmark_runner(
    backend: str,
    runner: Callable[[object], None],
    images: list[tuple[Path, object]],
    warmup: int,
    repeat: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    """
        Benchmark one YOLO backend.
    """

    for index in range(warmup):
        _, image = images[index % len(images)]
        runner(image)

    rows: list[dict[str, float | int | str]] = []
    latencies: list[float] = []

    for repeat_index in range(repeat):
        for image_path, image in images:
            started = time.perf_counter()
            runner(image)
            latency_ms = (time.perf_counter() - started) * 1000.0

            latencies.append(latency_ms)
            rows.append(
                {
                    "backend": backend,
                    "repeat": repeat_index + 1,
                    "image": str(image_path),
                    "latency_ms": round(latency_ms, 4),
                }
            )

    summary = {
        "backend": backend,
        "image_count": len(images),
        "repeat": repeat,
        "sample_count": len(latencies),
        "average_latency_ms_per_image": round(mean(latencies), 4),
    }

    return summary, rows


def benchmark_yolo(
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, float | int | str]]]:
    """
        Benchmark YOLO PyTorch checkpoint against YOLO OpenVINO model.
    """

    from ultralytics import YOLO

    check_model_path(args.pytorch_model, "YOLO PyTorch model")
    check_model_path(args.openvino_model, "YOLO OpenVINO model")

    image_paths = find_images(args.image_dir, args.max_images)
    images = load_images(image_paths)

    task = YOLO_COMPONENT_TASK if args.task == "detect" else YOLO_SCRATCH_TASK
    pytorch_model = YOLO(str(args.pytorch_model), task=task)
    openvino_model = YOLO(str(args.openvino_model), task=task)

    def run_pytorch(image: object) -> None:
        pytorch_model.predict(
            source=image,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

    def run_openvino(image: object) -> None:
        openvino_model.predict(
            source=image,
            imgsz=args.imgsz,
            verbose=False,
        )

    pytorch_summary, pytorch_rows = benchmark_runner(
        backend="pytorch",
        runner=run_pytorch,
        images=images,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    openvino_summary, openvino_rows = benchmark_runner(
        backend="openvino",
        runner=run_openvino,
        images=images,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    speedup = (
        pytorch_summary["average_latency_ms_per_image"]
        / openvino_summary["average_latency_ms_per_image"]
    )

    return (
        {
            "metric": "average_latency_ms_per_image",
            "image_dir": str(args.image_dir),
            "pytorch": pytorch_summary,
            "openvino": openvino_summary,
            "speedup_ratio": round(float(speedup), 4),
        },
        pytorch_rows + openvino_rows,
    )


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    """
        Write per-image benchmark rows.
    """

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """
        Run YOLO benchmark and save results.
    """

    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, rows = benchmark_yolo(args)
    report = {
        "model_name": f"yolo_{args.task}",
        "task": args.task,
        "metrics_root": str(output_dir),
        "results": summary,
    }

    summary_path = output_dir / "latency_summary.json"
    rows_path = output_dir / "latency_rows.csv"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    write_csv(rows_path, rows)

    pytorch_latency = summary["pytorch"]["average_latency_ms_per_image"]
    openvino_latency = summary["openvino"]["average_latency_ms_per_image"]
    speedup = summary["speedup_ratio"]

    print(f"Latency summary saved: {summary_path}")
    print(f"Per-image latency rows saved: {rows_path}")
    print(
        f"YOLO: PyTorch={pytorch_latency:.2f} ms/image | "
        f"OpenVINO={openvino_latency:.2f} ms/image | "
        f"Speedup={speedup:.2f}x"
    )


if __name__ == "__main__":
    main()
