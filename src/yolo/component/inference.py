"""Run YOLO component detection and save bounding-box outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[3]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

"""Keep matplotlib/Ultralytics cache away from the user's home folder."""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ultralytics import YOLO

from configs.data import IMAGE_EXTENSIONS
from configs.yolo import (
    YOLO_COMPONENT_CONFIDENCE_THRESHOLD,
    YOLO_COMPONENT_DEFAULT_IMAGE,
    YOLO_COMPONENT_IMAGE_SIZE,
    YOLO_COMPONENT_INFERENCE_OUTPUT,
    YOLO_COMPONENT_IOU_THRESHOLD,
    YOLO_COMPONENT_MODEL,
    YOLO_COMPONENT_TASK,
)


def parse_args() -> argparse.Namespace:
    """Parse inference arguments."""

    parser = argparse.ArgumentParser(
        description="Detect component bounding boxes for YOLO scratch ROI extraction."
    )

    parser.add_argument("--source", type=Path, default=YOLO_COMPONENT_DEFAULT_IMAGE)
    parser.add_argument("--model", type=Path, default=YOLO_COMPONENT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=YOLO_COMPONENT_INFERENCE_OUTPUT)
    parser.add_argument("--conf", type=float, default=YOLO_COMPONENT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--iou", type=float, default=YOLO_COMPONENT_IOU_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=YOLO_COMPONENT_IMAGE_SIZE)

    args = parser.parse_args()

    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou must be between 0 and 1")
    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")

    return args


def collect_images(source: Path) -> list[Path]:
    """Collect one image or all supported images from a folder."""

    if source.is_file():
        return [source]

    if not source.is_dir():
        raise FileNotFoundError(f"Source not found: {source}")

    images = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise ValueError(f"No supported images found in: {source}")

    return images


def main() -> None:
    """Run component detection."""

    args = parse_args()
    model_path = args.model.resolve()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path), task=YOLO_COMPONENT_TASK)
    images = collect_images(source)

    for image_path in images:
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )[0]

        plotted = result.plot(boxes=True, labels=True, conf=True)
        output_image = output_dir / f"{image_path.stem}_component.jpg"

        if not cv2.imwrite(str(output_image), plotted):
            raise RuntimeError(f"Failed to save prediction image: {output_image}")

    print(f"Images: {len(images)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
