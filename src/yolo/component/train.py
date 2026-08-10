"""Train YOLO26n for component bounding-box detection."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

"""Keep matplotlib/Ultralytics cache away from the user's home folder."""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ultralytics import YOLO

from configs.yolo import (
    YOLO_COMPONENT_BASE_MODEL,
    YOLO_COMPONENT_BATCH_SIZE,
    YOLO_COMPONENT_DEVICE,
    YOLO_COMPONENT_EPOCHS,
    YOLO_COMPONENT_EXIST_OK,
    YOLO_COMPONENT_IMAGE_SIZE,
    YOLO_COMPONENT_LEARNING_RATE,
    YOLO_COMPONENT_PATIENCE,
    YOLO_COMPONENT_RUN_NAME,
    YOLO_COMPONENT_SEED,
    YOLO_COMPONENT_TASK,
    YOLO_COMPONENT_TRAIN_DATA,
    YOLO_COMPONENT_TRAIN_PROJECT,
    YOLO_COMPONENT_WORKERS,
)


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""

    parser = argparse.ArgumentParser(
        description="Train YOLO26n to detect component bounding boxes."
    )

    parser.add_argument("--data", type=Path, default=YOLO_COMPONENT_TRAIN_DATA)
    parser.add_argument("--model", type=str, default=YOLO_COMPONENT_BASE_MODEL)
    parser.add_argument("--imgsz", type=int, default=YOLO_COMPONENT_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=YOLO_COMPONENT_EPOCHS)
    parser.add_argument("--batch", type=int, default=YOLO_COMPONENT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=YOLO_COMPONENT_LEARNING_RATE)
    parser.add_argument("--device", type=str, default=YOLO_COMPONENT_DEVICE)
    parser.add_argument("--workers", type=int, default=YOLO_COMPONENT_WORKERS)
    parser.add_argument("--patience", type=int, default=YOLO_COMPONENT_PATIENCE)
    parser.add_argument("--project", type=Path, default=YOLO_COMPONENT_TRAIN_PROJECT)
    parser.add_argument("--name", type=str, default=YOLO_COMPONENT_RUN_NAME)
    parser.add_argument("--seed", type=int, default=YOLO_COMPONENT_SEED)
    parser.add_argument(
        "--exist-ok",
        action=argparse.BooleanOptionalAction,
        default=YOLO_COMPONENT_EXIST_OK,
    )

    args = parser.parse_args()

    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch < 1:
        parser.error("--batch must be at least 1")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.lr <= 0:
        parser.error("--lr must be positive")

    return args


def resolve_device(device: str | None) -> str:
    """Return explicit device or auto-select CUDA when available."""

    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    """Train YOLO component detector."""

    args = parse_args()
    data_path = args.data.resolve()

    if not data_path.is_file():
        raise FileNotFoundError(f"YOLO data YAML not found: {data_path}")

    model = YOLO(args.model, task=YOLO_COMPONENT_TASK)
    device = resolve_device(args.device)

    print(f"Training YOLO component detector")
    print(f"Base model: {args.model}")
    print(f"Data: {data_path}")
    print(f"Device: {device}")

    model.train(
        data=str(data_path),
        task=YOLO_COMPONENT_TASK,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        lr0=args.lr,
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        exist_ok=args.exist_ok,
    )

    print(f"Best checkpoint: {args.project / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
