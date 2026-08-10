"""Train YOLO26 segmentation for scratch instance masks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

"""Keep matplotlib/Ultralytics cache away from the user's home folder."""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ultralytics import YOLO

from configs.yolo import (
    YOLO_SCRATCH_BASE_MODEL,
    YOLO_SCRATCH_BATCH_SIZE,
    YOLO_SCRATCH_DEVICE,
    YOLO_SCRATCH_EPOCHS,
    YOLO_SCRATCH_EXIST_OK,
    YOLO_SCRATCH_IMAGE_SIZE,
    YOLO_SCRATCH_LEARNING_RATE,
    YOLO_SCRATCH_PATIENCE,
    YOLO_SCRATCH_RUN_NAME,
    YOLO_SCRATCH_SEED,
    YOLO_SCRATCH_TASK,
    YOLO_SCRATCH_TRAIN_DATA,
    YOLO_SCRATCH_TRAIN_PROJECT,
    YOLO_SCRATCH_WORKERS,
)


SPLITS = ("train", "valid", "test")
YOLO_SPLIT_ALIASES = {
    "valid": ("valid", "val"),
}


def parse_args() -> argparse.Namespace:
    """Parse YOLO scratch segmentation training arguments."""

    parser = argparse.ArgumentParser(
        description="Train YOLO scratch segmentation on a YOLO-format dataset."
    )

    """Dataset and run paths"""
    parser.add_argument("--data", type=Path, default=YOLO_SCRATCH_TRAIN_DATA)
    parser.add_argument("--model", type=str, default=YOLO_SCRATCH_BASE_MODEL)
    parser.add_argument("--project", type=Path, default=YOLO_SCRATCH_TRAIN_PROJECT)
    parser.add_argument("--name", type=str, default=YOLO_SCRATCH_RUN_NAME)

    """Training settings"""
    parser.add_argument("--imgsz", type=int, default=YOLO_SCRATCH_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=YOLO_SCRATCH_EPOCHS)
    parser.add_argument("--batch", type=int, default=YOLO_SCRATCH_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=YOLO_SCRATCH_LEARNING_RATE)
    parser.add_argument("--device", type=str, default=YOLO_SCRATCH_DEVICE)
    parser.add_argument("--workers", type=int, default=YOLO_SCRATCH_WORKERS)
    parser.add_argument("--patience", type=int, default=YOLO_SCRATCH_PATIENCE)
    parser.add_argument("--seed", type=int, default=YOLO_SCRATCH_SEED)
    parser.add_argument(
        "--exist-ok",
        action=argparse.BooleanOptionalAction,
        default=YOLO_SCRATCH_EXIST_OK,
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume an interrupted Ultralytics training run.",
    )

    """Dataset checks and post-train validation"""
    parser.add_argument(
        "--check-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate data.yaml and train/valid/test folders before training.",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run validation on the test split after training finishes.",
    )
    parser.add_argument(
        "--val-split",
        choices=SPLITS,
        default="test",
        help="Dataset split used by the post-training validation step.",
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
    """Return explicit device or auto-select the first CUDA GPU when available."""

    return device or ("0" if torch.cuda.is_available() else "cpu")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file with a clear error if PyYAML is unavailable."""

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to inspect YOLO data.yaml.") from exc

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YOLO data YAML: {path}")

    return data


def resolve_dataset_root(data_yaml: Path, data: dict[str, Any]) -> Path:
    """Resolve YOLO dataset root from data.yaml."""

    yaml_dir = data_yaml.parent
    root_value = data.get("path")

    if root_value is None:
        return yaml_dir

    root = Path(str(root_value)).expanduser()
    if not root.is_absolute():
        root = yaml_dir / root

    return root.resolve()


def resolve_split_dir(dataset_root: Path, data_yaml: Path, split_value: Any) -> Path:
    """Resolve one split directory from a YOLO data.yaml value."""

    split_path = Path(str(split_value)).expanduser()
    if split_path.is_absolute():
        return split_path

    root_candidate = (dataset_root / split_path).resolve()
    if root_candidate.exists():
        return root_candidate

    return (data_yaml.parent / split_path).resolve()


def image_dir_to_label_dir(image_dir: Path) -> Path:
    """Infer labels directory from a YOLO images directory."""

    if image_dir.name == "images":
        return image_dir.with_name("labels")

    return image_dir.parent / "labels"


def get_split_value(data: dict[str, Any], split: str) -> Any:
    """
        Return one split path from data.yaml.

        Args:
            data : parsed YOLO data.yaml dictionary
            split: train, valid, or test

        Returns:
            split_value: image directory value for the requested split

        Notes:
            Ultralytics uses `val` as the canonical validation key. This project
            displays the folder as `valid`, so both `valid` and `val` are accepted.
    """

    for key in YOLO_SPLIT_ALIASES.get(split, (split,)):
        if key in data:
            return data[key]

    raise KeyError(split)


def check_yolo_dataset(data_yaml: Path) -> None:
    """Validate YOLO segmentation dataset folders before training."""

    if not data_yaml.is_file():
        raise FileNotFoundError(f"YOLO data YAML not found: {data_yaml}")

    data = load_yaml(data_yaml)
    dataset_root = resolve_dataset_root(data_yaml, data)

    missing_keys = []
    for split in SPLITS:
        try:
            get_split_value(data, split)
        except KeyError:
            missing_keys.append(split)

    if missing_keys:
        raise ValueError(f"data.yaml missing split keys: {', '.join(missing_keys)}")

    names = data.get("names")
    if not names:
        raise ValueError("data.yaml missing class names.")

    print("Checking YOLO scratch dataset")
    print(f"data.yaml: {data_yaml}")
    print(f"dataset root: {dataset_root}")
    print(f"classes: {names}")

    for split in SPLITS:
        image_dir = resolve_split_dir(dataset_root, data_yaml, get_split_value(data, split))
        label_dir = image_dir_to_label_dir(image_dir)

        if not image_dir.is_dir():
            raise FileNotFoundError(f"{split} images directory not found: {image_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"{split} labels directory not found: {label_dir}")

        images = sorted(path for path in image_dir.iterdir() if path.is_file())
        labels = sorted(path for path in label_dir.iterdir() if path.suffix == ".txt")
        print(f"- {split}: images={len(images)} | labels={len(labels)}")

        if not images:
            raise RuntimeError(f"No images found in {image_dir}")
        if not labels:
            raise RuntimeError(f"No labels found in {label_dir}")


def train_yolo(args: argparse.Namespace) -> tuple[Path, Path]:
    """Run Ultralytics YOLO segmentation training and return checkpoint paths."""

    data_path = args.data.resolve()
    project_path = args.project.resolve()
    device = resolve_device(args.device)

    if args.check_data:
        check_yolo_dataset(data_path)
    elif not data_path.is_file():
        raise FileNotFoundError(f"YOLO data YAML not found: {data_path}")

    print("\nTraining YOLO scratch segmentation")
    print(f"Base model: {args.model}")
    print(f"Data: {data_path}")
    print(f"Project: {project_path}")
    print(f"Run name: {args.name}")
    print(f"Device: {device}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch: {args.batch}")
    print(f"Epochs: {args.epochs}")

    model = YOLO(args.model, task=YOLO_SCRATCH_TASK)
    model.train(
        data=str(data_path),
        task=YOLO_SCRATCH_TASK,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        project=str(project_path),
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=args.pretrained,
        plots=args.plots,
        save=args.save,
        device=device,
        seed=args.seed,
        lr0=args.lr,
        resume=args.resume,
    )

    run_dir = project_path / args.name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    print(f"\nRun dir: {run_dir}")
    print(f"Best checkpoint: {best_pt} | exists={best_pt.is_file()}")
    print(f"Last checkpoint: {last_pt} | exists={last_pt.is_file()}")

    return best_pt, last_pt


def validate_best_checkpoint(args: argparse.Namespace, best_pt: Path) -> None:
    """Validate the best checkpoint on the requested split."""

    if not best_pt.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {best_pt}")

    device = resolve_device(args.device)
    print(f"\nValidating best checkpoint on split: {args.val_split}")

    best_model = YOLO(str(best_pt), task=YOLO_SCRATCH_TASK)
    metrics = best_model.val(
        data=str(args.data.resolve()),
        task=YOLO_SCRATCH_TASK,
        split=args.val_split,
        imgsz=args.imgsz,
        batch=args.batch,
        plots=args.plots,
        device=device,
    )

    print(metrics)


def main() -> None:
    """Train YOLO scratch segmentation."""

    args = parse_args()
    best_pt, _last_pt = train_yolo(args)

    if args.validate:
        validate_best_checkpoint(args, best_pt)


if __name__ == "__main__":
    main()
