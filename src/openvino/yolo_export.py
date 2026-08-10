"""
    Export YOLO segmentation checkpoint to OpenVINO.

    Default:
        models/yolo/scratch_yolo26n_seg/weights/best.pt

    Run:
        python -m src.openvino.yolo_export
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


"""Project Root"""
PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

"""Matplotlib Cache Directory"""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ultralytics import YOLO

from configs.yolo import YOLO_SCRATCH_IMAGE_SIZE, YOLO_SCRATCH_MODEL


def default_model_path() -> Path:
    """
        Build default YOLO PyTorch checkpoint path.
    """

    return YOLO_SCRATCH_MODEL


def default_output_path(model_path: Path) -> Path:
    """
        Build default OpenVINO output directory from checkpoint path.
    """

    return model_path.with_name("best_openvino_model")


def parse_args() -> argparse.Namespace:
    """
        Parse command line arguments for YOLO OpenVINO export.
    """

    parser = argparse.ArgumentParser(
        description="Export trained YOLO scratch segmentation model to OpenVINO."
    )

    """Input Argument"""
    parser.add_argument("--model", type=Path, default=default_model_path())

    """Export Arguments"""
    parser.add_argument("--imgsz", type=int, default=YOLO_SCRATCH_IMAGE_SIZE)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    """Output Argument"""
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    """Validate Image Size"""
    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")

    """Validate Batch Size"""
    if args.batch < 1:
        parser.error("--batch must be at least 1")

    return args


def export_yolo_to_openvino(
    model_path: Path,
    imgsz: int,
    batch: int,
    device: str,
    dynamic: bool,
    output_path: Path | None,
) -> Path:
    """
        Export trained YOLO segmentation checkpoint to OpenVINO format.

        Returns:
            exported_path: OpenVINO model directory.
    """

    """Check YOLO Checkpoint"""
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {model_path}")

    """Load YOLO Model"""
    model = YOLO(str(model_path))

    """Export YOLO To OpenVINO"""
    exported_path = Path(
        model.export(
            format="openvino",
            imgsz=imgsz,
            batch=batch,
            device=device,
            dynamic=dynamic,
        )
    )

    """Move Exported Directory To Custom Output Path"""
    if output_path is not None and exported_path.resolve() != output_path.resolve():
        import shutil

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            shutil.rmtree(output_path)

        shutil.move(str(exported_path), str(output_path))
        exported_path = output_path

    return exported_path


def main() -> None:
    """
        Main function for YOLO OpenVINO export.
    """

    args = parse_args()

    """Export YOLO To OpenVINO"""
    output_path = export_yolo_to_openvino(
        model_path=args.model,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        dynamic=args.dynamic,
        output_path=args.output,
    )

    """Print Export Summary"""
    print(f"YOLO OpenVINO exported: {output_path}")


if __name__ == "__main__":
    main()
