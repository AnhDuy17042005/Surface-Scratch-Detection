"""
    Standalone YOLO segmentation inference for scratch masks.

    This pipeline is isolated under src/yolo/scratch so scratch-specific
    YOLO scripts are managed separately from the component detector.

    Default flow:
        1. Load a full-size image.
        2. Run YOLO26-seg on 512x512 sliding-window patches.
        3. Merge patch masks back into one full-size scratch mask.
        4. Save a red overlay image, binary mask, and optional raw prediction.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

"""Keep Ultralytics/Matplotlib cache out of the project tree."""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ultralytics import YOLO

from configs.data import IMAGE_EXTENSIONS
from configs.yolo import (
    YOLO_SCRATCH_CONFIDENCE_THRESHOLD,
    YOLO_SCRATCH_DEFAULT_IMAGE,
    YOLO_SCRATCH_IMAGE_SIZE,
    YOLO_SCRATCH_INFERENCE_OUTPUT,
    YOLO_SCRATCH_IOU_THRESHOLD,
    YOLO_SCRATCH_MASK_THRESHOLD,
    YOLO_SCRATCH_MODEL,
    YOLO_SCRATCH_OVERLAY_ALPHA,
    YOLO_SCRATCH_TILE_OVERLAP,
    YOLO_SCRATCH_TILE_SIZE,
)

DEFAULT_MODEL = YOLO_SCRATCH_MODEL
DEFAULT_IMAGE = YOLO_SCRATCH_DEFAULT_IMAGE
DEFAULT_OUTPUT_DIR = YOLO_SCRATCH_INFERENCE_OUTPUT


@dataclass(frozen=True)
class Tile:
    """
        One sliding-window tile region in the original image.
    """

    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class InferenceStats:
    """
        Runtime and prediction summary for one image.
    """

    image_name: str
    width: int
    height: int
    tiles: int
    instances: int
    positive_pixels: int
    yolo_seconds: float
    total_seconds: float
    overlay_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    """
        Parse command-line arguments for the demo pipeline.
    """

    parser = argparse.ArgumentParser(
        description="Run standalone YOLO26-seg scratch inference."
    )

    """Input/output paths"""
    parser.add_argument("--source", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    """YOLO inference settings"""
    parser.add_argument("--imgsz", type=int, default=YOLO_SCRATCH_IMAGE_SIZE)
    parser.add_argument("--conf", type=float, default=YOLO_SCRATCH_CONFIDENCE_THRESHOLD)
    parser.add_argument("--iou", type=float, default=YOLO_SCRATCH_IOU_THRESHOLD)
    parser.add_argument("--device", type=str, default="cpu")

    """Patch inference settings"""
    parser.add_argument(
        "--mode",
        choices=("sliding", "full"),
        default="sliding",
        help="Use sliding for patch-trained YOLO, or full for one resized full-image pass.",
    )
    parser.add_argument("--tile-size", type=int, default=YOLO_SCRATCH_TILE_SIZE)
    parser.add_argument("--overlap", type=float, default=YOLO_SCRATCH_TILE_OVERLAP)

    """Output controls"""
    parser.add_argument("--mask-threshold", type=float, default=YOLO_SCRATCH_MASK_THRESHOLD)
    parser.add_argument("--alpha", type=float, default=YOLO_SCRATCH_OVERLAY_ALPHA)
    parser.add_argument("--draw-boxes", action="store_true")
    parser.add_argument("--save-raw", action="store_true")

    args = parser.parse_args()

    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be in [0, 1]")
    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou must be in [0, 1]")
    if args.tile_size < 32:
        parser.error("--tile-size must be at least 32")
    if not 0.0 <= args.overlap < 1.0:
        parser.error("--overlap must be in [0, 1)")
    if not 0.0 <= args.mask_threshold <= 1.0:
        parser.error("--mask-threshold must be in [0, 1]")
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")

    return args


def resolve_project_path(path: Path) -> Path:
    """
        Resolve relative paths from the project root.
    """

    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_ultralytics_model_path(model_path: Path) -> Path:
    """
        Return the path format expected by Ultralytics for each backend.

        Ultralytics loads OpenVINO exports from the exported directory
        (for example best_openvino_model), not from the inner best.xml file.
    """

    if model_path.is_file() and model_path.parent.name.endswith("_openvino_model"):
        return model_path.parent
    return model_path


def collect_images(source: Path) -> list[Path]:
    """
        Collect a single image or all supported images from a folder.
    """

    source = resolve_project_path(source)

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
        raise RuntimeError(f"No supported images found in: {source}")

    return images


def load_image(image_path: Path) -> np.ndarray:
    """
        Read one image as BGR.
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    """
        Save an image and raise a clear error on failure.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Cannot write image: {path}")


def sliding_coords(size: int, tile_size: int, stride: int) -> list[int]:
    """
        Build edge-aligned sliding-window coordinates for one axis.
    """

    if size <= tile_size:
        return [0]

    coords = list(range(0, size - tile_size + 1, stride))
    last = size - tile_size
    if coords[-1] != last:
        coords.append(last)
    return coords


def build_tiles(image_shape: tuple[int, int, int], tile_size: int, overlap: float) -> list[Tile]:
    """
        Build full-image sliding-window tiles.
    """

    height, width = image_shape[:2]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    xs = sliding_coords(width, tile_size, stride)
    ys = sliding_coords(height, tile_size, stride)

    tiles: list[Tile] = []
    for y1 in ys:
        for x1 in xs:
            x2 = min(x1 + tile_size, width)
            y2 = min(y1 + tile_size, height)
            tiles.append(Tile(x1=x1, y1=y1, x2=x2, y2=y2))

    return tiles


def result_to_mask(result, width: int, height: int, threshold: float) -> tuple[np.ndarray, int]:
    """
        Convert one Ultralytics segmentation result to a binary mask.
    """

    mask = np.zeros((height, width), dtype=np.uint8)
    if result.masks is None or result.masks.data is None:
        return mask, 0

    mask_data = result.masks.data.detach().cpu().numpy()
    if mask_data.size == 0:
        return mask, 0

    instance_count = int(mask_data.shape[0])
    combined = np.max(mask_data, axis=0)

    if combined.shape != (height, width):
        combined = cv2.resize(
            combined.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    mask[combined >= threshold] = 255
    return mask, instance_count


def draw_result_boxes(
    image: np.ndarray,
    result,
    offset_x: int = 0,
    offset_y: int = 0,
) -> None:
    """
        Draw YOLO boxes on an image in-place.
    """

    if result.boxes is None:
        return

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else []

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int).tolist()
        x1 += offset_x
        x2 += offset_x
        y1 += offset_y
        y2 += offset_y

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if len(confs) > index:
            cv2.putText(
                image,
                f"{float(confs[index]):.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )


def make_red_overlay(image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """
        Overlay a red scratch mask on top of the original image.
    """

    overlay = image.copy()
    red_layer = np.zeros_like(image)
    red_layer[:, :, 2] = 255

    positive = mask > 0
    overlay[positive] = cv2.addWeighted(
        image[positive],
        1.0 - alpha,
        red_layer[positive],
        alpha,
        0,
    )
    return overlay


def predict_full_image(
    model: YOLO,
    image: np.ndarray,
    args: argparse.Namespace,
    box_canvas: np.ndarray | None,
) -> tuple[np.ndarray, int, float]:
    """
        Run one full-image YOLO prediction.
    """

    start = time.perf_counter()
    result = model.predict(
        source=image,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        task="segment",
        verbose=False,
    )[0]
    seconds = time.perf_counter() - start

    height, width = image.shape[:2]
    mask, instances = result_to_mask(
        result=result,
        width=width,
        height=height,
        threshold=args.mask_threshold,
    )

    if box_canvas is not None:
        draw_result_boxes(box_canvas, result)

    return mask, instances, seconds


def predict_sliding_image(
    model: YOLO,
    image: np.ndarray,
    args: argparse.Namespace,
    box_canvas: np.ndarray | None,
) -> tuple[np.ndarray, int, float, int]:
    """
        Run patch-based YOLO prediction and paste masks into a full-size image.
    """

    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    instance_count = 0
    yolo_seconds = 0.0
    tiles = build_tiles(
        image_shape=image.shape,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )

    for tile in tiles:
        tile_image = image[tile.y1:tile.y2, tile.x1:tile.x2]

        start = time.perf_counter()
        result = model.predict(
            source=tile_image,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            task="segment",
            verbose=False,
        )[0]
        yolo_seconds += time.perf_counter() - start

        tile_h, tile_w = tile_image.shape[:2]
        tile_mask, tile_instances = result_to_mask(
            result=result,
            width=tile_w,
            height=tile_h,
            threshold=args.mask_threshold,
        )

        roi = full_mask[tile.y1:tile.y2, tile.x1:tile.x2]
        np.maximum(roi, tile_mask, out=roi)
        instance_count += tile_instances

        if box_canvas is not None:
            draw_result_boxes(
                image=box_canvas,
                result=result,
                offset_x=tile.x1,
                offset_y=tile.y1,
            )

    return full_mask, instance_count, yolo_seconds, len(tiles)


def predict_image(model: YOLO, image_path: Path, args: argparse.Namespace) -> InferenceStats:
    """
        Predict one image and save demo outputs.
    """

    total_start = time.perf_counter()
    image = load_image(image_path)
    box_canvas = image.copy() if args.draw_boxes else None

    if args.mode == "full":
        mask, instances, yolo_seconds = predict_full_image(
            model=model,
            image=image,
            args=args,
            box_canvas=box_canvas,
        )
        tile_count = 1
    else:
        mask, instances, yolo_seconds, tile_count = predict_sliding_image(
            model=model,
            image=image,
            args=args,
            box_canvas=box_canvas,
        )

    overlay = make_red_overlay(image, mask, args.alpha)
    if box_canvas is not None:
        positive = mask > 0
        box_canvas[positive] = overlay[positive]
        overlay = box_canvas

    output_dir = resolve_project_path(args.output_dir)
    overlay_path = output_dir / f"{image_path.stem}_yolo_seg_overlay.jpg"
    mask_path = output_dir / f"{image_path.stem}_yolo_seg_mask.png"

    save_image(overlay_path, overlay)
    save_image(mask_path, mask)

    if args.save_raw:
        raw_path = output_dir / f"{image_path.stem}_raw.jpg"
        save_image(raw_path, image)

    total_seconds = time.perf_counter() - total_start
    return InferenceStats(
        image_name=image_path.name,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        tiles=tile_count,
        instances=instances,
        positive_pixels=int((mask > 0).sum()),
        yolo_seconds=yolo_seconds,
        total_seconds=total_seconds,
        overlay_path=overlay_path,
        mask_path=mask_path,
    )


def write_summary(stats: Iterable[InferenceStats], output_dir: Path) -> None:
    """
        Save a CSV summary for all processed images.
    """

    rows = list(stats)
    if not rows:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "image",
                "width",
                "height",
                "tiles",
                "instances",
                "positive_pixels",
                "yolo_seconds",
                "total_seconds",
                "overlay",
                "mask",
            ),
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "image": row.image_name,
                    "width": row.width,
                    "height": row.height,
                    "tiles": row.tiles,
                    "instances": row.instances,
                    "positive_pixels": row.positive_pixels,
                    "yolo_seconds": f"{row.yolo_seconds:.4f}",
                    "total_seconds": f"{row.total_seconds:.4f}",
                    "overlay": row.overlay_path,
                    "mask": row.mask_path,
                }
            )


def main() -> None:
    """
        Run the standalone YOLO segmentation demo pipeline.
    """

    args = parse_args()

    model_path = normalize_ultralytics_model_path(resolve_project_path(args.model))
    output_dir = resolve_project_path(args.output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"YOLO segmentation model not found: {model_path}")

    images = collect_images(args.source)
    model = YOLO(str(model_path), task="segment")

    stats: list[InferenceStats] = []
    for image_path in images:
        row = predict_image(model=model, image_path=image_path, args=args)
        stats.append(row)
        print(
            f"{row.image_name} | tiles={row.tiles} instances={row.instances} "
            f"positive_pixels={row.positive_pixels} "
            f"yolo={row.yolo_seconds:.2f}s total={row.total_seconds:.2f}s"
        )
        print(f"  overlay: {row.overlay_path}")

    write_summary(stats=stats, output_dir=output_dir)
    print(f"\nProcessed images: {len(stats)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
