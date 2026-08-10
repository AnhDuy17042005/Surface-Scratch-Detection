"""
    Convert binary scratch masks into YOLO segmentation labels.

    Input:
        data/scratch_patches/
          train/
            images/
            masks/
          valid/
            images/
            masks/
          test/
            images/
            masks/

    Output:
        data/scratch_yolo_seg/
          train/
            images/
            labels/
          valid/
            images/
            labels/
          test/
            images/
            labels/
          data.yaml
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.data import IMAGE_EXTENSIONS, MASK_EXTENSION, SCRATCH_TRAIN_DATASET


SPLIT_MAP = {
    "train": "train",
    "valid": "valid",
    "test": "test",
}
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "scratch_yolo_seg"
OUTPUT_IMAGE_EXTENSION = ".png"


@dataclass(frozen=True)
class ConvertStats:
    """
        Summary for one converted split.
    """

    split: str
    image_count: int
    positive_count: int
    negative_count: int
    polygon_count: int
    skipped_contours: int


def resolve_project_path(path: Path) -> Path:
    """
        Resolve relative paths from the project root.
    """

    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def collect_images(image_dir: Path) -> list[Path]:
    """
        Collect supported image files from a split image folder.
    """

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(f"No supported images found in: {image_dir}")

    return image_paths


def prepare_output(dst_root: Path, overwrite: bool, dry_run: bool) -> None:
    """
        Create YOLO output folders or replace them when requested.
    """

    if dry_run:
        return

    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {dst_root}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(dst_root)

    for yolo_split in SPLIT_MAP.values():
        (dst_root / yolo_split / "images").mkdir(parents=True, exist_ok=True)
        (dst_root / yolo_split / "labels").mkdir(parents=True, exist_ok=True)


def write_or_link_png_image(src_path: Path, dst_path: Path, image_mode: str) -> None:
    """
        Write one image into the YOLO image folder as PNG.
    """

    if image_mode == "copy":
        image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {src_path}")
        if not cv2.imwrite(str(dst_path), image):
            raise RuntimeError(f"Failed to save image: {dst_path}")
        return

    if image_mode == "symlink":
        if src_path.suffix.lower() != OUTPUT_IMAGE_EXTENSION:
            raise ValueError(
                "Symlink image mode requires PNG source images because "
                f"YOLO output images are named with {OUTPUT_IMAGE_EXTENSION}: "
                f"{src_path}"
            )
        dst_path.symlink_to(src_path.resolve())
        return

    raise ValueError(f"Unsupported image mode: {image_mode}")


def contour_to_polygon(
    contour: np.ndarray,
    width: int,
    height: int,
    epsilon_ratio: float,
    min_points: int,
) -> list[float] | None:
    """
        Convert one OpenCV contour to a normalized YOLO polygon.
    """

    """Optionally simplify the contour, but default keeps scratch detail."""
    if epsilon_ratio > 0:
        epsilon = epsilon_ratio * cv2.arcLength(contour, closed=True)
        contour = cv2.approxPolyDP(contour, epsilon, closed=True)

    points = contour.reshape(-1, 2)

    """Remove repeated neighboring points to avoid malformed labels."""
    unique_points: list[tuple[int, int]] = []
    for x, y in points.tolist():
        point = (int(x), int(y))
        if not unique_points or unique_points[-1] != point:
            unique_points.append(point)

    if len(unique_points) > 1 and unique_points[0] == unique_points[-1]:
        unique_points.pop()

    if len(unique_points) < min_points:
        return None

    polygon: list[float] = []
    for x, y in unique_points:
        normalized_x = min(max(float(x) / float(width), 0.0), 1.0)
        normalized_y = min(max(float(y) / float(height), 0.0), 1.0)
        polygon.extend([normalized_x, normalized_y])

    return polygon


def mask_to_polygons(
    mask: np.ndarray,
    class_id: int,
    min_area: float,
    epsilon_ratio: float,
    min_points: int,
) -> tuple[list[list[float]], int]:
    """
        Convert one binary mask into YOLO segmentation rows.
    """

    height, width = mask.shape[:2]

    """Binarize 0/255 masks and find connected scratch components."""
    binary = np.zeros_like(mask, dtype=np.uint8)
    binary[mask > 0] = 255

    contours, _hierarchy = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    rows: list[list[float]] = []
    skipped_contours = 0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            skipped_contours += 1
            continue

        polygon = contour_to_polygon(
            contour=contour,
            width=width,
            height=height,
            epsilon_ratio=epsilon_ratio,
            min_points=min_points,
        )
        if polygon is None:
            skipped_contours += 1
            continue

        rows.append([float(class_id), *polygon])

    return rows, skipped_contours


def format_label_row(row: list[float]) -> str:
    """
        Format one YOLO segmentation label row.
    """

    class_id = int(row[0])
    coords = " ".join(f"{value:.6f}" for value in row[1:])
    return f"{class_id} {coords}"


def convert_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    class_id: int,
    min_area: float,
    epsilon_ratio: float,
    min_points: int,
    image_mode: str,
    dry_run: bool,
) -> ConvertStats:
    """
        Convert one train/valid/test split to YOLO segmentation format.
    """

    yolo_split = SPLIT_MAP[split]
    image_dir = src_root / split / "images"
    mask_dir = src_root / split / "masks"

    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_paths = collect_images(image_dir)
    positive_count = 0
    polygon_count = 0
    skipped_contours = 0

    out_image_dir = dst_root / yolo_split / "images"
    out_label_dir = dst_root / yolo_split / "labels"

    for image_path in image_paths:
        mask_path = mask_dir / f"{image_path.stem}{MASK_EXTENSION}"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found for image: {image_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image and mask size mismatch: {image_path.name} "
                f"{image.shape[:2]} != {mask.shape[:2]}"
            )

        rows, skipped = mask_to_polygons(
            mask=mask,
            class_id=class_id,
            min_area=min_area,
            epsilon_ratio=epsilon_ratio,
            min_points=min_points,
        )

        polygon_count += len(rows)
        skipped_contours += skipped
        if rows:
            positive_count += 1

        if dry_run:
            continue

        write_or_link_png_image(
            src_path=image_path,
            dst_path=out_image_dir / f"{image_path.stem}{OUTPUT_IMAGE_EXTENSION}",
            image_mode=image_mode,
        )

        label_path = out_label_dir / f"{image_path.stem}.txt"
        label_text = "\n".join(format_label_row(row) for row in rows)
        if label_text:
            label_text += "\n"
        label_path.write_text(label_text, encoding="utf-8")

    return ConvertStats(
        split=split,
        image_count=len(image_paths),
        positive_count=positive_count,
        negative_count=len(image_paths) - positive_count,
        polygon_count=polygon_count,
        skipped_contours=skipped_contours,
    )


def write_data_yaml(
    dst_root: Path,
    class_id: int,
    class_name: str,
    dry_run: bool,
) -> None:
    """
        Write Ultralytics data.yaml for YOLO segmentation training.
    """

    if dry_run:
        return

    yaml_text = "\n".join(
        [
            "path: .",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "",
            "names:",
            f"  {class_id}: {class_name}",
            "",
        ]
    )
    (dst_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def write_metadata(
    dst_root: Path,
    stats: list[ConvertStats],
    dry_run: bool,
) -> None:
    """
        Save conversion summary for reproducibility.
    """

    if dry_run:
        return

    metadata_path = dst_root / "conversion_summary.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "split",
                "images",
                "positive_images",
                "negative_images",
                "polygons",
                "skipped_contours",
            ),
        )
        writer.writeheader()

        for row in stats:
            writer.writerow(
                {
                    "split": row.split,
                    "images": row.image_count,
                    "positive_images": row.positive_count,
                    "negative_images": row.negative_count,
                    "polygons": row.polygon_count,
                    "skipped_contours": row.skipped_contours,
                }
            )


def convert_dataset(
    src_root: Path,
    dst_root: Path,
    class_id: int,
    class_name: str,
    min_area: float,
    epsilon_ratio: float,
    min_points: int,
    image_mode: str,
    overwrite: bool,
    dry_run: bool,
) -> list[ConvertStats]:
    """
        Convert the full scratch patch dataset to YOLO segmentation format.
    """

    src_root = resolve_project_path(src_root)
    dst_root = resolve_project_path(dst_root)

    if not src_root.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {src_root}")

    prepare_output(dst_root=dst_root, overwrite=overwrite, dry_run=dry_run)

    stats: list[ConvertStats] = []
    for split in SPLIT_MAP:
        split_stats = convert_split(
            src_root=src_root,
            dst_root=dst_root,
            split=split,
            class_id=class_id,
            min_area=min_area,
            epsilon_ratio=epsilon_ratio,
            min_points=min_points,
            image_mode=image_mode,
            dry_run=dry_run,
        )
        stats.append(split_stats)

    write_data_yaml(
        dst_root=dst_root,
        class_id=class_id,
        class_name=class_name,
        dry_run=dry_run,
    )
    write_metadata(dst_root=dst_root, stats=stats, dry_run=dry_run)
    return stats


def parse_args() -> argparse.Namespace:
    """
        Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Convert binary mask dataset to YOLO segmentation format."
    )

    parser.add_argument(
        "--src",
        type=Path,
        default=SCRATCH_TRAIN_DATASET,
        help="Source dataset root with train/valid/test images and masks.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output YOLO segmentation dataset root.",
    )
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--class-name", type=str, default="scratch")
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="Skip contours with area smaller than this value in pixels.",
    )
    parser.add_argument(
        "--epsilon-ratio",
        type=float,
        default=0.0,
        help=(
            "Contour simplification ratio. Keep 0.0 for thin scratches to "
            "preserve mask detail."
        ),
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Minimum polygon points required by YOLO segmentation.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("copy", "symlink"),
        default="copy",
        help=(
            "Write PNG copies for a portable dataset, or symlink only when "
            "source images are already PNG."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.class_id < 0:
        parser.error("--class-id must be non-negative")
    if args.min_area < 0:
        parser.error("--min-area must be non-negative")
    if args.epsilon_ratio < 0:
        parser.error("--epsilon-ratio must be non-negative")
    if args.min_points < 3:
        parser.error("--min-points must be at least 3")

    return args


def main() -> None:
    """
        Convert masks and print a compact summary.
    """

    args = parse_args()

    stats = convert_dataset(
        src_root=args.src,
        dst_root=args.dst,
        class_id=args.class_id,
        class_name=args.class_name,
        min_area=args.min_area,
        epsilon_ratio=args.epsilon_ratio,
        min_points=args.min_points,
        image_mode=args.image_mode,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    dst_root = resolve_project_path(args.dst)
    print(f"YOLO segmentation dataset: {dst_root}")
    if args.dry_run:
        print("Dry run only. No files were written.")

    for row in stats:
        print(
            f"{row.split}: images={row.image_count} "
            f"positive={row.positive_count} negative={row.negative_count} "
            f"polygons={row.polygon_count} skipped_contours={row.skipped_contours}"
        )

    if not args.dry_run:
        print(f"data.yaml: {dst_root / 'data.yaml'}")
        print("Train with:")
        print(f"  yolo segment train data={dst_root / 'data.yaml'} model=yolo11n-seg.pt imgsz=512")


if __name__ == "__main__":
    main()
