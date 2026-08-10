"""
    Split binary scratch masks into one-mask-per-sample datasets.

    Input:
        data/scratch_patches_sam2/
          train/
            images/
            masks/
            patches.csv

    Output:
        data/scratch_patches_sam2_instances/
          train/
            images/
            masks/
            instances.csv

    Each positive connected component becomes a new sample. The image is copied
    or symlinked for each component, while the output mask keeps only that
    component. This is useful before prompt-based SAM2 fine-tuning.
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

from configs.data import IMAGE_EXTENSIONS, MASK_EXTENSION, SPLITS


DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "scratch_patches_sam2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "scratch_patches_sam2_instances"
OUTPUT_IMAGE_EXTENSION = ".png"


@dataclass(frozen=True)
class InstanceRecord:
    """
        Metadata for one output instance sample.
    """

    split: str
    sample_name: str
    patch_image: str
    patch_mask: str
    original_image: str
    patch_x: int | None
    patch_y: int | None
    component_id: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    positive_pixels: int


@dataclass(frozen=True)
class SplitStats:
    """
        Summary for one split.
    """

    split: str
    input_images: int
    positive_images: int
    negative_images: int
    output_instances: int
    skipped_components: int
    output_negatives: int


def resolve_project_path(path: Path) -> Path:
    """
        Resolve relative paths from the project root.
    """

    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def collect_images(image_dir: Path) -> list[Path]:
    """
        Collect supported image files from one split.
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


def load_patch_metadata(split_root: Path) -> dict[str, dict[str, str]]:
    """
        Load optional patches.csv metadata keyed by patch_name.
    """

    metadata_path = split_root / "patches.csv"
    if not metadata_path.is_file():
        return {}

    with metadata_path.open(newline="", encoding="utf-8") as file:
        return {
            row["patch_name"]: row
            for row in csv.DictReader(file)
            if row.get("patch_name")
        }


def prepare_output(dst_root: Path, overwrite: bool, dry_run: bool) -> None:
    """
        Create output folders or replace them when requested.
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

    for split in SPLITS:
        (dst_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_root / split / "masks").mkdir(parents=True, exist_ok=True)


def write_or_link_png_image(src_path: Path, dst_path: Path, image_mode: str) -> None:
    """
        Write one image to a unique output sample name as PNG.
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
                f"instance output images are named with {OUTPUT_IMAGE_EXTENSION}: "
                f"{src_path}"
            )
        dst_path.symlink_to(src_path.resolve())
        return

    raise ValueError(f"Unsupported image mode: {image_mode}")


def component_label_mask(mask: np.ndarray, close_kernel: int) -> np.ndarray:
    """
        Return the binary mask used to discover connected components.
    """

    binary = np.zeros_like(mask, dtype=np.uint8)
    binary[mask > 0] = 255

    if close_kernel <= 1:
        return binary

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_kernel, close_kernel),
    )
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def iter_components(
    mask: np.ndarray,
    min_pixels: int,
    close_kernel: int,
) -> tuple[list[tuple[int, np.ndarray, tuple[int, int, int, int], int]], int]:
    """
        Yield filtered component masks and count skipped components.
    """

    original_binary = mask > 0
    label_binary = component_label_mask(mask=mask, close_kernel=close_kernel)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        label_binary,
        connectivity=8,
    )

    components: list[tuple[int, np.ndarray, tuple[int, int, int, int], int]] = []
    skipped_components = 0

    for component_id in range(1, component_count):
        component_region = labels == component_id
        component_pixels = int((component_region & original_binary).sum())

        if component_pixels < min_pixels:
            skipped_components += 1
            continue

        component_mask = np.zeros_like(mask, dtype=np.uint8)
        component_mask[component_region & original_binary] = 255

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])

        components.append(
            (
                component_id,
                component_mask,
                (x, y, w, h),
                component_pixels,
            )
        )

    return components, skipped_components


def parse_optional_int(value: str | None) -> int | None:
    """
        Parse optional integer metadata fields.
    """

    if value in (None, ""):
        return None
    return int(value)


def write_instances_csv(path: Path, records: list[InstanceRecord]) -> None:
    """
        Save split metadata for traceability and prompt generation.
    """

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "sample_name",
                "patch_image",
                "patch_mask",
                "original_image",
                "patch_x",
                "patch_y",
                "component_id",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "positive_pixels",
            ],
        )
        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "split": record.split,
                    "sample_name": record.sample_name,
                    "patch_image": record.patch_image,
                    "patch_mask": record.patch_mask,
                    "original_image": record.original_image,
                    "patch_x": "" if record.patch_x is None else record.patch_x,
                    "patch_y": "" if record.patch_y is None else record.patch_y,
                    "component_id": record.component_id,
                    "bbox_x": record.bbox_x,
                    "bbox_y": record.bbox_y,
                    "bbox_w": record.bbox_w,
                    "bbox_h": record.bbox_h,
                    "positive_pixels": record.positive_pixels,
                }
            )


def split_one_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    min_pixels: int,
    close_kernel: int,
    image_mode: str,
    include_negative: bool,
    dry_run: bool,
) -> SplitStats:
    """
        Split masks in one train/valid/test folder.
    """

    split_root = src_root / split
    image_dir = split_root / "images"
    mask_dir = split_root / "masks"
    out_image_dir = dst_root / split / "images"
    out_mask_dir = dst_root / split / "masks"

    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_paths = collect_images(image_dir)
    patch_metadata = load_patch_metadata(split_root)

    positive_images = 0
    skipped_components = 0
    output_negatives = 0
    records: list[InstanceRecord] = []

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

        components, skipped = iter_components(
            mask=mask,
            min_pixels=min_pixels,
            close_kernel=close_kernel,
        )
        skipped_components += skipped

        meta = patch_metadata.get(image_path.stem, {})
        original_image = meta.get("source_image", image_path.name)
        patch_x = parse_optional_int(meta.get("x"))
        patch_y = parse_optional_int(meta.get("y"))

        if components:
            positive_images += 1

        for instance_index, (
            component_id,
            component_mask,
            bbox,
            positive_pixels,
        ) in enumerate(components, start=1):
            sample_name = f"{image_path.stem}_i{instance_index:03d}"
            x, y, w, h = bbox

            if not dry_run:
                write_or_link_png_image(
                    src_path=image_path,
                    dst_path=out_image_dir / f"{sample_name}{OUTPUT_IMAGE_EXTENSION}",
                    image_mode=image_mode,
                )
                mask_output_path = out_mask_dir / f"{sample_name}{MASK_EXTENSION}"
                if not cv2.imwrite(str(mask_output_path), component_mask):
                    raise RuntimeError(f"Failed to save mask: {mask_output_path}")

            records.append(
                InstanceRecord(
                    split=split,
                    sample_name=sample_name,
                    patch_image=image_path.name,
                    patch_mask=mask_path.name,
                    original_image=original_image,
                    patch_x=patch_x,
                    patch_y=patch_y,
                    component_id=component_id,
                    bbox_x=x,
                    bbox_y=y,
                    bbox_w=w,
                    bbox_h=h,
                    positive_pixels=positive_pixels,
                )
            )

        if include_negative and not components:
            sample_name = f"{image_path.stem}_neg"
            empty_mask = np.zeros_like(mask, dtype=np.uint8)
            output_negatives += 1

            if not dry_run:
                write_or_link_png_image(
                    src_path=image_path,
                    dst_path=out_image_dir / f"{sample_name}{OUTPUT_IMAGE_EXTENSION}",
                    image_mode=image_mode,
                )
                mask_output_path = out_mask_dir / f"{sample_name}{MASK_EXTENSION}"
                if not cv2.imwrite(str(mask_output_path), empty_mask):
                    raise RuntimeError(f"Failed to save mask: {mask_output_path}")

            records.append(
                InstanceRecord(
                    split=split,
                    sample_name=sample_name,
                    patch_image=image_path.name,
                    patch_mask=mask_path.name,
                    original_image=original_image,
                    patch_x=patch_x,
                    patch_y=patch_y,
                    component_id=0,
                    bbox_x=0,
                    bbox_y=0,
                    bbox_w=0,
                    bbox_h=0,
                    positive_pixels=0,
                )
            )

    if not dry_run:
        write_instances_csv(dst_root / split / "instances.csv", records)

    return SplitStats(
        split=split,
        input_images=len(image_paths),
        positive_images=positive_images,
        negative_images=len(image_paths) - positive_images,
        output_instances=len(records) - output_negatives,
        skipped_components=skipped_components,
        output_negatives=output_negatives,
    )


def split_dataset(
    src_root: Path,
    dst_root: Path,
    min_pixels: int,
    close_kernel: int,
    image_mode: str,
    include_negative: bool,
    overwrite: bool,
    dry_run: bool,
) -> list[SplitStats]:
    """
        Split all dataset masks into per-instance samples.
    """

    src_root = resolve_project_path(src_root)
    dst_root = resolve_project_path(dst_root)

    if not src_root.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {src_root}")

    prepare_output(dst_root=dst_root, overwrite=overwrite, dry_run=dry_run)

    stats: list[SplitStats] = []
    for split in SPLITS:
        stats.append(
            split_one_split(
                src_root=src_root,
                dst_root=dst_root,
                split=split,
                min_pixels=min_pixels,
                close_kernel=close_kernel,
                image_mode=image_mode,
                include_negative=include_negative,
                dry_run=dry_run,
            )
        )

    return stats


def parse_args() -> argparse.Namespace:
    """
        Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Split binary masks into one instance mask per image sample."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Source patch dataset root with train/valid/test images and masks.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output per-instance dataset root.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=20,
        help="Skip components with fewer foreground pixels.",
    )
    parser.add_argument(
        "--close-kernel",
        type=int,
        default=0,
        help=(
            "Optional odd/even kernel size for morphological close before "
            "component labeling. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--image-mode",
        choices=("copy", "symlink"),
        default="copy",
        help=(
            "Write PNG copies for duplicated images, or symlink only when "
            "source images are already PNG."
        ),
    )
    parser.add_argument(
        "--include-negative",
        action="store_true",
        help="Also emit empty-mask samples for negative patches.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.min_pixels < 1:
        parser.error("--min-pixels must be positive")
    if args.close_kernel < 0:
        parser.error("--close-kernel must be non-negative")

    return args


def main() -> None:
    """
        Split masks and print a compact summary.
    """

    args = parse_args()
    stats = split_dataset(
        src_root=args.src,
        dst_root=args.dst,
        min_pixels=args.min_pixels,
        close_kernel=args.close_kernel,
        image_mode=args.image_mode,
        include_negative=args.include_negative,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    dst_root = resolve_project_path(args.dst)
    print(f"Instance mask dataset: {dst_root}")
    if args.dry_run:
        print("Dry run only. No files were written.")

    for row in stats:
        print(
            f"{row.split}: input={row.input_images} "
            f"positive={row.positive_images} negative={row.negative_images} "
            f"instances={row.output_instances} "
            f"negatives_written={row.output_negatives} "
            f"skipped_components={row.skipped_components}"
        )


if __name__ == "__main__":
    main()
