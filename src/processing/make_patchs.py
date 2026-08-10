"""
    Offline sliding-window patch generator for scratch segmentation.

    This script converts full-resolution image/mask pairs into fixed-size
    segmentation patches. Image and mask patches are always cropped with
    exactly the same coordinates.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.data import IMAGE_EXTENSIONS, SCRATCH_TRAIN_DATASET, SPLITS


DEFAULT_PATCH_SIZE = 512
DEFAULT_PATCH_OVERLAP = 0.25
OUTPUT_IMAGE_EXTENSION = ".png"


@dataclass(frozen=True)
class PatchRecord:
    """
        Metadata for one patch crop.
    """

    image_path: Path
    mask_path: Path
    patch_index: int
    x: int
    y: int
    positive_pixels: int

    @property
    def has_scratch(self) -> bool:
        """Return whether this patch contains any scratch pixel."""
        return self.positive_pixels > 0

    @property
    def output_stem(self) -> str:
        """Create deterministic output filename stem."""
        return f"{self.image_path.stem}_p{self.patch_index:03d}_x{self.x:04d}_y{self.y:04d}"


def validate_patch_settings(
    patch_size: int,
    overlap: float,
) -> int:
    """
        Validate patch settings and return stride.
    """

    """Patch size must be positive"""
    if patch_size < 1:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

    """Overlap must keep a positive stride"""
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    """Convert overlap ratio to integer stride"""
    stride = int(round(patch_size * (1.0 - overlap)))

    if stride < 1:
        raise ValueError(
            f"Invalid stride {stride}; reduce overlap or increase patch_size"
        )

    return stride


def sliding_window_coords(
    size: int,
    patch_size: int,
    stride: int,
) -> list[int]:
    """
        Generate top-left coordinates for sliding window along one axis.

        The final window is flush with the image edge so no border strip is
        lost when the image size is not divisible by stride.
    """

    """Small images cannot produce a full patch"""
    if size < patch_size:
        return []

    """Generate regular grid coordinates"""
    coords = list(range(0, size - patch_size + 1, stride))

    """Add last edge-aligned coordinate when needed"""
    last = size - patch_size

    if coords[-1] != last:
        coords.append(last)

    return coords


def iter_image_paths(image_dir: Path) -> list[Path]:
    """
        Collect supported image files from one split.
    """

    """Check image folder"""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    """Return only supported image files"""
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_patch_records(
    image_dir: Path,
    mask_dir: Path,
    patch_size: int,
    stride: int,
) -> tuple[list[PatchRecord], list[PatchRecord]]:
    """
        Collect positive and negative patch records without storing image crops.
    """

    """Check mask folder"""
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    positive_records: list[PatchRecord] = []
    negative_records: list[PatchRecord] = []

    """Process each source image and corresponding mask"""
    for image_path in iter_image_paths(image_dir):
        mask_path = mask_dir / f"{image_path.stem}.png"

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

        h, w = mask.shape[:2]
        ys = sliding_window_coords(h, patch_size, stride)
        xs = sliding_window_coords(w, patch_size, stride)

        patch_index = 0

        """Create patch metadata from mask only"""
        for y in ys:
            for x in xs:
                mask_patch = mask[y:y + patch_size, x:x + patch_size]
                positive_pixels = int((mask_patch > 0).sum())

                record = PatchRecord(
                    image_path=image_path,
                    mask_path=mask_path,
                    patch_index=patch_index,
                    x=x,
                    y=y,
                    positive_pixels=positive_pixels,
                )

                if record.has_scratch:
                    positive_records.append(record)
                else:
                    negative_records.append(record)

                patch_index += 1

    return positive_records, negative_records


def sample_negative_records(
    negative_records: list[PatchRecord],
    positive_count: int,
    max_negative_ratio: float | None,
    seed: int,
) -> list[PatchRecord]:
    """
        Sample negative patches according to the requested positive ratio.
    """

    """Keep all negatives when no cap is requested"""
    if max_negative_ratio is None:
        return negative_records

    """Validate cap ratio"""
    if max_negative_ratio < 0:
        raise ValueError(
            f"max_negative_ratio must be non-negative, got {max_negative_ratio}"
        )

    """Compute maximum number of negative patches to keep"""
    max_negatives = int(round(positive_count * max_negative_ratio))

    if len(negative_records) <= max_negatives:
        return negative_records

    """Randomly sample negative records deterministically"""
    rng = np.random.default_rng(seed=seed)
    indices = rng.choice(
        len(negative_records),
        size=max_negatives,
        replace=False,
    )

    return [negative_records[index] for index in sorted(indices.tolist())]


def save_patch(
    record: PatchRecord,
    out_image_dir: Path,
    out_mask_dir: Path,
    patch_size: int,
) -> None:
    """
        Crop one patch pair and save it to disk.
    """

    """Read source image and mask"""
    image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(record.mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Cannot read image: {record.image_path}")

    if mask is None:
        raise ValueError(f"Cannot read mask: {record.mask_path}")

    """Crop aligned image and mask patches"""
    y1 = record.y
    y2 = record.y + patch_size
    x1 = record.x
    x2 = record.x + patch_size

    image_patch = image[y1:y2, x1:x2]
    mask_patch = mask[y1:y2, x1:x2]

    """Write patch files"""
    image_path = out_image_dir / f"{record.output_stem}{OUTPUT_IMAGE_EXTENSION}"
    mask_path = out_mask_dir / f"{record.output_stem}.png"

    if not cv2.imwrite(str(image_path), image_patch):
        raise RuntimeError(f"Failed to save image patch: {image_path}")

    if not cv2.imwrite(str(mask_path), mask_patch):
        raise RuntimeError(f"Failed to save mask patch: {mask_path}")


def write_metadata(
    metadata_path: Path,
    split: str,
    records: list[PatchRecord],
) -> None:
    """
        Save patch metadata for traceability.
    """

    """Create metadata CSV"""
    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "patch_name",
                "source_image",
                "source_mask",
                "patch_index",
                "x",
                "y",
                "has_scratch",
                "positive_pixels",
            ],
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "split": split,
                    "patch_name": record.output_stem,
                    "source_image": record.image_path.name,
                    "source_mask": record.mask_path.name,
                    "patch_index": record.patch_index,
                    "x": record.x,
                    "y": record.y,
                    "has_scratch": int(record.has_scratch),
                    "positive_pixels": record.positive_pixels,
                }
            )


def warn_if_output_not_empty(
    out_image_dir: Path,
    out_mask_dir: Path,
) -> None:
    """
        Warn when generated patch folders already contain files.
    """

    """Count existing generated files"""
    existing_images = list(out_image_dir.glob("*")) if out_image_dir.exists() else []
    existing_masks = list(out_mask_dir.glob("*")) if out_mask_dir.exists() else []

    if existing_images or existing_masks:
        print(
            "Warning: output folders already contain files. "
            "Existing files with the same names will be overwritten, "
            "but stale files from old settings will remain."
        )


def process_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    patch_size: int,
    overlap: float,
    max_negative_ratio: float | None,
    seed: int,
) -> None:
    """
        Process one split and save patch image/mask pairs.
    """

    """Validate patch settings"""
    stride = validate_patch_settings(
        patch_size=patch_size,
        overlap=overlap,
    )

    """Source folders"""
    image_dir = src_root / split / "images"
    mask_dir = src_root / split / "masks"

    """Destination folders"""
    out_image_dir = dst_root / split / "images"
    out_mask_dir = dst_root / split / "masks"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    warn_if_output_not_empty(
        out_image_dir=out_image_dir,
        out_mask_dir=out_mask_dir,
    )

    """Collect patch coordinates and labels"""
    positive_records, negative_records = collect_patch_records(
        image_dir=image_dir,
        mask_dir=mask_dir,
        patch_size=patch_size,
        stride=stride,
    )

    """Sample negative patches if requested"""
    kept_negative_records = sample_negative_records(
        negative_records=negative_records,
        positive_count=len(positive_records),
        max_negative_ratio=max_negative_ratio,
        seed=seed,
    )

    """Keep positive patches first for easier inspection"""
    records = positive_records + kept_negative_records

    print(
        f"[{split}] positive={len(positive_records)} "
        f"negative_total={len(negative_records)} "
        f"negative_kept={len(kept_negative_records)} "
        f"total={len(records)}"
    )

    """Save selected patch pairs"""
    for record in records:
        save_patch(
            record=record,
            out_image_dir=out_image_dir,
            out_mask_dir=out_mask_dir,
            patch_size=patch_size,
        )

    """Save split metadata"""
    write_metadata(
        metadata_path=dst_root / split / "patches.csv",
        split=split,
        records=records,
    )


def parse_args() -> argparse.Namespace:
    """
        Parse command-line arguments.
    """

    """Build parser"""
    parser = argparse.ArgumentParser(
        description="Create sliding-window patches for scratch segmentation."
    )

    """Input and output paths"""
    parser.add_argument("--src", type=Path, default=SCRATCH_TRAIN_DATASET)
    parser.add_argument("--dst", type=Path, default=Path("data/scratch_patches"))

    """Patch settings"""
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--overlap", type=float, default=DEFAULT_PATCH_OVERLAP)

    """Negative sampling settings"""
    parser.add_argument("--train-negative-ratio", type=float, default=1.0)
    parser.add_argument("--valid-negative-ratio", type=float, default=None)
    parser.add_argument("--test-negative-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    """
        Generate train/valid/test patch datasets.
    """

    """Parse arguments"""
    args = parse_args()

    """Process all splits with split-specific negative sampling"""
    negative_ratios = {
        "train": args.train_negative_ratio,
        "valid": args.valid_negative_ratio,
        "test": args.test_negative_ratio,
    }

    for split in SPLITS:
        process_split(
            src_root=args.src,
            dst_root=args.dst,
            split=split,
            patch_size=args.patch_size,
            overlap=args.overlap,
            max_negative_ratio=negative_ratios[split],
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
