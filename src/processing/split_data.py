"""
    Split labeled scratch images into train/valid/test folders.

    Input format:
        source/
          images/
          masks/

    Output format:
        data/scratch_v2/
          train/
            images/
            masks/
          valid/
            images/
            masks/
          test/
            images/
            masks/
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

from configs.data import IMAGE_EXTENSIONS, LABELING_DATASET, MASK_EXTENSION, SPLITS


OUTPUT_IMAGE_EXTENSION = ".png"


@dataclass(frozen=True)
class ImageMaskPair:
    """
        One matched image and mask pair.
    """

    image_path: Path
    mask_path: Path

    @property
    def stem(self) -> str:
        """Return pair stem used for deterministic sorting and metadata."""
        return self.image_path.stem


def resolve_project_path(path: Path) -> Path:
    """
        Resolve relative paths from the project root.
    """

    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def collect_pairs(src_root: Path) -> list[ImageMaskPair]:
    """
        Collect image/mask pairs from a labeling-style dataset folder.
    """

    image_dir = src_root / "images"
    mask_dir = src_root / "masks"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"No supported images found in: {image_dir}")

    seen_stems: set[str] = set()
    pairs: list[ImageMaskPair] = []
    missing_masks: list[Path] = []

    for image_path in image_paths:
        if image_path.stem in seen_stems:
            raise RuntimeError(f"Duplicate image stem found: {image_path.stem}")
        seen_stems.add(image_path.stem)

        mask_path = mask_dir / f"{image_path.stem}{MASK_EXTENSION}"
        if not mask_path.is_file():
            missing_masks.append(image_path)
            continue

        pairs.append(ImageMaskPair(image_path=image_path, mask_path=mask_path))

    if missing_masks:
        names = ", ".join(path.name for path in missing_masks[:5])
        extra = "" if len(missing_masks) <= 5 else f", ... ({len(missing_masks)} total)"
        raise RuntimeError(f"Missing masks for images: {names}{extra}")

    return sorted(pairs, key=lambda pair: pair.stem)


def validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    """
        Validate and normalize train/valid/test ratios.
    """

    ratios = (train_ratio, valid_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError(f"Ratios must be non-negative, got {ratios}")

    total = sum(ratios)
    if total <= 0:
        raise ValueError("At least one split ratio must be positive.")

    return tuple(ratio / total for ratio in ratios)


def split_counts(total_count: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    """
        Convert split ratios into integer counts with largest-remainder rounding.
    """

    raw_counts = np.array(ratios, dtype=np.float64) * total_count
    counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(total_count - counts.sum())

    if remainder > 0:
        order = np.argsort(-(raw_counts - counts))
        for index in order[:remainder]:
            counts[index] += 1

    return {
        split: int(count)
        for split, count in zip(SPLITS, counts.tolist())
    }


def split_pairs(
    pairs: list[ImageMaskPair],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[ImageMaskPair]]:
    """
        Shuffle pairs deterministically and assign them to train/valid/test.
    """

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(pairs)).tolist()
    shuffled_pairs = [pairs[index] for index in indices]
    counts = split_counts(len(pairs), ratios)

    train_end = counts["train"]
    valid_end = train_end + counts["valid"]

    return {
        "train": shuffled_pairs[:train_end],
        "valid": shuffled_pairs[train_end:valid_end],
        "test": shuffled_pairs[valid_end:],
    }


def prepare_output_root(dst_root: Path, overwrite: bool, dry_run: bool) -> None:
    """
        Create an empty output root or fail if it already exists.
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


def copy_split(
    split: str,
    pairs: list[ImageMaskPair],
    dst_root: Path,
    dry_run: bool,
) -> None:
    """
        Copy one split into images and masks folders.
    """

    if dry_run:
        return

    image_dir = dst_root / split / "images"
    mask_dir = dst_root / split / "masks"

    for pair in pairs:
        image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {pair.image_path}")

        output_image_path = image_dir / f"{pair.stem}{OUTPUT_IMAGE_EXTENSION}"
        output_mask_path = mask_dir / f"{pair.stem}{MASK_EXTENSION}"

        if not cv2.imwrite(str(output_image_path), image):
            raise RuntimeError(f"Failed to save image: {output_image_path}")

        shutil.copy2(pair.mask_path, output_mask_path)


def write_metadata(
    split_map: dict[str, list[ImageMaskPair]],
    dst_root: Path,
    dry_run: bool,
) -> None:
    """
        Save split assignment metadata for reproducibility.
    """

    if dry_run:
        return

    metadata_path = dst_root / "split.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("split", "image", "mask"),
        )
        writer.writeheader()
        for split in SPLITS:
            for pair in split_map[split]:
                writer.writerow(
                    {
                        "split": split,
                        "image": f"{pair.stem}{OUTPUT_IMAGE_EXTENSION}",
                        "mask": f"{pair.stem}{MASK_EXTENSION}",
                    }
                )


def parse_args() -> argparse.Namespace:
    """
        Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Split labeled scratch data into train/valid/test folders."
    )

    parser.add_argument("--src", type=Path, default=LABELING_DATASET)
    parser.add_argument("--dst", type=Path, default=Path("data/scratch_v2"))

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main() -> None:
    """
        Split labeled image/mask pairs into train/valid/test folders.
    """

    args = parse_args()
    src_root = resolve_project_path(args.src)
    dst_root = resolve_project_path(args.dst)
    ratios = validate_ratios(
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )

    pairs = collect_pairs(src_root)
    split_map = split_pairs(
        pairs=pairs,
        ratios=ratios,
        seed=args.seed,
    )

    prepare_output_root(
        dst_root=dst_root,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    for split in SPLITS:
        copy_split(
            split=split,
            pairs=split_map[split],
            dst_root=dst_root,
            dry_run=args.dry_run,
        )

    write_metadata(
        split_map=split_map,
        dst_root=dst_root,
        dry_run=args.dry_run,
    )

    mode = "DRY RUN" if args.dry_run else "DONE"
    print(f"{mode}: {len(pairs)} image/mask pairs")
    print(f"Source: {src_root}")
    print(f"Output: {dst_root}")
    for split in SPLITS:
        print(f"{split}: {len(split_map[split])}")


if __name__ == "__main__":
    main()
