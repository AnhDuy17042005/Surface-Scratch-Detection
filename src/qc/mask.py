"""
    Quality check script for segmentation masks.

    Default target:
        data/scratch_patches/{train,valid,test}/masks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET = ROOT / "data" / "scratch_patches"
SPLITS = ("train", "valid", "test")


def summarize_values(values: set[int]) -> str:
    """
        Convert unique mask values to a compact string.
    """

    """Return empty marker when no masks were read"""
    if not values:
        return "[]"

    """Show values in sorted order"""
    return "[" + ", ".join(str(value) for value in sorted(values)) + "]"


def percentile(values: list[float], q: float) -> float:
    """
        Compute percentile and return 0 for empty input.
    """

    """Handle empty list safely"""
    if not values:
        return 0.0

    """Use numpy percentile for concise statistics"""
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def format_percent(ratio: float) -> str:
    """
        Format ratio as percentage.
    """

    return f"{ratio * 100:.6f}%"


def collect_mask_stats(mask_dir: Path) -> tuple[list[dict], list[Path]]:
    """
        Read masks and collect per-file statistics.
    """

    """Collect PNG masks in deterministic order"""
    mask_paths = sorted(mask_dir.glob("*.png"))

    rows: list[dict] = []
    bad_files: list[Path] = []

    """Read each mask and compute scratch pixel ratio"""
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            bad_files.append(mask_path)
            continue

        values = np.unique(mask)
        scratch_pixels = int(np.sum(mask > 0))
        total_pixels = int(mask.size)
        ratio = scratch_pixels / total_pixels if total_pixels else 0.0

        rows.append(
            {
                "name": mask_path.name,
                "path": mask_path,
                "width": int(mask.shape[1]),
                "height": int(mask.shape[0]),
                "values": [int(value) for value in values.tolist()],
                "scratch_pixels": scratch_pixels,
                "total_pixels": total_pixels,
                "ratio": ratio,
                "is_positive": scratch_pixels > 0,
            }
        )

    return rows, bad_files


def print_split_summary(
    split: str,
    mask_dir: Path,
    top_k: int,
    print_files: bool,
) -> None:
    """
        Print detailed mask quality summary for one split.
    """

    """Check split mask folder"""
    if not mask_dir.is_dir():
        print(f"\n== {split} ==")
        print(f"Mask directory not found: {mask_dir}")
        return

    """Collect per-mask statistics"""
    rows, bad_files = collect_mask_stats(mask_dir)

    """Prepare summary arrays"""
    positive_rows = [row for row in rows if row["is_positive"]]
    negative_rows = [row for row in rows if not row["is_positive"]]
    scratch_pixels = [row["scratch_pixels"] for row in rows]
    positive_pixels = [row["scratch_pixels"] for row in positive_rows]
    ratios = [row["ratio"] for row in rows]
    positive_ratios = [row["ratio"] for row in positive_rows]
    values = {value for row in rows for value in row["values"]}
    shapes = sorted({(row["width"], row["height"]) for row in rows})

    """Compute total foreground statistics"""
    total_scratch_pixels = int(sum(scratch_pixels))
    total_pixels = int(sum(row["total_pixels"] for row in rows))
    total_ratio = total_scratch_pixels / total_pixels if total_pixels else 0.0

    """Print split header"""
    print(f"\n== {split} ==")
    print(f"mask_dir           : {mask_dir}")
    print(f"mask_count         : {len(rows)}")
    print(f"bad_files          : {len(bad_files)}")
    print(f"positive_masks     : {len(positive_rows)}")
    print(f"negative_masks     : {len(negative_rows)}")
    print(f"shapes             : {shapes}")
    print(f"mask_values        : {summarize_values(values)}")
    print(f"total_scratch      : {total_scratch_pixels:,}/{total_pixels:,}")
    print(f"total_ratio        : {format_percent(total_ratio)}")

    """Print all-mask distribution"""
    if rows:
        print(
            "scratch_pixels all : "
            f"min={min(scratch_pixels):,} "
            f"p25={percentile(scratch_pixels, 25):.1f} "
            f"median={percentile(scratch_pixels, 50):.1f} "
            f"mean={np.mean(scratch_pixels):.1f} "
            f"p75={percentile(scratch_pixels, 75):.1f} "
            f"max={max(scratch_pixels):,}"
        )
        print(
            "ratio all          : "
            f"min={format_percent(min(ratios))} "
            f"median={format_percent(percentile(ratios, 50))} "
            f"mean={format_percent(float(np.mean(ratios)))} "
            f"max={format_percent(max(ratios))}"
        )

    """Print positive-only distribution"""
    if positive_rows:
        print(
            "scratch_pixels pos : "
            f"min={min(positive_pixels):,} "
            f"p25={percentile(positive_pixels, 25):.1f} "
            f"median={percentile(positive_pixels, 50):.1f} "
            f"mean={np.mean(positive_pixels):.1f} "
            f"p75={percentile(positive_pixels, 75):.1f} "
            f"max={max(positive_pixels):,}"
        )
        print(
            "ratio positive     : "
            f"min={format_percent(min(positive_ratios))} "
            f"median={format_percent(percentile(positive_ratios, 50))} "
            f"mean={format_percent(float(np.mean(positive_ratios)))} "
            f"max={format_percent(max(positive_ratios))}"
        )

    """Print files with read errors"""
    if bad_files:
        print("bad file examples:")
        for path in bad_files[:top_k]:
            print(f"  {path.name}")

    """Sort masks by scratch pixels"""
    rows_by_pixels = sorted(rows, key=lambda row: row["scratch_pixels"])


def parse_args() -> argparse.Namespace:
    """
        Parse command-line arguments.
    """

    """Build parser"""
    parser = argparse.ArgumentParser(
        description="QA segmentation masks and print summary statistics."
    )

    """Dataset root should contain train/valid/test/masks"""
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Dataset root containing train/valid/test folders.",
    )

    """Allow checking only one split"""
    parser.add_argument(
        "--split",
        choices=(*SPLITS, "all"),
        default="all",
        help="Dataset split to inspect.",
    )

    """Control number of example files printed in summary"""
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of smallest/largest mask examples to print.",
    )

    """Optional verbose per-file output"""
    parser.add_argument(
        "--print-files",
        action="store_true",
        help="Print statistics for every mask file.",
    )

    return parser.parse_args()


def main() -> None:
    """
        Run mask quality checks.
    """

    """Parse arguments"""
    args = parse_args()

    """Resolve dataset root"""
    dataset = args.dataset.resolve()

    print(f"Dataset: {dataset}")

    """Select splits"""
    splits = SPLITS if args.split == "all" else (args.split,)

    """Print each split summary"""
    for split in splits:
        print_split_summary(
            split=split,
            mask_dir=dataset / split / "masks",
            top_k=args.top_k,
            print_files=args.print_files,
        )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.stdout.close()
