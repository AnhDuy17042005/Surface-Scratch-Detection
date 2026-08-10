"""
    Convert one-instance scratch patches into SAM2 PNGRawDataset format.

    Input:
        data/scratch_patches_sam2_instances/
          train/
            images/
            masks/
            instances.csv

    Output:
        data/scratch_sam2_format/
          JPEGImages/
            train/<sample_id>/00000.png
          Annotations/
            train/<sample_id>/00000.png
          ImageSets/
            train.txt

    The output layout is intended for official SAM2 training with
    PNGRawDataset. Each static image is represented as a one-frame video.
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


DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "scratch_patches_sam2_instances"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "scratch_sam2_format"
FRAME_IMAGE_NAME = "00000.png"
FRAME_MASK_NAME = "00000.png"


@dataclass(frozen=True)
class ConvertedSample:
    """
        Metadata for one converted one-frame SAM2 sample.
    """

    split: str
    video_name: str
    sample_name: str
    source_image: str
    source_mask: str
    frame_image: str
    frame_mask: str
    positive_pixels: int
    bbox_x: str
    bbox_y: str
    bbox_w: str
    bbox_h: str
    patch_image: str
    patch_mask: str
    original_image: str
    patch_x: str
    patch_y: str


@dataclass(frozen=True)
class SplitStats:
    """
        Summary for one converted split.
    """

    split: str
    input_images: int
    converted: int
    skipped_empty: int


def resolve_project_path(path: Path) -> Path:
    """
        Resolve relative paths from the project root.
    """

    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def collect_images(image_dir: Path) -> list[Path]:
    """
        Collect supported images from one split.
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


def load_instances_csv(split_root: Path) -> dict[str, dict[str, str]]:
    """
        Load optional instance metadata keyed by sample_name.
    """

    instances_path = split_root / "instances.csv"
    if not instances_path.is_file():
        return {}

    with instances_path.open(newline="", encoding="utf-8") as file:
        return {
            row["sample_name"]: row
            for row in csv.DictReader(file)
            if row.get("sample_name")
        }


def prepare_output(dst_root: Path, overwrite: bool, dry_run: bool) -> None:
    """
        Create or replace the SAM2 output root.
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

    (dst_root / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (dst_root / "Annotations").mkdir(parents=True, exist_ok=True)
    (dst_root / "ImageSets").mkdir(parents=True, exist_ok=True)


def write_frame_image(src_path: Path, dst_path: Path, image_mode: str) -> None:
    """
        Save one source image as 00000.png.
    """

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if image_mode == "symlink":
        if src_path.suffix.lower() != ".png":
            raise ValueError(
                "Symlink image mode requires PNG source images because "
                f"SAM2 frame images are named with .png: {src_path}"
            )
        dst_path.symlink_to(src_path.resolve())
        return

    image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {src_path}")
    if not cv2.imwrite(str(dst_path), image):
        raise RuntimeError(f"Failed to save image: {dst_path}")


def convert_mask(mask_path: Path, dst_path: Path, mask_value: int) -> int:
    """
        Save a binary mask as a palettised-style object id mask.
    """

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")

    foreground = mask > 0
    positive_pixels = int(foreground.sum())
    if positive_pixels == 0:
        return 0

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    output_mask = np.zeros_like(mask, dtype=np.uint8)
    output_mask[foreground] = np.uint8(mask_value)

    if not cv2.imwrite(str(dst_path), output_mask):
        raise RuntimeError(f"Failed to save mask: {dst_path}")

    return positive_pixels


def write_file_list(path: Path, video_names: list[str], dry_run: bool) -> None:
    """
        Write one file list for PNGRawDataset file_list_txt.
    """

    if dry_run:
        return

    text = "\n".join(video_names)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def metadata_value(row: dict[str, str], key: str) -> str:
    """
        Return CSV metadata value or an empty string.
    """

    return row.get(key, "")


def convert_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    image_mode: str,
    mask_value: int,
    dry_run: bool,
) -> tuple[SplitStats, list[ConvertedSample], list[str]]:
    """
        Convert one split to SAM2 one-frame video folders.
    """

    split_root = src_root / split
    image_dir = split_root / "images"
    mask_dir = split_root / "masks"

    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_paths = collect_images(image_dir)
    instance_rows = load_instances_csv(split_root)

    samples: list[ConvertedSample] = []
    video_names: list[str] = []
    skipped_empty = 0

    for image_path in image_paths:
        mask_path = mask_dir / f"{image_path.stem}{MASK_EXTENSION}"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found for image: {image_path}")

        video_name = f"{split}/{image_path.stem}"
        out_image_path = dst_root / "JPEGImages" / video_name / FRAME_IMAGE_NAME
        out_mask_path = dst_root / "Annotations" / video_name / FRAME_MASK_NAME

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")
        positive_pixels = int((mask > 0).sum())
        if positive_pixels == 0:
            skipped_empty += 1
            continue

        if not dry_run:
            write_frame_image(
                src_path=image_path,
                dst_path=out_image_path,
                image_mode=image_mode,
            )
            positive_pixels = convert_mask(
                mask_path=mask_path,
                dst_path=out_mask_path,
                mask_value=mask_value,
            )

        row = instance_rows.get(image_path.stem, {})
        video_names.append(video_name)
        samples.append(
            ConvertedSample(
                split=split,
                video_name=video_name,
                sample_name=image_path.stem,
                source_image=str(image_path.relative_to(src_root)),
                source_mask=str(mask_path.relative_to(src_root)),
                frame_image=str(out_image_path.relative_to(dst_root)),
                frame_mask=str(out_mask_path.relative_to(dst_root)),
                positive_pixels=positive_pixels,
                bbox_x=metadata_value(row, "bbox_x"),
                bbox_y=metadata_value(row, "bbox_y"),
                bbox_w=metadata_value(row, "bbox_w"),
                bbox_h=metadata_value(row, "bbox_h"),
                patch_image=metadata_value(row, "patch_image"),
                patch_mask=metadata_value(row, "patch_mask"),
                original_image=metadata_value(row, "original_image"),
                patch_x=metadata_value(row, "patch_x"),
                patch_y=metadata_value(row, "patch_y"),
            )
        )

    return (
        SplitStats(
            split=split,
            input_images=len(image_paths),
            converted=len(samples),
            skipped_empty=skipped_empty,
        ),
        samples,
        video_names,
    )


def write_metadata(
    dst_root: Path,
    samples: list[ConvertedSample],
    dry_run: bool,
) -> None:
    """
        Save combined conversion metadata.
    """

    if dry_run:
        return

    metadata_path = dst_root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "video_name",
                "sample_name",
                "source_image",
                "source_mask",
                "frame_image",
                "frame_mask",
                "positive_pixels",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "patch_image",
                "patch_mask",
                "original_image",
                "patch_x",
                "patch_y",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.__dict__)


def write_readme(dst_root: Path, mask_value: int, dry_run: bool) -> None:
    """
        Write compact usage notes next to the converted dataset.
    """

    if dry_run:
        return

    readme = "\n".join(
        [
            "# Scratch SAM2 Dataset",
            "",
            "One-frame PNGRawDataset layout for official SAM2 fine-tuning.",
            "",
            "Use these config paths:",
            "",
            "```yaml",
            f"dataset.img_folder: {dst_root / 'JPEGImages'}",
            f"dataset.gt_folder: {dst_root / 'Annotations'}",
            f"dataset.file_list_txt: {dst_root / 'ImageSets' / 'train.txt'}",
            "scratch.resolution: 1024",
            "scratch.num_frames: 1",
            "scratch.max_num_objects: 1",
            "```",
            "",
            f"Mask values are `0` for background and `{mask_value}` for scratch.",
            "",
        ]
    )
    (dst_root / "README.md").write_text(readme, encoding="utf-8")


def convert_dataset(
    src_root: Path,
    dst_root: Path,
    image_mode: str,
    mask_value: int,
    overwrite: bool,
    dry_run: bool,
) -> tuple[list[SplitStats], list[ConvertedSample]]:
    """
        Convert all splits to SAM2 PNGRawDataset format.
    """

    src_root = resolve_project_path(src_root)
    dst_root = resolve_project_path(dst_root)

    if not src_root.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {src_root}")

    prepare_output(dst_root=dst_root, overwrite=overwrite, dry_run=dry_run)

    stats: list[SplitStats] = []
    all_samples: list[ConvertedSample] = []
    for split in SPLITS:
        split_stats, split_samples, video_names = convert_split(
            src_root=src_root,
            dst_root=dst_root,
            split=split,
            image_mode=image_mode,
            mask_value=mask_value,
            dry_run=dry_run,
        )
        stats.append(split_stats)
        all_samples.extend(split_samples)
        write_file_list(
            path=dst_root / "ImageSets" / f"{split}.txt",
            video_names=video_names,
            dry_run=dry_run,
        )
        write_file_list(
            path=dst_root / f"{split}.txt",
            video_names=video_names,
            dry_run=dry_run,
        )

    write_metadata(dst_root=dst_root, samples=all_samples, dry_run=dry_run)
    write_readme(dst_root=dst_root, mask_value=mask_value, dry_run=dry_run)

    return stats, all_samples


def parse_args() -> argparse.Namespace:
    """
        Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Convert one-instance scratch patches to SAM2 PNGRawDataset format."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Source one-instance dataset root.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output SAM2 dataset root.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("copy", "symlink"),
        default="copy",
        help=(
            "Write PNG copies for portability, or symlink only when source "
            "images are already PNG."
        ),
    )
    parser.add_argument(
        "--mask-value",
        type=int,
        default=1,
        help="Foreground object id written into SAM2 annotation PNGs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if not 1 <= args.mask_value <= 255:
        parser.error("--mask-value must be in [1, 255]")

    return args


def main() -> None:
    """
        Convert dataset and print a compact summary.
    """

    args = parse_args()
    stats, samples = convert_dataset(
        src_root=args.src,
        dst_root=args.dst,
        image_mode=args.image_mode,
        mask_value=args.mask_value,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    dst_root = resolve_project_path(args.dst)
    print(f"SAM2 dataset: {dst_root}")
    if args.dry_run:
        print("Dry run only. No files were written.")

    for row in stats:
        print(
            f"{row.split}: input={row.input_images} "
            f"converted={row.converted} skipped_empty={row.skipped_empty}"
        )

    print(f"total converted={len(samples)}")
    if not args.dry_run:
        print(f"img_folder: {dst_root / 'JPEGImages'}")
        print(f"gt_folder: {dst_root / 'Annotations'}")
        print(f"train list: {dst_root / 'ImageSets' / 'train.txt'}")


if __name__ == "__main__":
    main()
