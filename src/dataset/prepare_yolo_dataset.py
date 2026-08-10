"""
    Prepare YOLO instance-segmentation data from labeled scratch masks.

    This pipeline avoids writing intermediate split and patch image/mask
    folders. It computes split/patch metadata in memory, then exports only the
    final YOLO training dataset:

        outputs/labeling
          images/
          masks/

        data/scratch_yolo_seg
          train/images
          train/labels
          valid/images
          valid/labels
          test/images
          test/labels
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

from configs.data import LABELING_DATASET, MASK_EXTENSION, SPLITS
from src.processing.make_patchs import (
    sample_negative_records,
    sliding_window_coords,
    validate_patch_settings,
)
from src.processing.split_data import (
    ImageMaskPair,
    collect_pairs,
    split_pairs,
    validate_ratios,
)
from src.processing.yolo_convert_labels import (
    format_label_row,
    mask_to_polygons,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "scratch_yolo_seg"
DEFAULT_SPLIT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "scratch"
OUTPUT_IMAGE_EXTENSION = ".png"


@dataclass(frozen=True)
class PatchRecord:
    """
        Metadata for one selected YOLO patch.

        Args:
            split          : train, valid, or test
            source_image   : original labeled image path
            source_mask    : original labeled mask path
            source_stem    : original image stem
            patch_index    : patch order within the source image grid
            x, y           : top-left patch coordinate in the source image
            width, height  : patch size written to YOLO
            positive_pixels: foreground mask pixels inside the patch
    """

    split: str
    source_image: Path
    source_mask: Path
    source_stem: str
    patch_index: int
    x: int
    y: int
    width: int
    height: int
    positive_pixels: int

    @property
    def has_scratch(self) -> bool:
        """Return whether this patch contains any scratch pixel."""
        return self.positive_pixels > 0

    @property
    def output_stem(self) -> str:
        """Create deterministic patch filename stem."""
        return (
            f"{self.source_stem}_p{self.patch_index:03d}"
            f"_x{self.x:04d}_y{self.y:04d}"
        )


@dataclass(frozen=True)
class SplitStats:
    """
        Summary for one split after export.
    """

    split: str
    source_images: int
    selected_patches: int
    positive_patches: int
    negative_patches: int
    polygons: int
    skipped_contours: int


def resolve_project_path(path: Path) -> Path:
    """
        Resolve relative paths from the project root.
    """

    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def prepare_output_root(
    output_root: Path,
    overwrite: bool,
    dry_run: bool,
) -> Path:
    """
        Prepare the final YOLO dataset root.
    """

    if dry_run:
        return output_root

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. "
                "Use --overwrite or choose a new --output-root."
            )
        shutil.rmtree(output_root)

    for split in SPLITS:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

    return output_root


def prepare_split_output_root(
    split_output_root: Path,
    overwrite: bool,
    dry_run: bool,
) -> None:
    """
        Prepare a full-image train/valid/test dataset root.
    """

    if dry_run:
        return

    if split_output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Split output root already exists: {split_output_root}. "
                "Use --overwrite or choose a new --split-output-root."
            )
        shutil.rmtree(split_output_root)

    for split in SPLITS:
        (split_output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (split_output_root / split / "masks").mkdir(parents=True, exist_ok=True)


def export_full_split_dataset(
    split_map: dict[str, list[ImageMaskPair]],
    split_output_root: Path,
    dry_run: bool,
) -> None:
    """
        Copy full-size image/mask pairs into train/valid/test folders.
    """

    if dry_run:
        return

    for split in SPLITS:
        image_dir = split_output_root / split / "images"
        mask_dir = split_output_root / split / "masks"

        for pair in split_map[split]:
            image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot read image: {pair.image_path}")

            image_path = image_dir / f"{pair.stem}{OUTPUT_IMAGE_EXTENSION}"
            mask_path = mask_dir / f"{pair.stem}{MASK_EXTENSION}"

            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(f"Failed to save split image: {image_path}")
            shutil.copy2(pair.mask_path, mask_path)

        print(
            f"[{split}] exported full split images: {len(split_map[split])}",
            flush=True,
        )


def write_split_metadata(
    split_map: dict[str, list[ImageMaskPair]],
    split_output_root: Path,
    dry_run: bool,
) -> None:
    """
        Save full-image split assignments for reproducibility.
    """

    if dry_run:
        return

    metadata_path = split_output_root / "split.csv"
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


def load_mask(pair: ImageMaskPair) -> np.ndarray:
    """
        Load one mask and validate that the paired image can be read.
    """

    mask = cv2.imread(str(pair.mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Cannot read mask: {pair.mask_path}")

    image = cv2.imread(str(pair.image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {pair.image_path}")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image and mask size mismatch: {pair.image_path.name} "
            f"{image.shape[:2]} != {mask.shape[:2]}"
        )

    return mask


def collect_pair_patch_records(
    split: str,
    pair: ImageMaskPair,
    patch_size: int,
    stride: int,
) -> tuple[list[PatchRecord], list[PatchRecord]]:
    """
        Collect positive and negative patch metadata for one source pair.
    """

    mask = load_mask(pair)
    h, w = mask.shape[:2]
    ys = sliding_window_coords(
        size=h,
        patch_size=patch_size,
        stride=stride,
    )
    xs = sliding_window_coords(
        size=w,
        patch_size=patch_size,
        stride=stride,
    )

    positive_records: list[PatchRecord] = []
    negative_records: list[PatchRecord] = []
    patch_index = 0

    """Build patch metadata without writing intermediate crops."""
    for y in ys:
        for x in xs:
            mask_patch = mask[y:y + patch_size, x:x + patch_size]
            positive_pixels = int((mask_patch > 0).sum())

            record = PatchRecord(
                split=split,
                source_image=pair.image_path,
                source_mask=pair.mask_path,
                source_stem=pair.stem,
                patch_index=patch_index,
                x=x,
                y=y,
                width=patch_size,
                height=patch_size,
                positive_pixels=positive_pixels,
            )

            if record.has_scratch:
                positive_records.append(record)
            else:
                negative_records.append(record)

            patch_index += 1

    return positive_records, negative_records


def collect_split_patch_records(
    split: str,
    pairs: list[ImageMaskPair],
    patch_size: int,
    stride: int,
    negative_ratio: float | None,
    seed: int,
) -> list[PatchRecord]:
    """
        Collect and sample patch metadata for one split.
    """

    positive_records: list[PatchRecord] = []
    negative_records: list[PatchRecord] = []

    for pair in pairs:
        pair_positive, pair_negative = collect_pair_patch_records(
            split=split,
            pair=pair,
            patch_size=patch_size,
            stride=stride,
        )
        positive_records.extend(pair_positive)
        negative_records.extend(pair_negative)

    kept_negative_records = sample_negative_records(
        negative_records=negative_records,
        positive_count=len(positive_records),
        max_negative_ratio=negative_ratio,
        seed=seed,
    )

    records = positive_records + kept_negative_records
    print(
        f"[{split}] source_images={len(pairs)} "
        f"positive={len(positive_records)} "
        f"negative_total={len(negative_records)} "
        f"negative_kept={len(kept_negative_records)} "
        f"total={len(records)}"
    )
    return records


def crop_record_from_arrays(
    record: PatchRecord,
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
        Crop one image/mask patch from already-loaded arrays.
    """

    y1 = record.y
    y2 = record.y + record.height
    x1 = record.x
    x2 = record.x + record.width

    image_patch = image[y1:y2, x1:x2]
    mask_patch = mask[y1:y2, x1:x2]

    if image_patch.shape[:2] != (record.height, record.width):
        raise ValueError(
            f"Invalid image patch size for {record.output_stem}: "
            f"{image_patch.shape[:2]}"
        )
    if mask_patch.shape[:2] != (record.height, record.width):
        raise ValueError(
            f"Invalid mask patch size for {record.output_stem}: "
            f"{mask_patch.shape[:2]}"
        )

    return image_patch, mask_patch


def write_yolo_patch(
    record: PatchRecord,
    image_patch: np.ndarray,
    mask_patch: np.ndarray,
    yolo_root: Path,
    class_id: int,
    min_area: float,
    epsilon_ratio: float,
    min_points: int,
    dry_run: bool,
) -> tuple[int, int]:
    """
        Write one cropped patch as YOLO image + segmentation label.

        Returns:
            polygon_count, skipped_contours
    """

    rows, skipped_contours = mask_to_polygons(
        mask=mask_patch,
        class_id=class_id,
        min_area=min_area,
        epsilon_ratio=epsilon_ratio,
        min_points=min_points,
    )

    if dry_run:
        return len(rows), skipped_contours

    image_path = (
        yolo_root
        / record.split
        / "images"
        / f"{record.output_stem}{OUTPUT_IMAGE_EXTENSION}"
    )
    label_path = yolo_root / record.split / "labels" / f"{record.output_stem}.txt"

    if not cv2.imwrite(str(image_path), image_patch):
        raise RuntimeError(f"Failed to save YOLO image: {image_path}")

    label_text = "\n".join(format_label_row(row) for row in rows)
    if label_text:
        label_text += "\n"
    label_path.write_text(label_text, encoding="utf-8")

    return len(rows), skipped_contours


def export_split_records(
    split: str,
    records: list[PatchRecord],
    yolo_root: Path,
    class_id: int,
    min_area: float,
    epsilon_ratio: float,
    min_points: int,
    dry_run: bool,
) -> tuple[int, int]:
    """
        Export selected patches for one split.

        Source images are loaded once and reused for all selected patches from
        that image. This matters for large raw datasets.
    """

    polygon_count = 0
    skipped_contours = 0
    records_by_source: dict[tuple[Path, Path], list[PatchRecord]] = {}

    for record in records:
        key = (record.source_image, record.source_mask)
        records_by_source.setdefault(key, []).append(record)

    for index, ((image_path, mask_path), source_records) in enumerate(records_by_source.items(), 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")

        for record in source_records:
            image_patch, mask_patch = crop_record_from_arrays(
                record=record,
                image=image,
                mask=mask,
            )
            polygons, skipped = write_yolo_patch(
                record=record,
                image_patch=image_patch,
                mask_patch=mask_patch,
                yolo_root=yolo_root,
                class_id=class_id,
                min_area=min_area,
                epsilon_ratio=epsilon_ratio,
                min_points=min_points,
                dry_run=dry_run,
            )
            polygon_count += polygons
            skipped_contours += skipped

        if index % 10 == 0 or index == len(records_by_source):
            print(
                f"[{split}] exported source images: "
                f"{index}/{len(records_by_source)}",
                flush=True,
            )

    return polygon_count, skipped_contours


def write_data_yaml(
    yolo_root: Path,
    class_id: int,
    class_name: str,
    dry_run: bool,
) -> None:
    """
        Write Ultralytics data.yaml inside the final YOLO dataset root.
    """

    if dry_run:
        return

    yaml_text = "\n".join(
        [
            f"path: {yolo_root.resolve()}",
            "train: train/images",
            "val: valid/images",
            "valid: valid/images",
            "test: test/images",
            "",
            "names:",
            f"  {class_id}: {class_name}",
            "",
        ]
    )
    (yolo_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def prepare_yolo_dataset(args: argparse.Namespace) -> list[SplitStats]:
    """
        Run the full metadata-first YOLO dataset preparation pipeline.
    """

    src_root = resolve_project_path(args.src)
    output_root = resolve_project_path(args.output_root)
    split_output_root = (
        resolve_project_path(args.split_output_root)
        if args.split_output_root is not None
        else None
    )
    yolo_root = output_root

    if split_output_root is not None and split_output_root.resolve() == output_root.resolve():
        raise ValueError("--split-output-root must be different from --output-root.")

    stride = validate_patch_settings(
        patch_size=args.patch_size,
        overlap=args.overlap,
    )
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
        output_root=output_root,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    if split_output_root is not None:
        prepare_split_output_root(
            split_output_root=split_output_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        export_full_split_dataset(
            split_map=split_map,
            split_output_root=split_output_root,
            dry_run=args.dry_run,
        )
        write_split_metadata(
            split_map=split_map,
            split_output_root=split_output_root,
            dry_run=args.dry_run,
        )

    negative_seed = args.negative_seed if args.negative_seed is not None else args.seed

    negative_ratios = {
        "train": args.train_negative_ratio,
        "valid": args.valid_negative_ratio,
        "test": args.test_negative_ratio,
    }

    records_by_split: dict[str, list[PatchRecord]] = {}
    for split in SPLITS:
        records_by_split[split] = collect_split_patch_records(
            split=split,
            pairs=split_map[split],
            patch_size=args.patch_size,
            stride=stride,
            negative_ratio=negative_ratios[split],
            seed=negative_seed,
        )

    stats: list[SplitStats] = []
    for split in SPLITS:
        if args.dry_run:
            polygon_count = 0
            skipped_contours = 0
        else:
            polygon_count, skipped_contours = export_split_records(
                split=split,
                records=records_by_split[split],
                yolo_root=yolo_root,
                class_id=args.class_id,
                min_area=args.min_area,
                epsilon_ratio=args.epsilon_ratio,
                min_points=args.min_points,
                dry_run=args.dry_run,
            )

        positive_count = sum(1 for record in records_by_split[split] if record.has_scratch)
        total_count = len(records_by_split[split])
        stats.append(
            SplitStats(
                split=split,
                source_images=len(split_map[split]),
                selected_patches=total_count,
                positive_patches=positive_count,
                negative_patches=total_count - positive_count,
                polygons=polygon_count,
                skipped_contours=skipped_contours,
            )
        )

    write_data_yaml(
        yolo_root=yolo_root,
        class_id=args.class_id,
        class_name=args.class_name,
        dry_run=args.dry_run,
    )

    return stats


def parse_args() -> argparse.Namespace:
    """
        Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Prepare a YOLO scratch segmentation dataset."
    )

    """Input and output paths"""
    parser.add_argument("--src", type=Path, default=LABELING_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-output-root", type=Path, default=None)

    """Split settings"""
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.10)

    """Patch settings"""
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.25)

    """Negative sampling settings"""
    parser.add_argument("--train-negative-ratio", type=float, default=1.0)
    parser.add_argument("--valid-negative-ratio", type=float, default=1.5)
    parser.add_argument("--test-negative-ratio", type=float, default=1.5)

    """YOLO label conversion settings"""
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--class-name", type=str, default="scratch")
    parser.add_argument("--min-area", type=float, default=0.0)
    parser.add_argument("--epsilon-ratio", type=float, default=0.0)
    parser.add_argument("--min-points", type=int, default=3)

    """Runtime settings"""
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.patch_size < 1:
        parser.error("--patch-size must be positive")
    if args.class_id < 0:
        parser.error("--class-id must be non-negative")
    if args.min_area < 0:
        parser.error("--min-area must be non-negative")
    if args.epsilon_ratio < 0:
        parser.error("--epsilon-ratio must be non-negative")
    if args.min_points < 3:
        parser.error("--min-points must be at least 3")
    for name in ("train_negative_ratio", "valid_negative_ratio", "test_negative_ratio"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")

    return args


def main() -> None:
    """
        Prepare YOLO data and print a compact summary.
    """

    args = parse_args()
    stats = prepare_yolo_dataset(args)

    output_root = resolve_project_path(args.output_root)
    split_output_root = (
        resolve_project_path(args.split_output_root)
        if args.split_output_root is not None
        else None
    )
    mode = "DRY RUN" if args.dry_run else "DONE"

    print(f"{mode}: YOLO dataset")
    print(f"Source: {resolve_project_path(args.src)}")
    print(f"Output root: {output_root}")
    print(f"YOLO data: {output_root / 'data.yaml'}")
    if split_output_root is not None:
        print(f"Full split output: {split_output_root}")

    for row in stats:
        print(
            f"{row.split}: source_images={row.source_images} "
            f"patches={row.selected_patches} "
            f"positive={row.positive_patches} "
            f"negative={row.negative_patches} "
            f"polygons={row.polygons} "
            f"skipped_contours={row.skipped_contours}"
        )

    if not args.dry_run:
        print("Train with:")
        print(
            "  .venv/bin/python -m src.yolo.scratch.train "
            f"--data {output_root / 'data.yaml'} "
            "--model yolo26s-seg.pt --imgsz 512 --batch 8 "
            "--epochs 150 --patience 40 --lr 0.001"
        )


if __name__ == "__main__":
    main()
