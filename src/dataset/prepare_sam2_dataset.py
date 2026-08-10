"""
    Prepare SAM2 one-frame dataset directly from labeled scratch masks.

    This script skips intermediate patch and instance folders. It performs all
    steps in memory:

        outputs/labeling
          images/
          masks/

        split full images
          -> crop patch candidates
          -> split each positive patch mask into connected components
          -> write SAM2 PNGRawDataset format

    Output:
        data/scratch_sam2_format/
          JPEGImages/<split>/<sample_id>/00000.png
          Annotations/<split>/<sample_id>/00000.png
          ImageSets/<split>.txt
"""

from __future__ import annotations

import argparse
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


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "scratch_sam2_format"
FRAME_IMAGE_NAME = "00000.png"
FRAME_MASK_NAME = "00000.png"


@dataclass(frozen=True)
class ImageMaskPair:
    """
        One labeled image and binary mask pair.

        Args:
            image_path: source image path
            mask_path : source binary mask path
    """

    image_path: Path
    mask_path: Path

    @property
    def stem(self) -> str:
        """Return deterministic source stem."""
        return self.image_path.stem


@dataclass(frozen=True)
class PatchRecord:
    """
        Metadata for one selected patch candidate.

        Args:
            split          : train, valid, or test
            source_image   : full image path
            source_mask    : full mask path
            source_stem    : full image stem
            patch_index    : patch index inside source image grid
            x, y           : top-left patch coordinate
            width, height  : patch spatial size
            positive_pixels: foreground pixels inside the patch
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
        """Return whether this patch contains scratch pixels."""
        return self.positive_pixels > 0

    @property
    def patch_stem(self) -> str:
        """Create deterministic patch stem."""
        return (
            f"{self.source_stem}_p{self.patch_index:03d}"
            f"_x{self.x:04d}_y{self.y:04d}"
        )


@dataclass(frozen=True)
class ComponentRecord:
    """
        One connected scratch component inside a patch.
    """

    component_id: int
    component_mask: np.ndarray
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    positive_pixels: int


@dataclass(frozen=True)
class SplitStats:
    """
        Summary for one SAM2 split.
    """

    split: str
    source_images: int
    positive_patches: int
    output_instances: int
    skipped_components: int


def resolve_project_path(path: Path) -> Path:
    """
        Resolve relative paths from the project root.
    """

    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def collect_pairs(src_root: Path) -> list[ImageMaskPair]:
    """
        Collect labeled image/mask pairs from outputs/labeling style data.
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

    pairs: list[ImageMaskPair] = []
    missing_masks: list[Path] = []
    seen_stems: set[str] = set()

    for image_path in image_paths:
        if image_path.stem in seen_stems:
            raise RuntimeError(f"Duplicate image stem found: {image_path.stem}")
        seen_stems.add(image_path.stem)

        mask_path = mask_dir / f"{image_path.stem}{MASK_EXTENSION}"
        if not mask_path.is_file():
            missing_masks.append(image_path)
            continue

        pairs.append(
            ImageMaskPair(
                image_path=image_path,
                mask_path=mask_path,
            )
        )

    if missing_masks:
        names = ", ".join(path.name for path in missing_masks[:5])
        extra = "" if len(missing_masks) <= 5 else f", ... ({len(missing_masks)} total)"
        raise RuntimeError(f"Missing masks for images: {names}{extra}")

    return sorted(pairs, key=lambda pair: pair.stem)


def validate_ratios(
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> tuple[float, float, float]:
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


def split_counts(
    total_count: int,
    ratios: tuple[float, float, float],
) -> dict[str, int]:
    """
        Convert split ratios into integer counts.
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
        Shuffle image/mask pairs and assign them to train/valid/test.
    """

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(pairs)).tolist()
    shuffled_pairs = [pairs[index] for index in indices]
    counts = split_counts(
        total_count=len(pairs),
        ratios=ratios,
    )

    train_end = counts["train"]
    valid_end = train_end + counts["valid"]

    return {
        "train": shuffled_pairs[:train_end],
        "valid": shuffled_pairs[train_end:valid_end],
        "test": shuffled_pairs[valid_end:],
    }


def validate_patch_settings(
    patch_size: int,
    overlap: float,
) -> int:
    """
        Validate patch settings and return stride.
    """

    if patch_size < 1:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

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
        Generate top-left sliding-window coordinates along one axis.
    """

    if size < patch_size:
        return []

    coords = list(range(0, size - patch_size + 1, stride))
    last = size - patch_size

    if coords[-1] != last:
        coords.append(last)

    return coords


def prepare_output_root(
    dst_root: Path,
    overwrite: bool,
    dry_run: bool,
) -> None:
    """
        Create a clean SAM2 output root.
    """

    if dry_run:
        return

    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {dst_root}. "
                "Use --overwrite or choose a new --dst."
            )
        shutil.rmtree(dst_root)

    (dst_root / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (dst_root / "Annotations").mkdir(parents=True, exist_ok=True)
    (dst_root / "ImageSets").mkdir(parents=True, exist_ok=True)


def load_image_and_mask(pair: ImageMaskPair) -> tuple[np.ndarray, np.ndarray]:
    """
        Load one full image and mask pair.
    """

    image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(pair.mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Cannot read image: {pair.image_path}")
    if mask is None:
        raise ValueError(f"Cannot read mask: {pair.mask_path}")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image and mask size mismatch: {pair.image_path.name} "
            f"{image.shape[:2]} != {mask.shape[:2]}"
        )

    return image, mask


def collect_patch_records(
    split: str,
    pair: ImageMaskPair,
    mask: np.ndarray,
    patch_size: int,
    stride: int,
) -> list[PatchRecord]:
    """
        Collect positive patch metadata from one full mask.
    """

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

    records: list[PatchRecord] = []
    patch_index = 0

    """Only positive patches are useful for one-object SAM2 fine-tuning."""
    for y in ys:
        for x in xs:
            mask_patch = mask[y:y + patch_size, x:x + patch_size]
            positive_pixels = int((mask_patch > 0).sum())

            if positive_pixels > 0:
                records.append(
                    PatchRecord(
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
                )

            patch_index += 1

    return records


def component_label_mask(
    mask: np.ndarray,
    close_kernel: int,
) -> np.ndarray:
    """
        Return the binary mask used for connected-component labeling.
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
) -> tuple[list[ComponentRecord], int]:
    """
        Split one patch mask into connected scratch components.
    """

    original_binary = mask > 0
    label_binary = component_label_mask(
        mask=mask,
        close_kernel=close_kernel,
    )
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        label_binary,
        connectivity=8,
    )

    components: list[ComponentRecord] = []
    skipped_components = 0

    for component_id in range(1, component_count):
        component_region = labels == component_id
        component_pixels = int((component_region & original_binary).sum())

        if component_pixels < min_pixels:
            skipped_components += 1
            continue

        component_mask = np.zeros_like(mask, dtype=np.uint8)
        component_mask[component_region & original_binary] = 1

        components.append(
            ComponentRecord(
                component_id=component_id,
                component_mask=component_mask,
                bbox_x=int(stats[component_id, cv2.CC_STAT_LEFT]),
                bbox_y=int(stats[component_id, cv2.CC_STAT_TOP]),
                bbox_w=int(stats[component_id, cv2.CC_STAT_WIDTH]),
                bbox_h=int(stats[component_id, cv2.CC_STAT_HEIGHT]),
                positive_pixels=component_pixels,
            )
        )

    return components, skipped_components


def crop_patch(
    image: np.ndarray,
    mask: np.ndarray,
    record: PatchRecord,
) -> tuple[np.ndarray, np.ndarray]:
    """
        Crop one image/mask patch from loaded full arrays.
    """

    y1 = record.y
    y2 = record.y + record.height
    x1 = record.x
    x2 = record.x + record.width

    image_patch = image[y1:y2, x1:x2]
    mask_patch = mask[y1:y2, x1:x2]

    if image_patch.shape[:2] != (record.height, record.width):
        raise ValueError(
            f"Invalid image patch shape for {record.patch_stem}: "
            f"{image_patch.shape[:2]}"
        )
    if mask_patch.shape[:2] != (record.height, record.width):
        raise ValueError(
            f"Invalid mask patch shape for {record.patch_stem}: "
            f"{mask_patch.shape[:2]}"
        )

    return image_patch, mask_patch


def write_sam2_sample(
    dst_root: Path,
    split: str,
    sample_id: str,
    image_patch: np.ndarray,
    component_mask: np.ndarray,
    dry_run: bool,
) -> None:
    """
        Write one static patch as a one-frame SAM2 sample.
    """

    if dry_run:
        return

    image_path = dst_root / "JPEGImages" / split / sample_id / FRAME_IMAGE_NAME
    mask_path = dst_root / "Annotations" / split / sample_id / FRAME_MASK_NAME
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(image_path), image_patch):
        raise RuntimeError(f"Failed to save image frame: {image_path}")
    if not cv2.imwrite(str(mask_path), component_mask):
        raise RuntimeError(f"Failed to save mask frame: {mask_path}")


def write_file_list(
    dst_root: Path,
    split: str,
    video_names: list[str],
    dry_run: bool,
) -> None:
    """
        Write ImageSets/<split>.txt for PNGRawDataset.
    """

    if dry_run:
        return

    text = "\n".join(video_names)
    if text:
        text += "\n"

    (dst_root / "ImageSets" / f"{split}.txt").write_text(text, encoding="utf-8")


def process_split(
    dst_root: Path,
    split: str,
    pairs: list[ImageMaskPair],
    patch_size: int,
    stride: int,
    min_pixels: int,
    close_kernel: int,
    dry_run: bool,
) -> SplitStats:
    """
        Process one split from full image/mask pairs to SAM2 samples.
    """

    positive_patches = 0
    output_instances = 0
    skipped_components = 0
    video_names: list[str] = []

    for pair_index, pair in enumerate(pairs, start=1):
        image, mask = load_image_and_mask(pair)
        patch_records = collect_patch_records(
            split=split,
            pair=pair,
            mask=mask,
            patch_size=patch_size,
            stride=stride,
        )
        positive_patches += len(patch_records)

        for record in patch_records:
            image_patch, mask_patch = crop_patch(
                image=image,
                mask=mask,
                record=record,
            )
            components, skipped = iter_components(
                mask=mask_patch,
                min_pixels=min_pixels,
                close_kernel=close_kernel,
            )
            skipped_components += skipped

            for instance_index, component in enumerate(components, start=1):
                sample_id = f"{record.patch_stem}_i{instance_index:03d}"
                video_name = f"{split}/{sample_id}"

                write_sam2_sample(
                    dst_root=dst_root,
                    split=split,
                    sample_id=sample_id,
                    image_patch=image_patch,
                    component_mask=component.component_mask,
                    dry_run=dry_run,
                )

                video_names.append(video_name)
                output_instances += 1

        if pair_index % 10 == 0 or pair_index == len(pairs):
            print(
                f"[{split}] processed source images: {pair_index}/{len(pairs)}",
                flush=True,
            )

    write_file_list(
        dst_root=dst_root,
        split=split,
        video_names=video_names,
        dry_run=dry_run,
    )

    return SplitStats(
        split=split,
        source_images=len(pairs),
        positive_patches=positive_patches,
        output_instances=output_instances,
        skipped_components=skipped_components,
    )


def prepare_sam2_dataset(args: argparse.Namespace) -> list[SplitStats]:
    """
        Run the full SAM2 dataset preparation pipeline.
    """

    src_root = resolve_project_path(args.src)
    dst_root = resolve_project_path(args.dst)
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
        dst_root=dst_root,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    stats: list[SplitStats] = []
    for split in SPLITS:
        stats.append(
            process_split(
                dst_root=dst_root,
                split=split,
                pairs=split_map[split],
                patch_size=args.patch_size,
                stride=stride,
                min_pixels=args.min_pixels,
                close_kernel=args.close_kernel,
                dry_run=args.dry_run,
            )
        )

    return stats


def parse_args() -> argparse.Namespace:
    """
        Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Prepare SAM2 PNGRawDataset directly from labeled masks."
    )

    """Input and output paths"""
    parser.add_argument("--src", type=Path, default=LABELING_DATASET)
    parser.add_argument("--dst", type=Path, default=DEFAULT_OUTPUT_ROOT)

    """Split settings"""
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.10)

    """Patch settings"""
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.25)

    """Instance settings"""
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=20,
        help="Skip connected components with fewer foreground pixels.",
    )
    parser.add_argument(
        "--close-kernel",
        type=int,
        default=0,
        help="Optional morphology close kernel before component labeling.",
    )

    """Runtime settings"""
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.patch_size < 1:
        parser.error("--patch-size must be positive")
    if args.min_pixels < 1:
        parser.error("--min-pixels must be positive")
    if args.close_kernel < 0:
        parser.error("--close-kernel must be non-negative")

    return args


def main() -> None:
    """
        Prepare SAM2 dataset and print a compact summary.
    """

    args = parse_args()
    stats = prepare_sam2_dataset(args)

    dst_root = resolve_project_path(args.dst)
    mode = "DRY RUN" if args.dry_run else "DONE"

    print(f"{mode}: SAM2 dataset")
    print(f"Source: {resolve_project_path(args.src)}")
    print(f"Output root: {dst_root}")
    print(f"img_folder: {dst_root / 'JPEGImages'}")
    print(f"gt_folder: {dst_root / 'Annotations'}")
    print(f"train list: {dst_root / 'ImageSets' / 'train.txt'}")

    for row in stats:
        print(
            f"{row.split}: source_images={row.source_images} "
            f"positive_patches={row.positive_patches} "
            f"instances={row.output_instances} "
            f"skipped_components={row.skipped_components}"
        )


if __name__ == "__main__":
    main()
