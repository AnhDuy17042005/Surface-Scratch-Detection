"""
    Reconstruct Roboflow scratch export into U-Net folder structure.

    Input:
        data/scratch/train/
            img001.jpg
            img001_mask.png

    Output:
        data/scratch/train/
            images/img001.jpg
            masks/img001.png
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.data import IMAGE_EXTENSIONS, MASK_EXTENSION, SCRATCH_TRAIN_DATASET, SPLITS


def is_mask_file(path: Path, mask_suffix: str) -> bool:
    """
        Check whether a file is a Roboflow mask file.
    """

    """Roboflow segmentation masks use image_stem + _mask.png"""
    return path.suffix.lower() == MASK_EXTENSION and path.stem.endswith(mask_suffix)


def collect_flat_images(split_dir: Path, mask_suffix: str) -> list[Path]:
    """
        Collect image files from a flat Roboflow split folder.
    """

    """Read only files placed directly inside the split folder"""
    image_paths = []

    for path in split_dir.iterdir():
        """Skip folders such as images/ and masks/"""
        if not path.is_file():
            continue

        """Skip CSV, README, and other metadata files"""
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        """Skip mask files so they are not treated as training images"""
        if is_mask_file(path, mask_suffix):
            continue

        image_paths.append(path)

    """Keep a stable deterministic order"""
    return sorted(image_paths)


def corresponding_mask_path(
    image_path: Path,
    mask_suffix: str,
) -> Path:
    """
        Get the Roboflow mask path for one image.
    """

    """Roboflow keeps mask beside image with _mask suffix"""
    return image_path.with_name(f"{image_path.stem}{mask_suffix}{MASK_EXTENSION}")


def transfer_file(
    source_path: Path,
    destination_path: Path,
    mode: str,
    overwrite: bool,
) -> None:
    """
        Copy or move one file to its destination.
    """

    """Create destination folder before transferring file"""
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    """Avoid accidental overwrite unless the user explicitly allows it"""
    if destination_path.exists():
        if not overwrite:
            return

        destination_path.unlink()

    """Move files when reconstructing the dataset in place"""
    if mode == "move":
        shutil.move(str(source_path), str(destination_path))
        return

    """Copy files when the user wants to preserve the original flat export"""
    shutil.copy2(source_path, destination_path)


def transfer_image_mask_pair(
    image_path: Path,
    mask_path: Path,
    image_output_dir: Path,
    mask_output_dir: Path,
    mode: str,
    overwrite: bool,
) -> None:
    """
        Transfer one image-mask pair to U-Net folder structure.
    """

    """Keep original image filename"""
    output_image_path = image_output_dir / image_path.name

    """Normalize mask filename to match image stem for ScratchDataset"""
    output_mask_path = mask_output_dir / f"{image_path.stem}{MASK_EXTENSION}"

    """Transfer image first because mask name depends on the image stem"""
    transfer_file(
        source_path=image_path,
        destination_path=output_image_path,
        mode=mode,
        overwrite=overwrite,
    )

    """Transfer mask after image"""
    transfer_file(
        source_path=mask_path,
        destination_path=output_mask_path,
        mode=mode,
        overwrite=overwrite,
    )


def validate_structured_split(split_dir: Path) -> tuple[int, int]:
    """
        Validate one structured U-Net split folder.
    """

    """Expected U-Net folders"""
    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"

    """If structured folders are missing, this split is not ready yet"""
    if not image_dir.is_dir() or not mask_dir.is_dir():
        return 0, 0

    """Collect images after reconstruction"""
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    missing = 0

    """Every image must have a same-stem PNG mask"""
    for image_path in image_paths:
        mask_path = mask_dir / f"{image_path.stem}{MASK_EXTENSION}"

        if not mask_path.exists():
            print(f"Missing mask after reconstruct: {mask_path}")
            missing += 1

    return len(image_paths), missing


def reconstruct_split(
    split_dir: Path,
    mask_suffix: str,
    mode: str,
    overwrite: bool,
) -> tuple[int, int]:
    """
        Reconstruct one split folder from flat Roboflow format to U-Net format.
    """

    """Output folders for U-Net training"""
    image_output_dir = split_dir / "images"
    mask_output_dir  = split_dir / "masks"

    """Collect flat image files from Roboflow export"""
    image_paths = collect_flat_images(split_dir, mask_suffix)

    """If no flat files exist, only validate the current structured split"""
    if not image_paths:
        image_count, missing_count = validate_structured_split(split_dir)
        return image_count, missing_count

    copied_count = 0
    missing_count = 0

    """Transfer each flat image and its corresponding mask"""
    for image_path in image_paths:
        mask_path = corresponding_mask_path(
            image_path=image_path,
            mask_suffix=mask_suffix,
        )

        """Skip incomplete image-mask pairs"""
        if not mask_path.exists():
            print(f"Missing mask for: {image_path}")
            print(f"Expected mask : {mask_path}")
            missing_count += 1
            continue

        """Transfer pair into images/ and masks/"""
        transfer_image_mask_pair(
            image_path=image_path,
            mask_path=mask_path,
            image_output_dir=image_output_dir,
            mask_output_dir=mask_output_dir,
            mode=mode,
            overwrite=overwrite,
        )

        copied_count += 1

    return copied_count, missing_count


def reconstruct_scratch_dataset(
    dataset_root: str | Path,
    subsets: tuple[str, ...] = SPLITS,
    mask_suffix: str = "_mask",
    mode: str = "move",
    overwrite: bool = False,
) -> None:
    """
        Reconstruct scratch dataset into U-Net train/valid/test structure.
    """

    """Convert dataset path to Path object"""
    dataset_root = Path(dataset_root)

    """Check dataset root folder"""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    """Process every split independently"""
    for subset in subsets:
        split_dir = dataset_root / subset

        """Skip missing split folders"""
        if not split_dir.exists():
            print(f"{subset}: skip, folder not found")
            continue

        """Reconstruct or validate this split"""
        pair_count, missing_count = reconstruct_split(
            split_dir=split_dir,
            mask_suffix=mask_suffix,
            mode=mode,
            overwrite=overwrite,
        )

        """Print split summary"""
        print(f"{subset}: pairs={pair_count}, missing_masks={missing_count}")

    """Print final summary"""
    print("Done reconstructing scratch dataset.")
    print(f"Dataset root: {dataset_root}")


def parse_args() -> Namespace:
    """
        Build command-line arguments.
    """

    """Create parser for direct script usage"""
    parser = ArgumentParser(
        description="Reconstruct data/scratch into train/valid/test images/masks folders."
    )

    """Dataset root defaults to configs.data.SCRATCH_TRAIN_DATASET"""
    parser.add_argument(
        "--root",
        type=Path,
        default=SCRATCH_TRAIN_DATASET,
        help="Scratch dataset root containing train, valid, and test folders.",
    )

    """Choose move for in-place reconstruct or copy for preserving flat export"""
    parser.add_argument(
        "--mode",
        choices=("move", "copy"),
        default="move",
        help="Use move for in-place reconstruction or copy to preserve flat files.",
    )

    """Allow replacing files in images/ and masks/ if they already exist"""
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in images/ and masks/.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    """Run scratch dataset reconstruction script."""
    args = parse_args()

    reconstruct_scratch_dataset(
        dataset_root=args.root,
        mode=args.mode,
        overwrite=args.overwrite,
    )
