"""
    Sliding-window inference utilities for U-Net patch models.

    Train-time patch models should be inferred with the same patch size and
    overlap used during patch generation. Probability maps are merged first;
    thresholding is applied only after the full probability map is reconstructed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.unet import (
    IMAGENET_MEAN as IMAGENET_MEAN_VALUES,
    IMAGENET_STD as IMAGENET_STD_VALUES,
)


IMAGENET_MEAN = np.asarray(IMAGENET_MEAN_VALUES, dtype=np.float32)
IMAGENET_STD = np.asarray(IMAGENET_STD_VALUES, dtype=np.float32)


def validate_sliding_window_settings(
    patch_size: int,
    overlap: float,
    batch_size: int,
) -> int:
    """
        Validate sliding-window settings and return stride.
    """

    """Patch size must be positive"""
    if patch_size < 1:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

    """Overlap must keep a positive stride"""
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    """Patch batch size must be positive"""
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    """Convert overlap ratio to stride"""
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
        Generate top-left coordinates for one image axis.

        The final coordinate is aligned with the far edge so border pixels are
        included even when image size is not divisible by stride.
    """

    """Small padded images still produce one full patch"""
    if size <= patch_size:
        return [0]

    """Generate regular sliding-window coordinates"""
    coords = list(range(0, size - patch_size + 1, stride))

    """Append final edge-aligned coordinate when needed"""
    last = size - patch_size

    if coords[-1] != last:
        coords.append(last)

    return coords


def sliding_window_grid_shape(
    image_shape: tuple[int, ...],
    patch_size: int,
    overlap: float,
    batch_size: int,
) -> tuple[int, int, int]:
    """
        Return sliding-window grid as (num_x, num_y, stride).
    """

    """Validate settings and compute stride"""
    stride = validate_sliding_window_settings(
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
    )

    """Pad small images consistently with inference"""
    h, w = image_shape[:2]
    padded_h = max(h, patch_size)
    padded_w = max(w, patch_size)

    """Generate coordinates for both axes"""
    ys = sliding_window_coords(
        size=padded_h,
        patch_size=patch_size,
        stride=stride,
    )
    xs = sliding_window_coords(
        size=padded_w,
        patch_size=patch_size,
        stride=stride,
    )

    return len(xs), len(ys), stride


def pad_image_to_patch_size(
    image: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
        Pad image to guarantee at least one full patch.

        Returns:
            padded image
            original size as (height, width)
    """

    """Store original image size"""
    h, w = image.shape[:2]

    """Compute required padding"""
    pad_h = max(patch_size - h, 0)
    pad_w = max(patch_size - w, 0)

    if pad_h == 0 and pad_w == 0:
        return image, (h, w)

    """Use reflected border so padded pixels look like local texture"""
    padded = cv2.copyMakeBorder(
        image,
        top=0,
        bottom=pad_h,
        left=0,
        right=pad_w,
        borderType=cv2.BORDER_REFLECT_101,
    )

    return padded, (h, w)


def preprocess_patch_batch(
    patches: list[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    """
        Convert BGR patches to normalized NCHW tensor batch.
    """

    """Normalize each patch with the same preprocessing used in training"""
    batch = []

    for patch in patches:
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        normalized = (
            rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN
        ) / IMAGENET_STD
        batch.append(normalized.transpose(2, 0, 1))

    """Stack patches into NCHW tensor"""
    tensor = torch.from_numpy(np.stack(batch, axis=0))

    return tensor.to(device)


@torch.no_grad()
def predict_patch_batch_probability(
    model: torch.nn.Module,
    patches: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """
        Predict probability maps for one batch of image patches.
    """

    """Preprocess patch batch"""
    tensor = preprocess_patch_batch(
        patches=patches,
        device=device,
    )

    """Run U-Net and convert logits to probabilities"""
    logits = model(tensor)
    probabilities = torch.sigmoid(logits)[:, 0].detach().cpu().numpy()

    return probabilities.astype(np.float32)


@torch.no_grad()
def predict_sliding_window_probability(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: int,
    overlap: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
        Predict full-image probability map using sliding-window patches.
    """

    """Validate settings and compute stride"""
    stride = validate_sliding_window_settings(
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
    )

    """Pad small images and keep original size for final crop"""
    padded_image, (original_h, original_w) = pad_image_to_patch_size(
        image=image,
        patch_size=patch_size,
    )

    h, w = padded_image.shape[:2]

    """Generate patch coordinates"""
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

    """Accumulate probability and overlap counts"""
    full_prob = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    patch_batch: list[np.ndarray] = []
    coord_batch: list[tuple[int, int]] = []

    def flush_batch() -> None:
        """
            Predict current batch and merge probabilities into full map.
        """

        if not patch_batch:
            return

        probabilities = predict_patch_batch_probability(
            model=model,
            patches=patch_batch,
            device=device,
        )

        for probability, (x, y) in zip(probabilities, coord_batch):
            full_prob[y:y + patch_size, x:x + patch_size] += probability
            count_map[y:y + patch_size, x:x + patch_size] += 1.0

        patch_batch.clear()
        coord_batch.clear()

    """Create patch batches from sliding-window grid"""
    for y in ys:
        for x in xs:
            patch = padded_image[y:y + patch_size, x:x + patch_size]
            patch_batch.append(patch)
            coord_batch.append((x, y))

            if len(patch_batch) >= batch_size:
                flush_batch()

    """Predict final partial batch"""
    flush_batch()

    """Average overlapped probabilities"""
    count_map[count_map == 0.0] = 1.0
    full_prob = full_prob / count_map

    """Crop away padding"""
    return full_prob[:original_h, :original_w]


@torch.no_grad()
def predict_sliding_window_mask(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: int,
    overlap: float,
    threshold: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
        Predict full-image binary mask using sliding-window patches.
    """

    """Predict merged probability map"""
    probability = predict_sliding_window_probability(
        model=model,
        image=image,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
        device=device,
    )

    """Threshold after merging, not per patch"""
    return (probability >= threshold).astype(np.uint8) * 255
