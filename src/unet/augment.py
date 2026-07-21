"""
    Augmentation script for scratch full-image dataset.
"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.unet import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    UNET_BLUR_LIMIT,
    UNET_BLUR_PROBABILITY,
    UNET_BRIGHTNESS_LIMIT,
    UNET_BRIGHTNESS_PROBABILITY,
    UNET_CLAHE_CLIP_LIMIT,
    UNET_CLAHE_PROBABILITY,
    UNET_CONTRAST_LIMIT,
    UNET_CROP_PROBABILITY,
    UNET_CROP_RATIO,
    UNET_HORIZONTAL_FLIP_PROBABILITY,
    UNET_IMAGE_SIZE,
    UNET_NOISE_PROBABILITY,
    UNET_NOISE_STD_RANGE,
    UNET_ROTATION_LIMIT,
    UNET_ROTATION_PROBABILITY,
    UNET_VERTICAL_FLIP_PROBABILITY,
)


def get_train_transforms(img_size: int = UNET_IMAGE_SIZE):
    """
        Augmentation pipeline for training.
        Strategy: geometric transforms that preserve scratch structure +
                  photometric transforms to handle lighting variation.

        Heavy distortions are avoided because they can bend thin scratch masks
        into shapes that are not physically realistic.
    """
    return A.Compose([
        A.Resize(img_size, img_size),

        # Geometric transforms
        A.HorizontalFlip(p=UNET_HORIZONTAL_FLIP_PROBABILITY),
        A.VerticalFlip(p=UNET_VERTICAL_FLIP_PROBABILITY),
        A.Rotate(
            limit=UNET_ROTATION_LIMIT,
            border_mode=cv2.BORDER_REFLECT_101,
            p=UNET_ROTATION_PROBABILITY,
        ),
        A.CropNonEmptyMaskIfExists(
            height=int(img_size * UNET_CROP_RATIO),
            width=int(img_size * UNET_CROP_RATIO),
            p=UNET_CROP_PROBABILITY,
        ),
        A.Resize(img_size, img_size),

        # Photometric transforms
        A.RandomBrightnessContrast(
            brightness_limit=UNET_BRIGHTNESS_LIMIT,
            contrast_limit=UNET_CONTRAST_LIMIT,
            p=UNET_BRIGHTNESS_PROBABILITY,
        ),
        A.GaussNoise(std_range=UNET_NOISE_STD_RANGE, p=UNET_NOISE_PROBABILITY),
        A.GaussianBlur(blur_limit=UNET_BLUR_LIMIT, p=UNET_BLUR_PROBABILITY),
        A.CLAHE(clip_limit=UNET_CLAHE_CLIP_LIMIT, p=UNET_CLAHE_PROBABILITY),

        # Normalize to ImageNet stats
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = UNET_IMAGE_SIZE):
    """
        Transform for validation and test set.
        No random augmentation.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
