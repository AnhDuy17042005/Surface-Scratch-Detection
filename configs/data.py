"""Dataset paths and schema for surface scratch segmentation."""

from configs.path import DATA_DIR, OUTPUTS_DIR


"""Dataset split names used by training and evaluation"""
SPLITS = ("train", "valid", "test")

"""Supported image and mask formats"""
IMAGE_EXTENSIONS = frozenset({".bmp", ".png", ".jpg", ".jpeg", ".webp"})
MASK_EXTENSION   = ".png"

"""Default scratch patch dataset used by legacy/evaluation scripts"""
SCRATCH_DATASET = DATA_DIR / "scratch_patches"
SCRATCH_TRAIN_DATASET = SCRATCH_DATASET

"""Manual labeling output before train/valid/test split"""
LABELING_DATASET   = OUTPUTS_DIR / "labeling"
LABELING_IMAGE_DIR = LABELING_DATASET / "images"
LABELING_MASK_DIR  = LABELING_DATASET / "masks"

"""Class schema for binary scratch segmentation"""
BACKGROUND_VALUE = 0
SCRATCH_VALUE = 255
CLASS_NAMES = {
    0: "background",
    1: "scratch",
}
