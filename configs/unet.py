"""U-Net defaults for surface scratch binary segmentation."""

from configs.data import SCRATCH_TRAIN_DATASET
from configs.path import ROOT, UNET_OUTPUT_DIR as DEFAULT_UNET_OUTPUT_DIR, UNET_RUNS_DIR


"""Model registry"""
UNET_MODEL_VERSION = 3
UNET_MODEL  = UNET_RUNS_DIR / f"unet_v{UNET_MODEL_VERSION}" / "best.pth"
UNET_MODELS = {
    f"scratch_unet_v{version}": UNET_RUNS_DIR / f"unet_v{version}" / "best.pth"
    for version in range(1, 2)
}
UNET_MODEL_LABELS = {
    model_id: f"Scratch U-Net V{index}"
    for index, model_id in enumerate(UNET_MODELS, start=1)
}

"""Architecture and preprocessing defaults"""
UNET_INPUT_CHANNELS = 3
UNET_NUM_CLASSES    = 1
UNET_BASE_CHANNELS  = 32
UNET_NORM_TYPE      = "group"
UNET_IMAGE_SIZE     = 512
UNET_THRESHOLD      = 0.5
UNET_METRIC_THRESHOLD = 0.5
UNET_METRIC_THRES = UNET_METRIC_THRESHOLD
UNET_DEVICE         = "auto"
UNET_DEFAULT_IMAGE  = ROOT / "data" / "raw" / "Image_20260714163537759.bmp"

"""Image normalization values"""
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

"""Training defaults"""
UNET_TRAIN_DATA     = SCRATCH_TRAIN_DATASET
UNET_EPOCHS         = 100
UNET_BATCH_SIZE     = 8
UNET_LEARNING_RATE  = 1e-3
UNET_MIN_LEARNING_RATE = 1e-6
UNET_MIN_LR = UNET_MIN_LEARNING_RATE
UNET_WEIGHT_DECAY   = 1e-4
UNET_NUM_WORKERS    = 2
UNET_DICE_ALPHA     = 0.3
UNET_MAX_GRADIENT_NORM = 1.0
UNET_TRAIN_OUTPUT   = UNET_RUNS_DIR / "train_v1"

"""Tile settings for future preprocessing"""
UNET_TILE_SIZE      = 512
UNET_TILE_STRIDE    = 384
UNET_TILE_OVERLAP   = 0.25

"""Sliding-window inference defaults"""
UNET_INFERENCE_TILE_SIZE        = 512
UNET_INFERENCE_TILE_OVERLAP     = 0.15
UNET_INFERENCE_PATCH_BATCH_SIZE = 1

"""Online augmentation defaults"""
UNET_HORIZONTAL_FLIP_PROBABILITY = 0.5
UNET_VERTICAL_FLIP_PROBABILITY   = 0.5
UNET_ROTATION_LIMIT              = 10
UNET_ROTATION_PROBABILITY        = 0.4
UNET_CROP_RATIO                  = 0.9
UNET_CROP_PROBABILITY            = 0.0
UNET_BRIGHTNESS_LIMIT            = 0.2
UNET_CONTRAST_LIMIT              = 0.2
UNET_BRIGHTNESS_PROBABILITY      = 0.5
UNET_NOISE_STD_RANGE             = (0.02, 0.08)
UNET_NOISE_PROBABILITY           = 0.3
UNET_BLUR_LIMIT                  = (3, 3)
UNET_BLUR_PROBABILITY            = 0.15
UNET_CLAHE_CLIP_LIMIT            = 2.0
UNET_CLAHE_PROBABILITY           = 0.3

"""Scratch mask post-processing defaults"""
UNET_OPEN_KERNEL_SIZE            = 1
UNET_CLOSE_KERNEL_SIZE           = 3
UNET_MIN_AREA_PIXELS             = 8
UNET_MIN_AREA_RATIO              = 0.0
UNET_MIN_LARGEST_RATIO           = 0.0
UNET_FILL_HOLES                  = False

"""Inference output"""
UNET_OUTPUT_DIR = DEFAULT_UNET_OUTPUT_DIR
