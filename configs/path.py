"""Shared project paths for Surface Scratch Detection."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

"""Top-level project folders"""
CONFIG_DIR = ROOT / "configs"
SRC_DIR    = ROOT / "src"
ASSETS_DIR = ROOT / "assets"

DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

"""Model and output folders"""
UNET_RUNS_DIR   = MODELS_DIR / "unet"
UNET_OUTPUT_DIR = OUTPUTS_DIR / "unet_v3"
METRICS_DIR     = OUTPUTS_DIR / "metrics"
LABELING_OUTPUT_DIR = OUTPUTS_DIR / "labeling"
