# Surface Scratch Detection

A surface scratch inspection project built with U-Net, PyTorch, OpenCV, and
PySide6. The current pipeline focuses on binary scratch segmentation: the model
predicts a `scratch/background` mask from surface images and saves visual
prediction results for inspection.

## Pipeline

1. Prepare surface images and binary scratch masks.
2. Split the dataset into train, validation, and test sets.
3. Generate image patches for U-Net training when needed.
4. Train a U-Net segmentation model on scratch masks.
5. Evaluate the checkpoint with IoU, Dice, precision, recall, specificity, and
   pixel accuracy.
6. Run sliding-window or full-image inference and save the prediction overlay.

## Project Structure

```text
Surface-Scratch-Detection/
├── configs/
│   ├── data.py               # Dataset paths, splits, and class schema
│   ├── path.py               # Shared project paths
│   └── unet.py               # U-Net model, training, and inference defaults
├── data/
│   ├── raw/                  # Original surface images
│   ├── scratch/              # Full-image scratch dataset
│   └── scratch_patches/      # Patch dataset for U-Net training
├── gui/                      # PySide6 learning and UI examples
├── models/
│   └── unet/                 # U-Net checkpoints
├── notebooks/
│   └── train_unet.ipynb      # Notebook training workflow
├── outputs/
│   ├── metrics/              # Evaluation reports
│   └── unet_v3/              # Inference outputs
├── src/
│   ├── labeling/             # Smart labeling tools
│   ├── onnx/                 # ONNX export
│   ├── openvino/             # OpenVINO export and benchmark scripts
│   ├── processing/           # Patch creation and reconstruction utilities
│   ├── qc/                   # Mask quality checks
│   └── unet/                 # U-Net train, evaluation, and inference code
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.12
- pip
- CUDA-capable GPU is optional

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Model Checkpoints

The default U-Net checkpoint is configured in `configs/unet.py`:

```text
models/
└── unet/
    └── unet_v3/
        └── best.pth
```

## Training

Train U-Net with the default configuration:

```bash
python -m src.unet.train
```

Train with explicit dataset and output folder:

```bash
python -m src.unet.train \
  --data data/scratch_patches \
  --save_dir models/unet/train_v1
```

## Evaluation

Evaluate the default checkpoint on the test split:

```bash
python -m src.unet.evaluation --split test
```

Outputs are saved to `outputs/metrics/unet/`:

- `metrics.json`
- `per_image_metrics.csv`

## Inference

Run U-Net inference on the default image:

```bash
python -m src.unet.inference
```

Run inference on a specific image:

```bash
python -m src.unet.inference \
  --image data/raw/Image_20260714163537759.bmp \
  --output-dir outputs/unet_v3
```

Use full-image resized inference for a fast baseline:

```bash
python -m src.unet.inference --mode full
```

## Configuration

| File | Purpose |
|---|---|
| `configs/path.py` | Project directories |
| `configs/data.py` | Dataset paths, split names, and class values |
| `configs/unet.py` | U-Net checkpoint, training, evaluation, and inference defaults |

## Tech Stack

- PyTorch
- U-Net
- OpenCV
- Albumentations
- ONNX Runtime
- OpenVINO
- PySide6
