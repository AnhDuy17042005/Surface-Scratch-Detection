# Surface Scratch Detection

A surface scratch inspection system built with Ultralytics YOLO26, SAM2-assisted
labeling, OpenCV, and PySide6. The project focuses on detecting thin scratch
defects on metal/component surfaces. YOLO component detection first finds the
main part ROI, then YOLO scratch segmentation runs on ROI tiles to preserve fine
scratch details without resizing the whole high-resolution image.

The repository includes dataset preparation, training, evaluation, optimized
runtime export, command-line inference, and a desktop GUI for annotation,
training, and prediction review.

## Pipeline

1. Collect raw surface images under `data/raw/`.
2. Label scratches in the PySide6 Annotation tab with SAM2 point/box assistance.
3. Save reviewed image-mask pairs under `outputs/labeling/`.
4. Prepare a YOLO instance-segmentation dataset directly from labeled masks.
5. Train YOLO26 scratch segmentation from the GUI, CLI, or Colab notebook.
6. Detect the component ROI with a component YOLO detector.
7. Run scratch YOLO segmentation on ROI tiles.
8. Save overlays, masks, metrics, and optionally send predictions back to the
   Annotation tab for human correction and future retraining.

## Model Roles

| Model | Task | Purpose |
|---|---|---|
| YOLO component detector | Object detection | Find the component/part bounding box ROI |
| YOLO scratch segmenter | Instance segmentation | Segment scratch masks inside full images or ROI tiles |
| SAM2 | Prompt-based segmentation | Assist human labeling and mask correction |

## User Interface

The desktop GUI is implemented with PySide6:

```bash
.venv/bin/python -m src.gui.main
```

Tabs:

| Tab | Purpose |
|---|---|
| `Inference` | Run component ROI + scratch YOLO inference and send predictions to Annotation |
| `Training` | Train scratch YOLO26 segmentation with local GPU/CPU settings |
| `Data Processing` | Prepare YOLO datasets from labeled image-mask pairs |
| `Annotation` | Label or refine scratches with SAM2 point/box prompts, brush, and eraser |
| `Camera` | Camera/image acquisition utilities |

## Project Structure

```text
Surface-Scratch-Detection/
├── assets/                    # Optional README/demo assets
├── configs/
│   ├── data.py                # Split names, image extensions, class schema
│   ├── path.py                # Shared project paths
│   └── yolo.py                # Component and scratch YOLO defaults
├── data/
│   ├── raw/                   # Original high-resolution images
│   ├── component/             # YOLO component detection dataset
│   ├── scratch_yolo_seg/      # YOLO scratch segmentation dataset
│   └── scratch_sam2_format/   # SAM2 fine-tuning dataset
├── models/
│   ├── yolo/
│   │   ├── component/         # Component detector checkpoint
│   │   ├── scratch_yolo26n_seg/
│   │   └── scratch_yolo26s_seg/
│   └── sam2/                  # SAM2 fine-tuned checkpoint
├── notebooks/
│   ├── train_yolo.ipynb       # Google Colab YOLO training workflow
│   └── train_sam2.ipynb       # Google Colab SAM2 fine-tuning workflow
├── outputs/
│   ├── labeling/              # Human-reviewed images and masks
│   ├── metrics/               # Evaluation JSON, CSV, and plots
│   ├── onnx/                  # Optional ONNX runtime experiment outputs
│   └── yolo/                  # YOLO inference outputs
├── src/
│   ├── dataset/               # End-to-end dataset preparation pipelines
│   ├── evaluation/            # YOLO and SAM2 evaluation scripts
│   ├── gui/                   # Main PySide6 application
│   ├── onnx/                  # Optional YOLO ONNX export/runtime experiments
│   ├── openvino/              # YOLO OpenVINO export and benchmark
│   ├── processing/            # Legacy/experimental processing helpers
│   └── yolo/
│       ├── component/         # Component detector train/inference/ROI helpers
│       └── scratch/           # Scratch segmentation train/inference scripts
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

`data/`, `models/`, and `outputs/` are runtime artifacts. They are intentionally
ignored by Git and mounted into Docker containers at runtime.

## Prerequisites

- Python 3.12
- pip
- Docker and Docker Compose, optional
- CUDA-capable GPU, optional for training/inference acceleration
- Linux X11 display, required for Docker GUI mode

## Installation

Create and activate a local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Runtime Artifacts

The active local model paths are configured in `configs/yolo.py`:

```text
models/
├── yolo/
│   ├── component/
│   │   └── weights/
│   │       └── best.pt
│   ├── scratch_yolo26n_seg/
│   │   └── weights/
│   │       └── best.pt
│   └── scratch_yolo26s_seg/
│       └── weights/
│           └── best.pt
└── sam2/
    └── checkpoint.pt
```

A clean Git clone does not include model weights or datasets. Copy or mount
`data/`, `models/`, and `outputs/` separately before running full inference.

## Run With Docker

The Docker image contains only code and dependencies. Datasets, checkpoints, and
outputs are mounted from the host:

```text
./data    -> /app/data
./models  -> /app/models
./outputs -> /app/outputs
```

Allow Docker containers to use the host X11 display:

```bash
xhost +local:docker
```

Build and run the GUI:

```bash
docker compose up --build surface-scratch-gui
```

Run a CLI command inside the same image:

```bash
docker compose --profile cli run --rm surface-scratch-cli
```

The default CLI command prints scratch YOLO inference help. Override it when
needed:

```bash
docker compose --profile cli run --rm surface-scratch-cli \
  python -m src.yolo.scratch.inference \
  --source data/raw \
  --mode sliding \
  --output-dir outputs/yolo/scratch
```

## Dataset Preparation

### Labeling Output

The Annotation tab writes reviewed samples to:

```text
outputs/labeling/
  images/
  masks/
```

### YOLO Scratch Segmentation Dataset

Prepare a YOLO instance-segmentation dataset directly from labeled masks:

```bash
.venv/bin/python -m src.dataset.prepare_yolo_dataset \
  --src outputs/labeling \
  --output-root data/scratch_yolo_seg \
  --train-ratio 0.70 \
  --valid-ratio 0.20 \
  --test-ratio 0.10 \
  --patch-size 512 \
  --overlap 0.25 \
  --train-negative-ratio 1.0 \
  --valid-negative-ratio 1.5 \
  --test-negative-ratio 1.5 \
  --seed 42 \
  --overwrite
```

Output:

```text
data/scratch_yolo_seg/
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  test/images/
  test/labels/
  data.yaml
```

### SAM2 Fine-tuning Dataset

Prepare the one-frame SAM2 dataset format:

```bash
.venv/bin/python -m src.dataset.prepare_sam2_dataset \
  --src outputs/labeling \
  --dst data/scratch_sam2_format \
  --train-ratio 0.70 \
  --valid-ratio 0.20 \
  --test-ratio 0.10 \
  --patch-size 1024 \
  --overlap 0.25 \
  --min-pixels 20 \
  --seed 42 \
  --overwrite
```

Output:

```text
data/scratch_sam2_format/
  JPEGImages/<split>/<sample_id>/00000.png
  Annotations/<split>/<sample_id>/00000.png
  ImageSets/train.txt
  ImageSets/valid.txt
  ImageSets/test.txt
```

## Training

### Scratch YOLO26 Segmentation

Train from the CLI:

```bash
.venv/bin/python -m src.yolo.scratch.train \
  --data data/scratch_yolo_seg/data.yaml \
  --model yolo26n-seg.pt \
  --imgsz 512 \
  --batch 16 \
  --epochs 100 \
  --patience 30 \
  --lr 0.002 \
  --device auto \
  --project models/yolo \
  --name scratch_yolo26n_seg
```

Common base models:

```text
yolo26n-seg.pt
yolo26s-seg.pt
yolo26m-seg.pt
yolo26l-seg.pt
yolo26x-seg.pt
```

Use `n` for speed-sensitive CPU/local inference. Use larger models only when
quality gains justify the slower runtime.

### Component YOLO Detection

Train the component detector:

```bash
.venv/bin/python -m src.yolo.component.train \
  --data data/component/data.yaml \
  --model yolo26n.pt \
  --imgsz 640 \
  --batch 8 \
  --epochs 100
```

### Notebook Training

Google Colab workflows are provided for longer training runs:

```text
notebooks/train_yolo.ipynb
notebooks/train_sam2.ipynb
```

The notebooks are designed for uploading datasets to Drive, training on Colab,
and copying checkpoints/plots back to Drive.

## Evaluation

### YOLO Scratch Segmentation

```bash
.venv/bin/python -m src.evaluation.yolo_scratch \
  --data data/scratch_yolo_seg \
  --split test
```

Outputs are saved under:

```text
outputs/metrics/yolo_scratch/
```

Typical outputs include:

- `summary.json`
- threshold curves
- metric bar plots
- confusion matrix plots
- prediction visualizations

### YOLO Component Detection

```bash
.venv/bin/python -m src.evaluation.yolo_component \
  --data data/component/data.yaml
```

Outputs are saved under:

```text
outputs/metrics/yolo_component/
```

### SAM2 Prompt Evaluation

```bash
.venv/bin/python -m src.evaluation.sam2 \
  --data data/scratch_sam2_format \
  --split test
```

Outputs are saved under:

```text
outputs/metrics/sam2/
```

## Inference

### Scratch YOLO Full Image or Sliding Tiles

Run scratch segmentation on raw images:

```bash
.venv/bin/python -m src.yolo.scratch.inference \
  --source data/raw \
  --model models/yolo/scratch_yolo26n_seg/weights/best.pt \
  --mode sliding \
  --tile-size 512 \
  --overlap 0.15 \
  --imgsz 512 \
  --conf 0.10 \
  --output-dir outputs/yolo/scratch
```

The sliding mode preserves thin-scratch detail by avoiding direct full-image
downscaling.

### Component Detection

```bash
.venv/bin/python -m src.yolo.component.inference \
  --source data/raw \
  --model models/yolo/component/weights/best.pt \
  --output-dir outputs/yolo/component
```

### GUI ROI Inference

The GUI Inference tab combines both models:

```text
component YOLO -> ROI boxes -> scratch YOLO tiles -> full-size mask/overlay
```

It can also send a predicted mask to the Annotation tab so a worker can refine
the prediction and save it as a new training label.

## Export and Runtime Optimization

### OpenVINO

OpenVINO is the preferred optimized backend for CPU deployment:

```bash
.venv/bin/python -m src.openvino.yolo_export \
  --model models/yolo/scratch_yolo26n_seg/weights/best.pt \
  --imgsz 512
```

Benchmark PyTorch vs OpenVINO:

```bash
.venv/bin/python -m src.openvino.evaluation \
  --image-dir data/scratch_yolo_seg/test/images \
  --max-images 10 \
  --repeat 3
```

### ONNX

ONNX export/runtime experiments live in `src/onnx/`. They are optional and are
not the default deployment path. OpenVINO is recommended for the current local
CPU-focused GUI pipeline.

## Configuration

| File | Purpose |
|---|---|
| `configs/path.py` | Project root, `data/`, `models/`, `outputs/`, metrics, labeling paths |
| `configs/data.py` | Split names, image extensions, binary mask values, labeling paths |
| `configs/yolo.py` | Component and scratch YOLO datasets, checkpoints, thresholds, training defaults |

## Important Paths

| Path | Description |
|---|---|
| `data/raw/` | Original camera/raw images |
| `outputs/labeling/images/` | Labeled source images |
| `outputs/labeling/masks/` | Labeled binary masks |
| `data/scratch_yolo_seg/` | YOLO scratch segmentation dataset |
| `data/scratch_sam2_format/` | SAM2 fine-tuning dataset |
| `models/yolo/` | YOLO checkpoints |
| `models/sam2/` | SAM2 checkpoints |
| `outputs/yolo/` | YOLO inference outputs |
| `outputs/metrics/` | Evaluation outputs |

## Git and Artifact Policy

The repository is intended to track source code, configs, notebooks, Docker
files, and documentation. Large runtime artifacts are ignored:

```text
data/
models/
outputs/
.venv/
*.pt
*.pth
*.onnx
*.xml
*.bin
*.engine
```

To reproduce a full local run from a fresh clone, restore the required
`data/` and `models/` folders separately.

## Tech Stack

- Python 3.12
- PySide6
- Ultralytics YOLO26
- SAM2
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Albumentations
- Matplotlib
- OpenVINO
- ONNX Runtime, optional experiment backend
- Docker / Docker Compose
