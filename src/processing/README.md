# Data Processing

Scripts for preparing labeled scratch data before training.

These commands are CLI utilities. The GUI Data tab wraps the common data
preparation actions and can now prepare a YOLO dataset directly from
`outputs/labeling`.

All generated training images are written as `.png` to avoid JPEG compression
loss; masks are also kept as `.png`.

Prepare a YOLO segmentation dataset directly from labeled images:

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
  --overwrite
```

This writes only the final YOLO images, labels, and `data.yaml`:

```text
data/scratch_yolo_seg/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
  data.yaml
```

Train YOLO from the exported dataset:

```bash
.venv/bin/python -m src.yolo.scratch.train \
  --data data/scratch_yolo_seg/data.yaml \
  --model yolo26s-seg.pt \
  --imgsz 512 \
  --batch 8 \
  --epochs 150 \
  --patience 40 \
  --lr 0.001 \
  --name scratch_yolo26s_seg_v2
```

Legacy/experimental split helper:

```bash
.venv/bin/python -m src.processing.split_data \
  --src outputs/labeling \
  --dst data/scratch \
  --overwrite
```

Legacy/experimental patch helper:

```bash
.venv/bin/python -m src.processing.make_patchs \
  --src data/scratch \
  --dst data/scratch_patches \
  --patch-size 512 \
  --overlap 0.25 \
  --train-negative-ratio 1.0 \
  --valid-negative-ratio 1.2 \
  --test-negative-ratio 1.2
```

Generate 1024 patches for SAM2 experiments:

```bash
.venv/bin/python -m src.processing.make_patchs \
  --src data/scratch \
  --dst data/scratch_patches_sam2 \
  --patch-size 1024 \
  --overlap 0.25 \
  --train-negative-ratio 1.0 \
  --valid-negative-ratio 1.2 \
  --test-negative-ratio 1.2
```

Split semantic patch masks into one instance mask per sample:

```bash
.venv/bin/python -m src.processing.sam2_instances_mask \
  --src data/scratch_patches_sam2 \
  --dst data/scratch_patches_sam2_instances \
  --min-pixels 20 \
  --overwrite
```

Convert instance patches to SAM2 one-frame dataset format:

```bash
.venv/bin/python -m src.processing.sam2_convert_format \
  --src data/scratch_patches_sam2_instances \
  --dst data/scratch_sam2_format \
  --image-mode copy \
  --overwrite
```

Legacy helper for converting binary patch masks to YOLO segmentation labels:

```bash
.venv/bin/python -m src.processing.yolo_convert_labels \
  --src data/scratch_patches \
  --dst data/scratch_yolo_seg \
  --overwrite
```

The active GUI pipeline uses `src.dataset.prepare_yolo_dataset` instead of
running the split, patch, and YOLO-convert helpers separately.
