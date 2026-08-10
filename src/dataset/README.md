# Dataset Preparation

Scripts in this folder prepare training datasets directly from labeled scratch
data.

Input source:

```text
outputs/labeling/
  images/
  masks/
```

The scripts split full images first, then crop patches in memory. This avoids
creating extra intermediate `split`, `patches`, or `instances` folders.

## YOLO Scratch Segmentation

Prepare a YOLO instance-segmentation dataset:

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

Output:

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

Train YOLO:

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

## SAM2 Scratch Fine-tuning

Prepare a SAM2 one-frame dataset:

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

Use these paths in the SAM2 training config:

```yaml
dataset.img_folder: data/scratch_sam2_format/JPEGImages
dataset.gt_folder: data/scratch_sam2_format/Annotations
dataset.file_list_txt: data/scratch_sam2_format/ImageSets/train.txt
```

## Notes

- Use `--dry-run` to check counts without writing files.
- Use `--overwrite` only when replacing an existing dataset intentionally.
- YOLO output writes one image per selected patch.
- SAM2 output writes one image per scratch instance, so it can be much larger.
- Images are written as `.png` to avoid JPEG compression artifacts on thin scratches.
