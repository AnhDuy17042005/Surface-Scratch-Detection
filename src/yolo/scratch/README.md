# YOLO Scratch Segmentation

Scripts for YOLO instance segmentation of scratch masks.

Train the scratch segmentation model:

```bash
.venv/bin/python -m src.yolo.scratch.train \
  --data data/scratch_yolo_seg/data.yaml \
  --device 0
```

Run scratch inference on full images with sliding-window patches:

```bash
.venv/bin/python -m src.yolo.scratch.inference \
  --source data/raw \
  --mode sliding \
  --output-dir outputs/yolo/scratch
```

Default dataset, model, threshold, and patch settings are configured in
`configs/yolo.py`.
