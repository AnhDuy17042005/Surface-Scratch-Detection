# OpenVINO

Export and benchmark optimized runtime models.

Use these scripts from the project root.

Export YOLO scratch segmentation to OpenVINO:

```bash
.venv/bin/python -m src.openvino.yolo_export \
  --model models/yolo/scratch_yolo26n_seg/weights/best.pt
```

Benchmark YOLO PyTorch vs OpenVINO:

```bash
.venv/bin/python -m src.openvino.evaluation \
  --max-images 10 \
  --repeat 3
```

OpenVINO must be installed in `.venv` before these commands can run.
