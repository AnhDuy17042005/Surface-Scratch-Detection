# Evaluation

CLI evaluation scripts for comparing model quality and runtime.

Evaluate YOLO scratch segmentation:

```bash
.venv/bin/python -m src.evaluation.yolo_scratch \
  --data data/scratch_patches \
  --split test
```

Evaluate YOLO component detection:

```bash
.venv/bin/python -m src.evaluation.yolo_component \
  --data data/component/data.yaml
```

Evaluate SAM2 prompt quality:

```bash
.venv/bin/python -m src.evaluation.sam2 \
  --data data/scratch_sam2_format \
  --split test
```

Metrics are written under `outputs/metrics/`.
YOLO and SAM2 evaluators also write PNG plots under
`outputs/metrics/<model>/plots/`.
