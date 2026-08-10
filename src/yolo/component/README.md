# YOLO Component Detection

Scripts for the component bounding-box detector. This model finds the component
ROI before YOLO scratch segmentation runs inside that region.

Use these commands from the project root.

Run inference:

```bash
.venv/bin/python -m src.yolo.component.inference \
  --source data/raw \
  --output-dir outputs/yolo/component
```

Train the component detector:

```bash
.venv/bin/python -m src.yolo.component.train \
  --data data/component/data.yaml \
  --epochs 100
```

Default paths and thresholds are configured in `configs/yolo.py`.
