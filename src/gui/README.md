# GUI

PySide6 application tabs for camera capture, labeling, data preparation,
training, SAM2-assisted labeling, and YOLO inference.

Run the full GUI:

```bash
.venv/bin/python -m src.gui.main
```

The GUI calls the same backend scripts used by the CLI, including:

```text
src.processing.split_data
src.processing.make_patchs
src.dataset.prepare_yolo_dataset
src.yolo.scratch.inference
src.yolo.component.inference
src.yolo.component.roi
```

Evaluation scripts live in `src/evaluation/` and are intended for CLI runs.
Model paths and runtime defaults are configured in `configs/`.
