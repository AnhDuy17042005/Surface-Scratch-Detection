# U-Net Scratch Segmentation

Training, evaluation, and inference scripts for the U-Net scratch mask model.

Train U-Net:

```bash
.venv/bin/python -m src.unet.train \
  --data data/scratch_patches \
  --save_dir models/unet/unet_v6
```

Run inference:

```bash
.venv/bin/python -m src.unet.inference \
  --image data/raw/Image_20260714163537759.bmp \
  --mode sliding \
  --device auto
```

Run evaluation:

```bash
.venv/bin/python -m src.unet.evaluation \
  --data data/scratch_patches \
  --split test \
  --mode sliding
```

Default model paths, patch size, threshold, and training settings are configured
in `configs/unet.py`.
