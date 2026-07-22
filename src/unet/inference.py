"""
    Inference script for U-Net scratch segmentation.

    Purpose:
        1. Load trained U-Net checkpoint.
        2. Predict binary scratch mask from one input image or tile.
        3. Apply post-processing to clean the mask.
        4. Save overlay result for visual inspection.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support both direct script run and package import"""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.unet.sliding_windows import (
        predict_sliding_window_mask,
        sliding_window_grid_shape,
    )
    from src.unet.unet import build
else:
    from .sliding_windows import predict_sliding_window_mask, sliding_window_grid_shape
    from .unet import build

from configs.unet import (
    IMAGENET_MEAN as IMAGENET_MEAN_VALUES,
    IMAGENET_STD as IMAGENET_STD_VALUES,
    UNET_DEFAULT_IMAGE,
    UNET_DEVICE,
    UNET_IMAGE_SIZE,
    UNET_INFERENCE_PATCH_BATCH_SIZE,
    UNET_INFERENCE_TILE_OVERLAP,
    UNET_INFERENCE_TILE_SIZE,
    UNET_MODEL,
    UNET_OUTPUT_DIR,
    UNET_THRESHOLD,
)

IMAGENET_MEAN = np.asarray(IMAGENET_MEAN_VALUES, dtype=np.float32)
IMAGENET_STD  = np.asarray(IMAGENET_STD_VALUES, dtype=np.float32)


class OnnxUnetModel:
    """
        Small adapter so ONNX Runtime models can be called like PyTorch models.
    """

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        input_array = tensor.detach().cpu().numpy().astype(np.float32)
        logits = self.session.run(
            [self.output_name],
            {self.input_name: input_array},
        )[0]

        return torch.from_numpy(logits).to(tensor.device)


class OpenVINOUnetModel:
    """
        Small adapter so OpenVINO IR models can be called like PyTorch models.
    """

    def __init__(self, model_path: Path) -> None:
        import openvino as ov

        core = ov.Core()
        self.compiled_model = core.compile_model(model_path, "CPU")
        self.input = self.compiled_model.input(0)
        self.output = self.compiled_model.output(0)
        self.input_shape = list(self.input.partial_shape)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        input_array = tensor.detach().cpu().numpy().astype(np.float32)
        logits = self.compiled_model({self.input: input_array})[self.output]

        return torch.from_numpy(logits).to(tensor.device)


def parse_args() -> argparse.Namespace:
    """
        Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Predict scratch mask with U-Net."
    )

    """Input/output paths"""
    parser.add_argument("--image", type=Path, default=UNET_DEFAULT_IMAGE)
    parser.add_argument("--model", type=Path, default=UNET_MODEL)
    parser.add_argument("--output-dir", type=Path, default=UNET_OUTPUT_DIR)

    """Inference settings"""
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=UNET_THRESHOLD)
    parser.add_argument("--device", type=str, default=UNET_DEVICE)
    parser.add_argument(
        "--mode",
        choices=("sliding", "full"),
        default="sliding",
        help="Use sliding for patch-trained U-Net or full for resized full-image inference.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=UNET_INFERENCE_TILE_SIZE,
        help="Sliding-window patch size. Use 512 for the original training-like setting.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=UNET_INFERENCE_TILE_OVERLAP,
        help="Sliding-window overlap ratio. Higher overlap is slower but can reduce tile seams.",
    )
    parser.add_argument(
        "--patch-batch-size",
        type=int,
        default=UNET_INFERENCE_PATCH_BATCH_SIZE,
    )

    return parser.parse_args()


def load_image(image_path: Path) -> np.ndarray:
    """
        Load image with OpenCV.

        Returns:
            BGR image as numpy array.
    """

    """Read image from disk"""
    image = cv2.imread(str(image_path))

    """Raise error if image cannot be loaded"""
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    return image


def infer_norm_type_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    """
        Infer normalization type from checkpoint state_dict.
    """

    """BatchNorm checkpoints contain running statistics buffers"""
    has_batch_norm_buffers = any(
        key.endswith("running_mean")
        or key.endswith("running_var")
        or key.endswith("num_batches_tracked")
        for key in state_dict
    )

    if has_batch_norm_buffers:
        return "batch"

    return "group"


def get_device(device: str) -> torch.device:
    """
        Select inference device.

        Args:
            device:
                "auto", "cuda", or "cpu".
    """

    """Automatically use GPU if available"""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """Use user-defined device"""
    return torch.device(device)


def load_model(
    model_path: Path,
    device: torch.device,
    img_size: int | None
):
    """
        Load U-Net checkpoint saved by train.py.

        Supports:
            - full checkpoint dictionary
            - raw model state_dict

        Returns:
            model:
                Loaded U-Net model.

            input_size:
                Image size used for inference.
    """

    """Check checkpoint path"""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    """Load OpenVINO IR model with OpenVINO Runtime"""
    if model_path.suffix.lower() == ".xml":
        model = OpenVINOUnetModel(model_path)
        shape_size = model.input_shape[-1]
        input_size = img_size or (
            int(shape_size.get_length())
            if shape_size.is_static
            else UNET_IMAGE_SIZE
        )

        return model, input_size

    """Load ONNX model with ONNX Runtime"""
    if model_path.suffix.lower() == ".onnx":
        model = OnnxUnetModel(model_path)
        shape_size = model.input_shape[-1]
        input_size = img_size or (
            int(shape_size)
            if isinstance(shape_size, int)
            else UNET_IMAGE_SIZE
        )

        return model, input_size

    """Load checkpoint to selected device"""
    checkpoint = torch.load(model_path, map_location=device)

    """Extract model weights and training arguments"""
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        train_args = checkpoint.get("args", {}) or {}
    else:
        state_dict = checkpoint
        train_args = {}

    """Infer base channel from first convolution layer"""
    first_conv = state_dict.get("enc1.block.0.weight")
    base_channel = int(first_conv.shape[0]) if first_conv is not None else 64

    """Infer normalization type for old BatchNorm and new GroupNorm checkpoints"""
    norm_type = (
        train_args.get("norm")
        or train_args.get("norm_type")
        or infer_norm_type_from_state_dict(state_dict)
    )

    """Use input image size from argument or checkpoint"""
    input_size = img_size or int(train_args.get("img_size", 256))

    """Build U-Net model with the same width as checkpoint"""
    model = build(
        in_channels=3,
        num_classes=1,
        base_channels=base_channel,
        norm_type=norm_type,
    )

    """Load trained weights"""
    model.load_state_dict(state_dict)

    """Move model to device and set evaluation mode"""
    model.to(device)
    model.eval()

    return model, input_size


def preprocess(
    image: np.ndarray,
    img_size: int,
    device: torch.device
) -> torch.Tensor:
    """
        Convert OpenCV BGR image to normalized NCHW tensor.

        Processing:
            BGR → RGB
            resize
            normalize
            HWC → CHW
            add batch dimension
    """

    """Convert BGR to RGB"""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    """Resize image to model input size"""
    resized = cv2.resize(
        rgb,
        (img_size, img_size),
        interpolation=cv2.INTER_LINEAR
    )

    """Normalize image using ImageNet statistics"""
    normalized = (
        resized.astype(np.float32) / 255.0 - IMAGENET_MEAN
    ) / IMAGENET_STD

    """Convert HWC numpy image to NCHW torch tensor"""
    tensor = torch.from_numpy(
        normalized.transpose(2, 0, 1)
    ).unsqueeze(0)

    return tensor.to(device)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image: np.ndarray,
    img_size: int,
    threshold: float,
    device: torch.device,
) -> np.ndarray:
    """
        Run U-Net prediction on one image.

        Returns:
            Binary mask with values 0 and 255.
    """

    """Store original image size"""
    h, w = image.shape[:2]

    """Preprocess image for U-Net"""
    tensor = preprocess(image, img_size, device)

    """Forward pass"""
    logits = model(tensor)

    """Convert logits to probability map"""
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    """Resize probability map back to original image size"""
    prob = cv2.resize(
        prob,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    """Apply threshold to create binary mask"""
    return (prob >= threshold).astype(np.uint8) * 255


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
        Overlay predicted scratch mask on the original image.
    """

    """Create red color mask"""
    color = np.zeros_like(image)
    color[mask > 0] = (0, 0, 255)

    """Find active mask pixels"""
    overlay = image.copy()
    active = mask > 0

    """Blend original image and green mask only on scratch pixels"""
    overlay[active] = cv2.addWeighted(
        image,
        0.7,
        color,
        0.3,
        0
    )[active]

    return overlay


def save_image(path: Path, image: np.ndarray) -> None:
    """
        Save image and raise error if saving fails.
    """

    """Save image to disk"""
    success = cv2.imwrite(str(path), image)

    """Check saving result"""
    if not success:
        raise RuntimeError(f"Failed to save image: {path}")


def save_outputs(
    output_dir: Path,
    image_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
) -> None:
    """
        Save predicted overlay result.
    """

    """Create output directory"""
    output_dir.mkdir(parents=True, exist_ok=True)

    """Use input image name as output prefix"""
    stem = image_path.stem

    """Save overlay prediction"""
    save_image(
        output_dir / f"{stem}_prediction.jpg",
        make_overlay(image, mask)
    )


def main() -> None:
    """
        Main inference function.

        Steps:
            1. Parse arguments
            2. Load image
            3. Load U-Net model
            4. Predict scratch mask
            5. Post-process mask
            6. Save overlay result
    """

    """Parse command line arguments"""
    args = parse_args()

    """Create output directory"""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    """Print input information"""
    print(f"Image: {args.image}")
    print(f"Model: {args.model}")

    """Prepare device"""
    device = get_device(args.device)
    print(f"Device: {device}")

    """Load input image"""
    image = load_image(args.image)
    image_h, image_w = image.shape[:2]
    print(f"Original image size: {image_w}x{image_h}")

    """Load trained U-Net model"""
    model_start = time.perf_counter()
    model, img_size = load_model(
        args.model,
        device,
        args.img_size
    )
    model_seconds = time.perf_counter() - model_start

    """Predict raw binary mask"""
    predict_start = time.perf_counter()
    if args.mode == "sliding":
        grid_x, grid_y, stride = sliding_window_grid_shape(
            image_shape=image.shape,
            patch_size=args.patch_size,
            overlap=args.overlap,
            batch_size=args.patch_batch_size,
        )
        total_patches = grid_x * grid_y
        processed_megapixels = (
            total_patches * args.patch_size * args.patch_size / 1_000_000
        )
        print(
            "Sliding grid: "
            f"{grid_x}x{grid_y} = {total_patches} patches | "
            f"stride: {stride} | "
            f"processed: {processed_megapixels:.1f}MP"
        )
        mask = predict_sliding_window_mask(
            model=model,
            image=image,
            patch_size=args.patch_size,
            overlap=args.overlap,
            threshold=args.threshold,
            batch_size=args.patch_batch_size,
            device=device,
        )
    else:
        mask = predict(
            model=model,
            image=image,
            img_size=img_size,
            threshold=args.threshold,
            device=device,
        )
    predict_seconds = time.perf_counter() - predict_start

    """Save final output"""
    save_start = time.perf_counter()
    save_outputs(
        output_dir=args.output_dir,
        image_path=args.image,
        image=image,
        mask=mask,
    )
    save_seconds = time.perf_counter() - save_start
    total_seconds = time.perf_counter() - start_time

    """Print output information"""
    print(f"Mode: {args.mode}")
    print(f"Image size: {img_size}")
    if args.mode == "sliding":
        print(
            f"Patch size: {args.patch_size} | "
            f"Overlap: {args.overlap} | "
            f"Patch batch size: {args.patch_batch_size}"
        )
    print(f"Saved outputs to: {args.output_dir}")
    print(
        "Timing: "
        f"model load {model_seconds:.2f}s | "
        f"predict {predict_seconds:.2f}s | "
        f"save {save_seconds:.2f}s | "
        f"total {total_seconds:.2f}s"
    )


if __name__ == "__main__":
    main()
