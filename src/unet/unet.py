"""
    U-Net for surface scratch binary segmentation.

    Architecture:
    Encoder    : 4 conv blocks, each block = Conv→GN→ReLU→Conv→GN→ReLU, followed by MaxPool
    Bottleneck : deepest conv block (no pooling after)
    Decoder    : 4 up-blocks, each = Upsample→Cat(skip)→Conv→GN→ReLU→Conv→GN→ReLU
    Head       : 1x1 Conv → num_classes (1 for binary)

    Key insight: skip connections concatenate encoder feature maps to the decoder at the
    same resolution. This is what lets U-Net recover fine spatial detail (like thin scratch
    boundaries) that would be lost during downsampling.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Support direct script run from the project root."""
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.unet import (
    UNET_BASE_CHANNELS,
    UNET_IMAGE_SIZE,
    UNET_INPUT_CHANNELS,
    UNET_NORM_TYPE,
    UNET_NUM_CLASSES,
)

def make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """
        Build GroupNorm with a group count that divides the channel count.
    """

    """Use the largest valid group count up to max_groups"""
    for num_groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(
                num_groups=num_groups,
                num_channels=num_channels,
            )

    """This path is unreachable because 1 always divides num_channels"""
    raise ValueError(f"Cannot create GroupNorm for {num_channels} channels")


def make_norm_layer(
    num_channels: int,
    norm_type: str,
) -> nn.Module:
    """
        Build normalization layer by type.
    """

    """Use BatchNorm for old checkpoints trained before GroupNorm migration"""
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)

    """Use GroupNorm for small-batch scratch segmentation training"""
    if norm_type == "group":
        return make_group_norm(num_channels)

    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvBlock(nn.Module):
    """
        Two conv layers, each followed by GroupNorm + ReLU.
        This is the fundamental building block repeated throughout U-Net.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = UNET_NORM_TYPE,
    ):
        super().__init__()

        """Sequential layer"""
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                padding=1, 
                bias=False
            ),
            make_norm_layer(out_channels, norm_type),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                padding=1, 
                bias=False
            ),
            make_norm_layer(out_channels, norm_type),
            nn.ReLU(inplace=True),    
        )

    def forward(self, x):
        return self.block(x)
        
   
class UpBlock(nn.Module):
    """
        Upsample → concatenate skip → two conv layers.

        We use bilinear upsample + conv instead of ConvTranspose2d because:
        - ConvTranspose2d can produce 'checkerboard artifacts' on thin structures
        - Bilinear upsampling is smoother and more stable for scratch-like patterns
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = UNET_NORM_TYPE,
    ):
        super().__init__()

        """Scaling width and height"""
        self.up   = nn.Upsample(
            scale_factor=2, 
            mode="bilinear", 
            align_corners=True
        )
        self.conv = ConvBlock(in_channels, out_channels, norm_type=norm_type)

    def forward(self, x, skip):
        x = self.up(x)

        """Handle input size"""
        if x.shape[2:] != skip.shape[2:]:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                           0, skip.shape[2] - x.shape[2]]) 
            
        """Concatunate on channel dim"""
        x = torch.cat([skip, x], dim=1) 
        return self.conv(x)


class UNet(nn.Module):
    """
        U-Net for binary segmentation (scratch / background).

        Args:
            in_channels  : 1 for grayscale, 3 for RGB
            num_classes  : 1 for binary (use BCEWithLogitsLoss)
                        N for multi-class (use CrossEntropyLoss)
            base_channels: controls model width. Default 64.
                        Reduce to 32 for lighter model on Colab T4.
    """
    def __init__(
        self,
        in_channels: int = UNET_INPUT_CHANNELS,
        num_classes: int = UNET_NUM_CLASSES,
        base_channels: int = UNET_BASE_CHANNELS,
        norm_type: str = UNET_NORM_TYPE,
    ):
        super().__init__()
        c = base_channels
        self.norm_type = norm_type

        """Encoder Block"""
        self.enc1 = ConvBlock(in_channels, c, norm_type=norm_type)       # 256×256, 64ch
        self.enc2 = ConvBlock(c, 2 * c, norm_type=norm_type)             # 128×128, 128ch
        self.enc3 = ConvBlock(2 * c, 4 * c, norm_type=norm_type)         # 64×64,   256ch
        self.enc4 = ConvBlock(4 * c, 8 * c, norm_type=norm_type)         # 32×32,   512ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        """Bottleneck: enc5"""
        self.bottleneck = ConvBlock(8 * c, 16 * c, norm_type=norm_type)  # 16×16,   1024ch

        """Decoder Block"""
        self.dec4 = UpBlock(16 * c + 8 * c, 8 * c, norm_type=norm_type)
        self.dec3 = UpBlock(8  * c + 4 * c, 4 * c, norm_type=norm_type)
        self.dec2 = UpBlock(4  * c + 2 * c, 2 * c, norm_type=norm_type)
        self.dec1 = UpBlock(2  * c + c, c, norm_type=norm_type)

        """Head: conv 1x1"""
        self.head = nn.Conv2d(
            in_channels=c, 
            out_channels=num_classes, 
            kernel_size=1
        )

    def forward(self, x):
        """Encoder"""
        s1 = self.enc1(x)               # skip 1
        s2 = self.enc2(self.pool(s1))   # skip 2
        s3 = self.enc3(self.pool(s2))   # skip 3
        s4 = self.enc4(self.pool(s3))   # skip 4

        """Bottleneck"""
        b  = self.bottleneck(self.pool(s4))

        """Decoder"""
        x  = self.dec4(b, s4)
        x  = self.dec3(x, s3)
        x  = self.dec2(x, s2)
        x  = self.dec1(x, s1)

        """Head"""
        x  = self.head(x)
        return x
    

def build(
    in_channels=UNET_INPUT_CHANNELS,
    num_classes=UNET_NUM_CLASSES,
    base_channels=UNET_BASE_CHANNELS,
    norm_type=UNET_NORM_TYPE,
):
    model = UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        norm_type=norm_type,
    )   
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net built — norm = {norm_type} trainable params: {total_params:,}")
    
    return model

if __name__ == "__main__":
    model = build()

    """Test model with random tensor"""
    x = torch.randn(2, UNET_INPUT_CHANNELS, UNET_IMAGE_SIZE, UNET_IMAGE_SIZE)
    out = model(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
