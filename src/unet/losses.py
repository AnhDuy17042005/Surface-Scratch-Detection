"""
    Loss functions and metrics for binary segmentation.

    BCE treats every pixel equally. For scratch segmentation, scratch pixels are a
    minority vs background. 
    
    Dice Loss directly optimizes the overlap
    ratio between prediction and GT - better for imbalanced pixel distributions.

    Best practice: combine BCE + Dice for stable training.
"""

import torch
import torch.nn as nn

from configs.unet import UNET_DICE_ALPHA, UNET_METRIC_THRESHOLD

class DiceLoss(nn.Module):
    """
        Dice Loss = 1 - Dice coefficient.

        Dice = 2 * |Predict ∩ GroundTruth| / (|Predict| + |GroundTruth|)

        Works directly on predicted probabilities and binary GT mask.
        smooth=1 prevents division by zero and acts as Laplace smoothing.
    """
    def __init__(self, smooth: int = 1):
        super().__init__()

        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)

        """Flatten spatial dims: (B, 1, H, W) - (B, H*W)"""
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / \
               (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)

        return 1.0 - dice.mean()
    
class BCEDiceLoss(nn.Module):
    """
        Combined loss: a * BCE + (1-a) * Dice

        BCE: good gradient at the beginning
        Dice: good at handling class imbalance
        a = 0.5 is a solid default for scratch segmentation.
    """
    def __init__(
        self,
        alpha: float = UNET_DICE_ALPHA,
        smooth: int = 1,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

        self.alpha = alpha
        self.bce   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice  = DiceLoss(smooth=smooth)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        return self.alpha * self.bce(logits, targets) + (1 - self.alpha) * self.dice(logits, targets)


"""Metrics"""

@torch.no_grad()
def compute_iou(logits: torch.Tensor, targets: torch.Tensor, 
                threshold: float = UNET_METRIC_THRESHOLD) -> float:
    """
        The standard metric for segmentation evaluation.
    """
    
    preds   = (torch.sigmoid(logits) > threshold)
    targets = targets.float()

    """Flatten"""
    preds   = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    union        = (preds.sum() + targets.sum()) - intersection

    if union == 0:
        return 1.0
    iou = intersection / union
    return iou.item()

@torch.no_grad()
def compute_dice(logits: torch.Tensor, targets: torch.Tensor,
                 threshold: float = UNET_METRIC_THRESHOLD) -> float:
    """
        Dice coefficient (F1 score on pixels). Range [0, 1], higher is better.
    """
    
    preds   = (torch.sigmoid(logits) > threshold).float().view(-1)
    targets = targets.float().view(-1)

    intersection = (preds * targets).sum()
    denom = preds.sum() + targets.sum()

    if denom == 0:
        return 1.0

    return (2.0 * intersection / denom).item()
