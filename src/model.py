import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from datasets import get_training_transforms, get_validation_transforms

class SegmentationModel(nn.Module):
    """Unet CNN using segmentation_models_pytorch library."""
    def __init__(
            self, encoder_name: str,
            encoder_weights: str,
            in_channels: int,
            classes: int=1,
            image_size: tuple[int, int]=(512,512)):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

        self.training_transform = get_training_transforms(image_size)
        self.validation_transform = get_validation_transforms(image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
class DiceLoss(nn.Module):
    """Dice loss function."""
    def __init__(self, smooth: float=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:   
        overlap = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()

        dice_coefficient = (2. * overlap + self.smooth) / (union + self.smooth)

        return 1 - dice_coefficient
    
class BCEDiceLoss(nn.Module):
    """Combination of BCE and Dice loss."""
    def __init__(self, weight_bce: float=0.5, weight_dice: float=0.5):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss() 
        self.dice_loss = DiceLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y_pred_sigmoid = torch.sigmoid(inputs)
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(y_pred_sigmoid, targets)
        return self.weight_bce * bce + self.weight_dice * dice
    

def calculate_iou(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float):
    """Calculate intersection over union (IoU) metric."""
    y_pred = torch.flatten(y_pred)
    y_true = torch.flatten(y_true)
    y_pred = (y_pred > threshold).float()
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection

    return (intersection + 1e-6) / (union + 1e-6)


def main():
    model = SegmentationModel(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    )

    x = torch.randn((1, 3, 1080, 1920)) 

    y = model(x)
    print(y.shape)  # binary mask output shape
    print(y)

if __name__ == "__main__":
    main()
