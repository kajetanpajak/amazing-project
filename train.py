import sys
from pathlib import Path
import torch
import torch.nn as nn


sys.path.append(str(Path(__file__).parent / 'src'))
import engine
from model import SegmentationModel, BCEDiceLoss
from datasets import create_dataloaders 


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = SegmentationModel(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=2e-4,  # Slightly higher LR for larger batch
        weight_decay=1e-4,  # L2 regularization
        betas=(0.9, 0.999),  # Momentum parameters (default but explicit)
        eps=1e-8
    )
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=100,  # Match your epochs
        eta_min=1e-6
    )
    
    loss_function = BCEDiceLoss()

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        dataset_dir='datasets',
        batch_size=8,  # Increased as requested
        num_workers=2,
        image_size=(512, 512)
    )

    print(f"Train batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")

    results = engine.train_model(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=100,
        save_path='models/resnet34_model.pth'
    )

    print(results)

if __name__ == "__main__":
    main()

