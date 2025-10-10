import sys
from pathlib import Path
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent / 'src'))
from datasets import FluidDataset, create_dataloaders 
from model import SegmentationModel, BCEDiceLoss, calculate_iou
from tqdm import tqdm


def training_step(
        model: nn.Module,
        train_dataloader: DataLoader,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str
):

        model.train()

        train_loss = 0
        train_iou = 0

        for batch, (x, y) in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
                x, y = x.to(device), y.to(device)
                
                y_pred = model(x) # logits

                loss = loss_function(y_pred, y)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

                iou = calculate_iou(torch.sigmoid(y_pred), y, threshold=0.5)
                train_iou += iou.item()
        
        train_loss /= len(train_dataloader)
        train_iou = train_iou / len(train_dataloader) * 100

        return train_loss, train_iou

def test_step(
        model: nn.Module,
        test_dataloader: DataLoader,
        loss_function: nn.Module,
        device: str
):
        model.eval()

        test_loss = 0
        test_iou = 0

        with torch.inference_mode():
                for batch, (x, y) in enumerate(tqdm(test_dataloader, desc="Testing", leave=False)):
                        x, y = x.to(device), y.to(device)

                        y_pred = model(x) # logits

                        loss = loss_function(y_pred, y)
                        test_loss += loss.item()

                        iou = calculate_iou(torch.sigmoid(y_pred), y, threshold=0.5)
                        test_iou += iou.item()
        
        test_loss /= len(test_dataloader)
        test_iou = test_iou / len(test_dataloader) * 100

        return test_loss, test_iou


def train_model(
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        scheduler=None,
        epochs: int=50,
        save_path: str=None
):
        results = {
                "train_loss": [],
                "train_iou": [],
                "test_loss": [],
                "test_iou": []
        }
        
        # Track best model
        best_iou = 0.0
        best_model_path = None
        if save_path:
                best_model_path = save_path.replace('.pth', '_best.pth')

        for epoch in tqdm(range(epochs)):
                train_loss, train_iou = training_step(
                        model=model,
                        train_dataloader=train_dataloader,
                        loss_function=loss_function,
                        optimizer=optimizer,
                        device=device
                )

                test_loss, test_iou = test_step(
                        model=model,
                        test_dataloader=test_dataloader,
                        loss_function=loss_function,
                        device=device
                )

                print(
                        f"Epoch: {epoch+1} | "
                        f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.2f}% | "
                        f"Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.2f}%"
                )

                results["train_loss"].append(train_loss)
                results["train_iou"].append(train_iou)
                results["test_loss"].append(test_loss)
                results["test_iou"].append(test_iou)
                
                # Save best model based on validation IoU
                if test_iou > best_iou:
                        best_iou = test_iou
                        if best_model_path:
                                torch.save(model.state_dict(), best_model_path)
                                print(f"New best model saved! IoU: {best_iou:.2f}%")
                
                # Regular checkpoint saving
                if save_path:
                        torch.save(model.state_dict(), save_path)
                
                # Update learning rate
                if scheduler:
                        scheduler.step()
                        current_lr = scheduler.get_last_lr()[0]
                        print(f"Learning rate: {current_lr:.2e}")
                        
        print(f"Training completed! Best validation IoU: {best_iou:.2f}%")
        return results