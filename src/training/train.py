"""
src/training/train.py
Main training script for custom CNN or pretrained MobileNetV2.

Usage examples:
    python -m src.training.train --model_type custom --epochs 20
    python -m src.training.train --model_type pretrained --dataset sampled
"""

import argparse
import os
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, OUTPUTS_PATH,
    USE_SUBSAMPLED, EARLY_STOPPING_PATIENCE
)
from src.data.dataset import get_dataloaders
from src.models.custom_cnn import get_model as get_custom_model
from src.models.pretrained_model import get_pretrained_model
from src.training.validate import validate
from src.training.utils import set_seed, save_checkpoint, EarlyStopping

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_save_path, early_stopping_patience):
    """
    Execute the training loop with validation and checkpointing.
    Returns training history list and the best validation accuracy.
    """
    history = []
    best_val_acc = 0.0
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.001)
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # (batch, 1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs).cpu().detach().numpy()
            preds_binary = (preds >= 0.5).astype(int)
            correct += (preds_binary.flatten() == labels.cpu().numpy().flatten()).sum()
            total += labels.size(0)
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation phase
        val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1']
        })
        
        # Save best model based on validation accuracy
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_checkpoint(model, optimizer, epoch, val_metrics['loss'], model_save_path)
            print(f"  New best model saved with val_acc: {best_val_acc:.4f}")
        
        # Early stopping
        early_stopping(val_metrics['loss'])
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    return history, best_val_acc

def main():
    parser = argparse.ArgumentParser(description="Train drowsiness detection model.")
    parser.add_argument("--model_type", type=str, choices=["custom", "pretrained"], default="custom",
                        help="Type of model to train: 'custom' (CNN_Small) or 'pretrained' (MobileNetV2)")
    parser.add_argument("--dataset", type=str, choices=["sampled", "full"], default=None,
                        help="Override dataset type (sampled or full). If not set, uses config.USE_SUBSAMPLED.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Learning rate (for custom CNN). Pretrained uses lower LR.")
    args = parser.parse_args()
    
    # Determine dataset type
    if args.dataset is None:
        use_subsampled = USE_SUBSAMPLED
    else:
        use_subsampled = (args.dataset == "sampled")
    
    # Override config flag temporarily (for dataset loading)
    # We import config again to avoid modifying global, but we can just pass a flag.
    # Simpler: override by setting a variable; dataset.py reads USE_SUBSAMPLED from config.
    # We'll directly modify config.USE_SUBSAMPLED for this run (not persistent).
    import src.config as cfg
    cfg.USE_SUBSAMPLED = use_subsampled
    
    dataset_str = "sampled" if use_subsampled else "full"
    print(f"Training on {dataset_str} dataset")
    
    # Set seed for reproducibility
    set_seed()
    
    # Create data loaders
    train_loader, val_loader, _ = get_dataloaders(batch_size=args.batch_size, num_workers=cfg.NUM_WORKERS)
    
    # Create model
    if args.model_type == "custom":
        model = get_custom_model("small")  # Use small CNN for CPU training
        print("Using custom CNN (small)")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        model = get_pretrained_model(freeze_backbone=True, num_classes=1)
        print("Using pretrained MobileNetV2 (backbone frozen)")
        # Lower learning rate for pretrained fine-tuning
        lr_pretrained = 0.0001 if args.lr == LEARNING_RATE else args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr_pretrained)
    
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    
    # Prepare save path
    model_save_dir = Path(MODEL_SAVE_PATH)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model_filename = f"{args.model_type}_{dataset_str}_best.pth"
    model_save_path = model_save_dir / model_filename
    
    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    history, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        args.epochs, DEVICE, model_save_path, EARLY_STOPPING_PATIENCE
    )
    
    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to {model_save_path}")
    
    # Save training history as CSV
    output_dir = Path(OUTPUTS_PATH) / f"{args.model_type}_{dataset_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    csv_path = output_dir / "training_history.csv"
    history_df.to_csv(csv_path, index=False)
    print(f"Training history saved to {csv_path}")

if __name__ == "__main__":
    main()