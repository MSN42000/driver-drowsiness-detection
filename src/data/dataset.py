"""
src/data/dataset.py
PyTorch Dataset for loading eye images from the processed folder.

Reads images from data/processed/{sampled|full}/{train|val|test}/Drowsy/ and Non Drowsy/
and applies appropriate transforms (augmentation for train, basic for val/test).
"""

import torch
from torchvision import datasets
import src.config as cfg
from src.data.preprocessing import get_train_transforms, get_val_test_transforms
from pathlib import Path

def get_processed_root():
    """
    Returns the correct processed root folder based on USE_SUBSAMPLED flag.
    If USE_SUBSAMPLED = True -> data/processed/sampled/
    If USE_SUBSAMPLED = False -> data/processed/full/
    """
    processed_path = Path(cfg.DATA_PROCESSED_ROOT)
    if cfg.USE_SUBSAMPLED:
        return processed_path / "sampled"
    else:
        return processed_path / "full"

def get_dataloaders(batch_size, num_workers=2):
    """
    Creates train, validation, and test DataLoaders.

    Args:
        batch_size (int): Batch size for all loaders.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        train_loader, val_loader, test_loader (tuple of DataLoader objects)
    """
    processed_root = get_processed_root()
    
    # Define paths to each split
    train_root = processed_root / "train"
    val_root = processed_root / "val"
    test_root = processed_root / "test"
    
    # Create datasets with appropriate transforms
    train_dataset = datasets.ImageFolder(
        root=train_root,
        transform=get_train_transforms()
    )
    
    val_dataset = datasets.ImageFolder(
        root=val_root,
        transform=get_val_test_transforms()
    )
    
    test_dataset = datasets.ImageFolder(
        root=test_root,
        transform=get_val_test_transforms()
    )
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # CPU only, no need for pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    # Print dataset sizes for verification
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    print(f"Test dataset: {len(test_dataset)} images")
    print(f"Classes: {train_dataset.classes}")  # Should be ['Drowsy', 'Non Drowsy'] or similar order
    
    return train_loader, val_loader, test_loader