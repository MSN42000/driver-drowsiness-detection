"""
src/training/utils.py
Utility functions for training: checkpoint management, seeding, and early stopping.
"""

import torch
import random
import numpy as np
from src.config import RANDOM_SEED

def set_seed(seed=RANDOM_SEED):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    This ensures that runs with the same seed produce identical results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Does nothing on CPU but safe to call
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, val_loss, filepath):
    """
    Save model and optimizer state to a checkpoint file.

    Args:
        model (nn.Module): Trained model.
        optimizer (torch.optim.Optimizer): Optimizer used during training.
        epoch (int): Current epoch number.
        val_loss (float): Validation loss at this epoch.
        filepath (str or Path): Where to save the checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load a checkpoint and restore model (and optionally optimizer) state.

    Args:
        filepath (str or Path): Path to the checkpoint file.
        model (nn.Module): Model instance (architecture must match saved state).
        optimizer (torch.optim.Optimizer, optional): Optimizer to restore state.

    Returns:
        epoch (int), val_loss (float)
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    print(f"Checkpoint loaded from {filepath} (epoch {epoch}, val_loss {val_loss:.4f})")
    return epoch, val_loss


class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.

    Args:
        patience (int): Number of epochs with no improvement after which to stop.
        min_delta (float): Minimum change in monitored metric to qualify as improvement.
    """
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0