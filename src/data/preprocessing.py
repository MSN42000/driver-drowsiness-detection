"""
src/data/preprocessing.py
Image transforms for training, validation, and test sets.

Training uses data augmentation (random flips, rotation, color jitter) to reduce overfitting.
Validation and test use only resizing, tensor conversion, and normalization.
"""

from torchvision import transforms
from src.config import IMAGE_SIZE

def get_train_transforms():
    """
    Returns a composition of transforms for training data.
    Includes resizing, random horizontal flip, random rotation,
    color jitter, conversion to tensor, and normalization.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        # Normalize with mean and std calculated from typical eye image dataset
        # These values work reasonably for RGB images; adjust if needed.
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_val_test_transforms():
    """
    Returns a composition of transforms for validation and test data.
    Only resizing, tensor conversion, and normalization (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])