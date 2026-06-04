"""
src/models/custom_cnn.py
Custom CNN architectures for binary eye state classification.

Two models are provided:
- CNN_Small: 2 conv layers + 1 FC (lightweight, fast on CPU)
- CNN_Medium: 3 conv layers + 2 FC (slightly larger, more capacity)

Both output a single logit (no sigmoid) for use with BCEWithLogitsLoss.
"""

import torch
import torch.nn as nn
from src.config import IMAGE_SIZE

class CNN_Small(nn.Module):
    """
    Small CNN with 2 convolutional blocks and 1 fully connected layer.
    Designed for CPU training and fast inference.
    """
    def __init__(self, input_channels=3, num_classes=1):
        super(CNN_Small, self).__init__()
        
        # Convolutional block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Convolutional block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the size after convolutions and pooling
        # Input: (batch, 3, IMAGE_SIZE, IMAGE_SIZE)
        # After conv1: (batch, 16, IMAGE_SIZE/2, IMAGE_SIZE/2)
        # After conv2: (batch, 32, IMAGE_SIZE/4, IMAGE_SIZE/4)
        self.feature_size = 32 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)  # num_classes=1 for binary
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x


class CNN_Medium(nn.Module):
    """
    Medium CNN with 3 convolutional blocks and 2 fully connected layers.
    Slightly more capacity than CNN_Small.
    """
    def __init__(self, input_channels=3, num_classes=1):
        super(CNN_Medium, self).__init__()
        
        # Convolutional block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Convolutional block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Convolutional block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate feature size after 3 pooling layers (IMAGE_SIZE / 2^3)
        self.feature_size = 128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_model(model_name="small"):
    """
    Factory function to return a custom CNN model.

    Args:
        model_name (str): Either "small" (CNN_Small) or "medium" (CNN_Medium).

    Returns:
        nn.Module instance.
    """
    if model_name == "small":
        return CNN_Small()
    elif model_name == "medium":
        return CNN_Medium()
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'small' or 'medium'.")