"""
src/models/pretrained_model.py
Pretrained MobileNetV2 model for transfer learning.

MobileNetV2 is lightweight (~3.5M parameters) and runs reasonably on CPU.
The backbone is frozen, and only the classifier head is trained.
"""

import torch
import torch.nn as nn
from torchvision import models

def get_pretrained_model(freeze_backbone=True, num_classes=1):
    """
    Load a pretrained MobileNetV2 model and replace the classifier head.

    Args:
        freeze_backbone (bool): If True, freeze all layers except the final classifier.
                                If False, allow fine-tuning of all layers (slower).
        num_classes (int): Number of output classes (1 for binary classification).

    Returns:
        model (nn.Module): Modified MobileNetV2 model ready for training.
    """
    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(pretrained=True)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the classifier head
    # MobileNetV2's classifier is: nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 1000))
    # We replace the last linear layer to output num_classes (1 for binary)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    # Optional: If you want to unfreeze the last few layers for fine-tuning later,
    # you can set requires_grad=True for those layers here.
    # For CPU training, keeping backbone frozen is recommended for speed.
    
    return model


def get_pretrained_model_for_finetune(num_classes=1, unfreeze_last_n=2):
    """
    Alternative: Load pretrained model and unfreeze the last `unfreeze_last_n` layers.
    This allows limited fine-tuning while keeping most of the backbone frozen.
    Use this if you want slightly better accuracy at the cost of training time.

    Args:
        num_classes (int): Number of output classes.
        unfreeze_last_n (int): Number of final layers to unfreeze (counting from the end).

    Returns:
        model (nn.Module): MobileNetV2 with unfrozen final layers.
    """
    model = models.mobilenet_v2(pretrained=True)
    
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the classifier and last n layers of features
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Unfreeze the last `unfreeze_last_n` blocks in the feature extractor
    # MobileNetV2 features are in 18 inverted residual blocks (index 0 to 17)
    blocks = list(model.features.children())
    for block in blocks[-unfreeze_last_n:]:
        for param in block.parameters():
            param.requires_grad = True
    
    # Replace classifier output layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model