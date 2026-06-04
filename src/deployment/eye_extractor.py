"""
src/deployment/eye_extractor.py
Extract eye region from face bounding box and preprocess for model inference.

Uses heuristic ratios to locate the eye (assuming upright face).
The extracted eye is resized, normalized, and converted to a PyTorch tensor.
"""

import cv2
import torch
import numpy as np
from torchvision import transforms
from src.config import IMAGE_SIZE

# Normalization parameters (same as training)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Preprocessing transform for a single eye image (no augmentation, just resize, tensor, normalize)
_eye_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

def get_left_eye_bbox(face_bbox, eye_width_ratio=0.3, eye_height_ratio=0.2, left_offset_x=0.1, top_offset_y=0.3):
    """
    Calculate the bounding box for the left eye within the face bounding box.

    Args:
        face_bbox (tuple): (x, y, w, h) of the face.
        eye_width_ratio (float): Width of eye relative to face width.
        eye_height_ratio (float): Height of eye relative to face height.
        left_offset_x (float): X offset from left of face as fraction of face width.
        top_offset_y (float): Y offset from top of face as fraction of face height.

    Returns:
        tuple: (eye_x, eye_y, eye_w, eye_h) in absolute image coordinates.
    """
    x, y, w, h = face_bbox
    eye_w = int(w * eye_width_ratio)
    eye_h = int(h * eye_height_ratio)
    eye_x = x + int(w * left_offset_x)
    eye_y = y + int(h * top_offset_y)
    return (eye_x, eye_y, eye_w, eye_h)

def extract_eye_roi(frame, eye_bbox):
    """
    Crop the eye region from the frame.

    Args:
        frame (numpy.ndarray): BGR image.
        eye_bbox (tuple): (x, y, w, h) of the eye.

    Returns:
        numpy.ndarray: Cropped eye image in BGR (original format).
    """
    x, y, w, h = eye_bbox
    # Ensure coordinates are within frame boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    if w <= 0 or h <= 0:
        return None
    eye_roi = frame[y:y+h, x:x+w]
    return eye_roi

def preprocess_eye(eye_roi):
    """
    Convert eye ROI to RGB, resize, normalize, and return a PyTorch tensor.

    Args:
        eye_roi (numpy.ndarray): Cropped eye image in BGR.

    Returns:
        torch.Tensor: Tensor of shape (1, 3, IMAGE_SIZE, IMAGE_SIZE) ready for model.
    """
    if eye_roi is None or eye_roi.size == 0:
        return None
    # Convert BGR to RGB (model expects RGB)
    eye_rgb = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2RGB)
    # Apply transform
    eye_tensor = _eye_transform(eye_rgb)
    # Add batch dimension
    eye_tensor = eye_tensor.unsqueeze(0)  # (1, 3, H, W)
    return eye_tensor

def get_eye_for_model(frame, face_bbox, use_left_eye=True):
    """
    High-level function to extract and preprocess the eye from a face bounding box.

    Args:
        frame (numpy.ndarray): BGR frame.
        face_bbox (tuple): (x, y, w, h) of the face.
        use_left_eye (bool): If True, extract left eye; if False, extract right eye.
                             (Right eye extraction is similar but with different offset.)

    Returns:
        torch.Tensor or None: Preprocessed eye tensor ready for model, or None if extraction fails.
    """
    if use_left_eye:
        # Left eye is on the left side of the face (lower x)
        eye_bbox = get_left_eye_bbox(face_bbox)
    else:
        # For right eye: offset from left approx 0.6 to 0.7 of face width
        eye_bbox = get_left_eye_bbox(face_bbox, left_offset_x=0.6)  # reuse function with different offset
        # Note: For simplicity, we only implement left eye in this example.
        # You can extend for right eye similarly.
    
    eye_roi = extract_eye_roi(frame, eye_bbox)
    if eye_roi is None:
        return None
    eye_tensor = preprocess_eye(eye_roi)
    return eye_tensor