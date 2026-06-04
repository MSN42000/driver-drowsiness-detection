"""
src/deployment/webcam_demo.py
Real-time drowsiness detection using webcam.

Usage:
    python -m src.deployment.webcam_demo --model_path models/pretrained_sampled_best.pth
"""

import argparse
import cv2
import torch
import torch.nn as nn
import time
from pathlib import Path

from src.config import DEVICE, CLASS_NAMES
from src.deployment.face_detector import load_face_cascade, detect_face, draw_face_bbox
from src.deployment.eye_extractor import get_eye_for_model
from src.models.custom_cnn import get_model as get_custom_model
from src.models.pretrained_model import get_pretrained_model
from src.training.utils import load_checkpoint

def load_model(model_path, model_type):
    """
    Load a trained model from a checkpoint.

    Args:
        model_path (str): Path to .pth checkpoint.
        model_type (str): 'custom' or 'pretrained' (used to build architecture).

    Returns:
        nn.Module: Loaded model in eval mode.
    """
    if model_type == 'custom':
        model = get_custom_model('small')
    else:
        model = get_pretrained_model(freeze_backbone=True, num_classes=1)
    
    # Load weights (ignore optimizer)
    load_checkpoint(model_path, model, optimizer=None)
    model = model.to(DEVICE)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Real-time drowsiness detection from webcam.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model .pth file (e.g., models/custom_sampled_best.pth)")
    parser.add_argument("--use_left_eye", action="store_true", default=True,
                        help="Use left eye for inference (default: True). Right eye not fully implemented.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for classifying as drowsy (closed eyes).")
    args = parser.parse_args()

    # Infer model type from filename (heuristic)
    model_path = Path(args.model_path)
    if "custom" in model_path.stem:
        model_type = "custom"
    elif "pretrained" in model_path.stem:
        model_type = "pretrained"
    else:
        print("Could not infer model type from filename. Assuming 'custom'.")
        model_type = "custom"

    # Load model
    print(f"Loading model from {model_path} (type: {model_type})")
    model = load_model(model_path, model_type)

    # Load face cascade
    face_cascade = load_face_cascade()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # For FPS calculation
    frame_count = 0
    fps_start_time = time.time()
    fps = 0

    print("\nStarting webcam demo. Press 'q' to quit.")
    print("Face detection and eye classification running...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Mirror frame for natural self-view (optional)
        frame = cv2.flip(frame, 1)

        # Detect face
        face_bbox = detect_face(frame, face_cascade)
        if face_bbox is not None:
            # Draw face rectangle
            draw_face_bbox(frame, face_bbox, color=(0, 255, 0), thickness=2)

            # Extract and preprocess eye
            eye_tensor = get_eye_for_model(frame, face_bbox, use_left_eye=args.use_left_eye)
            if eye_tensor is not None:
                eye_tensor = eye_tensor.to(DEVICE)

                # Run model
                with torch.no_grad():
                    logit = model(eye_tensor)
                    prob = torch.sigmoid(logit).item()

                # Interpret result: prob > threshold -> drowsy (closed eyes)
                is_drowsy = prob > args.threshold
                label = "DROWSY" if is_drowsy else "ACTIVE"
                color = (0, 0, 255) if is_drowsy else (0, 255, 0)  # red for drowsy, green for active

                # Display probability and label on frame
                cv2.putText(frame, f"{label} ({prob:.2f})", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                cv2.putText(frame, "Eye extraction failed", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Calculate and display FPS
        frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            fps = frame_count / (time.time() - fps_start_time)
            frame_count = 0
            fps_start_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show frame
        cv2.imshow("Driver Drowsiness Detection", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam demo ended.")

if __name__ == "__main__":
    main()