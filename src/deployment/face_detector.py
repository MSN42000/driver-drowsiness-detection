"""
src/deployment/face_detector.py
Face detection using OpenCV Haar cascade.

This module loads the pre-trained frontal face cascade and provides
a function to detect the largest face in an image frame.
"""

import cv2

# Path to OpenCV's pre-trained Haar cascade (comes with opencv-python package)
# If the default path doesn't work, you can download the XML from OpenCV's GitHub.
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

def load_face_cascade():
    """
    Load the Haar cascade classifier for frontal face detection.

    Returns:
        cv2.CascadeClassifier: The loaded cascade classifier.
    """
    cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {HAAR_CASCADE_PATH}")
    return cascade

def detect_face(frame, cascade, scale_factor=1.1, min_neighbors=5, min_size=(60, 60)):
    """
    Detect the largest face in the given frame.

    Args:
        frame (numpy.ndarray): BGR image from webcam or file.
        cascade (cv2.CascadeClassifier): Loaded face cascade.
        scale_factor (float): Parameter specifying how much the image size is reduced at each scale.
        min_neighbors (int): Minimum number of neighbors to retain a detection.
        min_size (tuple): Minimum face size (width, height).

    Returns:
        tuple or None: (x, y, w, h) coordinates of the bounding box of the largest face,
                       or None if no face detected.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)
    
    if len(faces) == 0:
        return None
    
    # Return the largest face by area (width * height)
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    return (x, y, w, h)

def draw_face_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    """
    Draw a rectangle around the face bounding box.

    Args:
        frame (numpy.ndarray): Image to draw on.
        bbox (tuple): (x, y, w, h).
        color (tuple): BGR color.
        thickness (int): Line thickness.

    Returns:
        numpy.ndarray: Frame with rectangle drawn (modifies in place).
    """
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    return frame