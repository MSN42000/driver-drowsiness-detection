"""
src/config.py
Central configuration for the Driver Drowsiness Detection project.

Modify values here to control training, evaluation, and deployment.
"""

import torch
import os

# Dataset selection (change this to switch between sampled and full data)
# I used sampled here for resources (10k images total),
# but you can set to False if your machine is powerful enough for 41k images.
USE_SUBSAMPLED = True   # True = use data/processed/sampled/ , False = use data/processed/full/

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_ROOT = os.path.join(PROJECT_ROOT, "data")                     # contains Drowsy/ and Non Drowsy/
DATA_PROCESSED_ROOT = os.path.join(PROJECT_ROOT, "data", "processed")

# Subsampled symlink directory (created by src/data/subsample.py)
DATA_SUBSAMPLED_ROOT = os.path.join(PROJECT_ROOT, "data", "subsampled")

# Where to save trained models
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models")

# Where to save evaluation outputs (plots, CSVs)
OUTPUTS_PATH = os.path.join(PROJECT_ROOT, "outputs")

# Training hyperparameters
IMAGE_SIZE = 96           # Height and width (square) – reduced for CPU speed
BATCH_SIZE = 8            # Small batch due to 8GB RAM and CPU
NUM_EPOCHS = 20           # Enough for convergence with 10k images
LEARNING_RATE = 0.001     # For custom CNN; pretrained model uses lower LR inside train.py
NUM_WORKERS = 2           # DataLoader workers (CPU cores)

# Device (CPU only – no GPU detected)
DEVICE = torch.device("cpu")

# Class names (order matters: index 0 = Non Drowsy, index 1 = Drowsy)
CLASS_NAMES = ["Non Drowsy", "Drowsy"]
NUM_CLASSES = len(CLASS_NAMES)   # 2 for binary, but we use sigmoid output (1 neuron)

# Random seed for reproducibility
RANDOM_SEED = 42

# Early stopping patience (epochs with no improvement)
EARLY_STOPPING_PATIENCE = 5

# Training history CSV filename (will be placed inside OUTPUTS_PATH/model_name/)
HISTORY_FILENAME = "training_history.csv"