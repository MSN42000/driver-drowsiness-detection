"""
src/evaluation/plots.py
Plotting utilities for training history, confusion matrix, and ROC curve.

All plots are saved as PNG files.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score
from pathlib import Path

def plot_training_history(csv_path, save_path):
    """
    Plot training and validation loss/accuracy from a CSV file.

    Args:
        csv_path (str or Path): Path to training_history.csv.
        save_path (str or Path): Where to save the figure (PNG).
    """
    df = pd.read_csv(csv_path)
    epochs = df['epoch']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, df['train_loss'], label='Train Loss', marker='o', linewidth=2)
    ax1.plot(epochs, df['val_loss'], label='Val Loss', marker='s', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, df['train_acc'], label='Train Accuracy', marker='o', linewidth=2)
    ax2.plot(epochs, df['val_acc'], label='Val Accuracy', marker='s', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot a confusion matrix heatmap.

    Args:
        y_true (list or array): Ground truth labels.
        y_pred (list or array): Predicted labels.
        class_names (list): Names of classes in order [Non Drowsy, Drowsy].
        save_path (str or Path): Where to save the figure (PNG).
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_roc_curve(y_true, y_score, save_path):
    """
    Plot ROC curve and compute AUC.

    Args:
        y_true (list or array): Ground truth labels.
        y_score (list or array): Predicted probabilities (sigmoid output).
        save_path (str or Path): Where to save the figure (PNG).
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path} (AUC = {auc:.3f})")