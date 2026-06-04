"""
src/evaluation/test.py
Evaluate a trained model on the test set and generate metrics + plots.

Usage examples:
    python -m src.evaluation.test --model_type custom --dataset sampled
    python -m src.evaluation.test --model_type pretrained --dataset full
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path

from src.config import DEVICE, BATCH_SIZE, MODEL_SAVE_PATH, OUTPUTS_PATH, CLASS_NAMES
from src.data.dataset import get_dataloaders
from src.models.custom_cnn import get_model as get_custom_model
from src.models.pretrained_model import get_pretrained_model
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import plot_confusion_matrix, plot_roc_curve
from src.training.utils import load_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Test a trained model on the test set.")
    parser.add_argument("--model_type", type=str, choices=["custom", "pretrained"], required=True,
                        help="Model type to test: 'custom' or 'pretrained'")
    parser.add_argument("--dataset", type=str, choices=["sampled", "full"], required=True,
                        help="Dataset used for training: 'sampled' or 'full'")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for testing")
    args = parser.parse_args()
    
    # Override config dataset flag to match requested dataset
    import src.config as cfg
    cfg.USE_SUBSAMPLED = (args.dataset == "sampled")
    
    # Load test dataloader
    print(f"Loading test data from {args.dataset} dataset...")
    _, _, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=cfg.NUM_WORKERS)
    
    # Build model architecture
    if args.model_type == "custom":
        model = get_custom_model("small")
        print("Testing custom CNN (small)")
    else:
        model = get_pretrained_model(freeze_backbone=True, num_classes=1)
        print("Testing pretrained MobileNetV2")
    
    # Load best checkpoint
    model_filename = f"{args.model_type}_{args.dataset}_best.pth"
    model_path = Path(MODEL_SAVE_PATH) / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please train the model first.")
    
    load_checkpoint(model_path, model, optimizer=None)
    model = model.to(DEVICE)
    model.eval()
    
    # Run inference on test set
    all_labels = []
    all_probs = []
    
    criterion = nn.BCEWithLogitsLoss()
    test_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy().flatten())
    
    avg_test_loss = test_loss / len(test_loader.dataset)
    
    # Convert probabilities to binary predictions (threshold 0.5)
    all_preds = [1 if p >= 0.5 else 0 for p in all_probs]
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['test_loss'] = avg_test_loss
    
    print("\n" + "="*50)
    print(f"Test Results for {args.model_type} on {args.dataset} dataset")
    print("="*50)
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}" if not pd.isna(metrics['auc_roc']) else "AUC-ROC: N/A")
    print("="*50)
    
    # Prepare output directory
    output_dir = Path(OUTPUTS_PATH) / f"{args.model_type}_{args.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(all_labels, all_preds, CLASS_NAMES, cm_path)
    
    # Save ROC curve
    roc_path = output_dir / "roc_curve.png"
    plot_roc_curve(all_labels, all_probs, roc_path)
    
    # Save metrics to a text file
    metrics_path = output_dir / "test_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Test Loss: {metrics['test_loss']:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"AUC-ROC: {metrics['auc_roc']:.4f}\n")
    print(f"Metrics saved to {metrics_path}")
    
    # Also generate training history plot if the CSV exists
    history_csv = output_dir / "training_history.csv"
    if history_csv.exists():
        plot_path = output_dir / "accuracy_loss_plot.png"
        from src.evaluation.plots import plot_training_history
        plot_training_history(history_csv, plot_path)
    else:
        print(f"No training history CSV found at {history_csv}")

if __name__ == "__main__":
    import pandas as pd  # for isnull check
    main()