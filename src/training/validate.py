"""
src/training/validate.py
Validation loop for evaluating model performance on validation or test sets.

Computes loss, accuracy, precision, recall, and F1 score.
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def validate(model, dataloader, criterion, device):
    """
    Evaluate the model on a dataset.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        dataloader (DataLoader): DataLoader for validation or test set.
        criterion (loss function): Loss function (e.g., BCEWithLogitsLoss).
        device (torch.device): 'cpu' or 'cuda'.

    Returns:
        dict: Contains 'loss', 'accuracy', 'precision', 'recall', 'f1'.
    """
    model.eval()  # Set model to evaluation mode (disables dropout, batch norm uses running stats)
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # Disable gradient computation for efficiency
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # Shape: (batch, 1) for binary BCE
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            # Convert logits to predictions (threshold 0.5 after sigmoid)
            preds = torch.sigmoid(outputs).cpu().numpy()
            preds_binary = (preds >= 0.5).astype(int)
            all_preds.extend(preds_binary.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    # Compute metrics
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }