"""
src/evaluation/metrics.py
Compute standard classification metrics for binary eye state prediction.

Metrics: accuracy, precision, recall, f1, and AUC-ROC.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def compute_metrics(y_true, y_pred, y_score=None):
    """
    Compute classification metrics for binary labels.

    Args:
        y_true (list or array): Ground truth labels (0 or 1).
        y_pred (list or array): Predicted labels (0 or 1) after thresholding.
        y_score (list or array, optional): Predicted probabilities (for AUC-ROC).

    Returns:
        dict: Contains 'accuracy', 'precision', 'recall', 'f1', and optionally 'auc_roc'.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_score is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_score)
        except ValueError:
            # If only one class present in y_true, AUC is not defined
            metrics['auc_roc'] = float('nan')
    
    return metrics