# evaluation_metrics.py

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

class MILEvaluator:

    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and labels"""
        self.all_labels = []
        self.all_preds = []
        self.all_probs = []
        self.all_losses = []
    
    def update(self, labels, preds, probs, losses=None):
        """
        Update with batch predictions
        Args:
            labels: true labels
            preds: predicted classes 
            probs: predicted probabilities for positive class 
            losses: batch losses
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
            
        self.all_labels.extend(labels if isinstance(labels, list) else [labels])
        self.all_preds.extend(preds if isinstance(preds, list) else [preds])
        self.all_probs.extend(probs if isinstance(probs, list) else [probs])
        
        if losses is not None:
            if isinstance(losses, torch.Tensor):
                losses = losses.cpu().numpy()
            self.all_losses.extend(losses if isinstance(losses, list) else [losses])
    
    def compute_all_metrics(self, verbose=True):
       
        if len(self.all_labels) == 0:
            return {}
            
        labels = np.array(self.all_labels)
        preds = np.array(self.all_preds)
        probs = np.array(self.all_probs)
        
        metrics = {}
        
        # === Core Classification Metrics ===
        metrics['accuracy'] = np.mean(preds == labels)
        metrics['balanced_accuracy'] = balanced_accuracy_score(labels, preds)
        
        # Per-class metrics
        metrics['precision'] = precision_score(labels, preds, average='binary', zero_division=0)
        metrics['recall'] = recall_score(labels, preds, average='binary', zero_division=0)
        metrics['sensitivity'] = metrics['recall']  # Same as recall for binary
        metrics['f1_score'] = f1_score(labels, preds, average='binary', zero_division=0)
        
        # Specificity
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Precision)
        
        # === Advanced Metrics ===
        metrics['mcc'] = matthews_corrcoef(labels, preds)
        
        # ROC metrics
        if len(np.unique(labels)) > 1:  # Only if both classes present
            metrics['auc_roc'] = roc_auc_score(labels, probs)
            
            # Precision-Recall curve and AUC
            precision_vals, recall_vals, _ = precision_recall_curve(labels, probs)
            metrics['auc_pr'] = auc(recall_vals, precision_vals)
        else:
            metrics['auc_roc'] = 0.5
            metrics['auc_pr'] = 0.5
        
        # === Loss ===
        if self.all_losses:
            metrics['avg_loss'] = np.mean(self.all_losses)
        
        # === Class Distribution Analysis ===
        metrics['n_samples'] = len(labels)
        metrics['n_positive'] = np.sum(labels == 1)
        metrics['n_negative'] = np.sum(labels == 0)
        metrics['class_ratio'] = metrics['n_positive'] / metrics['n_samples']
        
        if verbose:
            self.print_metrics_summary(metrics)
        
        return metrics
    
    def print_metrics_summary(self, metrics):
        """Print formatted metrics summary"""
        print("\n" + "="*60)
        print("EVALUATION METRICS SUMMARY")
        print("="*60)
        
        print(f"Dataset Size: {metrics['n_samples']} samples")
        print(f"Class Distribution: {metrics['n_positive']} positive ({metrics['class_ratio']:.1%}), "
              f"{metrics['n_negative']} negative ({1-metrics['class_ratio']:.1%})")
        
        print(f"\n CORE METRICS:")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  F1-Score:          {metrics['f1_score']:.4f}")
        
        print(f"\n CLINICAL METRICS:")
        print(f"  Sensitivity (Recall): {metrics['sensitivity']:.4f}")
        print(f"  Specificity:          {metrics['specificity']:.4f}")
        print(f"  PPV (Precision):      {metrics['ppv']:.4f}")
        print(f"  NPV:                  {metrics['npv']:.4f}")
        
        print(f"\n DISCRIMINATIVE POWER:")
        print(f"  AUC-ROC:           {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:            {metrics['auc_pr']:.4f}")
        print(f"  Matthews Corr:     {metrics['mcc']:.4f}")
        
        if 'avg_loss' in metrics:
            print(f"\n LOSS:")
            print(f"  Average Loss:      {metrics['avg_loss']:.4f}")
        
        print("="*60)
    
    def get_confusion_matrix(self):
        """Return confusion matrix"""
        if len(self.all_labels) == 0:
            return None
        return confusion_matrix(self.all_labels, self.all_preds)
    
    def plot_confusion_matrix(self, save_path=None, title="Confusion Matrix"):
        """Plot and optionally save confusion matrix"""
        cm = self.get_confusion_matrix()
        if cm is None:
            return None
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Low Risk", "High Risk"],
                    yticklabels=["Low Risk", "High Risk"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label") 
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return plt.gcf()
    
    def get_classification_report(self):
        """Get detailed classification report"""
        if len(self.all_labels) == 0:
            return ""
        
        return classification_report(
            self.all_labels, 
            self.all_preds,
            target_names=["Low Risk", "High Risk"],
            digits=4
        )


def compute_metrics_for_epoch(labels, preds, probs, losses=None, prefix=""):
    """
    Utility function for computing metrics for a single epoch
    Returns: dict with metrics (keys prefixed with prefix)
    """
    evaluator = MILEvaluator()
    evaluator.update(labels, preds, probs, losses)
    metrics = evaluator.compute_all_metrics(verbose=False)
    
    # Add prefix to all keys
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    
    return metrics