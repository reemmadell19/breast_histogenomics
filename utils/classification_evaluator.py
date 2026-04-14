
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, precision_recall_curve, roc_curve, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class ClassificationEvaluator:
    def __init__(self, n_classes=2):
        self.n_classes = n_classes
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and labels"""
        self.all_labels = []
        self.all_preds = []
        self.all_probs = []
        self.all_losses = []
    
    def update(self, labels, preds, probs=None, losses=None):
        """
        Update with batch predictions
        Args:
            labels: true labels
            preds: predicted classes 
            probs: predicted probabilities (softmax outputs)
            losses: batch losses
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if probs is not None and isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
            
        self.all_labels.extend(labels if isinstance(labels, (list, np.ndarray)) else [labels])
        self.all_preds.extend(preds if isinstance(preds, (list, np.ndarray)) else [preds])
        
        if probs is not None:
            self.all_probs.extend(probs if isinstance(probs, (list, np.ndarray)) else [probs])
        
        if losses is not None:
            if isinstance(losses, torch.Tensor):
                losses = losses.cpu().numpy()
            self.all_losses.extend(losses if isinstance(losses, (list, np.ndarray)) else [losses])
    
    def compute_all_metrics(self, verbose=True):
        """Compute comprehensive classification metrics"""
        if len(self.all_labels) == 0:
            return {}
            
        labels = np.array(self.all_labels)
        preds = np.array(self.all_preds)
        
        metrics = {}
        
        # === Core Classification Metrics ===
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['balanced_accuracy'] = balanced_accuracy_score(labels, preds)
        
        # Per-class metrics for binary classification
        if self.n_classes == 2:
            metrics['precision'] = precision_score(labels, preds, zero_division=0)
            metrics['recall'] = recall_score(labels, preds, zero_division=0)
            metrics['sensitivity'] = metrics['recall']  # Same as recall
            metrics['f1_score'] = f1_score(labels, preds, zero_division=0)
            
            metrics['cohen_kappa'] = cohen_kappa_score(labels, preds)
            # Specificity calculation
            cm = confusion_matrix(labels, preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
                metrics['ppv'] = metrics['precision']  # Positive Predictive Value
            
            # Matthews Correlation Coefficient
            metrics['mcc'] = matthews_corrcoef(labels, preds)
            
            # ROC and PR metrics if probabilities available
            if len(self.all_probs) > 0:
                probs = np.array(self.all_probs)
                
                # Handle both softmax output and single probability
                if probs.ndim == 2 and probs.shape[1] == 2:
                    pos_probs = probs[:, 1]  # Probability of positive class
                else:
                    pos_probs = probs.flatten()
                
                if len(np.unique(labels)) > 1:  # Only if both classes present
                    metrics['auroc'] = roc_auc_score(labels, pos_probs)
                    metrics['auc_pr'] = average_precision_score(labels, pos_probs)
                else:
                    metrics['auroc'] = 0.5
                    metrics['auc_pr'] = 0.5
        
        # === Loss ===
        if self.all_losses:
            metrics['avg_loss'] = np.mean(self.all_losses)
        
        # === Class Distribution ===
        metrics['n_samples'] = len(labels)
        if self.n_classes == 2:
            metrics['n_positive'] = np.sum(labels == 1)
            metrics['n_negative'] = np.sum(labels == 0)
            metrics['class_ratio'] = metrics['n_positive'] / metrics['n_samples']
        
        if verbose:
            self.print_metrics_summary(metrics)
        
        return metrics
    
    def print_metrics_summary(self, metrics):
        """Print formatted metrics summary"""
        print("\n" + "="*60)
        print("CLASSIFICATION METRICS SUMMARY")
        print("="*60)
        
        print(f"Dataset Size: {metrics['n_samples']} samples")
        if self.n_classes == 2:
            print(f"Class Distribution: {metrics['n_positive']} high-risk ({metrics['class_ratio']:.1%}), "
                  f"{metrics['n_negative']} low-risk ({1-metrics['class_ratio']:.1%})")
        
        print(f"\nCORE METRICS:")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        if self.n_classes == 2:
            print(f"  F1-Score:          {metrics['f1_score']:.4f}")
        
        if self.n_classes == 2:
            print(f"\nCLINICAL METRICS:")
            print(f"  Sensitivity:       {metrics.get('sensitivity', 0):.4f}")
            print(f"  Specificity:       {metrics.get('specificity', 0):.4f}")
            print(f"  PPV (Precision):   {metrics.get('ppv', 0):.4f}")
            print(f"  NPV:               {metrics.get('npv', 0):.4f}")
            
            print(f"\nDISCRIMINATIVE POWER:")
            print(f"  AUC-ROC:           {metrics.get('auroc', 0):.4f}")
            print(f"  AUC-PR:            {metrics.get('auc_pr', 0):.4f}")
            print(f"  Matthews Corr:     {metrics.get('mcc', 0):.4f}")
            print(f"  Cohen's Kappa:     {metrics.get('cohen_kappa', 0):.4f}")  # ADD THIS LINE
    
        if 'avg_loss' in metrics:
            print(f"\nLOSS:")
            print(f"  Average Loss:      {metrics['avg_loss']:.4f}")
        
        print("="*60)
    
    def get_confusion_matrix(self):
        """Return confusion matrix"""
        if len(self.all_labels) == 0:
            return None
        return confusion_matrix(self.all_labels, self.all_preds)
    
    def plot_confusion_matrix(self, save_path=None, title="Confusion Matrix"):
        """Plot confusion matrix"""
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
    
    def plot_roc_curve(self, save_path=None, title="ROC Curve"):
        """Plot ROC curve"""
        if len(self.all_probs) == 0 or self.n_classes != 2:
            return None
        
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        
        if probs.ndim == 2 and probs.shape[1] == 2:
            pos_probs = probs[:, 1]
        else:
            pos_probs = probs.flatten()
        
        fpr, tpr, _ = roc_curve(labels, pos_probs)
        auc_score = roc_auc_score(labels, pos_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return plt.gcf()