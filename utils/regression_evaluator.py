# utils/regression_evaluator.py - FIXED VERSION
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class RegressionEvaluator:
    def __init__(self, threshold=25.0):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and targets"""
        self.all_targets = []
        self.all_preds = []
        self.all_losses = []
    
    def _compute_c_index(self, y_true, y_pred):
        """
        Compute concordance index (C-index) for ranking performance.
        
        C-index measures the probability that for a random pair of patients,
        the one with higher true RS score also has higher predicted RS score.
        
        Returns:
            float: C-index value between 0 and 1 (0.5 = random, 1.0 = perfect concordance)
        """
        if len(y_true) < 2:
            return 0.5
        
        concordant_pairs = 0
        total_pairs = 0
        
        n = len(y_true)
        for i in range(n):
            for j in range(i + 1, n):
                # Skip tied true values
                if y_true[i] == y_true[j]:
                    continue
                
                total_pairs += 1
                
                # Check if predicted ranking matches true ranking
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    concordant_pairs += 1
        
        return concordant_pairs / total_pairs if total_pairs > 0 else 0.5
    
    def _compute_classification_metrics(self, targets, preds, threshold):
        """Compute classification metrics at given threshold - FIXED VERSION"""
        binary_targets = (targets >= threshold).astype(int)
        binary_preds = (preds >= threshold).astype(int)
        
        metrics = {}
        
        # Basic classification metrics
        metrics['binary_accuracy'] = np.mean(binary_targets == binary_preds)
        
        # Handle case where all predictions are same class
        try:
            metrics['precision'] = precision_score(binary_targets, binary_preds, zero_division=0)
            metrics['recall'] = recall_score(binary_targets, binary_preds, zero_division=0) 
            metrics['f1_score'] = f1_score(binary_targets, binary_preds, zero_division=0)
            
            # Specificity (true negative rate)
            tn_mask = (binary_targets == 0) & (binary_preds == 0)
            fp_mask = (binary_targets == 0) & (binary_preds == 1)
            tn = np.sum(tn_mask)
            fp = np.sum(fp_mask)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Balanced accuracy
            metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
            
        except Exception as e:
            print(f"Warning in classification metrics: {e}")
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1_score'] = 0.0
            metrics['specificity'] = 0.0
            metrics['balanced_accuracy'] = 0.5
        
        # FIXED: AUROC and AUC-PR calculations
        if len(np.unique(binary_targets)) > 1:  # Need both classes
            try:
                # Use the raw continuous predictions as probability scores
                # This is more appropriate than normalizing
                metrics['auroc'] = roc_auc_score(binary_targets, preds)
                
                # For AUC-PR, we need probabilities in [0, 1]
                # Use sigmoid transformation for better calibration
                # This preserves relative ordering while mapping to probabilities
                pred_probs = 1 / (1 + np.exp(-(preds - threshold) / 10))
                metrics['auc_pr'] = average_precision_score(binary_targets, pred_probs)
                
            except Exception as e:
                print(f"Warning in AUROC/AUC-PR calculation: {e}")
                # Fallback to simpler calculation
                try:
                    # Alternative: use min-max normalization
                    if np.max(preds) > np.min(preds):
                        pred_probs_alt = (preds - np.min(preds)) / (np.max(preds) - np.min(preds))
                        metrics['auroc'] = roc_auc_score(binary_targets, pred_probs_alt)
                        metrics['auc_pr'] = average_precision_score(binary_targets, pred_probs_alt)
                    else:
                        metrics['auroc'] = 0.5
                        metrics['auc_pr'] = np.mean(binary_targets)
                except:
                    metrics['auroc'] = 0.5
                    metrics['auc_pr'] = np.mean(binary_targets)
        else:
            # Only one class present
            metrics['auroc'] = 0.5  # No discrimination possible
            metrics['auc_pr'] = np.mean(binary_targets)  # Baseline for imbalanced classes
        
        return metrics
    
    def update(self, targets, preds, losses=None):
        """
        Update with batch predictions
        Args:
            targets: true RS scores
            preds: predicted RS scores
            losses: batch losses
        """
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
            
        # Ensure we're working with flat arrays
        targets = np.atleast_1d(targets).flatten()
        preds = np.atleast_1d(preds).flatten()
        
        self.all_targets.extend(targets.tolist())
        self.all_preds.extend(preds.tolist())
        
        if losses is not None:
            if isinstance(losses, torch.Tensor):
                losses = losses.cpu().numpy()
            losses = np.atleast_1d(losses).flatten()
            self.all_losses.extend(losses.tolist())
    
    def compute_all_metrics(self, verbose=True):
        """Compute comprehensive regression and classification metrics - FIXED VERSION"""
        if len(self.all_targets) == 0:
            return {}
            
        targets = np.array(self.all_targets)
        preds = np.array(self.all_preds)
        
        metrics = {}
        
        # === Core Regression Metrics ===
        metrics['mse'] = mean_squared_error(targets, preds)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(targets, preds)
        
        # R² score (coefficient of determination)
        if len(np.unique(targets)) > 1:  # Need variance in targets
            metrics['r2'] = r2_score(targets, preds)
        else:
            metrics['r2'] = 0.0
        
        # FIXED: Spearman's rank correlation coefficient
        # Check for sufficient variation in BOTH arrays
        n_unique_targets = len(np.unique(targets))
        n_unique_preds = len(np.unique(preds))
        
        if n_unique_targets > 1 and n_unique_preds > 1:
            try:
                spearman_corr, spearman_p = spearmanr(targets, preds)
                # Handle NaN cases
                if np.isnan(spearman_corr):
                    metrics['spearman_correlation'] = 0.0
                    metrics['spearman_p_value'] = 1.0
                else:
                    metrics['spearman_correlation'] = float(spearman_corr)
                    metrics['spearman_p_value'] = float(spearman_p)
            except Exception as e:
                print(f"Warning in Spearman calculation: {e}")
                metrics['spearman_correlation'] = 0.0
                metrics['spearman_p_value'] = 1.0
        else:
            # Not enough variation for correlation
            if verbose:
                print(f"Warning: Insufficient variation for Spearman correlation")
                print(f"  Unique targets: {n_unique_targets}, Unique predictions: {n_unique_preds}")
            metrics['spearman_correlation'] = 0.0
            metrics['spearman_p_value'] = 1.0
        
        # C-index (concordance index) for ranking performance
        metrics['c_index'] = self._compute_c_index(targets, preds)
        
        # === Classification Metrics at Threshold ===
        classification_metrics = self._compute_classification_metrics(targets, preds, self.threshold)
        metrics.update(classification_metrics)
        
        # === Boundary-Specific Metrics ===
        # Performance near clinical threshold (RS=25)
        boundary_mask = np.abs(targets - self.threshold) <= 10.0
        if np.sum(boundary_mask) > 0:
            boundary_targets = targets[boundary_mask]
            boundary_preds = preds[boundary_mask]
            metrics['boundary_mae'] = mean_absolute_error(boundary_targets, boundary_preds)
            metrics['boundary_rmse'] = np.sqrt(mean_squared_error(boundary_targets, boundary_preds))
            metrics['n_boundary_samples'] = np.sum(boundary_mask)
        else:
            metrics['boundary_mae'] = 0.0
            metrics['boundary_rmse'] = 0.0
            metrics['n_boundary_samples'] = 0
        
        # === Distribution Analysis ===
        metrics['pred_mean'] = np.mean(preds)
        metrics['pred_std'] = np.std(preds)
        metrics['target_mean'] = np.mean(targets)
        metrics['target_std'] = np.std(targets)
        
        # Prediction range and diversity
        metrics['pred_min'] = np.min(preds)
        metrics['pred_max'] = np.max(preds)
        metrics['n_unique_preds'] = n_unique_preds
        metrics['n_unique_targets'] = n_unique_targets
        
        # === Loss ===
        if self.all_losses:
            metrics['avg_loss'] = np.mean(self.all_losses)
        
        # === Sample Statistics ===
        metrics['n_samples'] = len(targets)
        metrics['n_high_risk_actual'] = np.sum(targets >= self.threshold)
        metrics['n_high_risk_predicted'] = np.sum(preds >= self.threshold)
        
        # === Additional Diagnostics ===
        # Check for potential issues
        if n_unique_preds < 5:
            metrics['warning_low_pred_diversity'] = True
            if verbose:
                print(f"\n⚠️ WARNING: Low prediction diversity ({n_unique_preds} unique values)")
                print(f"   This may indicate model collapse or insufficient learning")
        
        if verbose:
            self.print_metrics_summary(metrics)
        
        return metrics
    
    def print_metrics_summary(self, metrics):
        """Print formatted metrics summary - ENHANCED VERSION"""
        print("\n" + "="*70)
        print("COMPREHENSIVE REGRESSION & CLASSIFICATION METRICS")
        print("="*70)
        
        print(f"Dataset Size: {metrics['n_samples']} samples")
        print(f"Target Range: {metrics['target_mean']:.2f} ± {metrics['target_std']:.2f}")
        print(f"Prediction Range: {metrics['pred_mean']:.2f} ± {metrics['pred_std']:.2f} "
              f"[{metrics['pred_min']:.2f}, {metrics['pred_max']:.2f}]")
        print(f"Prediction Diversity: {metrics['n_unique_preds']} unique values")
        
        print(f"\nCORE REGRESSION METRICS:")
        print(f"  RMSE:              {metrics['rmse']:.4f}")
        print(f"  MAE:               {metrics['mae']:.4f}")
        print(f"  R² Score:          {metrics['r2']:.4f}")
        print(f"  Spearman ρ:        {metrics['spearman_correlation']:.4f} (p={metrics['spearman_p_value']:.4f})")
        print(f"  C-index:           {metrics['c_index']:.4f}")
        
        print(f"\nCLASSIFICATION METRICS (RS≥{self.threshold}):")
        print(f"  AUROC:             {metrics['auroc']:.4f}")
        print(f"  AUC-PR:            {metrics['auc_pr']:.4f}")
        print(f"  Binary Accuracy:   {metrics['binary_accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics.get('balanced_accuracy', 0.5):.4f}")
        print(f"  Precision:         {metrics['precision']:.4f}")
        print(f"  Recall:            {metrics['recall']:.4f}")
        print(f"  F1-Score:          {metrics['f1_score']:.4f}")
        print(f"  Specificity:       {metrics['specificity']:.4f}")
        
        print(f"\nBOUNDARY PERFORMANCE (±10 from RS={self.threshold}):")
        print(f"  Boundary RMSE:     {metrics['boundary_rmse']:.4f}")
        print(f"  Boundary MAE:      {metrics['boundary_mae']:.4f}")
        print(f"  Boundary Samples:  {metrics.get('n_boundary_samples', 0)}")
        
        print(f"\nSAMPLE DISTRIBUTION:")
        print(f"  High Risk (Actual):     {metrics['n_high_risk_actual']}/{metrics['n_samples']}")
        print(f"  High Risk (Predicted):  {metrics['n_high_risk_predicted']}/{metrics['n_samples']}")
        
        if 'avg_loss' in metrics:
            print(f"\nLOSS:")
            print(f"  Average Loss:      {metrics['avg_loss']:.4f}")
        
        if metrics.get('warning_low_pred_diversity', False):
            print(f"\n⚠️ WARNINGS:")
            print(f"  - Low prediction diversity detected!")
        
        print("="*70)