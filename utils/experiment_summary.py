# utils/experiment_summary.py
import os
import json
from datetime import datetime

def save_experiment_summary(experiment_config, best_model_metrics, final_model_metrics, results_dir):
    """
    Save a comprehensive experiment summary text file that matches the console output format.
    
    Args:
        experiment_config: Dictionary containing experiment configuration
        best_model_metrics: Dictionary of best model performance metrics
        final_model_metrics: Dictionary of final model performance metrics
        results_dir: Directory to save the summary file
    """
    
    summary_path = os.path.join(results_dir, "experiment_summary.txt")
    
    with open(summary_path, 'w') as f:
        # Header with timestamp
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results Directory: {results_dir}\n")
        f.write("=" * 80 + "\n")
        
        # Experiment Configuration
        f.write("\nEXPERIMENT CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Feature Extractor: {experiment_config.get('feature_extractor', 'N/A')}\n")
        f.write(f"MIL Architecture: {experiment_config.get('mil_model_name', experiment_config.get('mil_pooling', 'N/A'))}\n")
        f.write(f"Input Dimension: {experiment_config.get('input_dim', 'N/A')}\n")
        f.write(f"Hidden Dimension: {experiment_config.get('hidden_dim', 'N/A')}\n")
        f.write(f"Learning Rate: {experiment_config.get('lr', 'N/A')}\n")
        f.write(f"Number of Epochs: {experiment_config.get('num_epochs', 'N/A')}\n")
        f.write(f"Loss Type: {experiment_config.get('loss_type', 'N/A')}\n")
        if experiment_config.get('loss_type') == 'huber':
            f.write(f"Huber Delta: {experiment_config.get('huber_delta', 'N/A')}\n")
        f.write(f"Class Balancing: {experiment_config.get('use_class_balancing', 'N/A')}\n")
        f.write(f"Boundary Weighting: {experiment_config.get('use_boundary_weighting', 'N/A')}\n")
        f.write(f"Random Seed: {experiment_config.get('random_seed', 'N/A')}\n")
        
        # Performance Summary (matching the console output format exactly)
        f.write("\n" + "=" * 80 + "\n")
        f.write("TRAINING COMPLETED - PERFORMANCE SUMMARY\n")
        f.write("=" * 80 + "\n")
        
        f.write(f"Feature Extractor: {experiment_config.get('feature_extractor', 'N/A')}\n")
        f.write(f"MIL Architecture: {experiment_config.get('mil_model_name', experiment_config.get('mil_pooling', 'N/A'))}\n")
        f.write(f"Input Dimension: {experiment_config.get('input_dim', 'N/A')}\n")
        
        # Best model performance
        if best_model_metrics is not None:
            f.write(f"\nBEST MODEL PERFORMANCE (saved model):\n")
            f.write(f"  AUROC: {best_model_metrics.get('auroc', 0):.4f}\n")
            f.write(f"  RMSE: {best_model_metrics.get('rmse', 0):.4f}\n")
            f.write(f"  MAE: {best_model_metrics.get('mae', 0):.4f}\n")
            f.write(f"  R²: {best_model_metrics.get('r2', 0):.4f}\n")
            f.write(f"  Spearman ρ: {best_model_metrics.get('spearman_correlation', 0):.4f}\n")
            f.write(f"  Binary Accuracy: {best_model_metrics.get('binary_accuracy', 0):.4f}\n")
            f.write(f"  F1-Score: {best_model_metrics.get('f1_score', 0):.4f}\n")
            f.write(f"  Boundary MAE: {best_model_metrics.get('boundary_mae', 0):.4f}\n")
        
        # Final model performance
        if final_model_metrics is not None:
            f.write(f"\nFINAL MODEL PERFORMANCE (last epoch):\n")
            f.write(f"  AUROC: {final_model_metrics.get('auroc', 0):.4f}\n")
            f.write(f"  RMSE: {final_model_metrics.get('rmse', 0):.4f}\n")
            f.write(f"  MAE: {final_model_metrics.get('mae', 0):.4f}\n")
            f.write(f"  R²: {final_model_metrics.get('r2', 0):.4f}\n")
            f.write(f"  Spearman ρ: {final_model_metrics.get('spearman_correlation', 0):.4f}\n")
            f.write(f"  Binary Accuracy: {final_model_metrics.get('binary_accuracy', 0):.4f}\n")
            f.write(f"  F1-Score: {final_model_metrics.get('f1_score', 0):.4f}\n")
            f.write(f"  Boundary MAE: {final_model_metrics.get('boundary_mae', 0):.4f}\n")
        
        f.write(f"\nAll results saved to: {results_dir}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("EXPERIMENT SUMMARY COMPLETED\n")
        f.write("=" * 80 + "\n")
    
    print(f"Experiment summary saved to: {summary_path}")
    return summary_path

def save_quick_comparison_summary(experiment_config, best_model_metrics, results_dir):
    """
    Save a quick one-line summary for easy comparison across experiments.
    
    Args:
        experiment_config: Dictionary containing experiment configuration
        best_model_metrics: Dictionary of best model performance metrics
        results_dir: Directory to save the summary file
    """
    
    comparison_path = os.path.join(results_dir, "quick_summary.txt")
    
    with open(comparison_path, 'w') as f:
        # One-line summary for easy comparison
        feature = experiment_config.get('feature_extractor', 'N/A')
        mil = experiment_config.get('mil_pooling', 'N/A')
        
        if best_model_metrics:
            auroc = best_model_metrics.get('auroc', 0)
            rmse = best_model_metrics.get('rmse', 0)
            mae = best_model_metrics.get('mae', 0)
            r2 = best_model_metrics.get('r2', 0)
            binary_acc = best_model_metrics.get('binary_accuracy', 0)
            f1 = best_model_metrics.get('f1_score', 0)
            
            f.write(f"{feature}+{mil}: AUROC={auroc:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, ")
            f.write(f"R²={r2:.4f}, BinAcc={binary_acc:.4f}, F1={f1:.4f}\n")
        else:
            f.write(f"{feature}+{mil}: No metrics available\n")
    
    return comparison_path