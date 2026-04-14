# File: train_enhanced_cleaned.py
# Enhanced Baseline Model Training (No Batch Balancing) - Cleaned Version

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import torch

# Project imports
from datasets.mil_dataset import MILDataset
from models.baseline_model import MeanPoolingMIL
from utils.mil_utils import mil_collate_fn
from utils.evaluation_metrics import MILEvaluator
from utils.training_helpers import (
    set_seed, setup_directories, initialize_wandb, create_dataloaders,
    setup_model_and_optimizer, train_one_epoch, validate_one_epoch,
    print_epoch_metrics, save_best_model, create_metrics_dict,
    log_to_wandb, create_comprehensive_plots, generate_baseline_summary,
    print_final_performance, perform_sanity_check, print_experiment_header
)

# Configuration
CONFIG = {
    "model_name": "enhanced",  # Enhanced baseline (no balancing)
    "experiment_type": "baseline",
    "input_dim": 512,  # Standard feature dimension
    "batch_size": 1,
    "num_classes": 2,
    "lr": 1e-4,
    "num_epochs": 10,
    "train_csv": "data/manifests/train_features.csv",
    "val_csv": "data/manifests/val_features.csv"
}

def generate_enhanced_summary(model_name: str, experiment_type: str, input_dim: int,
                             train_dataset_size: int, val_dataset_size: int,
                             num_epochs: int, lr: float, best_val_auc: float,
                             final_val_metrics: dict, results_dir: str) -> str:
    """Generate summary for enhanced baseline experiments."""
    
    summary_report = f"""
{model_name.upper()} {experiment_type.upper()} EXPERIMENT SUMMARY
{'='*50}
Model: MeanPoolingMIL (Enhanced Baseline)
Feature Dimension: {input_dim}
Sampling Strategy: Standard (No Class Balancing)
Dataset: Train={train_dataset_size}, Val={val_dataset_size}
Epochs: {num_epochs}
Learning Rate: {lr}

BEST PERFORMANCE:
- Best Validation AUC-ROC: {best_val_auc:.4f}

FINAL METRICS:
- Accuracy: {final_val_metrics['val_accuracy']:.4f}
- F1-Score: {final_val_metrics['val_f1']:.4f}
- Sensitivity: {final_val_metrics['val_sensitivity']:.4f}
- Specificity: {final_val_metrics['val_specificity']:.4f}
- AUC-PR: {final_val_metrics['val_auc_pr']:.4f}
- Matthews Correlation: {final_val_metrics['val_mcc']:.4f}

MODEL CONFIGURATION:
- Input dimension: {input_dim}
- Architecture: Mean Pooling → Linear({input_dim}→256) → ReLU → Linear(256→2)
- Loss function: CrossEntropy
- Optimizer: Adam
- Sampling: Standard (no class balancing)

FILES SAVED:
- Model: best_{model_name}_{experiment_type}_model.pt
- Metrics: comprehensive_metrics.csv
- Plots: comprehensive_training_curves.png, best_model_confusion_matrix.png
"""
    
    summary_path = os.path.join(results_dir, "experiment_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary_report)
    
    return summary_path

def main():
    # Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = setup_directories(CONFIG["model_name"], CONFIG["experiment_type"])
    
    # Initialize W&B
    wandb_config = {
        "model": "MeanPoolingMIL",
        "backbone": "Enhanced-Baseline",
        "loss": "CrossEntropy",
        "sampling": "Standard (No Balancing)",
        **{k: v for k, v in CONFIG.items() if k not in ["train_csv", "val_csv"]}
    }
    initialize_wandb("rs-baseline-mil", "enhanced-mean-pooling-baseline", wandb_config)
    
    # Create datasets and dataloaders (no weighted sampling)
    train_dataset = MILDataset(CONFIG["train_csv"])
    val_dataset = MILDataset(CONFIG["val_csv"])
    
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, CONFIG["batch_size"], mil_collate_fn,
        use_weighted_sampling=False  # Enhanced baseline, no balancing
    )
    
    # Setup model, criterion, optimizer
    model_params = {"input_dim": CONFIG["input_dim"], "num_classes": CONFIG["num_classes"]}
    model, criterion, optimizer = setup_model_and_optimizer(
        MeanPoolingMIL, model_params, CONFIG["lr"], device
    )
    
    # Initialize evaluators
    train_evaluator = MILEvaluator()
    val_evaluator = MILEvaluator()
    
    # Print experiment info
    print_experiment_header(
        CONFIG["model_name"], CONFIG["experiment_type"], CONFIG["input_dim"],
        device, CONFIG["lr"], CONFIG["num_epochs"]
    )
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    perform_sanity_check(train_loader, CONFIG["model_name"])
    
    # Training loop
    all_metrics = []
    best_val_auc = 0.0
    
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        # Reset evaluators
        train_evaluator.reset()
        val_evaluator.reset()
        
        # Training phase
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, train_evaluator
        )
        train_metrics = train_evaluator.compute_all_metrics(verbose=False)
        
        # Validation phase
        val_loss = validate_one_epoch(
            model, val_loader, criterion, device, val_evaluator
        )
        val_metrics = val_evaluator.compute_all_metrics(verbose=True)
        
        # Print metrics
        print_epoch_metrics(
            epoch, CONFIG["num_epochs"], train_loss, val_loss,
            train_metrics, val_metrics, CONFIG["model_name"]
        )
        
        print("Validation Confusion Matrix:")
        print(val_evaluator.get_confusion_matrix())
        
        # Save best model
        best_val_auc = save_best_model(
            model, val_metrics, best_val_auc, results_dir,
            CONFIG["model_name"], CONFIG["experiment_type"], val_evaluator
        )
        
        # Store metrics
        epoch_metrics = create_metrics_dict(
            epoch, train_loss, val_loss, train_metrics, val_metrics
        )
        all_metrics.append(epoch_metrics)
        
        # Log to W&B
        log_to_wandb(
            epoch, train_loss, val_loss, train_metrics, val_metrics,
            "Standard (No Balancing)"
        )
    
    # Post-training analysis
    print(f"\n{'='*60}")
    print(f"{CONFIG['model_name'].upper()} {CONFIG['experiment_type'].upper()} TRAINING COMPLETE - GENERATING REPORTS")
    print(f"{'='*60}")
    
    # Save metrics and create plots
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(results_dir, "comprehensive_metrics.csv"), index=False)
    create_comprehensive_plots(metrics_df, results_dir, CONFIG["model_name"])
    
    # Generate final summary using enhanced summary function
    final_val_metrics = metrics_df.iloc[-1]
    summary_path = generate_enhanced_summary(
        CONFIG["model_name"], CONFIG["experiment_type"], CONFIG["input_dim"],
        len(train_dataset), len(val_dataset), CONFIG["num_epochs"], 
        CONFIG["lr"], best_val_auc, final_val_metrics, results_dir
    )
    
    print_final_performance(
        CONFIG["model_name"], CONFIG["experiment_type"], best_val_auc,
        final_val_metrics, results_dir
    )
    
    print(f"🎯 {CONFIG['model_name'].upper()} {CONFIG['experiment_type'].upper()} Summary report saved to: {summary_path}")
    
    # Close W&B
    import wandb
    wandb.finish()
    
    print(f"\n{'='*60}")
    print(f"{CONFIG['model_name'].upper()} {CONFIG['experiment_type'].upper()} TRAINING COMPLETED")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()