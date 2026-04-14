# File: train_virchow2_bbb_cleaned.py
# Virchow-2 Baseline Model Training with Batch Balancing (BBB) - Fixed Version

# DISABLE WANDB FIRST - BEFORE ANY OTHER IMPORTS
import os
os.environ["WANDB_DISABLED"] = "true"

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
    set_seed, setup_directories, create_dataloaders,
    setup_model_and_optimizer, train_one_epoch, validate_one_epoch,
    print_epoch_metrics, save_best_model, create_metrics_dict,
    create_comprehensive_plots, generate_final_summary,
    print_final_performance, perform_sanity_check, print_experiment_header
)

# Configuration
CONFIG = {
    "model_name": "virchow2",
    "experiment_type": "bbb",
    "input_dim": 1280,  # Virchow-2 feature dimension
    "batch_size": 1,
    "num_classes": 2,
    "lr": 1e-4,
    "num_epochs": 10,
    "class_weights": {0: 0.19, 1: 0.81},
    "counts": {0: 0.811, 1: 0.189},
    "train_csv": "data/manifests/train_features_virchow2.csv",
    "val_csv": "data/manifests/val_features_virchow2.csv"
}

def main():
    # Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = setup_directories(CONFIG["model_name"], CONFIG["experiment_type"])
    
    # SKIP W&B INITIALIZATION - Comment out or remove these lines
    # wandb_config = {
    #     "model": "MeanPoolingMIL",
    #     "backbone": CONFIG["model_name"].upper(),
    #     "loss": "CrossEntropy",
    #     "sampling": "WeightedRandomSampler",
    #     **{k: v for k, v in CONFIG.items() if k not in ["train_csv", "val_csv"]}
    # }
    # initialize_wandb("rs-baseline-mil", f"{CONFIG['model_name']}-mean-pooling-bbb-baseline", wandb_config)
    
    # Create datasets and dataloaders
    train_dataset = MILDataset(CONFIG["train_csv"])
    val_dataset = MILDataset(CONFIG["val_csv"])
    
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, CONFIG["batch_size"], mil_collate_fn,
        use_weighted_sampling=True, class_weights=CONFIG["class_weights"]
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
    
    print(f"📊 Class distribution: {CONFIG['counts']}")
    print(f"🎯 Class weights for sampling: {CONFIG['class_weights']}")
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
        
        # SKIP W&B LOGGING - Comment out or remove this line
        # log_to_wandb(
        #     epoch, train_loss, val_loss, train_metrics, val_metrics,
        #     "WeightedRandomSampler"
        # )
    
    # Post-training analysis
    print(f"\n{'='*60}")
    print(f"{CONFIG['model_name'].upper()} {CONFIG['experiment_type'].upper()} TRAINING COMPLETE - GENERATING REPORTS")
    print(f"{'='*60}")
    
    # Save metrics and create plots
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(results_dir, "comprehensive_metrics.csv"), index=False)
    create_comprehensive_plots(metrics_df, results_dir, CONFIG["model_name"])
    
    # Generate final summary with comparison
    final_val_metrics = metrics_df.iloc[-1]
    
    # Enhanced summary for Virchow-2 with comparison to H-Optimus-1
    summary_content = f"""
{CONFIG['model_name'].upper()} {CONFIG['experiment_type'].upper()} EXPERIMENT SUMMARY
{'='*50}
Model: MeanPoolingMIL with WeightedRandomSampler
Feature Extractor: {CONFIG['model_name'].upper()} ({CONFIG['input_dim']}-dim features)
Sampling Strategy: Batch Balancing (BBB)
Class Weights: {CONFIG['class_weights']}
Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}
Epochs: {CONFIG['num_epochs']}
Learning Rate: {CONFIG['lr']}

BEST PERFORMANCE:
- Best Validation AUC-ROC: {best_val_auc:.4f}

FINAL METRICS:
- Accuracy: {final_val_metrics['val_accuracy']:.4f}
- F1-Score: {final_val_metrics['val_f1']:.4f}
- Sensitivity: {final_val_metrics['val_sensitivity']:.4f}
- Specificity: {final_val_metrics['val_specificity']:.4f}
- AUC-PR: {final_val_metrics['val_auc_pr']:.4f}
- Matthews Correlation: {final_val_metrics['val_mcc']:.4f}

COMPARISON WITH H-OPTIMUS-1:
- H-Optimus-1 AUC-ROC: 0.8855
- Virchow-2 AUC-ROC: {best_val_auc:.4f}
- Improvement: {best_val_auc - 0.8855:+.4f} ({(best_val_auc - 0.8855)*100:+.2f}%)

VIRCHOW-2 ADVANTAGES:
- Clinical-grade foundation model by Paige AI
- Trained on 1.5M+ whole slide images (3B+ patches)
- Feature dimension: {CONFIG['input_dim']} (ViT-Large architecture)
- Pan-cancer training across 30+ cancer types
- Extensive clinical validation studies
- Real-world clinical data training
- WeightedRandomSampler addresses class imbalance

TECHNICAL DETAILS:
- Architecture: Vision Transformer (ViT-Large)
- Feature dimension: {CONFIG['input_dim']}
- Training data: 1.5M+ WSIs, 3B+ patches
- Developer: Paige AI
- Clinical focus: Real-world pathology workflow

FILES SAVED:
- Model: best_{CONFIG['model_name']}_{CONFIG['experiment_type']}_model.pt
- Metrics: comprehensive_metrics.csv
- Plots: comprehensive_training_curves.png, best_model_confusion_matrix.png
"""
    
    summary_path = os.path.join(results_dir, "experiment_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary_content)
    
    print_final_performance(
        CONFIG["model_name"], CONFIG["experiment_type"], best_val_auc,
        final_val_metrics, results_dir
    )
    
    # Virchow-2 specific comparison output
    h_optimus_auc = 0.8855
    improvement = best_val_auc - h_optimus_auc
    
    print(f"\n🔍 VIRCHOW-2 vs H-OPTIMUS-1 COMPARISON:")
    print(f"H-Optimus-1 AUC-ROC: {h_optimus_auc:.4f}")
    print(f"Virchow-2 AUC-ROC: {best_val_auc:.4f}")
    print(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    if improvement > 0.01:
        print(f"✅ SIGNIFICANT IMPROVEMENT! Virchow-2 outperforms H-Optimus-1!")
    elif improvement > 0:
        print(f"📈 Modest improvement with Virchow-2")
    else:
        print(f"📊 H-Optimus-1 performs similarly or better")
    
    print(f"\n🎯 {CONFIG['model_name'].upper()} Summary report saved to: {summary_path}")
    
    # SKIP W&B FINISH - Comment out or remove this
    # import wandb
    # wandb.finish()
    
    print(f"\n{'='*60}")
    print(f"VIRCHOW-2 {CONFIG['experiment_type'].upper()} BASELINE TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"🎉 You now have VIRCHOW-2 results to compare!")
    print(f"📊 Expected: Clinical-grade performance from Paige AI model")
    print(f"🔬 Virchow-2 trained on real clinical data (1.5M+ WSIs)")
    print(f"🏥 Extensive clinical validation studies published")
    print(f"🔍 Compare with H-Optimus-1 (AUC: 0.8855) and UNI-2H (AUC: 0.8521)")

if __name__ == "__main__":
    main()