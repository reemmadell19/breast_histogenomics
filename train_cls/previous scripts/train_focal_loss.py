# Baseline Model Training with Focal Loss 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

# project imports
from datasets.mil_dataset import MILDataset
from models.baseline_model import MeanPoolingMIL
from utils.mil_utils import mil_collate_fn
from utils.focal_loss import FocalLoss
from utils.evaluation_metrics import MILEvaluator

print("Loaded MILDataset from:", MILDataset.__module__)

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Config
train_csv   = "data/manifests/train_features.csv"
val_csv     = "data/manifests/val_features.csv"
batch_size  = 1   # one slide (bag) at a time
input_dim   = 512
num_classes = 2
lr          = 1e-4
num_epochs  = 10
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Focal Loss hyperparameters
focal_alpha = 1.0  # Can tune this
focal_gamma = 2.0  # Can tune this (higher = more focus on hard examples)

# Create results directory (will overwrite previous runs)
results_dir = "results/focal_loss"
os.makedirs(results_dir, exist_ok=True)
print(f"📁 Results will be saved to: {results_dir}")

# Initialize Weights & Biases
wandb.init(project="rs-baseline-mil", name="enhanced-focal-loss-baseline")
wandb.config.update({
    "model": "MeanPoolingMIL",
    "loss": "FocalLoss",
    "focal_alpha": focal_alpha,
    "focal_gamma": focal_gamma,
    "batch_size": batch_size,
    "lr": lr,
    "epochs": num_epochs,
    "input_dim": input_dim,
    "num_classes": num_classes
})

# Datasets & Loaders
train_dataset = MILDataset(train_csv)
val_dataset   = MILDataset(val_csv)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=mil_collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=mil_collate_fn
)

# Model, Loss, Optimizer
model = MeanPoolingMIL(input_dim=input_dim, num_classes=num_classes).to(device)
criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
optimizer = optim.Adam(model.parameters(), lr=lr)

print(f"🎯 Using Focal Loss with alpha={focal_alpha}, gamma={focal_gamma}")

# Initialize evaluators
train_evaluator = MILEvaluator()
val_evaluator = MILEvaluator()

# Metrics storage for plotting
all_metrics = []
best_val_auc = 0.0

# Sanity check
print("Sanity check on train_loader item:")
for feats, lbl in train_loader:
    print("  features:", type(feats), feats.shape if hasattr(feats, 'shape') else f"list of {len(feats)} items")
    print("  label   :", type(lbl), lbl)
    break

print(f"\nDataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")

# ── Training & Validation Loop ───────────────────────────────────────────────
for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    
    # Reset evaluators for this epoch
    train_evaluator.reset()
    val_evaluator.reset()
    
    # ---- TRAINING PHASE ----
    model.train()
    train_loss_total = 0.0
    
    for features, labels in tqdm(train_loader, desc="Training"):
        # Handle single bag per batch (batch_size=1)
        if isinstance(features, list):
            features = features[0] if len(features) == 1 else features[0]
        if isinstance(labels, list):
            labels = labels[0] if len(labels) == 1 else labels[0]
        
        # Move to device
        features = features.to(device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor([labels], dtype=torch.long, device=device)
        else:
            labels = labels.to(device)
            if labels.dim() == 0:  # scalar tensor
                labels = labels.unsqueeze(0)
        
        optimizer.zero_grad()
        outputs = model(features)  # outputs: [num_classes]
        loss = criterion(outputs.unsqueeze(0), labels)  # [1, num_classes], [1]
        loss.backward()
        optimizer.step()
        
        # Extract predictions for metrics
        probs = torch.softmax(outputs, dim=0)
        pred_class = probs.argmax().item()
        prob_positive = probs[1].item()
        label_int = labels.item()
        
        # Update train evaluator
        train_evaluator.update(
            labels=[label_int],
            preds=[pred_class], 
            probs=[prob_positive],
            losses=[loss.item()]
        )
        
        train_loss_total += loss.item()

    # Compute training metrics
    train_metrics = train_evaluator.compute_all_metrics(verbose=False)
    print(f"📊 TRAIN - Loss: {train_loss_total:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
          f"AUC: {train_metrics['auc_roc']:.4f} | F1: {train_metrics['f1_score']:.4f}")

    # ---- VALIDATION PHASE ----
    model.eval()
    val_loss_total = 0.0
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Validating"):
            # Handle single bag per batch (batch_size=1)
            if isinstance(features, list):
                features = features[0] if len(features) == 1 else features[0]
            if isinstance(labels, list):
                labels = labels[0] if len(labels) == 1 else labels[0]
            
            # Move to device
            features = features.to(device)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor([labels], dtype=torch.long, device=device)
            else:
                labels = labels.to(device)
                if labels.dim() == 0:  # scalar tensor
                    labels = labels.unsqueeze(0)
            
            outputs = model(features)
            loss = criterion(outputs.unsqueeze(0), labels)
            
            # Extract predictions for metrics
            probs = torch.softmax(outputs, dim=0)
            pred_class = probs.argmax().item()
            prob_positive = probs[1].item()
            label_int = labels.item()
            
            # Update validation evaluator
            val_evaluator.update(
                labels=[label_int],
                preds=[pred_class],
                probs=[prob_positive], 
                losses=[loss.item()]
            )
            
            val_loss_total += loss.item()

    # Compute validation metrics  
    val_metrics = val_evaluator.compute_all_metrics(verbose=True)
    
    print(f"📊 VAL   - Loss: {val_loss_total:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
          f"AUC: {val_metrics['auc_roc']:.4f} | F1: {val_metrics['f1_score']:.4f}")
    
    # Print confusion matrix
    print("Validation Confusion Matrix:")
    print(val_evaluator.get_confusion_matrix())
    
    # Store all metrics for this epoch (ensure consistent naming)
    epoch_metrics = {
        "epoch": epoch,
        "train_loss": train_loss_total,
        "val_loss": val_loss_total,
        # Training metrics
        "train_accuracy": train_metrics['accuracy'],
        "train_auc_roc": train_metrics['auc_roc'],
        "train_f1": train_metrics['f1_score'],
        "train_precision": train_metrics['precision'],
        "train_recall": train_metrics['recall'],
        "train_sensitivity": train_metrics['sensitivity'],
        "train_specificity": train_metrics['specificity'],
        "train_auc_pr": train_metrics['auc_pr'],
        "train_mcc": train_metrics['mcc'],
        "train_balanced_accuracy": train_metrics['balanced_accuracy'],
        # Validation metrics
        "val_accuracy": val_metrics['accuracy'],
        "val_auc_roc": val_metrics['auc_roc'],
        "val_f1": val_metrics['f1_score'],
        "val_precision": val_metrics['precision'],
        "val_recall": val_metrics['recall'],
        "val_sensitivity": val_metrics['sensitivity'],
        "val_specificity": val_metrics['specificity'],
        "val_ppv": val_metrics['ppv'],
        "val_npv": val_metrics['npv'],
        "val_auc_pr": val_metrics['auc_pr'],
        "val_mcc": val_metrics['mcc'],
        "val_balanced_accuracy": val_metrics['balanced_accuracy'],
    }
    all_metrics.append(epoch_metrics)
    
    # Log to WandB with comprehensive metrics
    wandb_log = {
        "epoch": epoch,
        "train_loss": train_loss_total,
        "val_loss": val_loss_total,
        # Core metrics
        "train_accuracy": train_metrics['accuracy'],
        "val_accuracy": val_metrics['accuracy'],
        "train_auc_roc": train_metrics['auc_roc'],
        "val_auc_roc": val_metrics['auc_roc'],
        "train_f1": train_metrics['f1_score'],
        "val_f1": val_metrics['f1_score'],
        # Clinical metrics
        "val_sensitivity": val_metrics['sensitivity'],
        "val_specificity": val_metrics['specificity'],
        "val_ppv": val_metrics['ppv'],
        "val_npv": val_metrics['npv'],
        # Advanced metrics
        "val_auc_pr": val_metrics['auc_pr'],
        "val_mcc": val_metrics['mcc'],
        "val_balanced_accuracy": val_metrics['balanced_accuracy'],
        # Focal loss specific
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "loss_type": "FocalLoss"
    }
    wandb.log(wandb_log)
    
    # Save best model based on AUC-ROC (more robust than accuracy for imbalanced data)
    if val_metrics['auc_roc'] > best_val_auc:
        best_val_auc = val_metrics['auc_roc']
        save_path = os.path.join(results_dir, "best_focal_loss_model.pt")
        torch.save(model.state_dict(), save_path)
        print(f"🎯 NEW BEST MODEL! AUC: {best_val_auc:.4f} - Saved to {save_path}")
        
        # Save best model's confusion matrix
        val_evaluator.plot_confusion_matrix(
            save_path=os.path.join(results_dir, "best_model_confusion_matrix.png"),
            title=f"Best Focal Loss Model Confusion Matrix (AUC: {best_val_auc:.4f})"
        )

# ── Post-Training Analysis & Visualization ─────────────────────────────────
print("\n" + "="*60)
print("FOCAL LOSS TRAINING COMPLETE - GENERATING REPORTS")
print("="*60)

# Save comprehensive metrics CSV
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(os.path.join(results_dir, "comprehensive_metrics.csv"), index=False)

# Generate plots with error handling
try:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Focal Loss Training Progress - Comprehensive Metrics", fontsize=16)

    # Loss curves
    axes[0,0].plot(metrics_df['epoch'], metrics_df['train_loss'], 'b-o', label='Train')
    axes[0,0].plot(metrics_df['epoch'], metrics_df['val_loss'], 'r-o', label='Validation')
    axes[0,0].set_title('Loss Curves (Focal Loss)')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)

    # Accuracy curves
    axes[0,1].plot(metrics_df['epoch'], metrics_df['train_accuracy'], 'b-o', label='Train')
    axes[0,1].plot(metrics_df['epoch'], metrics_df['val_accuracy'], 'r-o', label='Validation')
    axes[0,1].set_title('Accuracy Curves')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True)

    # AUC curves
    axes[0,2].plot(metrics_df['epoch'], metrics_df['train_auc_roc'], 'b-o', label='Train AUC-ROC')
    axes[0,2].plot(metrics_df['epoch'], metrics_df['val_auc_roc'], 'r-o', label='Val AUC-ROC')
    axes[0,2].plot(metrics_df['epoch'], metrics_df['val_auc_pr'], 'g-o', label='Val AUC-PR')
    axes[0,2].set_title('AUC Curves')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('AUC')
    axes[0,2].legend()
    axes[0,2].grid(True)

    # F1 Score
    axes[1,0].plot(metrics_df['epoch'], metrics_df['train_f1'], 'b-o', label='Train')
    axes[1,0].plot(metrics_df['epoch'], metrics_df['val_f1'], 'r-o', label='Validation')
    axes[1,0].set_title('F1 Score')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('F1 Score')
    axes[1,0].legend()
    axes[1,0].grid(True)

    # Clinical Metrics
    axes[1,1].plot(metrics_df['epoch'], metrics_df['val_sensitivity'], 'r-o', label='Sensitivity')
    axes[1,1].plot(metrics_df['epoch'], metrics_df['val_specificity'], 'b-o', label='Specificity')
    axes[1,1].plot(metrics_df['epoch'], metrics_df['val_ppv'], 'g-o', label='PPV')
    axes[1,1].plot(metrics_df['epoch'], metrics_df['val_npv'], 'm-o', label='NPV')
    axes[1,1].set_title('Clinical Metrics')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Score')
    axes[1,1].legend()
    axes[1,1].grid(True)

    # Advanced Metrics
    axes[1,2].plot(metrics_df['epoch'], metrics_df['val_mcc'], 'r-o', label='Matthews Corr')
    axes[1,2].plot(metrics_df['epoch'], metrics_df['val_balanced_accuracy'], 'b-o', label='Balanced Acc')
    axes[1,2].set_title('Advanced Metrics')
    axes[1,2].set_xlabel('Epoch')
    axes[1,2].set_ylabel('Score')
    axes[1,2].legend()
    axes[1,2].grid(True)

    plt.tight_layout()
    comprehensive_plots_path = os.path.join(results_dir, "comprehensive_training_curves.png")
    plt.savefig(comprehensive_plots_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Comprehensive training curves saved to: {comprehensive_plots_path}")
    
except Exception as e:
    print(f"❌ Error creating comprehensive plots: {e}")
    print(f"Available columns in metrics_df: {list(metrics_df.columns)}")
    
    # Create simple plots as fallback
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curves
        axes[0].plot(metrics_df['epoch'], metrics_df['train_loss'], 'b-o', label='Train')
        axes[0].plot(metrics_df['epoch'], metrics_df['val_loss'], 'r-o', label='Validation')
        axes[0].set_title('Focal Loss Curves')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy curves
        axes[1].plot(metrics_df['epoch'], metrics_df['train_accuracy'], 'b-o', label='Train')
        axes[1].plot(metrics_df['epoch'], metrics_df['val_accuracy'], 'r-o', label='Validation')
        axes[1].set_title('Accuracy Curves')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        fallback_plots_path = os.path.join(results_dir, "basic_training_curves.png")
        plt.savefig(fallback_plots_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Basic training curves saved to: {fallback_plots_path}")
        
    except Exception as e2:
        print(f"❌ Error creating fallback plots: {e2}")
        print("Skipping plot generation...")

# Final model performance summary
print(f"\n📋 FINAL FOCAL LOSS PERFORMANCE SUMMARY:")
print(f"Best Validation AUC-ROC: {best_val_auc:.4f}")
final_val_metrics = metrics_df.iloc[-1]
print(f"Final Validation Metrics:")
print(f"  Accuracy: {final_val_metrics['val_accuracy']:.4f}")
print(f"  F1-Score: {final_val_metrics['val_f1']:.4f}")
print(f"  Sensitivity: {final_val_metrics['val_sensitivity']:.4f}")
print(f"  Specificity: {final_val_metrics['val_specificity']:.4f}")
print(f"  AUC-PR: {final_val_metrics['val_auc_pr']:.4f}")
print(f"  Matthews Correlation: {final_val_metrics['val_mcc']:.4f}")

print(f"\n✅ All results saved to: {results_dir}/")
print(f"   - comprehensive_metrics.csv")
print(f"   - comprehensive_training_curves.png (or basic_training_curves.png)") 
print(f"   - best_model_confusion_matrix.png")
print(f"   - best_focal_loss_model.pt")

# Also create a summary report
summary_report = f"""
FOCAL LOSS EXPERIMENT SUMMARY
{'='*50}
Model: MeanPoolingMIL with Focal Loss
Loss Function: Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})
Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}
Epochs: {num_epochs}
Learning Rate: {lr}

FOCAL LOSS PARAMETERS:
- Alpha: {focal_alpha} (class weighting)
- Gamma: {focal_gamma} (focusing parameter)
- Purpose: Focus learning on hard examples and handle class imbalance

BEST PERFORMANCE:
- Best Validation AUC-ROC: {best_val_auc:.4f}

FINAL METRICS:
- Accuracy: {final_val_metrics['val_accuracy']:.4f}
- F1-Score: {final_val_metrics['val_f1']:.4f}
- Sensitivity: {final_val_metrics['val_sensitivity']:.4f}
- Specificity: {final_val_metrics['val_specificity']:.4f}
- AUC-PR: {final_val_metrics['val_auc_pr']:.4f}
- Matthews Correlation: {final_val_metrics['val_mcc']:.4f}

FOCAL LOSS THEORY:
Focal Loss modifies cross-entropy by:
1. Down-weighting easy examples (high confidence predictions)
2. Focusing on hard examples (low confidence predictions)
3. Gamma controls focusing strength (higher = more focus on hard examples)
4. Alpha provides class balancing (can handle imbalanced datasets)

FILES SAVED:
- Model: best_focal_loss_model.pt
- Metrics: comprehensive_metrics.csv
- Plots: comprehensive_training_curves.png, best_model_confusion_matrix.png
"""

summary_path = os.path.join(results_dir, "experiment_summary.txt")
with open(summary_path, 'w') as f:
    f.write(summary_report)

print(f"   - experiment_summary.txt")
print(f"\n🎯 Focal Loss Summary report saved to: {summary_path}")

# Close wandb
wandb.finish()