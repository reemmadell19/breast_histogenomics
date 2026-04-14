# File: find_optimal_class_weights.py
# Find optimal class weight ratios for batch balancing using grid search

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
from itertools import product

# project imports
from datasets.mil_dataset import MILDataset
from models.baseline_model import MeanPoolingMIL
from utils.mil_utils import mil_collate_fn
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
train_csv   = "data/manifests/train_features_resnet18.csv"
val_csv     = "data/manifests/val_features_resnet18.csv"
batch_size  = 1   # one slide (bag) at a time
input_dim   = 512  # Update this based on your features (512 for ResNet18, 2048 for ResNet50, etc.)
num_classes = 2
lr          = 1e-4
num_epochs  = 5    # Reduced epochs for grid search
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create results directory
results_dir = "results/class_weight_optimization"
os.makedirs(results_dir, exist_ok=True)
print(f"📁 Results will be saved to: {results_dir}")

# Class weight ratios to test
# Format: (minority_class_weight_multiplier, majority_class_weight)
# E.g., (5.0, 1.0) means minority class gets 5x weight, majority class gets 1x weight
class_weight_ratios = [
    (1.0, 1.0),    # No balancing (baseline)
    (2.0, 1.0),    # 2x minority weight
    (3.0, 1.0),    # 3x minority weight
    (4.0, 1.0),    # 4x minority weight
    (5.0, 1.0),    # 5x minority weight
    (6.0, 1.0),    # 6x minority weight
    (7.0, 1.0),    # 7x minority weight
    (8.0, 1.0),    # 8x minority weight
    (10.0, 1.0),   # 10x minority weight
    # Inverse frequency based
    (5.29, 1.23),  # Based on your class distribution (0.189 vs 0.811)
    # More fine-grained around promising areas
    (4.5, 1.0),
    (5.5, 1.0),
    (6.5, 1.0),
    (7.5, 1.0),
    (9.0, 1.0),
]

print("="*60)
print("CLASS WEIGHT OPTIMIZATION FOR BATCH BALANCING")
print("="*60)
print(f"Feature dimension: {input_dim}")
print(f"Testing {len(class_weight_ratios)} different class weight ratios")
print(f"Epochs per ratio: {num_epochs}")
print(f"Total experiments: {len(class_weight_ratios)}")
print(f"Device: {device}")
print("="*60)

# Load datasets once
train_dataset = MILDataset(train_csv)
val_dataset   = MILDataset(val_csv)

print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")

# Calculate actual class distribution
actual_labels = [label for _, label in train_dataset]
actual_counts = np.bincount(actual_labels)
actual_distribution = actual_counts / len(actual_labels)
print(f"Actual class distribution: Class 0: {actual_distribution[0]:.3f}, Class 1: {actual_distribution[1]:.3f}")

def create_weighted_dataloader(class_weights_dict, train_dataset, batch_size):
    """Create a DataLoader with WeightedRandomSampler using given class weights."""
    labels = [label for _, label in train_dataset]
    sample_weights = [class_weights_dict[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=mil_collate_fn
    )

def train_and_evaluate(class_weights_dict, train_dataset, val_dataset, experiment_id):
    """Train model with given class weights and return validation metrics."""
    print(f"\n--- Experiment {experiment_id}: Class weights {class_weights_dict} ---")
    
    # Create data loaders
    train_loader = create_weighted_dataloader(class_weights_dict, train_dataset, batch_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=mil_collate_fn
    )
    
    # Create fresh model for each experiment
    model = MeanPoolingMIL(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_auc = 0.0
    best_metrics = None
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        train_evaluator = MILEvaluator()
        
        for features, label in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            # Handle features and labels
            if isinstance(features, list):
                features = features[0] if len(features) == 1 else torch.stack(features)
            features = features.to(device)
            
            if isinstance(label, list):
                label = label[0] if len(label) == 1 else label
            if not isinstance(label, torch.Tensor):
                label = torch.tensor([label], dtype=torch.long, device=device)
            else:
                label = label.to(device)
                if label.dim() == 0:
                    label = label.unsqueeze(0)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.unsqueeze(0), label)
            loss.backward()
            optimizer.step()

            # Update evaluator
            probs = torch.softmax(outputs, dim=0)
            pred_class = probs.argmax().item()
            prob_positive = probs[1].item()
            label_int = label.item()
            
            train_evaluator.update(
                labels=[label_int],
                preds=[pred_class], 
                probs=[prob_positive],
                losses=[loss.item()]
            )
        
        # Validation phase
        model.eval()
        val_evaluator = MILEvaluator()
        
        with torch.no_grad():
            for features, label in val_loader:
                # Handle features and labels
                if isinstance(features, list):
                    features = features[0] if len(features) == 1 else torch.stack(features)
                features = features.to(device)
                
                if isinstance(label, list):
                    label = label[0] if len(label) == 1 else label
                if not isinstance(label, torch.Tensor):
                    label = torch.tensor([label], dtype=torch.long, device=device)
                else:
                    label = label.to(device)
                    if label.dim() == 0:
                        label = label.unsqueeze(0)

                outputs = model(features)
                loss = criterion(outputs.unsqueeze(0), label)

                # Update evaluator
                probs = torch.softmax(outputs, dim=0)
                pred_class = probs.argmax().item()
                prob_positive = probs[1].item()
                label_int = label.item()
                
                val_evaluator.update(
                    labels=[label_int],
                    preds=[pred_class],
                    probs=[prob_positive], 
                    losses=[loss.item()]
                )
        
        # Get metrics
        train_metrics = train_evaluator.compute_all_metrics(verbose=False)
        val_metrics = val_evaluator.compute_all_metrics(verbose=False)
        
        # Track best validation performance
        if val_metrics['auc_roc'] > best_val_auc:
            best_val_auc = val_metrics['auc_roc']
            best_metrics = {
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'class_weights': class_weights_dict
            }
        
        print(f"Epoch {epoch}: Train F1={train_metrics['f1_score']:.3f}, Val F1={val_metrics['f1_score']:.3f}, "
              f"Val AUC={val_metrics['auc_roc']:.3f}, Val Sens={val_metrics['sensitivity']:.3f}")
    
    return best_metrics

# Run grid search
all_results = []
experiment_id = 1

print(f"\nStarting grid search over {len(class_weight_ratios)} class weight configurations...")

for minority_weight, majority_weight in class_weight_ratios:
    class_weights_dict = {0: majority_weight, 1: minority_weight}
    
    try:
        result = train_and_evaluate(class_weights_dict, train_dataset, val_dataset, experiment_id)
        
        # Store results
        result_summary = {
            'experiment_id': experiment_id,
            'minority_weight': minority_weight,
            'majority_weight': majority_weight,
            'weight_ratio': minority_weight / majority_weight,
            'best_epoch': result['epoch'],
            'val_auc_roc': result['val_metrics']['auc_roc'],
            'val_f1_score': result['val_metrics']['f1_score'],
            'val_accuracy': result['val_metrics']['accuracy'],
            'val_sensitivity': result['val_metrics']['sensitivity'],
            'val_specificity': result['val_metrics']['specificity'],
            'val_precision': result['val_metrics']['precision'],
            'val_recall': result['val_metrics']['recall'],
            'val_mcc': result['val_metrics']['mcc'],
            'val_balanced_accuracy': result['val_metrics']['balanced_accuracy'],
            'val_auc_pr': result['val_metrics']['auc_pr'],
            'train_f1_score': result['train_metrics']['f1_score'],
            'train_auc_roc': result['train_metrics']['auc_roc'],
        }
        
        all_results.append(result_summary)
        
        print(f"✅ Experiment {experiment_id} completed")
        print(f"   Best Val AUC: {result['val_metrics']['auc_roc']:.4f}")
        print(f"   Best Val F1:  {result['val_metrics']['f1_score']:.4f}")
        print(f"   Best Val Sens: {result['val_metrics']['sensitivity']:.4f}")
        
    except Exception as e:
        print(f"❌ Experiment {experiment_id} failed: {e}")
        continue
    
    experiment_id += 1

# Analyze results
print("\n" + "="*60)
print("GRID SEARCH COMPLETED - ANALYZING RESULTS")
print("="*60)

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(results_dir, "class_weight_optimization_results.csv"), index=False)

# Find best configurations for different metrics
best_auc = results_df.loc[results_df['val_auc_roc'].idxmax()]
best_f1 = results_df.loc[results_df['val_f1_score'].idxmax()]
best_sensitivity = results_df.loc[results_df['val_sensitivity'].idxmax()]
best_mcc = results_df.loc[results_df['val_mcc'].idxmax()]
best_balanced_acc = results_df.loc[results_df['val_balanced_accuracy'].idxmax()]

print(f"\n🏆 BEST CONFIGURATIONS:")
print(f"Best AUC-ROC ({best_auc['val_auc_roc']:.4f}): Minority={best_auc['minority_weight']:.1f}, Majority={best_auc['majority_weight']:.1f}")
print(f"Best F1-Score ({best_f1['val_f1_score']:.4f}): Minority={best_f1['minority_weight']:.1f}, Majority={best_f1['majority_weight']:.1f}")
print(f"Best Sensitivity ({best_sensitivity['val_sensitivity']:.4f}): Minority={best_sensitivity['minority_weight']:.1f}, Majority={best_sensitivity['majority_weight']:.1f}")
print(f"Best MCC ({best_mcc['val_mcc']:.4f}): Minority={best_mcc['minority_weight']:.1f}, Majority={best_mcc['majority_weight']:.1f}")
print(f"Best Balanced Acc ({best_balanced_acc['val_balanced_accuracy']:.4f}): Minority={best_balanced_acc['minority_weight']:.1f}, Majority={best_balanced_acc['majority_weight']:.1f}")

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Class Weight Optimization Results', fontsize=16)

# AUC vs Weight Ratio
axes[0,0].scatter(results_df['weight_ratio'], results_df['val_auc_roc'], alpha=0.7, s=50)
axes[0,0].set_xlabel('Weight Ratio (Minority/Majority)')
axes[0,0].set_ylabel('Validation AUC-ROC')
axes[0,0].set_title('AUC-ROC vs Weight Ratio')
axes[0,0].grid(True, alpha=0.3)

# F1 Score vs Weight Ratio
axes[0,1].scatter(results_df['weight_ratio'], results_df['val_f1_score'], alpha=0.7, s=50, color='orange')
axes[0,1].set_xlabel('Weight Ratio (Minority/Majority)')
axes[0,1].set_ylabel('Validation F1-Score')
axes[0,1].set_title('F1-Score vs Weight Ratio')
axes[0,1].grid(True, alpha=0.3)

# Sensitivity vs Weight Ratio
axes[0,2].scatter(results_df['weight_ratio'], results_df['val_sensitivity'], alpha=0.7, s=50, color='green')
axes[0,2].set_xlabel('Weight Ratio (Minority/Majority)')
axes[0,2].set_ylabel('Validation Sensitivity')
axes[0,2].set_title('Sensitivity vs Weight Ratio')
axes[0,2].grid(True, alpha=0.3)

# MCC vs Weight Ratio
axes[1,0].scatter(results_df['weight_ratio'], results_df['val_mcc'], alpha=0.7, s=50, color='red')
axes[1,0].set_xlabel('Weight Ratio (Minority/Majority)')
axes[1,0].set_ylabel('Validation MCC')
axes[1,0].set_title('Matthews Correlation vs Weight Ratio')
axes[1,0].grid(True, alpha=0.3)

# Balanced Accuracy vs Weight Ratio
axes[1,1].scatter(results_df['weight_ratio'], results_df['val_balanced_accuracy'], alpha=0.7, s=50, color='purple')
axes[1,1].set_xlabel('Weight Ratio (Minority/Majority)')
axes[1,1].set_ylabel('Validation Balanced Accuracy')
axes[1,1].set_title('Balanced Accuracy vs Weight Ratio')
axes[1,1].grid(True, alpha=0.3)

# Multi-metric comparison (top 5 by AUC)
top_5 = results_df.nlargest(5, 'val_auc_roc')
metrics_to_plot = ['val_auc_roc', 'val_f1_score', 'val_sensitivity', 'val_specificity', 'val_mcc']
x_pos = np.arange(len(top_5))

for i, metric in enumerate(metrics_to_plot):
    axes[1,2].plot(x_pos, top_5[metric], 'o-', label=metric.replace('val_', '').upper(), linewidth=2, markersize=6)

axes[1,2].set_xlabel('Top 5 Configurations (by AUC)')
axes[1,2].set_ylabel('Metric Value')
axes[1,2].set_title('Multi-Metric Comparison (Top 5)')
axes[1,2].set_xticks(x_pos)
axes[1,2].set_xticklabels([f"{row['weight_ratio']:.1f}" for _, row in top_5.iterrows()], rotation=45)
axes[1,2].legend()
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "class_weight_optimization_plots.png"), dpi=300, bbox_inches='tight')
plt.close()

# Create detailed results table
print(f"\n📊 TOP 10 CONFIGURATIONS BY AUC-ROC:")
top_10 = results_df.nlargest(10, 'val_auc_roc')[['weight_ratio', 'val_auc_roc', 'val_f1_score', 'val_sensitivity', 'val_specificity', 'val_mcc']]
print(top_10.to_string(index=False, float_format='%.4f'))

# Save recommendations
recommendations = {
    'best_overall': {
        'minority_weight': float(best_auc['minority_weight']),
        'majority_weight': float(best_auc['majority_weight']),
        'weight_ratio': float(best_auc['weight_ratio']),
        'metrics': {
            'val_auc_roc': float(best_auc['val_auc_roc']),
            'val_f1_score': float(best_auc['val_f1_score']),
            'val_sensitivity': float(best_auc['val_sensitivity']),
            'val_specificity': float(best_auc['val_specificity']),
            'val_mcc': float(best_auc['val_mcc'])
        }
    },
    'best_for_f1': {
        'minority_weight': float(best_f1['minority_weight']),
        'majority_weight': float(best_f1['majority_weight']),
        'weight_ratio': float(best_f1['weight_ratio']),
        'metrics': {
            'val_auc_roc': float(best_f1['val_auc_roc']),
            'val_f1_score': float(best_f1['val_f1_score']),
            'val_sensitivity': float(best_f1['val_sensitivity']),
            'val_specificity': float(best_f1['val_specificity']),
            'val_mcc': float(best_f1['val_mcc'])
        }
    },
    'best_for_sensitivity': {
        'minority_weight': float(best_sensitivity['minority_weight']),
        'majority_weight': float(best_sensitivity['majority_weight']),
        'weight_ratio': float(best_sensitivity['weight_ratio']),
        'metrics': {
            'val_auc_roc': float(best_sensitivity['val_auc_roc']),
            'val_f1_score': float(best_sensitivity['val_f1_score']),
            'val_sensitivity': float(best_sensitivity['val_sensitivity']),
            'val_specificity': float(best_sensitivity['val_specificity']),
            'val_mcc': float(best_sensitivity['val_mcc'])
        }
    }
}

with open(os.path.join(results_dir, "recommended_class_weights.json"), 'w') as f:
    json.dump(recommendations, f, indent=2)

# Create summary report
summary_report = f"""
CLASS WEIGHT OPTIMIZATION SUMMARY
{'='*50}

EXPERIMENT CONFIGURATION:
- Total configurations tested: {len(class_weight_ratios)}
- Epochs per configuration: {num_epochs}
- Feature dimension: {input_dim}
- Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}
- Actual class distribution: {actual_distribution[0]:.3f} / {actual_distribution[1]:.3f}

BEST CONFIGURATIONS:
1. OVERALL BEST (AUC-ROC = {best_auc['val_auc_roc']:.4f}):
   - Class weights: {{0: {best_auc['majority_weight']:.1f}, 1: {best_auc['minority_weight']:.1f}}}
   - Weight ratio: {best_auc['weight_ratio']:.1f}
   - F1-Score: {best_auc['val_f1_score']:.4f}
   - Sensitivity: {best_auc['val_sensitivity']:.4f}
   - Specificity: {best_auc['val_specificity']:.4f}
   - MCC: {best_auc['val_mcc']:.4f}

2. BEST FOR F1-SCORE ({best_f1['val_f1_score']:.4f}):
   - Class weights: {{0: {best_f1['majority_weight']:.1f}, 1: {best_f1['minority_weight']:.1f}}}
   - Weight ratio: {best_f1['weight_ratio']:.1f}

3. BEST FOR SENSITIVITY ({best_sensitivity['val_sensitivity']:.4f}):
   - Class weights: {{0: {best_sensitivity['majority_weight']:.1f}, 1: {best_sensitivity['minority_weight']:.1f}}}
   - Weight ratio: {best_sensitivity['weight_ratio']:.1f}

RECOMMENDED FOR FUTURE EXPERIMENTS:
Use class weights: {{0: {best_auc['majority_weight']:.1f}, 1: {best_auc['minority_weight']:.1f}}}

FILES SAVED:
- class_weight_optimization_results.csv: Detailed results
- class_weight_optimization_plots.png: Visualization
- recommended_class_weights.json: Best configurations
"""

summary_path = os.path.join(results_dir, "optimization_summary.txt")
with open(summary_path, 'w') as f:
    f.write(summary_report)

print(f"\n✅ All results saved to: {results_dir}/")
print(f"   - class_weight_optimization_results.csv")
print(f"   - class_weight_optimization_plots.png")
print(f"   - recommended_class_weights.json")
print(f"   - optimization_summary.txt")

print(f"\n🎯 RECOMMENDATION FOR FUTURE EXPERIMENTS:")
print(f"Use class weights: {{0: {best_auc['majority_weight']:.1f}, 1: {best_auc['minority_weight']:.1f}}}")
print(f"Expected performance: AUC={best_auc['val_auc_roc']:.4f}, F1={best_auc['val_f1_score']:.4f}, Sensitivity={best_auc['val_sensitivity']:.4f}")

print(f"\n{'='*60}")
print("CLASS WEIGHT OPTIMIZATION COMPLETED")
print(f"{'='*60}")