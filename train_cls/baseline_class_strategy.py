import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import json

# Project imports
from datasets.classification_mil_dataset import ClassificationMILDataset, create_classification_weighted_sampler
from utils.mil_utils import mil_collate_fn
from utils.classification_evaluator import ClassificationEvaluator
from utils.training_helpers import set_seed

# =====================================================================
# FIXED CONFIGURATION FOR BASELINE EXPERIMENTS
# =====================================================================

# Model configuration - FIXED for baseline
MODEL_NAME = "resnet18"
MIL_ARCHITECTURE = "mean"
INPUT_DIM = 512
COMBINED_CSV = "data/manifests/combined_features_resnet18.csv"

# Training hyperparameters - FIXED
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30
BATCH_SIZE = 1
OPTIMIZER_TYPE = "adam"

# Model architecture - FIXED
HIDDEN_DIM = 256
DROPOUT = 0.25

# Training settings - FIXED
USE_GRADIENT_CLIP = True
GRADIENT_CLIP_VAL = 1.0

# Learning rate scheduler - FIXED
USE_SCHEDULER = True
SCHEDULER_TYPE = "cosine"

# Cross-validation - FIXED
N_FOLDS = 5
RANDOM_SEED = 42

# Early stopping - FIXED
USE_EARLY_STOPPING = True
PATIENCE = 5

# =====================================================================
# CLASS BALANCING STRATEGIES TO TEST
# =====================================================================

BALANCING_STRATEGIES = {
    "no_balancing": {
        "name": "No Balancing (Baseline)",
        "use_class_weights": False,
        "use_focal_loss": False,
        "use_balanced_sampling": False,
        "description": "Standard cross-entropy loss without any balancing"
    },
    "focal_loss": {
        "name": "Focal Loss",
        "use_class_weights": False,
        "use_focal_loss": True,
        "use_balanced_sampling": False,
        "focal_gamma": 2.0,
        "focal_alpha": [0.25, 0.75],
        "description": "Focal loss to focus on hard examples"
    },
    "weighted_ce": {
        "name": "Weighted Cross-Entropy",
        "use_class_weights": True,
        "use_focal_loss": False,
        "use_balanced_sampling": False,
        "description": "Cross-entropy with inverse frequency class weights"
    },
    "balanced_sampling": {
        "name": "Balanced Sampling",
        "use_class_weights": False,
        "use_focal_loss": False,
        "use_balanced_sampling": True,
        "description": "Balanced sampling during training"
    },
    "focal_balanced": {
        "name": "Focal Loss + Balanced Sampling",
        "use_class_weights": False,
        "use_focal_loss": True,
        "use_balanced_sampling": True,
        "focal_gamma": 2.0,
        "focal_alpha": [0.25, 0.75],
        "description": "Combination of focal loss and balanced sampling"
    },
    "weighted_balanced": {
        "name": "Weighted CE + Balanced Sampling",
        "use_class_weights": True,
        "use_focal_loss": False,
        "use_balanced_sampling": True,
        "description": "Combination of weighted CE and balanced sampling"
    }
}

# =====================================================================
# MODEL DEFINITION
# =====================================================================

# Import the MeanPoolingMILClassifier from your models file
from models.classification_model_updated import MeanPoolingMILClassifier

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        device = inputs.device
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1-pt)**self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha = torch.tensor(self.alpha, device=device)
            focal_loss = alpha[targets] * focal_loss
        
        return focal_loss.mean()

# =====================================================================
# TRAINING FUNCTIONS
# =====================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, evaluator):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    evaluator.reset()
    
    for batch_features, batch_labels in dataloader:
        optimizer.zero_grad()
        
        if not isinstance(batch_features, list):
            batch_features = [batch_features]
            batch_labels = [batch_labels] if not isinstance(batch_labels, torch.Tensor) else batch_labels.unsqueeze(0)
        
        all_logits = []
        
        for features in batch_features:
            features = features.to(device)
            features = F.dropout(features, p=0.3, training=True)
            logits = model(features)
            
            if logits.dim() == 2 and logits.shape[0] == 1:
                logits = logits.squeeze(0)
            
            all_logits.append(logits)
        
        batch_logits = torch.stack(all_logits)
        
        if isinstance(batch_labels, list):
            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
        else:
            batch_labels = batch_labels.to(device).long()
        
        if batch_labels.dim() == 0:
            batch_labels = batch_labels.unsqueeze(0)
        
        loss = criterion(batch_logits, batch_labels)
        loss.backward()
        
        if USE_GRADIENT_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
        
        optimizer.step()
        
        probs = torch.softmax(batch_logits, dim=1)
        preds = torch.argmax(batch_logits, dim=1)
        
        total_loss += loss.item()
        evaluator.update(
            labels=batch_labels.cpu().numpy(),
            preds=preds.detach().cpu().numpy(),
            probs=probs.detach().cpu().numpy(),
            losses=[loss.item()]
        )
    
    return total_loss / len(dataloader)

def evaluate_epoch(model, dataloader, criterion, device, evaluator):
    """Evaluate for one epoch"""
    model.eval()
    total_loss = 0.0
    evaluator.reset()
    
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            if not isinstance(batch_features, list):
                batch_features = [batch_features]
                batch_labels = [batch_labels] if not isinstance(batch_labels, torch.Tensor) else batch_labels.unsqueeze(0)
            
            all_logits = []
            
            for features in batch_features:
                features = features.to(device)
                logits = model(features)
                
                if logits.dim() == 2 and logits.shape[0] == 1:
                    logits = logits.squeeze(0)
                
                all_logits.append(logits)
            
            batch_logits = torch.stack(all_logits)
            
            if isinstance(batch_labels, list):
                batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
            else:
                batch_labels = batch_labels.to(device).long()
            
            if batch_labels.dim() == 0:
                batch_labels = batch_labels.unsqueeze(0)
            
            loss = criterion(batch_logits, batch_labels)
            
            probs = torch.softmax(batch_logits, dim=1)
            preds = torch.argmax(batch_logits, dim=1)
            
            total_loss += loss.item()
            evaluator.update(
                labels=batch_labels.cpu().numpy(),
                preds=preds.cpu().numpy(),
                probs=probs.cpu().numpy(),
                losses=[loss.item()]
            )
    
    avg_loss = total_loss / len(dataloader)
    metrics = evaluator.compute_all_metrics(verbose=False)
    
    return avg_loss, metrics

def train_fold(train_loader, val_loader, device, fold_num, save_dir, strategy_config, class_weights=None):
    """Train a single fold with specified balancing strategy"""
    set_seed(RANDOM_SEED + fold_num)
    
    model = MeanPoolingMILClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        n_classes=2,
        dropout=DROPOUT
    ).to(device)
    
    # Setup loss function based on strategy
    if strategy_config["use_focal_loss"]:
        focal_gamma = strategy_config.get("focal_gamma", 2.0)
        focal_alpha = strategy_config.get("focal_alpha", None)
        criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    elif strategy_config["use_class_weights"] and class_weights is not None:
        weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    scheduler = None
    if USE_SCHEDULER:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    train_evaluator = ClassificationEvaluator(n_classes=2)
    val_evaluator = ClassificationEvaluator(n_classes=2)
    
    fold_metrics_history = []
    best_val_auroc = 0.0
    best_metrics = {}
    best_epoch = 0
    epochs_without_improvement = 0
    
    pbar = tqdm(range(1, NUM_EPOCHS + 1), desc=f"Fold {fold_num}")
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, train_evaluator)
        train_metrics = train_evaluator.compute_all_metrics(verbose=False)
        
        val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device, val_evaluator)
        
        epoch_metrics = {
            'epoch': epoch,
            'fold': fold_num,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        for k, v in train_metrics.items():
            epoch_metrics[f'train_{k}'] = v
        for k, v in val_metrics.items():
            epoch_metrics[f'val_{k}'] = v
        
        fold_metrics_history.append(epoch_metrics)
        
        if scheduler:
            scheduler.step()
        
        current_auroc = val_metrics.get('auroc', 0.0)
        if current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            best_metrics = val_metrics.copy()
            best_metrics['val_loss'] = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_auroc': best_val_auroc,
                'metrics': best_metrics,
                'strategy': strategy_config["name"]
            }
            checkpoint_path = os.path.join(save_dir, f'best_model_fold_{fold_num}.pt')
            torch.save(checkpoint, checkpoint_path)
        else:
            epochs_without_improvement += 1
        
        if USE_EARLY_STOPPING and epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        
        pbar.set_postfix({
            'val_auroc': f'{current_auroc:.4f}',
            'val_f1': f'{val_metrics.get("f1_score", 0):.4f}',
            'val_bal_acc': f'{val_metrics.get("balanced_accuracy", 0):.4f}',
            'best_auroc': f'{best_val_auroc:.4f}'
        })
    
    print(f"Fold {fold_num}: Best AUROC={best_val_auroc:.4f} at epoch {best_epoch}")
    
    best_metrics['fold'] = fold_num
    best_metrics['best_epoch'] = best_epoch
    best_metrics['best_val_auroc'] = best_val_auroc
    
    return best_metrics, fold_metrics_history

def run_strategy_cv(strategy_key, strategy_config):
    """Run complete CV for a single balancing strategy"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(RANDOM_SEED)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_baseline_class_balancing/{strategy_key}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"TESTING STRATEGY: {strategy_config['name']}")
    print(f"{'='*80}")
    print(f"Description: {strategy_config['description']}")
    print(f"Results directory: {results_dir}")
    
    # Save configuration
    config_info = {
        'strategy': strategy_key,
        'strategy_name': strategy_config['name'],
        'strategy_config': strategy_config,
        'model': MODEL_NAME,
        'mil_architecture': MIL_ARCHITECTURE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'hidden_dim': HIDDEN_DIM,
        'dropout': DROPOUT,
        'n_folds': N_FOLDS,
        'random_seed': RANDOM_SEED
    }
    
    with open(os.path.join(results_dir, "config.json"), 'w') as f:
        json.dump(config_info, f, indent=2)
    
    # Load dataset
    dataset = ClassificationMILDataset(COMBINED_CSV, label_column='RSHigh', threshold=25.0)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Calculate class weights
    class_counts = np.bincount(labels)
    class_weights = len(labels) / (len(class_counts) * class_counts)
    
    print(f"Dataset: {len(dataset)} samples")
    print(f"Class distribution: {class_counts[0]} low-risk, {class_counts[1]} high-risk")
    print(f"Class weights: Low={class_weights[0]:.3f}, High={class_weights[1]:.3f}")
    
    # Setup cross-validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    splits = list(skf.split(range(len(dataset)), labels))
    
    all_results = []
    all_histories = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\nFold {fold_idx}/{N_FOLDS}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # Setup data loaders based on strategy
        if strategy_config["use_balanced_sampling"]:
            # Create temporary dataset wrapper for sampler
            class TempDataset:
                def __init__(self, indices, full_dataset):
                    self.indices = indices
                    self.full_dataset = full_dataset
                def __len__(self):
                    return len(self.indices)
                def __getitem__(self, idx):
                    return self.full_dataset[self.indices[idx]]
            
            temp_dataset = TempDataset(train_idx, dataset)
            sampler = create_classification_weighted_sampler(temp_dataset, balance_classes=True)
            train_loader = DataLoader(
                train_subset, 
                batch_size=BATCH_SIZE, 
                sampler=sampler,
                collate_fn=mil_collate_fn, 
                num_workers=0
            )
        else:
            train_loader = DataLoader(
                train_subset, 
                batch_size=BATCH_SIZE, 
                shuffle=True,
                collate_fn=mil_collate_fn, 
                num_workers=0
            )
        
        val_loader = DataLoader(
            val_subset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            collate_fn=mil_collate_fn, 
            num_workers=0
        )
        
        # Train fold
        fold_results, fold_history = train_fold(
            train_loader, val_loader, device, fold_idx, results_dir,
            strategy_config, class_weights if strategy_config["use_class_weights"] else None
        )
        
        all_results.append(fold_results)
        all_histories.extend(fold_history)
    
    # Compute summary statistics
    results_df = pd.DataFrame(all_results)
    metrics = ['auroc', 'auc_pr', 'accuracy', 'balanced_accuracy', 'f1_score',
               'precision', 'recall', 'sensitivity', 'specificity', 'mcc']
    
    summary = {
        'strategy': strategy_key,
        'strategy_name': strategy_config['name']
    }
    
    for metric in metrics:
        if metric in results_df.columns:
            summary[f'{metric}_mean'] = results_df[metric].mean()
            summary[f'{metric}_std'] = results_df[metric].std()
    
    # Save results
    results_df.to_csv(os.path.join(results_dir, "fold_results.csv"), index=False)
    all_histories_df = pd.DataFrame(all_histories)
    all_histories_df.to_csv(os.path.join(results_dir, "training_history.csv"), index=False)
    
    with open(os.path.join(results_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def create_comparison_table(all_summaries):
    """Create comparison table of all strategies"""
    comparison_data = []
    
    for summary in all_summaries:
        row = {
            'Strategy': summary['strategy_name'],
            'AUROC': f"{summary.get('auroc_mean', 0):.3f} ± {summary.get('auroc_std', 0):.3f}",
            'AUC-PR': f"{summary.get('auc_pr_mean', 0):.3f} ± {summary.get('auc_pr_std', 0):.3f}",
            'Balanced Acc': f"{summary.get('balanced_accuracy_mean', 0):.3f} ± {summary.get('balanced_accuracy_std', 0):.3f}",
            'F1-Score': f"{summary.get('f1_score_mean', 0):.3f} ± {summary.get('f1_score_std', 0):.3f}",
            'Sensitivity': f"{summary.get('sensitivity_mean', 0):.3f} ± {summary.get('sensitivity_std', 0):.3f}",
            'Specificity': f"{summary.get('specificity_mean', 0):.3f} ± {summary.get('specificity_std', 0):.3f}",
            'MCC': f"{summary.get('mcc_mean', 0):.3f} ± {summary.get('mcc_std', 0):.3f}"
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

def plot_strategy_comparison(all_summaries, save_dir):
    """Create comparison plots for all strategies"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Extract metrics for plotting
    strategies = [s['strategy_name'] for s in all_summaries]
    metrics_to_plot = ['auroc', 'auc_pr', 'balanced_accuracy', 'f1_score', 'sensitivity', 'mcc']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Class Balancing Strategy Comparison\nResNet-18 + Mean Pooling', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        means = [s.get(f'{metric}_mean', 0) for s in all_summaries]
        stds = [s.get(f'{metric}_std', 0) for s in all_summaries]
        
        x_pos = np.arange(len(strategies))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        
        # Color code: baseline in gray, others in blue
        bars[0].set_color('gray')
        for bar in bars[1:]:
            bar.set_color('steelblue')
        
        # Highlight best performing strategy
        best_idx = np.argmax(means)
        bars[best_idx].set_color('darkgreen')
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.split()[0] for s in strategies], rotation=45, ha='right')
        ax.set_ylabel(metric.upper().replace('_', ' '))
        ax.set_title(metric.upper().replace('_', ' '))
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.01, f'{m:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'strategy_comparison_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def main():
    """Run all class balancing experiments"""
    print(f"{'='*80}")
    print(f"BASELINE CLASS BALANCING EXPERIMENTS")
    print(f"Model: ResNet-18 + Mean Pooling")
    print(f"{'='*80}")
    
    # Create main results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_results_dir = f"results_final/baseline/results_baseline_comparison_{timestamp}"
    os.makedirs(main_results_dir, exist_ok=True)
    
    all_summaries = []
    
    # Run each strategy
    for strategy_key, strategy_config in BALANCING_STRATEGIES.items():
        summary = run_strategy_cv(strategy_key, strategy_config)
        all_summaries.append(summary)
    
    # Create comparison table
    comparison_df = create_comparison_table(all_summaries)
    comparison_df.to_csv(os.path.join(main_results_dir, "strategy_comparison_table.csv"), index=False)
    
    # Create comparison plots
    plot_path = plot_strategy_comparison(all_summaries, main_results_dir)
    
    # Print final comparison
    print(f"\n{'='*80}")
    print(f"FINAL COMPARISON - ALL STRATEGIES")
    print(f"{'='*80}")
    print("\n" + comparison_df.to_string(index=False))
    
    # Identify best strategy for each metric
    print(f"\n{'='*80}")
    print(f"BEST STRATEGY PER METRIC")
    print(f"{'='*80}")
    
    metrics_to_check = ['auroc', 'auc_pr', 'balanced_accuracy', 'f1_score', 'sensitivity', 'mcc']
    for metric in metrics_to_check:
        means = [(s['strategy_name'], s.get(f'{metric}_mean', 0)) for s in all_summaries]
        best = max(means, key=lambda x: x[1])
        print(f"{metric.upper()}: {best[0]} ({best[1]:.4f})")
    
    # Save all summaries
    with open(os.path.join(main_results_dir, "all_summaries.json"), 'w') as f:
        json.dump(all_summaries, f, indent=2)
    
    print(f"\nAll results saved to: {main_results_dir}")
    print(f"Comparison plot saved to: {plot_path}")
    
    return all_summaries, comparison_df

if __name__ == "__main__":
    all_summaries, comparison_df = main()