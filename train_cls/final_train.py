# run_final_cv_classification.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

# Project imports
from datasets.classification_mil_dataset import ClassificationMILDataset, create_classification_weighted_sampler
from utils.mil_utils import mil_collate_fn
from utils.classification_evaluator import ClassificationEvaluator
from utils.training_helpers import set_seed

# Foundation model configurations
FOUNDATION_MODELS = {
    "h-optimus": {
        "input_dim": 1536,
        "combined_csv": "data/manifests/combined_features_h-optimus.csv"
    },
    "virchow2": {
        "input_dim": 1280,
        "combined_csv": "data/manifests/combined_features_virchow2.csv"
    },
    "uni2-h": {
        "input_dim": 1536,
        "combined_csv": "data/manifests/combined_features_uni2-h.csv"
    },
    "resnet18": {
        "input_dim": 512,
        "combined_csv": "data/manifests/combined_features_resnet18.csv"
    }
}

def load_config(config_path):
    """Load hyperparameters from JSON file"""
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)

def create_model(config, input_dim, device, n_classes=2):
    """Create classification model based on config"""
    # Check model type based on config parameters
    if 'n_branches' in config:
        # ACMIL models (both ACMIL and ACMIL_CLAM_Hybrid have n_branches)
        if 'gate' in config:
            # ACMIL_CLAM_Hybrid classifier
            from models.classification_model import ACMIL_CLAM_HybridClassifier
            model = ACMIL_CLAM_HybridClassifier(
                input_dim=input_dim,
                n_branches=config['n_branches'],
                hidden_dim=config['hidden_dim'],
                attention_hidden_dim=config['attention_hidden_dim'],
                n_classes=n_classes,
                mask_ratio=config.get('mask_ratio', 0.0),
                n_masked_patch=config.get('n_masked_patch', 10),
                dropout=config.get('dropout', 0.25),
                gate=config.get('gate', True)
            ).to(device)
        else:
            # Pure ACMIL classifier
            from models.classification_model import ACMILClassifier
            model = ACMILClassifier(
                input_dim=input_dim,
                hidden_dim=config['hidden_dim'],
                n_branches=config['n_branches'],
                n_classes=n_classes,
                n_masked_patch=config.get('n_masked_patch', 10),
                mask_ratio=config.get('mask_ratio', 0.6),
                dropout=config.get('dropout', 0.25)
            ).to(device)
    else:
        # Standard CLAM classifier
        from models.classification_model import CLAMClassifier
        model = CLAMClassifier(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            attention_hidden_dim=config['attention_hidden_dim'],
            n_classes=n_classes,
            dropout=config.get('dropout', 0.25),
            gate=config.get('gate', False)
        ).to(device)
    
    return model

def create_optimizer(model, config):
    """Create optimizer from config"""
    opt_type = config.get('optimizer_type', 'Adam')
    lr = config['learning_rate']
    wd = config.get('weight_decay', 0.0)
    
    if opt_type == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

def train_epoch(model, dataloader, criterion, optimizer, device, evaluator, grad_clip=False, clip_val=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    evaluator.reset()
    
    for features, label in dataloader:
        # Process data
        if isinstance(features, list):
            features = features[0]
        features = features.to(device)
        
        if isinstance(label, list):
            label = label[0]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor([label], dtype=torch.long)
        label = label.to(device).long()
        if label.dim() == 0:
            label = label.unsqueeze(0)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(features)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        
        loss = criterion(logits, label)
        loss.backward()
        
        # Gradient clipping
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        
        optimizer.step()
        
        # Get predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        total_loss += loss.item()
        evaluator.update(
            labels=label.cpu().numpy(),
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
        for features, label in dataloader:
            # Process data
            if isinstance(features, list):
                features = features[0]
            features = features.to(device)
            
            if isinstance(label, list):
                label = label[0]
            if not isinstance(label, torch.Tensor):
                label = torch.tensor([label], dtype=torch.long)
            label = label.to(device).long()
            if label.dim() == 0:
                label = label.unsqueeze(0)
            
            # Forward pass
            logits = model(features)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            
            loss = criterion(logits, label)
            
            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Update metrics
            total_loss += loss.item()
            evaluator.update(
                labels=label.cpu().numpy(),
                preds=preds.cpu().numpy(),
                probs=probs.cpu().numpy(),
                losses=[loss.item()]
            )
    
    avg_loss = total_loss / len(dataloader)
    metrics = evaluator.compute_all_metrics(verbose=False)
    
    return avg_loss, metrics

def train_fold(config, model_config, train_loader, val_loader, device, fold_num, save_dir, class_weights=None):
    """Train a single fold and save best model"""
    set_seed(42 + fold_num)
    
    # Create model
    n_classes = 2  # Binary classification
    model = create_model(config, model_config["input_dim"], device, n_classes)
    
    # Setup training
    loss_type = config.get('loss_type', 'ce')
    if loss_type == 'ce' and class_weights is not None:
        weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = create_optimizer(model, config)
    
    n_epochs = config.get('n_epochs', 20)
    grad_clip = config.get('use_gradient_clip', False)
    clip_val = config.get('gradient_clip_val', 1.0)
    
    # Setup scheduler if specified
    scheduler = None
    scheduler_type = config.get('scheduler_type', 'none')
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Initialize evaluators
    train_evaluator = ClassificationEvaluator(n_classes=n_classes)
    val_evaluator = ClassificationEvaluator(n_classes=n_classes)
    
    # Track metrics for plotting
    fold_metrics_history = []
    
    # Training loop
    best_auroc = 0.0
    best_metrics = {}
    best_epoch = 0
    
    pbar = tqdm(range(1, n_epochs + 1), desc=f"Fold {fold_num}")
    for epoch in pbar:
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, 
                                train_evaluator, grad_clip, clip_val)
        train_metrics = train_evaluator.compute_all_metrics(verbose=False)
        
        # Evaluate on validation
        val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device, val_evaluator)
        
        # Store epoch metrics
        epoch_metrics = {
            'epoch': epoch,
            'fold': fold_num,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        # Add train metrics with prefix
        for k, v in train_metrics.items():
            epoch_metrics[f'train_{k}'] = v
        # Add val metrics with prefix  
        for k, v in val_metrics.items():
            epoch_metrics[f'val_{k}'] = v
        
        fold_metrics_history.append(epoch_metrics)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Track best model based on AUROC
        current_auroc = val_metrics.get('auroc', 0.0)
        if current_auroc > best_auroc:
            best_auroc = current_auroc
            best_metrics = val_metrics.copy()
            best_epoch = epoch
            
            # Save best model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auroc': best_auroc,
                'metrics': best_metrics,
                'config': config
            }
            checkpoint_path = os.path.join(save_dir, f'best_model_fold_{fold_num}.pt')
            torch.save(checkpoint, checkpoint_path)
        
        # Update progress bar
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_auroc': f'{current_auroc:.4f}',
            'val_f1': f'{val_metrics.get("f1_score", 0):.4f}',
            'best': f'{best_auroc:.4f}'
        })
    
    print(f"Fold {fold_num}: Best AUROC={best_auroc:.4f} at epoch {best_epoch}")
    
    # Save fold training history
    fold_history_df = pd.DataFrame(fold_metrics_history)
    fold_history_df.to_csv(os.path.join(save_dir, f'fold_{fold_num}_history.csv'), index=False)
    
    # Save confusion matrix and ROC curve for best model
    val_evaluator.plot_confusion_matrix(
        save_path=os.path.join(save_dir, f'fold_{fold_num}_confusion_matrix.png'),
        title=f'Fold {fold_num} - Confusion Matrix'
    )
    val_evaluator.plot_roc_curve(
        save_path=os.path.join(save_dir, f'fold_{fold_num}_roc_curve.png'),
        title=f'Fold {fold_num} - ROC Curve'
    )
    
    best_metrics['fold'] = fold_num
    best_metrics['best_epoch'] = best_epoch
    
    return best_metrics, fold_metrics_history

def run_cv_training(model_name, config_path, n_folds=5):
    """Run complete CV training for classification"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    # Load config
    config = load_config(config_path)
    model_config = FOUNDATION_MODELS[model_name]
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_classification_cv/{model_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save config for reference
    with open(os.path.join(results_dir, "config_used.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"{'='*80}")
    print(f"CLASSIFICATION CV TRAINING: {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Folds: {n_folds}")
    print(f"Results directory: {results_dir}")
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Load dataset
    print(f"\nLoading dataset: {model_config['combined_csv']}")
    
    # Check if RSHigh column exists, otherwise use RS column
    df_check = pd.read_csv(model_config["combined_csv"])
    if 'RSHigh' in df_check.columns:
        label_column = 'RSHigh'
    elif 'RS' in df_check.columns:
        label_column = 'RS'
    else:
        raise ValueError("Neither RSHigh nor RS column found in dataset")
    
    dataset = ClassificationMILDataset(
        model_config["combined_csv"],
        label_column=label_column,
        threshold=25.0
    )
    
    # Get labels for stratification
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(labels)
    class_weights = len(labels) / (len(class_counts) * class_counts)
    
    print(f"Total samples: {len(dataset)}")
    print(f"Class distribution: {class_counts[0]} low-risk, {class_counts[1]} high-risk")
    print(f"Class weights: Low={class_weights[0]:.3f}, High={class_weights[1]:.3f}")
    
    # Create CV splits
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(skf.split(range(len(dataset)), labels))
    
    # Run CV training
    all_results = []
    all_histories = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx}/{n_folds}")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        # Create dataloaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # Apply sampling strategy if specified
        if config.get('use_class_balancing', True):
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
            sampler = create_classification_weighted_sampler(
                temp_dataset, 
                balance_classes=True
            )
            train_loader = DataLoader(
                train_subset, 
                batch_size=1, 
                sampler=sampler,
                collate_fn=mil_collate_fn, 
                num_workers=0
            )
        else:
            train_loader = DataLoader(
                train_subset, 
                batch_size=1, 
                shuffle=True,
                collate_fn=mil_collate_fn, 
                num_workers=0
            )
        
        val_loader = DataLoader(
            val_subset, 
            batch_size=1, 
            shuffle=False,
            collate_fn=mil_collate_fn, 
            num_workers=0
        )
        
        # Train fold
        fold_results, fold_history = train_fold(
            config, model_config, train_loader, val_loader,
            device, fold_idx, results_dir,
            class_weights if config.get('use_class_weights', True) else None
        )
        all_results.append(fold_results)
        all_histories.extend(fold_history)
    
    # Calculate statistics
    results_df = pd.DataFrame(all_results)
    
    # Key metrics to summarize
    metrics = ['auroc', 'auc_pr', 'accuracy', 'balanced_accuracy', 'f1_score',
               'precision', 'recall', 'sensitivity', 'specificity', 'mcc']
    
    summary = {}
    for metric in metrics:
        if metric in results_df.columns:
            summary[f'{metric}_mean'] = results_df[metric].mean()
            summary[f'{metric}_std'] = results_df[metric].std()
    
    # Print final results
    print(f"\n{'='*80}")
    print(f"FINAL CLASSIFICATION CV RESULTS")
    print(f"{'='*80}")
    print(f"AUROC:             {summary.get('auroc_mean', 0):.4f} ± {summary.get('auroc_std', 0):.4f}")
    print(f"AUC-PR:            {summary.get('auc_pr_mean', 0):.4f} ± {summary.get('auc_pr_std', 0):.4f}")
    print(f"Accuracy:          {summary.get('accuracy_mean', 0):.4f} ± {summary.get('accuracy_std', 0):.4f}")
    print(f"Balanced Accuracy: {summary.get('balanced_accuracy_mean', 0):.4f} ± {summary.get('balanced_accuracy_std', 0):.4f}")
    print(f"F1-Score:          {summary.get('f1_score_mean', 0):.4f} ± {summary.get('f1_score_std', 0):.4f}")
    print(f"Sensitivity:       {summary.get('sensitivity_mean', 0):.4f} ± {summary.get('sensitivity_std', 0):.4f}")
    print(f"Specificity:       {summary.get('specificity_mean', 0):.4f} ± {summary.get('specificity_std', 0):.4f}")
    print(f"MCC:               {summary.get('mcc_mean', 0):.4f} ± {summary.get('mcc_std', 0):.4f}")
    
    # Create summary visualizations
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics_to_plot = [
        ('auroc', 'AUROC', axes[0,0]),
        ('auc_pr', 'AUC-PR', axes[0,1]),
        ('f1_score', 'F1 Score', axes[0,2]),
        ('sensitivity', 'Sensitivity', axes[1,0]),
        ('specificity', 'Specificity', axes[1,1]),
        ('balanced_accuracy', 'Balanced Accuracy', axes[1,2])
    ]
    
    for metric_name, display_name, ax in metrics_to_plot:
        if metric_name in results_df.columns:
            fold_values = results_df[metric_name].values
            bp = ax.boxplot([fold_values], tick_labels=['CV Folds'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            
            # Add mean line
            mean_val = np.mean(fold_values)
            ax.axhline(y=mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.3f}')
            
            # Add individual fold points
            x = np.ones(len(fold_values))
            ax.scatter(x, fold_values, color='darkblue', alpha=0.6, s=50)
            
            ax.set_ylabel(display_name)
            ax.set_title(f'{display_name} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name.upper()} - Classification CV Metrics Distribution', 
                 fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cv_metrics_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save all results
    results_df.to_csv(os.path.join(results_dir, "fold_results.csv"), index=False)
    
    # Save all training histories
    all_histories_df = pd.DataFrame(all_histories)
    all_histories_df.to_csv(os.path.join(results_dir, "all_training_history.csv"), index=False)
    
    with open(os.path.join(results_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Find and copy best model overall
    best_fold = results_df['auroc'].idxmax() + 1
    best_model_path = os.path.join(results_dir, f'best_model_fold_{best_fold}.pt')
    final_model_path = os.path.join(results_dir, 'best_model_overall.pt')
    
    import shutil
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, final_model_path)
        print(f"\nBest model from fold {best_fold} saved as: {final_model_path}")
    
    print(f"\nAll results saved to: {results_dir}")
    
    return summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run classification CV training')
    parser.add_argument('--model', type=str, required=True,
                      choices=['resnet18', 'uni2-h', 'virchow2', 'h-optimus'],
                      help='Model name')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config JSON file with hyperparameters')
    parser.add_argument('--folds', type=int, default=5,
                      help='Number of CV folds (default: 5)')
    
    args = parser.parse_args()
    
    run_cv_training(args.model, args.config, args.folds)