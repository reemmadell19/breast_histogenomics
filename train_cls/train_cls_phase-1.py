# train_cls_phase1_cv.py
# 5-Fold Cross-Validation Training Script for Phase 1 Foundation Model Comparison - Classification
# Uses SAME seed and shared CV splits for fair foundation model comparison

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# Project imports
from datasets.classification_mil_dataset import ClassificationMILDataset, create_classification_weighted_sampler
from models.classification_model import MeanPoolingMILClassifier
from utils.mil_utils import mil_collate_fn
from utils.classification_evaluator import ClassificationEvaluator
from utils.training_helpers import set_seed, print_experiment_header

# Feature extractor configurations
FEATURE_EXTRACTORS = {
    "resnet18": {
        "input_dim": 512,
        "combined_csv": "data/manifests/combined_features_resnet18.csv"
    },
    "resnet50": {
        "input_dim": 2048,
        "combined_csv": "data/manifests/combined_features_resnet50.csv"
    },
    "conch": {
        "input_dim": 512,
        "combined_csv": "data/manifests/combined_features_conch.csv"
    },
    "uni2-h": {
        "input_dim": 1536,
        "combined_csv": "data/manifests/combined_features_uni2-h.csv"
    },
    "virchow2": {
        "input_dim": 1280,
        "combined_csv": "data/manifests/combined_features_virchow2.csv"
    },
    "h-optimus": {
        "input_dim": 1536,
        "combined_csv": "data/manifests/combined_features_h-optimus.csv"
    }
}

# Configuration for Phase 1: Foundation Model Comparison - Classification
CONFIG = {
    "feature_extractor": "h-optimus",  # CHANGE THIS FOR EACH FOUNDATION MODEL
    "mil_pooling": "mean",  # Fixed for Phase 1
    "experiment_type": "phase1_cv_classification", 
    "hidden_dim": 128,
    "n_classes": 2,  # Binary classification
    "batch_size": 1,
    "lr": 1e-4,
    "num_epochs": 15,
    "n_folds": 5,
    "use_class_balancing": True,  # Balance sampling
    "use_class_weights": False,  # Weighted loss
    "loss_type": "ce",  # CrossEntropy for classification
    "random_seed": 42,
    "threshold": 25.0  # RS threshold for low/high risk
}

def setup_cv_directories(feature_extractor: str) -> str:
    """Create results directory for CV experiments."""
    results_dir = f"results_classification_phase1_cv/{feature_extractor}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def create_cv_dataset_and_splits(combined_csv: str, n_folds: int, random_seed: int, threshold: float = 25.0):
    """
    Create dataset and stratified K-fold splits for cross-validation
    """
    # Check if RSHigh column exists
    df_check = pd.read_csv(combined_csv)
    if 'RSHigh' in df_check.columns:
        label_column = 'RSHigh'
    elif 'RS' in df_check.columns:
        label_column = 'RS'
    else:
        raise ValueError("Neither RSHigh nor RS column found in dataset")
    
    # Load combined dataset (train + val combined for CV)
    full_dataset = ClassificationMILDataset(combined_csv, label_column=label_column, threshold=threshold)
    
    # Extract labels for stratification
    labels = []
    for i in range(len(full_dataset)):
        _, label = full_dataset[i]
        labels.append(label)
    
    labels = np.array(labels)
    
    # Create stratified K-fold splits (already have binary labels)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    cv_splits = list(skf.split(range(len(full_dataset)), labels))
    
    return full_dataset, cv_splits, labels

def create_fold_dataloaders(full_dataset, train_indices, val_indices, batch_size: int,
                          use_class_balancing: bool = True):
    """Create dataloaders for a specific CV fold"""
    
    # Create subset datasets
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    # Create dataloaders
    if use_class_balancing:
        # Create temporary dataset for sampler
        class TempDataset:
            def __init__(self, indices, full_dataset):
                self.indices = indices
                self.full_dataset = full_dataset
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                return self.full_dataset[self.indices[idx]]
        
        temp_train_dataset = TempDataset(train_indices, full_dataset)
        
        sampler = create_classification_weighted_sampler(
            temp_train_dataset, 
            balance_classes=True
        )
        
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=mil_collate_fn,
            num_workers=0
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=mil_collate_fn,
            num_workers=0
        )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=mil_collate_fn,
        num_workers=0
    )
    
    return train_loader, val_loader

def train_single_fold(fold_num: int, train_loader, val_loader, experiment_config, 
                     device, class_weights=None):
    """Train and validate a single CV fold for classification"""
    
    # Use SAME seed for all foundation models
    set_seed(experiment_config["random_seed"])
    
    # Create fresh model and optimizer for each fold
    model = MeanPoolingMILClassifier(
        input_dim=experiment_config["input_dim"], 
        hidden_dim=experiment_config["hidden_dim"],
        n_classes=experiment_config["n_classes"]
    ).to(device)
    
    # Use only unweighted CrossEntropy loss
    criterion = nn.CrossEntropyLoss()  # No class weights
    
    optimizer = optim.Adam(model.parameters(), lr=experiment_config["lr"])
    
    # Initialize fresh evaluators for this fold
    train_evaluator = ClassificationEvaluator(n_classes=experiment_config["n_classes"])
    val_evaluator = ClassificationEvaluator(n_classes=experiment_config["n_classes"])
    
    # Track the PRIMARY metric - AUROC for classification
    best_val_auroc = 0.0
    best_fold_metrics = None
    all_epoch_metrics = []
    
    # Training loop for this fold
    for epoch in range(1, experiment_config["num_epochs"] + 1):
        # Reset evaluators each epoch
        train_evaluator.reset()
        val_evaluator.reset()
        
        # Training epoch
        model.train()
        
        for features, label in train_loader:
            # Process batch data
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
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(features)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
                
            loss = criterion(logits, label)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Update metrics
            train_evaluator.update(
                labels=label.cpu().numpy(),
                preds=preds.detach().cpu().numpy(),
                probs=probs.detach().cpu().numpy()
            )
        
        train_metrics = train_evaluator.compute_all_metrics(verbose=False)
        
        # Validation epoch
        model.eval()
        
        with torch.no_grad():
            for features, label in val_loader:
                # Process batch data
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
                val_evaluator.update(
                    labels=label.cpu().numpy(),
                    preds=preds.cpu().numpy(),
                    probs=probs.cpu().numpy()
                )
        
        val_metrics = val_evaluator.compute_all_metrics(verbose=False)
        all_epoch_metrics.append(val_metrics.copy())
        
        # Track best model based on AUROC
        current_auroc = val_metrics.get('auroc', 0.0)
        if current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            best_fold_metrics = val_metrics.copy()
            best_fold_metrics['best_epoch'] = epoch
        
        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == experiment_config["num_epochs"]:
            print(f"  Epoch {epoch}: AUROC={val_metrics.get('auroc', 0):.3f}, "
                  f"F1={val_metrics.get('f1_score', 0):.3f}, "
                  f"Acc={val_metrics.get('accuracy', 0):.3f}, "
                  f"Sens={val_metrics.get('sensitivity', 0):.3f}, "
                  f"Spec={val_metrics.get('specificity', 0):.3f}")
    
    # Add final metrics
    if best_fold_metrics is not None:
        best_fold_metrics['final_auroc'] = all_epoch_metrics[-1]['auroc']
        best_fold_metrics['final_f1'] = all_epoch_metrics[-1]['f1_score']
        best_fold_metrics['best_auroc'] = best_val_auroc
    else:
        best_fold_metrics = all_epoch_metrics[-1]
        best_fold_metrics['best_epoch'] = experiment_config["num_epochs"]
        best_fold_metrics['best_auroc'] = best_fold_metrics['auroc']
    
    return best_fold_metrics

def run_cross_validation_experiment(experiment_config, device, shared_cv_splits, 
                                   full_dataset, class_weights):
    """Run complete cross-validation experiment for one foundation model using shared splits"""
    
    feature_name = experiment_config["feature_extractor"]
    
    print(f"\n{'='*80}")
    print(f"TESTING FOUNDATION MODEL: {feature_name.upper()}")
    print(f"{'='*80}")
    print(f"Input Dimension: {experiment_config['input_dim']}")
    print(f"Cross-Validation: {experiment_config['n_folds']}-fold stratified")
    print(f"MIL Architecture: Mean Pooling (baseline)")
    print(f"Loss Function: CrossEntropy")
    print(f"Class Weights: {class_weights}")
    print(f"SAME INITIALIZATION SEED: {experiment_config['random_seed']} for all foundation models")
    
    print(f"Total samples: {len(full_dataset)}")
    
    # Store results for each fold
    fold_results = []
    
    # Run cross-validation using shared splits
    for fold, (train_idx, val_idx) in enumerate(shared_cv_splits):
        print(f"\nFold {fold+1}/{experiment_config['n_folds']} - {feature_name}")
        print(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        
        # Create fold-specific dataloaders
        train_loader, val_loader = create_fold_dataloaders(
            full_dataset, train_idx, val_idx, experiment_config["batch_size"],
            experiment_config["use_class_balancing"]
        )
        
        # Train this fold
        fold_metrics = train_single_fold(
            fold+1, train_loader, val_loader, experiment_config, device, class_weights
        )
        
        # Store results with metadata
        fold_metrics['fold'] = fold + 1
        fold_metrics['train_size'] = len(train_idx)
        fold_metrics['val_size'] = len(val_idx)
        fold_metrics['foundation_model'] = feature_name
        fold_metrics['initialization_seed'] = experiment_config['random_seed']
        
        # Print fold summary
        print(f"  Fold {fold+1} Results: AUROC={fold_metrics.get('auroc', 0):.4f}, "
              f"F1={fold_metrics.get('f1_score', 0):.4f}, "
              f"Accuracy={fold_metrics.get('accuracy', 0):.4f}, "
              f"Sensitivity={fold_metrics.get('sensitivity', 0):.4f}, "
              f"Specificity={fold_metrics.get('specificity', 0):.4f}, "
              f"MCC={fold_metrics.get('mcc', 0):.4f}")
        
        fold_results.append(fold_metrics)
    
    return fold_results

def analyze_cv_results(fold_results, feature_name, results_dir):
    """Analyze and summarize cross-validation results for classification"""
    
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION RESULTS SUMMARY: {feature_name.upper()}")
    print(f"{'='*80}")
    
    # Convert to DataFrame for easy analysis
    cv_df = pd.DataFrame(fold_results)
    
    # Key metrics to analyze for classification
    key_metrics = ['auroc', 'auc_pr', 'accuracy', 'balanced_accuracy', 'f1_score', 
                   'precision', 'recall', 'sensitivity', 'specificity', 'mcc']
    
    # Calculate statistics across folds
    cv_stats = {}
    for metric in key_metrics:
        if metric in cv_df.columns:
            values = cv_df[metric].values
            cv_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            }
    
    # Print detailed statistics table
    print(f"{'Metric':<20} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'CV':<8}")
    print(f"{'-'*70}")
    
    for metric in key_metrics:
        if metric in cv_stats:
            stats = cv_stats[metric]
            print(f"{metric.upper():<20} {stats['mean']:<8.3f} {stats['std']:<8.3f} "
                  f"{stats['min']:<8.3f} {stats['max']:<8.3f} {stats['cv']:<8.3f}")
    
    # Performance summary
    print(f"\n{'='*60}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"AUROC: {cv_stats.get('auroc', {}).get('mean', 0):.4f} ± {cv_stats.get('auroc', {}).get('std', 0):.4f}")
    print(f"AUC-PR: {cv_stats.get('auc_pr', {}).get('mean', 0):.4f} ± {cv_stats.get('auc_pr', {}).get('std', 0):.4f}")
    print(f"F1 Score: {cv_stats.get('f1_score', {}).get('mean', 0):.4f} ± {cv_stats.get('f1_score', {}).get('std', 0):.4f}")
    print(f"Accuracy: {cv_stats.get('accuracy', {}).get('mean', 0):.4f} ± {cv_stats.get('accuracy', {}).get('std', 0):.4f}")
    print(f"Balanced Accuracy: {cv_stats.get('balanced_accuracy', {}).get('mean', 0):.4f} ± {cv_stats.get('balanced_accuracy', {}).get('std', 0):.4f}")
    print(f"Sensitivity: {cv_stats.get('sensitivity', {}).get('mean', 0):.4f} ± {cv_stats.get('sensitivity', {}).get('std', 0):.4f}")
    print(f"Specificity: {cv_stats.get('specificity', {}).get('mean', 0):.4f} ± {cv_stats.get('specificity', {}).get('std', 0):.4f}")
    print(f"MCC: {cv_stats.get('mcc', {}).get('mean', 0):.4f} ± {cv_stats.get('mcc', {}).get('std', 0):.4f}")
    
    # Model stability assessment
    auroc_cv = cv_stats.get('auroc', {}).get('cv', 1.0)
    stability = "Highly Stable" if auroc_cv < 0.02 else "Moderately Stable" if auroc_cv < 0.05 else "Variable"
    print(f"Model Stability (AUROC CV): {auroc_cv:.4f} - {stability}")
    
    # Save detailed results
    cv_df.to_csv(os.path.join(results_dir, f"{feature_name}_cv_detailed_results.csv"), index=False)
    
    # Save summary statistics
    summary_df = pd.DataFrame(cv_stats).T
    summary_df.to_csv(os.path.join(results_dir, f"{feature_name}_cv_summary_stats.csv"))
    
    return cv_stats, cv_df

def create_cv_summary_for_comparison(feature_name, cv_stats):
    """Create summary row for comparing across foundation models"""
    
    def safe_get_metric(stats_dict, metric_name, stat_type):
        """Safely extract metric statistics with fallback to 0"""
        return stats_dict.get(metric_name, {}).get(stat_type, 0.0)
    
    return {
        'Model': feature_name,
        'AUROC_mean': safe_get_metric(cv_stats, 'auroc', 'mean'),
        'AUROC_std': safe_get_metric(cv_stats, 'auroc', 'std'),
        'AUROC_cv': safe_get_metric(cv_stats, 'auroc', 'cv'),
        'AUC_PR_mean': safe_get_metric(cv_stats, 'auc_pr', 'mean'),
        'AUC_PR_std': safe_get_metric(cv_stats, 'auc_pr', 'std'),
        'F1_Score_mean': safe_get_metric(cv_stats, 'f1_score', 'mean'),
        'F1_Score_std': safe_get_metric(cv_stats, 'f1_score', 'std'),
        'Accuracy_mean': safe_get_metric(cv_stats, 'accuracy', 'mean'),
        'Accuracy_std': safe_get_metric(cv_stats, 'accuracy', 'std'),
        'Balanced_Accuracy_mean': safe_get_metric(cv_stats, 'balanced_accuracy', 'mean'),
        'Balanced_Accuracy_std': safe_get_metric(cv_stats, 'balanced_accuracy', 'std'),
        'Sensitivity_mean': safe_get_metric(cv_stats, 'sensitivity', 'mean'),
        'Sensitivity_std': safe_get_metric(cv_stats, 'sensitivity', 'std'),
        'Specificity_mean': safe_get_metric(cv_stats, 'specificity', 'mean'),
        'Specificity_std': safe_get_metric(cv_stats, 'specificity', 'std'),
        'MCC_mean': safe_get_metric(cv_stats, 'mcc', 'mean'),
        'MCC_std': safe_get_metric(cv_stats, 'mcc', 'std')
    }

def create_comparison_plot(comparison_df, output_dir):
    """Create visualization comparing foundation models"""
    
    # Sort by AUROC
    comparison_df_sorted = comparison_df.sort_values('AUROC_mean', ascending=False)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. AUROC comparison
    ax1 = axes[0, 0]
    models = comparison_df_sorted['Model'].values
    auroc_means = comparison_df_sorted['AUROC_mean'].values
    auroc_stds = comparison_df_sorted['AUROC_std'].values
    
    x_pos = np.arange(len(models))
    ax1.bar(x_pos, auroc_means, yerr=auroc_stds, capsize=5, color='skyblue')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45)
    ax1.set_ylabel('AUROC')
    ax1.set_title('AUROC Comparison')
    ax1.grid(True, alpha=0.3)
    
    # 2. F1 Score comparison
    ax2 = axes[0, 1]
    f1_means = comparison_df_sorted['F1_Score_mean'].values
    f1_stds = comparison_df_sorted['F1_Score_std'].values
    
    ax2.bar(x_pos, f1_means, yerr=f1_stds, capsize=5, color='lightcoral')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45)
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 3. Sensitivity vs Specificity
    ax3 = axes[0, 2]
    sens_means = comparison_df_sorted['Sensitivity_mean'].values
    spec_means = comparison_df_sorted['Specificity_mean'].values
    
    width = 0.35
    ax3.bar(x_pos - width/2, sens_means, width, label='Sensitivity', color='lightgreen')
    ax3.bar(x_pos + width/2, spec_means, width, label='Specificity', color='lightblue')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models, rotation=45)
    ax3.set_ylabel('Score')
    ax3.set_title('Sensitivity vs Specificity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. MCC comparison
    ax4 = axes[1, 0]
    mcc_means = comparison_df_sorted['MCC_mean'].values
    mcc_stds = comparison_df_sorted['MCC_std'].values
    
    ax4.bar(x_pos, mcc_means, yerr=mcc_stds, capsize=5, color='gold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(models, rotation=45)
    ax4.set_ylabel('MCC')
    ax4.set_title('Matthews Correlation Coefficient')
    ax4.grid(True, alpha=0.3)
    
    # 5. Model stability (CV of AUROC)
    ax5 = axes[1, 1]
    auroc_cvs = comparison_df_sorted['AUROC_cv'].values
    
    ax5.bar(x_pos, auroc_cvs, color='salmon')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(models, rotation=45)
    ax5.set_ylabel('Coefficient of Variation')
    ax5.set_title('Model Stability (Lower is Better)')
    ax5.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Stability Threshold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Overall ranking heatmap
    ax6 = axes[1, 2]
    
    # Create ranking matrix
    metrics_for_ranking = ['AUROC_mean', 'F1_Score_mean', 'MCC_mean', 
                           'Sensitivity_mean', 'Specificity_mean']
    ranking_data = []
    
    for metric in metrics_for_ranking:
        sorted_idx = np.argsort(-comparison_df[metric].values)  # Descending
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(len(comparison_df)) + 1
        ranking_data.append(ranks)
    
    ranking_matrix = np.array(ranking_data)
    
    sns.heatmap(ranking_matrix, annot=True, fmt='d', cmap='RdYlGn_r',
                xticklabels=comparison_df['Model'].values,
                yticklabels=['AUROC', 'F1', 'MCC', 'Sens', 'Spec'],
                ax=ax6, vmin=1, vmax=len(comparison_df))
    ax6.set_title('Model Rankings (1=Best)')
    
    plt.suptitle('Phase 1: Foundation Model Comparison - Classification', fontsize=16, weight='bold')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'foundation_models_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to: {output_dir}")

def run_phase1_foundation_comparison():
    """
    Run complete Phase 1 comparison across all foundation models using SAME conditions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(CONFIG["random_seed"])
    
    print(f"{'='*80}")
    print(f"PHASE 1: FOUNDATION MODEL COMPARISON - CLASSIFICATION")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Configuration: Mean Pooling + CrossEntropy Loss + Class Balancing")
    print(f"Foundation Models: {list(FEATURE_EXTRACTORS.keys())}")
    print(f"SAME CV splits and initialization for all foundation models")
    
    # Create CV splits ONCE and use for ALL foundation models
    print(f"Creating ONE set of CV splits for ALL foundation models (seed: {CONFIG['random_seed']})")
    
    # Use first foundation model to create the splits
    first_model = list(FEATURE_EXTRACTORS.keys())[0]
    first_csv = FEATURE_EXTRACTORS[first_model]["combined_csv"]
    
    _, shared_cv_splits, shared_labels = create_cv_dataset_and_splits(
        first_csv, 
        CONFIG["n_folds"], 
        CONFIG["random_seed"],
        CONFIG["threshold"]
    )
    
    # Calculate class weights
    class_counts = np.bincount(shared_labels)
    class_weights = len(shared_labels) / (len(class_counts) * class_counts)
    
    print(f"Shared CV splits created with {CONFIG['n_folds']} folds")
    print(f"Class distribution: {class_counts[0]} low-risk, {class_counts[1]} high-risk")
    print(f"Class weights: Low={class_weights[0]:.3f}, High={class_weights[1]:.3f}")
    
    # Results storage
    all_foundation_results = []
    
    # Test each foundation model with IDENTICAL CV splits and initialization
    for model_idx, feature_name in enumerate(FEATURE_EXTRACTORS.keys()):
        print(f"\n{'#'*80}")
        print(f"TESTING FOUNDATION MODEL {model_idx+1}/{len(FEATURE_EXTRACTORS)}: {feature_name.upper()}")
        print(f"{'#'*80}")
        
        # Update config for current foundation model
        current_config = CONFIG.copy()
        current_config["feature_extractor"] = feature_name
        current_config["input_dim"] = FEATURE_EXTRACTORS[feature_name]["input_dim"]
        
        # Create results directory
        results_dir = setup_cv_directories(feature_name)
        
        # Load dataset for this foundation model
        feature_config = FEATURE_EXTRACTORS[feature_name]
        
        # Check label column
        df_check = pd.read_csv(feature_config["combined_csv"])
        if 'RSHigh' in df_check.columns:
            label_column = 'RSHigh'
        else:
            label_column = 'RS'
        
        full_dataset = ClassificationMILDataset(
            feature_config["combined_csv"],
            label_column=label_column,
            threshold=CONFIG["threshold"]
        )
        
        # Run CV using shared splits
        fold_results = run_cross_validation_experiment(
            current_config, device, shared_cv_splits, full_dataset, class_weights
        )
        
        # Analyze results for this foundation model
        cv_stats, cv_df = analyze_cv_results(fold_results, feature_name, results_dir)
        
        # Store summary for comparison
        summary_row = create_cv_summary_for_comparison(feature_name, cv_stats)
        all_foundation_results.append(summary_row)
    
    # Create final comparison across all foundation models
    print(f"\n{'='*100}")
    print(f"PHASE 1 FINAL RESULTS: ALL FOUNDATION MODELS COMPARISON - CLASSIFICATION")
    print(f"{'='*100}")
    
    comparison_df = pd.DataFrame(all_foundation_results)
    
    # Sort by mean AUROC (primary metric)
    comparison_df_sorted = comparison_df.sort_values('AUROC_mean', ascending=False)
    
    # Print comparison table
    print(f"{'Model':<12} {'AUROC':<15} {'F1':<15} {'Sensitivity':<15} {'Specificity':<15} {'MCC':<15}")
    print(f"{'-'*105}")
    
    for _, row in comparison_df_sorted.iterrows():
        print(f"{row['Model']:<12} "
              f"{row['AUROC_mean']:.3f}±{row['AUROC_std']:.3f}    "
              f"{row['F1_Score_mean']:.3f}±{row['F1_Score_std']:.3f}      "
              f"{row['Sensitivity_mean']:.3f}±{row['Sensitivity_std']:.3f}    "
              f"{row['Specificity_mean']:.3f}±{row['Specificity_std']:.3f}    "
              f"{row['MCC_mean']:.3f}±{row['MCC_std']:.3f}")
    
    # Identify top performers for Phase 2
    print(f"\n{'='*60}")
    print(f"TOP FOUNDATION MODELS FOR PHASE 2:")
    print(f"{'='*60}")
    
    top_4 = comparison_df_sorted.head(4)
    for i, (_, row) in enumerate(top_4.iterrows(), 1):
        print(f"{i}. {row['Model'].upper()}: "
              f"AUROC = {row['AUROC_mean']:.4f} ± {row['AUROC_std']:.4f}, "
              f"F1 = {row['F1_Score_mean']:.4f} ± {row['F1_Score_std']:.4f}, "
              f"MCC = {row['MCC_mean']:.4f} ± {row['MCC_std']:.4f}")
    
    # Save complete comparison
    os.makedirs("results_phase1_classification_cv", exist_ok=True)
    comparison_df_sorted.to_csv("results_phase1_classification_cv/foundation_models_cv_comparison.csv", index=False)
    
    # Create comparison visualizations
    create_comparison_plot(comparison_df, "results_phase1_classification_cv")
    
    print(f"\nComplete Phase 1 results saved to: results_phase1_classification_cv/")
    print(f"  - foundation_models_cv_comparison.csv")
    print(f"  - foundation_models_comparison.png")
    
    return comparison_df_sorted

def main():
    """
    Main function - can run single foundation model or all foundation models
    """
    
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        # Single model mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(CONFIG["random_seed"])
        
        print(f"Running Phase 1 CV for single model: {CONFIG['feature_extractor']}")
        
        current_config = CONFIG.copy()
        current_config["input_dim"] = FEATURE_EXTRACTORS[CONFIG["feature_extractor"]]["input_dim"]
        
        # Create shared CV splits for single model test
        feature_config = FEATURE_EXTRACTORS[CONFIG["feature_extractor"]]
        full_dataset, shared_cv_splits, labels = create_cv_dataset_and_splits(
            feature_config["combined_csv"], 
            CONFIG["n_folds"], 
            CONFIG["random_seed"],
            CONFIG["threshold"]
        )
        
        # Calculate class weights
        class_counts = np.bincount(labels)
        class_weights = len(labels) / (len(class_counts) * class_counts)
        
        fold_results = run_cross_validation_experiment(
            current_config, device, shared_cv_splits, full_dataset, class_weights
        )
        
        results_dir = setup_cv_directories(CONFIG["feature_extractor"])
        cv_stats, cv_df = analyze_cv_results(fold_results, CONFIG["feature_extractor"], results_dir)
        
        print(f"\nSingle model CV completed for {CONFIG['feature_extractor']}!")
        
    else:
        # Run complete Phase 1 comparison
        print("Running complete Phase 1: All foundation models with identical conditions - CLASSIFICATION")
        final_results = run_phase1_foundation_comparison()
        print(f"\nPhase 1 completed! {len(final_results)} foundation models tested for classification.")

if __name__ == "__main__":
    main()