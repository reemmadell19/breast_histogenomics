# train_cls_phase2_complete.py
# Complete Phase 2 Classification: All Foundation Models × All MIL Architectures
# Unified script including Mean, Max, Attention, CLAM, ACMIL, and ACMIL_CLAM_Hybrid

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
from datetime import datetime

# Project imports
from datasets.classification_mil_dataset import ClassificationMILDataset, create_classification_weighted_sampler
from models.classification_model import (
    MeanPoolingMILClassifier, MaxPoolingMILClassifier, 
    AttentionMILClassifier, CLAMClassifier,
    ACMILClassifier, ACMIL_CLAM_HybridClassifier
)
from utils.mil_utils import mil_collate_fn
from utils.classification_evaluator import ClassificationEvaluator
from utils.training_helpers import set_seed

# ALL foundation models to test
SELECTED_MODELS = {
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

# Complete MIL architecture configurations
MIL_ARCHITECTURES = {
    "mean": {
        "class": MeanPoolingMILClassifier,
        "name": "Mean Pooling",
        "description": "Simple mean pooling aggregation",
        "params": {}
    },
    "max": {
        "class": MaxPoolingMILClassifier,
        "name": "Max Pooling", 
        "description": "Max pooling aggregation",
        "params": {}
    },
    "attention": {
        "class": AttentionMILClassifier,
        "name": "Attention MIL",
        "description": "Attention-based aggregation",
        "params": {"attention_hidden_dim": 64}
    },
    "clam": {
        "class": CLAMClassifier,
        "name": "CLAM",
        "description": "Clustering-constrained Attention MIL",
        "params": {"attention_hidden_dim": 256, "gate": True}
    },
    "acmil": {
        "class": ACMILClassifier,
        "name": "ACMIL",
        "description": "Attention-Challenging Multiple Instance Learning",
        "params": {
            "n_branches": 5,
            "n_masked_patch": 10,
            "mask_ratio": 0.6,  # Can be adjusted based on validation
            "dropout": 0.25
        }
    },
    "acmil_clam": {
        "class": ACMIL_CLAM_HybridClassifier,
        "name": "ACMIL-CLAM Hybrid",
        "description": "ACMIL with CLAM-style gated attention",
        "params": {
            "n_branches": 5,
            "n_masked_patch": 10,
            "mask_ratio": 0.6,
            "attention_hidden_dim": 256,
            "gate": True,
            "dropout": 0.25
        }
    }
}

# Configuration for Phase 2: MIL Architecture Comparison
CONFIG = {
    "experiment_type": "phase2_cv_classification",
    "hidden_dim": 128,
    "n_classes": 2,
    "batch_size": 1,
    "lr": 1e-4,
    "num_epochs": 15,
    "n_folds": 5,
    "use_class_balancing": True,  # Batch balancing
    "use_class_weights": False,    # No weighted loss (as per your preference)
    "loss_type": "ce",
    "random_seed": 42,
    "threshold": 25.0
}

def setup_phase2_cv_directories(feature_extractor: str, mil_architecture: str) -> str:
    """Create hierarchical results directory for Phase 2 CV experiments."""
    results_dir = f"results_phase2_classification_cv/{feature_extractor}/{mil_architecture}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def create_cv_dataset_and_splits(combined_csv: str, n_folds: int, random_seed: int, threshold: float = 25.0):
    """Create dataset and stratified K-fold splits for cross-validation"""
    
    # Check if RSHigh column exists
    df_check = pd.read_csv(combined_csv)
    if 'RSHigh' in df_check.columns:
        label_column = 'RSHigh'
    elif 'RS' in df_check.columns:
        label_column = 'RS'
    else:
        raise ValueError("Neither RSHigh nor RS column found in dataset")
    
    # Load dataset
    full_dataset = ClassificationMILDataset(combined_csv, label_column=label_column, threshold=threshold)
    
    # Extract labels for stratification
    labels = []
    for i in range(len(full_dataset)):
        _, label = full_dataset[i]
        labels.append(label)
    
    labels = np.array(labels)
    
    # Create stratified K-fold splits
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

def create_mil_model(mil_architecture: str, input_dim: int, config: dict, device):
    """Create MIL model based on architecture specification"""
    
    mil_config = MIL_ARCHITECTURES[mil_architecture]
    mil_class = mil_config["class"]
    
    # Base parameters
    model_params = {
        "input_dim": input_dim,
        "hidden_dim": config["hidden_dim"],
        "n_classes": config["n_classes"]
    }
    
    # Add architecture-specific parameters
    model_params.update(mil_config["params"])
    
    # Create model
    model = mil_class(**model_params).to(device)
    
    return model, mil_config

def train_single_fold(fold_num: int, train_loader, val_loader, experiment_config, 
                     mil_architecture: str, device):
    """Train and validate a single CV fold with specified MIL architecture"""
    
    # Set seed for reproducibility
    set_seed(experiment_config["random_seed"])
    
    # Create model
    model, mil_config = create_mil_model(
        mil_architecture,
        experiment_config["input_dim"],
        experiment_config,
        device
    )
    
    # Loss function (no class weights as requested)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=experiment_config["lr"])
    
    # Initialize evaluators
    train_evaluator = ClassificationEvaluator(n_classes=experiment_config["n_classes"])
    val_evaluator = ClassificationEvaluator(n_classes=experiment_config["n_classes"])
    
    # Track metrics
    best_val_auroc = 0.0
    best_fold_metrics = None
    all_epoch_metrics = []
    
    # Training loop
    for epoch in range(1, experiment_config["num_epochs"] + 1):
        # Reset evaluators
        train_evaluator.reset()
        val_evaluator.reset()
        
        # Training
        model.train()
        for features, label in train_loader:
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
            
            optimizer.zero_grad()
            
            # Forward pass
            if mil_architecture in ["acmil", "acmil_clam"]:
                logits = model(features, return_branch_outputs=False)
            else:
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
        
        # Validation
        model.eval()
        with torch.no_grad():
            for features, label in val_loader:
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
                if mil_architecture in ["acmil", "acmil_clam"]:
                    logits = model(features, return_branch_outputs=False)
                else:
                    logits = model(features)
                
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                
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
        
        # Track best model
        current_auroc = val_metrics.get('auroc', 0.0)
        if current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            best_fold_metrics = val_metrics.copy()
            best_fold_metrics['best_epoch'] = epoch
    
    # Add final metrics
    if best_fold_metrics:
        best_fold_metrics['final_auroc'] = all_epoch_metrics[-1]['auroc']
        best_fold_metrics['final_f1'] = all_epoch_metrics[-1]['f1_score']
        best_fold_metrics['best_auroc'] = best_val_auroc
    else:
        best_fold_metrics = all_epoch_metrics[-1]
        best_fold_metrics['best_epoch'] = experiment_config["num_epochs"]
        best_fold_metrics['best_auroc'] = best_fold_metrics['auroc']
    
    return best_fold_metrics
def check_if_results_exist(feature_name: str, mil_arch: str, results_dir: str) -> bool:
    """Check if CV results already exist for this combination"""
    results_file = os.path.join(results_dir, f"{feature_name}_{mil_arch}_cv_results.csv")
    if os.path.exists(results_file):
        # Verify the file has content
        try:
            df = pd.read_csv(results_file)
            if len(df) > 0:
                return True
        except:
            return False
    return False

def run_single_model_mil_comparison(feature_name: str, experiment_config, device):
    """Run CV comparison of all MIL architectures for one foundation model"""
    
    model_config = SELECTED_MODELS[feature_name]
    
    print(f"\n{'='*80}")
    print(f"PHASE 2: MIL ARCHITECTURE COMPARISON - CLASSIFICATION")
    print(f"{'='*80}")
    print(f"Foundation Model: {feature_name.upper()}")
    
    # Create CV splits ONCE for fair comparison
    full_dataset, cv_splits, labels = create_cv_dataset_and_splits(
        model_config["combined_csv"], 
        experiment_config["n_folds"], 
        experiment_config["random_seed"],
        experiment_config["threshold"]
    )
    
    # Store results for each MIL architecture
    all_mil_results = []
    
    # Test each MIL architecture
    for mil_arch in MIL_ARCHITECTURES.keys():
        # Create results directory
        results_dir = setup_phase2_cv_directories(feature_name, mil_arch)
        
        # CHECK IF RESULTS ALREADY EXIST
        if check_if_results_exist(feature_name, mil_arch, results_dir):
            print(f"\n{'#'*60}")
            print(f"SKIPPING: {feature_name.upper()} + {MIL_ARCHITECTURES[mil_arch]['name']}")
            print(f"Results already exist in: {results_dir}")
            
            # Load existing results
            existing_results = pd.read_csv(os.path.join(results_dir, f"{feature_name}_{mil_arch}_cv_results.csv"))
            
            # Calculate stats from existing results
            key_metrics = ['auroc', 'auc_pr', 'accuracy', 'f1_score', 'sensitivity', 'specificity', 'mcc']
            cv_stats = {}
            for metric in key_metrics:
                if metric in existing_results.columns:
                    values = existing_results[metric].values
                    cv_stats[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
            
            # Add to summary
            summary_row = create_mil_summary(feature_name, mil_arch, cv_stats)
            all_mil_results.append(summary_row)
            continue  # Skip to next MIL architecture
        
        print(f"\n{'#'*60}")
        print(f"Testing: {MIL_ARCHITECTURES[mil_arch]['name'].upper()}")
        print(f"{'#'*60}")
        
        # Update config
        current_config = experiment_config.copy()
        current_config["mil_architecture"] = mil_arch
        current_config["input_dim"] = model_config["input_dim"]
        
        # Store fold results
        fold_results = []
        
        # Run cross-validation
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"  Fold {fold+1}/{experiment_config['n_folds']}", end=" ")
            
            # Create dataloaders
            train_loader, val_loader = create_fold_dataloaders(
                full_dataset, train_idx, val_idx, 
                experiment_config["batch_size"],
                experiment_config["use_class_balancing"]
            )
            
            # Train fold
            fold_metrics = train_single_fold(
                fold+1, train_loader, val_loader, 
                current_config, mil_arch, device
            )
            
            # Store results
            fold_metrics['fold'] = fold + 1
            fold_metrics['mil_architecture'] = mil_arch
            fold_metrics['foundation_model'] = feature_name
            
            print(f"AUROC={fold_metrics.get('auroc', 0):.3f}, F1={fold_metrics.get('f1_score', 0):.3f}")
            
            fold_results.append(fold_metrics)
        
        # Analyze results
        cv_stats = analyze_cv_results(fold_results, mil_arch, results_dir, feature_name)
        
        # Store summary
        summary_row = create_mil_summary(feature_name, mil_arch, cv_stats)
        all_mil_results.append(summary_row)
    
    return all_mil_results

def analyze_cv_results(fold_results, mil_arch, results_dir, feature_name):
    """Analyze and summarize cross-validation results"""
    
    # Convert to DataFrame
    cv_df = pd.DataFrame(fold_results)
    
    # Key metrics for classification
    key_metrics = ['auroc', 'auc_pr', 'accuracy', 'balanced_accuracy', 
                   'f1_score', 'sensitivity', 'specificity', 'mcc']
    
    # Calculate statistics
    cv_stats = {}
    for metric in key_metrics:
        if metric in cv_df.columns:
            values = cv_df[metric].values
            cv_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    # Print summary
    print(f"\n  Summary for {feature_name.upper()} + {MIL_ARCHITECTURES[mil_arch]['name']}:")
    print(f"    AUROC: {cv_stats.get('auroc', {}).get('mean', 0):.4f} ± {cv_stats.get('auroc', {}).get('std', 0):.4f}")
    print(f"    F1: {cv_stats.get('f1_score', {}).get('mean', 0):.4f} ± {cv_stats.get('f1_score', {}).get('std', 0):.4f}")
    print(f"    Sensitivity: {cv_stats.get('sensitivity', {}).get('mean', 0):.4f} ± {cv_stats.get('sensitivity', {}).get('std', 0):.4f}")
    
    # Save results
    cv_df.to_csv(os.path.join(results_dir, f"{feature_name}_{mil_arch}_cv_results.csv"), index=False)
    
    return cv_stats

def create_mil_summary(feature_name, mil_arch, cv_stats):
    """Create summary row for comparison"""
    
    def safe_get(stats_dict, metric, stat):
        return stats_dict.get(metric, {}).get(stat, 0.0)
    
    return {
        'Foundation_Model': feature_name,
        'MIL_Architecture': mil_arch,
        'MIL_Description': MIL_ARCHITECTURES[mil_arch]['description'],
        'AUROC_mean': safe_get(cv_stats, 'auroc', 'mean'),
        'AUROC_std': safe_get(cv_stats, 'auroc', 'std'),
        'AUC_PR_mean': safe_get(cv_stats, 'auc_pr', 'mean'),
        'AUC_PR_std': safe_get(cv_stats, 'auc_pr', 'std'),
        'F1_Score_mean': safe_get(cv_stats, 'f1_score', 'mean'),
        'F1_Score_std': safe_get(cv_stats, 'f1_score', 'std'),
        'Accuracy_mean': safe_get(cv_stats, 'accuracy', 'mean'),
        'Accuracy_std': safe_get(cv_stats, 'accuracy', 'std'),
        'Balanced_Accuracy_mean': safe_get(cv_stats, 'balanced_accuracy', 'mean'),
        'Balanced_Accuracy_std': safe_get(cv_stats, 'balanced_accuracy', 'std'),
        'Sensitivity_mean': safe_get(cv_stats, 'sensitivity', 'mean'),
        'Sensitivity_std': safe_get(cv_stats, 'sensitivity', 'std'),
        'Specificity_mean': safe_get(cv_stats, 'specificity', 'mean'),
        'Specificity_std': safe_get(cv_stats, 'specificity', 'std'),
        'MCC_mean': safe_get(cv_stats, 'mcc', 'mean'),
        'MCC_std': safe_get(cv_stats, 'mcc', 'std')
    }

def create_visualization(comparison_df, output_dir):
    """Create comprehensive visualization of results"""
    
    # Sort by AUROC
    comparison_df = comparison_df.sort_values('AUROC_mean', ascending=False)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Heatmap of AUROC across all combinations
    ax1 = plt.subplot(2, 3, 1)
    pivot_auroc = comparison_df.pivot(index='MIL_Architecture', 
                                      columns='Foundation_Model', 
                                      values='AUROC_mean')
    sns.heatmap(pivot_auroc, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, ax=ax1)
    ax1.set_title('AUROC Heatmap')
    
    # 2. Heatmap of F1 Score
    ax2 = plt.subplot(2, 3, 2)
    pivot_f1 = comparison_df.pivot(index='MIL_Architecture', 
                                   columns='Foundation_Model', 
                                   values='F1_Score_mean')
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0.3, vmax=0.8, ax=ax2)
    ax2.set_title('F1 Score Heatmap')
    
    # 3. Top 10 combinations bar plot
    ax3 = plt.subplot(2, 3, 3)
    top10 = comparison_df.head(10)
    labels = [f"{row['Foundation_Model'][:4]}+{row['MIL_Architecture'][:4]}" 
              for _, row in top10.iterrows()]
    y_pos = np.arange(len(labels))
    
    ax3.barh(y_pos, top10['AUROC_mean'].values, xerr=top10['AUROC_std'].values,
             color='skyblue', capsize=3)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels)
    ax3.set_xlabel('AUROC')
    ax3.set_title('Top 10 Combinations')
    ax3.set_xlim([0.5, 1.0])
    
    # 4. Sensitivity vs Specificity scatter
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(comparison_df['Specificity_mean'], 
                         comparison_df['Sensitivity_mean'],
                         c=comparison_df['AUROC_mean'], 
                         s=100, cmap='viridis', alpha=0.6)
    
    # Add labels for top 5
    for i, row in comparison_df.head(5).iterrows():
        ax4.annotate(f"{row['Foundation_Model'][:3]}+{row['MIL_Architecture'][:3]}",
                    (row['Specificity_mean'], row['Sensitivity_mean']),
                    fontsize=8)
    
    ax4.set_xlabel('Specificity')
    ax4.set_ylabel('Sensitivity')
    ax4.set_title('Sensitivity vs Specificity')
    plt.colorbar(scatter, ax=ax4, label='AUROC')
    
    # 5. MIL Architecture comparison
    ax5 = plt.subplot(2, 3, 5)
    mil_avg = comparison_df.groupby('MIL_Architecture')['AUROC_mean'].agg(['mean', 'std'])
    mil_avg = mil_avg.sort_values('mean', ascending=False)
    
    ax5.bar(range(len(mil_avg)), mil_avg['mean'], yerr=mil_avg['std'], 
            capsize=5, color='lightcoral')
    ax5.set_xticks(range(len(mil_avg)))
    ax5.set_xticklabels(mil_avg.index, rotation=45)
    ax5.set_ylabel('Average AUROC')
    ax5.set_title('MIL Architecture Performance')
    ax5.grid(True, alpha=0.3)
    
    # 6. Foundation Model comparison
    ax6 = plt.subplot(2, 3, 6)
    model_avg = comparison_df.groupby('Foundation_Model')['AUROC_mean'].agg(['mean', 'std'])
    model_avg = model_avg.sort_values('mean', ascending=False)
    
    ax6.bar(range(len(model_avg)), model_avg['mean'], yerr=model_avg['std'], 
            capsize=5, color='lightgreen')
    ax6.set_xticks(range(len(model_avg)))
    ax6.set_xticklabels(model_avg.index, rotation=45)
    ax6.set_ylabel('Average AUROC')
    ax6.set_title('Foundation Model Performance')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Phase 2 Classification: Complete Results Analysis', fontsize=16, weight='bold')
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join(output_dir, 'phase2_complete_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def run_complete_phase2_comparison():
    """Run complete Phase 2: all models × all MIL architectures"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(CONFIG["random_seed"])
    
    print(f"{'='*100}")
    print(f"PHASE 2: COMPLETE FOUNDATION MODEL × MIL ARCHITECTURE COMPARISON - CLASSIFICATION")
    print(f"{'='*100}")
    print(f"Device: {device}")
    print(f"Foundation Models: {list(SELECTED_MODELS.keys())}")
    print(f"MIL Architectures: {list(MIL_ARCHITECTURES.keys())}")
    print(f"Total Experiments: {len(SELECTED_MODELS)} × {len(MIL_ARCHITECTURES)} = {len(SELECTED_MODELS) * len(MIL_ARCHITECTURES)}")
    
    # Store all results
    all_phase2_results = []
    
    # Test each foundation model
    for model_idx, feature_name in enumerate(SELECTED_MODELS.keys(), 1):
        print(f"\n{'#'*100}")
        print(f"FOUNDATION MODEL {model_idx}/{len(SELECTED_MODELS)}: {feature_name.upper()}")
        print(f"{'#'*100}")
        
        # Run MIL comparison
        model_results = run_single_model_mil_comparison(feature_name, CONFIG, device)
        all_phase2_results.extend(model_results)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_phase2_results)
    comparison_df_sorted = comparison_df.sort_values('AUROC_mean', ascending=False)
    
    # Save results
    os.makedirs("results_phase2_classification_cv", exist_ok=True)
    comparison_df_sorted.to_csv("results_phase2_classification_cv/phase2_complete_results.csv", index=False)
    
    # Print final results
    print(f"\n{'='*120}")
    print(f"PHASE 2 FINAL RESULTS - CLASSIFICATION")
    print(f"{'='*120}")
    
    print(f"\nTOP 15 COMBINATIONS:")
    print(f"{'Rank':<5} {'Foundation':<12} {'MIL':<15} {'AUROC':<15} {'F1':<15} {'Sensitivity':<15} {'MCC':<15}")
    print(f"{'-'*105}")
    
    for i, (_, row) in enumerate(comparison_df_sorted.head(15).iterrows(), 1):
        print(f"{i:<5} {row['Foundation_Model']:<12} {row['MIL_Architecture']:<15} "
              f"{row['AUROC_mean']:.3f}±{row['AUROC_std']:.3f}   "
              f"{row['F1_Score_mean']:.3f}±{row['F1_Score_std']:.3f}   "
              f"{row['Sensitivity_mean']:.3f}±{row['Sensitivity_std']:.3f}   "
              f"{row['MCC_mean']:.3f}±{row['MCC_std']:.3f}")
    
    # Best MIL per foundation model
    print(f"\n{'='*80}")
    print(f"BEST MIL ARCHITECTURE PER FOUNDATION MODEL:")
    print(f"{'='*80}")
    
    for foundation in SELECTED_MODELS.keys():
        foundation_results = comparison_df[comparison_df['Foundation_Model'] == foundation]
        best = foundation_results.loc[foundation_results['AUROC_mean'].idxmax()]
        print(f"{foundation.upper():<12} → {best['MIL_Architecture']:<15}: "
              f"AUROC={best['AUROC_mean']:.4f}, F1={best['F1_Score_mean']:.4f}, "
              f"Sens={best['Sensitivity_mean']:.4f}")
    
    # Best foundation model per MIL
    print(f"\n{'='*80}")
    print(f"BEST FOUNDATION MODEL PER MIL ARCHITECTURE:")
    print(f"{'='*80}")
    
    for mil in MIL_ARCHITECTURES.keys():
        mil_results = comparison_df[comparison_df['MIL_Architecture'] == mil]
        best = mil_results.loc[mil_results['AUROC_mean'].idxmax()]
        print(f"{mil.upper():<15} → {best['Foundation_Model']:<12}: "
              f"AUROC={best['AUROC_mean']:.4f}, F1={best['F1_Score_mean']:.4f}")
    
    # Create visualizations
    create_visualization(comparison_df, "results_phase2_classification_cv")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: results_phase2_classification_cv/")
    print(f"  - phase2_complete_results.csv")
    print(f"  - phase2_complete_analysis.png")
    
    return comparison_df_sorted

def main():
    """Main function for Phase 2"""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "single_model" and len(sys.argv) > 2:
            # Test single foundation model
            model_name = sys.argv[2]
            if model_name in SELECTED_MODELS:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Running Phase 2 for single model: {model_name}")
                
                results = run_single_model_mil_comparison(model_name, CONFIG, device)
                
                # Save results
                results_df = pd.DataFrame(results)
                os.makedirs("results_phase2_classification_cv", exist_ok=True)
                results_df.to_csv(f"results_phase2_classification_cv/{model_name}_mil_comparison.csv", index=False)
                
                print(f"\nResults for {model_name} saved!")
            else:
                print(f"Model {model_name} not found. Available: {list(SELECTED_MODELS.keys())}")
        else:
            print("Usage: python script.py single_model <model_name>")
    else:
        # Run complete comparison
        print("Running complete Phase 2: All foundation models × All MIL architectures")
        final_results = run_complete_phase2_comparison()
        print(f"\nPhase 2 completed! {len(final_results)} total combinations tested.")

if __name__ == "__main__":
    main()