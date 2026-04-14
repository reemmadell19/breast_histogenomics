# train_reg_cv.py
# 5-Fold Cross-Validation Training Script for Phase 1 Foundation Model Comparison
# FIXED: Same seed and shared CV splits for fair foundation model comparison

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

# Project imports
from datasets.regression_mil_dataset import RegressionMILDataset, create_rs_weighted_sampler
from models.regression_model import MeanPoolingMIL
from utils.mil_utils import mil_collate_fn
from utils.regression_evaluator import RegressionEvaluator
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

# Configuration for Phase 1: Foundation Model Comparison
CONFIG = {
    "feature_extractor": "h-optimus",  # CHANGE THIS FOR EACH FOUNDATION MODEL
    "mil_pooling": "mean",  # Fixed for Phase 1
    "experiment_type": "phase1_cv", 
    "hidden_dim": 128,
    "batch_size": 1,
    "lr": 1e-4,
    "num_epochs": 15,
    "n_folds": 5,
    "use_class_balancing": True,
    "use_boundary_weighting": False,  
    "loss_type": "huber",
    "huber_delta": 5.0,
    "random_seed": 42,
    "stratify_threshold": 25.0
}

def setup_cv_directories(feature_extractor: str) -> str:
    """Create results directory for CV experiments."""
    results_dir = f"results_regression_phase1_cv/{feature_extractor}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def create_cv_dataset_and_splits(combined_csv: str, n_folds: int, random_seed: int, 
                               stratify_threshold: float = 25.0):
    """
    Create dataset and stratified K-fold splits for cross-validation
    """
    # Load combined dataset (train + val combined for CV)
    full_dataset = RegressionMILDataset(combined_csv)
    
    # Extract RS scores for stratification
    rs_scores = []
    for i in range(len(full_dataset)):
        _, rs_score = full_dataset[i]
        rs_scores.append(rs_score)
    
    rs_scores = np.array(rs_scores)
    
    # Create binary labels for stratification (RS < 25 vs RS >= 25)
    binary_labels = (rs_scores >= stratify_threshold).astype(int)
    
    # Create stratified K-fold splits
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    cv_splits = list(skf.split(range(len(full_dataset)), binary_labels))
    
    return full_dataset, cv_splits, binary_labels

def create_fold_dataloaders(full_dataset, train_indices, val_indices, batch_size: int,
                          use_class_balancing: bool = True):
    """Create dataloaders for a specific CV fold"""
    
    # Create subset datasets
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    # Create dataloaders
    if use_class_balancing:
        # Extract RS scores for weighted sampling
        train_rs_scores = []
        for idx in train_indices:
            _, rs_score = full_dataset[idx]
            train_rs_scores.append(rs_score)
        
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
        
        sampler = create_rs_weighted_sampler(
            temp_train_dataset, 
            boundary_focus=False,  # Keep simple for Phase 1
            class_balance=True, 
            threshold=25.0
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

def train_single_fold(fold_num: int, train_loader, val_loader, experiment_config, device):
    """Train and validate a single CV fold"""
    
    # FIXED: Use SAME seed for all foundation models
    set_seed(experiment_config["random_seed"])  # Always use base seed (e.g., 42)
    
    # CRITICAL: Create fresh model and optimizer for each fold
    model = MeanPoolingMIL(
        input_dim=experiment_config["input_dim"], 
        hidden_dim=experiment_config["hidden_dim"]
    ).to(device)
    
    # Fresh loss function and optimizer
    if experiment_config["loss_type"] == "huber":
        criterion = nn.HuberLoss(delta=experiment_config["huber_delta"])
    elif experiment_config["loss_type"] == "mse":
        criterion = nn.MSELoss()
    elif experiment_config["loss_type"] == "mae":
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss type: {experiment_config['loss_type']}")
    
    optimizer = optim.Adam(model.parameters(), lr=experiment_config["lr"])
    
    # Initialize fresh evaluators for this fold
    train_evaluator = RegressionEvaluator()
    val_evaluator = RegressionEvaluator()
    
    # Track the PRIMARY metric - AUROC for consistency across phases
    best_val_auroc = 0.0  # Changed from best_val_r2
    best_fold_metrics = None
    all_epoch_metrics = []  # Track all epochs for analysis
    
    # Training loop for this fold
    for epoch in range(1, experiment_config["num_epochs"] + 1):
        # Reset evaluators each epoch
        train_evaluator.reset()
        val_evaluator.reset()
        
        # Training epoch
        model.train()
        
        for features, rs_target in train_loader:
            # Process batch data
            if isinstance(features, list):
                features = features[0] if len(features) == 1 else torch.stack(features)
            features = features.to(device)
            
            if isinstance(rs_target, list):
                rs_target = rs_target[0] if len(rs_target) == 1 else rs_target
            if not isinstance(rs_target, torch.Tensor):
                rs_target = torch.tensor([rs_target], dtype=torch.float32, device=device)
            else:
                rs_target = rs_target.to(device)
                if rs_target.dim() == 0:
                    rs_target = rs_target.unsqueeze(0)
            
            optimizer.zero_grad()
            
            # Forward pass
            prediction = model(features)
            if prediction.dim() == 0:
                prediction = prediction.unsqueeze(0)
                
            loss = criterion(prediction, rs_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_evaluator.update(
                targets=rs_target.cpu().numpy(),
                preds=prediction.detach().cpu().numpy()
            )
        
        train_metrics = train_evaluator.compute_all_metrics(verbose=False)
        
        # Validation epoch
        model.eval()
        
        with torch.no_grad():
            for features, rs_target in val_loader:
                # Process batch data
                if isinstance(features, list):
                    features = features[0] if len(features) == 1 else torch.stack(features)
                features = features.to(device)
                
                if isinstance(rs_target, list):
                    rs_target = rs_target[0] if len(rs_target) == 1 else rs_target
                if not isinstance(rs_target, torch.Tensor):
                    rs_target = torch.tensor([rs_target], dtype=torch.float32, device=device)
                else:
                    rs_target = rs_target.to(device)
                    if rs_target.dim() == 0:
                        rs_target = rs_target.unsqueeze(0)
                
                # Forward pass
                prediction = model(features)
                if prediction.dim() == 0:
                    prediction = prediction.unsqueeze(0)
                    
                loss = criterion(prediction, rs_target)
                
                # Update metrics
                val_evaluator.update(
                    targets=rs_target.cpu().numpy(),
                    preds=prediction.cpu().numpy()
                )
        
        val_metrics = val_evaluator.compute_all_metrics(verbose=False)
        all_epoch_metrics.append(val_metrics.copy())
        
        # Use AUROC as selection criterion for consistency with other phases
        current_auroc = val_metrics.get('auroc', 0.0)
        if current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            best_fold_metrics = val_metrics.copy()
            best_fold_metrics['best_epoch'] = epoch  # Track which epoch was best
        
        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == experiment_config["num_epochs"]:
            print(f"  Epoch {epoch}: AUROC={val_metrics.get('auroc', 0):.3f}, "
                  f"R²={val_metrics.get('r2', 0):.3f}, "
                  f"F1={val_metrics.get('f1_score', 0):.3f}, "
                  f"Spearman={val_metrics.get('spearman_correlation', 0):.3f}")
    
    # Add final epoch metrics for comparison
    if best_fold_metrics is not None:
        best_fold_metrics['final_auroc'] = all_epoch_metrics[-1]['auroc']
        best_fold_metrics['final_r2'] = all_epoch_metrics[-1]['r2']
        best_fold_metrics['best_auroc'] = best_val_auroc
        
        # Also track R² at best AUROC epoch for analysis
        best_fold_metrics['r2_at_best_auroc'] = best_fold_metrics['r2']
    else:
        # If no improvement was found, use final epoch
        best_fold_metrics = all_epoch_metrics[-1]
        best_fold_metrics['best_epoch'] = experiment_config["num_epochs"]
        best_fold_metrics['best_auroc'] = best_fold_metrics['auroc']
    
    return best_fold_metrics

def run_cross_validation_experiment(experiment_config, device, shared_cv_splits, full_dataset):
    """Run complete cross-validation experiment for one foundation model using shared splits"""
    
    feature_name = experiment_config["feature_extractor"]
    
    print(f"\n{'='*80}")
    print(f"TESTING FOUNDATION MODEL: {feature_name.upper()}")
    print(f"{'='*80}")
    print(f"Input Dimension: {experiment_config['input_dim']}")
    print(f"Cross-Validation: {experiment_config['n_folds']}-fold stratified")
    print(f"MIL Architecture: Mean Pooling (baseline)")
    print(f"Loss Function: {experiment_config['loss_type']}")
    print(f"SAME INITIALIZATION SEED: {experiment_config['random_seed']} for all foundation models")
    print(f"Using shared CV splits for fair foundation model comparison")
    
    print(f"Total samples: {len(full_dataset)}")
    
    # Store results for each fold
    fold_results = []
    
    # Run cross-validation using shared splits
    for fold, (train_idx, val_idx) in enumerate(shared_cv_splits):
        print(f"\nFold {fold+1}/{experiment_config['n_folds']} - {feature_name}")
        print(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        print(f"  Using shared CV split and seed: {experiment_config['random_seed']}")
        
        # Create fold-specific dataloaders
        train_loader, val_loader = create_fold_dataloaders(
            full_dataset, train_idx, val_idx, experiment_config["batch_size"],
            experiment_config["use_class_balancing"]
        )
        
        # Train this fold (uses same seed inside train_single_fold)
        fold_metrics = train_single_fold(
            fold+1, train_loader, val_loader, experiment_config, device
        )
        
        # Store results with metadata
        fold_metrics['fold'] = fold + 1
        fold_metrics['train_size'] = len(train_idx)
        fold_metrics['val_size'] = len(val_idx)
        fold_metrics['foundation_model'] = feature_name
        fold_metrics['initialization_seed'] = experiment_config['random_seed']  # Same for all
        
        # Print fold summary with ALL key metrics including F1
        print(f"  Fold {fold+1} Results: AUROC={fold_metrics.get('auroc', 0):.4f}, "
              f"R²={fold_metrics.get('r2', 0):.4f}, "
              f"RMSE={fold_metrics.get('rmse', 0):.4f}, "
              f"Spearman={fold_metrics.get('spearman_correlation', 0):.4f}, "
              f"F1={fold_metrics.get('f1_score', 0):.4f}, "
              f"Binary_Acc={fold_metrics.get('binary_accuracy', 0):.4f}")
        
        fold_results.append(fold_metrics)
    
    return fold_results

def analyze_cv_results(fold_results, feature_name, results_dir):
    """Analyze and summarize cross-validation results"""
    
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION RESULTS SUMMARY: {feature_name.upper()}")
    print(f"{'='*80}")
    
    # Convert to DataFrame for easy analysis
    cv_df = pd.DataFrame(fold_results)
    
    # Key metrics to analyze (including F1 and spearman p-value)
    key_metrics = ['auroc', 'rmse', 'mae', 'r2', 'spearman_correlation', 'spearman_p_value', 
                   'binary_accuracy', 'f1_score', 'boundary_mae']
    
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
    
    # Performance summary with all key metrics
    print(f"\n{'='*60}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Primary Metric (AUROC): {cv_stats.get('auroc', {}).get('mean', 0):.4f} ± {cv_stats.get('auroc', {}).get('std', 0):.4f}")
    print(f"Regression Performance (R²): {cv_stats.get('r2', {}).get('mean', 0):.4f} ± {cv_stats.get('r2', {}).get('std', 0):.4f}")
    print(f"Error Magnitude (RMSE): {cv_stats.get('rmse', {}).get('mean', 0):.4f} ± {cv_stats.get('rmse', {}).get('std', 0):.4f}")
    print(f"Spearman Correlation: {cv_stats.get('spearman_correlation', {}).get('mean', 0):.4f} ± {cv_stats.get('spearman_correlation', {}).get('std', 0):.4f}")
    print(f"Spearman p-value: {cv_stats.get('spearman_p_value', {}).get('mean', 1):.6f} ± {cv_stats.get('spearman_p_value', {}).get('std', 0):.6f}")
    print(f"F1 Score: {cv_stats.get('f1_score', {}).get('mean', 0):.4f} ± {cv_stats.get('f1_score', {}).get('std', 0):.4f}")
    print(f"Clinical Decision (Binary Acc): {cv_stats.get('binary_accuracy', {}).get('mean', 0):.4f} ± {cv_stats.get('binary_accuracy', {}).get('std', 0):.4f}")
    print(f"Boundary Performance (MAE): {cv_stats.get('boundary_mae', {}).get('mean', 0):.4f} ± {cv_stats.get('boundary_mae', {}).get('std', 0):.4f}")
    
    # Model stability assessment
    auroc_cv = cv_stats.get('auroc', {}).get('cv', 1.0)
    stability = "Highly Stable" if auroc_cv < 0.02 else "Moderately Stable" if auroc_cv < 0.05 else "Variable"
    print(f"Model Stability (AUROC CV): {auroc_cv:.4f} - {stability}")
    
    # Check for suspicious patterns (systematic bias detection)
    if 'auroc' in cv_stats:
        auroc_values = cv_df['auroc'].values
        fold_numbers = cv_df['fold'].values
        correlation = np.corrcoef(fold_numbers, auroc_values)[0, 1]
        if abs(correlation) > 0.7:
            print(f"WARNING: Strong correlation between fold number and AUROC: {correlation:.3f}")
            print("This may indicate systematic bias in CV splits or training process.")
    
    # Statistical significance check
    spearman_p_mean = cv_stats.get('spearman_p_value', {}).get('mean', 1.0)
    if spearman_p_mean < 0.001:
        significance = "Highly Significant (p < 0.001)"
    elif spearman_p_mean < 0.01:
        significance = "Significant (p < 0.01)"
    elif spearman_p_mean < 0.05:
        significance = "Significant (p < 0.05)"
    else:
        significance = "Not Significant (p ≥ 0.05)"
    print(f"Statistical Significance: {significance}")
    
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
        'RMSE_mean': safe_get_metric(cv_stats, 'rmse', 'mean'),
        'RMSE_std': safe_get_metric(cv_stats, 'rmse', 'std'),
        'R2_mean': safe_get_metric(cv_stats, 'r2', 'mean'),
        'R2_std': safe_get_metric(cv_stats, 'r2', 'std'),
        'Spearman_mean': safe_get_metric(cv_stats, 'spearman_correlation', 'mean'),
        'Spearman_std': safe_get_metric(cv_stats, 'spearman_correlation', 'std'),
        'Spearman_pvalue_mean': safe_get_metric(cv_stats, 'spearman_p_value', 'mean'),
        'Spearman_pvalue_std': safe_get_metric(cv_stats, 'spearman_p_value', 'std'),
        'Binary_Accuracy_mean': safe_get_metric(cv_stats, 'binary_accuracy', 'mean'),
        'Binary_Accuracy_std': safe_get_metric(cv_stats, 'binary_accuracy', 'std'),
        'F1_Score_mean': safe_get_metric(cv_stats, 'f1_score', 'mean'),
        'F1_Score_std': safe_get_metric(cv_stats, 'f1_score', 'std'),
        'Boundary_MAE_mean': safe_get_metric(cv_stats, 'boundary_mae', 'mean'),
        'Boundary_MAE_std': safe_get_metric(cv_stats, 'boundary_mae', 'std')
    }

def run_phase1_foundation_comparison():
    """
    Run complete Phase 1 comparison across all foundation models using SAME conditions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(CONFIG["random_seed"])
    
    print(f"{'='*80}")
    print(f"PHASE 1: FOUNDATION MODEL COMPARISON WITH IDENTICAL CONDITIONS")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Configuration: Mean Pooling + {CONFIG['loss_type'].upper()} Loss + Class Balancing")
    print(f"Foundation Models: {list(FEATURE_EXTRACTORS.keys())}")
    print(f"SAME CV splits and initialization for all foundation models")
    
    # CRITICAL: Create CV splits ONCE and use for ALL foundation models
    # This ensures truly fair comparison - only the features differ
    print(f"Creating ONE set of CV splits for ALL foundation models (seed: {CONFIG['random_seed']})")
    
    # Use first foundation model to create the splits (they all have same samples anyway)
    first_model = list(FEATURE_EXTRACTORS.keys())[0]
    first_csv = FEATURE_EXTRACTORS[first_model]["combined_csv"]
    
    _, shared_cv_splits, shared_binary_labels = create_cv_dataset_and_splits(
        first_csv, 
        CONFIG["n_folds"], 
        CONFIG["random_seed"],  # SAME seed for CV splits
        CONFIG["stratify_threshold"]
    )
    
    print(f"Shared CV splits created with {CONFIG['n_folds']} folds")
    print(f"Class distribution: {np.sum(shared_binary_labels == 0)} low-risk, {np.sum(shared_binary_labels == 1)} high-risk")
    
    # Results storage
    all_foundation_results = []
    
    # Test each foundation model with IDENTICAL CV splits and initialization
    for model_idx, feature_name in enumerate(FEATURE_EXTRACTORS.keys()):
        print(f"\n{'#'*80}")
        print(f"TESTING FOUNDATION MODEL {model_idx+1}/{len(FEATURE_EXTRACTORS)}: {feature_name.upper()}")
        print(f"{'#'*80}")
        print(f"Using SAME CV splits and initialization as all other foundation models")
        
        # Update config for current foundation model
        current_config = CONFIG.copy()
        current_config["feature_extractor"] = feature_name
        current_config["input_dim"] = FEATURE_EXTRACTORS[feature_name]["input_dim"]
        
        # Create results directory
        results_dir = setup_cv_directories(feature_name)
        
        # Load dataset for this foundation model
        feature_config = FEATURE_EXTRACTORS[feature_name]
        full_dataset = RegressionMILDataset(feature_config["combined_csv"])
        
        # Run CV using shared splits
        fold_results = run_cross_validation_experiment(current_config, device, shared_cv_splits, full_dataset)
        
        # Analyze results for this foundation model
        cv_stats, cv_df = analyze_cv_results(fold_results, feature_name, results_dir)
        
        # Store summary for comparison
        summary_row = create_cv_summary_for_comparison(feature_name, cv_stats)
        all_foundation_results.append(summary_row)
    
    # Create final comparison across all foundation models
    print(f"\n{'='*100}")
    print(f"PHASE 1 FINAL RESULTS: ALL FOUNDATION MODELS COMPARISON")
    print(f"{'='*100}")
    
    comparison_df = pd.DataFrame(all_foundation_results)
    
    # Sort by mean AUROC (primary clinical metric)
    comparison_df_sorted = comparison_df.sort_values('AUROC_mean', ascending=False)
    
    # Print comparison table with F1 included
    print(f"{'Model':<12} {'AUROC':<15} {'R²':<15} {'F1':<15} {'Spearman':<15} {'Binary Acc':<15}")
    print(f"{'-'*105}")
    
    for _, row in comparison_df_sorted.iterrows():
        print(f"{row['Model']:<12} "
              f"{row['AUROC_mean']:.3f}±{row['AUROC_std']:.3f}    "
              f"{row['R2_mean']:.3f}±{row['R2_std']:.3f}      "
              f"{row['F1_Score_mean']:.3f}±{row['F1_Score_std']:.3f}      "
              f"{row['Spearman_mean']:.3f}±{row['Spearman_std']:.3f}    "
              f"{row['Binary_Accuracy_mean']:.3f}±{row['Binary_Accuracy_std']:.3f}")
    
    # Identify top performers for Phase 2
    print(f"\n{'='*60}")
    print(f"TOP FOUNDATION MODELS FOR PHASE 2:")
    print(f"{'='*60}")
    
    top_4 = comparison_df_sorted.head(4)
    for i, (_, row) in enumerate(top_4.iterrows(), 1):
        significance = "***" if row['Spearman_pvalue_mean'] < 0.001 else "**" if row['Spearman_pvalue_mean'] < 0.01 else "*" if row['Spearman_pvalue_mean'] < 0.05 else ""
        print(f"{i}. {row['Model'].upper()}: "
              f"AUROC = {row['AUROC_mean']:.4f} ± {row['AUROC_std']:.4f}, "
              f"F1 = {row['F1_Score_mean']:.4f} ± {row['F1_Score_std']:.4f}, "
              f"Spearman = {row['Spearman_mean']:.4f}{significance}")
    
    # Statistical significance summary
    significant_models = comparison_df_sorted[comparison_df_sorted['Spearman_pvalue_mean'] < 0.05]
    print(f"\n{len(significant_models)}/{len(comparison_df_sorted)} models show significant Spearman correlation (p < 0.05)")
    
    # Save complete comparison
    os.makedirs("results_phase1_cv", exist_ok=True)
    comparison_df_sorted.to_csv("results_phase1_cv/foundation_models_cv_comparison.csv", index=False)
    
    print(f"\nComplete Phase 1 results saved to: results_phase1_cv/foundation_models_cv_comparison.csv")
    
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
        _, shared_cv_splits, _ = create_cv_dataset_and_splits(
            feature_config["combined_csv"], 
            CONFIG["n_folds"], 
            CONFIG["random_seed"],
            CONFIG["stratify_threshold"]
        )
        
        # Load dataset
        full_dataset = RegressionMILDataset(feature_config["combined_csv"])
        
        fold_results = run_cross_validation_experiment(current_config, device, shared_cv_splits, full_dataset)
        
        results_dir = setup_cv_directories(CONFIG["feature_extractor"])
        cv_stats, cv_df = analyze_cv_results(fold_results, CONFIG["feature_extractor"], results_dir)
        
        print(f"\nSingle model CV completed for {CONFIG['feature_extractor']}!")
        
    else:
        # Run complete Phase 1 comparison
        print("Running complete Phase 1: All foundation models with identical conditions")
        final_results = run_phase1_foundation_comparison()
        print(f"\nPhase 1 completed! {len(final_results)} foundation models tested.")

if __name__ == "__main__":
    main()