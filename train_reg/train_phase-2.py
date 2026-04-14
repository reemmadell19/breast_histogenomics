# train_phase2_cv.py
# 5-Fold Cross-Validation Training Script for Phase 2 MIL Architecture Comparison
# FIXED: Independent CV splits for each MIL architecture to prevent systematic bias

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
from models.regression_model import MeanPoolingMIL, MaxPoolingMIL, AttentionMIL, CLAM
from utils.mil_utils import mil_collate_fn
from utils.regression_evaluator import RegressionEvaluator
from utils.training_helpers import set_seed, print_experiment_header

# Selected foundation models from Phase 1 + ResNet-18 baseline
SELECTED_MODELS = {
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

# MIL architecture configurations
MIL_ARCHITECTURES = {
    "mean": {
        "class": MeanPoolingMIL,
        "name": "Mean Pooling",
        "description": "Simple mean pooling aggregation"
    },
    "max": {
        "class": MaxPoolingMIL,
        "name": "Max Pooling", 
        "description": "Max pooling aggregation"
    },
    "attention": {
        "class": AttentionMIL,
        "name": "Attention MIL",
        "description": "Attention-based aggregation"
    },
    "clam": {
        "class": CLAM,
        "name": "CLAM",
        "description": "Clustering-constrained Attention MIL"
    }
}

# Configuration for Phase 2: MIL Architecture Comparison
CONFIG = {
    "feature_extractor": "h-optimus",  # CHANGE THIS FOR EACH FOUNDATION MODEL
    "mil_architecture": "clam",        # CHANGE THIS FOR EACH MIL METHOD
    "experiment_type": "phase2_cv", 
    "hidden_dim": 128,
    "attention_hidden_dim": 128,       # For attention-based models
    "batch_size": 1,
    "lr": 1e-4,
    "num_epochs": 15,
    "n_folds": 5,
    "use_class_balancing": True,
    "use_boundary_weighting": False,  
    "loss_type": "huber",              # Fixed from Phase 1 results
    "huber_delta": 5.0,
    "random_seed": 42,
    "stratify_threshold": 25.0
}

def setup_phase2_cv_directories(feature_extractor: str, mil_architecture: str) -> str:
    """Create hierarchical results directory for Phase 2 CV experiments."""
    results_dir = f"results_phase2_cv/{feature_extractor}/{mil_architecture}"
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
            boundary_focus=False,
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

def create_mil_model(mil_architecture: str, input_dim: int, hidden_dim: int, 
                    attention_hidden_dim: int, device):
    """Create MIL model based on architecture specification"""
    
    mil_config = MIL_ARCHITECTURES[mil_architecture]
    mil_class = mil_config["class"]
    
    # Create model with appropriate parameters
    if mil_architecture in ["mean", "max"]:
        model = mil_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        ).to(device)
    elif mil_architecture == "attention":
        model = mil_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_hidden_dim=attention_hidden_dim
        ).to(device)
    elif mil_architecture == "clam":
        model = mil_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_hidden_dim=attention_hidden_dim
        ).to(device)
    else:
        raise ValueError(f"Unknown MIL architecture: {mil_architecture}")
    
    return model, mil_config

def train_single_fold(fold_num: int, train_loader, val_loader, experiment_config, device):
    """Train and validate a single CV fold with specified MIL architecture"""
    
    # FIXED: Use SAME seed for all MIL architectures and all folds
    set_seed(experiment_config["random_seed"])  # Always use base seed (e.g., 42)
    
    # Create fresh model and optimizer for each fold
    model, mil_config = create_mil_model(
        experiment_config["mil_architecture"],
        experiment_config["input_dim"],
        experiment_config["hidden_dim"],
        experiment_config["attention_hidden_dim"],
        device
    )
    
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
    
    # Track the PRIMARY metric you care about
    best_val_auroc = 0.0  # Or use best_val_r2 if that's your primary metric
    best_fold_metrics = None
    all_epoch_metrics = []  # Optionally track all epochs
    
    # Training loop for this fold with epoch progress bar
    epoch_pbar = tqdm(range(1, experiment_config["num_epochs"] + 1), 
                      desc=f"Fold {fold_num} - Training", 
                      unit="epoch")
    
    for epoch in epoch_pbar:
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
        
        # Use consistent selection criterion across all phases
        current_auroc = val_metrics.get('auroc', 0.0)
        if current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            best_fold_metrics = val_metrics.copy()
            best_fold_metrics['best_epoch'] = epoch  # Track which epoch was best
    
    # Add final epoch metrics for comparison
    best_fold_metrics['final_auroc'] = all_epoch_metrics[-1]['auroc']
    best_fold_metrics['final_r2'] = all_epoch_metrics[-1]['r2']
    
    return best_fold_metrics

def run_single_model_mil_comparison(experiment_config, device):
    """Run CV comparison of all MIL architectures for one foundation model"""
    
    feature_name = experiment_config["feature_extractor"]
    model_config = SELECTED_MODELS[feature_name]
    
    print(f"\n{'='*80}")
    print(f"PHASE 2: MIL ARCHITECTURE COMPARISON")
    print(f"{'='*80}")
    print(f"Foundation Model: {feature_name.upper()}")
    print(f"Input Dimension: {experiment_config['input_dim']}")
    print(f"Cross-Validation: {experiment_config['n_folds']}-fold stratified")
    print(f"Loss Function: {experiment_config['loss_type']}")
    print(f"SAME INITIALIZATION SEED: {experiment_config['random_seed']} for all MIL architectures")
    
    # Create CV splits ONCE for fair comparison across MIL architectures
    print(f"Creating shared CV splits for fair MIL comparison (seed: {experiment_config['random_seed']})")
    
    full_dataset, cv_splits, binary_labels = create_cv_dataset_and_splits(
        model_config["combined_csv"], 
        experiment_config["n_folds"], 
        experiment_config["random_seed"],  # SAME seed for CV splits
        experiment_config["stratify_threshold"]
    )
    
    print(f"Total samples: {len(full_dataset)}")
    print(f"Class distribution: {np.sum(binary_labels == 0)} low-risk, {np.sum(binary_labels == 1)} high-risk")
    
    # Store results for each MIL architecture
    all_mil_results = []
    
    # Test each MIL architecture using the SAME CV splits AND SAME initialization
    for mil_arch in MIL_ARCHITECTURES.keys():
        print(f"\n{'#'*60}")
        print(f"TESTING MIL ARCHITECTURE: {MIL_ARCHITECTURES[mil_arch]['name'].upper()}")
        print(f"{'#'*60}")
        print(f"Using SAME initialization seed for pure architecture comparison")
        
        # Update config for current MIL architecture
        current_config = experiment_config.copy()
        current_config["mil_architecture"] = mil_arch
        
        # Create results directory for this combination
        results_dir = setup_phase2_cv_directories(feature_name, mil_arch)
        
        # Store results for each fold
        fold_results = []
        
        # Run cross-validation for this MIL architecture using SAME splits
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\nFold {fold+1}/{experiment_config['n_folds']} - {mil_arch}")
            print(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
            print(f"  Using seed: {experiment_config['random_seed']} (same for all MIL)")
            
            # Create fold-specific dataloaders (same data splits for all MIL)
            train_loader, val_loader = create_fold_dataloaders(
                full_dataset, train_idx, val_idx, experiment_config["batch_size"],
                experiment_config["use_class_balancing"]
            )
            
            # Train this fold (uses same seed inside train_single_fold)
            fold_metrics = train_single_fold(
                fold+1, train_loader, val_loader, current_config, device
            )
            
            # Store results with metadata
            fold_metrics['fold'] = fold + 1
            fold_metrics['train_size'] = len(train_idx)
            fold_metrics['val_size'] = len(val_idx)
            fold_metrics['mil_architecture'] = mil_arch
            fold_metrics['foundation_model'] = feature_name
            fold_metrics['initialization_seed'] = experiment_config['random_seed']  # Same for all
            
            # Print fold summary with all key metrics
            print(f"  Fold {fold+1} Results: AUROC={fold_metrics.get('auroc', 0):.4f}, "
                  f"R²={fold_metrics.get('r2', 0):.4f}, "
                  f"RMSE={fold_metrics.get('rmse', 0):.4f}, "
                  f"Spearman={fold_metrics.get('spearman_correlation', 0):.4f}, "
                  f"F1={fold_metrics.get('f1_score', 0):.4f}, "
                  f"Binary_Acc={fold_metrics.get('binary_accuracy', 0):.4f}")
            
            fold_results.append(fold_metrics)
        
        # Analyze results for this MIL architecture
        cv_stats, cv_df = analyze_cv_results(fold_results, mil_arch, results_dir, feature_name)
        
        # Store summary for comparison
        summary_row = create_mil_summary_for_comparison(feature_name, mil_arch, cv_stats)
        all_mil_results.append(summary_row)
    
    return all_mil_results

def analyze_cv_results(fold_results, mil_arch, results_dir, feature_name):
    """Analyze and summarize cross-validation results for a MIL architecture"""
    
    print(f"\n{'='*80}")
    print(f"CV RESULTS: {feature_name.upper()} + {MIL_ARCHITECTURES[mil_arch]['name'].upper()}")
    print(f"{'='*80}")
    
    # Convert to DataFrame for easy analysis
    cv_df = pd.DataFrame(fold_results)
    
    # Key metrics to analyze (including spearman p-value)
    key_metrics = ['auroc', 'rmse', 'mae', 'r2', 'spearman_correlation', 'spearman_p_value', 'binary_accuracy', 'f1_score', 'boundary_mae']
    
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
    
    # Print summary
    print(f"AUROC: {cv_stats.get('auroc', {}).get('mean', 0):.4f} ± {cv_stats.get('auroc', {}).get('std', 0):.4f}")
    print(f"R²: {cv_stats.get('r2', {}).get('mean', 0):.4f} ± {cv_stats.get('r2', {}).get('std', 0):.4f}")
    print(f"RMSE: {cv_stats.get('rmse', {}).get('mean', 0):.4f} ± {cv_stats.get('rmse', {}).get('std', 0):.4f}")
    print(f"Spearman Correlation: {cv_stats.get('spearman_correlation', {}).get('mean', 0):.4f} ± {cv_stats.get('spearman_correlation', {}).get('std', 0):.4f}")
    print(f"Spearman p-value: {cv_stats.get('spearman_p_value', {}).get('mean', 0):.6f} ± {cv_stats.get('spearman_p_value', {}).get('std', 0):.6f}")
    print(f"Binary Accuracy: {cv_stats.get('binary_accuracy', {}).get('mean', 0):.4f} ± {cv_stats.get('binary_accuracy', {}).get('std', 0):.4f}")
    print(f"F1 Score: {cv_stats.get('f1_score', {}).get('mean', 0):.4f} ± {cv_stats.get('f1_score', {}).get('std', 0):.4f}")
    
    # Check for suspicious patterns
    if 'auroc' in cv_stats:
        auroc_values = cv_df['auroc'].values
        # Check if there's a strong upward trend (correlation with fold number)
        fold_numbers = cv_df['fold'].values
        correlation = np.corrcoef(fold_numbers, auroc_values)[0, 1]
        if abs(correlation) > 0.7:
            print(f"WARNING: Strong correlation between fold number and AUROC: {correlation:.3f}")
            print("This may indicate systematic bias in CV splits or training process.")
    
    # Save detailed results
    cv_df.to_csv(os.path.join(results_dir, f"{feature_name}_{mil_arch}_cv_detailed_results.csv"), index=False)
    
    # Save summary statistics
    summary_df = pd.DataFrame(cv_stats).T
    summary_df.to_csv(os.path.join(results_dir, f"{feature_name}_{mil_arch}_cv_summary_stats.csv"))
    
    return cv_stats, cv_df

def create_mil_summary_for_comparison(feature_name, mil_arch, cv_stats):
    """Create summary row for comparing MIL architectures with safe metric extraction"""
    
    def safe_get_metric(stats_dict, metric_name, stat_type):
        """Safely extract metric statistics with fallback to 0"""
        return stats_dict.get(metric_name, {}).get(stat_type, 0.0)
    
    return {
        'Foundation_Model': feature_name,
        'MIL_Architecture': mil_arch,
        'MIL_Description': MIL_ARCHITECTURES[mil_arch]['description'],
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

def run_complete_phase2_comparison():
    """
    Run complete Phase 2: all selected models × all MIL architectures
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(CONFIG["random_seed"])
    
    print(f"{'='*100}")
    print(f"PHASE 2: COMPLETE FOUNDATION MODEL × MIL ARCHITECTURE COMPARISON")
    print(f"{'='*100}")
    print(f"Device: {device}")
    print(f"Selected Models: {list(SELECTED_MODELS.keys())}")
    print(f"MIL Architectures: {list(MIL_ARCHITECTURES.keys())}")
    print(f"Total Experiments: {len(SELECTED_MODELS)} × {len(MIL_ARCHITECTURES)} = {len(SELECTED_MODELS) * len(MIL_ARCHITECTURES)}")
    print(f"Each experiment: {CONFIG['n_folds']}-fold CV with {CONFIG['num_epochs']} epochs per fold")
    
    # Store all results
    all_phase2_results = []
    
    # Test each foundation model
    for model_idx, feature_name in enumerate(SELECTED_MODELS.keys()):
        print(f"\n{'#'*100}")
        print(f"TESTING FOUNDATION MODEL {model_idx+1}/{len(SELECTED_MODELS)}: {feature_name.upper()}")
        print(f"{'#'*100}")
        
        # Update config for current foundation model
        current_config = CONFIG.copy()
        current_config["feature_extractor"] = feature_name
        current_config["input_dim"] = SELECTED_MODELS[feature_name]["input_dim"]
        
        # Run MIL comparison for this foundation model
        model_mil_results = run_single_model_mil_comparison(current_config, device)
        all_phase2_results.extend(model_mil_results)
    
    # Create final comparison across all combinations
    print(f"\n{'='*120}")
    print(f"PHASE 2 FINAL RESULTS: ALL MODEL × MIL COMBINATIONS")
    print(f"{'='*120}")
    
    comparison_df = pd.DataFrame(all_phase2_results)
    
    # Sort by mean AUROC for ranking
    comparison_df_sorted = comparison_df.sort_values('AUROC_mean', ascending=False)
    
    # Save complete comparison
    os.makedirs("results_phase2_cv", exist_ok=True)
    comparison_df_sorted.to_csv("results_phase2_cv/phase2_complete_comparison.csv", index=False)
    
    # Print top combinations
    print(f"\nTOP 10 FOUNDATION MODEL × MIL ARCHITECTURE COMBINATIONS:")
    print(f"{'Rank':<4} {'Foundation':<12} {'MIL':<12} {'AUROC':<15} {'R²':<15} {'Binary Acc':<15}")
    print(f"{'-'*90}")
    
    for i, (_, row) in enumerate(comparison_df_sorted.head(10).iterrows(), 1):
        print(f"{i:<4} {row['Foundation_Model'].upper():<12} {row['MIL_Architecture']:<12} "
              f"{row['AUROC_mean']:.3f}±{row['AUROC_std']:.3f}   "
              f"{row['R2_mean']:.3f}±{row['R2_std']:.3f}       "
              f"{row['Binary_Accuracy_mean']:.3f}±{row['Binary_Accuracy_std']:.3f}")
    
    # Print per-foundation-model best MIL
    print(f"\nBEST MIL ARCHITECTURE PER FOUNDATION MODEL:")
    print(f"{'Foundation':<12} {'Best MIL':<12} {'AUROC':<15} {'R²':<15}")
    print(f"{'-'*60}")
    
    for foundation in SELECTED_MODELS.keys():
        foundation_results = comparison_df[comparison_df['Foundation_Model'] == foundation]
        best_for_foundation = foundation_results.loc[foundation_results['AUROC_mean'].idxmax()]
        print(f"{foundation.upper():<12} {best_for_foundation['MIL_Architecture']:<12} "
              f"{best_for_foundation['AUROC_mean']:.3f}±{best_for_foundation['AUROC_std']:.3f}   "
              f"{best_for_foundation['R2_mean']:.3f}±{best_for_foundation['R2_std']:.3f}")
    
    print(f"\nComplete Phase 2 results saved to: results_phase2_cv/phase2_complete_comparison.csv")
    
    return comparison_df_sorted

def main():
    """
    Main function for Phase 2
    """
    
    if len(sys.argv) > 1 and sys.argv[1] == "single_model":
        # Test all MIL architectures for one foundation model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(CONFIG["random_seed"])
        
        print(f"Running Phase 2 for single model: {CONFIG['feature_extractor']}")
        
        current_config = CONFIG.copy()
        current_config["input_dim"] = SELECTED_MODELS[CONFIG["feature_extractor"]]["input_dim"]
        
        model_results = run_single_model_mil_comparison(current_config, device)
        
        print(f"\nSingle model MIL comparison completed for {CONFIG['feature_extractor']}!")
        print(f"Results for {len(model_results)} MIL architectures saved.")
        
    else:
        # Run complete Phase 2 comparison
        print("Running complete Phase 2: All foundation models × All MIL architectures")
        final_results = run_complete_phase2_comparison()
        print(f"\nPhase 2 completed! {len(final_results)} total combinations tested.")

if __name__ == "__main__":
    main()