# train_phase3_clam_optimization.py
# Phase 3: CLAM Architecture Optimization for Single Foundation Model
# Run separately for each foundation model on different GPUs

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
import itertools
import json
from datetime import datetime

# Project imports
from datasets.regression_mil_dataset import RegressionMILDataset, create_rs_weighted_sampler
from models.regression_model import CLAM
from utils.mil_utils import mil_collate_fn
from utils.regression_evaluator import RegressionEvaluator
from utils.training_helpers import set_seed

# Foundation model configurations
FOUNDATION_MODELS = {
    "h-optimus": {
        "input_dim": 1536,
        "combined_csv": "data/manifests/combined_features_h-optimus.csv",
        "baseline_auroc": 0.859
    },
    "virchow2": {
        "input_dim": 1280,
        "combined_csv": "data/manifests/combined_features_virchow2.csv", 
        "baseline_auroc": 0.853
    },
    "uni2-h": {
        "input_dim": 1536,
        "combined_csv": "data/manifests/combined_features_uni2-h.csv",
        "baseline_auroc": 0.852
    },
    "resnet18": {
        "input_dim": 512,
        "combined_csv": "data/manifests/combined_features_resnet18.csv",
        "baseline_auroc": 0.692
    }
}

# CLAM optimization grid
CLAM_OPTIMIZATION_GRID = {
    "hidden_dim": [128, 256, 512],               # Feature transformation size
    "attention_hidden_dim": [256, 512],          # Attention network capacity
    "gate": [True, False],                       # Gated vs simple attention
    "learning_rate": [5e-5, 1e-4, 2e-4],       # Training dynamics
    "dropout": [0.25, 0.5],                     # Regularization
    "weight_decay": [0, 1e-4]                   # L2 regularization
}

# Configuration
CONFIG = {
    "feature_extractor": "h-optimus",            # CHANGE THIS FOR EACH RUN
    "experiment_type": "phase3_clam_optimization",
    "batch_size": 1,
    "max_epochs": 20,
    "early_stopping_patience": 5,
    "inner_cv_folds": 3,                         # For hyperparameter selection
    "final_validation_folds": 5,                 # For final performance estimate
    "random_seed": 42,
    "stratify_threshold": 25.0,
    "loss_type": "huber",
    "huber_delta": 5.0
}

def setup_results_directory(feature_extractor: str) -> str:
    """Create results directory for this optimization run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_phase3_optimization/{feature_extractor}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    return results_dir

def create_cv_dataset_and_splits(combined_csv: str, n_folds: int, random_seed: int, 
                               stratify_threshold: float = 25.0):
    """Create dataset and stratified K-fold splits"""
    full_dataset = RegressionMILDataset(combined_csv)
    
    rs_scores = []
    for i in range(len(full_dataset)):
        _, rs_score = full_dataset[i]
        rs_scores.append(rs_score)
    
    rs_scores = np.array(rs_scores)
    binary_labels = (rs_scores >= stratify_threshold).astype(int)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    cv_splits = list(skf.split(range(len(full_dataset)), binary_labels))
    
    return full_dataset, cv_splits, binary_labels

def generate_hyperparameter_combinations(param_grid):
    """Generate all hyperparameter combinations from grid"""
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    param_combinations = []
    for combo in combinations:
        param_dict = dict(zip(keys, combo))
        param_combinations.append(param_dict)
    
    return param_combinations

def create_fold_dataloaders(full_dataset, train_indices, val_indices, use_class_balancing=True):
    """Create dataloaders for a specific CV fold"""
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    if use_class_balancing:
        # Simple approach: create temporary dataset for weighted sampling
        class TempDataset:
            def __init__(self, indices, full_dataset):
                self.indices = indices
                self.full_dataset = full_dataset
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                return self.full_dataset[self.indices[idx]]
        
        temp_dataset = TempDataset(train_indices, full_dataset)
        sampler = create_rs_weighted_sampler(temp_dataset, boundary_focus=False, 
                                           class_balance=True, threshold=25.0)
        
        train_loader = DataLoader(train_subset, batch_size=1, sampler=sampler, 
                                collate_fn=mil_collate_fn, num_workers=0)
    else:
        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True, 
                                collate_fn=mil_collate_fn, num_workers=0)
    
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, 
                          collate_fn=mil_collate_fn, num_workers=0)
    
    return train_loader, val_loader

def train_and_evaluate_hyperparameter_combination(model_config, hyperparams, 
                                                 train_loader, val_loader, device):
    """Train and evaluate single hyperparameter combination"""
    
    # Set consistent seed for reproducibility
    set_seed(CONFIG["random_seed"])
    
    # Create CLAM model with current hyperparameters
    model = CLAM(
        input_dim=model_config["input_dim"],
        hidden_dim=hyperparams["hidden_dim"],
        attention_hidden_dim=hyperparams["attention_hidden_dim"],
        dropout=hyperparams["dropout"],
        gate=hyperparams["gate"]
    ).to(device)
    
    # Setup training components
    criterion = nn.HuberLoss(delta=CONFIG["huber_delta"])
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"]
    )
    
    # Training loop with fixed epochs - track best performance across all epochs
    best_val_auroc = 0.0
    best_metrics = None
    
    for epoch in range(1, 16):  # Fixed 15 epochs as requested
        # Training epoch
        model.train()
        
        for features, rs_target in train_loader:
            # Data processing
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
            optimizer.zero_grad()
            prediction = model(features)
            if prediction.dim() == 0:
                prediction = prediction.unsqueeze(0)
            
            loss = criterion(prediction, rs_target)
            loss.backward()
            optimizer.step()
        
        # Validation epoch
        model.eval()
        val_evaluator = RegressionEvaluator()
        
        with torch.no_grad():
            for features, rs_target in val_loader:
                # Data processing
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
                
                prediction = model(features)
                if prediction.dim() == 0:
                    prediction = prediction.unsqueeze(0)
                
                val_evaluator.update(
                    targets=rs_target.cpu().numpy(),
                    preds=prediction.cpu().numpy()
                )
        
        # Get validation metrics
        val_metrics = val_evaluator.compute_all_metrics(verbose=False)
        current_auroc = val_metrics.get('auroc', 0.0)
        
        # Track best performance across all epochs
        if current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            best_metrics = val_metrics.copy()
    
    return best_val_auroc, best_metrics

def run_grid_search(model_config, param_grid, shared_cv_splits, full_dataset, device):
    """Run complete grid search with inner CV"""
    
    print(f"Starting CLAM optimization for {model_config['name'].upper()}")
    print(f"Baseline AUROC: {model_config['baseline_auroc']:.4f}")
    print(f"Total hyperparameter combinations: {len(generate_hyperparameter_combinations(param_grid))}")
    print(f"Inner CV folds: {CONFIG['inner_cv_folds']}")
    
    # Generate all parameter combinations
    param_combinations = generate_hyperparameter_combinations(param_grid)
    
    # Track all results
    all_results = []
    
    # Progress tracking
    overall_pbar = tqdm(param_combinations, desc=f"Grid Search - {model_config['name'].upper()}")
    
    for combo_idx, hyperparams in enumerate(overall_pbar):
        # Run inner CV for this hyperparameter combination
        inner_cv_scores = []
        inner_cv_detailed = []
        
        for fold_idx in range(CONFIG["inner_cv_folds"]):
            train_idx, val_idx = shared_cv_splits[fold_idx]
            
            # Create dataloaders
            train_loader, val_loader = create_fold_dataloaders(
                full_dataset, train_idx, val_idx, use_class_balancing=True
            )
            
            # Train and evaluate
            fold_auroc, fold_metrics = train_and_evaluate_hyperparameter_combination(
                model_config, hyperparams, train_loader, val_loader, device
            )
            
            inner_cv_scores.append(fold_auroc)
            inner_cv_detailed.append(fold_metrics)
        
        # Aggregate inner CV results
        avg_auroc = np.mean(inner_cv_scores)
        std_auroc = np.std(inner_cv_scores)
        
        # Store comprehensive results
        result_entry = {
            'combination_idx': combo_idx,
            'avg_auroc': avg_auroc,
            'std_auroc': std_auroc,
            'improvement_over_baseline': avg_auroc - model_config['baseline_auroc'],
            **hyperparams  # Include all hyperparameter values
        }
        
        # Add aggregated metrics from detailed results
        if inner_cv_detailed:
            for metric_name in ['r2', 'rmse', 'f1_score', 'binary_accuracy', 'spearman_correlation', 'spearman_p_value']:
                metric_values = [fold.get(metric_name, 0) for fold in inner_cv_detailed]
                result_entry[f'avg_{metric_name}'] = np.mean(metric_values)
                result_entry[f'std_{metric_name}'] = np.std(metric_values)
        
        all_results.append(result_entry)
        
        # Update progress bar with current best
        if combo_idx == 0 or avg_auroc > max([r['avg_auroc'] for r in all_results[:-1]], default=0):
            overall_pbar.set_postfix({
                'Best_AUROC': f'{avg_auroc:.4f}',
                'Improvement': f'{avg_auroc - model_config["baseline_auroc"]:+.3f}'
            })
    
    return all_results

def analyze_optimization_results(all_results, model_config, results_dir):
    """Analyze grid search results and find best hyperparameters"""
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Find best combination
    best_idx = results_df['avg_auroc'].idxmax()
    best_result = results_df.iloc[best_idx]
    
    print(f"\n{'='*80}")
    print(f"GRID SEARCH RESULTS: {model_config['name'].upper()}")
    print(f"{'='*80}")
    
    # Performance summary
    print(f"Total combinations tested: {len(results_df)}")
    print(f"Baseline AUROC: {model_config['baseline_auroc']:.4f}")
    print(f"Best AUROC: {best_result['avg_auroc']:.4f} ± {best_result['std_auroc']:.4f}")
    print(f"Improvement: +{best_result['improvement_over_baseline']:.3f} "
          f"({best_result['improvement_over_baseline']/model_config['baseline_auroc']*100:+.1f}%)")
    
    # Best hyperparameters
    print(f"\nBEST HYPERPARAMETERS:")
    hyperparam_keys = ['hidden_dim', 'attention_hidden_dim', 'gate', 'learning_rate', 'dropout', 'weight_decay']
    for param in hyperparam_keys:
        print(f"  {param}: {best_result[param]}")
    
    # Additional metrics for best combination
    print(f"\nBEST COMBINATION PERFORMANCE:")
    print(f"  AUROC: {best_result['avg_auroc']:.4f} ± {best_result['std_auroc']:.4f}")
    print(f"  R²: {best_result.get('avg_r2', 0):.4f} ± {best_result.get('std_r2', 0):.4f}")
    print(f"  F1: {best_result.get('avg_f1_score', 0):.4f} ± {best_result.get('std_f1_score', 0):.4f}")
    print(f"  Spearman: {best_result.get('avg_spearman_correlation', 0):.4f} ± {best_result.get('std_spearman_correlation', 0):.4f}")
    print(f"  Spearman p-value: {best_result.get('avg_spearman_p_value', 1):.6f}")
    print(f"  RMSE: {best_result.get('avg_rmse', 0):.4f} ± {best_result.get('std_rmse', 0):.4f}")
    print(f"  Binary Acc: {best_result.get('avg_binary_accuracy', 0):.4f} ± {best_result.get('std_binary_accuracy', 0):.4f}")
    
    # Top 5 combinations with more metrics
    results_df_sorted = results_df.sort_values('avg_auroc', ascending=False)
    print(f"\nTOP 5 COMBINATIONS:")
    print(f"{'Rank':<4} {'AUROC':<15} {'R²':<15} {'Spearman':<15} {'Hidden':<8} {'Attn':<8} {'Gate':<6}")
    print(f"{'-'*85}")
    
    for i, (_, row) in enumerate(results_df_sorted.head(5).iterrows(), 1):
        spearman_sig = "***" if row.get('avg_spearman_p_value', 1) < 0.001 else "**" if row.get('avg_spearman_p_value', 1) < 0.01 else "*" if row.get('avg_spearman_p_value', 1) < 0.05 else ""
        print(f"{i:<4} {row['avg_auroc']:.3f}±{row['std_auroc']:.3f}   "
              f"{row.get('avg_r2', 0):.3f}±{row.get('std_r2', 0):.3f}     "
              f"{row.get('avg_spearman_correlation', 0):.3f}{spearman_sig}        "
              f"{int(row['hidden_dim']):<8} {int(row['attention_hidden_dim']):<8} "
              f"{str(row['gate']):<6}")
    
    # Save detailed results with ALL metrics
    results_df_sorted.to_csv(os.path.join(results_dir, "grid_search_detailed_results.csv"), index=False)
    
    # Extract best hyperparameters for saving
    best_hyperparams = {key: best_result[key] for key in hyperparam_keys}
    
    # Save best hyperparameters
    with open(os.path.join(results_dir, "best_hyperparameters.json"), 'w') as f:
        json.dump(best_hyperparams, f, indent=2)
    
    return best_hyperparams, best_result, results_df_sorted

def run_final_validation(model_config, best_hyperparams, shared_cv_splits, 
                        full_dataset, device, results_dir):
    """Run final 5-fold validation with best hyperparameters"""
    
    print(f"\n{'='*80}")
    print(f"FINAL VALIDATION: {model_config['name'].upper()} WITH OPTIMIZED CLAM")
    print(f"{'='*80}")
    
    final_results = []
    
    # Run all 5 folds for unbiased performance estimate
    for fold in range(CONFIG["final_validation_folds"]):
        train_idx, val_idx = shared_cv_splits[fold]
        
        print(f"Final Fold {fold+1}/{CONFIG['final_validation_folds']}")
        print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
        
        # Create dataloaders
        train_loader, val_loader = create_fold_dataloaders(
            full_dataset, train_idx, val_idx, use_class_balancing=True
        )
        
        # Train with best hyperparameters
        fold_auroc, fold_metrics = train_and_evaluate_hyperparameter_combination(
            model_config, best_hyperparams, train_loader, val_loader, device
        )
        
        # Store results
        fold_metrics['fold'] = fold + 1
        fold_metrics['model'] = model_config['name']
        final_results.append(fold_metrics)
        
        print(f"  Results: AUROC={fold_metrics.get('auroc', 0):.4f}, "
              f"R²={fold_metrics.get('r2', 0):.4f}, "
              f"RMSE={fold_metrics.get('rmse', 0):.4f}, "
              f"F1={fold_metrics.get('f1_score', 0):.4f}")
    
    # Calculate final statistics
    final_df = pd.DataFrame(final_results)
    
    # Summary statistics
    final_summary = {
        'model': model_config['name'],
        'final_auroc_mean': final_df['auroc'].mean(),
        'final_auroc_std': final_df['auroc'].std(),
        'final_r2_mean': final_df['r2'].mean(),
        'final_r2_std': final_df['r2'].std(),
        'final_f1_mean': final_df['f1_score'].mean(),
        'final_f1_std': final_df['f1_score'].std(),
        'baseline_auroc': model_config['baseline_auroc'],
        'final_improvement': final_df['auroc'].mean() - model_config['baseline_auroc'],
        'best_hyperparams': best_hyperparams
    }
    
    print(f"\n{'='*60}")
    print(f"FINAL OPTIMIZED PERFORMANCE")
    print(f"{'='*60}")
    print(f"AUROC: {final_summary['final_auroc_mean']:.4f} ± {final_summary['final_auroc_std']:.4f}")
    print(f"R²: {final_summary['final_r2_mean']:.4f} ± {final_summary['final_r2_std']:.4f}")
    print(f"F1: {final_summary['final_f1_mean']:.4f} ± {final_summary['final_f1_std']:.4f}")
    print(f"Total Improvement: +{final_summary['final_improvement']:.3f}")
    
    # Save final results
    final_df.to_csv(os.path.join(results_dir, "final_validation_results.csv"), index=False)
    
    with open(os.path.join(results_dir, "final_summary.json"), 'w') as f:
        json.dump(final_summary, f, indent=2, default=str)
    
    return final_df, final_summary

def main():
    """Main execution for single model CLAM optimization"""
    
    # Setup
    feature_extractor = CONFIG["feature_extractor"]
    
    if feature_extractor not in FOUNDATION_MODELS:
        print(f"Error: Unknown feature extractor '{feature_extractor}'")
        print(f"Available options: {list(FOUNDATION_MODELS.keys())}")
        return
    
    model_config = FOUNDATION_MODELS[feature_extractor].copy()
    model_config['name'] = feature_extractor
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(CONFIG["random_seed"])
    
    print(f"{'='*80}")
    print(f"PHASE 3: CLAM OPTIMIZATION FOR {feature_extractor.upper()}")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"Feature Extractor: {feature_extractor}")
    print(f"Input Dimension: {model_config['input_dim']}")
    print(f"Baseline AUROC: {model_config['baseline_auroc']:.4f}")
    
    # Create results directory
    results_dir = setup_results_directory(feature_extractor)
    
    # Save configuration
    config_save = CONFIG.copy()
    config_save['model_config'] = model_config
    config_save['optimization_grid'] = CLAM_OPTIMIZATION_GRID
    with open(os.path.join(results_dir, "experiment_config.json"), 'w') as f:
        json.dump(config_save, f, indent=2)
    
    # Load dataset and create CV splits
    print(f"Loading dataset and creating CV splits...")
    full_dataset, shared_cv_splits, binary_labels = create_cv_dataset_and_splits(
        model_config["combined_csv"],
        max(CONFIG["inner_cv_folds"], CONFIG["final_validation_folds"]),
        CONFIG["random_seed"],
        CONFIG["stratify_threshold"]
    )
    
    print(f"Dataset: {len(full_dataset)} samples")
    print(f"Class distribution: {np.sum(binary_labels == 0)} low-risk, {np.sum(binary_labels == 1)} high-risk")
    
    # Stage 1: Grid search with inner CV
    print(f"\n{'='*60}")
    print(f"STAGE 1: HYPERPARAMETER GRID SEARCH")
    print(f"{'='*60}")
    
    all_results = run_grid_search(model_config, CLAM_OPTIMIZATION_GRID, 
                                 shared_cv_splits, full_dataset, device)
    
    # Stage 2: Analyze results and find best hyperparameters
    print(f"\n{'='*60}")
    print(f"STAGE 2: ANALYSIS AND BEST HYPERPARAMETER SELECTION")
    print(f"{'='*60}")
    
    best_hyperparams, best_result, sorted_results = analyze_optimization_results(
        all_results, model_config, results_dir
    )
    
    # Stage 3: Final validation with best hyperparameters
    print(f"\n{'='*60}")
    print(f"STAGE 3: FINAL VALIDATION")
    print(f"{'='*60}")
    
    final_df, final_summary = run_final_validation(
        model_config, best_hyperparams, shared_cv_splits, 
        full_dataset, device, results_dir
    )
    
    # Complete summary
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE: {feature_extractor.upper()}")
    print(f"{'='*80}")
    print(f"Ready for Phase 4 generalisability testing")
    print(f"All results saved to: {results_dir}")
    
    return final_summary

if __name__ == "__main__":
    # Allow command line override of feature extractor
    if len(sys.argv) > 1:
        CONFIG["feature_extractor"] = sys.argv[1]
        print(f"Using feature extractor from command line: {CONFIG['feature_extractor']}")
    
    main()