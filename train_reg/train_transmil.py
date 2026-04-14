# train_transmil_fixed.py
# Fixed version with patch limiting and command-line arguments

import os
import sys
import argparse
import gc
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
from models.regression_model import MeanPoolingMIL, MaxPoolingMIL, AttentionMIL, CLAM, TransMIL
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

# Configuration for Phase 2: MIL Architecture Comparison
CONFIG = {
    "feature_extractor": "virchow2",  # Will be overridden by CLI args
    "mil_architecture": "transmil",
    "experiment_type": "phase2_cv", 
    "hidden_dim": 128,
    "attention_hidden_dim": 128,
    "batch_size": 1,
    "lr": 1e-4,
    "num_epochs": 15,
    "n_folds": 5,
    "use_class_balancing": True,
    "use_boundary_weighting": False,  
    "loss_type": "huber",
    "huber_delta": 5.0,
    "random_seed": 42,
    "stratify_threshold": 25.0,
    "max_patches_transmil": 3000  # CRITICAL: Limit patches for TransMIL
}

def setup_phase2_cv_directories(feature_extractor: str, mil_architecture: str) -> str:
    """Create hierarchical results directory for Phase 2 CV experiments."""
    results_dir = f"results_phase2_cv/{feature_extractor}/{mil_architecture}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def create_cv_dataset_and_splits(combined_csv: str, n_folds: int, random_seed: int, 
                               stratify_threshold: float = 25.0):
    """Create dataset and stratified K-fold splits for cross-validation"""
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

def create_fold_dataloaders(full_dataset, train_indices, val_indices, batch_size: int,
                          use_class_balancing: bool = True):
    """Create dataloaders for a specific CV fold"""
    
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    if use_class_balancing:
        train_rs_scores = []
        for idx in train_indices:
            _, rs_score = full_dataset[idx]
            train_rs_scores.append(rs_score)
        
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
    
    if mil_architecture == "transmil":
        # Reduced parameters for memory efficiency
        model = TransMIL(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            proj_dim=256,  # Reduced from 512
            num_heads=4,    # Reduced from 8
            num_layers=1,   # Reduced from 2
            pos_enc='PPEG',
            dropout=0.1
        ).to(device)
    else:
        raise ValueError(f"This script is for TransMIL only")
    
    return model

def train_single_fold(fold_num: int, train_loader, val_loader, experiment_config, device):
    """Train and validate a single CV fold with TransMIL and patch limiting"""
    
    # Clear GPU cache before starting
    torch.cuda.empty_cache()
    gc.collect()
    
    set_seed(experiment_config["random_seed"])
    
    # Create fresh model
    model = create_mil_model(
        experiment_config["mil_architecture"],
        experiment_config["input_dim"],
        experiment_config["hidden_dim"],
        experiment_config["attention_hidden_dim"],
        device
    )
    
    # Loss and optimizer
    if experiment_config["loss_type"] == "huber":
        criterion = nn.HuberLoss(delta=experiment_config["huber_delta"])
    elif experiment_config["loss_type"] == "mse":
        criterion = nn.MSELoss()
    elif experiment_config["loss_type"] == "mae":
        criterion = nn.L1Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=experiment_config["lr"])
    
    # Initialize evaluators
    train_evaluator = RegressionEvaluator()
    val_evaluator = RegressionEvaluator()
    
    best_val_auroc = 0.0
    best_fold_metrics = None
    all_epoch_metrics = []
    
    # Get max patches from config
    max_patches = experiment_config.get("max_patches_transmil", 500)
    
    epoch_pbar = tqdm(range(1, experiment_config["num_epochs"] + 1), 
                      desc=f"Fold {fold_num} - Training", 
                      unit="epoch")
    
    for epoch in epoch_pbar:
        train_evaluator.reset()
        val_evaluator.reset()
        
        # Training epoch
        model.train()
        
        for features, rs_target in train_loader:
            if isinstance(features, list):
                features = features[0] if len(features) == 1 else torch.stack(features)
            
            # CRITICAL: Limit patches BEFORE moving to GPU
            if features.shape[0] > max_patches:
                indices = torch.randperm(features.shape[0])[:max_patches]
                features = features[indices]
                if epoch == 1:  # Only print on first epoch
                    print(f"  Sampled {max_patches} patches (original had {features.shape[0]})")
            
            # Now move to GPU
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
            
            prediction = model(features)
            if prediction.dim() == 0:
                prediction = prediction.unsqueeze(0)
                
            loss = criterion(prediction, rs_target)
            
            loss.backward()
            optimizer.step()
            
            train_evaluator.update(
                targets=rs_target.cpu().numpy(),
                preds=prediction.detach().cpu().numpy()
            )
        
        train_metrics = train_evaluator.compute_all_metrics(verbose=False)
        
        # Validation epoch
        model.eval()
        
        with torch.no_grad():
            for features, rs_target in val_loader:
                if isinstance(features, list):
                    features = features[0] if len(features) == 1 else torch.stack(features)
                
                # Also limit patches for validation
                if features.shape[0] > max_patches:
                    indices = torch.randperm(features.shape[0])[:max_patches]
                    features = features[indices]
                
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
                    
                loss = criterion(prediction, rs_target)
                
                val_evaluator.update(
                    targets=rs_target.cpu().numpy(),
                    preds=prediction.cpu().numpy()
                )
        
        val_metrics = val_evaluator.compute_all_metrics(verbose=False)
        all_epoch_metrics.append(val_metrics.copy())
        
        current_auroc = val_metrics.get('auroc', 0.0)
        if current_auroc > best_val_auroc:
            best_val_auroc = current_auroc
            best_fold_metrics = val_metrics.copy()
            best_fold_metrics['best_epoch'] = epoch
    
    # Clean up memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    best_fold_metrics['final_auroc'] = all_epoch_metrics[-1]['auroc']
    best_fold_metrics['final_r2'] = all_epoch_metrics[-1]['r2']
    
    return best_fold_metrics

def analyze_cv_results(fold_results, mil_arch, results_dir, feature_name):
    """Analyze and summarize cross-validation results"""
    
    print(f"\n{'='*80}")
    print(f"CV RESULTS: {feature_name.upper()} + TransMIL")
    print(f"{'='*80}")
    
    cv_df = pd.DataFrame(fold_results)
    
    key_metrics = ['auroc', 'rmse', 'mae', 'r2', 'spearman_correlation', 
                   'binary_accuracy', 'f1_score', 'boundary_mae']
    
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
    
    print(f"AUROC: {cv_stats.get('auroc', {}).get('mean', 0):.4f} ± {cv_stats.get('auroc', {}).get('std', 0):.4f}")
    print(f"R²: {cv_stats.get('r2', {}).get('mean', 0):.4f} ± {cv_stats.get('r2', {}).get('std', 0):.4f}")
    print(f"RMSE: {cv_stats.get('rmse', {}).get('mean', 0):.4f} ± {cv_stats.get('rmse', {}).get('std', 0):.4f}")
    
    cv_df.to_csv(os.path.join(results_dir, f"{feature_name}_transmil_cv_results.csv"), index=False)
    
    return cv_stats, cv_df

def run_single_model_transmil(feature_name: str, device, max_patches: int = 5000):
    """Run TransMIL for a single foundation model"""
    
    if feature_name not in SELECTED_MODELS:
        raise ValueError(f"Unknown model: {feature_name}. Choose from {list(SELECTED_MODELS.keys())}")
    
    model_config = SELECTED_MODELS[feature_name]
    
    print(f"\n{'='*80}")
    print(f"Running TransMIL for: {feature_name.upper()}")
    print(f"Max patches per slide: {max_patches}")
    print(f"{'='*80}")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Setup config
    current_config = CONFIG.copy()
    current_config["feature_extractor"] = feature_name
    current_config["input_dim"] = model_config["input_dim"]
    current_config["mil_architecture"] = "transmil"
    current_config["max_patches_transmil"] = max_patches
    
    # Create CV splits
    full_dataset, cv_splits, binary_labels = create_cv_dataset_and_splits(
        model_config["combined_csv"], 
        current_config["n_folds"], 
        current_config["random_seed"],
        current_config["stratify_threshold"]
    )
    
    print(f"Total samples: {len(full_dataset)}")
    print(f"Class distribution: {np.sum(binary_labels == 0)} low-risk, {np.sum(binary_labels == 1)} high-risk")
    
    # Create results directory
    results_dir = setup_phase2_cv_directories(feature_name, "transmil")
    
    fold_results = []
    
    # Run cross-validation
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\nFold {fold+1}/{current_config['n_folds']}")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        train_loader, val_loader = create_fold_dataloaders(
            full_dataset, train_idx, val_idx, 
            current_config["batch_size"],
            current_config["use_class_balancing"]
        )
        
        fold_metrics = train_single_fold(
            fold+1, train_loader, val_loader, current_config, device
        )
        
        fold_metrics['fold'] = fold + 1
        fold_metrics['foundation_model'] = feature_name
        fold_results.append(fold_metrics)
        
        print(f"  Fold {fold+1} Results: AUROC={fold_metrics.get('auroc', 0):.4f}, "
              f"R²={fold_metrics.get('r2', 0):.4f}")
    
    # Analyze results
    cv_stats, cv_df = analyze_cv_results(fold_results, "transmil", results_dir, feature_name)
    
    return cv_stats

def main():
    """Main function with command-line argument support"""
    
    parser = argparse.ArgumentParser(description='Run TransMIL for specified foundation model')
    parser.add_argument('--model', type=str, required=True,
                       choices=['h-optimus', 'virchow2', 'uni2-h', 'resnet18'],
                       help='Foundation model to use')
    parser.add_argument('--max-patches', type=int, default=500,
                       help='Maximum patches per slide (default: 500)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Device: {device}")
    
    # Set seed
    set_seed(CONFIG["random_seed"])
    
    # Run TransMIL for specified model
    cv_stats = run_single_model_transmil(args.model, device, args.max_patches)
    
    print(f"\n{'='*80}")
    print(f"COMPLETED: TransMIL for {args.model.upper()}")
    print(f"Final AUROC: {cv_stats.get('auroc', {}).get('mean', 0):.4f} ± {cv_stats.get('auroc', {}).get('std', 0):.4f}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()