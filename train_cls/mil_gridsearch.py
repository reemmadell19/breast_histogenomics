# gridsearch_critical_hyperparameters.py

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
import itertools
import json

# Project imports
from datasets.classification_mil_dataset import ClassificationMILDataset, create_classification_weighted_sampler
from utils.mil_utils import mil_collate_fn
from utils.classification_evaluator import ClassificationEvaluator
from utils.training_helpers import set_seed

# Foundation models
FOUNDATION_MODELS = {
    "resnet18": {"input_dim": 512, "combined_csv": "data/manifests/combined_features_resnet18.csv"},
    "resnet50": {"input_dim": 2048, "combined_csv": "data/manifests/combined_features_resnet50.csv"},
    "conch": {"input_dim": 512, "combined_csv": "data/manifests/combined_features_conch.csv"},
    "uni2-h": {"input_dim": 1536, "combined_csv": "data/manifests/combined_features_uni2-h.csv"},
    "virchow2": {"input_dim": 1280, "combined_csv": "data/manifests/combined_features_virchow2.csv"},
    "h-optimus": {"input_dim": 1536, "combined_csv": "data/manifests/combined_features_h-optimus.csv"}
}

# =====================================================================
# CRITICAL PARAMETERS ONLY - FOCUSED GRID SEARCH
# =====================================================================

CRITICAL_PARAMS_GRID = {
    "mean": {
        "learning_rate": [1e-4, 2e-4],
        "hidden_dim": [128, 256],
        # Fixed parameters
        "weight_decay": [1e-5],
        "dropout": [0.25],
    },
    
    "attention": {
        "learning_rate": [1e-4, 2e-4],
        "hidden_dim": [128],  # Fix based on results
        "attention_hidden_dim": [64, 128, 256],
        # Fixed parameters
        "weight_decay": [1e-5],
        "dropout": [0.25],
    },
    
    "clam": {
        "learning_rate": [1e-4],  # Fix at optimal
        "hidden_dim": [128],
        "attention_hidden_dim": [256, 512],
        # Fixed parameters
        "weight_decay": [1e-5],
        "dropout": [0.25],
        "gate": [True],
    },
    
    "acmil": {
        "learning_rate": [1e-4, 2e-4],
        "hidden_dim": [128,256],  # Fix based on results
        "n_branches": [5, 7],
        "mask_ratio": [0.0, 0.6],
        # Fixed parameters
        "n_masked_patch": [10],
        "weight_decay": [1e-5],
        "dropout": [0.25],
    },
    
    "acmil_clam": {
        "learning_rate": [1e-4],
        "hidden_dim": [128,256],
        "attention_hidden_dim": [256, 512],
        "n_branches": [5, 7],
        "mask_ratio": [0.0,0.6],  # Fix at optimal
        # Fixed parameters
        "n_masked_patch": [10],
        "weight_decay": [1e-5],
        "dropout": [0.25],
        "gate": [True],
    }
}

# Fixed parameters for all experiments
FIXED_PARAMS = {
    "num_epochs": 10,  # Quick evaluation
    "batch_size": 1,
    "use_class_balancing": True,
    "use_gradient_clip": True,
    "gradient_clip_val": 1.0,
    "n_folds": 3,  # 3-fold for speed
    "random_seed": 42
}

def print_grid_summary():
    """Print summary of combinations to be tested"""
    print("\n" + "="*80)
    print("CRITICAL PARAMETERS GRID SEARCH - COMBINATIONS SUMMARY")
    print("="*80)
    
    total_combinations = 0
    for arch, params in CRITICAL_PARAMS_GRID.items():
        # Calculate combinations
        non_fixed = {k: v for k, v in params.items() 
                    if len(v) > 1 or k in ['learning_rate', 'hidden_dim', 'attention_hidden_dim', 'n_branches', 'mask_ratio']}
        combinations = 1
        for values in non_fixed.values():
            combinations *= len(values)
        
        print(f"\n{arch.upper()}:")
        print(f"  Critical parameters being tested:")
        for k, v in non_fixed.items():
            if len(v) > 1:
                print(f"    - {k}: {v}")
        print(f"  Total combinations: {combinations}")
        total_combinations += combinations
    
    print(f"\n{'='*80}")
    print(f"TOTAL COMBINATIONS ACROSS ALL ARCHITECTURES: {total_combinations}")
    print(f"{'='*80}\n")
    
    return total_combinations

def create_model(params, input_dim, mil_architecture, device, n_classes=2):
    """Create model with given parameters"""
    
    from models.classification_model import (
        MeanPoolingMILClassifier, AttentionMILClassifier, 
        CLAMClassifier, ACMILClassifier, ACMIL_CLAM_HybridClassifier
    )
    
    if mil_architecture == "mean":
        model = MeanPoolingMILClassifier(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            n_classes=n_classes
        ).to(device)
    
    elif mil_architecture == "attention":
        model = AttentionMILClassifier(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            attention_hidden_dim=params['attention_hidden_dim'],
            n_classes=n_classes
        ).to(device)
    
    elif mil_architecture == "clam":
        model = CLAMClassifier(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            attention_hidden_dim=params['attention_hidden_dim'],
            n_classes=n_classes,
            dropout=params['dropout'],
            gate=params.get('gate', True)
        ).to(device)
    
    elif mil_architecture == "acmil":
        model = ACMILClassifier(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            n_branches=params['n_branches'],
            n_classes=n_classes,
            n_masked_patch=params['n_masked_patch'],
            mask_ratio=params['mask_ratio'],
            dropout=params['dropout']
        ).to(device)
    
    elif mil_architecture == "acmil_clam":
        model = ACMIL_CLAM_HybridClassifier(
            input_dim=input_dim,
            n_branches=params['n_branches'],
            hidden_dim=params['hidden_dim'],
            attention_hidden_dim=params['attention_hidden_dim'],
            n_classes=n_classes,
            mask_ratio=params['mask_ratio'],
            n_masked_patch=params['n_masked_patch'],
            dropout=params['dropout'],
            gate=params.get('gate', True)
        ).to(device)
    
    return model

def train_with_params(params, train_loader, val_loader, input_dim, mil_architecture, device):
    """Quick training to evaluate hyperparameters"""
    
    set_seed(FIXED_PARAMS['random_seed'])
    
    # Create model
    model = create_model(params, input_dim, mil_architecture, device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), 
                           lr=params['learning_rate'], 
                           weight_decay=params['weight_decay'])
    
    # Track best validation performance
    best_auroc = 0.0
    best_f1 = 0.0
    
    for epoch in range(FIXED_PARAMS['num_epochs']):
        # Train
        model.train()
        for features, label in train_loader:
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
            
            if mil_architecture in ["acmil", "acmil_clam"]:
                logits = model(features, return_branch_outputs=False)
            else:
                logits = model(features)
            
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            
            loss = criterion(logits, label)
            loss.backward()
            
            if FIXED_PARAMS['use_gradient_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                              FIXED_PARAMS['gradient_clip_val'])
            
            optimizer.step()
        
        # Validate
        model.eval()
        evaluator = ClassificationEvaluator(n_classes=2)
        
        with torch.no_grad():
            for features, label in val_loader:
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
                
                if mil_architecture in ["acmil", "acmil_clam"]:
                    logits = model(features, return_branch_outputs=False)
                else:
                    logits = model(features)
                
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                evaluator.update(
                    labels=label.cpu().numpy(),
                    preds=preds.cpu().numpy(),
                    probs=probs.cpu().numpy()
                )
        
        metrics = evaluator.compute_all_metrics(verbose=False)
        best_auroc = max(best_auroc, metrics.get('auroc', 0.0))
        best_f1 = max(best_f1, metrics.get('f1_score', 0.0))
    
    return best_auroc, best_f1

def generate_param_combinations(param_grid):
    """Generate all combinations of parameters"""
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]

def run_critical_grid_search(model_name, mil_architecture):
    """Run focused grid search on critical parameters only"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(FIXED_PARAMS['random_seed'])
    
    # Load configuration
    model_config = FOUNDATION_MODELS[model_name]
    param_grid = CRITICAL_PARAMS_GRID[mil_architecture]
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"gridsearch_critical/{model_name}_{mil_architecture}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"CRITICAL PARAMETERS GRID SEARCH: {model_name.upper()} + {mil_architecture.upper()}")
    print(f"{'='*80}")
    
    # Load dataset
    df_check = pd.read_csv(model_config["combined_csv"])
    label_column = 'RSHigh' if 'RSHigh' in df_check.columns else 'RS'
    
    dataset = ClassificationMILDataset(
        model_config["combined_csv"],
        label_column=label_column,
        threshold=25.0
    )
    
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Generate parameter combinations
    param_combinations = generate_param_combinations(param_grid)
    
    # Filter to show only varying parameters
    varying_params = {k: v for k, v in param_grid.items() if len(v) > 1}
    
    print(f"Critical parameters being tested:")
    for k, v in varying_params.items():
        print(f"  {k}: {v}")
    print(f"Total combinations: {len(param_combinations)}")
    
    # Store results
    all_results = []
    
    # Progress bar
    pbar = tqdm(param_combinations, desc="Testing combinations")
    
    for params in pbar:
        # Run 3-fold CV
        cv_aurocs = []
        cv_f1s = []
        
        skf = StratifiedKFold(n_splits=FIXED_PARAMS['n_folds'], 
                             shuffle=True, 
                             random_state=FIXED_PARAMS['random_seed'])
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
            # Create dataloaders
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            # Create sampler for balanced training
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
            
            train_loader = DataLoader(train_subset, batch_size=1, sampler=sampler,
                                    collate_fn=mil_collate_fn, num_workers=0)
            val_loader = DataLoader(val_subset, batch_size=1, shuffle=False,
                                  collate_fn=mil_collate_fn, num_workers=0)
            
            # Train and evaluate
            fold_auroc, fold_f1 = train_with_params(
                params, train_loader, val_loader, 
                model_config["input_dim"], mil_architecture, device
            )
            cv_aurocs.append(fold_auroc)
            cv_f1s.append(fold_f1)
        
        # Calculate statistics
        mean_auroc = np.mean(cv_aurocs)
        std_auroc = np.std(cv_aurocs)
        mean_f1 = np.mean(cv_f1s)
        
        # Store result
        result = params.copy()
        result['mean_auroc'] = mean_auroc
        result['std_auroc'] = std_auroc
        result['mean_f1'] = mean_f1
        all_results.append(result)
        
        # Update progress
        pbar.set_postfix({'auroc': f'{mean_auroc:.3f}±{std_auroc:.3f}'})
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('mean_auroc', ascending=False)
    
    # Save results
    results_df.to_csv(os.path.join(results_dir, 'critical_params_results.csv'), index=False)
    
    # Save best configuration
    best_params = results_df.iloc[0].to_dict()
    best_config = {
        'model_name': model_name,
        'mil_architecture': mil_architecture,
        'best_auroc': best_params['mean_auroc'],
        'auroc_std': best_params['std_auroc'],
        'best_f1': best_params['mean_f1'],
        'parameters': {k: v for k, v in best_params.items() 
                      if k not in ['mean_auroc', 'std_auroc', 'mean_f1']}
    }
    
    with open(os.path.join(results_dir, 'best_critical_params.json'), 'w') as f:
        json.dump(best_config, f, indent=2)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"TOP 3 CONFIGURATIONS:")
    print(f"{'='*80}")
    
    for idx, row in results_df.head(3).iterrows():
        print(f"\nRank {idx+1}:")
        print(f"  AUROC: {row['mean_auroc']:.4f} ± {row['std_auroc']:.4f}")
        print(f"  F1: {row['mean_f1']:.4f}")
        for k in varying_params.keys():
            print(f"  {k}: {row[k]}")
    
    print(f"\nBest configuration saved to: {results_dir}")
    
    return best_config

def run_all_architectures_critical_search(model_name):
    """Run critical parameter search for all architectures"""
    
    print_grid_summary()
    
    all_best_configs = {}
    
    for mil_arch in CRITICAL_PARAMS_GRID.keys():
        best_config = run_critical_grid_search(model_name, mil_arch)
        all_best_configs[mil_arch] = best_config
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = f"gridsearch_critical/{model_name}_summary_{timestamp}"
    os.makedirs(summary_dir, exist_ok=True)
    
    # Create summary table
    summary_data = []
    for mil_arch, config in all_best_configs.items():
        row = {
            'mil_architecture': mil_arch,
            'best_auroc': config['best_auroc'],
            'auroc_std': config['auroc_std'],
            'best_f1': config['best_f1']
        }
        row.update(config['parameters'])
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('best_auroc', ascending=False)
    summary_df.to_csv(os.path.join(summary_dir, 'all_architectures_best_critical.csv'), index=False)
    
    # Print final summary
    print(f"\n{'='*100}")
    print(f"FINAL SUMMARY - CRITICAL PARAMETERS: {model_name.upper()}")
    print(f"{'='*100}")
    
    for _, row in summary_df.iterrows():
        print(f"\n{row['mil_architecture'].upper()}:")
        print(f"  AUROC: {row['best_auroc']:.4f} ± {row['auroc_std']:.4f}")
        print(f"  Best params: lr={row.get('learning_rate')}, hidden={row.get('hidden_dim')}", end="")
        if 'attention_hidden_dim' in row:
            print(f", att_dim={row['attention_hidden_dim']}", end="")
        if 'n_branches' in row:
            print(f", branches={row['n_branches']}", end="")
        if 'mask_ratio' in row:
            print(f", mask={row['mask_ratio']}", end="")
        print()
    
    return all_best_configs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Critical parameters grid search')
    parser.add_argument('--model', type=str, required=True,
                      choices=list(FOUNDATION_MODELS.keys()),
                      help=f'Model name. Options: {list(FOUNDATION_MODELS.keys())}')
    parser.add_argument('--mil', type=str, default='all',
                      choices=list(CRITICAL_PARAMS_GRID.keys()) + ['all'],
                      help='MIL architecture (default: all)')
    
    args = parser.parse_args()
    
    if args.mil == 'all':
        print(f"Running critical parameter search for ALL architectures with {args.model}")
        run_all_architectures_critical_search(args.model)
    else:
        print(f"Running critical parameter search for {args.model} + {args.mil}")
        run_critical_grid_search(args.model, args.mil)