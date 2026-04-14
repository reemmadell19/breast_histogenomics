# train_phase4_training_optimization.py
# Phase 4: Training Strategy Optimization for Best Models
# Optimizes training dynamics for Virchow2 and ResNet18

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import itertools
import json
from datetime import datetime
import argparse

# Project imports
from datasets.regression_mil_dataset import RegressionMILDataset, create_rs_weighted_sampler
from models.regression_model import CLAM
from utils.mil_utils import mil_collate_fn
from utils.regression_evaluator import RegressionEvaluator
from utils.training_helpers import set_seed

# Custom JSON encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        return super(NumpyEncoder, self).default(obj)

# Model configurations from Phase 3
PHASE3_BEST_CONFIGS = {
    "virchow2": {
        "input_dim": 1280,
        "combined_csv": "data/manifests/combined_features_virchow2.csv",
        "baseline_auroc": 0.8832,  # Phase 3 result
        "clam_params": {
            "hidden_dim": 512,
            "attention_hidden_dim": 512,
            "dropout": 0.25,
            "gate": False
        },
        # Phase 3 original hyperparameters
        "phase3_training": {
            "learning_rate": 0.0002,
            "weight_decay": 0.0
        }
    },
    "resnet18": {
        "input_dim": 512,
        "combined_csv": "data/manifests/combined_features_resnet18.csv",
        "baseline_auroc": 0.7408,  # Phase 3 result
        "clam_params": {
            "hidden_dim": 256,
            "attention_hidden_dim": 512,
            "dropout": 0.25,
            "gate": False
        },
        # Phase 3 original hyperparameters
        "phase3_training": {
            "learning_rate": 0.0002,
            "weight_decay": 0.0
        }
    }
}

# Phase 4 Training Optimization Grid - includes original Phase 3 settings
TRAINING_OPTIMIZATION_GRID = {
    'optimizer': ['Adam', 'AdamW'],  # Test both
    'base_lr': [1e-4, 2e-4],
    'lr_schedule': ['none', 'cosine', 'reduce_plateau'],  # Include no scheduler
    'weight_decay': [0, 1e-4, 1e-3],  # Include 0 from Phase 3
    'warmup_epochs': [0, 5],
    'gradient_clip_value': [None, 1.0, 5.0],
    'huber_delta': [1.0, 5.0, 10.0]
}

# Schedule-specific parameters
SCHEDULE_PARAMS = {
    'reduce_plateau': {
        'patience': 5,
        'factor': 0.5,
        'min_lr': 1e-6,
        'mode': 'max'
    },
    'cosine': {
        'T_max': 50,  # Full 50 epochs
        'eta_min': 1e-6
    }
}

# Training configuration
TRAINING_CONFIG = {
    'max_epochs': 50,
    'early_stopping_patience': 10,
    'early_stopping_metric': 'auroc',
    'inner_cv_folds': 3,
    'final_cv_folds': 5,
    'batch_size': 1,
    'random_seed': 42,
    'stratify_threshold': 25.0
}

def setup_results_directory(model_name: str) -> str:
    """Create results directory for Phase 4"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_phase4_training/{model_name}_{timestamp}"
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

def create_fold_dataloaders(full_dataset, train_indices, val_indices, batch_size=1):
    """Create dataloaders for a specific CV fold"""
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    # Create weighted sampler for training
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
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, 
                            collate_fn=mil_collate_fn, num_workers=0)
    
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, 
                          collate_fn=mil_collate_fn, num_workers=0)
    
    return train_loader, val_loader

def create_lr_scheduler(optimizer, schedule_type: str, warmup_epochs: int):
    """Create learning rate scheduler with optional warmup"""
    if schedule_type == 'cosine':
        main_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=TRAINING_CONFIG['max_epochs'] - warmup_epochs,
            eta_min=SCHEDULE_PARAMS['cosine']['eta_min']
        )
    elif schedule_type == 'reduce_plateau':
        main_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=SCHEDULE_PARAMS['reduce_plateau']['mode'],
            patience=SCHEDULE_PARAMS['reduce_plateau']['patience'],
            factor=SCHEDULE_PARAMS['reduce_plateau']['factor'],
            min_lr=SCHEDULE_PARAMS['reduce_plateau']['min_lr']
        )
    elif schedule_type == 'none':
        main_scheduler = None
    else:
        main_scheduler = None
    
    return main_scheduler, warmup_epochs

def train_epoch(model, train_loader, criterion, optimizer, device, gradient_clip=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
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
        
        # Gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    val_evaluator = RegressionEvaluator()
    running_loss = 0.0
    
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
            
            loss = criterion(prediction, rs_target)
            running_loss += loss.item()
            
            val_evaluator.update(
                targets=rs_target.cpu().numpy(),
                preds=prediction.cpu().numpy()
            )
    
    val_metrics = val_evaluator.compute_all_metrics(verbose=False)
    val_metrics['loss'] = running_loss / len(val_loader)
    
    return val_metrics

def train_with_config(model_config, train_config, train_loader, val_loader, device):
    """Train model with specific configuration - WITH EARLY STOPPING"""
    
    # Create model
    model = CLAM(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["clam_params"]["hidden_dim"],
        attention_hidden_dim=model_config["clam_params"]["attention_hidden_dim"],
        dropout=model_config["clam_params"]["dropout"],
        gate=model_config["clam_params"]["gate"]
    ).to(device)
    
    # Create criterion
    criterion = nn.HuberLoss(delta=train_config['huber_delta'])
    
    # Create optimizer - support both Adam and AdamW
    if train_config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config['base_lr'],
            weight_decay=train_config['weight_decay']
        )
    else:  # AdamW
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config['base_lr'],
            weight_decay=train_config['weight_decay']
        )
    
    # Create scheduler
    scheduler, warmup_epochs = create_lr_scheduler(
        optimizer, 
        train_config['lr_schedule'],
        train_config['warmup_epochs']
    )
    
    # Training loop with early stopping
    best_auroc = 0.0
    best_metrics = None
    patience_counter = 0
    actual_epochs = 0
    
    for epoch in range(1, TRAINING_CONFIG['max_epochs'] + 1):
        actual_epochs = epoch
        
        # Warmup handling
        if epoch <= warmup_epochs:
            warmup_lr = train_config['base_lr'] * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            gradient_clip=train_config['gradient_clip_value']
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        if epoch > warmup_epochs and scheduler is not None:
            if train_config['lr_schedule'] == 'reduce_plateau':
                scheduler.step(val_metrics['auroc'])
            else:
                scheduler.step()
        
        # Track best model
        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            best_metrics = val_metrics.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
            break
    
    return best_auroc, best_metrics, actual_epochs

def run_hyperparameter_search(model_config, param_grid, cv_splits, full_dataset, device):
    """Run grid search for training hyperparameters"""
    
    # Generate parameter combinations
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    print(f"Total training configurations: {len(param_combinations)}")
    print(f"Using {TRAINING_CONFIG['inner_cv_folds']}-fold inner CV")
    print(f"Max epochs: {TRAINING_CONFIG['max_epochs']}, Early stopping patience: {TRAINING_CONFIG['early_stopping_patience']}")
    
    all_results = []
    
    # Progress bar
    pbar = tqdm(param_combinations, desc="Training Grid Search")
    
    for config_idx, train_config in enumerate(pbar):
        # Run inner CV
        fold_scores = []
        fold_details = []
        fold_epochs = []
        
        for fold_idx in range(TRAINING_CONFIG['inner_cv_folds']):
            train_idx, val_idx = cv_splits[fold_idx]
            
            # Create dataloaders
            train_loader, val_loader = create_fold_dataloaders(
                full_dataset, train_idx, val_idx, TRAINING_CONFIG['batch_size']
            )
            
            # Train with configuration
            fold_auroc, fold_metrics, epochs_run = train_with_config(
                model_config, train_config, train_loader, val_loader, device
            )
            
            fold_scores.append(fold_auroc)
            fold_details.append(fold_metrics)
            fold_epochs.append(epochs_run)
        
        # Aggregate results
        result = {
            'config_idx': config_idx,
            'avg_auroc': np.mean(fold_scores),
            'std_auroc': np.std(fold_scores),
            'avg_epochs': np.mean(fold_epochs),
            'min_epochs': np.min(fold_epochs),
            'max_epochs': np.max(fold_epochs),
            **train_config
        }
        
        # Add other metrics
        for metric in ['r2', 'rmse', 'f1_score', 'binary_accuracy']:
            values = [f.get(metric, 0) for f in fold_details]
            result[f'avg_{metric}'] = np.mean(values)
            result[f'std_{metric}'] = np.std(values)
        
        all_results.append(result)
        
        # Update progress bar with early stopping info
        if config_idx == 0 or result['avg_auroc'] > max([r['avg_auroc'] for r in all_results[:-1]], default=0):
            pbar.set_postfix({
                'Best_AUROC': f"{result['avg_auroc']:.4f}",
                'Epochs': f"{result['avg_epochs']:.1f}"
            })
    
    return all_results

def analyze_results(all_results, model_name, model_config, results_dir):
    """Analyze grid search results"""
    
    results_df = pd.DataFrame(all_results)
    results_df_sorted = results_df.sort_values('avg_auroc', ascending=False)
    
    # Find best configuration
    best_result = results_df_sorted.iloc[0]
    
    print(f"\n{'='*80}")
    print(f"TRAINING OPTIMIZATION RESULTS: {model_name.upper()}")
    print(f"{'='*80}")
    
    print(f"Total configurations tested: {len(results_df)}")
    print(f"Best AUROC: {best_result['avg_auroc']:.4f} ± {best_result['std_auroc']:.4f}")
    print(f"Best config stopped at epoch: {best_result['avg_epochs']:.1f} (range: {best_result['min_epochs']}-{best_result['max_epochs']})")
    print(f"Phase 3 Baseline: {model_config['baseline_auroc']:.4f}")
    
    # Check if Phase 3 settings are in top results
    phase3_match = results_df[
        (results_df['optimizer'] == 'Adam') &
        (results_df['base_lr'] == model_config['phase3_training']['learning_rate']) &
        (results_df['lr_schedule'] == 'none') &
        (results_df['weight_decay'] == model_config['phase3_training']['weight_decay']) &
        (results_df['huber_delta'] == 5.0)
    ]
    
    if not phase3_match.empty:
        phase3_rank = (results_df_sorted.index == phase3_match.index[0]).argmax() + 1
        print(f"Phase 3 configuration rank: {phase3_rank}/{len(results_df)}")
        print(f"Phase 3 configuration AUROC: {phase3_match.iloc[0]['avg_auroc']:.4f}")
        print(f"Phase 3 config epochs: {phase3_match.iloc[0]['avg_epochs']:.1f}")
    
    print(f"\nBEST TRAINING CONFIGURATION:")
    for param in ['optimizer', 'base_lr', 'lr_schedule', 'weight_decay', 
                  'warmup_epochs', 'gradient_clip_value', 'huber_delta']:
        print(f"  {param}: {best_result[param]}")
    
    print(f"\nBEST CONFIGURATION METRICS:")
    print(f"  AUROC: {best_result['avg_auroc']:.4f} ± {best_result['std_auroc']:.4f}")
    print(f"  R²: {best_result.get('avg_r2', 0):.4f} ± {best_result.get('std_r2', 0):.4f}")
    print(f"  RMSE: {best_result.get('avg_rmse', 0):.4f} ± {best_result.get('std_rmse', 0):.4f}")
    print(f"  F1: {best_result.get('avg_f1_score', 0):.4f} ± {best_result.get('std_f1_score', 0):.4f}")
    
    # Top 5 configurations with epochs info
    print(f"\nTOP 5 CONFIGURATIONS:")
    print(f"{'Rank':<5} {'AUROC':<12} {'Epochs':<8} {'Opt':<6} {'LR':<8} {'Sched':<12} {'WD':<8} {'Clip':<6} {'Huber':<6}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(results_df_sorted.head(5).iterrows(), 1):
        early_stop_indicator = "*" if row['avg_epochs'] < TRAINING_CONFIG['max_epochs'] else ""
        print(f"{i:<5} {row['avg_auroc']:.3f}±{row['std_auroc']:.3f}  "
              f"{row['avg_epochs']:.0f}{early_stop_indicator:<7} "
              f"{row['optimizer']:<6} {row['base_lr']:<8.0e} {row['lr_schedule']:<12} "
              f"{row['weight_decay']:<8.0e} {str(row['gradient_clip_value']):<6} {row['huber_delta']:<6.1f}")
    
    print("\n* = Early stopped before max epochs")
    
    # Statistics on early stopping
    early_stopped = results_df[results_df['avg_epochs'] < TRAINING_CONFIG['max_epochs']]
    print(f"\nEarly stopping statistics:")
    print(f"  Configurations that early stopped: {len(early_stopped)}/{len(results_df)} ({len(early_stopped)/len(results_df)*100:.1f}%)")
    if len(early_stopped) > 0:
        print(f"  Average epochs when early stopped: {early_stopped['avg_epochs'].mean():.1f}")
    
    # Save results
    results_df_sorted.to_csv(os.path.join(results_dir, "training_optimization_results.csv"), index=False)
    
    # Extract best configuration
    best_config = {}
    for param in ['optimizer', 'base_lr', 'lr_schedule', 'weight_decay', 
                  'warmup_epochs', 'gradient_clip_value', 'huber_delta']:
        value = best_result[param]
        if isinstance(value, np.number):
            best_config[param] = float(value) if value is not None else None
        else:
            best_config[param] = value
    
    with open(os.path.join(results_dir, "best_training_config.json"), 'w') as f:
        json.dump(best_config, f, indent=2, cls=NumpyEncoder)
    
    return best_config, results_df_sorted

def run_final_validation(model_name, model_config, best_train_config, cv_splits, 
                         full_dataset, device, results_dir):
    """Run final 5-fold CV with best training configuration"""
    
    print(f"\n{'='*80}")
    print(f"FINAL VALIDATION: {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Max epochs: {TRAINING_CONFIG['max_epochs']}, Early stopping patience: {TRAINING_CONFIG['early_stopping_patience']}")
    
    final_results = []
    epochs_list = []
    
    for fold in range(TRAINING_CONFIG['final_cv_folds']):
        train_idx, val_idx = cv_splits[fold]
        
        print(f"\nFold {fold+1}/{TRAINING_CONFIG['final_cv_folds']}")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        # Create dataloaders
        train_loader, val_loader = create_fold_dataloaders(
            full_dataset, train_idx, val_idx, TRAINING_CONFIG['batch_size']
        )
        
        # Train with best configuration
        fold_auroc, fold_metrics, epochs_run = train_with_config(
            model_config, best_train_config, train_loader, val_loader, device
        )
        
        fold_metrics['fold'] = fold + 1
        fold_metrics['epochs_run'] = epochs_run
        final_results.append(fold_metrics)
        epochs_list.append(epochs_run)
        
        early_stop_msg = f" (early stopped)" if epochs_run < TRAINING_CONFIG['max_epochs'] else ""
        print(f"  AUROC: {fold_metrics['auroc']:.4f}, R²: {fold_metrics.get('r2', 0):.4f}, "
              f"RMSE: {fold_metrics.get('rmse', 0):.4f}, Epochs: {epochs_run}{early_stop_msg}")
    
    # Calculate final statistics
    final_df = pd.DataFrame(final_results)
    
    final_summary = {
        'model': model_name,
        'phase3_baseline_auroc': model_config['baseline_auroc'],
        'phase4_auroc_mean': final_df['auroc'].mean(),
        'phase4_auroc_std': final_df['auroc'].std(),
        'phase4_r2_mean': final_df['r2'].mean(),
        'phase4_r2_std': final_df['r2'].std(),
        'phase4_rmse_mean': final_df['rmse'].mean(),
        'phase4_rmse_std': final_df['rmse'].std(),
        'phase4_f1_mean': final_df['f1_score'].mean(),
        'phase4_f1_std': final_df['f1_score'].std(),
        'avg_epochs': np.mean(epochs_list),
        'improvement': final_df['auroc'].mean() - model_config['baseline_auroc'],
        'best_training_config': best_train_config
    }
    
    print(f"\n{'='*60}")
    print(f"FINAL PHASE 4 PERFORMANCE")
    print(f"{'='*60}")
    print(f"AUROC: {final_summary['phase4_auroc_mean']:.4f} ± {final_summary['phase4_auroc_std']:.4f}")
    print(f"R²: {final_summary['phase4_r2_mean']:.4f} ± {final_summary['phase4_r2_std']:.4f}")
    print(f"RMSE: {final_summary['phase4_rmse_mean']:.4f} ± {final_summary['phase4_rmse_std']:.4f}")
    print(f"F1: {final_summary['phase4_f1_mean']:.4f} ± {final_summary['phase4_f1_std']:.4f}")
    print(f"Average epochs: {final_summary['avg_epochs']:.1f}")
    print(f"Improvement from Phase 3: {final_summary['improvement']:+.4f}")
    
    # Save results
    final_df.to_csv(os.path.join(results_dir, "final_validation_results.csv"), index=False)
    
    with open(os.path.join(results_dir, "final_summary.json"), 'w') as f:
        json.dump(final_summary, f, indent=2, cls=NumpyEncoder)
    
    return final_df, final_summary

def main():
    """Main execution"""
    
    parser = argparse.ArgumentParser(description='Phase 4: Training Optimization')
    parser.add_argument('--model', type=str, required=True, 
                      choices=['virchow2', 'resnet18'],
                      help='Model to optimize')
    parser.add_argument('--skip_search', action='store_true',
                      help='Skip grid search and use default config')
    args = parser.parse_args()
    
    # Setup
    model_name = args.model
    model_config = PHASE3_BEST_CONFIGS[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(TRAINING_CONFIG['random_seed'])
    
    print(f"{'='*80}")
    print(f"PHASE 4: TRAINING OPTIMIZATION FOR {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Phase 3 Baseline AUROC: {model_config['baseline_auroc']:.4f}")
    print(f"Phase 3 used: Adam, LR={model_config['phase3_training']['learning_rate']}, "
          f"WD={model_config['phase3_training']['weight_decay']}, no scheduler")
    print(f"Training: Max {TRAINING_CONFIG['max_epochs']} epochs with early stopping (patience={TRAINING_CONFIG['early_stopping_patience']})")
    
    # Create results directory
    results_dir = setup_results_directory(model_name)
    
    # Save configuration
    with open(os.path.join(results_dir, "phase4_config.json"), 'w') as f:
        config_save = {
            'model_config': model_config,
            'training_grid': TRAINING_OPTIMIZATION_GRID,
            'training_config': TRAINING_CONFIG
        }
        json.dump(config_save, f, indent=2, cls=NumpyEncoder)
    
    # Load dataset
    print("\nLoading dataset and creating CV splits...")
    full_dataset, cv_splits, binary_labels = create_cv_dataset_and_splits(
        model_config["combined_csv"],
        max(TRAINING_CONFIG['inner_cv_folds'], TRAINING_CONFIG['final_cv_folds']),
        TRAINING_CONFIG['random_seed'],
        TRAINING_CONFIG['stratify_threshold']
    )
    
    print(f"Dataset: {len(full_dataset)} samples")
    print(f"Class distribution: {np.sum(binary_labels == 0)} low-risk, {np.sum(binary_labels == 1)} high-risk")
    
    if not args.skip_search:
        # Stage 1: Grid search
        print(f"\n{'='*60}")
        print("STAGE 1: TRAINING HYPERPARAMETER SEARCH")
        print(f"{'='*60}")
        
        all_results = run_hyperparameter_search(
            model_config, TRAINING_OPTIMIZATION_GRID, cv_splits, full_dataset, device
        )
        
        # Save raw results immediately
        with open(os.path.join(results_dir, "grid_search_raw_results.json"), 'w') as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)
        
        # Stage 2: Analysis
        print(f"\n{'='*60}")
        print("STAGE 2: ANALYSIS")
        print(f"{'='*60}")
        
        best_train_config, results_df = analyze_results(all_results, model_name, model_config, results_dir)
    else:
        # Use default configuration
        best_train_config = {
            'optimizer': 'Adam',
            'base_lr': 1e-4,
            'lr_schedule': 'cosine',
            'weight_decay': 1e-4,
            'warmup_epochs': 5,
            'gradient_clip_value': 1.0,
            'huber_delta': 5.0
        }
        print("Using default training configuration (skipping search)")
    
    # Stage 3: Final validation
    print(f"\n{'='*60}")
    print("STAGE 3: FINAL VALIDATION WITH BEST CONFIGURATION")
    print(f"{'='*60}")
    
    final_df, final_summary = run_final_validation(
        model_name, model_config, best_train_config, cv_splits,
        full_dataset, device, results_dir
    )
    
    print(f"\n{'='*80}")
    print(f"PHASE 4 COMPLETE FOR {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Results saved to: {results_dir}")
    
    return final_summary

if __name__ == "__main__":
    main()