# run_final_validation.py
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
from datasets.regression_mil_dataset import RegressionMILDataset, create_rs_weighted_sampler
from models.regression_model import CLAM
from utils.mil_utils import mil_collate_fn
from utils.regression_evaluator import RegressionEvaluator
from utils.training_helpers import set_seed

# Custom JSON encoder for numpy types
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

# Best hyperparameters from grid search results
BEST_HYPERPARAMS = {
    "resnet18": {
        "hidden_dim": 256,
        "attention_hidden_dim": 512,
        "gate": False,
        "learning_rate": 0.0002,
        "dropout": 0.25,
        "weight_decay": 0.0
    },
    "uni2-h": {
        "hidden_dim": 512,
        "attention_hidden_dim": 256,
        "gate": False,
        "learning_rate": 0.0002,
        "dropout": 0.5,
        "weight_decay": 0.0
    },
    "virchow2": {
        "hidden_dim": 512,
        "attention_hidden_dim": 512,
        "gate": False,
        "learning_rate": 0.0002,
        "dropout": 0.25,
        "weight_decay": 0.0
    },
    "h-optimus": {
        "hidden_dim": 512,
        "attention_hidden_dim": 512,
        "gate": True,
        "learning_rate": 0.0001,
        "dropout": 0.5,
        "weight_decay": 0.0001
    }
}

def create_fold_dataloaders(full_dataset, train_indices, val_indices, use_class_balancing=True):
    """Create dataloaders for a specific CV fold"""
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    if use_class_balancing:
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
    set_seed(42)
    
    # Create CLAM model with current hyperparameters
    model = CLAM(
        input_dim=model_config["input_dim"],
        hidden_dim=hyperparams["hidden_dim"],
        attention_hidden_dim=hyperparams["attention_hidden_dim"],
        dropout=hyperparams["dropout"],
        gate=hyperparams["gate"]
    ).to(device)
    
    # Setup training components
    criterion = nn.HuberLoss(delta=5.0)
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"]
    )
    
    # Training loop with fixed epochs
    best_val_auroc = 0.0
    best_metrics = None
    
    for epoch in range(1, 16):  # Fixed 15 epochs
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

def run_final_validation_for_model(model_name):
    """Run final validation for a specific model"""
    
    if model_name not in FOUNDATION_MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available options: {list(FOUNDATION_MODELS.keys())}")
        return
    
    if model_name not in BEST_HYPERPARAMS:
        print(f"Error: No hyperparameters defined for '{model_name}'")
        return
    
    # Setup
    model_config = FOUNDATION_MODELS[model_name].copy()
    model_config['name'] = model_name
    best_hyperparams = BEST_HYPERPARAMS[model_name]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_phase3_final_validation/{model_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"{'='*80}")
    print(f"FINAL VALIDATION FOR {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Input Dimension: {model_config['input_dim']}")
    print(f"Baseline AUROC: {model_config['baseline_auroc']:.4f}")
    print(f"\nUsing best hyperparameters from grid search:")
    for k, v in best_hyperparams.items():
        print(f"  {k}: {v}")
    
    # Load dataset and create CV splits
    print(f"\nLoading dataset from: {model_config['combined_csv']}")
    
    # Create dataset
    full_dataset = RegressionMILDataset(model_config["combined_csv"])
    
    # Get labels for stratification
    rs_scores = []
    for i in range(len(full_dataset)):
        _, rs_score = full_dataset[i]
        rs_scores.append(rs_score)
    
    rs_scores = np.array(rs_scores)
    binary_labels = (rs_scores >= 25.0).astype(int)
    
    # Create 5-fold stratified splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(skf.split(range(len(full_dataset)), binary_labels))
    
    print(f"Dataset: {len(full_dataset)} samples")
    print(f"Class distribution: {np.sum(binary_labels == 0)} low-risk, {np.sum(binary_labels == 1)} high-risk")
    
    # Run 5-fold validation
    final_results = []
    
    for fold in range(5):
        train_idx, val_idx = cv_splits[fold]
        
        print(f"\nFold {fold+1}/5")
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
        fold_metrics['model'] = model_name
        final_results.append(fold_metrics)
        
        print(f"  Results: AUROC={fold_metrics.get('auroc', 0):.4f}, "
              f"R²={fold_metrics.get('r2', 0):.4f}, "
              f"RMSE={fold_metrics.get('rmse', 0):.4f}, "
              f"F1={fold_metrics.get('f1_score', 0):.4f}")
    
    # Calculate final statistics
    final_df = pd.DataFrame(final_results)
    
    # Summary statistics
    final_summary = {
        'model': model_name,
        'final_auroc_mean': float(final_df['auroc'].mean()),
        'final_auroc_std': float(final_df['auroc'].std()),
        'final_r2_mean': float(final_df['r2'].mean()),
        'final_r2_std': float(final_df['r2'].std()),
        'final_f1_mean': float(final_df['f1_score'].mean()),
        'final_f1_std': float(final_df['f1_score'].std()),
        'final_rmse_mean': float(final_df['rmse'].mean()),
        'final_rmse_std': float(final_df['rmse'].std()),
        'baseline_auroc': float(model_config['baseline_auroc']),
        'final_improvement': float(final_df['auroc'].mean() - model_config['baseline_auroc']),
        'best_hyperparams': best_hyperparams
    }
    
    print(f"\n{'='*80}")
    print(f"FINAL OPTIMIZED PERFORMANCE")
    print(f"{'='*80}")
    print(f"AUROC: {final_summary['final_auroc_mean']:.4f} ± {final_summary['final_auroc_std']:.4f}")
    print(f"R²: {final_summary['final_r2_mean']:.4f} ± {final_summary['final_r2_std']:.4f}")
    print(f"F1: {final_summary['final_f1_mean']:.4f} ± {final_summary['final_f1_std']:.4f}")
    print(f"RMSE: {final_summary['final_rmse_mean']:.4f} ± {final_summary['final_rmse_std']:.4f}")
    print(f"Baseline AUROC: {final_summary['baseline_auroc']:.4f}")
    print(f"Total Improvement: +{final_summary['final_improvement']:.4f} "
          f"({final_summary['final_improvement']/final_summary['baseline_auroc']*100:+.1f}%)")
    
    # Save results
    final_df.to_csv(os.path.join(results_dir, "final_validation_results.csv"), index=False)
    
    with open(os.path.join(results_dir, "final_summary.json"), 'w') as f:
        json.dump(final_summary, f, indent=2, cls=NumpyEncoder)
    
    # Save best hyperparameters separately
    with open(os.path.join(results_dir, "best_hyperparameters.json"), 'w') as f:
        json.dump(best_hyperparams, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nAll results saved to: {results_dir}")
    
    return final_summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run final validation with best hyperparameters')
    parser.add_argument('--model', type=str, required=True,
                      choices=['resnet18', 'uni2-h', 'virchow2', 'h-optimus'],
                      help='Model to run final validation for')
    
    args = parser.parse_args()
    
    # If you need to update hyperparameters for virchow2, do it here before running:
    if args.model == 'virchow2':
        # UPDATE THESE with your actual virchow2 best hyperparameters from grid search
        print("NOTE: Please update virchow2 hyperparameters in the script if you have the actual values")
        # You can manually set them here if needed:
        # BEST_HYPERPARAMS['virchow2'] = {
        #     "hidden_dim": YOUR_VALUE,
        #     "attention_hidden_dim": YOUR_VALUE,
        #     "gate": YOUR_VALUE,
        #     "learning_rate": YOUR_VALUE,
        #     "dropout": YOUR_VALUE,
        #     "weight_decay": YOUR_VALUE
        # }
    
    run_final_validation_for_model(args.model)