# train_reg/train_acmil_all.py - Run ACMIL for all foundation models automatically

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
from models.regression_model import ACMIL, ACMIL_CLAM_Hybrid
from utils.mil_utils import mil_collate_fn
from utils.regression_evaluator import RegressionEvaluator
from utils.training_helpers import set_seed

# All foundation models to test
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

# Base configuration
BASE_CONFIG = {
    "hidden_dim": 128,
    "attention_hidden_dim": 128,
    
    # ACMIL parameters (based on stability test)
    "n_branches": 5,
    "n_masked_patch": 10,
    "mask_ratio": 0.0,  # Based on your stability test results
    
    "batch_size": 1,
    "lr": 1e-4,
    "num_epochs": 15,
    "n_folds": 5,
    "use_class_balancing": True,
    "loss_type": "huber",
    "huber_delta": 5.0,
    "random_seed": 42,
    "stratify_threshold": 25.0,
    
    # Which ACMIL variants to test
    "test_acmil": True,
    "test_acmil_clam": True
}

def run_acmil_for_single_model(feature_extractor, model_config, base_config, device):
    """Run ACMIL experiments for a single foundation model"""
    
    print(f"\n{'='*80}")
    print(f"Testing {feature_extractor.upper()}")
    print(f"Input dimension: {model_config['input_dim']}")
    print(f"{'='*80}")
    
    # Load dataset
    full_dataset = RegressionMILDataset(model_config["combined_csv"])
    
    # Extract RS scores for stratification
    rs_scores = []
    for i in range(len(full_dataset)):
        _, rs_score = full_dataset[i]
        rs_scores.append(rs_score)
    rs_scores = np.array(rs_scores)
    
    # Create binary labels for stratification
    binary_labels = (rs_scores >= base_config["stratify_threshold"]).astype(int)
    
    # Create CV splits
    skf = StratifiedKFold(n_splits=base_config["n_folds"], shuffle=True, 
                         random_state=base_config["random_seed"])
    cv_splits = list(skf.split(range(len(full_dataset)), binary_labels))
    
    print(f"Total samples: {len(full_dataset)}")
    print(f"Class distribution: {np.sum(binary_labels == 0)} low-risk, {np.sum(binary_labels == 1)} high-risk")
    
    model_results = []
    
    # Test ACMIL variants
    models_to_test = []
    if base_config["test_acmil"]:
        models_to_test.append(("acmil", ACMIL))
    if base_config["test_acmil_clam"]:
        models_to_test.append(("acmil_clam", ACMIL_CLAM_Hybrid))
    
    for model_name, model_class in models_to_test:
        print(f"\n  Testing {model_name}...")
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            # Create dataloaders
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)
            
            if base_config["use_class_balancing"]:
                class TempDataset:
                    def __init__(self, indices, full_dataset):
                        self.indices = indices
                        self.full_dataset = full_dataset
                    def __len__(self):
                        return len(self.indices)
                    def __getitem__(self, idx):
                        return self.full_dataset[self.indices[idx]]
                
                temp_train = TempDataset(train_idx, full_dataset)
                sampler = create_rs_weighted_sampler(temp_train, boundary_focus=False, 
                                                    class_balance=True, threshold=25.0)
                train_loader = DataLoader(train_subset, batch_size=base_config["batch_size"],
                                        sampler=sampler, collate_fn=mil_collate_fn, num_workers=0)
            else:
                train_loader = DataLoader(train_subset, batch_size=base_config["batch_size"],
                                        shuffle=True, collate_fn=mil_collate_fn, num_workers=0)
            
            val_loader = DataLoader(val_subset, batch_size=base_config["batch_size"],
                                  shuffle=False, collate_fn=mil_collate_fn, num_workers=0)
            
            # Create model
            if model_name == "acmil":
                model = ACMIL(
                    input_dim=model_config["input_dim"],
                    hidden_dim=base_config["hidden_dim"],
                    n_branches=base_config["n_branches"],
                    n_masked_patch=base_config["n_masked_patch"],
                    mask_ratio=base_config["mask_ratio"],
                    dropout=0.25
                ).to(device)
            else:  # acmil_clam
                model = ACMIL_CLAM_Hybrid(
                    input_dim=model_config["input_dim"],
                    hidden_dim=base_config["hidden_dim"],
                    attention_hidden_dim=base_config["attention_hidden_dim"],
                    n_branches=base_config["n_branches"],
                    n_masked_patch=base_config["n_masked_patch"],
                    mask_ratio=base_config["mask_ratio"],
                    dropout=0.25,
                    gate=True
                ).to(device)
            
            # Loss and optimizer
            criterion = nn.HuberLoss(delta=base_config["huber_delta"])
            optimizer = optim.Adam(model.parameters(), lr=base_config["lr"])
            
            # Training
            train_evaluator = RegressionEvaluator()
            val_evaluator = RegressionEvaluator()
            best_auroc = 0
            best_metrics = None
            
            # Progress bar for epochs
            for epoch in range(1, base_config["num_epochs"] + 1):
                # Train
                model.train()
                train_evaluator.reset()
                
                for features, rs_target in train_loader:
                    if isinstance(features, list):
                        features = features[0]
                    features = features.to(device)
                    
                    if isinstance(rs_target, list):
                        rs_target = rs_target[0]
                    if not isinstance(rs_target, torch.Tensor):
                        rs_target = torch.tensor([rs_target], dtype=torch.float32, device=device)
                    else:
                        rs_target = rs_target.to(device)
                        if rs_target.dim() == 0:
                            rs_target = rs_target.unsqueeze(0)
                    
                    optimizer.zero_grad()
                    prediction = model(features, return_branch_outputs=False)
                    if prediction.dim() == 0:
                        prediction = prediction.unsqueeze(0)
                    
                    loss = criterion(prediction, rs_target)
                    loss.backward()
                    optimizer.step()
                    
                    train_evaluator.update(rs_target.cpu().numpy(), prediction.detach().cpu().numpy())
                
                # Validate
                model.eval()
                val_evaluator.reset()
                
                with torch.no_grad():
                    for features, rs_target in val_loader:
                        if isinstance(features, list):
                            features = features[0]
                        features = features.to(device)
                        
                        if isinstance(rs_target, list):
                            rs_target = rs_target[0]
                        if not isinstance(rs_target, torch.Tensor):
                            rs_target = torch.tensor([rs_target], dtype=torch.float32, device=device)
                        else:
                            rs_target = rs_target.to(device)
                            if rs_target.dim() == 0:
                                rs_target = rs_target.unsqueeze(0)
                        
                        prediction = model(features, return_branch_outputs=False)
                        if prediction.dim() == 0:
                            prediction = prediction.unsqueeze(0)
                        
                        val_evaluator.update(rs_target.cpu().numpy(), prediction.cpu().numpy())
                
                val_metrics = val_evaluator.compute_all_metrics(verbose=False)
                
                if val_metrics.get('auroc', 0) > best_auroc:
                    best_auroc = val_metrics.get('auroc', 0)
                    best_metrics = val_metrics.copy()
            
            fold_results.append(best_metrics)
            print(f"    Fold {fold+1}/{base_config['n_folds']}: AUROC={best_auroc:.4f}")
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in ['auroc', 'r2', 'rmse', 'mae', 'binary_accuracy', 'c_index']:
            values = [f.get(metric, 0) for f in fold_results]
            avg_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        print(f"  {model_name.upper()} Results:")
        print(f"    AUROC: {avg_metrics['auroc']['mean']:.4f} ± {avg_metrics['auroc']['std']:.4f}")
        print(f"    R²: {avg_metrics['r2']['mean']:.4f} ± {avg_metrics['r2']['std']:.4f}")
        print(f"    C-index: {avg_metrics['c_index']['mean']:.4f} ± {avg_metrics['c_index']['std']:.4f}")
        
        model_results.append({
            'feature_extractor': feature_extractor,
            'mil_architecture': model_name,
            'auroc_mean': avg_metrics['auroc']['mean'],
            'auroc_std': avg_metrics['auroc']['std'],
            'r2_mean': avg_metrics['r2']['mean'],
            'r2_std': avg_metrics['r2']['std'],
            'rmse_mean': avg_metrics['rmse']['mean'],
            'rmse_std': avg_metrics['rmse']['std'],
            'mae_mean': avg_metrics['mae']['mean'],
            'mae_std': avg_metrics['mae']['std'],
            'c_index_mean': avg_metrics['c_index']['mean'],
            'c_index_std': avg_metrics['c_index']['std'],
            'binary_accuracy_mean': avg_metrics['binary_accuracy']['mean'],
            'binary_accuracy_std': avg_metrics['binary_accuracy']['std']
        })
    
    return model_results

def run_all_acmil_experiments():
    """Run ACMIL experiments for all foundation models"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(BASE_CONFIG["random_seed"])
    
    print(f"{'='*100}")
    print(f"ACMIL TESTING FOR ALL FOUNDATION MODELS")
    print(f"{'='*100}")
    print(f"Device: {device}")
    print(f"Foundation models to test: {list(FOUNDATION_MODELS.keys())}")
    print(f"ACMIL Settings: n_branches={BASE_CONFIG['n_branches']}, mask_ratio={BASE_CONFIG['mask_ratio']}")
    print(f"Cross-validation: {BASE_CONFIG['n_folds']}-fold")
    
    all_results = []
    
    # Test each foundation model
    for idx, (feature_extractor, model_config) in enumerate(FOUNDATION_MODELS.items(), 1):
        print(f"\n{'#'*100}")
        print(f"FOUNDATION MODEL {idx}/{len(FOUNDATION_MODELS)}: {feature_extractor.upper()}")
        print(f"{'#'*100}")
        
        try:
            model_results = run_acmil_for_single_model(
                feature_extractor, model_config, BASE_CONFIG, device
            )
            all_results.extend(model_results)
        except Exception as e:
            print(f"ERROR with {feature_extractor}: {str(e)}")
            continue
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('auroc_mean', ascending=False)
    
    os.makedirs("results_acmil", exist_ok=True)
    results_df.to_csv("results_acmil/all_models_acmil_results.csv", index=False)
    
    # Print summary
    print(f"\n{'='*100}")
    print(f"ACMIL TESTING COMPLETE - SUMMARY")
    print(f"{'='*100}")
    
    print(f"\nTOP RESULTS (by AUROC):")
    print(f"{'Rank':<5} {'Foundation':<12} {'MIL':<12} {'AUROC':<15} {'R²':<15} {'C-index':<15}")
    print(f"{'-'*80}")
    
    for i, row in enumerate(results_df.head(8).iterrows(), 1):
        r = row[1]
        print(f"{i:<5} {r['feature_extractor']:<12} {r['mil_architecture']:<12} "
              f"{r['auroc_mean']:.3f}±{r['auroc_std']:.3f}   "
              f"{r['r2_mean']:.3f}±{r['r2_std']:.3f}   "
              f"{r['c_index_mean']:.3f}±{r['c_index_std']:.3f}")
    
    print(f"\nResults saved to: results_acmil/all_models_acmil_results.csv")
    
    return results_df

if __name__ == "__main__":
    results = run_all_acmil_experiments()