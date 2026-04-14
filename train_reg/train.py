# train_reg_flexible.py
# Flexible Regression Training Script for RS Score Prediction with Multiple MIL Architectures

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Project imports
from datasets.regression_mil_dataset import RegressionMILDataset, create_rs_weighted_sampler
from models.regression_model import MeanPoolingMIL, MaxPoolingMIL, AttentionMIL, CLAM, Attn_Net_Gated
from utils.mil_utils import mil_collate_fn
from utils.regression_evaluator import RegressionEvaluator
from utils.training_helpers import set_seed, print_experiment_header
from utils.experiment_summary import save_experiment_summary, save_quick_comparison_summary

# Feature extractor configurations
FEATURE_EXTRACTORS = {
    "resnet18": {
        "input_dim": 512,
        "train_csv": "data/manifests/train_features_resnet18.csv",
        "val_csv": "data/manifests/val_features_resnet18.csv"
    },
    "resnet50": {
        "input_dim": 2048,
        "train_csv": "data/manifests/train_features_resnet50.csv",
        "val_csv": "data/manifests/val_features_resnet50.csv"
    },
    "conch": {
        "input_dim": 512,
        "train_csv": "data/manifests/train_features_conch.csv",
        "val_csv": "data/manifests/val_features_conch.csv"
    },
    "uni2-h": {
        "input_dim": 1536,
        "train_csv": "data/manifests/train_features_uni2-h.csv",
        "val_csv": "data/manifests/val_features_uni2-h.csv"
    },
    "virchow2": {
        "input_dim": 1280,
        "train_csv": "data/manifests/train_features_virchow2.csv",
        "val_csv": "data/manifests/val_features_virchow2.csv"
    },
    "h-optimus": {
        "input_dim": 1536,
        "train_csv": "data/manifests/train_features_h-optimus.csv",
        "val_csv": "data/manifests/val_features_h-optimus.csv"
    }
}

# MIL Model configurations
MIL_MODELS = {
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
        "description": "Clustering-constrained Attention MIL with gated attention"
    }
}


# Configuration - Change these to switch models and pooling
CONFIG = {
    "feature_extractor": "resnet50",  # CHANGE THIS TO SWITCH FEATURE EXTRACTORS
    "mil_pooling": "mean",  # CHANGE THIS TO SWITCH MIL ARCHITECTURES: "mean", "max", "attention"
    "model_name": "regression_mil",
    "experiment_type": "mse_balanced", 
    "hidden_dim": 128,
    "attention_hidden_dim":128,  # For attention models
    "batch_size": 1,
    "lr": 1e-4,
    "num_epochs": 15,
    "use_class_balancing": True,
    "use_boundary_weighting": False,  
    "boundary_range": 10.0,
    "loss_type": "mse",
    "huber_delta": 5.0,
    "random_seed": 42
}

def get_feature_config(feature_extractor_name):
    """Get configuration for specified feature extractor"""
    if feature_extractor_name not in FEATURE_EXTRACTORS:
        available = list(FEATURE_EXTRACTORS.keys())
        raise ValueError(f"Unknown feature extractor: {feature_extractor_name}. "
                        f"Available options: {available}")
    
    return FEATURE_EXTRACTORS[feature_extractor_name]

def get_mil_model_config(mil_pooling_name):
    """Get MIL model configuration"""
    if mil_pooling_name not in MIL_MODELS:
        available = list(MIL_MODELS.keys())
        raise ValueError(f"Unknown MIL pooling method: {mil_pooling_name}. "
                        f"Available options: {available}")
    
    return MIL_MODELS[mil_pooling_name]

def setup_experiment_config(base_config):
    """Setup complete experiment configuration"""
    feature_name = base_config["feature_extractor"]
    mil_pooling = base_config["mil_pooling"]
    experiment_type = base_config["experiment_type"]
    
    # Get feature and MIL configurations
    feature_config = get_feature_config(feature_name)
    mil_config = get_mil_model_config(mil_pooling)
    
    # Merge configurations
    experiment_config = base_config.copy()
    experiment_config.update(feature_config)
    
    # Create hierarchical experiment structure
    experiment_config["method_name"] = f"{experiment_config['model_name']}_{mil_pooling}_{experiment_type}"
    experiment_config["feature_extractor_name"] = feature_name
    experiment_config["mil_model_class"] = mil_config["class"]
    experiment_config["mil_model_name"] = mil_config["name"]
    experiment_config["mil_model_description"] = mil_config["description"]
    
    return experiment_config

def setup_regression_directories(method_name: str, feature_extractor_name: str) -> str:
    """Create hierarchical results directory structure."""
    # Main method directory
    method_dir = f"results_regression/{method_name}"
    # Feature extractor subdirectory
    results_dir = os.path.join(method_dir, feature_extractor_name)
    
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    return results_dir

def save_experiment_config(config, results_dir):
    """Save experiment configuration for reproducibility"""
    import json
    
    config_path = os.path.join(results_dir, "experiment_config.json")
    
    # Convert any non-serializable values
    config_to_save = {}
    for key, value in config.items():
        if key == "mil_model_class":
            # Store class name instead of class object
            config_to_save[key] = value.__name__
        elif isinstance(value, (str, int, float, bool, list, dict)):
            config_to_save[key] = value
        else:
            config_to_save[key] = str(value)
    
    # Add system info
    config_to_save.update({
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    })
    
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    
    print(f"Experiment configuration saved to: {config_path}")
    return config_path

def create_regression_dataloaders(train_dataset, val_dataset, batch_size: int,
                                use_class_balancing: bool = True,
                                use_boundary_weighting: bool = True,
                                boundary_range: float = 10.0) -> tuple:
    """Create dataloaders for regression training."""
    
    if use_class_balancing or use_boundary_weighting:
        sampler = create_rs_weighted_sampler(
            train_dataset, 
            boundary_focus=use_boundary_weighting,
            class_balance=use_class_balancing, 
            threshold=25.0, 
            boundary_range=boundary_range
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=mil_collate_fn,
            num_workers=0
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=mil_collate_fn,
            num_workers=0
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=mil_collate_fn,
        num_workers=0
    )
    
    return train_loader, val_loader

def setup_regression_model_and_criterion(input_dim: int, hidden_dim: int, lr: float, 
                                       device: torch.device, loss_type: str = "huber",
                                       huber_delta: float = 5.0, mil_model_class=None,
                                       attention_hidden_dim: int = 64):
    """Initialize regression model, criterion, and optimizer with flexible MIL architecture."""
    
    # Create model - pass all relevant parameters
    model_kwargs = {
        'input_dim': input_dim, 
        'hidden_dim': hidden_dim,
        'attention_hidden_dim': attention_hidden_dim  # Always pass this, models will ignore if not needed
    }
    
    model = mil_model_class(**model_kwargs).to(device)
    print(f"Model: {mil_model_class.__name__}")
    
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    if mil_model_class.__name__ == 'AttentionMIL':
        print(f"Attention hidden dimension: {attention_hidden_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    if loss_type == "huber":
        criterion = nn.HuberLoss(delta=huber_delta)
        print(f"Loss: Huber Loss (delta={huber_delta})")
    elif loss_type == "mse":
        criterion = nn.MSELoss()
        print(f"Loss: MSE Loss")
    elif loss_type == "mae":
        criterion = nn.L1Loss()  # MAE
        print(f"Loss: MAE Loss")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    return model, criterion, optimizer

def train_regression_epoch(model, dataloader, criterion, optimizer, device: torch.device,
                         evaluator: RegressionEvaluator) -> float:
    """Train one epoch for regression."""
    model.train()
    running_loss = 0.0
    evaluator.reset()
    
    for features, rs_target in tqdm(dataloader, desc="Training"):
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
            if rs_target.dim() == 0:  # scalar tensor
                rs_target = rs_target.unsqueeze(0)
        
        optimizer.zero_grad()
        
        # Forward pass
        prediction = model(features)
        if prediction.dim() == 0:  # scalar prediction
            prediction = prediction.unsqueeze(0)
            
        loss = criterion(prediction, rs_target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        evaluator.update(
            targets=rs_target.cpu().numpy(),
            preds=prediction.detach().cpu().numpy(),
            losses=[loss.item()]
        )
    
    return running_loss / len(dataloader)

def validate_regression_epoch(model, dataloader, criterion, device: torch.device,
                           evaluator: RegressionEvaluator) -> float:
    """Validate one epoch for regression."""
    model.eval()
    running_loss = 0.0
    evaluator.reset()
    
    with torch.no_grad():
        for features, rs_target in tqdm(dataloader, desc="Validating"):
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
                if rs_target.dim() == 0:  # scalar tensor
                    rs_target = rs_target.unsqueeze(0)
            
            # Forward pass
            prediction = model(features)
            if prediction.dim() == 0:  # scalar prediction
                prediction = prediction.unsqueeze(0)
                
            loss = criterion(prediction, rs_target)
            
            # Update metrics
            running_loss += loss.item()
            evaluator.update(
                targets=rs_target.cpu().numpy(),
                preds=prediction.cpu().numpy(),
                losses=[loss.item()]
            )
    
    return running_loss / len(dataloader)

def print_regression_epoch_metrics(epoch: int, num_epochs: int, train_loss: float, 
                                  val_loss: float, train_metrics: dict, val_metrics: dict):
    """Print formatted epoch metrics for regression."""
    print(f"\nEpoch {epoch}/{num_epochs}")
    print(f"TRAIN - Loss: {train_loss:.4f} | RMSE: {train_metrics.get('rmse', 0):.3f} | "
          f"MAE: {train_metrics.get('mae', 0):.3f} | R²: {train_metrics.get('r2', 0):.3f} | "
          f"Spearman: {train_metrics.get('spearman_correlation', 0):.3f}")
    print(f"VAL   - Loss: {val_loss:.4f} | RMSE: {val_metrics.get('rmse', 0):.3f} | "
          f"MAE: {val_metrics.get('mae', 0):.3f} | R²: {val_metrics.get('r2', 0):.3f} | "
          f"Spearman: {val_metrics.get('spearman_correlation', 0):.3f}")
    print(f"Classification - AUROC: {val_metrics.get('auroc', 0):.3f} | "
          f"Binary Acc: {val_metrics.get('binary_accuracy', 0):.3f} | "
          f"F1: {val_metrics.get('f1_score', 0):.3f} | "
          f"Boundary MAE: {val_metrics.get('boundary_mae', 0):.3f}")

def save_best_regression_model(model, val_metrics: dict, best_val_auroc: float, 
                             results_dir: str, model_name: str,
                             val_evaluator: RegressionEvaluator) -> tuple:
    """Save model if it achieves best validation AUROC and return best metrics."""
    current_auroc = val_metrics.get('auroc', 0.0)
    best_metrics = None
    
    if current_auroc > best_val_auroc:
        best_val_auroc = current_auroc
        best_metrics = val_metrics.copy()  # Store best metrics
        
        save_path = os.path.join(results_dir, f"best_{model_name}_model.pt")
        torch.save(model.state_dict(), save_path)
        print(f"NEW BEST MODEL! AUROC: {best_val_auroc:.4f} - Saved to {save_path}")
        
        # Save prediction plots for best model
        val_evaluator.plot_predictions_vs_targets(
            save_path=os.path.join(results_dir, "best_model_predictions_vs_targets.png"),
            title=f"Best {model_name.upper()} Model (AUROC: {best_val_auroc:.4f})"
        )
        
        val_evaluator.plot_residuals(
            save_path=os.path.join(results_dir, "best_model_residuals.png"),
            title=f"Best {model_name.upper()} Model Residuals"
        )
    
    return best_val_auroc, best_metrics

def evaluate_best_model(model, val_loader, device, results_dir, model_name):
    """Load and evaluate the best saved model"""
    best_model_path = os.path.join(results_dir, f"best_{model_name}_model.pt")
    
    if not os.path.exists(best_model_path):
        print("Best model file not found!")
        return None
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    # Create fresh evaluator for best model
    best_evaluator = RegressionEvaluator()
    
    print("Evaluating best model...")
    with torch.no_grad():
        for features, rs_target in tqdm(val_loader, desc="Evaluating Best Model"):
            # Process batch data (same as validation loop)
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
            
            # Update metrics
            best_evaluator.update(
                targets=rs_target.cpu().numpy(),
                preds=prediction.cpu().numpy()
            )
    
    return best_evaluator.compute_all_metrics(verbose=False)

def print_model_comparison(best_metrics, final_metrics):
    """Print comparison between best and final model performance"""
    print(f"\n{'='*80}")
    print(f"BEST MODEL vs FINAL MODEL COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Metric':<25} {'Best Model':<15} {'Final Model':<15} {'Difference':<15}")
    print(f"{'-'*70}")
    
    key_metrics = [
        ('AUROC', 'auroc'),
        ('RMSE', 'rmse'), 
        ('MAE', 'mae'),
        ('R² Score', 'r2'),
        ('Spearman ρ', 'spearman_correlation'),
        ('Binary Accuracy', 'binary_accuracy'),
        ('F1-Score', 'f1_score'),
        ('Boundary MAE', 'boundary_mae'),
        ('C-index', 'c_index')
    ]
    
    for display_name, key in key_metrics:
        if key in best_metrics and key in final_metrics:
            best_val = best_metrics[key]
            final_val = final_metrics[key] 
            diff = final_val - best_val
            
            # Format difference with + or - sign
            diff_str = f"{diff:+.4f}" if abs(diff) > 0.0001 else "±0.0000"
            
            print(f"{display_name:<25} {best_val:<15.4f} {final_val:<15.4f} {diff_str:<15}")
    
    print(f"{'-'*70}")
    
    # Summary interpretation
    auroc_diff = final_metrics['auroc'] - best_metrics['auroc']
    
    if auroc_diff < -0.01:
        print("Model performance degraded after best epoch (possible overfitting)")
    elif auroc_diff > 0.01:
        print("Model continued improving after best AUROC epoch")
    else:
        print("Model performance remained stable after best epoch")
    
    print(f"{'='*80}")

def generate_training_analysis_plots(val_evaluator, results_dir, feature_extractor, mil_name):
    """Generate comprehensive analysis plots for final model"""
    print(f"\n{'='*60}")
    print(f"GENERATING ANALYSIS PLOTS")
    print(f"{'='*60}")
    
    # Generate plots with model-specific titles
    model_title = f"{feature_extractor.upper()} + {mil_name}"
    
    # 1. Error Analysis
    error_path = os.path.join(results_dir, "error_analysis.png")
    val_evaluator.plot_error_analysis(save_path=error_path)
    print(f"Error analysis plot saved to: {error_path}")
    
    # 2. Risk Distribution Analysis
    risk_path = os.path.join(results_dir, "risk_distribution_analysis.png")
    val_evaluator.plot_risk_distribution_analysis(save_path=risk_path)
    print(f"Risk distribution analysis saved to: {risk_path}")
    
    # 3. Final residuals plot
    residuals_path = os.path.join(results_dir, "final_residuals_analysis.png")
    val_evaluator.plot_residuals(
        save_path=residuals_path,
        title=f"Final {model_title} Model - Residual Analysis"
    )
    print(f"Residuals analysis saved to: {residuals_path}")
    
    # 4. Final predictions vs targets
    predictions_path = os.path.join(results_dir, "final_predictions_vs_targets.png")
    val_evaluator.plot_predictions_vs_targets(
        save_path=predictions_path,
        title=f"Final {model_title} Model - Predictions vs Targets"
    )
    print(f"Predictions analysis saved to: {predictions_path}")
    
    print(f"All analysis plots generated successfully!")
    return [error_path, risk_path, residuals_path, predictions_path]

def print_experiment_summary(experiment_config):
    """Print comprehensive experiment setup summary"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SETUP SUMMARY")
    print(f"{'='*80}")
    
    print(f"Feature Extractor: {experiment_config['feature_extractor']} "
          f"(input_dim={experiment_config['input_dim']})")
    print(f"MIL Architecture: {experiment_config['mil_model_name']}")
    print(f"  └─ {experiment_config['mil_model_description']}")
    print(f"Experiment Type: {experiment_config['experiment_type']}")
    print(f"Training Configuration:")
    print(f"  └─ Learning Rate: {experiment_config['lr']}")
    print(f"  └─ Epochs: {experiment_config['num_epochs']}")
    print(f"  └─ Hidden Dim: {experiment_config['hidden_dim']}")
    print(f"  └─ Loss Type: {experiment_config['loss_type']}")
    if experiment_config['loss_type'] == 'huber':
        print(f"  └─ Huber Delta: {experiment_config['huber_delta']}")
    print(f"Data Balancing:")
    print(f"  └─ Class Balancing: {experiment_config['use_class_balancing']}")
    print(f"  └─ Boundary Weighting: {experiment_config['use_boundary_weighting']}")
    if experiment_config['use_boundary_weighting']:
        print(f"  └─ Boundary Range: ±{experiment_config['boundary_range']}")
    print(f"Data Sources:")
    print(f"  └─ Train: {experiment_config['train_csv']}")
    print(f"  └─ Val: {experiment_config['val_csv']}")
    
    print(f"{'='*80}")

def main():
    # Setup experiment configuration
    experiment_config = setup_experiment_config(CONFIG)
    
    # Set seed for reproducibility
    set_seed(experiment_config["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = setup_regression_directories(experiment_config["method_name"], 
                                             experiment_config["feature_extractor_name"])
    
    # Save experiment configuration for reproducibility
    save_experiment_config(experiment_config, results_dir)
    
    # Print experiment summary
    print_experiment_summary(experiment_config)
    print(f"Device: {device}")
    print(f"Results directory: {results_dir}")
    
    # Create datasets using feature-specific CSV files
    train_dataset = RegressionMILDataset(experiment_config["train_csv"])
    val_dataset = RegressionMILDataset(experiment_config["val_csv"])
    
    # Create dataloaders
    train_loader, val_loader = create_regression_dataloaders(
        train_dataset, val_dataset, experiment_config["batch_size"],
        use_class_balancing=experiment_config["use_class_balancing"],
        use_boundary_weighting=experiment_config["use_boundary_weighting"],
        boundary_range=experiment_config["boundary_range"]
    )
    
    # Setup model and criterion using flexible MIL architecture
    model, criterion, optimizer = setup_regression_model_and_criterion(
        experiment_config["input_dim"], 
        experiment_config["hidden_dim"], 
        experiment_config["lr"], 
        device,
        experiment_config["loss_type"], 
        experiment_config["huber_delta"],
        mil_model_class=experiment_config["mil_model_class"],
        attention_hidden_dim=experiment_config.get("attention_hidden_dim", 64)
    )
    
    # Initialize evaluators
    train_evaluator = RegressionEvaluator()
    val_evaluator = RegressionEvaluator()
    
    print(f"\nDataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    # Training loop
    all_metrics = []
    best_val_auroc = 0.0
    best_metrics = None  # Track best model metrics
    
    print(f"\n{'='*60}")
    print(f"STARTING REGRESSION TRAINING")
    print(f"{'='*60}")
    
    for epoch in range(1, experiment_config["num_epochs"] + 1):
        # Reset evaluators
        train_evaluator.reset()
        val_evaluator.reset()
        
        # Training
        train_loss = train_regression_epoch(
            model, train_loader, criterion, optimizer, device, train_evaluator
        )
        train_metrics = train_evaluator.compute_all_metrics(verbose=False)
        
        # Validation
        val_loss = validate_regression_epoch(
            model, val_loader, criterion, device, val_evaluator
        )
        val_metrics = val_evaluator.compute_all_metrics(verbose=True)
        
        # Print metrics
        print_regression_epoch_metrics(
            epoch, experiment_config["num_epochs"], train_loss, val_loss, 
            train_metrics, val_metrics
        )
        
        # Save best model (based on AUROC) and track best metrics
        model_name = f"{experiment_config['feature_extractor']}_{experiment_config['mil_pooling']}"
        best_val_auroc, current_best_metrics = save_best_regression_model(
            model, val_metrics, best_val_auroc, results_dir,
            model_name, val_evaluator
        )
        
        # Update best metrics if we found a new best
        if current_best_metrics is not None:
            best_metrics = current_best_metrics
        
        # Store metrics
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }
        all_metrics.append(epoch_metrics)
    
    # Post-training analysis
    print(f"\n{'='*60}")
    print(f"REGRESSION TRAINING COMPLETE")
    print(f"{'='*60}")
    
    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(results_dir, "regression_training_metrics.csv"), index=False)
    
    # Evaluate best model (reload and test)
    print(f"\n{'='*60}")
    print(f"EVALUATING BEST SAVED MODEL")
    print(f"{'='*60}")
    
    model_name = f"{experiment_config['feature_extractor']}_{experiment_config['mil_pooling']}"
    best_model_metrics = evaluate_best_model(model, val_loader, device, results_dir, model_name)
    
    # Final model metrics (current state)
    final_model_metrics = val_evaluator.compute_all_metrics(verbose=False)
    
    # Compare best vs final
    if best_model_metrics is not None and best_metrics is not None:
        print_model_comparison(best_model_metrics, final_model_metrics)
    
    # Generate comprehensive analysis plots
    print(f"\n{'='*60}")
    print(f"GENERATING TRAINING ANALYSIS PLOTS")  
    print(f"{'='*60}")
    
    # 1. Training curves
    RegressionEvaluator.plot_training_curves(metrics_df, results_dir)
    
    # 2. Final model analysis plots
    generate_training_analysis_plots(
        val_evaluator, 
        results_dir, 
        experiment_config["feature_extractor"],
        experiment_config["mil_model_name"]
    )
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED - PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    print(f"Feature Extractor: {experiment_config['feature_extractor']}")
    print(f"MIL Architecture: {experiment_config['mil_model_name']}")
    print(f"Input Dimension: {experiment_config['input_dim']}")
    
    # Best model summary
    if best_model_metrics is not None:
        print(f"\nBEST MODEL PERFORMANCE (saved model):")
        print(f"  AUROC: {best_model_metrics.get('auroc', 0):.4f}")
        print(f"  RMSE: {best_model_metrics.get('rmse', 0):.4f}")
        print(f"  MAE: {best_model_metrics.get('mae', 0):.4f}")
        print(f"  R²: {best_model_metrics.get('r2', 0):.4f}")
        print(f"  Spearman ρ: {best_model_metrics.get('spearman_correlation', 0):.4f}")
        print(f"  Binary Accuracy: {best_model_metrics.get('binary_accuracy', 0):.4f}")
        print(f"  F1-Score: {best_model_metrics.get('f1_score', 0):.4f}")
        print(f"  Boundary MAE: {best_model_metrics.get('boundary_mae', 0):.4f}")
    
    print(f"\nFINAL MODEL PERFORMANCE (last epoch):")
    print(f"  AUROC: {final_model_metrics.get('auroc', 0):.4f}")
    print(f"  RMSE: {final_model_metrics.get('rmse', 0):.4f}")
    print(f"  MAE: {final_model_metrics.get('mae', 0):.4f}")
    print(f"  R²: {final_model_metrics.get('r2', 0):.4f}")
    print(f"  Spearman ρ: {final_model_metrics.get('spearman_correlation', 0):.4f}")
    print(f"  Binary Accuracy: {final_model_metrics.get('binary_accuracy', 0):.4f}")
    print(f"  F1-Score: {final_model_metrics.get('f1_score', 0):.4f}")
    print(f"  Boundary MAE: {final_model_metrics.get('boundary_mae', 0):.4f}")
    
    print(f"\nAll results saved to: {results_dir}")
    
    # Save experiment summary text file
    save_experiment_summary(experiment_config, best_model_metrics, final_model_metrics, results_dir)
    save_quick_comparison_summary(experiment_config, best_model_metrics, results_dir)
    
    print(f"\n{'='*80}")
    print(f"REGRESSION TRAINING COMPLETED")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()