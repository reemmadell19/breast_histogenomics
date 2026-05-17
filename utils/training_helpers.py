import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any

def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_directories(model_name: str, experiment_type: str = "") -> str:
    """Create results directory with proper naming."""
    suffix = f"_{experiment_type}" if experiment_type else ""
    results_dir = f"results/{model_name}{suffix}"
    os.makedirs(results_dir, exist_ok=True)
    print(f" Results will be saved to: {results_dir}")
    return results_dir

def initialize_wandb(project_name: str, run_name: str, config: Dict[str, Any]):
    """Initialize Weights & Biases with given configuration."""
    wandb.init(project=project_name, name=run_name)
    wandb.config.update(config)

def create_weighted_sampler(dataset, class_weights: Dict[int, float]) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler for handling class imbalance."""
    labels = [label for _, label in dataset]
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def create_dataloaders(train_dataset, val_dataset, batch_size: int, 
                      collate_fn, use_weighted_sampling: bool = False,
                      class_weights: Optional[Dict[int, float]] = None) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with optional weighted sampling."""
    
    if use_weighted_sampling and class_weights:
        sampler = create_weighted_sampler(train_dataset, class_weights)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader

def setup_model_and_optimizer(model_class, model_params: Dict, lr: float, device: torch.device):
    """Initialize model, criterion, and optimizer."""
    model = model_class(**model_params).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer

def process_batch_data(features, label, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process and prepare batch data for training/validation."""
    # Handle features - move to device
    if isinstance(features, list):
        features = features[0] if len(features) == 1 else torch.stack(features)
    features = features.to(device)
    
    # Handle labels - ensure proper tensor conversion
    if isinstance(label, list):
        label = label[0] if len(label) == 1 else label
    if not isinstance(label, torch.Tensor):
        label = torch.tensor([label], dtype=torch.long, device=device)
    else:
        label = label.to(device)
        if label.dim() == 0:  # scalar tensor
            label = label.unsqueeze(0)
    
    return features, label

def extract_predictions(outputs: torch.Tensor, label: torch.Tensor) -> Tuple[int, float, int]:
    """Extract predictions and probabilities from model outputs."""
    probs = torch.softmax(outputs, dim=0)
    pred_class = probs.argmax().item()
    prob_positive = probs[1].item()
    label_int = label.item()
    return pred_class, prob_positive, label_int

def train_one_epoch(model, train_loader, criterion, optimizer, device: torch.device, 
                   train_evaluator, desc: str = "Training"):
    """Train model for one epoch and return total loss."""
    model.train()
    train_loss_total = 0.0
    
    for features, label in tqdm(train_loader, desc=desc):
        features, label = process_batch_data(features, label, device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.unsqueeze(0), label)
        loss.backward()
        optimizer.step()
        
        # Extract predictions for metrics
        pred_class, prob_positive, label_int = extract_predictions(outputs, label)
        
        # Update train evaluator
        train_evaluator.update(
            labels=[label_int],
            preds=[pred_class], 
            probs=[prob_positive],
            losses=[loss.item()]
        )
        
        train_loss_total += loss.item()
    
    return train_loss_total

def validate_one_epoch(model, val_loader, criterion, device: torch.device, 
                      val_evaluator, desc: str = "Validating"):
    """Validate model for one epoch and return total loss."""
    model.eval()
    val_loss_total = 0.0
    
    with torch.no_grad():
        for features, label in tqdm(val_loader, desc=desc):
            features, label = process_batch_data(features, label, device)
            
            outputs = model(features)
            loss = criterion(outputs.unsqueeze(0), label)
            
            # Extract predictions for metrics
            pred_class, prob_positive, label_int = extract_predictions(outputs, label)
            
            # Update validation evaluator
            val_evaluator.update(
                labels=[label_int],
                preds=[pred_class],
                probs=[prob_positive], 
                losses=[loss.item()]
            )
            
            val_loss_total += loss.item()
    
    return val_loss_total

def print_epoch_metrics(epoch: int, num_epochs: int, train_loss: float, val_loss: float,
                       train_metrics: Dict, val_metrics: Dict, model_name: str):
    """Print formatted epoch metrics."""
    print(f"\nEpoch {epoch}/{num_epochs}")
    print(f" TRAIN - Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
          f"AUC: {train_metrics['auc_roc']:.4f} | F1: {train_metrics['f1_score']:.4f}")
    print(f" VAL   - Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
          f"AUC: {val_metrics['auc_roc']:.4f} | F1: {val_metrics['f1_score']:.4f}")

def save_best_model(model, val_metrics: Dict, best_val_auc: float, results_dir: str, 
                   model_name: str, experiment_type: str, val_evaluator) -> float:
    """Save model if it achieves best validation AUC."""
    current_auc = val_metrics['auc_roc']
    
    if current_auc > best_val_auc:
        best_val_auc = current_auc
        save_path = os.path.join(results_dir, f"best_{model_name}_{experiment_type}_model.pt")
        torch.save(model.state_dict(), save_path)
        print(f" NEW BEST MODEL! AUC: {best_val_auc:.4f} - Saved to {save_path}")
        
        # Save best model's confusion matrix
        val_evaluator.plot_confusion_matrix(
            save_path=os.path.join(results_dir, "best_model_confusion_matrix.png"),
            title=f"Best {model_name.upper()} {experiment_type.upper()} Model Confusion Matrix (AUC: {best_val_auc:.4f})"
        )
    
    return best_val_auc

def create_metrics_dict(epoch: int, train_loss: float, val_loss: float,
                       train_metrics: Dict, val_metrics: Dict) -> Dict:
    """Create comprehensive metrics dictionary for an epoch."""
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        # Training metrics
        "train_accuracy": train_metrics['accuracy'],
        "train_auc_roc": train_metrics['auc_roc'],
        "train_f1": train_metrics['f1_score'],
        "train_precision": train_metrics['precision'],
        "train_recall": train_metrics['recall'],
        "train_sensitivity": train_metrics['sensitivity'],
        "train_specificity": train_metrics['specificity'],
        "train_auc_pr": train_metrics['auc_pr'],
        "train_mcc": train_metrics['mcc'],
        "train_balanced_accuracy": train_metrics['balanced_accuracy'],
        # Validation metrics
        "val_accuracy": val_metrics['accuracy'],
        "val_auc_roc": val_metrics['auc_roc'],
        "val_f1": val_metrics['f1_score'],
        "val_precision": val_metrics['precision'],
        "val_recall": val_metrics['recall'],
        "val_sensitivity": val_metrics['sensitivity'],
        "val_specificity": val_metrics['specificity'],
        "val_ppv": val_metrics['ppv'],
        "val_npv": val_metrics['npv'],
        "val_auc_pr": val_metrics['auc_pr'],
        "val_mcc": val_metrics['mcc'],
        "val_balanced_accuracy": val_metrics['balanced_accuracy'],
    }

def log_to_wandb(epoch: int, train_loss: float, val_loss: float,
                train_metrics: Dict, val_metrics: Dict, sampling_method: str = ""):
    """Log metrics to Weights & Biases."""
    wandb_log = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        # Core metrics
        "train_accuracy": train_metrics['accuracy'],
        "val_accuracy": val_metrics['accuracy'],
        "train_auc_roc": train_metrics['auc_roc'],
        "val_auc_roc": val_metrics['auc_roc'],
        "train_f1": train_metrics['f1_score'],
        "val_f1": val_metrics['f1_score'],
        # Clinical metrics
        "val_sensitivity": val_metrics['sensitivity'],
        "val_specificity": val_metrics['specificity'],
        "val_ppv": val_metrics['ppv'],
        "val_npv": val_metrics['npv'],
        # Advanced metrics
        "val_auc_pr": val_metrics['auc_pr'],
        "val_mcc": val_metrics['mcc'],
        "val_balanced_accuracy": val_metrics['balanced_accuracy'],
    }
    
    if sampling_method:
        wandb_log["sampling_method"] = sampling_method
    
    wandb.log(wandb_log)

def create_comprehensive_plots(metrics_df: pd.DataFrame, results_dir: str, model_name: str):
    """Create comprehensive training visualization plots."""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"{model_name.upper()} Training Progress - Comprehensive Metrics", fontsize=16)

        # Loss curves
        axes[0,0].plot(metrics_df['epoch'], metrics_df['train_loss'], 'b-o', label='Train')
        axes[0,0].plot(metrics_df['epoch'], metrics_df['val_loss'], 'r-o', label='Validation')
        axes[0,0].set_title('Loss Curves')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True)

        # Accuracy curves
        axes[0,1].plot(metrics_df['epoch'], metrics_df['train_accuracy'], 'b-o', label='Train')
        axes[0,1].plot(metrics_df['epoch'], metrics_df['val_accuracy'], 'r-o', label='Validation')
        axes[0,1].set_title('Accuracy Curves')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].legend()
        axes[0,1].grid(True)

        # AUC curves
        axes[0,2].plot(metrics_df['epoch'], metrics_df['train_auc_roc'], 'b-o', label='Train AUC-ROC')
        axes[0,2].plot(metrics_df['epoch'], metrics_df['val_auc_roc'], 'r-o', label='Val AUC-ROC')
        axes[0,2].plot(metrics_df['epoch'], metrics_df['val_auc_pr'], 'g-o', label='Val AUC-PR')
        axes[0,2].set_title('AUC Curves')
        axes[0,2].set_xlabel('Epoch')
        axes[0,2].set_ylabel('AUC')
        axes[0,2].legend()
        axes[0,2].grid(True)

        # F1 Score
        axes[1,0].plot(metrics_df['epoch'], metrics_df['train_f1'], 'b-o', label='Train')
        axes[1,0].plot(metrics_df['epoch'], metrics_df['val_f1'], 'r-o', label='Validation')
        axes[1,0].set_title('F1 Score')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('F1 Score')
        axes[1,0].legend()
        axes[1,0].grid(True)

        # Clinical Metrics
        axes[1,1].plot(metrics_df['epoch'], metrics_df['val_sensitivity'], 'r-o', label='Sensitivity')
        axes[1,1].plot(metrics_df['epoch'], metrics_df['val_specificity'], 'b-o', label='Specificity')
        axes[1,1].plot(metrics_df['epoch'], metrics_df['val_ppv'], 'g-o', label='PPV')
        axes[1,1].plot(metrics_df['epoch'], metrics_df['val_npv'], 'm-o', label='NPV')
        axes[1,1].set_title('Clinical Metrics')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Score')
        axes[1,1].legend()
        axes[1,1].grid(True)

        # Advanced Metrics
        axes[1,2].plot(metrics_df['epoch'], metrics_df['val_mcc'], 'r-o', label='Matthews Corr')
        axes[1,2].plot(metrics_df['epoch'], metrics_df['val_balanced_accuracy'], 'b-o', label='Balanced Acc')
        axes[1,2].set_title('Advanced Metrics')
        axes[1,2].set_xlabel('Epoch')
        axes[1,2].set_ylabel('Score')
        axes[1,2].legend()
        axes[1,2].grid(True)

        plt.tight_layout()
        comprehensive_plots_path = os.path.join(results_dir, "comprehensive_training_curves.png")
        plt.savefig(comprehensive_plots_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Comprehensive training curves saved to: {comprehensive_plots_path}")
        
    except Exception as e:
        print(f" Error creating comprehensive plots: {e}")
        print(f"Available columns in metrics_df: {list(metrics_df.columns)}")
        
        # Create simple plots as fallback
        create_basic_plots(metrics_df, results_dir, model_name)

def create_basic_plots(metrics_df: pd.DataFrame, results_dir: str, model_name: str):
    """Create basic training plots as fallback."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curves
        axes[0].plot(metrics_df['epoch'], metrics_df['train_loss'], 'b-o', label='Train')
        axes[0].plot(metrics_df['epoch'], metrics_df['val_loss'], 'r-o', label='Validation')
        axes[0].set_title('Loss Curves')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy curves
        axes[1].plot(metrics_df['epoch'], metrics_df['train_accuracy'], 'b-o', label='Train')
        axes[1].plot(metrics_df['epoch'], metrics_df['val_accuracy'], 'r-o', label='Validation')
        axes[1].set_title('Accuracy Curves')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        fallback_plots_path = os.path.join(results_dir, "basic_training_curves.png")
        plt.savefig(fallback_plots_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Basic training curves saved to: {fallback_plots_path}")
        
    except Exception as e2:
        print(f" Error creating fallback plots: {e2}")
        print("Skipping plot generation...")

def generate_final_summary(model_name: str, experiment_type: str, input_dim: int,
                          class_weights: Dict, train_dataset_size: int, val_dataset_size: int,
                          num_epochs: int, lr: float, best_val_auc: float,
                          final_val_metrics: Dict, results_dir: str, counts: Dict = None) -> str:
    """Generate and save comprehensive experiment summary."""
    
    # Add CONCH-specific details if needed
    model_description = f"{model_name.upper()} ({input_dim}-dim features)"
    if model_name.lower() == "conch":
        model_description += " - Vision-Language Foundation Model"
    
    summary_report = f"""
{model_name.upper()} {experiment_type.upper()} EXPERIMENT SUMMARY
{'='*50}
Model: MeanPoolingMIL with WeightedRandomSampler
Feature Extractor: {model_description}
Sampling Strategy: Batch Balancing ({experiment_type.upper()})
Class Weights: {class_weights}
Dataset: Train={train_dataset_size}, Val={val_dataset_size}
Epochs: {num_epochs}
Learning Rate: {lr}

BEST PERFORMANCE:
- Best Validation AUC-ROC: {best_val_auc:.4f}

FINAL METRICS:
- Accuracy: {final_val_metrics['val_accuracy']:.4f}
- F1-Score: {final_val_metrics['val_f1']:.4f}
- Sensitivity: {final_val_metrics['val_sensitivity']:.4f}
- Specificity: {final_val_metrics['val_specificity']:.4f}
- AUC-PR: {final_val_metrics['val_auc_pr']:.4f}
- Matthews Correlation: {final_val_metrics['val_mcc']:.4f}

BATCH BALANCING DETAILS:
- Original class distribution: {counts if counts else 'Not specified'}
- Sampling weights: {class_weights}
- Sampling method: WeightedRandomSampler with replacement

FILES SAVED:
- Model: best_{model_name}_{experiment_type}_model.pt
- Metrics: comprehensive_metrics.csv
- Plots: comprehensive_training_curves.png, best_model_confusion_matrix.png
"""
    
    summary_path = os.path.join(results_dir, "experiment_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary_report)
    
    return summary_path

def print_final_performance(model_name: str, experiment_type: str, best_val_auc: float,
                           final_val_metrics: Dict, results_dir: str):
    """Print final performance summary."""
    print(f"\n FINAL {model_name.upper()} {experiment_type.upper()} PERFORMANCE SUMMARY:")
    print(f"Best Validation AUC-ROC: {best_val_auc:.4f}")
    print(f"Final Validation Metrics:")
    print(f"  Accuracy: {final_val_metrics['val_accuracy']:.4f}")
    print(f"  F1-Score: {final_val_metrics['val_f1']:.4f}")
    print(f"  Sensitivity: {final_val_metrics['val_sensitivity']:.4f}")
    print(f"  Specificity: {final_val_metrics['val_specificity']:.4f}")
    print(f"  AUC-PR: {final_val_metrics['val_auc_pr']:.4f}")
    print(f"  Matthews Correlation: {final_val_metrics['val_mcc']:.4f}")
    print(f"\n All results saved to: {results_dir}/")

def perform_sanity_check(train_loader, model_name: str):
    """Perform sanity check on data loader."""
    print("Sanity check on train_loader item:")
    for feats, lbl in train_loader:
        print("  features:", type(feats), feats.shape if hasattr(feats, 'shape') else len(feats))
        print("  label   :", type(lbl), lbl)
        break

def print_experiment_header(model_name: str, experiment_type: str, input_dim: int, device: torch.device, lr: float, num_epochs: int):
    """Print formatted experiment header."""
    print("="*60)
    print(f"{model_name.upper()} {experiment_type.upper()} BASELINE TRAINING")
    print("="*60)
    print(f"Feature extractor: {model_name.upper()}")
    print(f"Feature dimension: {input_dim}")
    print(f"MIL aggregation: Mean Pooling")
    print(f"Sampling strategy: {experiment_type}")
    print(f"Device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {num_epochs}")
    print("="*60)