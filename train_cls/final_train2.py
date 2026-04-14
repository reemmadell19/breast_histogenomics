# run_final_cv_classification.py

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
import matplotlib.pyplot as plt
import seaborn as sns

# Project imports
from datasets.classification_mil_dataset import ClassificationMILDataset, create_classification_weighted_sampler
from utils.mil_utils import mil_collate_fn
from utils.classification_evaluator import ClassificationEvaluator
from utils.training_helpers import set_seed

# All foundation model configurations
FOUNDATION_MODELS = {
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

# MIL architecture mappings (excluding max pooling)
MIL_ARCHITECTURES = {
    "mean": "MeanPoolingMILClassifier",
    "attention": "AttentionMILClassifier",
    "clam": "CLAMClassifier",
    "acmil": "ACMILClassifier",
    "acmil_clam": "ACMIL_CLAM_HybridClassifier"
}

# =====================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS AS NEEDED
# =====================================================================

# Training hyperparameters
LEARNING_RATE = 5e-5  # Learning rate for optimizer
WEIGHT_DECAY = 1e-3  # Weight decay for optimizer
NUM_EPOCHS = 50    # Number of training epochs
BATCH_SIZE = 1        # Batch size (usually 1 for WSI)

# Optimizer selection
OPTIMIZER_TYPE = "adamw"  # Options: "adam", "adamw"

# MIL-specific hype rparameters
HIDDEN_DIM = 128    # Hidden dimension for MIL classifier

# Attention-based models (attention, clam, acmil_clam)
ATTENTION_HIDDEN_DIM = 256  # Attention hidden dimension

# CLAM-specific
GATE = True           # Use gated attention in CLAM

# ACMIL-specific
N_BRANCHES = 5        # Number of branches in ACMIL
N_MASKED_PATCH = 10   # Number of patches to mask
MASK_RATIO = 0.6     # Ratio of patches to mask

# General model parameters
DROPOUT = 0.25       # Dropout rate

# Training strategies
USE_CLASS_BALANCING = True   # Use balanced sampling for training
USE_CLASS_WEIGHTS = False     # Use weighted loss (False as requested)
USE_GRADIENT_CLIP = True      # Use gradient clipping
GRADIENT_CLIP_VAL = 1.0       # Gradient clipping value

# Learning rate scheduler
USE_SCHEDULER = False         # Whether to use LR scheduler
SCHEDULER_TYPE = "cosine"     # Options: "cosine", "step", "none"
STEP_SIZE = 10               # For step scheduler
GAMMA = 0.1                  # For step scheduler

# Cross-validation
N_FOLDS = 5          # Number of CV folds
RANDOM_SEED = 42     # Random seed for reproducibility

# Early stopping (optional)
USE_EARLY_STOPPING = True    # Whether to use early stopping
PATIENCE = 10                # Patience for early stopping

# =====================================================================

def plot_training_curves(fold_history_df, save_dir, fold_num):
    """
    Plot training curves for a single fold
    
    Args:
        fold_history_df: DataFrame with training history
        save_dir: Directory to save plots
        fold_num: Fold number
    """
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Curves - Fold {fold_num}', fontsize=16, fontweight='bold')
    
    epochs = fold_history_df['epoch'].values
    
    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, fold_history_df['train_loss'], label='Train Loss', marker='o', markersize=4, linewidth=2)
    ax.plot(epochs, fold_history_df['val_loss'], label='Val Loss', marker='s', markersize=4, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss over Epochs', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. AUROC curves
    ax = axes[0, 1]
    if 'train_auroc' in fold_history_df.columns:
        ax.plot(epochs, fold_history_df['train_auroc'], label='Train AUROC', marker='o', markersize=4, linewidth=2)
    ax.plot(epochs, fold_history_df['val_auroc'], label='Val AUROC', marker='s', markersize=4, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('AUROC over Epochs', fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. F1-Score curves
    ax = axes[0, 2]
    if 'train_f1_score' in fold_history_df.columns:
        ax.plot(epochs, fold_history_df['train_f1_score'], label='Train F1', marker='o', markersize=4, linewidth=2)
    ax.plot(epochs, fold_history_df['val_f1_score'], label='Val F1', marker='s', markersize=4, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score over Epochs', fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. Accuracy curves
    ax = axes[1, 0]
    if 'train_accuracy' in fold_history_df.columns:
        ax.plot(epochs, fold_history_df['train_accuracy'], label='Train Acc', marker='o', markersize=4, linewidth=2)
    ax.plot(epochs, fold_history_df['val_accuracy'], label='Val Acc', marker='s', markersize=4, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy over Epochs', fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 5. Sensitivity and Specificity
    ax = axes[1, 1]
    if 'val_sensitivity' in fold_history_df.columns:
        ax.plot(epochs, fold_history_df['val_sensitivity'], label='Sensitivity', marker='o', markersize=4, linewidth=2)
    if 'val_specificity' in fold_history_df.columns:
        ax.plot(epochs, fold_history_df['val_specificity'], label='Specificity', marker='s', markersize=4, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Sensitivity & Specificity over Epochs', fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 6. MCC curves
    ax = axes[1, 2]
    if 'val_mcc' in fold_history_df.columns:
        ax.plot(epochs, fold_history_df['val_mcc'], label='Val MCC', marker='s', markersize=4, linewidth=2)
        if 'train_mcc' in fold_history_df.columns:
            ax.plot(epochs, fold_history_df['train_mcc'], label='Train MCC', marker='o', markersize=4, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MCC', fontsize=12)
    ax.set_title('Matthews Correlation Coefficient over Epochs', fontsize=14)
    ax.set_ylim([-1, 1])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, f'fold_{fold_num}_training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {save_path}")
    
    return save_path

def plot_combined_cv_curves(all_histories_df, save_dir, model_name, mil_architecture):
    """
    Plot aggregated training curves across all CV folds
    
    Args:
        all_histories_df: DataFrame with all training histories
        save_dir: Directory to save plots
        model_name: Name of the foundation model
        mil_architecture: Name of the MIL architecture
    """
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Group by epoch and calculate mean and std
    grouped = all_histories_df.groupby('epoch')
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Cross-Validation Training Curves\n{model_name.upper()} + {mil_architecture.upper()}', 
                 fontsize=16, fontweight='bold')
    
    epochs = sorted(all_histories_df['epoch'].unique())
    
    # Helper function to plot with error bands
    def plot_with_std(ax, epochs, mean_values, std_values, label, color):
        ax.plot(epochs, mean_values, label=label, color=color, linewidth=2)
        ax.fill_between(epochs, mean_values - std_values, mean_values + std_values, 
                        alpha=0.2, color=color)
    
    # 1. Loss curves
    ax = axes[0, 0]
    train_loss_mean = grouped['train_loss'].mean().values
    train_loss_std = grouped['train_loss'].std().values
    val_loss_mean = grouped['val_loss'].mean().values
    val_loss_std = grouped['val_loss'].std().values
    
    plot_with_std(ax, epochs, train_loss_mean, train_loss_std, 'Train Loss', 'blue')
    plot_with_std(ax, epochs, val_loss_mean, val_loss_std, 'Val Loss', 'orange')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss over Epochs (Mean ± Std)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. AUROC curves
    ax = axes[0, 1]
    if 'val_auroc' in all_histories_df.columns:
        val_auroc_mean = grouped['val_auroc'].mean().values
        val_auroc_std = grouped['val_auroc'].std().values
        plot_with_std(ax, epochs, val_auroc_mean, val_auroc_std, 'Val AUROC', 'green')
        
        if 'train_auroc' in all_histories_df.columns:
            train_auroc_mean = grouped['train_auroc'].mean().values
            train_auroc_std = grouped['train_auroc'].std().values
            plot_with_std(ax, epochs, train_auroc_mean, train_auroc_std, 'Train AUROC', 'blue')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('AUROC over Epochs (Mean ± Std)', fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. F1-Score curves
    ax = axes[0, 2]
    if 'val_f1_score' in all_histories_df.columns:
        val_f1_mean = grouped['val_f1_score'].mean().values
        val_f1_std = grouped['val_f1_score'].std().values
        plot_with_std(ax, epochs, val_f1_mean, val_f1_std, 'Val F1', 'purple')
        
        if 'train_f1_score' in all_histories_df.columns:
            train_f1_mean = grouped['train_f1_score'].mean().values
            train_f1_std = grouped['train_f1_score'].std().values
            plot_with_std(ax, epochs, train_f1_mean, train_f1_std, 'Train F1', 'blue')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score over Epochs (Mean ± Std)', fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. Accuracy curves
    ax = axes[1, 0]
    if 'val_accuracy' in all_histories_df.columns:
        val_acc_mean = grouped['val_accuracy'].mean().values
        val_acc_std = grouped['val_accuracy'].std().values
        plot_with_std(ax, epochs, val_acc_mean, val_acc_std, 'Val Accuracy', 'red')
        
        if 'train_accuracy' in all_histories_df.columns:
            train_acc_mean = grouped['train_accuracy'].mean().values
            train_acc_std = grouped['train_accuracy'].std().values
            plot_with_std(ax, epochs, train_acc_mean, train_acc_std, 'Train Accuracy', 'blue')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy over Epochs (Mean ± Std)', fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 5. Sensitivity and Specificity
    ax = axes[1, 1]
    if 'val_sensitivity' in all_histories_df.columns:
        sens_mean = grouped['val_sensitivity'].mean().values
        sens_std = grouped['val_sensitivity'].std().values
        plot_with_std(ax, epochs, sens_mean, sens_std, 'Sensitivity', 'darkgreen')
    
    if 'val_specificity' in all_histories_df.columns:
        spec_mean = grouped['val_specificity'].mean().values
        spec_std = grouped['val_specificity'].std().values
        plot_with_std(ax, epochs, spec_mean, spec_std, 'Specificity', 'darkblue')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Sensitivity & Specificity (Mean ± Std)', fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 6. MCC curves
    ax = axes[1, 2]
    if 'val_mcc' in all_histories_df.columns:
        mcc_mean = grouped['val_mcc'].mean().values
        mcc_std = grouped['val_mcc'].std().values
        plot_with_std(ax, epochs, mcc_mean, mcc_std, 'Val MCC', 'brown')
        
        if 'train_mcc' in all_histories_df.columns:
            train_mcc_mean = grouped['train_mcc'].mean().values
            train_mcc_std = grouped['train_mcc'].std().values
            plot_with_std(ax, epochs, train_mcc_mean, train_mcc_std, 'Train MCC', 'blue')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MCC', fontsize=12)
    ax.set_title('Matthews Correlation Coefficient (Mean ± Std)', fontsize=14)
    ax.set_ylim([-1, 1])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'cv_aggregated_training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Aggregated CV training curves saved to: {save_path}")
    
    return save_path

def create_model(input_dim, mil_architecture, device, n_classes=2):
    """Create classification model based on MIL architecture type"""
    
    from models.classification_model import (
        MeanPoolingMILClassifier, AttentionMILClassifier, 
        CLAMClassifier, ACMILClassifier, ACMIL_CLAM_HybridClassifier
    )
    
    # Map architecture string to class
    mil_classes = {
        "mean": MeanPoolingMILClassifier,
        "attention": AttentionMILClassifier,
        "clam": CLAMClassifier,
        "acmil": ACMILClassifier,
        "acmil_clam": ACMIL_CLAM_HybridClassifier
    }
    
    if mil_architecture not in mil_classes:
        raise ValueError(f"Unknown MIL architecture: {mil_architecture}")
    
    model_class = mil_classes[mil_architecture]
    
    # Create model with appropriate parameters
    if mil_architecture == "mean":
        model = model_class(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            n_classes=n_classes
        ).to(device)
    
    elif mil_architecture == "attention":
        model = model_class(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            attention_hidden_dim=ATTENTION_HIDDEN_DIM,
            n_classes=n_classes
        ).to(device)
    
    elif mil_architecture == "clam":
        model = model_class(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            attention_hidden_dim=ATTENTION_HIDDEN_DIM,
            n_classes=n_classes,
            dropout=DROPOUT,
            gate=GATE
        ).to(device)
    
    elif mil_architecture == "acmil":
        model = model_class(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            n_branches=N_BRANCHES,
            n_classes=n_classes,
            n_masked_patch=N_MASKED_PATCH,
            mask_ratio=MASK_RATIO,
            dropout=DROPOUT
        ).to(device)
    
    elif mil_architecture == "acmil_clam":
        model = model_class(
            input_dim=input_dim,
            n_branches=N_BRANCHES,
            hidden_dim=HIDDEN_DIM,
            attention_hidden_dim=ATTENTION_HIDDEN_DIM,
            n_classes=n_classes,
            mask_ratio=MASK_RATIO,
            n_masked_patch=N_MASKED_PATCH,
            dropout=DROPOUT,
            gate=GATE
        ).to(device)
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device, evaluator, 
                mil_architecture):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    evaluator.reset()
    
    for features, label in dataloader:
        # Process data
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
        
        # Forward pass
        optimizer.zero_grad()
        
        # Handle ACMIL models differently
        if mil_architecture in ["acmil", "acmil_clam"]:
            logits = model(features, return_branch_outputs=False)
        else:
            logits = model(features)
            
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        
        loss = criterion(logits, label)
        loss.backward()
        
        # Gradient clipping
        if USE_GRADIENT_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
        
        optimizer.step()
        
        # Get predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        total_loss += loss.item()
        evaluator.update(
            labels=label.cpu().numpy(),
            preds=preds.detach().cpu().numpy(),
            probs=probs.detach().cpu().numpy(),
            losses=[loss.item()]
        )
    
    return total_loss / len(dataloader)

def evaluate_epoch(model, dataloader, criterion, device, evaluator, mil_architecture):
    """Evaluate for one epoch"""
    model.eval()
    total_loss = 0.0
    evaluator.reset()
    
    with torch.no_grad():
        for features, label in dataloader:
            # Process data
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
            
            # Forward pass
            if mil_architecture in ["acmil", "acmil_clam"]:
                logits = model(features, return_branch_outputs=False)
            else:
                logits = model(features)
                
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            
            loss = criterion(logits, label)
            
            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Update metrics
            total_loss += loss.item()
            evaluator.update(
                labels=label.cpu().numpy(),
                preds=preds.cpu().numpy(),
                probs=probs.cpu().numpy(),
                losses=[loss.item()]
            )
    
    avg_loss = total_loss / len(dataloader)
    metrics = evaluator.compute_all_metrics(verbose=False)
    
    return avg_loss, metrics

def train_fold(model_config, train_loader, val_loader, device, 
               fold_num, save_dir, mil_architecture, class_weights=None):
    """Train a single fold and save best model"""
    set_seed(RANDOM_SEED + fold_num)
    
    # Create model
    n_classes = 2  # Binary classification
    model = create_model(model_config["input_dim"], mil_architecture, device, n_classes)
    
    # Setup training
    if USE_CLASS_WEIGHTS and class_weights is not None:
        weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Select optimizer based on configuration
    if OPTIMIZER_TYPE.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        print(f"Using Adam optimizer (LR={LEARNING_RATE}, WD={WEIGHT_DECAY})")
    elif OPTIMIZER_TYPE.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        print(f"Using AdamW optimizer (LR={LEARNING_RATE}, WD={WEIGHT_DECAY})")
    else:
        raise ValueError(f"Unknown optimizer type: {OPTIMIZER_TYPE}. Choose 'adam' or 'adamw'")
    
    # Setup scheduler if specified
    scheduler = None
    if USE_SCHEDULER:
        if SCHEDULER_TYPE == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        elif SCHEDULER_TYPE == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    
    # Initialize evaluators
    train_evaluator = ClassificationEvaluator(n_classes=n_classes)
    val_evaluator = ClassificationEvaluator(n_classes=n_classes)
    
    # Track metrics for plotting
    fold_metrics_history = []
    
    # Training loop
    best_auroc = 0.0
    best_metrics = {}
    best_epoch = 0
    epochs_without_improvement = 0
    
    pbar = tqdm(range(1, NUM_EPOCHS + 1), desc=f"Fold {fold_num}")
    for epoch in pbar:
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, 
                                train_evaluator, mil_architecture)
        train_metrics = train_evaluator.compute_all_metrics(verbose=False)
        
        # Evaluate on validation
        val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device, 
                                               val_evaluator, mil_architecture)
        
        # Store epoch metrics
        epoch_metrics = {
            'epoch': epoch,
            'fold': fold_num,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        # Add train metrics with prefix
        for k, v in train_metrics.items():
            epoch_metrics[f'train_{k}'] = v
        # Add val metrics with prefix  
        for k, v in val_metrics.items():
            epoch_metrics[f'val_{k}'] = v
        
        fold_metrics_history.append(epoch_metrics)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Track best model based on AUROC
        current_auroc = val_metrics.get('auroc', 0.0)
        if current_auroc > best_auroc:
            best_auroc = current_auroc
            best_metrics = val_metrics.copy()
            best_epoch = epoch
            epochs_without_improvement = 0
            
            # Save best model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auroc': best_auroc,
                'metrics': best_metrics,
                'mil_architecture': mil_architecture,
                'hyperparameters': {
                    'optimizer_type': OPTIMIZER_TYPE,
                    'learning_rate': LEARNING_RATE,
                    'weight_decay': WEIGHT_DECAY,
                    'hidden_dim': HIDDEN_DIM,
                    'attention_hidden_dim': ATTENTION_HIDDEN_DIM,
                    'n_branches': N_BRANCHES,
                    'dropout': DROPOUT
                }
            }
            checkpoint_path = os.path.join(save_dir, f'best_model_fold_{fold_num}.pt')
            torch.save(checkpoint, checkpoint_path)
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if USE_EARLY_STOPPING and epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break
        
        # Update progress bar
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_auroc': f'{current_auroc:.4f}',
            'val_f1': f'{val_metrics.get("f1_score", 0):.4f}',
            'val_balanced_accuracy': f'{val_metrics.get("balanced_accuracy", 0):.4f}',
            'best': f'{best_auroc:.4f}'
        })
    
    print(f"Fold {fold_num}: Best AUROC={best_auroc:.4f} at epoch {best_epoch}")
    
    # Save fold training history
    fold_history_df = pd.DataFrame(fold_metrics_history)
    fold_history_df.to_csv(os.path.join(save_dir, f'fold_{fold_num}_history.csv'), index=False)
    
    # Plot training curves for this fold
    plot_training_curves(fold_history_df, save_dir, fold_num)
    
    # Save confusion matrix and ROC curve for best model
    val_evaluator.plot_confusion_matrix(
        save_path=os.path.join(save_dir, f'fold_{fold_num}_confusion_matrix.png'),
        title=f'Fold {fold_num} - Confusion Matrix'
    )
    val_evaluator.plot_roc_curve(
        save_path=os.path.join(save_dir, f'fold_{fold_num}_roc_curve.png'),
        title=f'Fold {fold_num} - ROC Curve'
    )
    
    best_metrics['fold'] = fold_num
    best_metrics['best_epoch'] = best_epoch
    
    return best_metrics, fold_metrics_history

def run_cv_training(model_name, mil_architecture):
    """Run complete CV training for classification"""
    
    # Validate inputs
    if model_name not in FOUNDATION_MODELS:
        raise ValueError(f"Invalid model name. Choose from: {list(FOUNDATION_MODELS.keys())}")
    
    if mil_architecture not in MIL_ARCHITECTURES:
        raise ValueError(f"Invalid MIL architecture. Choose from: {list(MIL_ARCHITECTURES.keys())}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(RANDOM_SEED)
    
    model_config = FOUNDATION_MODELS[model_name]
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_classification_cv/{model_name}_{mil_architecture}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save configuration for reference
    config_info = {
        'model_name': model_name,
        'mil_architecture': mil_architecture,
        'optimizer_type': OPTIMIZER_TYPE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'hidden_dim': HIDDEN_DIM,
        'attention_hidden_dim': ATTENTION_HIDDEN_DIM,
        'n_branches': N_BRANCHES,
        'n_masked_patch': N_MASKED_PATCH,
        'mask_ratio': MASK_RATIO,
        'dropout': DROPOUT,
        'use_class_balancing': USE_CLASS_BALANCING,
        'use_class_weights': USE_CLASS_WEIGHTS,
        'use_gradient_clip': USE_GRADIENT_CLIP,
        'gradient_clip_val': GRADIENT_CLIP_VAL,
        'use_scheduler': USE_SCHEDULER,
        'scheduler_type': SCHEDULER_TYPE,
        'n_folds': N_FOLDS,
        'random_seed': RANDOM_SEED
    }
    
    # Save config
    import json
    with open(os.path.join(results_dir, "config_used.json"), 'w') as f:
        json.dump(config_info, f, indent=2)
    
    print(f"{'='*80}")
    print(f"CLASSIFICATION CV TRAINING")
    print(f"{'='*80}")
    print(f"Foundation Model: {model_name.upper()}")
    print(f"MIL Architecture: {mil_architecture.upper()}")
    print(f"Device: {device}")
    print(f"Folds: {N_FOLDS}")
    print(f"Results directory: {results_dir}")
    print(f"\nHyperparameters:")
    print(f"  Optimizer: {OPTIMIZER_TYPE.upper()}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Hidden Dim: {HIDDEN_DIM}")
    print(f"  Dropout: {DROPOUT}")
    
    # Load dataset
    print(f"\nLoading dataset: {model_config['combined_csv']}")
    
    # Check if RSHigh column exists, otherwise use RS column
    df_check = pd.read_csv(model_config["combined_csv"])
    if 'RSHigh' in df_check.columns:
        label_column = 'RSHigh'
    elif 'RS' in df_check.columns:
        label_column = 'RS'
    else:
        raise ValueError("Neither RSHigh nor RS column found in dataset")
    
    dataset = ClassificationMILDataset(
        model_config["combined_csv"],
        label_column=label_column,
        threshold=25.0
    )
    
    # Get labels for stratification
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(labels)
    class_weights = len(labels) / (len(class_counts) * class_counts)
    
    print(f"Total samples: {len(dataset)}")
    print(f"Class distribution: {class_counts[0]} low-risk, {class_counts[1]} high-risk")
    print(f"Class weights: Low={class_weights[0]:.3f}, High={class_weights[1]:.3f}")
    
    # Create CV splits
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    splits = list(skf.split(range(len(dataset)), labels))
    
    # Run CV training
    all_results = []
    all_histories = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx}/{N_FOLDS}")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        # Create dataloaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # Apply sampling strategy if specified
        if USE_CLASS_BALANCING:
            # Create temporary dataset wrapper for sampler
            class TempDataset:
                def __init__(self, indices, full_dataset):
                    self.indices = indices
                    self.full_dataset = full_dataset
                def __len__(self):
                    return len(self.indices)
                def __getitem__(self, idx):
                    return self.full_dataset[self.indices[idx]]
            
            temp_dataset = TempDataset(train_idx, dataset)
            sampler = create_classification_weighted_sampler(
                temp_dataset, 
                balance_classes=True
            )
            train_loader = DataLoader(
                train_subset, 
                batch_size=BATCH_SIZE, 
                sampler=sampler,
                collate_fn=mil_collate_fn, 
                num_workers=0
            )
        else:
            train_loader = DataLoader(
                train_subset, 
                batch_size=BATCH_SIZE, 
                shuffle=True,
                collate_fn=mil_collate_fn, 
                num_workers=0
            )
        
        val_loader = DataLoader(
            val_subset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            collate_fn=mil_collate_fn, 
            num_workers=0
        )
        
        # Train fold
        fold_results, fold_history = train_fold(
            model_config, train_loader, val_loader,
            device, fold_idx, results_dir, mil_architecture,
            class_weights if USE_CLASS_WEIGHTS else None
        )
        all_results.append(fold_results)
        all_histories.extend(fold_history)
    
    # Calculate statistics
    results_df = pd.DataFrame(all_results)
    
    # Key metrics to summarize
    metrics = ['auroc', 'auc_pr', 'accuracy', 'balanced_accuracy', 'f1_score',
               'precision', 'recall', 'sensitivity', 'specificity', 'mcc']
    
    summary = {}
    for metric in metrics:
        if metric in results_df.columns:
            summary[f'{metric}_mean'] = results_df[metric].mean()
            summary[f'{metric}_std'] = results_df[metric].std()
    
    # Add model and architecture info
    summary['model_name'] = model_name
    summary['mil_architecture'] = mil_architecture
    
    # Save all training histories
    all_histories_df = pd.DataFrame(all_histories)
    all_histories_df.to_csv(os.path.join(results_dir, "all_training_history.csv"), index=False)
    
    # Plot aggregated CV curves
    plot_combined_cv_curves(all_histories_df, results_dir, model_name, mil_architecture)
    
    # Print final results
    print(f"\n{'='*80}")
    print(f"FINAL CLASSIFICATION CV RESULTS")
    print(f"Model: {model_name.upper()} + {mil_architecture.upper()}")
    print(f"{'='*80}")
    print(f"AUROC:             {summary.get('auroc_mean', 0):.4f} ± {summary.get('auroc_std', 0):.4f}")
    print(f"AUC-PR:            {summary.get('auc_pr_mean', 0):.4f} ± {summary.get('auc_pr_std', 0):.4f}")
    print(f"Accuracy:          {summary.get('accuracy_mean', 0):.4f} ± {summary.get('accuracy_std', 0):.4f}")
    print(f"Balanced Accuracy: {summary.get('balanced_accuracy_mean', 0):.4f} ± {summary.get('balanced_accuracy_std', 0):.4f}")
    print(f"F1-Score:          {summary.get('f1_score_mean', 0):.4f} ± {summary.get('f1_score_std', 0):.4f}")
    print(f"Sensitivity:       {summary.get('sensitivity_mean', 0):.4f} ± {summary.get('sensitivity_std', 0):.4f}")
    print(f"Specificity:       {summary.get('specificity_mean', 0):.4f} ± {summary.get('specificity_std', 0):.4f}")
    print(f"MCC:               {summary.get('mcc_mean', 0):.4f} ± {summary.get('mcc_std', 0):.4f}")
    
    # Save all results
    results_df.to_csv(os.path.join(results_dir, "fold_results.csv"), index=False)
    
    with open(os.path.join(results_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAll results saved to: {results_dir}")
    print(f"Training curves saved for each fold and aggregated across CV")
    
    return summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run classification CV training')
    parser.add_argument('--model', type=str, required=True,
                      choices=list(FOUNDATION_MODELS.keys()),
                      help=f'Model name. Options: {list(FOUNDATION_MODELS.keys())}')
    parser.add_argument('--mil', type=str, required=True,
                      choices=list(MIL_ARCHITECTURES.keys()),
                      help=f'MIL architecture. Options: {list(MIL_ARCHITECTURES.keys())}')
    
    args = parser.parse_args()
    
    print(f"Starting training: {args.model} + {args.mil}")
    print(f"Configuration can be modified at the top of this script")
    run_cv_training(args.model, args.mil)