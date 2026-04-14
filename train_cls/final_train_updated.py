# final_train_updated.py

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
import torch.nn.functional as F
import json

# Project imports
from datasets.classification_mil_dataset import ClassificationMILDataset, create_classification_weighted_sampler
from utils.mil_utils import mil_collate_fn
from utils.classification_evaluator import ClassificationEvaluator
from utils.training_helpers import set_seed

# Import from your updated model file
from models.classification_model_updated import (
    CLAMClassifier, MeanPoolingMILClassifier, MaxPoolingMILClassifier,
    AttentionMILClassifier, ACMILClassifier
)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=[0.25, 0.75]):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        device = inputs.device
        alpha = torch.tensor(self.alpha, device=device)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha[targets] * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

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

# MIL architecture mappings
MIL_ARCHITECTURES = {
    "mean": "MeanPoolingMILClassifier",
    "attention": "AttentionMILClassifier",
    "clam": "CLAMClassifier",
    "acmil": "ACMILClassifier",
}

# =====================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS AS NEEDED
# =====================================================================

# Training hyperparameters
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 5e-3
NUM_EPOCHS = 50
BATCH_SIZE = 4

# Optimizer selection
OPTIMIZER_TYPE = "adamw"  # Options: "adam", "adamw"

# MIL-specific hyperparameters
HIDDEN_DIM = 48 # clam uses 512 as standard
ATTENTION_HIDDEN_DIM = 128

# CLAM-specific
GATE = True  # Use gated attention in CLAM

# CLAM Instance Learning Parameters
USE_INSTANCE_LEARNING = True  # Enable instance-level learning for CLAM
K_SAMPLE = 8                   # Number of top/bottom patches for instance learning
INSTANCE_LOSS_FN = 'svm'        # 'ce' for cross-entropy or 'svm' for SVM loss
INSTANCE_LOSS_WEIGHT = 0.3     # Weight for instance loss in total loss

# ACMIL-specific parameters
N_BRANCHES = 10                # Number of attention branches (official uses 10)
TOP_K = 10                     # Top-k instances for masking
MASK_RATIO = 0.7              # Ratio of top instances to mask (70%)
LAMBDA_P = 1.0                # Weight for semantic regularization
LAMBDA_D = 1.0               # Weight for diversity loss
ACMIL_GATE = True             # Use gated attention (ABMIL-style)
# General model parameters
DROPOUT = 0.35

# Training strategies
USE_CLASS_BALANCING = True
USE_CLASS_WEIGHTS = False
USE_GRADIENT_CLIP = True
GRADIENT_CLIP_VAL = 1.0
LABEL_SMOOTHING = 0.2
ALPHA = [0.25, 0.75]

# Learning rate scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = "cosine"
STEP_SIZE = 10
GAMMA = 0.1

# Cross-validation
N_FOLDS = 5
RANDOM_SEED = 42

# Early stopping
USE_EARLY_STOPPING = True
PATIENCE = 10

# =====================================================================

def plot_training_curves(fold_history_df, save_dir, fold_num, best_epoch=None):
    """Plot training curves for a single fold with best epoch marked"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Curves - Fold {fold_num}', fontsize=16, fontweight='bold')
    
    epochs = fold_history_df['epoch'].values
    
    if best_epoch is None and 'val_auroc' in fold_history_df.columns:
        best_epoch = fold_history_df.loc[fold_history_df['val_auroc'].idxmax(), 'epoch']
    
    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, fold_history_df['train_loss'], label='Train Loss', marker='s', markersize=4, linewidth=2)
    ax.plot(epochs, fold_history_df['val_loss'], label='Val Loss', marker='o', markersize=4, linewidth=2)
    if best_epoch:
        ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss over Epochs', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. AUROC curves
    ax = axes[0, 1]
    if 'train_auroc' in fold_history_df.columns:
        ax.plot(epochs, fold_history_df['train_auroc'], label='Train AUROC', marker='s', markersize=4, linewidth=2)
    ax.plot(epochs, fold_history_df['val_auroc'], label='Val AUROC', marker='o', markersize=4, linewidth=2)
    if best_epoch:
        ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('AUROC over Epochs', fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. F1-Score curves
    ax = axes[0, 2]
    if 'train_f1_score' in fold_history_df.columns:
        ax.plot(epochs, fold_history_df['train_f1_score'], label='Train F1', marker='s', markersize=4, linewidth=2)
    ax.plot(epochs, fold_history_df['val_f1_score'], label='Val F1', marker='o', markersize=4, linewidth=2)
    if best_epoch:
        ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score over Epochs', fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. Balanced Accuracy curves
    ax = axes[1, 0]
    if 'train_balanced_accuracy' in fold_history_df.columns:
        ax.plot(epochs, fold_history_df['train_balanced_accuracy'], label='Train Balanced Acc', marker='s', markersize=4, linewidth=2)
    ax.plot(epochs, fold_history_df['val_balanced_accuracy'], label='Val Balanced Acc', marker='o', markersize=4, linewidth=2)
    if best_epoch:
        ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_title('Balanced Accuracy over Epochs', fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 5. Sensitivity and Specificity
    ax = axes[1, 1]
    if 'val_sensitivity' in fold_history_df.columns:
        ax.plot(epochs, fold_history_df['val_sensitivity'], label='Sensitivity', marker='s', markersize=4, linewidth=2)
    if 'val_specificity' in fold_history_df.columns:
        ax.plot(epochs, fold_history_df['val_specificity'], label='Specificity', marker='o', markersize=4, linewidth=2)
    if best_epoch:
        ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
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
    if best_epoch:
        ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MCC', fontsize=12)
    ax.set_title('Matthews Correlation Coefficient over Epochs', fontsize=14)
    ax.set_ylim([-1, 1])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'fold_{fold_num}_training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {save_path}")
    return save_path

def plot_combined_cv_curves(all_histories_df, save_dir, model_name, mil_architecture, fold_best_epochs=None):
    """Plot aggregated training curves across all CV folds"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    grouped = all_histories_df.groupby('epoch')
    
    mean_best_epoch = None
    if fold_best_epochs:
        mean_best_epoch = np.mean(list(fold_best_epochs.values()))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Cross-Validation Training Curves\n{model_name.upper()} + {mil_architecture.upper()}', 
                 fontsize=16, fontweight='bold')
    
    epochs = sorted(all_histories_df['epoch'].unique())
    
    def plot_with_std(ax, epochs, mean_values, std_values, label, color):
        ax.plot(epochs, mean_values, label=label, color=color, linewidth=2)
        ax.fill_between(epochs, mean_values - std_values, mean_values + std_values, 
                        alpha=0.2, color=color)
    
    # Plot all metrics
    metrics_config = [
        ((0, 0), 'loss', 'Loss', 'Loss'),
        ((0, 1), 'auroc', 'AUROC', 'AUROC'),
        ((0, 2), 'f1_score', 'F1-Score', 'F1'),
        ((1, 0), 'balanced_accuracy', 'Balanced Accuracy', 'Balanced Accuracy'),
        ((1, 1), None, 'Sensitivity & Specificity', None),
        ((1, 2), 'mcc', 'MCC', 'MCC')
    ]
    
    for (row, col), metric, title, label in metrics_config:
        ax = axes[row, col]
        
        if metric:
            if f'val_{metric}' in all_histories_df.columns:
                val_mean = grouped[f'val_{metric}'].mean().values
                val_std = grouped[f'val_{metric}'].std().values
                plot_with_std(ax, epochs, val_mean, val_std, f'Val {label}', 'orange')
            
            if f'train_{metric}' in all_histories_df.columns:
                train_mean = grouped[f'train_{metric}'].mean().values
                train_std = grouped[f'train_{metric}'].std().values
                plot_with_std(ax, epochs, train_mean, train_std, f'Train {label}', 'blue')
        else:
            # Special case for sensitivity/specificity
            if 'val_sensitivity' in all_histories_df.columns:
                sens_mean = grouped['val_sensitivity'].mean().values
                sens_std = grouped['val_sensitivity'].std().values
                plot_with_std(ax, epochs, sens_mean, sens_std, 'Sensitivity', 'darkgreen')
            
            if 'val_specificity' in all_histories_df.columns:
                spec_mean = grouped['val_specificity'].mean().values
                spec_std = grouped['val_specificity'].std().values
                plot_with_std(ax, epochs, spec_mean, spec_std, 'Specificity', 'darkblue')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(title.split()[0], fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'cv_aggregated_training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Aggregated CV training curves saved to: {save_path}")
    if mean_best_epoch:
        print(f"Average best epoch across folds: {mean_best_epoch:.1f}")
    
    return save_path

def plot_validation_roc_curves(all_histories_df, save_dir, model_name, mil_architecture):
    """Plot validation ROC curves across all CV folds"""
    best_epochs = []
    for fold in range(1, 6):
        fold_data = all_histories_df[all_histories_df['fold'] == fold]
        if len(fold_data) > 0:
            best_epoch = fold_data.loc[fold_data['val_auroc'].idxmax(), 'epoch']
            best_epochs.append(best_epoch)
    
    plt.figure(figsize=(10, 8))
    
    folds = all_histories_df['fold'].unique()
    epochs = sorted(all_histories_df['epoch'].unique())
    
    auroc_by_epoch = {epoch: [] for epoch in epochs}
    
    for fold in folds:
        fold_data = all_histories_df[all_histories_df['fold'] == fold]
        fold_aurocs = []
        fold_epochs = []
        
        for epoch in epochs:
            epoch_data = fold_data[fold_data['epoch'] == epoch]
            if len(epoch_data) > 0:
                auroc = epoch_data['val_auroc'].values[0]
                fold_aurocs.append(auroc)
                fold_epochs.append(epoch)
                auroc_by_epoch[epoch].append(auroc)
        
        plt.plot(fold_epochs, fold_aurocs, alpha=0.3, linewidth=1,
                label=f'Fold {fold} (Best: {max(fold_aurocs):.3f})')
    
    mean_aurocs = []
    std_aurocs = []
    valid_epochs = []
    
    for epoch in epochs:
        if len(auroc_by_epoch[epoch]) > 0:
            mean_aurocs.append(np.mean(auroc_by_epoch[epoch]))
            std_aurocs.append(np.std(auroc_by_epoch[epoch]))
            valid_epochs.append(epoch)
    
    mean_aurocs = np.array(mean_aurocs)
    std_aurocs = np.array(std_aurocs)
    
    plt.plot(valid_epochs, mean_aurocs, 'b-', linewidth=2.5,
            label=f'Mean Val AUROC (Max: {max(mean_aurocs):.3f})', alpha=0.8)
    
    plt.fill_between(valid_epochs,
                     mean_aurocs - std_aurocs,
                     mean_aurocs + std_aurocs,
                     color='blue', alpha=0.2,
                     label='± 1 std. dev.')
    
    if best_epochs:
        avg_best_epoch = np.mean(best_epochs)
        plt.axvline(x=avg_best_epoch, color='red', linestyle='--', 
                   alpha=0.5, label=f'Avg Best Epoch: {avg_best_epoch:.0f}')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation AUROC', fontsize=12)
    plt.title(f'Validation AUROC Across Folds\n{model_name.upper()} + {mil_architecture.upper()}', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.5, 1.0])
    
    final_mean = mean_aurocs[-1] if len(mean_aurocs) > 0 else 0
    final_std = std_aurocs[-1] if len(std_aurocs) > 0 else 0
    best_mean = max(mean_aurocs) if len(mean_aurocs) > 0 else 0
    
    textstr = f'Final: {final_mean:.3f} ± {final_std:.3f}\n'
    textstr += f'Best: {best_mean:.3f}'
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'validation_roc_curves_all_folds.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Validation ROC curves saved to: {os.path.join(save_dir, 'validation_roc_curves_all_folds.png')}")
    print(f"Mean Final Val AUROC: {final_mean:.3f} ± {final_std:.3f}")

def create_model(input_dim, mil_architecture, device, n_classes=2):
    """Create classification model based on MIL architecture type"""
    
    mil_classes = {
        "mean": MeanPoolingMILClassifier,
        "attention": AttentionMILClassifier,
        "clam": CLAMClassifier,
        "acmil": ACMILClassifier,
    }
    
    if mil_architecture not in mil_classes:
        raise ValueError(f"Unknown MIL architecture: {mil_architecture}")
    
    model_class = mil_classes[mil_architecture]
    
    if mil_architecture == "mean":
        model = model_class(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            n_classes=n_classes,
            dropout=DROPOUT
        ).to(device)
    
    elif mil_architecture == "attention":
        model = model_class(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            attention_hidden_dim=ATTENTION_HIDDEN_DIM,
            n_classes=n_classes,
            dropout=DROPOUT
        ).to(device)
    
    elif mil_architecture == "clam":
        model = model_class(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,  # CLAM uses 512 as standard
            attention_hidden_dim=ATTENTION_HIDDEN_DIM,
            n_classes=n_classes,
            dropout=DROPOUT,
            gate=GATE,
            instance_eval=USE_INSTANCE_LEARNING,
            k_sample=K_SAMPLE,
            instance_loss_fn=INSTANCE_LOSS_FN
        ).to(device)
    
    elif mil_architecture == "acmil":
        model = model_class(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            n_classes=n_classes,
            n_branches=N_BRANCHES,
            dropout=DROPOUT,
            top_k=TOP_K,
            mask_ratio=MASK_RATIO,
            lambda_p=LAMBDA_P,
            lambda_d=LAMBDA_D,
            gate=ACMIL_GATE
        ).to(device)
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device, evaluator, mil_architecture):
    """Train for one epoch - handles special losses for CLAM and ACMIL"""
    model.train()
    total_loss = 0.0
    evaluator.reset()
    
    for batch_features, batch_labels in dataloader:
        optimizer.zero_grad()
        
        if not isinstance(batch_features, list):
            batch_features = [batch_features]
            batch_labels = [batch_labels] if not isinstance(batch_labels, torch.Tensor) else batch_labels.unsqueeze(0)
        
        all_logits = []
        total_instance_loss = 0.0
        total_acmil_loss = 0.0
        
        for i, features in enumerate(batch_features):
            features = features.to(device)
            features = F.dropout(features, p=0.3, training=True)
            
            # Get current label for this sample (ACMIL needs it)
            if isinstance(batch_labels, list):
                current_label = torch.tensor([batch_labels[i]], dtype=torch.long).to(device)
            else:
                current_label = batch_labels[i:i+1].to(device)
                if current_label.dim() > 1:
                    current_label = current_label.squeeze()
            
            # Handle different architectures
            if mil_architecture == "clam" and USE_INSTANCE_LEARNING:
                outputs = model(features, return_instance_loss=True)
                if isinstance(outputs, tuple):
                    logits, instance_loss = outputs
                    total_instance_loss += instance_loss
                else:
                    logits = outputs
                    
            elif mil_architecture == "acmil":
                # ACMIL returns (logits, total_loss) during training when label is provided
                outputs = model(features, label=current_label)
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    logits, acmil_loss = outputs
                    total_acmil_loss += acmil_loss
                else:
                    logits = outputs
                    
            else:
                logits = model(features)
            
            if logits.dim() == 2 and logits.shape[0] == 1:
                logits = logits.squeeze(0)
            
            all_logits.append(logits)
        
        batch_logits = torch.stack(all_logits)
        
        if isinstance(batch_labels, list):
            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
        else:
            batch_labels = batch_labels.to(device).long()
        
        if batch_labels.dim() == 0:
            batch_labels = batch_labels.unsqueeze(0)
        
        # Calculate total loss based on architecture
        if mil_architecture == "acmil" and total_acmil_loss > 0:
            # ACMIL already computed its complete loss internally (bag + cluster + consistency)
            loss = total_acmil_loss / len(batch_features)
        elif mil_architecture == "clam" and USE_INSTANCE_LEARNING and total_instance_loss > 0:
            # CLAM: combine bag loss with instance loss
            bag_loss = criterion(batch_logits, batch_labels)
            total_instance_loss = total_instance_loss / len(batch_features)
            loss = bag_loss + INSTANCE_LOSS_WEIGHT * total_instance_loss
        else:
            # Standard loss for other architectures
            loss = criterion(batch_logits, batch_labels)
        
        loss.backward()
        
        if USE_GRADIENT_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
        
        optimizer.step()
        
        probs = torch.softmax(batch_logits, dim=1)
        preds = torch.argmax(batch_logits, dim=1)
        
        total_loss += loss.item()
        evaluator.update(
            labels=batch_labels.cpu().numpy(),
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
        for batch_features, batch_labels in dataloader:
            if not isinstance(batch_features, list):
                batch_features = [batch_features]
                batch_labels = [batch_labels] if not isinstance(batch_labels, torch.Tensor) else batch_labels.unsqueeze(0)
            
            all_logits = []
            
            for features in batch_features:
                features = features.to(device)
                
                # ACMIL in eval mode just returns logits
                logits = model(features)
                
                if logits.dim() == 2 and logits.shape[0] == 1:
                    logits = logits.squeeze(0)
                
                all_logits.append(logits)
            
            batch_logits = torch.stack(all_logits)
            
            if isinstance(batch_labels, list):
                batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
            else:
                batch_labels = batch_labels.to(device).long()
            
            if batch_labels.dim() == 0:
                batch_labels = batch_labels.unsqueeze(0)
            
            loss = criterion(batch_logits, batch_labels)
            
            probs = torch.softmax(batch_logits, dim=1)
            preds = torch.argmax(batch_logits, dim=1)
            
            total_loss += loss.item()
            evaluator.update(
                labels=batch_labels.cpu().numpy(),
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
    
    n_classes = 2
    model = create_model(model_config["input_dim"], mil_architecture, device, n_classes)
    
    if USE_CLASS_WEIGHTS and class_weights is not None:
        scaled_weights = 1 + (class_weights - 1) * 0.3
        weights = torch.tensor(scaled_weights, dtype=torch.float32).to(device)
        criterion = FocalLoss(gamma=2.0, alpha=ALPHA)
    else:
        criterion = FocalLoss(gamma=2.0, alpha=ALPHA)
    
    if OPTIMIZER_TYPE.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        print(f"Using Adam optimizer (LR={LEARNING_RATE}, WD={WEIGHT_DECAY})")
    elif OPTIMIZER_TYPE.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        print(f"Using AdamW optimizer (LR={LEARNING_RATE}, WD={WEIGHT_DECAY})")
    else:
        raise ValueError(f"Unknown optimizer type: {OPTIMIZER_TYPE}")
    
    scheduler = None
    if USE_SCHEDULER:
        if SCHEDULER_TYPE == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        elif SCHEDULER_TYPE == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    
    train_evaluator = ClassificationEvaluator(n_classes=n_classes)
    val_evaluator = ClassificationEvaluator(n_classes=n_classes)
    
    fold_metrics_history = []
    best_val_loss = float('inf')
    best_metrics = {}
    best_epoch = 0
    epochs_without_improvement = 0
    
    pbar = tqdm(range(1, NUM_EPOCHS + 1), desc=f"Fold {fold_num}")
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, 
                                train_evaluator, mil_architecture)
        train_metrics = train_evaluator.compute_all_metrics(verbose=False)
        
        val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device, 
                                               val_evaluator, mil_architecture)
        
        epoch_metrics = {
            'epoch': epoch,
            'fold': fold_num,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        for k, v in train_metrics.items():
            epoch_metrics[f'train_{k}'] = v
        for k, v in val_metrics.items():
            epoch_metrics[f'val_{k}'] = v
        
        fold_metrics_history.append(epoch_metrics)
        
        if scheduler:
            scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics.copy()
            best_metrics['val_loss'] = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_auroc': val_metrics.get('auroc', 0.0),
                'metrics': best_metrics,
                'mil_architecture': mil_architecture,
                'hyperparameters': {
                        'optimizer_type': OPTIMIZER_TYPE,
                        'learning_rate': LEARNING_RATE,
                        'weight_decay': WEIGHT_DECAY,
                        'hidden_dim': HIDDEN_DIM,
                        'attention_hidden_dim': ATTENTION_HIDDEN_DIM,
                        'dropout': DROPOUT,
                        'use_instance_learning': USE_INSTANCE_LEARNING if mil_architecture == "clam" else False,
                        'k_sample': K_SAMPLE if mil_architecture == "clam" else None,
                        'instance_loss_fn': INSTANCE_LOSS_FN if mil_architecture == "clam" else None,
                        'instance_loss_weight': INSTANCE_LOSS_WEIGHT if mil_architecture == "clam" else None,
                        'n_branches': N_BRANCHES if mil_architecture == "acmil" else None,
                        'top_k': TOP_K if mil_architecture == "acmil" else None,
                        'mask_ratio': MASK_RATIO if mil_architecture == "acmil" else None,
                        'lambda_p': LAMBDA_P if mil_architecture == "acmil" else None,
                        'lambda_d': LAMBDA_D if mil_architecture == "acmil" else None,
                        'gate': ACMIL_GATE if mil_architecture == "acmil" else None,
                    }
            }
            checkpoint_path = os.path.join(save_dir, f'best_model_fold_{fold_num}.pt')
            torch.save(checkpoint, checkpoint_path)
        else:
            epochs_without_improvement += 1
        
        if USE_EARLY_STOPPING and epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break
        
        current_auroc = val_metrics.get('auroc', 0.0)
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'train_auroc': f'{train_metrics.get("auroc", 0):.4f}',
            'val_auroc': f'{current_auroc:.4f}',
            'val_f1': f'{val_metrics.get("f1_score", 0):.4f}',
            'val_balanced_accuracy': f'{val_metrics.get("balanced_accuracy", 0):.4f}',
            'best_loss': f'{best_val_loss:.4f}'
        })
    
    print(f"Fold {fold_num}: Best Val Loss={best_val_loss:.4f} at epoch {best_epoch} (AUROC={best_metrics.get('auroc', 0):.4f})")
    
    fold_history_df = pd.DataFrame(fold_metrics_history)
    fold_history_df.to_csv(os.path.join(save_dir, f'fold_{fold_num}_history.csv'), index=False)
    
    plot_training_curves(fold_history_df, save_dir, fold_num, best_epoch=best_epoch)
    
    val_evaluator.plot_confusion_matrix(
        save_path=os.path.join(save_dir, f'fold_{fold_num}_confusion_matrix.png'),
        title=f'Fold {fold_num} - Confusion Matrix (Best Epoch: {best_epoch})'
    )
    val_evaluator.plot_roc_curve(
        save_path=os.path.join(save_dir, f'fold_{fold_num}_roc_curve.png'),
        title=f'Fold {fold_num} - ROC Curve (Best Epoch: {best_epoch})'
    )
    
    best_metrics['fold'] = fold_num
    best_metrics['best_epoch'] = best_epoch
    best_metrics['best_val_loss'] = best_val_loss
    
    return best_metrics, fold_metrics_history

def run_cv_training(model_name, mil_architecture):
    """Run complete CV training for classification"""
    
    if model_name not in FOUNDATION_MODELS:
        raise ValueError(f"Invalid model name. Choose from: {list(FOUNDATION_MODELS.keys())}")
    
    if mil_architecture not in MIL_ARCHITECTURES:
        raise ValueError(f"Invalid MIL architecture. Choose from: {list(MIL_ARCHITECTURES.keys())}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(RANDOM_SEED)
    
    model_config = FOUNDATION_MODELS[model_name]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_classification_cv_updated/{model_name}_{mil_architecture}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
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
        'dropout': DROPOUT,
        'use_class_balancing': USE_CLASS_BALANCING,
        'use_class_weights': USE_CLASS_WEIGHTS,
        'use_gradient_clip': USE_GRADIENT_CLIP,
        'gradient_clip_val': GRADIENT_CLIP_VAL,
        'label_smoothing': LABEL_SMOOTHING,
        'use_scheduler': USE_SCHEDULER,
        'scheduler_type': SCHEDULER_TYPE,
        'n_folds': N_FOLDS,
        'random_seed': RANDOM_SEED,
        'loss_function': "focal_loss",
        'alpha': ALPHA,
    }
    
    # Add CLAM-specific parameters if relevant
    if mil_architecture == "clam":
        config_info.update({
            'gate': GATE,
            'use_instance_learning': USE_INSTANCE_LEARNING,
            'k_sample': K_SAMPLE,
            'instance_loss_fn': INSTANCE_LOSS_FN,
            'instance_loss_weight': INSTANCE_LOSS_WEIGHT,
        })
    
    # Add ACMIL-specific parameters if relevant
    if mil_architecture == "acmil":
        config_info.update({
            'n_branches': N_BRANCHES,
            'top_k': TOP_K,
            'mask_ratio': MASK_RATIO,
            'lambda_p': LAMBDA_P,
            'lambda_d': LAMBDA_D,
            'gate': ACMIL_GATE,
        })
    
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
    
    if mil_architecture == "clam":
        print(f"\nCLAM Settings:")
        print(f"  Gated Attention: {GATE}")
        print(f"  Instance Learning: {USE_INSTANCE_LEARNING}")
        if USE_INSTANCE_LEARNING:
            print(f"  K-Sample: {K_SAMPLE}")
            print(f"  Instance Loss: {INSTANCE_LOSS_FN}")
            print(f"  Instance Loss Weight: {INSTANCE_LOSS_WEIGHT}")
    
    if mil_architecture == "acmil":
        print(f"\nACMIL Settings:")
        print(f"  N Branches: {N_BRANCHES}")
        print(f"  Top-K: {TOP_K}")
        print(f"  Mask Ratio: {MASK_RATIO}")
        print(f"  Lambda P (semantic): {LAMBDA_P}")
        print(f"  Lambda D (diversity): {LAMBDA_D}")
        print(f"  Gated Attention: {ACMIL_GATE}")
    
    print(f"\nLoading dataset: {model_config['combined_csv']}")
    
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
    
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    class_counts = np.bincount(labels)
    class_weights = len(labels) / (len(class_counts) * class_counts)
    
    print(f"Total samples: {len(dataset)}")
    print(f"Class distribution: {class_counts[0]} low-risk, {class_counts[1]} high-risk")
    print(f"Class weights: Low={class_weights[0]:.3f}, High={class_weights[1]:.3f}")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    splits = list(skf.split(range(len(dataset)), labels))
    
    all_results = []
    all_histories = []
    fold_best_epochs = {}
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx}/{N_FOLDS}")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        if USE_CLASS_BALANCING:
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
        
        fold_results, fold_history = train_fold(
            model_config, train_loader, val_loader,
            device, fold_idx, results_dir, mil_architecture,
            class_weights if USE_CLASS_WEIGHTS else None
        )
        all_results.append(fold_results)
        all_histories.extend(fold_history)
        
        if 'best_epoch' in fold_results:
            fold_best_epochs[fold_idx] = fold_results['best_epoch']
    
    results_df = pd.DataFrame(all_results)
    
    metrics = ['auroc', 'auc_pr', 'accuracy', 'balanced_accuracy', 'f1_score',
               'precision', 'recall', 'sensitivity', 'specificity', 'mcc']
    
    summary = {}
    for metric in metrics:
        if metric in results_df.columns:
            summary[f'{metric}_mean'] = results_df[metric].mean()
            summary[f'{metric}_std'] = results_df[metric].std()
    
    summary['model_name'] = model_name
    summary['mil_architecture'] = mil_architecture
    
    all_histories_df = pd.DataFrame(all_histories)
    all_histories_df.to_csv(os.path.join(results_dir, "all_training_history.csv"), index=False)
    
    plot_combined_cv_curves(all_histories_df, results_dir, model_name, mil_architecture, fold_best_epochs=fold_best_epochs)
    plot_validation_roc_curves(all_histories_df, results_dir, model_name, mil_architecture)
    
    if fold_best_epochs:
        print(f"\nBest epochs per fold: {fold_best_epochs}")
        print(f"Average best epoch: {np.mean(list(fold_best_epochs.values())):.1f} ± {np.std(list(fold_best_epochs.values())):.1f}")
    
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