# external validation inference save script for ECU dataset
# test_cls/ecu_inference_save.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Project imports
from datasets.classification_mil_dataset import ClassificationMILDataset
from utils.mil_utils import mil_collate_fn
from utils.classification_evaluator import ClassificationEvaluator

# Updated import to match training script
from models.classification_model_updated import (
    MeanPoolingMILClassifier, MaxPoolingMILClassifier,
    AttentionMILClassifier, CLAMClassifier, ACMILClassifier
)

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_config_from_folder(model_folder):
    """Load configuration from training folder"""
    config_path = os.path.join(model_folder, "config_used.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found in {model_folder}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def get_input_dim_from_model_name(model_name):
    """Get input dimensions for foundation models"""
    dim_map = {
        "resnet18": 512,
        "resnet50": 2048,
        "conch": 512,
        "uni2-h": 1536,
        "virchow2": 1280,
        "h-optimus": 1536
    }
    return dim_map.get(model_name, 512)

def create_model_from_config(config, device):
    """Recreate model architecture from config"""
    
    mil_architecture = config['mil_architecture']
    input_dim = get_input_dim_from_model_name(config['model_name'])
    n_classes = 2
    
    # Get architecture-specific params
    arch_params = config.get('architecture_specific_params', {})
    if not arch_params:
        arch_params = config
    
    if mil_architecture == "mean":
        model = MeanPoolingMILClassifier(
            input_dim=input_dim,
            hidden_dim=arch_params.get('hidden_dim', 256),
            n_classes=n_classes,
            dropout=arch_params.get('dropout', 0.25)
        )
    
    elif mil_architecture == "attention":
        model = AttentionMILClassifier(
            input_dim=input_dim,
            hidden_dim=arch_params.get('hidden_dim', 256),
            attention_hidden_dim=arch_params.get('attention_hidden_dim', 128),
            n_classes=n_classes,
            dropout=arch_params.get('dropout', 0.25)
        )
    
    elif mil_architecture == "clam":
        model = CLAMClassifier(
            input_dim=input_dim,
            hidden_dim=arch_params.get('hidden_dim', 512),
            attention_hidden_dim=arch_params.get('attention_hidden_dim', 384),
            n_classes=n_classes,
            dropout=arch_params.get('dropout', 0.25),
            gate=arch_params.get('gate', True),
            instance_eval=arch_params.get('use_instance_learning', True),
            k_sample=arch_params.get('k_sample', 8),
            instance_loss_fn=arch_params.get('instance_loss_fn', 'svm')
        )
    
    elif mil_architecture == "acmil":
        model = ACMILClassifier(
            input_dim=input_dim,
            hidden_dim=arch_params.get('hidden_dim', 256),
            n_classes=n_classes,
            n_branches=arch_params.get('n_branches', 10),
            dropout=arch_params.get('dropout', 0.5),
            top_k=arch_params.get('top_k', 10),
            mask_ratio=arch_params.get('mask_ratio', 0.7),
            lambda_p=arch_params.get('lambda_p', 1.0),
            lambda_d=arch_params.get('lambda_d', 1.0),
            gate=arch_params.get('gate', True)
        )
    
    else:
        raise ValueError(f"Unknown MIL architecture: {mil_architecture}")
    
    return model.to(device)

def evaluate_single_model(model, dataloader, device, mil_architecture):
    """Evaluate a single fold model"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating"):
            if not isinstance(features, list):
                features = [features]
                labels = [labels] if not isinstance(labels, torch.Tensor) else labels.unsqueeze(0)
            
            batch_probs = []
            
            for feat in features:
                feat = feat.to(device)
                logits = model(feat)
                
                if logits.dim() == 2 and logits.shape[0] == 1:
                    logits = logits.squeeze(0)
                
                probs = torch.softmax(logits, dim=0)
                batch_probs.append(probs.cpu().numpy())
            
            if isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.long)
            
            all_probs.extend(batch_probs)
            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
    
    return np.array(all_probs), np.array(all_labels)

def get_sensitivity_at_specificity(fpr, tpr, target_specificity):
    """Get sensitivity at a given specificity level"""
    specificities = 1 - fpr
    idx = np.argmin(np.abs(specificities - target_specificity))
    return float(tpr[idx])

def get_specificity_at_sensitivity(fpr, tpr, target_sensitivity):
    """Get specificity at a given sensitivity level"""
    idx = np.argmin(np.abs(tpr - target_sensitivity))
    return float(1 - fpr[idx])

def save_model_mean_roc_data_ecu(all_fold_probs, labels, config, output_dir):
    """
    Save mean ROC curve data for ECU external validation
    """
    
    # Calculate mean probabilities across all folds
    mean_probs = np.mean([probs[:, 1] for probs in all_fold_probs], axis=0)
    
    # Calculate mean ROC curve
    fpr_mean, tpr_mean, thresholds_mean = roc_curve(labels, mean_probs)
    auc_mean = auc(fpr_mean, tpr_mean)
    
    # Calculate std of probabilities across folds
    std_probs = np.std([probs[:, 1] for probs in all_fold_probs], axis=0)
    
    # Calculate ROC for each fold to get confidence intervals
    fold_aucs = []
    fold_tprs = []
    mean_fpr_grid = np.linspace(0, 1, 100)
    
    for probs in all_fold_probs:
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        fold_aucs.append(auc(fpr, tpr))
        
        # Interpolate TPR at standard FPR points
        interp_tpr = np.interp(mean_fpr_grid, fpr, tpr)
        interp_tpr[0] = 0.0
        fold_tprs.append(interp_tpr)
    
    # Calculate mean and std of TPR at each FPR point
    mean_tpr_interp = np.mean(fold_tprs, axis=0)
    std_tpr_interp = np.std(fold_tprs, axis=0)
    mean_tpr_interp[-1] = 1.0
    
    # Create model identifier
    model_identifier = f"{config['model_name']}_{config['mil_architecture']}"
    
    # Comprehensive data for this model configuration
    model_roc_data = {
        'dataset': 'ECU_external_validation',
        'model_config': {
            'feature_extractor': config['model_name'],
            'mil_architecture': config['mil_architecture'],
            'identifier': model_identifier,
            'architecture_params': config.get('architecture_specific_params', {})
        },
        'mean_roc': {
            'fpr': fpr_mean.tolist(),
            'tpr': tpr_mean.tolist(),
            'thresholds': thresholds_mean.tolist(),
            'auc': float(auc_mean),
            'auc_std': float(np.std(fold_aucs))
        },
        'interpolated_roc': {
            'fpr_grid': mean_fpr_grid.tolist(),
            'tpr_mean': mean_tpr_interp.tolist(),
            'tpr_std': std_tpr_interp.tolist(),
            'tpr_upper': np.minimum(mean_tpr_interp + std_tpr_interp, 1).tolist(),
            'tpr_lower': np.maximum(mean_tpr_interp - std_tpr_interp, 0).tolist()
        },
        'fold_statistics': {
            'n_folds': len(all_fold_probs),
            'fold_aucs': fold_aucs,
            'mean_auc': float(np.mean(fold_aucs)),
            'std_auc': float(np.std(fold_aucs)),
            'min_auc': float(np.min(fold_aucs)),
            'max_auc': float(np.max(fold_aucs))
        },
        'operating_points': {
            'sensitivity_at_90_spec': float(get_sensitivity_at_specificity(fpr_mean, tpr_mean, 0.9)),
            'specificity_at_90_sens': float(get_specificity_at_sensitivity(fpr_mean, tpr_mean, 0.9)),
            'youden_index': float(np.max(tpr_mean - fpr_mean)),
            'youden_threshold': float(thresholds_mean[np.argmax(tpr_mean - fpr_mean)])
        }
    }
    
    # Save in multiple formats
    
    # 1. JSON format for the mean model performance
    json_path = os.path.join(output_dir, f'{model_identifier}_ecu_mean_roc_data.json')
    with open(json_path, 'w') as f:
        json.dump(model_roc_data, f, indent=2)
    
    # 2. NumPy format for computational efficiency
    npz_path = os.path.join(output_dir, f'{model_identifier}_ecu_mean_roc_arrays.npz')
    np.savez(
        npz_path,
        fpr_mean=fpr_mean,
        tpr_mean=tpr_mean,
        thresholds_mean=thresholds_mean,
        auc_mean=auc_mean,
        mean_probs=mean_probs,
        std_probs=std_probs,
        labels=labels,
        fpr_grid=mean_fpr_grid,
        tpr_mean_interp=mean_tpr_interp,
        tpr_std_interp=std_tpr_interp,
        fold_aucs=np.array(fold_aucs)
    )
    
    # 3. Simple CSV for quick reference
    summary_df = pd.DataFrame({
        'Model': [model_identifier],
        'Dataset': ['ECU'],
        'Feature_Extractor': [config['model_name']],
        'MIL_Architecture': [config['mil_architecture']],
        'Mean_AUC': [auc_mean],
        'Std_AUC': [np.std(fold_aucs)],
        'CI_Lower': [auc_mean - 1.96 * np.std(fold_aucs) / np.sqrt(len(fold_aucs))],
        'CI_Upper': [auc_mean + 1.96 * np.std(fold_aucs) / np.sqrt(len(fold_aucs))],
        'Sensitivity_at_90_Spec': [get_sensitivity_at_specificity(fpr_mean, tpr_mean, 0.9)],
        'Specificity_at_90_Sens': [get_specificity_at_sensitivity(fpr_mean, tpr_mean, 0.9)],
        'N_Folds': [len(all_fold_probs)]
    })
    csv_path = os.path.join(output_dir, f'{model_identifier}_ecu_mean_performance.csv')
    summary_df.to_csv(csv_path, index=False)
    
    print(f"\nECU Mean ROC data saved for {model_identifier}:")
    print(f"  JSON: {json_path}")
    print(f"  NumPy: {npz_path}")
    print(f"  Summary: {csv_path}")
    print(f"  Mean AUC: {auc_mean:.3f} ± {np.std(fold_aucs):.3f}")
    
    return model_roc_data

def test_individual_folds_ecu(model_folder, test_csv, output_dir=None):
    """Test each fold on ECU dataset and save mean ROC data for comparison"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load config
    config = load_config_from_folder(model_folder)
    print(f"\nECU External Validation - Inference Save")
    print(f"Model: {config['model_name']}")
    print(f"Architecture: {config['mil_architecture']}")
    
    # Create output directory
    if output_dir is None:
        model_name = os.path.basename(model_folder)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"ecu_results/{model_name}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")
    
    # Load ECU test dataset
    print(f"\nLoading ECU test dataset: {test_csv}")
    test_df = pd.read_csv(test_csv)
    test_dataset = ClassificationMILDataset(
        test_csv,
        label_column='RSHigh' if 'RSHigh' in test_df.columns else 'RS',
        threshold=25.0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=mil_collate_fn,
        num_workers=0
    )
    
    print(f"ECU test samples: {len(test_dataset)}")
    
    # Test each fold
    fold_results = []
    all_fold_probs = []
    saved_labels = None
    
    for fold in range(1, 6):
        checkpoint_path = os.path.join(model_folder, f'best_model_fold_{fold}.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found for fold {fold}")
            continue
        
        print(f"\nFold {fold} evaluation...")
        
        # Load model
        model = create_model_from_config(config, device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Evaluate
        probs, labels = evaluate_single_model(
            model, test_loader, device, config['mil_architecture']
        )
        
        # Save labels from first fold
        if fold == 1:
            saved_labels = labels
        
        # Calculate metrics
        evaluator = ClassificationEvaluator(n_classes=2)
        preds = np.argmax(probs, axis=1)
        evaluator.update(labels, preds, probs)
        metrics = evaluator.compute_all_metrics(verbose=False)
        
        fold_results.append({
            'fold': fold,
            **metrics
        })
        all_fold_probs.append(probs)
        
        print(f"  Fold {fold} AUC: {metrics['auroc']:.3f}")
    
    # Save mean ROC data for model comparison
    mean_roc_data = None
    if len(all_fold_probs) > 0 and saved_labels is not None:
        mean_roc_data = save_model_mean_roc_data_ecu(
            all_fold_probs,
            saved_labels,
            config,
            output_dir
        )
    
    # Summary
    fold_df = pd.DataFrame(fold_results)
    
    print(f"\n{'='*60}")
    print(f"ECU EXTERNAL VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Mean AUC: {fold_df['auroc'].mean():.3f} ± {fold_df['auroc'].std():.3f}")
    print(f"Results saved to: {output_dir}")
    
    return mean_roc_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ECU external validation inference save')
    parser.add_argument('--model-folder', type=str, required=True,
                      help='Path to folder containing fold models')
    parser.add_argument('--test-csv', type=str, required=True,
                      help='Path to ECU test CSV file')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory (default: auto-generated)')
    
    args = parser.parse_args()
    
    test_individual_folds_ecu(
        args.model_folder, 
        args.test_csv, 
        args.output_dir
    )