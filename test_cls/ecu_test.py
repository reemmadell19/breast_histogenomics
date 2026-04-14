# external validation test pipeline on ECU dataset
# test_cls/ecu_test.py

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
from models.classification_model import (
    MeanPoolingMILClassifier, AttentionMILClassifier, 
    CLAMClassifier, ACMILClassifier
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
    
    if mil_architecture == "mean":
        model = MeanPoolingMILClassifier(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            n_classes=n_classes
        )
    
    elif mil_architecture == "attention":
        model = AttentionMILClassifier(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            attention_hidden_dim=config['attention_hidden_dim'],
            n_classes=n_classes
        )
    
    elif mil_architecture == "clam":
        model = CLAMClassifier(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            attention_hidden_dim=config['attention_hidden_dim'],
            n_classes=n_classes,
            dropout=config['dropout'],
            gate=config.get('gate', True)
        )
    
    elif mil_architecture == "acmil":
        model = ACMILClassifier(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            n_branches=config['n_branches'],
            n_classes=n_classes,
            n_masked_patch=config['n_masked_patch'],
            mask_ratio=config['mask_ratio'],
            dropout=config['dropout']
        )
    
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
                
                if mil_architecture in ["acmil"]:
                    logits = model(feat, return_branch_outputs=False)
                else:
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

def test_individual_folds(model_folder, test_csv, output_dir=None):
    """Test each fold individually without ensemble"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load config
    config = load_config_from_folder(model_folder)
    print(f"\nLoaded configuration:")
    print(f"Model: {config['model_name']}")
    print(f"Architecture: {config['mil_architecture']}")
    
    # Create output directory
    if output_dir is None:
        model_name = os.path.basename(model_folder)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"external_validation/ecu/{model_name}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")
    
    # Load test dataset
    print(f"\nLoading test dataset: {test_csv}")
    test_dataset = ClassificationMILDataset(
        test_csv,
        label_column='RSHigh' if 'RSHigh' in pd.read_csv(test_csv).columns else 'RS',
        threshold=25.0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=mil_collate_fn,
        num_workers=0
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Test each fold
    fold_results = []
    all_fold_probs = []
    
    for fold in range(1, 6):
        checkpoint_path = os.path.join(model_folder, f'best_model_fold_{fold}.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found for fold {fold}")
            continue
        
        print(f"\n{'='*60}")
        print(f"FOLD {fold} EVALUATION")
        print(f"{'='*60}")
        
        # Load model
        model = create_model_from_config(config, device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded from epoch {checkpoint.get('best_epoch', 0)}")
        print(f"Validation AUROC: {checkpoint.get('best_auroc', 0):.4f}")
        
        # Evaluate
        probs, labels = evaluate_single_model(
            model, test_loader, device, config['mil_architecture']
        )
        
        # Calculate metrics
        evaluator = ClassificationEvaluator(n_classes=2)
        preds = np.argmax(probs, axis=1)
        evaluator.update(labels, preds, probs)
        metrics = evaluator.compute_all_metrics(verbose=True)
        
        # Store results
        fold_results.append({
            'fold': fold,
            'val_auroc': checkpoint.get('best_auroc', 0),
            'best_epoch': checkpoint.get('best_epoch', 0),
            **metrics
        })
        all_fold_probs.append(probs)
        
        # Save individual fold plots
        evaluator.plot_confusion_matrix(
            save_path=os.path.join(output_dir, f'fold_{fold}_confusion_matrix.png'),
            title=f'Fold {fold} - Confusion Matrix'
        )
        
        evaluator.plot_roc_curve(
            save_path=os.path.join(output_dir, f'fold_{fold}_roc_curve.png'),
            title=f'Fold {fold} - ROC Curve (AUC={metrics["auroc"]:.3f})'
        )
    
    # Summary statistics
    fold_df = pd.DataFrame(fold_results)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY ACROSS ALL FOLDS")
    print(f"{'='*80}")
    
    metrics_to_report = ['auroc', 'auc_pr', 'accuracy', 'balanced_accuracy', 
                        'f1_score', 'sensitivity', 'specificity', 'mcc']
    
    for metric in metrics_to_report:
        if metric in fold_df.columns:
            mean_val = fold_df[metric].mean()
            std_val = fold_df[metric].std()
            min_val = fold_df[metric].min()
            max_val = fold_df[metric].max()
            print(f"{metric.upper():18s}: {mean_val:.4f} ± {std_val:.4f} "
                  f"[{min_val:.4f}, {max_val:.4f}]")
    
    # Best and worst folds
    best_fold = fold_df.loc[fold_df['auroc'].idxmax()]
    worst_fold = fold_df.loc[fold_df['auroc'].idxmin()]
    
    print(f"\n{'='*80}")
    print(f"BEST FOLD: {int(best_fold['fold'])}")
    print(f"  AUROC: {best_fold['auroc']:.4f}")
    print(f"  Sensitivity: {best_fold['sensitivity']:.4f}")
    print(f"  Specificity: {best_fold['specificity']:.4f}")
    
    print(f"\nWORST FOLD: {int(worst_fold['fold'])}")
    print(f"  AUROC: {worst_fold['auroc']:.4f}")
    print(f"  Sensitivity: {worst_fold['sensitivity']:.4f}")
    print(f"  Specificity: {worst_fold['specificity']:.4f}")
    
    # Save results
    results_summary = {
        'model_folder': model_folder,
        'test_csv': test_csv,
        'n_test_samples': len(test_dataset),
        'n_folds': len(fold_results),
        'fold_metrics': fold_results,
        'summary_statistics': {
            metric: {
                'mean': fold_df[metric].mean(),
                'std': fold_df[metric].std(),
                'min': fold_df[metric].min(),
                'max': fold_df[metric].max()
            } for metric in metrics_to_report if metric in fold_df.columns
        },
        'best_fold': best_fold.to_dict(),
        'worst_fold': worst_fold.to_dict()
    }
    
    # Save files
    with open(os.path.join(output_dir, 'ecu_individual_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, cls=NumpyEncoder)
    
    fold_df.to_csv(os.path.join(output_dir, 'fold_results.csv'), index=False)
    
    # Plot all ROC curves on one plot
    plot_all_roc_curves(all_fold_probs, labels, fold_df, output_dir)
    
    # Plot performance comparison
    plot_fold_comparison(fold_df, output_dir)
    
    print(f"\n{'='*80}")
    print(f"All results saved to: {output_dir}")
    
    return results_summary
def plot_all_roc_curves(all_fold_probs, labels, fold_df, output_dir):
    """Plot all fold ROC curves with mean and std on one plot"""
    
    plt.figure(figsize=(10, 8))
    
    # Plot individual fold ROC curves
    for fold_idx, (probs, row) in enumerate(zip(all_fold_probs, fold_df.iterrows()), 1):
        _, fold_metrics = row
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        fold_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=1.5, alpha=0.4,
                label=f'Fold {int(fold_metrics["fold"])} (AUC = {fold_auc:.3f})')
    
    # Calculate mean and std ROC curves
    mean_fpr = np.linspace(0, 1, 100)
    tpr_list = []
    
    for probs in all_fold_probs:
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_list.append(interp_tpr)
    
    mean_tpr = np.mean(tpr_list, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tpr_list, axis=0)
    
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std([auc(mean_fpr, tpr) for tpr in tpr_list])
    
    # Plot mean curve
    plt.plot(mean_fpr, mean_tpr, 'b-', linewidth=2.5,
            label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})', 
            alpha=0.8)
    
    # Plot ± 1 std deviation
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, 
                     color='grey', alpha=0.2,
                     label='± 1 std. dev.')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    # Formatting
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - All Folds with Mean ± Std', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_folds_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curves plot saved to: {os.path.join(output_dir, 'all_folds_roc_curves.png')}")
    print(f"Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}")


def plot_fold_comparison(fold_df, output_dir):
    """Plot performance metrics comparison across folds"""
    
    metrics = ['auroc', 'sensitivity', 'specificity', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        if metric in fold_df.columns:
            ax = axes[idx]
            folds = fold_df['fold'].values
            values = fold_df[metric].values
            
            bars = ax.bar(folds, values, color='skyblue', edgecolor='navy')
            
            # Highlight best and worst
            best_idx = np.argmax(values)
            worst_idx = np.argmin(values)
            bars[best_idx].set_color('green')
            bars[worst_idx].set_color('red')
            
            ax.axhline(y=values.mean(), color='orange', linestyle='--', 
                      label=f'Mean: {values.mean():.3f}')
            ax.set_xlabel('Fold')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Across Folds')
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fold_performance_comparison.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test individual CV fold models')
    parser.add_argument('--model-folder', type=str, required=True,
                      help='Path to folder containing fold models')
    parser.add_argument('--test-csv', type=str, required=True,
                      help='Path to test CSV file')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory (default: auto-generated)')
    
    args = parser.parse_args()
    
    test_individual_folds(args.model_folder, args.test_csv, args.output_dir)
################

# test commands

# h-optimus acmil, focal
# python test_cls/ecu_test.py --model-folder results_classification_cv/bestf_h-optimus_acmil_20250904_134437 --test-csv data/manifests/ecu_features_h-optimus.csv

# h-optimus acmil, cross
# python test_cls/ecu_test.py --model-folder results_classification_cv/best_h-optimus_acmil_20250904_113517 --test-csv data/manifests/ecu_features_h-optimus.csv

# batch size = 1, best performance = 0.87 auc, cross, deleted the folder as perfroamnce dropped from 0.87 to 0.81 (does not generalise well)
# python test_cls/ecu_test.py --model-folder results_classification_cv/h-optimus_acmil_20250902_193855 --test-csv data/manifests/ecu_features_h-optimus.csv

# h-optimus clam, focal
# python test_cls/ecu_test.py --model-folder results_classification_cv/h-optimus_clam_20250904_160103 --test-csv data/manifests/ecu_features_h-optimus.csv

# h-optimus attention, focal
# python test_cls/ecu_test.py --model-folder results_classification_cv/h-optimus_attention_20250904_145044 --test-csv data/manifests/ecu_features_h-optimus.csv

# virchow2 acmil, focal
# python test_cls/ecu_test.py --model-folder results_classification_cv/virchow2_acmil_20250904_150000 --test-csv data/manifests/ecu_features_virchow2.csv

# virchow2 clam, focal
# python test_cls/ecu_test.py --model-folder results_classification_cv/virchow2_clam_20250904_160938 --test-csv data/manifests/ecu_features_virchow2.csv

# virchow2 attention, focal
# python test_cls/ecu_test.py --model-folder results_classification_cv/virchow2_attention_20250904_165125 --test-csv data/manifests/ecu_features_virchow2.csv

# uni2-h acmil, focal, hidden_dim = 64
# python test_cls/ecu_test.py --model-folder results_classification_cv/uni2-h_acmil_20250904_154233 --test-csv data/manifests/ecu_features_uni2-h.csv

# uni2-h acmil, focal, hidden_dim = 48, best so far (auc = 0.8338)
# python test_cls/ecu_test.py --model-folder results_classification_cv/uni2-h_acmil_20250904_150045 --test-csv data/manifests/ecu_features_uni2-h.csv

# uni2-h clam, focal
# python test_cls/ecu_test.py --model-folder results_classification_cv/uni2-h_clam_20250904_161003 --test-csv data/manifests/ecu_features_uni2-h.csv

# uni2-h attention, focal
# python test_cls/ecu_test.py --model-folder results_classification_cv/uni2-h_attention_20250904_165149 --test-csv data/manifests/ecu_features_uni2-h.csv

# conch acmil, focal
# python test_cls/ecu_test.py --model-folder results_classification_cv/conch_acmil_20250904_150112 --test-csv data/manifests/ecu_features_conch.csv

# conch clam, focal
# python test_cls/ecu_test.py --model-folder results_classification_cv/conch_clam_20250904_161028 --test-csv data/manifests/ecu_features_conch.csv

# conch attention, focal
# python test_cls/ecu_test.py --model-folder results_classification_cv/conch_attention_20250904_165217 --test-csv data/manifests/ecu_features_conch.csv

# resnet18 acmil, focal
# python test_cls/ecu_test.py --model-folder results_classification_cv/resnet18_acmil_20250904_150222 --test-csv data/manifests/ecu_features_resnet18.csv

# resnet18 clam, focal
# python test_cls/ecu_test.py --model-folder results_classification_cv/resnet18_clam_20250904_161048 --test-csv data/manifests/ecu_features_resnet18.csv

# resnet18 attention, focal
# python test_cls/ecu_test.py --model-folder results_classification_cv/resnet18_attention_20250904_165232 --test-csv data/manifests/ecu_features_resnet18.csv

# resnet50 acmil, focal
# python test_cls/ecu_test.py --model-folder results_classification_cv/resnet50_acmil_20250904_152341 --test-csv data/manifests/ecu_features_resnet50.csv

# resnet50 clam, focal
# python test_cls/ecu_test.py --model-folder results_classification_cv/resnet50_clam_20250904_161110 --test-csv data/manifests/ecu_features_resnet50.csv

# resnet50 attention, focal
# python test_cls/ecu_test.py --model-folder results_classification_cv/resnet50_attention_20250904_165356 --test-csv data/manifests/ecu_features_resnet50.csv


