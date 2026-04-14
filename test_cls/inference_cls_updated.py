# test_cls/inference_cls_updated.py

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

# Import interpretability module
try:
    from interpretability.unified_interpretability import UnifiedInterpretability
    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    INTERPRETABILITY_AVAILABLE = False
    print("Warning: Interpretability module not found. Interpretability analysis will be skipped.")

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
    """Recreate model architecture from config - updated to match new config structure"""
    
    mil_architecture = config['mil_architecture']
    input_dim = get_input_dim_from_model_name(config['model_name'])
    n_classes = 2
    
    # Get architecture-specific params from the new config structure
    arch_params = config.get('architecture_specific_params', {})
    
    # Also check for old config format (backward compatibility)
    if not arch_params:
        # Old format - parameters at root level
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
    """Evaluate a single fold model - simplified since all models behave the same in eval"""
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
                
                # All models just return logits in eval mode (no special handling needed)
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
def run_interpretability_analysis(model, test_loader, test_csv, device, mil_architecture, 
                                 output_dir, n_visualize=10, dataset_name="test", save_pdf=True):
    """Run interpretability analysis on test set
    
    Args:
        save_pdf: If True, save visualizations as both PNG and PDF (default: True)
    """
    
    if not INTERPRETABILITY_AVAILABLE:
        print("Interpretability module not available. Skipping analysis.")
        return None
    
    # Create interpretability analyzer
    interp_dir = os.path.join(output_dir, f'interpretability_{dataset_name}')
    analyzer = UnifiedInterpretability(model_type=mil_architecture, save_dir=interp_dir)
    
    if mil_architecture == 'mean':
        print(f"Skipping interpretability for {mil_architecture} (no attention mechanism)")
        return None
    
    print(f"\nRunning interpretability analysis for {dataset_name} set...")
    print(f"  Saving visualizations as: PNG{' and PDF' if save_pdf else ''}")
    
    # Load the test CSV to get slide IDs and paths
    test_df = pd.read_csv(test_csv)
    
    all_metrics = []
    visualized_count = 0
    high_risk_visualized = 0
    low_risk_visualized = 0
    misclass_visualized = 0
    
    model.eval()
    with torch.no_grad():
        for idx, (features, label) in enumerate(tqdm(test_loader, desc="Interpretability")):
            # Handle batch format
            if isinstance(features, list):
                features = features[0]
            if isinstance(label, list):
                label = label[0]
            elif isinstance(label, torch.Tensor):
                label = label.item()
            
            # Get the corresponding row from the CSV
            csv_row = test_df.iloc[idx]
            feature_path = csv_row['path']
            slide_id = csv_row['slide_id']
            
            # Load the feature file to get coordinates
            if os.path.exists(feature_path):
                full_data = torch.load(feature_path, map_location='cpu')
                
                # Extract coordinates
                if 'coords' in full_data:
                    coordinates = full_data['coords'].numpy()
                else:
                    print(f"Warning: No coordinates found for {slide_id}")
                    continue
            else:
                print(f"Warning: Feature file not found: {feature_path}")
                continue
            
            # Get prediction
            features = features.to(device)
            logits = model(features)
            if logits.dim() == 2 and logits.shape[0] == 1:
                logits = logits.squeeze(0)
            
            probs = torch.softmax(logits, dim=0)
            pred_label = torch.argmax(logits).item()
            pred_prob = probs[1].item()  # Probability of high-risk
            
            # Decide whether to visualize - prioritize interesting cases
            is_misclassified = (pred_label != label)
            is_high_risk_correct = (pred_label == 1 and label == 1)
            is_low_risk_correct = (pred_label == 0 and label == 0)
            
            visualize = False
            if visualized_count < n_visualize:
                # Prioritize misclassifications
                if is_misclassified and misclass_visualized < n_visualize // 3:
                    visualize = True
                    misclass_visualized += 1
                # Then high-risk correct
                elif is_high_risk_correct and high_risk_visualized < n_visualize // 3:
                    visualize = True
                    high_risk_visualized += 1
                # Then low-risk correct
                elif is_low_risk_correct and low_risk_visualized < n_visualize // 3:
                    visualize = True
                    low_risk_visualized += 1
                # Fill remaining slots with any cases
                elif visualized_count < n_visualize:
                    visualize = True
                
                if visualize:
                    visualized_count += 1
            
            # Analyze slide with PDF saving option
            slide_metrics = analyzer.analyze_slide(
                model=model,
                features=features,
                coordinates=coordinates,
                slide_id=slide_id,
                true_label=int(label),
                pred_label=pred_label,
                pred_prob=pred_prob,
                visualize=visualize,
                save_pdf=save_pdf  # Pass the PDF save flag
            )
            
            # Add RS score if available
            slide_metrics['RS_score'] = csv_row['RS']
            slide_metrics['RS_category'] = csv_row['RSHigh']
            
            all_metrics.append(slide_metrics)
    
    # Create summary DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Filter out any rows with errors
    valid_metrics_df = metrics_df[~metrics_df.get('error', False).astype(bool)] if 'error' in metrics_df.columns else metrics_df
    
    # Calculate aggregate statistics (rest of the function remains the same...)
    if len(valid_metrics_df) > 0:
        summary = {
            'dataset': dataset_name,
            'n_samples': len(metrics_df),
            'n_valid_samples': len(valid_metrics_df),
            'mean_entropy': float(valid_metrics_df['entropy'].mean()) if 'entropy' in valid_metrics_df.columns else 0,
            'std_entropy': float(valid_metrics_df['entropy'].std()) if 'entropy' in valid_metrics_df.columns else 0,
            'mean_gini': float(valid_metrics_df['gini'].mean()) if 'gini' in valid_metrics_df.columns else 0,
            'std_gini': float(valid_metrics_df['gini'].std()) if 'gini' in valid_metrics_df.columns else 0,
            'mean_spatial_coherence': float(valid_metrics_df['spatial_coherence'].mean()) if 'spatial_coherence' in valid_metrics_df.columns else 0,
            'std_spatial_coherence': float(valid_metrics_df['spatial_coherence'].std()) if 'spatial_coherence' in valid_metrics_df.columns else 0,
            'mean_top_10_mass': float(valid_metrics_df['top_10_mass'].mean()) if 'top_10_mass' in valid_metrics_df.columns else 0,
            'std_top_10_mass': float(valid_metrics_df['top_10_mass'].std()) if 'top_10_mass' in valid_metrics_df.columns else 0,
            'saved_as_pdf': save_pdf  # Track whether PDFs were saved
        }
        
        # Compare correct vs incorrect predictions
        if 'correct' in valid_metrics_df.columns:
            correct_df = valid_metrics_df[valid_metrics_df['correct'] == True]
            incorrect_df = valid_metrics_df[valid_metrics_df['correct'] == False]
            
            if len(correct_df) > 0:
                summary['correct_mean_entropy'] = float(correct_df['entropy'].mean())
                summary['correct_mean_coherence'] = float(correct_df['spatial_coherence'].mean())
                summary['n_correct'] = len(correct_df)
            
            if len(incorrect_df) > 0:
                summary['incorrect_mean_entropy'] = float(incorrect_df['entropy'].mean())
                summary['incorrect_mean_coherence'] = float(incorrect_df['spatial_coherence'].mean())
                summary['n_incorrect'] = len(incorrect_df)
        
        # Compare high-risk vs low-risk
        if 'true_label' in valid_metrics_df.columns:
            high_risk_df = valid_metrics_df[valid_metrics_df['true_label'] == 1]
            low_risk_df = valid_metrics_df[valid_metrics_df['true_label'] == 0]
            
            if len(high_risk_df) > 0:
                summary['high_risk_mean_entropy'] = float(high_risk_df['entropy'].mean())
                summary['high_risk_mean_top10'] = float(high_risk_df['top_10_mass'].mean())
            
            if len(low_risk_df) > 0:
                summary['low_risk_mean_entropy'] = float(low_risk_df['entropy'].mean())
                summary['low_risk_mean_top10'] = float(low_risk_df['top_10_mass'].mean())
        
        # Save results
        metrics_df.to_csv(os.path.join(interp_dir, f'{dataset_name}_interpretability_metrics.csv'), index=False)
        
        with open(os.path.join(interp_dir, f'{dataset_name}_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nInterpretability Summary ({dataset_name}):")
        print(f"  Samples analyzed: {summary['n_samples']}")
        print(f"  Mean Entropy: {summary['mean_entropy']:.3f} ± {summary['std_entropy']:.3f}")
        print(f"  Mean Gini: {summary['mean_gini']:.3f} ± {summary['std_gini']:.3f}")
        print(f"  Mean Spatial Coherence: {summary['mean_spatial_coherence']:.3f}")
        print(f"  Mean Top-10 Mass: {summary['mean_top_10_mass']:.3f}")
        
        if 'n_correct' in summary and 'n_incorrect' in summary:
            print(f"\n  Correct predictions: {summary['n_correct']}")
            print(f"    Mean entropy: {summary.get('correct_mean_entropy', 0):.3f}")
            print(f"  Incorrect predictions: {summary['n_incorrect']}")
            print(f"    Mean entropy: {summary.get('incorrect_mean_entropy', 0):.3f}")
        
        print(f"\n  Visualizations saved: {visualized_count}")
        print(f"    Format: PNG{' and PDF' if save_pdf else ''}")
        print(f"    Misclassified: {misclass_visualized}")
        print(f"    High-risk correct: {high_risk_visualized}")
        print(f"    Low-risk correct: {low_risk_visualized}")
        print(f"  Results saved to: {interp_dir}")
        
        return metrics_df
    
    return None
def test_individual_folds(model_folder, test_csv, output_dir=None, run_interpretability=False, 
                         analyze_all_folds=False, n_visualize=15, save_pdf=True):
    """Test each fold individually without ensemble
    
    Args:
        model_folder: Path to folder containing fold models
        test_csv: Path to test CSV file
        output_dir: Output directory (default: auto-generated)
        run_interpretability: Whether to run interpretability analysis
        analyze_all_folds: Run interpretability on all folds vs best fold only
        n_visualize: Number of cases to visualize
        save_pdf: Save interpretability visualizations as PDF in addition to PNG
    """
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load config
    config = load_config_from_folder(model_folder)
    print(f"\nLoaded configuration:")
    print(f"Model: {config['model_name']}")
    print(f"Architecture: {config['mil_architecture']}")
    
    # Print architecture-specific settings from the new location
    arch_params = config.get('architecture_specific_params', {})
    
    if config['mil_architecture'] == 'attention':
        print(f"ABMIL Settings:")
        print(f"  Hidden Dim: {arch_params.get('hidden_dim', 256)}")
        print(f"  Attention Hidden Dim: {arch_params.get('attention_hidden_dim', 128)}")
        print(f"  Dropout: {arch_params.get('dropout', 0.25)}")
        print(f"  Learning Rate: {arch_params.get('learning_rate', 2e-4)}")
        print(f"  Weight Decay: {arch_params.get('weight_decay', 1e-5)}")
    
    elif config['mil_architecture'] == 'clam':
        print(f"CLAM Settings:")
        print(f"  Hidden Dim: {arch_params.get('hidden_dim', 512)}")
        print(f"  Attention Hidden Dim: {arch_params.get('attention_hidden_dim', 384)}")
        print(f"  Instance Learning: {arch_params.get('use_instance_learning', True)}")
        if arch_params.get('use_instance_learning'):
            print(f"  K-Sample: {arch_params.get('k_sample', 8)}")
            print(f"  Instance Loss: {arch_params.get('instance_loss_fn', 'svm')}")
            print(f"  Instance Loss Weight: {arch_params.get('instance_loss_weight', 0.7)}")
        print(f"  Learning Rate: {arch_params.get('learning_rate', 2e-4)}")
        print(f"  Weight Decay: {arch_params.get('weight_decay', 1e-5)}")
    
    elif config['mil_architecture'] == 'acmil':
        print(f"ACMIL Settings:")
        print(f"  Hidden Dim: {arch_params.get('hidden_dim', 256)}")
        print(f"  Branches: {arch_params.get('n_branches', 10)}")
        print(f"  Top-K: {arch_params.get('top_k', 10)}")
        print(f"  Mask Ratio: {arch_params.get('mask_ratio', 0.7)}")
        print(f"  Lambda P: {arch_params.get('lambda_p', 1.0)}")
        print(f"  Lambda D: {arch_params.get('lambda_d', 1.0)}")
        print(f"  Learning Rate: {arch_params.get('learning_rate', 1e-4)}")
        print(f"  Weight Decay: {arch_params.get('weight_decay', 5e-4)}")
        print(f"  Scheduler: {arch_params.get('scheduler_type', 'None')}")
    
    # Create output directory
    if output_dir is None:
        model_name = os.path.basename(model_folder)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"test_results_cls_updated/{model_name}_{timestamp}"
    
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
    best_fold_num = None
    best_fold_auroc = -1
    
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
        
        print(f"Model loaded from epoch {checkpoint.get('epoch', checkpoint.get('best_epoch', 0))}")
        print(f"Validation AUROC: {checkpoint.get('best_auroc', checkpoint.get('metrics', {}).get('auroc', 0)):.4f}")
        
        # Evaluate
        probs, labels = evaluate_single_model(
            model, test_loader, device, config['mil_architecture']
        )
        
        # Calculate metrics
        evaluator = ClassificationEvaluator(n_classes=2)
        preds = np.argmax(probs, axis=1)
        evaluator.update(labels, preds, probs)
        metrics = evaluator.compute_all_metrics(verbose=True)
        
        # Track best fold
        if metrics['auroc'] > best_fold_auroc:
            best_fold_auroc = metrics['auroc']
            best_fold_num = fold
        
        # Store results
        fold_results.append({
            'fold': fold,
            'val_auroc': checkpoint.get('best_auroc', checkpoint.get('metrics', {}).get('auroc', 0)),
            'best_epoch': checkpoint.get('epoch', checkpoint.get('best_epoch', 0)),
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
        
        # Run interpretability analysis if requested
        if run_interpretability and (analyze_all_folds or fold == best_fold_num):
            interp_results = run_interpretability_analysis(
                model=model,
                test_loader=test_loader,
                test_csv=test_csv,
                device=device,
                mil_architecture=config['mil_architecture'],
                output_dir=output_dir,
                n_visualize=n_visualize,
                dataset_name=f"fold_{fold}",
                save_pdf=save_pdf  # Pass PDF saving flag
            )
    
    # If we haven't run interpretability on best fold yet (because we didn't know which was best), run it now
    if run_interpretability and not analyze_all_folds and best_fold_num is not None:
        print(f"\n{'='*60}")
        print(f"Running interpretability analysis on best fold (Fold {best_fold_num})")
        print(f"{'='*60}")
        
        checkpoint_path = os.path.join(model_folder, f'best_model_fold_{best_fold_num}.pt')
        model = create_model_from_config(config, device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        interp_results = run_interpretability_analysis(
            model=model,
            test_loader=test_loader,
            test_csv=test_csv,
            device=device,
            mil_architecture=config['mil_architecture'],
            output_dir=output_dir,
            n_visualize=n_visualize,
            dataset_name=f"best_fold_{best_fold_num}",
            save_pdf=save_pdf  # Pass PDF saving flag
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
    if len(fold_df) > 0 and 'auroc' in fold_df.columns:
        best_fold = fold_df.loc[fold_df['auroc'].idxmax()]
        worst_fold = fold_df.loc[fold_df['auroc'].idxmin()]
        
        print(f"\n{'='*80}")
        print(f"BEST FOLD: {int(best_fold['fold'])}")
        print(f"  AUROC: {best_fold['auroc']:.4f}")
        print(f"  Sensitivity: {best_fold.get('sensitivity', 0):.4f}")
        print(f"  Specificity: {best_fold.get('specificity', 0):.4f}")
        
        print(f"\nWORST FOLD: {int(worst_fold['fold'])}")
        print(f"  AUROC: {worst_fold['auroc']:.4f}")
        print(f"  Sensitivity: {worst_fold.get('sensitivity', 0):.4f}")
        print(f"  Specificity: {worst_fold.get('specificity', 0):.4f}")
        
        best_fold_dict = best_fold.to_dict()
        worst_fold_dict = worst_fold.to_dict()
    else:
        best_fold_dict = {}
        worst_fold_dict = {}
    
    # Save results
    results_summary = {
        'model_folder': model_folder,
        'test_csv': test_csv,
        'n_test_samples': len(test_dataset),
        'n_folds': len(fold_results),
        'fold_metrics': fold_results,
        'summary_statistics': {
            metric: {
                'mean': float(fold_df[metric].mean()) if metric in fold_df.columns else 0,
                'std': float(fold_df[metric].std()) if metric in fold_df.columns else 0,
                'min': float(fold_df[metric].min()) if metric in fold_df.columns else 0,
                'max': float(fold_df[metric].max()) if metric in fold_df.columns else 0
            } for metric in metrics_to_report
        },
        'best_fold': best_fold_dict,
        'worst_fold': worst_fold_dict,
        'config': config,
        'interpretability_run': run_interpretability,
        'interpretability_pdf_saved': save_pdf if run_interpretability else False
    }
    
    # Save files
    with open(os.path.join(output_dir, 'individual_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, cls=NumpyEncoder)
    
    fold_df.to_csv(os.path.join(output_dir, 'fold_results.csv'), index=False)
    
    # Plot all ROC curves on one plot if we have results
    if len(all_fold_probs) > 0:
        plot_all_roc_curves(all_fold_probs, labels, fold_df, output_dir)
        plot_fold_comparison(fold_df, output_dir)
    
    print(f"\n{'='*80}")
    print(f"All results saved to: {output_dir}")
    if run_interpretability:
        print(f"Interpretability analysis completed and saved")
        print(f"  Formats: PNG{' and PDF' if save_pdf else ''}")
    
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
    plt.title('Internal Validation - ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save as PNG
    png_path = os.path.join(output_dir, 'all_folds_roc_curves.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    # Save as PDF for LaTeX
    pdf_path = os.path.join(output_dir, 'all_folds_roc_curves.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close()
    
    print(f"ROC curves plot saved to:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
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
    
    print(f"Fold comparison plot saved to: {os.path.join(output_dir, 'fold_performance_comparison.png')}")
# Add to your main function's argparse section:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test individual CV fold models')
    parser.add_argument('--model-folder', type=str, required=True,
                      help='Path to folder containing fold models')
    parser.add_argument('--test-csv', type=str, required=True,
                      help='Path to test CSV file')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory (default: auto-generated)')
    parser.add_argument('--run-interpretability', action='store_true',
                      help='Run interpretability analysis on test set')
    parser.add_argument('--analyze-all-folds', action='store_true',
                      help='Run interpretability on all folds (default: best fold only)')
    parser.add_argument('--n-visualize', type=int, default=15,
                      help='Number of cases to visualize (default: 15)')
    parser.add_argument('--no-pdf', action='store_true',
                      help='Disable PDF saving for interpretability visualizations')
    
    args = parser.parse_args()
    
    # Determine whether to save PDFs (default is True unless --no-pdf is specified)
    save_pdf = not args.no_pdf
    
    test_individual_folds(
        model_folder=args.model_folder, 
        test_csv=args.test_csv, 
        output_dir=args.output_dir,
        run_interpretability=args.run_interpretability,
        analyze_all_folds=args.analyze_all_folds,
        n_visualize=args.n_visualize,
        save_pdf=save_pdf
    )
################


# Test with interpretability on best fold only
# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/h-optimus_clam_20250912_145351 --test-csv data/manifests/test_features_h-optimus.csv --run-interpretability

### new

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/h-optimus_acmil_20250914_204945 --test-csv data/manifests/test_features_h-optimus.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/h-optimus_attention_20250914_231455 --test-csv data/manifests/test_features_h-optimus.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/h-optimus_clam_20250914_231419 --test-csv data/manifests/test_features_h-optimus.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/h-optimus_mean_20250914_231513 --test-csv data/manifests/test_features_h-optimus.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/virchow2_acmil_20250915_000404 --test-csv data/manifests/test_features_virchow2.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/virchow2_attention_20250915_003518 --test-csv data/manifests/test_features_virchow2.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/virchow2_clam_20250915_002422 --test-csv data/manifests/test_features_virchow2.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/virchow2_mean_20250915_004613 --test-csv data/manifests/test_features_virchow2.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/uni2-h_acmil_20250915_000434 --test-csv data/manifests/test_features_uni2-h.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/uni2-h_attention_20250915_003535 --test-csv data/manifests/test_features_uni2-h.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/uni2-h_clam_20250915_002443 --test-csv data/manifests/test_features_uni2-h.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/uni2-h_mean_20250915_004627 --test-csv data/manifests/test_features_uni2-h.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/conch_acmil_20250915_000515 --test-csv data/manifests/test_features_conch.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/conch_attention_20250915_005221 --test-csv data/manifests/test_features_conch.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/conch_clam_20250915_004600 --test-csv data/manifests/test_features_conch.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/conch_mean_20250915_005832 --test-csv data/manifests/test_features_conch.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/resnet50_acmil_20250915_000619 --test-csv data/manifests/test_features_resnet50.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/resnet50_attention_20250915_003616 --test-csv data/manifests/test_features_resnet50.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/resnet50_clam_20250915_002521 --test-csv data/manifests/test_features_resnet50.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/resnet50_mean_20250915_004639 --test-csv data/manifests/test_features_resnet50.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/resnet18_acmil_20250915_000643 --test-csv data/manifests/test_features_resnet18.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/resnet18_attention_20250915_003630 --test-csv data/manifests/test_features_resnet18.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/resnet18_clam_20250915_002540 --test-csv data/manifests/test_features_resnet18.csv 

# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/resnet18_mean_20250915_004647 --test-csv data/manifests/test_features_resnet18.csv 
