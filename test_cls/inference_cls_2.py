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

def save_detailed_predictions(all_fold_probs, labels, test_df, output_dir, fold_results):
    """
    Save detailed predictions for each sample including slide names, true labels, 
    predicted labels, and probabilities for all folds.
    """
    # Create a comprehensive predictions dataframe
    predictions_list = []
    
    # Get slide information from test_df
    slide_ids = test_df['slide_id'].values
    paths = test_df['path'].values
    
    # Additional metadata if available
    has_rs = 'RS' in test_df.columns
    has_rshigh = 'RSHigh' in test_df.columns
    
    if has_rs:
        rs_scores = test_df['RS'].values
    if has_rshigh:
        rs_high = test_df['RSHigh'].values
    
    # Process predictions for each sample
    for idx in range(len(labels)):
        pred_dict = {
            'sample_index': idx,
            'slide_id': slide_ids[idx],
            'file_path': paths[idx],
            'true_label': int(labels[idx])
        }
        
        # Add RS metadata if available
        if has_rs:
            pred_dict['RS_score'] = rs_scores[idx]
        if has_rshigh:
            pred_dict['RS_high_category'] = rs_high[idx]
        
        # Add predictions from each fold
        fold_predictions = []
        fold_probs_0 = []
        fold_probs_1 = []
        
        for fold_idx, probs in enumerate(all_fold_probs):
            if idx < len(probs):
                prob_0 = probs[idx, 0]
                prob_1 = probs[idx, 1]
                pred = np.argmax(probs[idx])
                
                pred_dict[f'fold_{fold_idx+1}_pred'] = int(pred)
                pred_dict[f'fold_{fold_idx+1}_prob_class0'] = float(prob_0)
                pred_dict[f'fold_{fold_idx+1}_prob_class1'] = float(prob_1)
                
                fold_predictions.append(pred)
                fold_probs_0.append(prob_0)
                fold_probs_1.append(prob_1)
        
        # Calculate ensemble predictions (mean across folds)
        if fold_predictions:
            pred_dict['ensemble_prob_class0'] = float(np.mean(fold_probs_0))
            pred_dict['ensemble_prob_class1'] = float(np.mean(fold_probs_1))
            pred_dict['ensemble_pred'] = int(pred_dict['ensemble_prob_class1'] >= 0.5)
            pred_dict['ensemble_prob_std'] = float(np.std(fold_probs_1))
            
            # Voting-based ensemble
            unique, counts = np.unique(fold_predictions, return_counts=True)
            pred_dict['majority_vote_pred'] = int(unique[np.argmax(counts)])
            pred_dict['vote_confidence'] = float(np.max(counts) / len(fold_predictions))
            
            # Agreement across folds
            pred_dict['fold_agreement'] = int(all(p == fold_predictions[0] for p in fold_predictions))
            
            # Correct prediction flags
            pred_dict['ensemble_correct'] = int(pred_dict['ensemble_pred'] == pred_dict['true_label'])
            pred_dict['majority_vote_correct'] = int(pred_dict['majority_vote_pred'] == pred_dict['true_label'])
        
        predictions_list.append(pred_dict)
    
    # Create DataFrame
    predictions_df = pd.DataFrame(predictions_list)
    
    # Sort by slide_id for easier reference
    predictions_df = predictions_df.sort_values('slide_id')
    
    # Save to CSV
    predictions_path = os.path.join(output_dir, 'all_predictions_detailed.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nDetailed predictions saved to: {predictions_path}")
    
    # Create summary statistics
    summary_stats = {
        'total_samples': len(predictions_df),
        'ensemble_accuracy': float(predictions_df['ensemble_correct'].mean()) if 'ensemble_correct' in predictions_df else 0,
        'majority_vote_accuracy': float(predictions_df['majority_vote_correct'].mean()) if 'majority_vote_correct' in predictions_df else 0,
        'samples_with_fold_agreement': int(predictions_df['fold_agreement'].sum()) if 'fold_agreement' in predictions_df else 0,
        'agreement_rate': float(predictions_df['fold_agreement'].mean()) if 'fold_agreement' in predictions_df else 0
    }
    
    # Add class-specific metrics
    for true_class in [0, 1]:
        class_df = predictions_df[predictions_df['true_label'] == true_class]
        if len(class_df) > 0:
            summary_stats[f'class_{true_class}_samples'] = len(class_df)
            if 'ensemble_correct' in class_df:
                summary_stats[f'class_{true_class}_ensemble_accuracy'] = float(class_df['ensemble_correct'].mean())
            if 'fold_agreement' in class_df:
                summary_stats[f'class_{true_class}_agreement_rate'] = float(class_df['fold_agreement'].mean())
    
    # Identify challenging cases (high uncertainty or disagreement)
    if 'ensemble_prob_std' in predictions_df and 'fold_agreement' in predictions_df:
        high_uncertainty = predictions_df.nlargest(10, 'ensemble_prob_std')[['slide_id', 'true_label', 'ensemble_pred', 'ensemble_prob_std']]
        disagreement_cases = predictions_df[predictions_df['fold_agreement'] == 0][['slide_id', 'true_label', 'ensemble_pred', 'vote_confidence']]
        
        # Save challenging cases
        high_uncertainty.to_csv(os.path.join(output_dir, 'high_uncertainty_cases.csv'), index=False)
        if len(disagreement_cases) > 0:
            disagreement_cases.to_csv(os.path.join(output_dir, 'fold_disagreement_cases.csv'), index=False)
        
        summary_stats['high_uncertainty_cases'] = len(high_uncertainty)
        summary_stats['fold_disagreement_cases'] = len(disagreement_cases)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'prediction_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"Prediction summary saved to: {summary_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Total samples: {summary_stats['total_samples']}")
    if 'ensemble_accuracy' in summary_stats:
        print(f"Ensemble accuracy: {summary_stats['ensemble_accuracy']:.4f}")
    if 'majority_vote_accuracy' in summary_stats:
        print(f"Majority vote accuracy: {summary_stats['majority_vote_accuracy']:.4f}")
    if 'agreement_rate' in summary_stats:
        print(f"Fold agreement rate: {summary_stats['agreement_rate']:.4f}")
    
    # Print class-specific results
    for true_class in [0, 1]:
        if f'class_{true_class}_samples' in summary_stats:
            print(f"\nClass {true_class} (n={summary_stats[f'class_{true_class}_samples']}):")
            if f'class_{true_class}_ensemble_accuracy' in summary_stats:
                print(f"  Accuracy: {summary_stats[f'class_{true_class}_ensemble_accuracy']:.4f}")
            if f'class_{true_class}_agreement_rate' in summary_stats:
                print(f"  Fold agreement: {summary_stats[f'class_{true_class}_agreement_rate']:.4f}")
    
    return predictions_df, summary_stats

def test_format_loading(test_csv, tfrecord_dir, h5_dir):
    """Quick test to verify both formats can be loaded"""
    from interpretability.dual_format_interpretability import DualFormatInterpretability
    
    analyzer = DualFormatInterpretability(
        model_type='test',
        save_dir='test_output',
        tfrecord_dir=tfrecord_dir,
        h5_dir=h5_dir
    )
    
    test_df = pd.read_csv(test_csv)
    
    # Find one example of each format
    h5_example = None
    tfrecord_example = None
    
    for _, row in test_df.iterrows():
        slide_id = str(row['slide_id'])
        if slide_id.strip().isdigit() and h5_example is None:
            h5_example = slide_id
        elif 'UCH_BRCA_RS' in slide_id and tfrecord_example is None:
            tfrecord_example = slide_id
        
        if h5_example and tfrecord_example:
            break
    
    print("\n" + "="*60)
    print("FORMAT LOADING TEST")
    print("="*60)
    
    if h5_example:
        print(f"Testing H5 (BCR_NET) loading with: {h5_example}")
        patches = analyzer.load_patches_by_index(h5_example, max_patches=5)
        print(f"  ✓ Loaded {len(patches)} patches from H5")
    else:
        print("  ✗ No H5 examples found in test set")
    
    if tfrecord_example:
        print(f"Testing TFRecord (UCMC) loading with: {tfrecord_example}")
        patches = analyzer.load_patches_by_index(tfrecord_example, max_patches=5)
        print(f"  ✓ Loaded {len(patches)} patches from TFRecord")
    else:
        print("  ✗ No TFRecord examples found in test set")
    
    print("="*60 + "\n")

def run_interpretability_analysis(model, test_loader, test_csv, device, mil_architecture, 
                                 output_dir, n_visualize=10, dataset_name="test",
                                 tfrecord_dir=None, h5_dir=None, visualize_patches=True,
                                 n_patches_per_category=5):
    """Run patch visualization analysis with balanced format selection"""
    
    # Import the dual-format interpretability module
    try:
        from interpretability.dual_format_interpretability import DualFormatInterpretability
        INTERPRETABILITY_AVAILABLE = True
    except ImportError:
        print("Dual-format interpretability module not found.")
        return None
    
    if not INTERPRETABILITY_AVAILABLE:
        print("Interpretability not available")
        return None
    
    # Create interpretability analyzer
    interp_dir = os.path.join(output_dir, f'interpretability_{dataset_name}')
    analyzer = DualFormatInterpretability(
        model_type=mil_architecture, 
        save_dir=interp_dir,
        tfrecord_dir=tfrecord_dir,
        h5_dir=h5_dir
    )
    
    if mil_architecture == 'mean':
        print(f"Skipping interpretability for {mil_architecture} (no attention mechanism)")
        return None
    
    print(f"\nRunning patch visualization analysis for {dataset_name} set...")
    print(f"Configuration:")
    if tfrecord_dir:
        print(f"  TFRecord directory (UCMC): {tfrecord_dir}")
    if h5_dir:
        print(f"  H5 directory (BCR_NET): {h5_dir}")
    print(f"  Target visualizations: {n_visualize} cases")
    print(f"  Patches per attention level: {n_patches_per_category}")
    
    # Load the test CSV
    test_df = pd.read_csv(test_csv)
    
    # First pass: categorize all samples AND identify their format
    all_predictions = []
    model.eval()
    
    print("\nCategorizing samples and identifying formats...")
    with torch.no_grad():
        for idx, (features, label) in enumerate(test_loader):
            # Handle batch format
            if isinstance(features, list):
                features = features[0]
            if isinstance(label, list):
                label = label[0]
            elif isinstance(label, torch.Tensor):
                label = label.item()
            
            # Get prediction
            features = features.to(device)
            logits = model(features)
            if logits.dim() == 2 and logits.shape[0] == 1:
                logits = logits.squeeze(0)
            
            probs = torch.softmax(logits, dim=0)
            pred_label = torch.argmax(logits).item()
            pred_prob = probs[1].item()
            
            # Categorize
            is_correct = (pred_label == label)
            category = ""
            if not is_correct:
                category = "misclassified"
            elif pred_label == 1 and label == 1:
                category = "correct_high_risk"
            elif pred_label == 0 and label == 0:
                category = "correct_low_risk"
            
            # Identify the format based on slide_id
            slide_id = str(test_df.iloc[idx]['slide_id'])
            data_format = 'h5' if slide_id.strip().isdigit() else 'tfrecord'
            
            all_predictions.append({
                'idx': idx,
                'true_label': label,
                'pred_label': pred_label,
                'pred_prob': pred_prob,
                'category': category,
                'slide_id': slide_id,
                'data_format': data_format  # Add format tracking
            })
    
    # Count by both category and format
    format_category_counts = {}
    for pred in all_predictions:
        key = (pred['data_format'], pred['category'])
        format_category_counts[key] = format_category_counts.get(key, 0) + 1
    
    print(f"\nSample distribution by format and category:")
    for format_type in ['h5', 'tfrecord']:
        format_name = 'BCR_NET' if format_type == 'h5' else 'UCMC'
        print(f"  {format_name} ({format_type}):")
        for cat in ['misclassified', 'correct_high_risk', 'correct_low_risk']:
            count = format_category_counts.get((format_type, cat), 0)
            print(f"    {cat}: {count}")
    
    # Select samples with balanced format representation
    samples_to_visualize = []
    
    # Try to get equal representation from both formats
    target_per_format = n_visualize // 2
    remainder = n_visualize % 2
    
    # Collect samples from each format
    h5_samples = []
    tfrecord_samples = []
    
    # Prioritize diverse categories within each format
    for format_type, format_list in [('h5', h5_samples), ('tfrecord', tfrecord_samples)]:
        format_preds = [p for p in all_predictions if p['data_format'] == format_type]
        
        if not format_preds:
            print(f"  Warning: No {format_type} samples found in predictions")
            continue
        
        # Calculate target samples for this format
        format_target = target_per_format + (remainder if format_type == 'h5' else 0)
        
        # Try to get samples from each category
        categories_priority = ['correct_high_risk', 'misclassified', 'correct_low_risk']
        samples_per_category = max(1, format_target // 3)
        
        for cat in categories_priority:
            cat_samples = [p for p in format_preds if p['category'] == cat]
            
            if not cat_samples:
                continue
            
            # Sort by prediction probability for diversity
            cat_samples.sort(key=lambda x: x['pred_prob'])
            
            # Take samples evenly distributed across probability range
            n_to_take = min(samples_per_category, len(cat_samples))
            if n_to_take > 0:
                if n_to_take == 1:
                    indices = [0]
                else:
                    indices = np.linspace(0, len(cat_samples)-1, n_to_take, dtype=int)
                
                for i in indices:
                    if cat_samples[i]['idx'] not in format_list:
                        format_list.append(cat_samples[i]['idx'])
                    if len(format_list) >= format_target:
                        break
            
            if len(format_list) >= format_target:
                break
        
        # If we still need more samples from this format, add any available
        if len(format_list) < format_target:
            remaining = [p['idx'] for p in format_preds if p['idx'] not in format_list]
            format_list.extend(remaining[:format_target - len(format_list)])
    
    # Combine samples
    samples_to_visualize = h5_samples + tfrecord_samples
    
    print(f"\nWill visualize {len(samples_to_visualize)} samples:")
    print(f"  H5 (BCR_NET): {len(h5_samples)} samples")
    print(f"  TFRecord (UCMC): {len(tfrecord_samples)} samples")
    
    # Show which specific samples were selected
    selected_details = {'h5': [], 'tfrecord': []}
    for idx in samples_to_visualize:
        pred = all_predictions[idx]
        selected_details[pred['data_format']].append({
            'slide_id': pred['slide_id'],
            'category': pred['category'],
            'pred_prob': pred['pred_prob']
        })
    
    print("\nSelected samples by format:")
    for format_type in ['h5', 'tfrecord']:
        format_name = 'BCR_NET' if format_type == 'h5' else 'UCMC'
        print(f"  {format_name} ({format_type}):")
        for sample in selected_details[format_type][:5]:  # Show first 5 of each
            print(f"    - {sample['slide_id'][:20]}... ({sample['category']}, prob={sample['pred_prob']:.3f})")
        if len(selected_details[format_type]) > 5:
            print(f"    ... and {len(selected_details[format_type])-5} more")
    
    # Second pass: generate visualizations
    all_metrics = []
    visualized_count = 0
    h5_visualized = 0
    tfrecord_visualized = 0
    
    model.eval()
    with torch.no_grad():
        for idx, (features, label) in enumerate(tqdm(test_loader, desc="Generating patch visualizations")):
            
            # Skip if not selected
            if idx not in samples_to_visualize:
                continue
            
            # Handle batch format
            if isinstance(features, list):
                features = features[0]
            if isinstance(label, list):
                label = label[0]
            elif isinstance(label, torch.Tensor):
                label = label.item()
            
            # Get slide info
            csv_row = test_df.iloc[idx]
            feature_path = csv_row['path']
            slide_id = str(csv_row['slide_id'])
            
            # Track format
            is_h5 = slide_id.strip().isdigit()
            if is_h5:
                h5_visualized += 1
            else:
                tfrecord_visualized += 1
            
            # Load coordinates
            coordinates = None
            if os.path.exists(feature_path):
                full_data = torch.load(feature_path, map_location='cpu')
                if 'coords' in full_data:
                    coordinates = full_data['coords'].numpy()
            
            # Get prediction
            features = features.to(device)
            logits = model(features)
            if logits.dim() == 2 and logits.shape[0] == 1:
                logits = logits.squeeze(0)
            
            probs = torch.softmax(logits, dim=0)
            pred_label = torch.argmax(logits).item()
            pred_prob = probs[1].item()
            
            # Find category
            pred_info = all_predictions[idx]
            
            format_name = 'BCR_NET' if is_h5 else 'UCMC'
            print(f"\nVisualizing {slide_id[:30]}... ({format_name} - {pred_info['category']})")
            
            # Analyze slide with patch visualization
            slide_metrics = analyzer.analyze_slide(
                model=model,
                features=features,
                coordinates=coordinates,
                slide_id=slide_id,
                true_label=int(label),
                pred_label=pred_label,
                pred_prob=pred_prob,
                visualize_patches=True,
                n_patches_per_category=n_patches_per_category
            )
            
            # Check if slide_metrics is None
            if slide_metrics is None:
                slide_metrics = {'slide_id': slide_id, 'error': 'No patches found'}
            
            # Add metadata
            if 'RS' in csv_row:
                slide_metrics['RS_score'] = csv_row['RS']
            if 'RSHigh' in csv_row:
                slide_metrics['RS_category'] = csv_row['RSHigh']
            slide_metrics['visualization_category'] = pred_info['category']
            slide_metrics['data_format'] = pred_info['data_format']
            slide_metrics['data_source'] = format_name
            
            all_metrics.append(slide_metrics)
            visualized_count += 1
    
    # Save summary
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        
        # Save metrics
        metrics_df.to_csv(os.path.join(interp_dir, f'{dataset_name}_patch_visualization_metrics.csv'), index=False)
        
        # Create summary
        summary = {
            'dataset': dataset_name,
            'model_architecture': mil_architecture,
            'n_visualized': visualized_count,
            'n_h5_visualized': h5_visualized,
            'n_tfrecord_visualized': tfrecord_visualized,
            'patches_per_category': n_patches_per_category,
            'total_patches_shown': visualized_count * n_patches_per_category * 3,
            'format_distribution': {
                'BCR_NET (H5)': h5_visualized,
                'UCMC (TFRecord)': tfrecord_visualized
            },
            'categories_visualized': {
                'total': visualized_count,
                'by_format': {
                    'h5': len(h5_samples),
                    'tfrecord': len(tfrecord_samples)
                }
            }
        }
        
        # Add format information
        if tfrecord_dir and h5_dir:
            summary['patch_formats'] = 'TFRecord and H5'
        elif tfrecord_dir:
            summary['patch_formats'] = 'TFRecord only'
        elif h5_dir:
            summary['patch_formats'] = 'H5 only'
        
        with open(os.path.join(interp_dir, f'{dataset_name}_patch_viz_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        print(f"\n{'='*60}")
        print(f"Patch Visualization Complete")
        print(f"{'='*60}")
        print(f"  Total cases visualized: {visualized_count}")
        print(f"    - BCR_NET (H5): {h5_visualized}")
        print(f"    - UCMC (TFRecord): {tfrecord_visualized}")
        print(f"  Total patches shown: {visualized_count * n_patches_per_category * 3}")
        print(f"  PDFs saved in: {os.path.join(interp_dir, 'patch_visualizations')}")
        
        return metrics_df
    
    return None

def test_individual_folds(model_folder, test_csv, output_dir=None, run_interpretability=False, 
                         analyze_all_folds=False, n_visualize=15,
                         tfrecord_dir=None, h5_dir=None, visualize_patches=True,
                         n_patches_per_category=5, test_formats=False, save_predictions=True):
    """Test each fold with patch visualization support and prediction saving"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test format loading if requested
    if test_formats and (tfrecord_dir or h5_dir):
        test_format_loading(test_csv, tfrecord_dir, h5_dir)
    
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
    
    # Check data format distribution
    test_df = pd.read_csv(test_csv)
    h5_count = sum(1 for _, row in test_df.iterrows() if str(row['slide_id']).strip().isdigit())
    tfrecord_count = sum(1 for _, row in test_df.iterrows() if 'UCH_BRCA_RS' in str(row['slide_id']))
    print(f"Data format distribution:")
    print(f"  H5 (BCR_NET): {h5_count} samples")
    print(f"  TFRecord (UCMC): {tfrecord_count} samples")
    
    # Test each fold
    fold_results = []
    all_fold_probs = []
    best_fold_num = None
    best_fold_auroc = -1
    saved_labels = None  # To store labels from first fold
    
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
        
        # Save labels from first fold (they're the same for all folds)
        if fold == 1:
            saved_labels = labels
        
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
                tfrecord_dir=tfrecord_dir,
                h5_dir=h5_dir,
                visualize_patches=visualize_patches,
                n_patches_per_category=n_patches_per_category
            )
    
    # Save detailed predictions if requested
    if save_predictions and len(all_fold_probs) > 0 and saved_labels is not None:
        predictions_df, pred_summary = save_detailed_predictions(
            all_fold_probs, 
            saved_labels,
            test_df, 
            output_dir, 
            fold_results
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
        'data_format_distribution': {
            'h5_bcr_net': h5_count,
            'tfrecord_ucmc': tfrecord_count
        },
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
        'patch_directories': {
            'tfrecord_dir': tfrecord_dir,
            'h5_dir': h5_dir
        },
        'predictions_saved': save_predictions
    }
    
    # Add prediction summary if available
    if save_predictions and 'pred_summary' in locals():
        results_summary['prediction_summary'] = pred_summary
    
    # Save files
    with open(os.path.join(output_dir, 'individual_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, cls=NumpyEncoder)
    
    fold_df.to_csv(os.path.join(output_dir, 'fold_results.csv'), index=False)
    
    # Plot all ROC curves on one plot if we have results
    if len(all_fold_probs) > 0 and saved_labels is not None:
        plot_all_roc_curves(all_fold_probs, saved_labels, fold_df, output_dir)
        plot_fold_comparison(fold_df, output_dir)
    
    print(f"\n{'='*80}")
    print(f"All results saved to: {output_dir}")
    if save_predictions:
        print(f"Detailed predictions saved with slide names and probabilities")
    if run_interpretability:
        print(f"Interpretability analysis completed with balanced format selection")
        if h5_dir and tfrecord_dir:
            print(f"  Patch visualizations include both BCR_NET (H5) and UCMC (TFRecord) samples")
    
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
    
    # Patch visualization arguments
    parser.add_argument('--tfrecord-dir', type=str, default=None,
                      help='Directory containing TFRecord files for patch visualization (UCMC)')
    parser.add_argument('--h5-dir', type=str, default=None,
                      help='Directory containing H5 files for patch visualization (BCR_NET)')
    parser.add_argument('--visualize-patches', action='store_true',
                      help='Visualize actual patches from high/medium/low attention regions')
    parser.add_argument('--n-patches-per-category', type=int, default=5,
                      help='Number of patches to visualize per attention category (default: 5)')
    parser.add_argument('--test-formats', action='store_true',
                      help='Test loading from both H5 and TFRecord formats before main analysis')
    
    # Prediction saving argument
    parser.add_argument('--save-predictions', action='store_true', default=True,
                      help='Save detailed predictions for all samples (default: True)')
    
    args = parser.parse_args()
    
    test_individual_folds(
        args.model_folder, 
        args.test_csv, 
        args.output_dir,
        args.run_interpretability,
        args.analyze_all_folds,
        args.n_visualize,
        tfrecord_dir=args.tfrecord_dir,
        h5_dir=args.h5_dir,
        visualize_patches=args.visualize_patches,
        n_patches_per_category=args.n_patches_per_category,
        test_formats=args.test_formats,
        save_predictions=args.save_predictions
    )
################


# Test with interpretability on best fold only
# python -B test_cls/inference_cls_updated.py --model-folder results_classification_cv_updated/h-optimus_clam_20250912_145351 --test-csv data/manifests/test_features_h-optimus.csv --run-interpretability

### new

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/h-optimus_acmil_20250914_204945 --test-csv data/manifests/test_features_h-optimus.csv --output-dir detailed_predictions/internal/h-optimus_acmil  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/h-optimus_attention_20250914_231455 --test-csv data/manifests/test_features_h-optimus.csv --output-dir detailed_predictions/internal/h-optimus_attention  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/h-optimus_clam_20250914_231419 --test-csv data/manifests/test_features_h-optimus.csv --output-dir detailed_predictions/internal/h-optimus_clam  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/h-optimus_mean_20250914_231513 --test-csv data/manifests/test_features_h-optimus.csv --output-dir detailed_predictions/internal/h-optimus_mean  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/virchow2_acmil_20250915_000404 --test-csv data/manifests/test_features_virchow2.csv --output-dir detailed_predictions/internal/virchow2_acmil  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/virchow2_attention_20250915_003518 --test-csv data/manifests/test_features_virchow2.csv --output-dir detailed_predictions/internal/virchow2_attention  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/virchow2_clam_20250915_002422 --test-csv data/manifests/test_features_virchow2.csv --output-dir detailed_predictions/internal/virchow2_clam  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/virchow2_mean_20250915_004613 --test-csv data/manifests/test_features_virchow2.csv --output-dir detailed_predictions/internal/virchow2_mean  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/uni2-h_acmil_20250915_000434 --test-csv data/manifests/test_features_uni2-h.csv --output-dir detailed_predictions/internal/uni2h_acmil  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/uni2-h_attention_20250915_003535 --test-csv data/manifests/test_features_uni2-h.csv --output-dir detailed_predictions/internal/uni2h_attention  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/uni2-h_clam_20250915_002443 --test-csv data/manifests/test_features_uni2-h.csv --output-dir detailed_predictions/internal/uni2h_clam  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/uni2-h_mean_20250915_004627 --test-csv data/manifests/test_features_uni2-h.csv --output-dir detailed_predictions/internal/uni2h_mean  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/conch_acmil_20250915_000515 --test-csv data/manifests/test_features_conch.csv --output-dir detailed_predictions/internal/conch_acmil  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/conch_attention_20250915_005221 --test-csv data/manifests/test_features_conch.csv --output-dir detailed_predictions/internal/conch_attention  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/conch_clam_20250915_004600 --test-csv data/manifests/test_features_conch.csv --output-dir detailed_predictions/internal/conch_clam  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/conch_mean_20250915_005832 --test-csv data/manifests/test_features_conch.csv --output-dir detailed_predictions/internal/conch_mean  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/resnet50_acmil_20250915_000619 --test-csv data/manifests/test_features_resnet50.csv --output-dir detailed_predictions/internal/resnet50_acmil  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/resnet50_attention_20250915_003616 --test-csv data/manifests/test_features_resnet50.csv --output-dir detailed_predictions/internal/resnet50_attention  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/resnet50_clam_20250915_002521 --test-csv data/manifests/test_features_resnet50.csv --output-dir detailed_predictions/internal/resnet50_clam  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/resnet50_mean_20250915_004639 --test-csv data/manifests/test_features_resnet50.csv --output-dir detailed_predictions/internal/resnet50_mean  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/resnet18_acmil_20250915_000643 --test-csv data/manifests/test_features_resnet18.csv --output-dir detailed_predictions/internal/resnet18_acmil  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/resnet18_attention_20250915_003630 --test-csv data/manifests/test_features_resnet18.csv --output-dir detailed_predictions/internal/resnet18_attention  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/resnet18_clam_20250915_002540 --test-csv data/manifests/test_features_resnet18.csv --output-dir detailed_predictions/internal/resnet18_clam  --save-predictions

# python -B test_cls/inference_cls_2.py --model-folder results_classification_cv_updated/resnet18_mean_20250915_004647 --test-csv data/manifests/test_features_resnet18.csv --output-dir detailed_predictions/internal/resnet18_mean  --save-predictions
