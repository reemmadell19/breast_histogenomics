# run_test_inference.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report, r2_score, mean_absolute_error
from scipy.stats import spearmanr

# Project imports
from datasets.regression_mil_dataset import RegressionMILDataset
from utils.mil_utils import mil_collate_fn
from utils.regression_evaluator import RegressionEvaluator

def plot_test_results(targets, predictions, output_dir, model_name):
    """Generate comprehensive test result visualizations"""
    
    targets = np.array(targets)
    predictions = np.array(predictions)
    threshold = 25.0
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. ROC Curve
    ax1 = plt.subplot(3, 3, 1)
    binary_targets = (targets >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(binary_targets, predictions)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    ax2 = plt.subplot(3, 3, 2)
    precision, recall, _ = precision_recall_curve(binary_targets, predictions)
    avg_precision = average_precision_score(binary_targets, predictions)
    
    ax2.plot(recall, precision, color='darkgreen', lw=2, 
             label=f'PR curve (AP = {avg_precision:.3f})')
    ax2.axhline(y=np.mean(binary_targets), color='navy', linestyle='--', 
                label=f'Baseline (AP = {np.mean(binary_targets):.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    ax3 = plt.subplot(3, 3, 3)
    binary_preds = (predictions >= threshold).astype(int)
    cm = confusion_matrix(binary_targets, binary_preds)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Low Risk\n(RS<25)', 'High Risk\n(RS≥25)'],
                yticklabels=['Low Risk\n(RS<25)', 'High Risk\n(RS≥25)'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title('Confusion Matrix')
    
    # 4. Predictions vs Targets Scatter
    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(targets, predictions, alpha=0.6, s=30)
    
    # Add perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Add threshold lines
    ax4.axvline(x=threshold, color='orange', linestyle='-', alpha=0.7, label='RS=25 Threshold')
    ax4.axhline(y=threshold, color='orange', linestyle='-', alpha=0.7)
    
    # Calculate and display metrics
    from sklearn.metrics import r2_score
    r2 = r2_score(targets, predictions)
    spearman_corr, _ = spearmanr(targets, predictions)
    
    ax4.text(0.05, 0.95, f'R² = {r2:.3f}\nSpearman ρ = {spearman_corr:.3f}', 
             transform=ax4.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    ax4.set_xlabel('True RS Score')
    ax4.set_ylabel('Predicted RS Score')
    ax4.set_title('Predictions vs Targets')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Residual Plot
    ax5 = plt.subplot(3, 3, 5)
    residuals = targets - predictions
    ax5.scatter(predictions, residuals, alpha=0.6, s=30)
    ax5.axhline(y=0, color='red', linestyle='--', lw=2)
    ax5.axvspan(threshold-10, threshold+10, alpha=0.2, color='yellow', 
                label='Decision Boundary Zone')
    
    # Add residual statistics
    ax5.text(0.05, 0.95, f'Mean: {np.mean(residuals):.2f}\nStd: {np.std(residuals):.2f}', 
             transform=ax5.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    ax5.set_xlabel('Predicted RS Score')
    ax5.set_ylabel('Residuals (True - Predicted)')
    ax5.set_title('Residual Plot')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Distribution Comparison
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(targets, bins=20, alpha=0.5, label='True RS', color='blue', density=True)
    ax6.hist(predictions, bins=20, alpha=0.5, label='Predicted RS', color='red', density=True)
    ax6.axvline(x=threshold, color='orange', linestyle='--', lw=2, label='RS=25 Threshold')
    ax6.set_xlabel('RS Score')
    ax6.set_ylabel('Density')
    ax6.set_title('Distribution Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Calibration Plot
    ax7 = plt.subplot(3, 3, 7)
    n_bins = 10
    bin_edges = np.linspace(0, 100, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_means_pred = []
    bin_means_true = []
    
    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i+1])
        if np.sum(mask) > 0:
            bin_means_pred.append(predictions[mask].mean())
            bin_means_true.append(targets[mask].mean())
        else:
            bin_means_pred.append(bin_centers[i])
            bin_means_true.append(bin_centers[i])
    
    ax7.plot(bin_means_pred, bin_means_true, 'o-', label='Calibration')
    ax7.plot([0, 100], [0, 100], 'r--', label='Perfect Calibration')
    ax7.set_xlabel('Mean Predicted RS')
    ax7.set_ylabel('Mean True RS')
    ax7.set_title('Calibration Plot')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Error by RS Range
    ax8 = plt.subplot(3, 3, 8)
    rs_ranges = [(0, 15), (15, 25), (25, 35), (35, 50), (50, 100)]
    range_labels = [f'{r[0]}-{r[1]}' for r in rs_ranges]
    range_maes = []
    range_counts = []
    
    for r_min, r_max in rs_ranges:
        mask = (targets >= r_min) & (targets < r_max)
        if np.sum(mask) > 0:
            mae = np.mean(np.abs(targets[mask] - predictions[mask]))
            range_maes.append(mae)
            range_counts.append(np.sum(mask))
        else:
            range_maes.append(0)
            range_counts.append(0)
    
    bars = ax8.bar(range_labels, range_maes, 
                   color=['lightblue' if not (15 <= r[0] <= 35) else 'yellow' 
                          for r in rs_ranges])
    
    # Add sample counts
    for bar, count in zip(bars, range_counts):
        if count > 0:
            ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'n={count}', ha='center', va='bottom', fontsize=9)
    
    ax8.set_xlabel('True RS Range')
    ax8.set_ylabel('Mean Absolute Error')
    ax8.set_title('Error by RS Range')
    ax8.grid(True, alpha=0.3)
    
    # 9. Box plots by risk category
    ax9 = plt.subplot(3, 3, 9)
    low_risk_mask = targets < threshold
    high_risk_mask = targets >= threshold
    
    data_to_plot = [predictions[low_risk_mask], predictions[high_risk_mask]]
    bp = ax9.boxplot(data_to_plot, labels=['True Low Risk', 'True High Risk'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax9.axhline(y=threshold, color='orange', linestyle='--', 
                label='Decision Threshold')
    ax9.set_ylabel('Predicted RS Score')
    ax9.set_title('Predictions by True Risk Category')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Overall title
    plt.suptitle(f'{model_name.upper()} - Test Set Performance Analysis', 
                 fontsize=16, weight='bold')
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, 'test_performance_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Test performance plots saved to: {plot_path}")
    
    # Generate additional individual plots for paper figures
    generate_paper_figures(targets, predictions, output_dir, model_name)

def generate_paper_figures(targets, predictions, output_dir, model_name):
    """Generate individual high-quality figures for paper/thesis"""
    
    targets = np.array(targets)
    predictions = np.array(predictions)
    threshold = 25.0
    binary_targets = (targets >= threshold).astype(int)
    
    # Set style for publication
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. ROC Curve (standalone)
    fig, ax = plt.subplots(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(binary_targets, predictions)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='#FF6B6B', lw=3, label=f'{model_name} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#4ECDC4', lw=2, linestyle='--', label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.2, color='#FF6B6B')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14, weight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve_paper.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Precision-Recall Curve (standalone)
    fig, ax = plt.subplots(figsize=(6, 6))
    precision, recall, _ = precision_recall_curve(binary_targets, predictions)
    avg_precision = average_precision_score(binary_targets, predictions)
    baseline_ap = np.mean(binary_targets)
    
    ax.plot(recall, precision, color='#95E77E', lw=3, 
            label=f'{model_name} (AP = {avg_precision:.3f})')
    ax.axhline(y=baseline_ap, color='#FFA07A', linestyle='--', lw=2,
               label=f'Baseline (AP = {baseline_ap:.3f})')
    ax.fill_between(recall, precision, alpha=0.2, color='#95E77E')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, weight='bold')
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curve_paper.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Regression Performance (standalone)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create hexbin plot for better visualization of density
    hb = ax.hexbin(targets, predictions, gridsize=20, cmap='YlOrRd', mincnt=1)
    
    # Add perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'b--', lw=2, 
            label='Perfect Prediction', alpha=0.8)
    
    # Add threshold lines
    ax.axvline(x=threshold, color='green', linestyle='-', alpha=0.7, 
               label='Clinical Threshold (RS=25)', lw=2)
    ax.axhline(y=threshold, color='green', linestyle='-', alpha=0.7, lw=2)
    
    # Add shaded regions for risk categories
    ax.axvspan(min_val, threshold, alpha=0.1, color='blue', label='Low Risk Region')
    ax.axhspan(min_val, threshold, alpha=0.1, color='blue')
    
    # Calculate metrics
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(np.mean((targets - predictions)**2))
    
    # Add text box with metrics
    textstr = f'R² = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel('True RS Score', fontsize=12)
    ax.set_ylabel('Predicted RS Score', fontsize=12)
    ax.set_title(f'Regression Performance - {model_name}', fontsize=14, weight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('Number of Samples', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regression_performance_paper.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Paper-quality figures saved successfully!")

def load_trained_model(checkpoint_path, input_dim, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    # Set weights_only=False to handle numpy scalars in checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    
    # Determine model type and create model
    if 'n_branches' in config:
        # ACMIL models
        if 'gate' in config:
            # ACMIL_CLAM_Hybrid model
            from models.regression_model import ACMIL_CLAM_Hybrid
            model = ACMIL_CLAM_Hybrid(
                input_dim=input_dim,
                n_branches=config['n_branches'],
                hidden_dim=config['hidden_dim'],
                attention_hidden_dim=config['attention_hidden_dim'],
                mask_ratio=config.get('mask_ratio', 0.0),
                n_masked_patch=config.get('n_masked_patch', 10),
                dropout=config.get('dropout', 0.25),
                gate=config.get('gate', True)
            )
        else:
            # Pure ACMIL model
            from models.regression_model import ACMIL
            model = ACMIL(
                input_dim=input_dim,
                hidden_dim=config['hidden_dim'],
                n_branches=config['n_branches'],
                n_masked_patch=config.get('n_masked_patch', 10),
                mask_ratio=config.get('mask_ratio', 0.6),
                dropout=config.get('dropout', 0.25)
            )
    else:
        # Standard CLAM model
        from models.regression_model import CLAM
        model = CLAM(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            attention_hidden_dim=config['attention_hidden_dim'],
            dropout=config.get('dropout', 0.25),
            gate=config.get('gate', False)
        )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config

def run_test_inference(model_name, checkpoint_path, test_csv_path, input_dim):
    """Run inference on test set"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model with input_dim
    model, config = load_trained_model(checkpoint_path, input_dim, device)
    
    print(f"{'='*60}")
    print(f"TEST SET INFERENCE - {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Test data: {test_csv_path}")
    print(f"Device: {device}")
    
    # Load test dataset
    test_dataset = RegressionMILDataset(test_csv_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                           collate_fn=mil_collate_fn, num_workers=0)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Run inference
    evaluator = RegressionEvaluator()
    all_predictions = []
    all_targets = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for features, rs_target in tqdm(test_loader, desc="Testing"):
            # Process data
            if isinstance(features, list):
                features = features[0]
            features = features.to(device)
            
            if isinstance(rs_target, list):
                rs_target = rs_target[0]
            if not isinstance(rs_target, torch.Tensor):
                rs_target = torch.tensor([rs_target], dtype=torch.float32)
            
            # Get prediction
            prediction = model(features)
            if prediction.dim() == 0:
                prediction = prediction.unsqueeze(0)
            
            # Store results
            pred_value = prediction.cpu().numpy()
            target_value = rs_target.numpy() if isinstance(rs_target, torch.Tensor) else rs_target
            
            all_predictions.append(float(pred_value))
            all_targets.append(float(target_value))
            
            evaluator.update(
                targets=[target_value],
                preds=[pred_value]
            )
    
    # Calculate metrics
    test_metrics = evaluator.compute_all_metrics(verbose=True)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'true_rs': all_targets,
        'predicted_rs': all_predictions,
        'true_risk': ['high' if rs >= 25 else 'low' for rs in all_targets],
        'predicted_risk': ['high' if rs >= 25 else 'low' for rs in all_predictions]
    })
    
    # Save results
    output_dir = os.path.dirname(checkpoint_path)
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    
    results_df.to_csv(os.path.join(test_dir, 'test_predictions.csv'), index=False)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        """Convert numpy types to native Python types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(val) for key, val in obj.items()}
        return obj
    
    test_metrics_native = convert_to_native(test_metrics)
    
    with open(os.path.join(test_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics_native, f, indent=2)
    
    # Generate comprehensive test plots
    print("\nGenerating test result visualizations...")
    plot_test_results(all_targets, all_predictions, test_dir, model_name)
    
    print(f"\n{'='*60}")
    print(f"TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"AUROC: {test_metrics.get('auroc', 0):.4f}")
    print(f"AUC-PR: {test_metrics.get('auc_pr', 0):.4f}")
    print(f"R²: {test_metrics.get('r2', 0):.4f}")
    print(f"RMSE: {test_metrics.get('rmse', 0):.4f}")
    print(f"MAE: {test_metrics.get('mae', 0):.4f}")
    print(f"F1 Score: {test_metrics.get('f1_score', 0):.4f}")
    print(f"C-index: {test_metrics.get('c_index', 0):.4f}")
    print(f"Binary Accuracy: {test_metrics.get('binary_accuracy', 0):.4f}")
    
    print(f"\nAll results saved to: {test_dir}")
    print(f"Files generated:")
    print(f"  - test_predictions.csv")
    print(f"  - test_metrics.json")
    print(f"  - test_performance_analysis.png")
    print(f"  - roc_curve_paper.png")
    print(f"  - pr_curve_paper.png")
    print(f"  - regression_performance_paper.png")
    
    return test_metrics, results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run test set inference')
    parser.add_argument('--model', type=str, required=True,
                      choices=['resnet18', 'uni2-h', 'virchow2', 'h-optimus'],
                      help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--test_csv', type=str, required=True,
                      help='Path to test set CSV file')
    
    args = parser.parse_args()
    
    # Get input dimensions
    input_dims = {
        'resnet18': 512,
        'uni2-h': 1536,
        'virchow2': 1280,
        'h-optimus': 1536
    }
    
    run_test_inference(args.model, args.checkpoint, args.test_csv, input_dims[args.model])