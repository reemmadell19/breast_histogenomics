import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import os

def plot_pr_curves_with_folds(model_folder, test_csv_internal, test_csv_external, output_dir="."):
    """
    Plot PR curves showing all folds, mean, and std for both internal and external validation
    Similar to the ROC curve visualization
    """
    
    # Import necessary functions from your existing code
    import torch
    import pandas as pd
    from torch.utils.data import DataLoader
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from datasets.classification_mil_dataset import ClassificationMILDataset
    from utils.mil_utils import mil_collate_fn
    from models.classification_model_updated import AttentionMILClassifier
    
    def load_config_from_folder(model_folder):
        import json
        config_path = os.path.join(model_folder, "config_used.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def create_model(config, device):
        return AttentionMILClassifier(
            input_dim=1536,  # h-optimus
            hidden_dim=config.get('architecture_specific_params', {}).get('hidden_dim', 256),
            attention_hidden_dim=config.get('architecture_specific_params', {}).get('attention_hidden_dim', 128),
            n_classes=2,
            dropout=config.get('architecture_specific_params', {}).get('dropout', 0.25)
        ).to(device)
    
    def evaluate_model(model, dataloader, device):
        model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in dataloader:
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
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config_from_folder(model_folder)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Process internal validation
    print("Processing internal validation...")
    test_dataset_int = ClassificationMILDataset(
        test_csv_internal,
        label_column='RSHigh' if 'RSHigh' in pd.read_csv(test_csv_internal).columns else 'RS',
        threshold=25.0
    )
    test_loader_int = DataLoader(test_dataset_int, batch_size=1, shuffle=False, 
                                 collate_fn=mil_collate_fn, num_workers=0)
    
    # Process external validation (ECU)
    print("Processing external validation...")
    test_dataset_ext = ClassificationMILDataset(
        test_csv_external,
        label_column='RSHigh' if 'RSHigh' in pd.read_csv(test_csv_external).columns else 'RS',
        threshold=25.0
    )
    test_loader_ext = DataLoader(test_dataset_ext, batch_size=1, shuffle=False,
                                collate_fn=mil_collate_fn, num_workers=0)
    
    # Colors for each fold
    fold_colors = ['#FFB6C1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
    
    # Process each fold for both datasets
    for dataset_type, test_loader, ax, title in [
        ('internal', test_loader_int, ax1, 'Internal Validation - PR Curves'),
        ('external', test_loader_ext, ax2, 'ECU External Validation - PR Curves')
    ]:
        
        all_fold_precisions = []
        all_fold_recalls = []
        all_fold_aps = []
        labels_saved = None
        
        # Evaluate each fold
        for fold in range(1, 6):
            checkpoint_path = os.path.join(model_folder, f'best_model_fold_{fold}.pt')
            if not os.path.exists(checkpoint_path):
                continue
            
            # Load model
            model = create_model(config, device)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Evaluate
            probs, labels = evaluate_model(model, test_loader, device)
            if fold == 1:
                labels_saved = labels
            
            # Calculate PR curve for this fold
            precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
            ap = average_precision_score(labels, probs[:, 1])
            
            # Plot individual fold with transparency
            ax.plot(recall, precision, color=fold_colors[fold-1], linewidth=1.5, alpha=0.4,
                   label=f'Fold {fold} (AP = {ap:.3f})')
            
            all_fold_precisions.append(precision)
            all_fold_recalls.append(recall)
            all_fold_aps.append(ap)
        
        # Calculate mean PR curve
        # Interpolate all curves to common recall points
        mean_recall = np.linspace(0, 1, 100)
        precision_interp_list = []
        
        for precision, recall in zip(all_fold_precisions, all_fold_recalls):
            # Reverse arrays to make recall increasing for interpolation
            recall_rev = np.flip(recall)
            precision_rev = np.flip(precision)
            
            # Remove duplicates
            unique_indices = np.unique(recall_rev, return_index=True)[1]
            recall_unique = recall_rev[unique_indices]
            precision_unique = precision_rev[unique_indices]
            
            # Interpolate
            from scipy.interpolate import interp1d
            if len(recall_unique) > 1:
                f = interp1d(recall_unique, precision_unique, kind='linear',
                           bounds_error=False, fill_value=(precision_unique[-1], precision_unique[0]))
                precision_interp = f(mean_recall)
                precision_interp_list.append(precision_interp)
        
        # Calculate mean and std
        mean_precision = np.mean(precision_interp_list, axis=0)
        std_precision = np.std(precision_interp_list, axis=0)
        mean_ap = np.mean(all_fold_aps)
        std_ap = np.std(all_fold_aps)
        
        # Plot mean curve
        ax.plot(mean_recall, mean_precision, 'b-', linewidth=2.5,
               label=f'Mean PR (AP = {mean_ap:.3f} ± {std_ap:.3f})', alpha=0.8)
        
        # Plot ± 1 std deviation
        precision_upper = np.minimum(mean_precision + std_precision, 1)
        precision_lower = np.maximum(mean_precision - std_precision, 0)
        ax.fill_between(mean_recall, precision_lower, precision_upper,
                       color='grey', alpha=0.2, label='± 1 std. dev.')
        
        # Plot baseline
        n_positive = np.sum(labels_saved)
        n_total = len(labels_saved)
        baseline = n_positive / n_total
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                  label=f'Baseline = {baseline:.3f}', alpha=0.5)
        
        # Formatting
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        print(f"{dataset_type.capitalize()} - Mean AP: {mean_ap:.3f} ± {std_ap:.3f}")
    
    plt.suptitle('H-Optimus Attention MIL - Precision-Recall Curves', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'h_optimus_attention_pr_curves_all_folds.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    
    print(f"\nSaved to: {output_path}")
    plt.show()

# Run the function
plot_pr_curves_with_folds(
    model_folder="results_classification_cv_updated/h-optimus_attention_20250914_231455",
    test_csv_internal="data/manifests/test_features_h-optimus.csv",
    test_csv_external="data/manifests/ecu_features_h-optimus.csv",
    output_dir="."
)