import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def load_all_roc_data(base_dir="roc_data/internal"):
    """Load all ROC data from the saved JSON files"""
    
    roc_data_list = []
    base_path = Path(base_dir)
    
    for model_dir in base_path.iterdir():
        if model_dir.is_dir():
            json_files = list(model_dir.glob("*_mean_roc_data.json"))
            if json_files:
                with open(json_files[0], 'r') as f:
                    data = json.load(f)
                    roc_data_list.append(data)
    
    print(f"Total models loaded: {len(roc_data_list)}")
    return roc_data_list

def plot_smoothed_roc_curves(roc_data_list, save_path="roc_comparison_smooth.png", 
                            n_points=1000, show_all=False):
    """
    Plot smoothed ROC curves using interpolation
    
    Args:
        roc_data_list: List of ROC data
        save_path: Path to save the plot
        n_points: Number of points for interpolation (higher = smoother)
        show_all: If False, shows only best MIL per feature extractor
    """
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color scheme for feature extractors
    feature_colors = {
        'resnet18': '#8B4513',
        'resnet50': '#D2691E', 
        'conch': '#4169E1',
        'uni2-h': '#32CD32',
        'virchow2': '#FF6347',
        'h-optimus': '#9370DB'
    }
    
    # If not showing all, select best per feature extractor
    if not show_all:
        # Group by feature extractor
        feature_groups = {}
        for data in roc_data_list:
            feature = data['model_config']['feature_extractor']
            if feature not in feature_groups:
                feature_groups[feature] = []
            feature_groups[feature].append(data)
        
        # Get best model per feature
        selected_models = []
        for feature, models in feature_groups.items():
            models.sort(key=lambda x: x['mean_roc']['auc'], reverse=True)
            selected_models.append(models[0])
        
        roc_data_list = selected_models
    
    # Sort by AUC for legend ordering
    roc_data_list.sort(key=lambda x: x['mean_roc']['auc'], reverse=True)
    
    # Common FPR grid for interpolation
    fpr_common = np.linspace(0, 1, n_points)
    
    for data in roc_data_list:
        feature = data['model_config']['feature_extractor']
        mil = data['model_config']['mil_architecture']
        
        # Use interpolated data if available, otherwise interpolate from raw data
        if 'interpolated_roc' in data:
            fpr = np.array(data['interpolated_roc']['fpr_grid'])
            tpr = np.array(data['interpolated_roc']['tpr_mean'])
        else:
            fpr = np.array(data['mean_roc']['fpr'])
            tpr = np.array(data['mean_roc']['tpr'])
        
        auc_val = data['mean_roc']['auc']
        auc_std = data['mean_roc']['auc_std']
        
        # Create interpolation function
        # Use cubic interpolation for smoothness, but linear near endpoints
        if len(fpr) > 3:
            # Remove duplicate points
            unique_indices = np.unique(fpr, return_index=True)[1]
            fpr_unique = fpr[unique_indices]
            tpr_unique = tpr[unique_indices]
            
            # Interpolate
            f = interp1d(fpr_unique, tpr_unique, kind='cubic', 
                        bounds_error=False, fill_value=(0, 1))
            tpr_smooth = f(fpr_common)
            
            # Ensure endpoints are correct
            tpr_smooth[0] = 0
            tpr_smooth[-1] = 1
            
            # Optional: Apply additional smoothing with Savitzky-Golay filter
            # This removes any remaining jaggedness
            if len(tpr_smooth) > 51:
                tpr_smooth = savgol_filter(tpr_smooth, 51, 3)
                # Ensure values stay in [0, 1]
                tpr_smooth = np.clip(tpr_smooth, 0, 1)
        else:
            # If too few points, use linear interpolation
            f = interp1d(fpr, tpr, kind='linear', 
                        bounds_error=False, fill_value=(0, 1))
            tpr_smooth = f(fpr_common)
        
        # Get color
        color = feature_colors.get(feature, '#333333')
        
        # Create label
        label = f"{feature}_{mil} (AUC: {auc_val:.3f} ± {auc_std:.3f})"
        
        # Plot smoothed ROC curve
        ax.plot(fpr_common, tpr_smooth, 
                color=color, 
                linewidth=2.5,
                label=label,
                alpha=0.9)
        

    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1.5, label='Random Classifier')
    
    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    
    if show_all:
        title = 'Smoothed ROC Curves - All Models\nInternal Validation Set'
    else:
        title = 'Internal Valdiation ROC curves (best MIL per feature extarctor)'
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    
    # Legend
    ax.legend(loc='lower right', fontsize=10, 
             framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    print(f"PDF saved to: {save_path.replace('.png', '.pdf')}")
    
    plt.show()

def main():
    """Main function to create smoothed ROC plots"""
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Load all ROC data
    print("Loading ROC data from roc_data/internal/...")
    roc_data_list = load_all_roc_data("roc_data/internal")
    
    if not roc_data_list:
        print("No ROC data found! Please check the directory path.")
        return
    
    # Create smoothed plot with best models only
    print("\nCreating smoothed ROC plot (best models only)...")
    plot_smoothed_roc_curves(roc_data_list, 
                           save_path="roc_smooth_best_only.png",
                           n_points=1000,
                           show_all=False)
    
    # Create smoothed plot with all models
    print("\nCreating smoothed ROC plot (all models)...")
    plot_smoothed_roc_curves(roc_data_list,
                           save_path="roc_smooth_all_models.png", 
                           n_points=1000,
                           show_all=True)
    
    print("\n" + "="*60)
    print("Smoothed plots created successfully!")
    print("="*60)

if __name__ == "__main__":
    main()