import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def load_all_roc_data(base_dir="roc_data/external"):
    """Load all ROC data from the saved JSON files"""
    
    roc_data_list = []
    base_path = Path(base_dir)
    
    for model_dir in base_path.iterdir():
        if model_dir.is_dir():
            # Look for ECU-specific ROC data files
            json_files = list(model_dir.glob("*_ecu_mean_roc_data.json"))
            if json_files:
                with open(json_files[0], 'r') as f:
                    data = json.load(f)
                    roc_data_list.append(data)
    
    print(f"Total models loaded: {len(roc_data_list)}")
    return roc_data_list

def plot_smoothed_roc_curves(roc_data_list, save_path="ecu_roc_comparison_smooth.png", 
                            n_points=1000, show_all=False):
    """
    Plot smoothed ROC curves using interpolation for ECU external validation
    
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
    
    # Track best model
    best_model = None
    best_auc = 0
    
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
        
        # Track best model
        if auc_val > best_auc:
            best_auc = auc_val
            best_model = f"{feature}_{mil}"
        
        # Create interpolation function
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
            
            # Apply additional smoothing with Savitzky-Golay filter
            if len(tpr_smooth) > 51:
                tpr_smooth = savgol_filter(tpr_smooth, 51, 3)
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
        title = 'ECU External Validation - ROC Curves (All Models)'
    else:
        title = 'ECU External Validation - ROC Curves (Best MIL per Feature Extractor)'
    
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

def create_performance_summary_table(roc_data_list, save_path="ecu_model_performance_summary.csv"):
    """Create a summary table of all model performances on ECU"""
    
    summary_data = []
    
    for data in roc_data_list:
        summary_data.append({
            'Model': data['model_config']['identifier'],
            'Feature_Extractor': data['model_config']['feature_extractor'],
            'MIL_Architecture': data['model_config']['mil_architecture'],
            'AUC_Mean': data['mean_roc']['auc'],
            'AUC_Std': data['mean_roc']['auc_std'],
            'Sensitivity_at_90_Spec': data['operating_points']['sensitivity_at_90_spec'],
            'Specificity_at_90_Sens': data['operating_points']['specificity_at_90_sens'],
            'Youden_Index': data['operating_points']['youden_index'],
            'N_Folds': data['fold_statistics']['n_folds']
        })
    
    # Create DataFrame and sort by AUC
    df = pd.DataFrame(summary_data)
    df = df.sort_values('AUC_Mean', ascending=False)
    
    # Format numbers
    df['AUC_Mean'] = df['AUC_Mean'].round(4)
    df['AUC_Std'] = df['AUC_Std'].round(4)
    df['Sensitivity_at_90_Spec'] = df['Sensitivity_at_90_Spec'].round(4)
    df['Specificity_at_90_Sens'] = df['Specificity_at_90_Sens'].round(4)
    df['Youden_Index'] = df['Youden_Index'].round(4)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"\nPerformance summary saved to: {save_path}")
    
    # Print top performers
    print("\n" + "="*60)
    print("ECU EXTERNAL VALIDATION - TOP 5 MODELS BY AUC:")
    print("="*60)
    print(df.head()[['Model', 'AUC_Mean', 'AUC_Std']].to_string(index=False))
    
    # Best by feature extractor
    print("\n" + "="*60)
    print("ECU - BEST MODEL PER FEATURE EXTRACTOR:")
    print("="*60)
    best_per_feature = df.loc[df.groupby('Feature_Extractor')['AUC_Mean'].idxmax()]
    print(best_per_feature[['Feature_Extractor', 'MIL_Architecture', 'AUC_Mean']].to_string(index=False))
    
    # Best by MIL architecture
    print("\n" + "="*60)
    print("ECU - BEST MODEL PER MIL ARCHITECTURE:")
    print("="*60)
    best_per_mil = df.loc[df.groupby('MIL_Architecture')['AUC_Mean'].idxmax()]
    print(best_per_mil[['MIL_Architecture', 'Feature_Extractor', 'AUC_Mean']].to_string(index=False))
    
    return df

def main():
    """Main function to create smoothed ROC plots for ECU external validation"""
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Load all ECU ROC data
    print("Loading ECU ROC data from roc_data/external/...")
    roc_data_list = load_all_roc_data("roc_data/external")
    
    if not roc_data_list:
        print("No ECU ROC data found! Please check the directory path.")
        return
    
    # Create smoothed plot with best models only
    print("\nCreating ECU smoothed ROC plot (best models only)...")
    plot_smoothed_roc_curves(roc_data_list, 
                           save_path="ecu_roc_smooth_best_only.png",
                           n_points=1000,
                           show_all=False)
    
    # Create smoothed plot with all models
    print("\nCreating ECU smoothed ROC plot (all models)...")
    plot_smoothed_roc_curves(roc_data_list,
                           save_path="ecu_roc_smooth_all_models.png", 
                           n_points=1000,
                           show_all=True)
    
    # Create performance summary table
    print("\nCreating ECU performance summary table...")
    summary_df = create_performance_summary_table(roc_data_list,
                                                  save_path="ecu_model_performance_summary.csv")
    
    print("\n" + "="*60)
    print("ECU external validation plots created successfully!")
    print("="*60)
    print("Files created:")
    print("  - ecu_roc_smooth_best_only.png/pdf")
    print("  - ecu_roc_smooth_all_models.png/pdf")
    print("  - ecu_model_performance_summary.csv")

if __name__ == "__main__":
    main()