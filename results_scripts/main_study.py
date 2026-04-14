import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os
from pathlib import Path
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import datetime

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"main_study_results/{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Results will be saved to: {output_dir}")
print("-" * 60)

# Define consistent color schemes
FEATURE_COLORS = {
    'resnet18': '#1f77b4',  # Blue
    'resnet50': '#2ca02c',  # Green  
    'conch': '#ff7f0e',     # Orange
    'uni2-h': '#d62728',       # Red
    'virchow2': '#9467bd',   # Purple
    'h-optimus': '#8c564b'    # Brown
}

MIL_MARKERS = {
    'mean': 'o',
    'attention': 's', 
    'clam': '^',
    'acmil': 'D'
}

def load_all_model_results():
    """Load internal and external results for all model combinations"""
    
    # Define paths
    internal_base_path = Path('test_results_cls_updated')
    external_base_path = Path('external_validation/ecu_updated')
    
    # Model combinations to look for
    feature_extractors = ['resnet18', 'resnet50', 'conch', 'uni2-h', 'virchow2', 'h-optimus']
    mil_architectures = ['mean', 'attention', 'clam', 'acmil']
    
    internal_results = {}
    external_results = {}
    
    print("Loading model results...")
    print("-" * 40)
    
    # Iterate through all possible combinations
    for feature in feature_extractors:
        for mil in mil_architectures:
            # Handle the naming convention (might be with underscore or hyphen)
            model_name_underscore = f"{feature.replace('-', '')}_{mil}"  # e.g., hoptimus_mean
            model_name_hyphen = f"{feature.replace('-', '_')}_{mil}"  # e.g., h_optimus_mean
            model_name_standard = f"{feature}_{mil}"  # e.g., h-optimus_mean
            
            # Try different naming patterns for internal results
            internal_loaded = False
            for name_variant in [model_name_underscore, model_name_hyphen, model_name_standard]:
                internal_folders = list(internal_base_path.glob(f"{name_variant}_*"))
                if internal_folders:
                    internal_json_path = internal_folders[0] / 'individual_results.json'
                    if internal_json_path.exists():
                        with open(internal_json_path, 'r') as f:
                            data = json.load(f)
                            # Extract the summary statistics mean values
                            internal_results[f"{feature}_{mil}"] = {
                                'auroc': data['summary_statistics']['auroc']['mean'],
                                'auc_pr': data['summary_statistics']['auc_pr']['mean'],
                                'accuracy': data['summary_statistics']['accuracy']['mean'],
                                'balanced_accuracy': data['summary_statistics']['balanced_accuracy']['mean'],
                                'f1_score': data['summary_statistics']['f1_score']['mean'],
                                'sensitivity': data['summary_statistics']['sensitivity']['mean'],
                                'specificity': data['summary_statistics']['specificity']['mean'],
                                'mcc': data['summary_statistics']['mcc']['mean'],
                                # Also store std for error bars if needed
                                'auroc_std': data['summary_statistics']['auroc']['std'],
                                'f1_std': data['summary_statistics']['f1_score']['std'],
                                # Store individual fold results for more detailed analysis
                                'fold_metrics': data['fold_metrics']
                            }
                            # Also get precision if available
                            if 'precision' in data['fold_metrics'][0]:
                                precisions = [fold['precision'] for fold in data['fold_metrics']]
                                internal_results[f"{feature}_{mil}"]['precision'] = np.mean(precisions)
                            else:
                                internal_results[f"{feature}_{mil}"]['precision'] = 0
                                
                        print(f"✓ Loaded internal results for {feature}_{mil}")
                        internal_loaded = True
                        break
            
            if not internal_loaded:
                print(f"✗ Internal results not found for {feature}_{mil}")
            
            # Try different naming patterns for external results
            external_loaded = False
            for name_variant in [model_name_underscore, model_name_hyphen, model_name_standard]:
                external_folders = list(external_base_path.glob(f"{name_variant}_*"))
                if external_folders:
                    external_json_path = external_folders[0] / 'ecu_external_validation_results.json'
                    if external_json_path.exists():
                        with open(external_json_path, 'r') as f:
                            data = json.load(f)
                            # External results have summary_statistics directly
                            external_results[f"{feature}_{mil}"] = {
                                'auroc': data['summary_statistics']['auroc']['mean'],
                                'auc_pr': data['summary_statistics']['auc_pr']['mean'],
                                'accuracy': data['summary_statistics']['accuracy']['mean'],
                                'balanced_accuracy': data['summary_statistics']['balanced_accuracy']['mean'],
                                'f1_score': data['summary_statistics']['f1_score']['mean'],
                                'sensitivity': data['summary_statistics']['sensitivity']['mean'],
                                'specificity': data['summary_statistics']['specificity']['mean'],
                                'mcc': data['summary_statistics']['mcc']['mean'],
                                # Store std
                                'auroc_std': data['summary_statistics']['auroc']['std'],
                                'f1_std': data['summary_statistics']['f1_score']['std'],
                            }
                            # Get precision from fold metrics
                            if 'fold_metrics' in data and data['fold_metrics']:
                                precisions = [fold['precision'] for fold in data['fold_metrics']]
                                external_results[f"{feature}_{mil}"]['precision'] = np.mean(precisions)
                            else:
                                external_results[f"{feature}_{mil}"]['precision'] = 0
                                
                        print(f"✓ Loaded external results for {feature}_{mil}")
                        external_loaded = True
                        break
            
            if not external_loaded:
                print(f"✗ External results not found for {feature}_{mil}")
    
    print("-" * 40)
    print(f"Total models loaded: {len(internal_results)} internal, {len(external_results)} external")
    
    return internal_results, external_results

def prepare_dataframe(internal_results, external_results):
    """Convert results to unified dataframe"""
    rows = []
    
    # Get common model keys (models that have both internal and external results)
    common_models = set(internal_results.keys()) & set(external_results.keys())
    
    print(f"\nProcessing {len(common_models)} models with both internal and external results...")
    
    for model_name in common_models:
        # Parse model name
        parts = model_name.split('_')
        feature = parts[0]
        mil = '_'.join(parts[1:])
        
        # Get metrics
        int_metrics = internal_results[model_name]
        ext_metrics = external_results[model_name]
        
        row = {
            'feature_extractor': feature,
            'mil_architecture': mil,
            'model_name': model_name,
            # Internal metrics
            'int_auroc': int_metrics.get('auroc', 0),
            'int_auroc_std': int_metrics.get('auroc_std', 0),
            'int_f1': int_metrics.get('f1_score', 0),
            'int_f1_std': int_metrics.get('f1_std', 0),
            'int_sensitivity': int_metrics.get('sensitivity', 0),
            'int_specificity': int_metrics.get('specificity', 0),
            'int_accuracy': int_metrics.get('accuracy', 0),
            'int_balanced_acc': int_metrics.get('balanced_accuracy', 0),
            'int_mcc': int_metrics.get('mcc', 0),
            'int_precision': int_metrics.get('precision', 0),
            'int_auc_pr': int_metrics.get('auc_pr', 0),
            # External metrics
            'ext_auroc': ext_metrics.get('auroc', 0),
            'ext_auroc_std': ext_metrics.get('auroc_std', 0),
            'ext_f1': ext_metrics.get('f1_score', 0),
            'ext_f1_std': ext_metrics.get('f1_std', 0),
            'ext_sensitivity': ext_metrics.get('sensitivity', 0),
            'ext_specificity': ext_metrics.get('specificity', 0),
            'ext_accuracy': ext_metrics.get('accuracy', 0),
            'ext_balanced_acc': ext_metrics.get('balanced_accuracy', 0),
            'ext_mcc': ext_metrics.get('mcc', 0),
            'ext_precision': ext_metrics.get('precision', 0),
            'ext_auc_pr': ext_metrics.get('auc_pr', 0),
        }
        
        # Calculate performance drops
        row['auroc_drop'] = ((row['int_auroc'] - row['ext_auroc']) / row['int_auroc'] * 100 
                             if row['int_auroc'] > 0 else 0)
        row['f1_drop'] = ((row['int_f1'] - row['ext_f1']) / row['int_f1'] * 100 
                          if row['int_f1'] > 0 else 100)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by external AUROC for consistent ordering
    df = df.sort_values('ext_auroc', ascending=False)
    
    return df

def create_generalization_plot_minimal(df):
    """Create minimalist scatter plot with generalization zones"""
    
    save_path = output_dir / 'figure_5_2_generalization_minimal.png'
    
    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Create generalization zones (shaded areas)
    x_range = np.linspace(0.4, 1.0, 100)
    
    # 5% drop zone (excellent generalization)
    y_5percent_lower = x_range - 0.05  # 5% below diagonal
    ax.fill_between(x_range, x_range, y_5percent_lower, 
                    where=(y_5percent_lower >= 0.35),
                    alpha=0.08, color='green', 
                    label='Excellent (≤5% drop)')
    
    # 10% drop zone (good generalization) 
    y_10percent_lower = x_range - 0.10  # 10% below diagonal
    ax.fill_between(x_range, y_5percent_lower, y_10percent_lower,
                    where=(y_10percent_lower >= 0.35),
                    alpha=0.08, color='orange',
                    label='Good (5-10% drop)')
    
    # Plot diagonal line (perfect generalization)
    ax.plot([0.4, 1], [0.4, 1], 'k--', alpha=0.4, linewidth=1.2, 
            label='Perfect Generalization', zorder=2)
    
    # Plot 5% and 10% boundary lines
    ax.plot(x_range, y_5percent_lower, '--', color='green', alpha=0.3, linewidth=1, zorder=1)
    ax.plot(x_range, y_10percent_lower, '--', color='orange', alpha=0.3, linewidth=1, zorder=1)
    
    # Plot all models
    for _, row in df.iterrows():
        marker = MIL_MARKERS[row['mil_architecture']]
        color = FEATURE_COLORS.get(row['feature_extractor'], '#333333')
        
        ax.scatter(row['int_auroc'], row['ext_auroc'], 
                  marker=marker, c=[color], s=80, 
                  alpha=0.7, edgecolors='white', linewidth=1,
                  zorder=3)
    
    # Create legend
    from matplotlib.lines import Line2D
    
    legend_elements = []
    
    # Add feature extractors (colors) with updated labels
    legend_elements.append(Line2D([0], [0], linestyle='', label='Feature Extractor:', marker=''))
    
    fe_labels = {
        'resnet18': 'ResNet-18',
        'resnet50': 'ResNet-50',
        'conch': 'CONCH',
        'uni2-h': 'UNI2-H',
        'virchow2': 'Virchow-2',
        'h-optimus': 'H-Optimus-1'
    }
    
    for fe, color in FEATURE_COLORS.items():
        label = fe_labels.get(fe, fe.upper())
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=8, 
                                     label=f'  {label}', alpha=0.7))
    
    # Add separator
    legend_elements.append(Line2D([0], [0], linestyle='', label='', marker=''))
    
    # Add MIL architectures (shapes)
    legend_elements.append(Line2D([0], [0], linestyle='', label='MIL Architecture:', marker=''))
    
    mil_labels = {
        'mean': 'Mean',
        'attention': 'ABMIL',
        'clam': 'CLAM',
        'acmil': 'ACMIL'
    }
    
    for mil, marker in MIL_MARKERS.items():
        label = mil_labels.get(mil, mil.upper())
        legend_elements.append(Line2D([0], [0], marker=marker, color='w', 
                                     markerfacecolor='gray', markersize=8, 
                                     label=f'  {label}', alpha=0.7))
    
    # First legend for models
    first_legend = ax.legend(handles=legend_elements, 
                            loc='upper left',
                            fontsize=12,
                            framealpha=0.9)
    ax.add_artist(first_legend)
    
    # Second legend for zones
    from matplotlib.patches import Patch
    zone_elements = [
        Line2D([0], [0], color='black', linestyle='--', alpha=0.4, label='Perfect Generalization'),
        Patch(facecolor='green', alpha=0.15, label='Excellent (≤5% drop)'),
        Patch(facecolor='orange', alpha=0.15, label='Good (5-10% drop)')
    ]
    
    ax.legend(handles=zone_elements,
             loc='lower right',
             fontsize=12,
             title='Generalization Zones',
             framealpha=0.95)
    
    # Clean styling
    ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.5)
    ax.set_xlabel('Internal Validation AUROC (mean)', fontsize=14, fontweight='semibold', labelpad=12)
    ax.set_ylabel('External Validation AUROC (mean)', fontsize=14, fontweight='semibold', labelpad=12)
    ax.set_title('Model Generalization Performance', 
                fontsize=16, fontweight='bold',  pad=20)
    
    # Set limits
    ax.set_xlim(0.45, 0.85)
    ax.set_ylim(0.35, 0.80)
    # ADD THIS LINE HERE to match the external AUROC plot
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)  # Optional: also set y-axis for consistency
    
    
    plt.tight_layout()
    
    # Save with high DPI for thesis
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    
    # Also save as PDF
    pdf_path = output_dir / 'figure_5_2_generalization_minimal.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.show()
    
    print(f"Saved: {save_path}")
    print(f"Saved PDF: {pdf_path}")
    return fig


def create_top_models_table(df, n_top=10):
    """Create table of top performing models"""
    
    save_path = output_dir / 'table_5_1_top_models.csv'
    
    # Sort by external AUROC
    top_models = df.nlargest(min(n_top, len(df)), 'ext_auroc').copy()
    
    # Create formatted table
    table_data = []
    for idx, (_, row) in enumerate(top_models.iterrows()):
        table_row = {
            'Rank': idx + 1,
            'Feature Extractor': row['feature_extractor'].upper(),
            'MIL Architecture': row['mil_architecture'].upper(),
            'Internal AUROC': f"{row['int_auroc']:.3f} ± {row['int_auroc_std']:.3f}",
            'Internal F1': f"{row['int_f1']:.3f}",
            'Internal Sens.': f"{row['int_sensitivity']:.3f}",
            'External AUROC': f"{row['ext_auroc']:.3f} ± {row['ext_auroc_std']:.3f}",
            'External F1': f"{row['ext_f1']:.3f}",
            'External Sens.': f"{row['ext_sensitivity']:.3f}",
            'AUROC Drop (%)': f"{row['auroc_drop']:.1f}",
            'External MCC': f"{row['ext_mcc']:.3f}"
        }
        table_data.append(table_row)
    
    table_df = pd.DataFrame(table_data)
    
    # Save to CSV
    table_df.to_csv(save_path, index=False)
    
    # Display
    print("\nTable 5.1: Top Model Configurations (Ranked by External AUROC)")
    print("="*140)
    print(table_df.to_string(index=False))
    print(f"\nSaved: {save_path}")
    
    return table_df

# Summary statistics function with file output
def calculate_summary_statistics(df):
    """Calculate and print summary statistics"""
    
    stats_path = output_dir / 'summary_statistics.txt'
    
    # Open file for writing
    with open(stats_path, 'w') as f:
        # Print to both console and file
        def print_both(text):
            print(text)
            f.write(text + '\n')
        
        print_both("\n" + "="*60)
        print_both("SUMMARY STATISTICS")
        print_both("="*60)
        
        # Group statistics
        imagenet_models = df[df['feature_extractor'].isin(['resnet18', 'resnet50'])]
        foundation_models = df[~df['feature_extractor'].isin(['resnet18', 'resnet50'])]
        
        print_both("\n1. Feature Extractor Comparison:")
        if len(imagenet_models) > 0:
            print_both(f"   ImageNet Models (n={len(imagenet_models)}):")
            print_both(f"      Internal AUROC: {imagenet_models['int_auroc'].mean():.3f} ± {imagenet_models['int_auroc'].std():.3f}")
            print_both(f"      External AUROC: {imagenet_models['ext_auroc'].mean():.3f} ± {imagenet_models['ext_auroc'].std():.3f}")
            print_both(f"      Avg Drop: {imagenet_models['auroc_drop'].mean():.1f}%")
        
        if len(foundation_models) > 0:
            print_both(f"   Foundation Models (n={len(foundation_models)}):")
            print_both(f"      Internal AUROC: {foundation_models['int_auroc'].mean():.3f} ± {foundation_models['int_auroc'].std():.3f}")
            print_both(f"      External AUROC: {foundation_models['ext_auroc'].mean():.3f} ± {foundation_models['ext_auroc'].std():.3f}")
            print_both(f"      Avg Drop: {foundation_models['auroc_drop'].mean():.1f}%")
        
        # Statistical test
        if len(imagenet_models) > 0 and len(foundation_models) > 0:
            _, p_value = stats.mannwhitneyu(imagenet_models['ext_auroc'], foundation_models['ext_auroc'])
            print_both(f"   Mann-Whitney U test: p = {p_value:.3e}")
        
        print_both("\n2. MIL Architecture Comparison:")
        for mil in df['mil_architecture'].unique():
            mil_data = df[df['mil_architecture'] == mil]
            print_both(f"   {mil.upper()}:")
            print_both(f"      Internal AUROC: {mil_data['int_auroc'].mean():.3f} ± {mil_data['int_auroc'].std():.3f}")
            print_both(f"      External AUROC: {mil_data['ext_auroc'].mean():.3f} ± {mil_data['ext_auroc'].std():.3f}")
            print_both(f"      Avg Drop: {mil_data['auroc_drop'].mean():.1f}%")
        
        print_both("\n3. Generalization Metrics:")
        print_both(f"   Average AUROC Drop: {df['auroc_drop'].mean():.1f}% ± {df['auroc_drop'].std():.1f}%")
        if len(df) > 0:
            best_gen = df.nsmallest(1, 'auroc_drop').iloc[0]
            worst_gen = df.nlargest(1, 'auroc_drop').iloc[0]
            print_both(f"   Best Generalizer: {best_gen['model_name']} ({best_gen['auroc_drop']:.1f}% drop)")
            print_both(f"   Worst Generalizer: {worst_gen['model_name']} ({worst_gen['auroc_drop']:.1f}% drop)")
        
        print_both("\n4. Best Overall Model:")
        if len(df) > 0:
            best_model = df.nlargest(1, 'ext_auroc').iloc[0]
            print_both(f"   Model: {best_model['model_name']}")
            print_both(f"   External AUROC: {best_model['ext_auroc']:.3f} ± {best_model['ext_auroc_std']:.3f}")
            print_both(f"   External F1-Score: {best_model['ext_f1']:.3f}")
            print_both(f"   External Sensitivity: {best_model['ext_sensitivity']:.3f}")
            print_both(f"   External Specificity: {best_model['ext_specificity']:.3f}")
            print_both(f"   External MCC: {best_model['ext_mcc']:.3f}")
        
        print_both("\n" + "="*60)
    
    print(f"Summary statistics saved to: {stats_path}")

# FIGURE: External AUROC Performance - Streamlined with Performance Colors
def create_external_auroc_vertical(df):
    """Create streamlined vertical bar plot with performance-based colors"""
    
    save_path = output_dir / 'figure_5_external_auroc_vertical.png'
    
    # Sort by external AUROC (descending for vertical)
    df_sorted = df.sort_values('ext_auroc', ascending=False)
    
    # Use a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Prepare data
    n_models = len(df_sorted)
    x_pos = np.arange(n_models)
    
    # Create colors based on performance zones (no gradients)
    colors = []
    for auroc in df_sorted['ext_auroc'].values:
        if auroc >= 0.7:
            # Green for clinically acceptable
            colors.append('#27ae60')  # Single green color
        elif auroc >= 0.6:
            # Yellow for borderline
            colors.append('#f1c40f')  # Single yellow color
        elif auroc >= 0.5:
            # Orange for poor
            colors.append('#f39c12')  # Single orange color
        else:
            # Red for below random
            colors.append('#e74c3c')  # Single red color
    
    # Create bars with gradient colors
    bars = ax.bar(x_pos, df_sorted['ext_auroc'].values, 
                  color=colors,
                  alpha=0.85, 
                  edgecolor='#2c3e50', 
                  linewidth=1.0,
                  yerr=df_sorted['ext_auroc_std'].values,
                  error_kw={'elinewidth': 1.0, 'capsize': 3, 'alpha': 0.6, 'color': '#34495e'})
    
    # Add value labels on top of bars
    for i, (bar, auroc, std) in enumerate(zip(bars, df_sorted['ext_auroc'].values, 
                                              df_sorted['ext_auroc_std'].values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.006,
                f'{auroc:.3f}', ha='center', va='bottom', 
                fontsize=10, fontweight='semibold',
                color='#2c3e50')
    
    # Customize x-axis with professional labels
    ax.set_xticks(x_pos)
    
    # Create custom labels with clear distinction
    model_labels = []
    for _, row in df_sorted.iterrows():
        feature = row['feature_extractor']
        mil = row['mil_architecture']
        
        # Professional feature names
        if feature == 'resnet18':
            feature_abbr = 'ResNet-18'
        elif feature == 'resnet50':
            feature_abbr = 'ResNet-50'
        elif feature == 'conch':
            feature_abbr = 'CONCH'
        elif feature == 'uni2-h':
            feature_abbr = 'UNI2-H'
        elif feature == 'virchow2':
            feature_abbr = 'Virchow-2'
        elif feature == 'h-optimus':
            feature_abbr = 'H-Optimus-1'
        else:
            feature_abbr = feature.upper()
        
        # Professional MIL names
        if mil == 'mean':
            mil_abbr = 'Mean'
        elif mil == 'attention':
            mil_abbr = 'ABMIL'
        elif mil == 'clam':
            mil_abbr = 'CLAM'
        elif mil == 'acmil':
            mil_abbr = 'ACMIL'
        else:
            mil_abbr = mil.upper()
        
        # Combine with newline
        label = f"{feature_abbr}\n{mil_abbr}"
        model_labels.append(label)
    
    ax.set_xticklabels(model_labels, fontsize=11, ha='right', rotation=45, rotation_mode='anchor')
    
    # Color x-labels by feature extractor for consistency with other figures
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        color = FEATURE_COLORS.get(row['feature_extractor'], '#333333')
        ax.get_xticklabels()[i].set_color(color)
        ax.get_xticklabels()[i].set_weight('bold')
    
    # Add x-axis label
    ax.set_xlabel('Model Configuration (Feature Extractor + MIL Architecture)', 
                  fontsize=14, fontweight='semibold', labelpad=12)
    
    # Subtle performance zones in background (no legend)
    ax.axhspan(0.35, 0.5, alpha=0.03, color='red', zorder=0)
    ax.axhspan(0.5, 0.6, alpha=0.03, color='orange', zorder=0)
    ax.axhspan(0.6, 0.7, alpha=0.03, color='yellow', zorder=0)
    ax.axhspan(0.7, 0.8, alpha=0.03, color='lightgreen', zorder=0)
    ax.axhspan(0.8, 0.85, alpha=0.03, color='green', zorder=0)
    
    # Add reference lines
    ax.axhline(y=0.5, color='#c0392b', linestyle='--', alpha=0.5, linewidth=1.3, 
               label='(AUROC = 0.5)', zorder=1)
    ax.axhline(y=0.7, color='#27ae60', linestyle='--', alpha=0.5, linewidth=1.3, 
               label='(AUROC = 0.7)', zorder=1)
    
    
    # Customize y-axis
    ax.set_ylabel('External Validation mean AUROC (mean ± std)', fontsize=14, fontweight='semibold', labelpad=12)
    ax.set_ylim(0.35, 0.85)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.025))
    
    # Add refined grid
    ax.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5, which='major')
    ax.grid(True, axis='y', alpha=0.1, linestyle=':', linewidth=0.3, which='minor')
    ax.grid(True, axis='x', alpha=0.1, linestyle='-', linewidth=0.3)
    ax.set_axisbelow(True)
    
    # Single clean title
    ax.set_title('External Validation Performance Across Model Configurations by AUROC', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Only legend for reference lines
    ax.legend(loc='upper right',
              fontsize=12,
              framealpha=0.95,
              edgecolor='#ddd',
              fancybox=False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with high DPI for thesis quality
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    # Also save as PDF for LaTeX inclusion
    pdf_path = output_dir / 'figure_5_external_auroc_vertical.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.show()
    
    print(f"Saved: {save_path}")
    print(f"Saved PDF: {pdf_path}")
    return fig

# Main execution function
def generate_all_figures():
    """Generate all figures for Section 5.1"""
    
    print("="*60)
    print("GENERATING FIGURES FOR SECTION 5.1")
    print("="*60)
    
    # Load all model results
    internal_results, external_results = load_all_model_results()
    
    # Prepare dataframe
    print("\nPreparing unified dataframe...")
    df = prepare_dataframe(internal_results, external_results)
    
    if df.empty:
        print("ERROR: No data found. Please check your folder structure.")
        return None
    
    print(f"Successfully loaded {len(df)} model configurations")
    
    # Generate all figures
    print("\n" + "="*40)
    print("GENERATING FIGURES")
    print("="*40)
    
    # Optional: Also create the minimal version
    print("\nCreating Figure 5.2b: Minimal Generalization Plot...")
    create_generalization_plot_minimal(df)

    create_external_auroc_vertical(df)
    print("\nCreating Table 5.1: Top Models...")
    create_top_models_table(df)
    
    print("\nCalculating summary statistics...")
    calculate_summary_statistics(df)
    
    # Save the processed dataframe
    csv_path = output_dir / 'processed_results_all_models.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nProcessed results saved to: {csv_path}")
    
    # Create a README file with information about the run
    readme_path = output_dir / 'README.txt'
    with open(readme_path, 'w') as f:
        f.write(f"Main Study Results - Generated {timestamp}\n")
        f.write("="*60 + "\n\n")
        f.write("Files in this directory:\n")
        f.write("- figure_5_1_heatmap.png: Performance heatmap for all models\n")
        f.write("- figure_5_2_generalization.png: Generalization scatter plot\n")
        f.write("- figure_5_2_generalization_minimal.png: Minimal version of generalization plot\n")
        f.write("- table_5_1_top_models.csv: Top performing models table\n")
        f.write("- processed_results_all_models.csv: Complete results dataframe\n")
        f.write("- summary_statistics.txt: Statistical analysis summary\n")
        f.write("\n")
        f.write(f"Total models analyzed: {len(df)}\n")
        f.write(f"Best performing model: {df.nlargest(1, 'ext_auroc')['model_name'].values[0] if len(df) > 0 else 'N/A'}\n")
    
    print(f"\nREADME created: {readme_path}")
    
    print("\n" + "="*40)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print(f"Results saved in: {output_dir}")
    print("="*40)
    
    return df

if __name__ == "__main__":
    # Generate all figures
    df_results = generate_all_figures()