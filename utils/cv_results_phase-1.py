# cv_visualization.py


import pandas as pd
import os
import numpy as np

def consolidate_cv_results(base_dir="results_regression_phase1_cv"):
    """
    Consolidate individual CV results into master comparison file
    """
    feature_extractors = ["resnet18", "resnet50", "conch", "uni2-h", "virchow2", "h-optimus"]
    
    all_summaries = []
    
    print("="*60)
    print("CONSOLIDATING PHASE 1 CV RESULTS")
    print("="*60)
    
    for feature in feature_extractors:
        summary_file = os.path.join(base_dir, feature, f"{feature}_cv_summary_stats.csv")
        
        if os.path.exists(summary_file):
            print(f"Loading {feature.upper()}...")
            
            # Load summary stats
            stats_df = pd.read_csv(summary_file, index_col=0)
            
            # Extract key metrics
            summary_row = {
                'Model': feature,
                'AUROC_mean': stats_df.loc['auroc', 'mean'],
                'AUROC_std': stats_df.loc['auroc', 'std'],
                'AUROC_cv': stats_df.loc['auroc', 'cv'],
                'RMSE_mean': stats_df.loc['rmse', 'mean'],
                'RMSE_std': stats_df.loc['rmse', 'std'],
                'R2_mean': stats_df.loc['r2', 'mean'],
                'R2_std': stats_df.loc['r2', 'std'],
                'Binary_Accuracy_mean': stats_df.loc['binary_accuracy', 'mean'],
                'Binary_Accuracy_std': stats_df.loc['binary_accuracy', 'std'],
                'F1_Score_mean': stats_df.loc['f1_score', 'mean'],
                'F1_Score_std': stats_df.loc['f1_score', 'std'],
                'Boundary_MAE_mean': stats_df.loc['boundary_mae', 'mean'],
                'Boundary_MAE_std': stats_df.loc['boundary_mae', 'std']
            }
            
            all_summaries.append(summary_row)
            
            # Print quick summary
            auroc = summary_row['AUROC_mean']
            auroc_std = summary_row['AUROC_std']
            print(f"  {feature.upper()}: AUROC = {auroc:.3f} ± {auroc_std:.3f}")
            
        else:
            print(f"Warning: {summary_file} not found - skipping {feature}")
    
    if not all_summaries:
        print("No CV results found! Run CV experiments first.")
        return None
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_summaries)
    
    # Sort by mean AUROC
    comparison_df = comparison_df.sort_values('AUROC_mean', ascending=False)
    
    # Save master comparison file
    output_path = os.path.join(base_dir, "foundation_models_cv_comparison.csv")
    comparison_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"CONSOLIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Master comparison saved to: {output_path}")
    print(f"Found results for {len(all_summaries)} foundation models")
    
    # Print ranking
    print(f"\nFOUNDATION MODEL RANKING (by AUROC):")
    for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
        print(f"  {i}. {row['Model'].upper()}: {row['AUROC_mean']:.3f} ± {row['AUROC_std']:.3f}")
    
    return comparison_df, output_path

if __name__ == "__main__":
    # Run consolidation
    comparison_df, output_path = consolidate_cv_results()
    
    if comparison_df is not None:
        print(f"\nReady for visualization! Run:")
        print(f"python cv_visualization.py")
        
        # Quick preview of results
        print(f"\nQuick Preview:")
        print(comparison_df[['Model', 'AUROC_mean', 'AUROC_std', 'RMSE_mean']].to_string(index=False))


# cv_visualization.py
# Visualization functions for Phase 1 cross-validation results with error bars

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

def plot_cv_foundation_ranking_with_errorbars(comparison_df, save_path=None):
    """
    Create horizontal bar chart with error bars showing CV results
    """
    # Sort by mean AUROC
    sorted_df = comparison_df.sort_values('AUROC_mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color scheme: Foundation models vs ImageNet models
    colors = []
    for model in sorted_df['Model']:
        if model in ['resnet18', 'resnet50']:
            colors.append('#CD5C5C')  # Red for ImageNet
        else:
            colors.append('#2E8B57')  # Green for Foundation
    
    # Create horizontal bars with error bars
    y_pos = np.arange(len(sorted_df))
    bars = ax.barh(y_pos, sorted_df['AUROC_mean'], 
                   xerr=sorted_df['AUROC_std'],  # Error bars showing standard deviation
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.8,
                   capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
    
    # Add value labels on bars (mean ± std)
    for i, (mean_val, std_val) in enumerate(zip(sorted_df['AUROC_mean'], sorted_df['AUROC_std'])):
        ax.text(mean_val + std_val + 0.005, i, 
                f'{mean_val:.3f}±{std_val:.3f}', 
                va='center', fontweight='bold', fontsize=10)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df['Model'].str.upper())
    ax.set_xlabel('AUROC (5-Fold CV)', fontweight='bold')
    ax.set_title('Phase 1: Foundation Model Performance Ranking\n(5-Fold Cross-Validation with 95% Confidence Intervals)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [Rectangle((0,0),1,1, facecolor='#2E8B57', alpha=0.8, label='Foundation Models'),
                      Rectangle((0,0),1,1, facecolor='#CD5C5C', alpha=0.8, label='ImageNet Models')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Grid and styling
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(sorted_df['AUROC_mean'] + sorted_df['AUROC_std']) * 1.15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_cv_multi_metric_comparison(comparison_df, save_path=None):
    """
    Create multi-metric comparison with error bars for CV results
    """
    # Select key metrics
    metrics_config = [
        ('AUROC', 'AUROC_mean', 'AUROC_std', '#1f77b4', 'Clinical Decision'),
        ('Binary_Accuracy', 'Binary_Accuracy_mean', 'Binary_Accuracy_std', '#ff7f0e', 'Threshold Accuracy'),
        ('F1_Score', 'F1_Score_mean', 'F1_Score_std', '#2ca02c', 'Balanced Classification'),
        ('R2', 'R2_mean', 'R2_std', '#d62728', 'Regression Quality')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()
    
    for idx, (metric_name, mean_col, std_col, color, description) in enumerate(metrics_config):
        ax = axes_flat[idx]
        
        # Sort by this metric
        sorted_df = comparison_df.sort_values(mean_col, ascending=False)
        
        # Create bars with error bars
        x_pos = np.arange(len(sorted_df))
        bars = ax.bar(x_pos, sorted_df[mean_col], 
                     yerr=sorted_df[std_col],
                     color=color, alpha=0.7, edgecolor='black',
                     capsize=5, error_kw={'linewidth': 1.5, 'capthick': 2})
        
        # Add value labels
        for i, (mean_val, std_val) in enumerate(zip(sorted_df[mean_col], sorted_df[std_col])):
            ax.text(i, mean_val + std_val + max(sorted_df[mean_col]) * 0.02,
                   f'{mean_val:.3f}±{std_val:.3f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=9,
                   rotation=45 if len(sorted_df) > 4 else 0)
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_df['Model'].str.upper(), rotation=45, ha='right')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name}: {description}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set y-limits to show error bars properly
        max_val = max(sorted_df[mean_col] + sorted_df[std_col])
        ax.set_ylim(0, max_val * 1.15)
    
    plt.suptitle('Phase 1: Multi-Metric Foundation Model Comparison\n(5-Fold Cross-Validation Results)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_cv_stability_analysis(comparison_df, save_path=None):
    """
    Analyze model stability using coefficient of variation
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Performance vs Stability scatter plot
    ax1.scatter(comparison_df['AUROC_mean'], comparison_df['AUROC_cv'], 
               s=100, alpha=0.7, c=['red' if model in ['resnet18', 'resnet50'] else 'green' 
                                   for model in comparison_df['Model']])
    
    # Add model labels
    for i, model in enumerate(comparison_df['Model']):
        ax1.annotate(model.upper(), 
                    (comparison_df['AUROC_mean'].iloc[i], comparison_df['AUROC_cv'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax1.set_xlabel('Mean AUROC Performance', fontweight='bold')
    ax1.set_ylabel('AUROC Coefficient of Variation', fontweight='bold')
    ax1.set_title('Performance vs Stability Trade-off', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add stability zones
    ax1.axhline(y=0.02, color='green', linestyle='--', alpha=0.5, label='Highly Stable')
    ax1.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Moderately Stable')
    ax1.legend()
    
    # 2. Error bar comparison for top metrics
    top_3_models = comparison_df.nlargest(3, 'AUROC_mean')
    
    x = np.arange(len(top_3_models))
    width = 0.35
    
    # AUROC bars
    bars1 = ax2.bar(x - width/2, top_3_models['AUROC_mean'], width,
                   yerr=top_3_models['AUROC_std'], 
                   label='AUROC', color='skyblue', alpha=0.8,
                   capsize=5, error_kw={'linewidth': 2})
    
    # Binary Accuracy bars  
    bars2 = ax2.bar(x + width/2, top_3_models['Binary_Accuracy_mean'], width,
                   yerr=top_3_models['Binary_Accuracy_std'],
                   label='Binary Accuracy', color='lightcoral', alpha=0.8,
                   capsize=5, error_kw={'linewidth': 2})
    
    # Add value labels
    for bars, means, stds in [(bars1, top_3_models['AUROC_mean'], top_3_models['AUROC_std']),
                             (bars2, top_3_models['Binary_Accuracy_mean'], top_3_models['Binary_Accuracy_std'])]:
        for bar, mean_val, std_val in zip(bars, means, stds):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std_val + 0.01,
                    f'{mean_val:.3f}±{std_val:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.set_xlabel('Top 3 Foundation Models', fontweight='bold')
    ax2.set_ylabel('Performance Score', fontweight='bold')
    ax2.set_title('Top 3 Models: Detailed Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_3_models['Model'].str.upper())
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_cv_imagenet_vs_foundation_comparison(comparison_df, save_path=None):
    """
    Compare ImageNet vs Foundation models with error bars
    """
    # Separate model categories
    imagenet_models = comparison_df[comparison_df['Model'].isin(['resnet18', 'resnet50'])]
    foundation_models = comparison_df[~comparison_df['Model'].isin(['resnet18', 'resnet50'])]
    
    # Calculate category averages and pooled standard errors
    def calculate_pooled_stats(df, metric_prefix):
        means = df[f'{metric_prefix}_mean'].values
        stds = df[f'{metric_prefix}_std'].values
        n_folds = 5  # Each model trained with 5-fold CV
        
        # Pooled mean
        pooled_mean = np.mean(means)
        
        # Pooled standard error (approximate)
        pooled_se = np.sqrt(np.sum(stds**2) / len(stds)) / np.sqrt(n_folds)
        
        return pooled_mean, pooled_se
    
    metrics = ['AUROC', 'Binary_Accuracy', 'F1_Score', 'R2']
    
    imagenet_stats = {}
    foundation_stats = {}
    
    for metric in metrics:
        imagenet_stats[metric] = calculate_pooled_stats(imagenet_models, metric)
        foundation_stats[metric] = calculate_pooled_stats(foundation_models, metric)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Main performance metrics with error bars
    x = np.arange(len(metrics))
    width = 0.35
    
    imagenet_means = [imagenet_stats[m][0] for m in metrics]
    imagenet_errs = [imagenet_stats[m][1] for m in metrics]
    foundation_means = [foundation_stats[m][0] for m in metrics] 
    foundation_errs = [foundation_stats[m][1] for m in metrics]
    
    bars1 = ax1.bar(x - width/2, imagenet_means, width, 
                   yerr=imagenet_errs, label='ImageNet Models', 
                   color='#CD5C5C', alpha=0.8, capsize=5)
    bars2 = ax1.bar(x + width/2, foundation_means, width,
                   yerr=foundation_errs, label='Foundation Models', 
                   color='#2E8B57', alpha=0.8, capsize=5)
    
    # Add value labels with error
    for bars, means, errs in [(bars1, imagenet_means, imagenet_errs), 
                             (bars2, foundation_means, foundation_errs)]:
        for bar, mean_val, err in zip(bars, means, errs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.01,
                    f'{mean_val:.3f}±{err:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax1.set_ylabel('Performance Score', fontweight='bold')
    ax1.set_title('ImageNet vs Foundation Models\n(5-Fold CV Performance)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. Individual model error bars (AUROC only)
    models = comparison_df['Model'].str.upper()
    auroc_means = comparison_df['AUROC_mean'] 
    auroc_stds = comparison_df['AUROC_std']
    
    # Sort for cleaner visualization
    sort_idx = np.argsort(auroc_means)[::-1]  # Descending order
    models_sorted = models.iloc[sort_idx]
    means_sorted = auroc_means.iloc[sort_idx] 
    stds_sorted = auroc_stds.iloc[sort_idx]
    
    colors_sorted = ['#CD5C5C' if model.lower() in ['resnet18', 'resnet50'] else '#2E8B57' 
                    for model in models_sorted]
    
    x_pos = np.arange(len(models_sorted))
    bars = ax2.bar(x_pos, means_sorted, yerr=stds_sorted,
                  color=colors_sorted, alpha=0.8, edgecolor='black',
                  capsize=4, error_kw={'linewidth': 1.5})
    
    # Add value labels
    for bar, mean_val, std_val in zip(bars, means_sorted, stds_sorted):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.005,
                f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models_sorted, rotation=45, ha='right')
    ax2.set_ylabel('AUROC', fontweight='bold')
    ax2.set_title('Detailed AUROC Comparison\n(Mean ± Standard Deviation)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_cv_regression_vs_classification_performance(comparison_df, save_path=None):
    """
    Compare regression performance vs classification performance with error bars
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sort by AUROC for consistency
    sorted_df = comparison_df.sort_values('AUROC_mean', ascending=False)
    
    x_pos = np.arange(len(sorted_df))
    width = 0.4
    
    # 1. Regression Metrics (RMSE and R²)
    # Note: For RMSE, lower is better, so we'll invert for visualization
    rmse_normalized = 1 - ((sorted_df['RMSE_mean'] - sorted_df['RMSE_mean'].min()) / 
                          (sorted_df['RMSE_mean'].max() - sorted_df['RMSE_mean'].min()))
    
    bars1 = ax1.bar(x_pos - width/2, sorted_df['R2_mean'], width,
                   yerr=sorted_df['R2_std'], label='R² Score',
                   color='#4CAF50', alpha=0.8, capsize=4)
    
    bars2 = ax1.bar(x_pos + width/2, rmse_normalized, width,
                   label='RMSE (Inverted)', color='#2196F3', alpha=0.8)
    
    ax1.set_ylabel('Performance Score', fontweight='bold')
    ax1.set_title('Regression Performance\n(R² and Normalized RMSE)', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sorted_df['Model'].str.upper(), rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Add R² value labels only
    for bar, r2_val, r2_std in zip(bars1, sorted_df['R2_mean'], sorted_df['R2_std']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + r2_std + 0.02,
                f'{r2_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 2. Classification Metrics
    bars3 = ax2.bar(x_pos - width/2, sorted_df['AUROC_mean'], width,
                   yerr=sorted_df['AUROC_std'], label='AUROC',
                   color='#FF9800', alpha=0.8, capsize=4)
    
    bars4 = ax2.bar(x_pos + width/2, sorted_df['F1_Score_mean'], width,
                   yerr=sorted_df['F1_Score_std'], label='F1-Score',
                   color='#9C27B0', alpha=0.8, capsize=4)
    
    ax2.set_ylabel('Performance Score', fontweight='bold') 
    ax2.set_title('Classification Performance\n(AUROC and F1-Score)', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sorted_df['Model'].str.upper(), rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add value labels
    for bars, means, stds in [(bars3, sorted_df['AUROC_mean'], sorted_df['AUROC_std']),
                             (bars4, sorted_df['F1_Score_mean'], sorted_df['F1_Score_std'])]:
        for bar, mean_val, std_val in zip(bars, means, stds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.01,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def create_cv_summary_table(comparison_df, save_path=None):
    """
    Create a publication-ready summary table
    """
    # Sort by AUROC
    sorted_df = comparison_df.sort_values('AUROC_mean', ascending=False).copy()
    
    # Add ranking
    sorted_df['Rank'] = range(1, len(sorted_df) + 1)
    
    print("="*120)
    print("PHASE 1: FOUNDATION MODEL CROSS-VALIDATION RESULTS")
    print("="*120)
    print(f"{'Rank':<4} {'Model':<12} {'AUROC':<13} {'RMSE':<13} {'R²':<13} {'F1':<13} {'Boundary MAE':<13}")
    print("-" * 120)
    
    for _, row in sorted_df.iterrows():
        print(f"{int(row['Rank']):<4} {row['Model'].upper():<12} "
              f"{row['AUROC_mean']:.3f}±{row['AUROC_std']:.3f}  "
              f"{row['RMSE_mean']:.2f}±{row['RMSE_std']:.2f}  "
              f"{row['R2_mean']:.3f}±{row['R2_std']:.3f}   "
              f"{row['F1_Score_mean']:.3f}±{row['F1_Score_std']:.3f}   "
              f"{row['Boundary_MAE_mean']:.2f}±{row['Boundary_MAE_std']:.2f}")
    
    print("="*120)
    
    # Statistical significance analysis
    print(f"\nSTATISTICAL SIGNIFICANCE ANALYSIS:")
    print(f"Top performer: {sorted_df.iloc[0]['Model'].upper()} "
          f"(AUROC: {sorted_df.iloc[0]['AUROC_mean']:.3f}±{sorted_df.iloc[0]['AUROC_std']:.3f})")
    
    # Check if top model is significantly better than second
    top_auroc = sorted_df.iloc[0]['AUROC_mean']
    top_std = sorted_df.iloc[0]['AUROC_std']
    second_auroc = sorted_df.iloc[1]['AUROC_mean'] 
    second_std = sorted_df.iloc[1]['AUROC_std']
    
    # Approximate significance using non-overlapping confidence intervals
    top_ci_lower = top_auroc - 1.96 * top_std
    second_ci_upper = second_auroc + 1.96 * second_std
    
    if top_ci_lower > second_ci_upper:
        print(f"✓ Top model is likely significantly better than second-best")
    else:
        print(f"? Performance difference may not be statistically significant")
    
    # Identify models for Phase 2
    print(f"\nRECOMMENDED FOR PHASE 2:")
    top_3 = sorted_df.head(3)
    for i, (_, row) in enumerate(top_3.iterrows(), 1):
        print(f"  {i}. {row['Model'].upper()} - AUROC: {row['AUROC_mean']:.3f}±{row['AUROC_std']:.3f}")
    
    if save_path:
        sorted_df.to_csv(save_path, index=False)
        print(f"\nDetailed results saved to: {save_path}")
    
    return sorted_df

def plot_cv_error_plot_style(comparison_df, save_path=None):
    """
    Create error plot (line chart with error bars) for foundation model comparison
    """
    # Sort by AUROC
    sorted_df = comparison_df.sort_values('AUROC_mean', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create x positions
    x_pos = np.arange(len(sorted_df))
    
    # Color mapping
    colors = ['#CD5C5C' if model in ['resnet18', 'resnet50'] else '#2E8B57' 
             for model in sorted_df['Model']]
    
    # Create error plot
    for i, (idx, row) in enumerate(sorted_df.iterrows()):
        ax.errorbar(i, row['AUROC_mean'], yerr=row['AUROC_std'], 
                   marker='o', markersize=10, capsize=8, capthick=2,
                   color=colors[i], linewidth=2, alpha=0.8)
        
        # Add value labels
        ax.text(i, row['AUROC_mean'] + row['AUROC_std'] + 0.01,
               f'{row["AUROC_mean"]:.3f}±{row["AUROC_std"]:.3f}',
               ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Connect points with line
    ax.plot(x_pos, sorted_df['AUROC_mean'], 'k--', alpha=0.5, linewidth=1)
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_df['Model'].str.upper(), rotation=45, ha='right')
    ax.set_ylabel('AUROC (5-Fold CV)', fontweight='bold')
    ax.set_title('Foundation Model Performance: Error Plot Style\n(Ranked by Mean AUROC)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(sorted_df['AUROC_mean'] + sorted_df['AUROC_std']) * 1.15)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='#2E8B57', linewidth=2, 
                             markersize=8, label='Foundation Models'),
                      Line2D([0], [0], marker='o', color='#CD5C5C', linewidth=2, 
                            markersize=8, label='ImageNet Models')]
    ax.legend(handles=legend_elements, loc='lower left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_cv_dot_plot_style(comparison_df, save_path=None):
    """
    Create Cleveland dot plot for foundation model comparison
    """
    # Sort by AUROC
    sorted_df = comparison_df.sort_values('AUROC_mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create y positions
    y_pos = np.arange(len(sorted_df))
    
    # Color mapping
    colors = ['#CD5C5C' if model in ['resnet18', 'resnet50'] else '#2E8B57' 
             for model in sorted_df['Model']]
    
    # Create dot plot with error bars
    for i, (idx, row) in enumerate(sorted_df.iterrows()):
        # Draw line from 0 to mean (lollipop style)
        ax.plot([0, row['AUROC_mean']], [i, i], color='lightgray', 
               linewidth=1, alpha=0.7, zorder=1)
        
        # Draw error bar
        ax.errorbar(row['AUROC_mean'], i, xerr=row['AUROC_std'],
                   marker='o', markersize=12, capsize=6, capthick=2,
                   color=colors[i], linewidth=2, alpha=0.9, zorder=3)
        
        # Add value labels
        ax.text(row['AUROC_mean'] + row['AUROC_std'] + 0.01, i,
               f'{row["AUROC_mean"]:.3f}±{row["AUROC_std"]:.3f}',
               va='center', ha='left', fontweight='bold', fontsize=10)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df['Model'].str.upper())
    ax.set_xlabel('AUROC (5-Fold CV)', fontweight='bold')
    ax.set_title('Foundation Model Performance: Dot Plot Style\n(Cleveland Plot with Error Bars)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(sorted_df['AUROC_mean'] + sorted_df['AUROC_std']) * 1.15)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='#2E8B57', linewidth=2, 
                                 markersize=10, label='Foundation Models', linestyle='None'),
                      plt.Line2D([0], [0], marker='o', color='#CD5C5C', linewidth=2, 
                                markersize=10, label='ImageNet Models', linestyle='None')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_cv_violin_plot_style(comparison_df, detailed_results_dir, save_path=None):
    """
    Create violin plot showing full distribution across CV folds
    Requires individual fold results from detailed CSV files
    """
    feature_extractors = ["resnet18", "resnet50", "conch", "uni2-h", "virchow2", "h-optimus"]
    
    # Load individual fold results for violin plot
    all_fold_data = []
    model_names = []
    
    for feature in feature_extractors:
        detailed_file = os.path.join(detailed_results_dir, feature, f"{feature}_cv_detailed_results.csv")
        if os.path.exists(detailed_file):
            fold_df = pd.read_csv(detailed_file)
            if 'auroc' in fold_df.columns:
                for auroc_val in fold_df['auroc']:
                    all_fold_data.append({'Model': feature.upper(), 'AUROC': auroc_val})
                model_names.append(feature.upper())
    
    if not all_fold_data:
        print("Warning: No detailed fold results found for violin plot")
        return None
    
    # Create DataFrame for seaborn
    violin_df = pd.DataFrame(all_fold_data)
    
    # Sort by median AUROC
    model_order = violin_df.groupby('Model')['AUROC'].median().sort_values(ascending=False).index
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create violin plot
    sns.violinplot(data=violin_df, x='Model', y='AUROC', order=model_order, ax=ax)
    
    # Color the violins
    for i, model in enumerate(model_order):
        if model.lower() in ['resnet18', 'resnet50']:
            ax.collections[i].set_facecolor('#CD5C5C')
        else:
            ax.collections[i].set_facecolor('#2E8B57')
        ax.collections[i].set_alpha(0.7)
    
    # Add mean markers
    for i, model in enumerate(model_order):
        model_data = violin_df[violin_df['Model'] == model]['AUROC']
        mean_val = model_data.mean()
        std_val = model_data.std()
        ax.plot(i, mean_val, 'wo', markersize=8, markeredgecolor='black', markeredgewidth=2)
        
        # Add text label
        ax.text(i, max(model_data) + 0.01, f'{mean_val:.3f}±{std_val:.3f}',
               ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Formatting
    ax.set_xlabel('Foundation Models', fontweight='bold')
    ax.set_ylabel('AUROC (5-Fold CV)', fontweight='bold')
    ax.set_title('Foundation Model Performance: Violin Plot Style\n(Full Distribution Across CV Folds)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor='#2E8B57', alpha=0.7, label='Foundation Models'),
                      plt.Rectangle((0,0),1,1, facecolor='#CD5C5C', alpha=0.7, label='ImageNet Models')]
    ax.legend(handles=legend_elements, loc='lower left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_cv_box_plot_style(comparison_df, detailed_results_dir, save_path=None):
    """
    Create box plot showing distribution statistics across CV folds
    """
    feature_extractors = ["resnet18", "resnet50", "conch", "uni2-h", "virchow2", "h-optimus"]
    
    # Load individual fold results
    fold_data_by_model = []
    model_names = []
    
    for feature in feature_extractors:
        detailed_file = os.path.join(detailed_results_dir, feature, f"{feature}_cv_detailed_results.csv")
        if os.path.exists(detailed_file):
            fold_df = pd.read_csv(detailed_file)
            if 'auroc' in fold_df.columns:
                fold_data_by_model.append(fold_df['auroc'].values)
                model_names.append(feature.upper())
    
    if not fold_data_by_model:
        print("Warning: No detailed fold results found for box plot")
        return None
    
    # Sort by median performance
    median_performances = [np.median(data) for data in fold_data_by_model]
    sorted_indices = np.argsort(median_performances)[::-1]  # Descending
    
    sorted_data = [fold_data_by_model[i] for i in sorted_indices]
    sorted_names = [model_names[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create box plot
    bp = ax.boxplot(sorted_data, labels=sorted_names, patch_artist=True,
                   showmeans=True, meanline=False, meanprops={'marker': 'D', 'markerfacecolor': 'white',
                                                              'markeredgecolor': 'black', 'markersize': 6})
    
    # Color the boxes
    for i, model in enumerate(sorted_names):
        if model.lower() in ['resnet18', 'resnet50']:
            bp['boxes'][i].set_facecolor('#CD5C5C')
        else:
            bp['boxes'][i].set_facecolor('#2E8B57')
        bp['boxes'][i].set_alpha(0.7)
    
    # Add mean ± std labels above boxes
    for i, data in enumerate(sorted_data):
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.text(i + 1, max(data) + 0.01, f'{mean_val:.3f}±{std_val:.3f}',
               ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Formatting
    ax.set_xlabel('Foundation Models', fontweight='bold')
    ax.set_ylabel('AUROC (5-Fold CV)', fontweight='bold')
    ax.set_title('Foundation Model Performance: Box Plot Style\n(Distribution Statistics Across CV Folds)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-labels if needed
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor='#2E8B57', alpha=0.7, label='Foundation Models'),
                      plt.Rectangle((0,0),1,1, facecolor='#CD5C5C', alpha=0.7, label='ImageNet Models')]
    ax.legend(handles=legend_elements, loc='lower left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def generate_phase1_cv_report(comparison_csv_path, output_dir="results_phase1_cv", detailed_results_dir="results_regression_phase1_cv"):
    """
    Generate complete Phase 1 CV visualization report with multiple plot styles
    """
    # Load results
    comparison_df = pd.read_csv(comparison_csv_path)
    
    print("Generating Phase 1 Cross-Validation Visualization Report...")
    print("Creating multiple plot styles for comparison...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all visualizations
    print("1. Creating foundation model ranking chart (bar plot)...")
    plot_cv_foundation_ranking_with_errorbars(
        comparison_df, 
        save_path=os.path.join(output_dir, "1_foundation_ranking_bar_plot.png")
    )
    
    print("2. Creating error plot style...")
    plot_cv_error_plot_style(
        comparison_df,
        save_path=os.path.join(output_dir, "2_foundation_ranking_error_plot.png")
    )
    
    print("3. Creating dot plot style...")
    plot_cv_dot_plot_style(
        comparison_df,
        save_path=os.path.join(output_dir, "3_foundation_ranking_dot_plot.png")
    )
    
    print("4. Creating violin plot style...")
    plot_cv_violin_plot_style(
        comparison_df, detailed_results_dir,
        save_path=os.path.join(output_dir, "4_foundation_ranking_violin_plot.png")
    )
    
    print("5. Creating box plot style...")
    plot_cv_box_plot_style(
        comparison_df, detailed_results_dir,
        save_path=os.path.join(output_dir, "5_foundation_ranking_box_plot.png")
    )
    
    print("6. Creating multi-metric comparison...")
    plot_cv_multi_metric_comparison(
        comparison_df,
        save_path=os.path.join(output_dir, "6_multi_metric_comparison_cv.png")
    )
    
    print("7. Creating stability analysis...")
    plot_cv_stability_analysis(
        comparison_df,
        save_path=os.path.join(output_dir, "7_stability_analysis_cv.png")
    )
    
    print("8. Creating ImageNet vs Foundation comparison...")
    plot_cv_imagenet_vs_foundation_comparison(
        comparison_df,
        save_path=os.path.join(output_dir, "8_imagenet_vs_foundation_cv.png")
    )
    
    print("9. Creating summary table...")
    summary_df = create_cv_summary_table(
        comparison_df,
        save_path=os.path.join(output_dir, "phase1_cv_summary_table.csv")
    )
    
    print(f"\nComplete Phase 1 CV report generated in: {output_dir}")
    print("Generated files:")
    print("  PLOT STYLE COMPARISON:")
    print("    1_foundation_ranking_bar_plot.png")
    print("    2_foundation_ranking_error_plot.png")
    print("    3_foundation_ranking_dot_plot.png")
    print("    4_foundation_ranking_violin_plot.png")
    print("    5_foundation_ranking_box_plot.png")
    print("  ADDITIONAL ANALYSES:")
    print("    6_multi_metric_comparison_cv.png")
    print("    7_stability_analysis_cv.png")
    print("    8_imagenet_vs_foundation_cv.png")
    print("    phase1_cv_summary_table.csv")
    
    print(f"\nRECOMMENDATION:")
    print(f"Review plots 1-5 to choose your preferred visualization style for the thesis.")
    print(f"Each shows the same data with different visual approaches:")
    print(f"  - Bar plot: Traditional, clear ranking")
    print(f"  - Error plot: Connected line, shows progression")  
    print(f"  - Dot plot: Minimalist, focused on values")
    print(f"  - Violin/Box plots: Full statistical distribution")
    
    return summary_df

# Example usage
if __name__ == "__main__":
    # Example: Generate report from your CV results
    comparison_csv = "results_regression_phase1_cv/foundation_models_cv_comparison.csv"
    
    if os.path.exists(comparison_csv):
        generate_phase1_cv_report(comparison_csv)
    else:
        print(f"Results file not found: {comparison_csv}")
        print("Please run the CV training script first to generate results.")