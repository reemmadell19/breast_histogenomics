#!/usr/bin/env python3
"""
Standalone script to replot baseline class balancing results from saved JSON/CSV files
Usage: python replot_baseline_results.py --results_dir path/to/results_baseline_comparison_[timestamp]
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import argparse
from pathlib import Path

def load_results(results_dir):
    """Load saved results from JSON and CSV files"""
    results_dir = Path(results_dir)
    
    # Load summaries JSON
    json_path = results_dir / "all_summaries.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Cannot find {json_path}")
    
    with open(json_path, 'r') as f:
        all_summaries = json.load(f)
    
    # Load comparison table CSV
    csv_path = results_dir / "strategy_comparison_table.csv"
    if csv_path.exists():
        comparison_df = pd.read_csv(csv_path)
    else:
        comparison_df = None
        print(f"Warning: Cannot find {csv_path}, will recreate from JSON")
    
    return all_summaries, comparison_df

def plot_comprehensive_comparison(all_summaries, save_dir):
    """Create comprehensive visualization of all strategy results"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with custom layout - adjusted for fewer subplots
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # Define consistent colors for each strategy
    strategy_colors = {
        'No Balancing (Baseline)': '#808080',  # Gray
        'Focal Loss': '#FF6B6B',              # Red
        'Weighted Cross-Entropy': '#4ECDC4',   # Teal
        'Balanced Sampling': '#45B7D1',        # Blue
        'Focal Loss + Balanced Sampling': '#96CEB4',  # Green
        'Weighted CE + Balanced Sampling': '#FFEAA7'  # Yellow
    }
    
    strategies = [s['strategy_name'] for s in all_summaries]
    short_names = ['No Bal', 'Focal', 'WCE', 'Bal Samp', 'Focal+Samp', 'WCE+Samp']
    
    # 1. MAIN METRICS COMPARISON (Top left, larger)
    ax1 = fig.add_subplot(gs[0, :2])
    
    main_metrics = ['auroc', 'balanced_accuracy', 'f1_score', 'auc_pr']
    metric_labels = ['AUROC', 'Balanced Acc', 'F1-Score', 'AUC-PR']
    
    x = np.arange(len(main_metrics))
    width = 0.13
    
    # Find best F1 strategy for highlighting
    best_f1_idx = np.argmax([s.get('f1_score_mean', 0) for s in all_summaries])
    
    for i, (strategy_name, summary) in enumerate(zip(strategies, all_summaries)):
        values = [summary.get(f'{m}_mean', 0) for m in main_metrics]
        errors = [summary.get(f'{m}_std', 0) for m in main_metrics]
        offset = (i - len(strategies)/2 + 0.5) * width
        
        # Highlight the best F1 strategy
        if i == best_f1_idx:
            edgecolor = 'red'
            linewidth = 2
            alpha = 1.0
        else:
            edgecolor = 'black'
            linewidth = 1
            alpha = 0.8
            
        bars = ax1.bar(x + offset, values, width, 
                      label=short_names[i],
                      color=strategy_colors[strategy_name],
                      yerr=errors, capsize=3, alpha=alpha,
                      edgecolor=edgecolor, linewidth=linewidth)
    
    ax1.set_xlabel('Metrics', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Primary Classification Metrics Comparison\n(Red outline = Best F1-Score)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels)
    ax1.legend(loc='upper left', ncol=3, framealpha=0.9)
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # 2. RADAR CHART (Top right)
    ax2 = fig.add_subplot(gs[0, 2], projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(main_metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    # Find best strategy based on F1-score ONLY
    best_idx = np.argmax([s.get('f1_score_mean', 0) for s in all_summaries])
    
    # Plot baseline (index 0) and best F1 strategy
    for i in [0, best_idx]:
        summary = all_summaries[i]
        values = [summary.get(f'{m}_mean', 0) for m in main_metrics]
        values += values[:1]
        
        ax2.plot(angles, values, 'o-', linewidth=2, 
                label=short_names[i], 
                color=strategy_colors[strategies[i]],
                markersize=6)
        ax2.fill(angles, values, alpha=0.15, color=strategy_colors[strategies[i]])
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metric_labels, size=9)
    ax2.set_ylim([0, 1])
    ax2.set_title('Baseline vs Best F1-Score Strategy', fontsize=12, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax2.grid(True)
    
    # 3. HEATMAP of all metrics (Bottom row, spanning all columns)
    ax3 = fig.add_subplot(gs[1, :])
    
    all_metrics = ['accuracy', 'balanced_accuracy', 'auroc', 'auc_pr', 
                   'f1_score', 'precision', 'recall', 'specificity', 'mcc']
    
    matrix_data = []
    for summary in all_summaries:
        row = [summary.get(f'{m}_mean', 0) for m in all_metrics]
        matrix_data.append(row)
    
    matrix_data = np.array(matrix_data)
    
    im = ax3.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax3.set_xticks(np.arange(len(all_metrics)))
    ax3.set_yticks(np.arange(len(strategies)))
    ax3.set_xticklabels([m.upper().replace('_', ' ') for m in all_metrics], rotation=45, ha='right')
    ax3.set_yticklabels(short_names)
    
    # Add text annotations
    for i in range(len(strategies)):
        for j in range(len(all_metrics)):
            text = ax3.text(j, i, f'{matrix_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax3.set_title('Complete Metrics Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    plt.suptitle('Class Balancing Strategy Analysis\nResNet-18 + Mean Pooling', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'comprehensive_strategy_comparison_replot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive plot saved to: {save_path}")
    return save_path

def plot_thesis_figure(all_summaries, save_dir):
    """Create a clean figure specifically for thesis publication"""
    plt.style.use('seaborn-v0_8-paper')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    strategies = [s['strategy_name'] for s in all_summaries]
    short_names = ['None', 'Focal', 'WCE', 'Sampling', 'Focal+S', 'WCE+S']
    colors = ['#808080', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Key metrics for thesis
    key_metrics = {
        'balanced_accuracy': 'Balanced Accuracy',
        'f1_score': 'F1-Score', 
        'sensitivity': 'Sensitivity',
        'auroc': 'AUROC'
    }
    
    # 1. Bar plot of key metrics
    ax = axes[0, 0]
    x = np.arange(len(short_names))
    width = 0.2
    
    for i, (metric, label) in enumerate(key_metrics.items()):
        values = [s.get(f'{metric}_mean', 0) for s in all_summaries]
        offset = (i - 1.5) * width
        ax.bar(x + offset, values, width, label=label, alpha=0.8)
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Score')
    ax.set_title('(a) Key Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Sensitivity vs Specificity
    ax = axes[0, 1]
    for i, summary in enumerate(all_summaries):
        ax.scatter(summary.get('specificity_mean', 0),
                  summary.get('sensitivity_mean', 0),
                  s=100, color=colors[i], label=short_names[i],
                  edgecolor='black', linewidth=1, alpha=0.8)
    
    ax.set_xlabel('Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('(b) Sensitivity-Specificity Trade-off')
    ax.legend(loc='best', fontsize=8)
    ax.set_xlim([0.4, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # 3. Improvement over baseline
    ax = axes[1, 0]
    baseline_metrics = all_summaries[0]
    improvements = []
    
    for metric in ['balanced_accuracy', 'f1_score', 'sensitivity']:
        baseline_val = baseline_metrics.get(f'{metric}_mean', 0)
        strategy_improvements = []
        for s in all_summaries[1:]:  # Skip baseline
            val = s.get(f'{metric}_mean', 0)
            improvement = ((val - baseline_val) / baseline_val) * 100
            strategy_improvements.append(improvement)
        improvements.append(strategy_improvements)
    
    x = np.arange(len(short_names[1:]))
    width = 0.25
    
    for i, (metric_impr, metric_name) in enumerate(zip(improvements, ['Bal. Acc', 'F1', 'Sens.'])):
        offset = (i - 1) * width
        ax.bar(x + offset, metric_impr, width, label=metric_name, alpha=0.8)
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Improvement over Baseline (%)')
    ax.set_title('(c) Relative Performance Gain')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names[1:])
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary statistics table (using recall instead of sensitivity)
    table_data = []
    for i, (name, summary) in enumerate(zip(short_names, all_summaries)):
        bal_acc = summary.get('balanced_accuracy_mean', 0)
        f1 = summary.get('f1_score_mean', 0)
        recall = summary.get('recall_mean', 0)  # Changed from sensitivity to recall
        spec = summary.get('specificity_mean', 0)
        
        row = [name, 
               f'{bal_acc:.3f}',
               f'{f1:.3f}',
               f'{recall:.3f}',  # Now using recall
               f'{spec:.3f}']
        table_data.append(row)
    
    headers = ['Strategy', 'Bal. Acc', 'F1', 'Recall', 'Spec.']  # Changed from Sens. to Recall
    
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    
    # Highlight best values in each column
    for col in range(1, len(headers)):
        values = [float(table_data[row][col]) for row in range(len(table_data))]
        best_idx = np.argmax(values) + 1  # +1 for header
        table[(best_idx, col)].set_facecolor('#C8E6C9')
    
    ax.set_title('(d) Performance Summary', y=0.95)
    
    plt.suptitle('Class Balancing Strategy Comparison for Baseline Model', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'thesis_figure_class_balancing.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Thesis figure saved to: {save_path}")
    return save_path

def analyze_best_strategy(all_summaries):
    """Analyze and report the best strategy based on multiple criteria"""
    
    print("\n" + "="*80)
    print("STRATEGY SELECTION ANALYSIS")
    print("="*80)
    
    # Define evaluation criteria and weights
    weights = {
        'balanced_accuracy': 0.30,
        'f1_score': 0.25,
        'sensitivity': 0.25,
        'auroc': 0.10,
        'auc_pr': 0.10
    }
    
    print("\nEvaluation Criteria Weights:")
    for metric, weight in weights.items():
        print(f"  - {metric.replace('_', ' ').title()}: {weight*100:.0f}%")
    
    # Calculate weighted scores
    scores = []
    for summary in all_summaries:
        score = sum(summary.get(f'{m}_mean', 0) * w for m, w in weights.items())
        scores.append({
            'strategy': summary['strategy_name'],
            'weighted_score': score,
            'metrics': {m: summary.get(f'{m}_mean', 0) for m in weights.keys()}
        })
    
    # Sort by weighted score
    scores.sort(key=lambda x: x['weighted_score'], reverse=True)
    
    print("\n" + "-"*80)
    print("Weighted Score Rankings:")
    print("-"*80)
    
    for i, score_info in enumerate(scores, 1):
        print(f"\n{i}. {score_info['strategy']}")
        print(f"   Weighted Score: {score_info['weighted_score']:.4f}")
        print("   Component Scores:")
        for metric, value in score_info['metrics'].items():
            contribution = value * weights[metric]
            print(f"     - {metric}: {value:.3f} (contributes {contribution:.3f})")
    
    print("\n" + "="*80)
    print(f"RECOMMENDED STRATEGY: {scores[0]['strategy']}")
    print(f"Weighted Score: {scores[0]['weighted_score']:.4f}")
    print("="*80)
    
    return scores[0]['strategy']

def main():
    parser = argparse.ArgumentParser(description='Replot baseline class balancing results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to results directory containing all_summaries.json')
    parser.add_argument('--thesis_figure', action='store_true',
                       help='Create clean figure for thesis publication')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    all_summaries, comparison_df = load_results(args.results_dir)
    
    print(f"Loaded {len(all_summaries)} strategy results")
    
    # Create plots
    save_dir = args.results_dir
    
    # Create comprehensive comparison
    comprehensive_path = plot_comprehensive_comparison(all_summaries, save_dir)
    
    # Create thesis figure if requested
    if args.thesis_figure:
        thesis_path = plot_thesis_figure(all_summaries, save_dir)
    
    # Analyze best strategy
    best_strategy = analyze_best_strategy(all_summaries)
    
    # Display comparison table
    if comparison_df is not None:
        print("\n" + "="*80)
        print("STRATEGY COMPARISON TABLE")
        print("="*80)
        print(comparison_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("REPLOTTING COMPLETE")
    print("="*80)
    print(f"Results saved to: {save_dir}")
    
    return all_summaries, best_strategy

if __name__ == "__main__":
    all_summaries, best_strategy = main()