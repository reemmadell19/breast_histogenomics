import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_predictions_by_rs_score(csv_path, output_dir='external_validation'):
    """
    Analyze model predictions across different RS score ranges
    to identify where the model fails
    """
    
    # Load predictions
    print(f"Loading predictions from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Create RS score bins for analysis (5-point ranges)
    # RS scores typically range from 0-100, with 25 as the threshold
    rs_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]
    rs_labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '50+']
    
    df['rs_bin'] = pd.cut(df['RS_score'], bins=rs_bins, labels=rs_labels, include_lowest=True)
    
    # Create binary prediction accuracy
    df['correct'] = df['ensemble_correct'].astype(int)
    
    # ========== ANALYSIS 1: Accuracy by RS Score Range ==========
    print("\n" + "="*60)
    print("ACCURACY BY RS SCORE RANGE")
    print("="*60)
    
    accuracy_by_rs = df.groupby('rs_bin').agg({
        'correct': 'mean',
        'RS_score': 'count',
        'true_label': lambda x: sum(x==1)/len(x)  # Proportion of high-risk in each bin
    }).round(3)
    accuracy_by_rs.columns = ['Accuracy', 'N_Samples', 'Prop_HighRisk']
    
    print(accuracy_by_rs)
    
    # ========== ANALYSIS 2: Confusion by RS Score (especially around threshold) ==========
    print("\n" + "="*60)
    print("DETAILED ANALYSIS AROUND THRESHOLD (RS=25)")
    print("="*60)
    
    # Focus on borderline cases (RS 20-30)
    borderline_df = df[(df['RS_score'] >= 20) & (df['RS_score'] <= 30)]
    
    print(f"\nBorderline cases (RS 20-30): {len(borderline_df)} samples")
    print(f"Accuracy in borderline region: {borderline_df['correct'].mean():.3f}")
    
    # Break down by exact RS scores near threshold
    near_threshold = df[(df['RS_score'] >= 23) & (df['RS_score'] <= 27)]
    if len(near_threshold) > 0:
        print("\nExact RS scores near threshold:")
        for rs in sorted(near_threshold['RS_score'].unique()):
            rs_samples = near_threshold[near_threshold['RS_score'] == rs]
            acc = rs_samples['correct'].mean()
            n = len(rs_samples)
            print(f"  RS={rs}: {n} samples, Accuracy={acc:.2f}")
    
    # ========== SEPARATE PLOTS ==========
    print("\nGenerating plots...")
    
    # Prepare data for plots
    misclassified = df[df['correct'] == 0]
    false_pos = misclassified[misclassified['true_label'] == 0]  # Predicted high, actually low
    false_neg = misclassified[misclassified['true_label'] == 1]  # Predicted low, actually high
    
    # ========== PLOT 1: Accuracy by RS Range ==========
    fig1, ax = plt.subplots(figsize=(12, 7))
    
    x_pos = np.arange(len(accuracy_by_rs))
    
    # Use a professional single color scheme
    bars = ax.bar(x_pos, accuracy_by_rs['Accuracy'], color='#2563eb', alpha=0.8, edgecolor='#1e40af', linewidth=1.5)
    ax.set_xlabel('RS Score Range', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title('Model Accuracy by RS Score Range', fontsize=15, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rs_labels, rotation=45, ha='right')
    ax.axhline(y=0.75, color='blue', linestyle='--', alpha=0.5, linewidth=2, label='Target (0.75)')
    ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add accuracy percentage and sample size on bars
    for bar, (idx, row) in zip(bars, accuracy_by_rs.iterrows()):
        height = bar.get_height()
        accuracy_pct = row['Accuracy'] * 100
        n_samples = int(row['N_Samples'])
        # Show accuracy boldly, n smaller
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{accuracy_pct:.1f}%\n(n={n_samples})', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rs_accuracy_by_range.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/rs_accuracy_by_range.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    
    # ========== PLOT 1B: Line plot version of accuracy by RS range ==========
    fig1b, ax = plt.subplots(figsize=(12, 7))
    
    x_pos = np.arange(len(accuracy_by_rs))
    accuracies = accuracy_by_rs['Accuracy'].values
    
    # Create line plot with markers
    line = ax.plot(x_pos, accuracies, 
                   marker='o', markersize=10, linewidth=2.5,
                   color='#1e40af', markerfacecolor='#2563eb', 
                   markeredgecolor='#1e40af', markeredgewidth=2,
                   label='Model Accuracy', zorder=5)
    
    # Create smooth shaded area with color gradient based on accuracy
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create a custom colormap for the fill
    # We'll create segments for the fill area
    for i in range(len(x_pos) - 1):
        x_fill = np.linspace(x_pos[i], x_pos[i+1], 100)
        y_fill = np.interp(x_fill, [x_pos[i], x_pos[i+1]], [accuracies[i], accuracies[i+1]])
        
        # Color based on the accuracy at each point
        for j in range(len(x_fill) - 1):
            acc_at_point = y_fill[j]
            if acc_at_point >= 0.6:
                color = '#22c55e'  # Green
                alpha = 0.25
            else: 
                color = '#dc2626'  # Red
                alpha = 0.25
            
            ax.fill_between([x_fill[j], x_fill[j+1]], 0, [y_fill[j], y_fill[j+1]], 
                           color=color, alpha=alpha, edgecolor='none')
    
    # Add accuracy values at each point
    for x, (idx, row) in zip(x_pos, accuracy_by_rs.iterrows()):
        accuracy_pct = row['Accuracy'] * 100
        n_samples = int(row['N_Samples'])
        # Show accuracy above the point
        ax.text(x, row['Accuracy'] + 0.03,
                f'{accuracy_pct:.1f}%', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Show sample size below the point
        ax.text(x, row['Accuracy'] - 0.03,
                f'n={n_samples}', 
                ha='center', va='top', fontsize=9, style='italic', alpha=0.7)
    

    # Formatting
    ax.set_xlabel('RS Score Range', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title('Model Accuracy by RS Score Range (External)', fontsize=15, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rs_labels, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rs_accuracy_by_range_line.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/rs_accuracy_by_range_line.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    
    # ========== PLOT 2: Scatter plot of predictions vs RS scores ==========
    fig2, ax = plt.subplots(figsize=(10, 8))
    
    # Color by correctness
    correct_mask = df['correct'] == 1
    ax.scatter(df.loc[correct_mask, 'RS_score'], 
               df.loc[correct_mask, 'ensemble_prob_class1'],
               alpha=0.6, c='green', label='Correct', s=50)
    ax.scatter(df.loc[~correct_mask, 'RS_score'], 
               df.loc[~correct_mask, 'ensemble_prob_class1'],
               alpha=0.6, c='red', label='Incorrect', s=50)
    
    # Add threshold lines with shaded regions
    ax.axvline(x=25, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='RS=25 threshold')
    ax.axhline(y=0.5, color='purple', linestyle='--', alpha=0.7, linewidth=2, label='Prob=0.5 threshold')
    
    # Add shaded quadrants
    ax.fill_between([0, 25], 0.5, 1, alpha=0.1, color='red', label='False Positive Region')
    ax.fill_between([25, 50], 0, 0.5, alpha=0.1, color='orange', label='False Negative Region')
    
    ax.set_xlabel('Actual RS Score', fontsize=13)
    ax.set_ylabel('Predicted Probability (High Risk)', fontsize=13)
    ax.set_title('Predictions vs RS Scores', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max(df['RS_score'].max() + 2, 50)])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rs_scatter_predictions.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/rs_scatter_predictions.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    
    # ========== PLOT 3: Misclassification distribution ==========
    fig3, ax = plt.subplots(figsize=(10, 7))
    
    bins_misc = np.arange(0, 55, 5)
    n_fp, bins_fp, patches_fp = ax.hist(false_pos['RS_score'], bins=bins_misc, alpha=0.7, 
                                        label=f'False Positives (n={len(false_pos)})', 
                                        color='orange', edgecolor='black', linewidth=1.5)
    n_fn, bins_fn, patches_fn = ax.hist(false_neg['RS_score'], bins=bins_misc, alpha=0.7,
                                        label=f'False Negatives (n={len(false_neg)})', 
                                        color='purple', edgecolor='black', linewidth=1.5)
    
    ax.axvline(x=25, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
    ax.text(25.5, ax.get_ylim()[1]*0.9, 'RS=25\nThreshold', fontsize=11, color='red', fontweight='bold')
    
    ax.set_xlabel('RS Score', fontsize=13)
    ax.set_ylabel('Number of Misclassifications', fontsize=13)
    ax.set_title('Distribution of Misclassified Cases', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rs_misclassification_dist.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/rs_misclassification_dist.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    
    # ========== PLOT 4: Performance by RS quartiles ==========
    fig4, ax = plt.subplots(figsize=(10, 7))
    
    # Create quartiles
    df['rs_quartile'] = pd.qcut(df['RS_score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    metrics_by_quartile = df.groupby('rs_quartile').agg({
        'correct': 'mean',
        'ensemble_prob_class1': 'mean',
        'RS_score': ['mean', 'min', 'max', 'count']
    }).round(3)
    
    x_pos = np.arange(4)
    width = 0.35
    
    # Create dual y-axis
    ax2 = ax.twinx()
    
    # Plot accuracy bars
    bars1 = ax.bar(x_pos - width/2, metrics_by_quartile['correct']['mean'], 
                   width, label='Accuracy', color='skyblue', edgecolor='navy', linewidth=1.5)
    
    # Plot mean probability bars
    bars2 = ax2.bar(x_pos + width/2, metrics_by_quartile['ensemble_prob_class1']['mean'],
                    width, label='Mean Pred Prob', color='coral', edgecolor='darkred', linewidth=1.5)
    
    ax.set_xlabel('RS Score Quartile', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13, color='skyblue')
    ax2.set_ylabel('Mean Predicted Probability', fontsize=13, color='coral')
    ax.set_title('Performance by RS Score Quartiles', fontsize=15, fontweight='bold')
    ax.set_xticks(x_pos)
    
    # Create quartile labels with ranges
    quartile_labels = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_data = metrics_by_quartile.loc[q, 'RS_score']
        label = f"{q}\n({q_data['min']:.0f}-{q_data['max']:.0f})\nn={int(q_data['count'])}"
        quartile_labels.append(label)
    
    ax.set_xticklabels(quartile_labels)
    ax.set_ylim([0, 1])
    ax2.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add legends
    ax.legend(loc='upper left', fontsize=11)
    ax2.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rs_quartile_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/rs_quartile_performance.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("\n" + "="*60)
    print("PLOTS SAVED")
    print("="*60)
    print(f"All plots saved to: {output_dir}/")
    print("  1. rs_accuracy_by_range.png/.pdf - Accuracy by RS score ranges")
    print("  2. rs_scatter_predictions.png/.pdf - Scatter plot of all predictions")
    print("  3. rs_misclassification_dist.png/.pdf - Distribution of errors")
    print("  4. rs_quartile_performance.png/.pdf - Performance by quartiles")
    
    # ========== DETAILED FAILURE ANALYSIS ==========
    print("\n" + "="*60)
    print("FAILURE ANALYSIS SUMMARY")
    print("="*60)
    
    # Identify problematic RS ranges
    problem_ranges = accuracy_by_rs[accuracy_by_rs['Accuracy'] < 0.7]
    if len(problem_ranges) > 0:
        print("\nProblematic RS ranges (accuracy < 70%):")
        for idx, row in problem_ranges.iterrows():
            print(f"  {idx}: Accuracy={row['Accuracy']:.2f}, N={int(row['N_Samples'])}")
    
    # False negative analysis (most critical for cancer screening)
    false_negs = df[(df['true_label'] == 1) & (df['ensemble_pred'] == 0)]
    if len(false_negs) > 0:
        print(f"\nFalse Negatives (missed high-risk): {len(false_negs)}")
        print(f"  Mean RS score: {false_negs['RS_score'].mean():.1f}")
        print(f"  RS range: {false_negs['RS_score'].min():.0f} - {false_negs['RS_score'].max():.0f}")
        
        # Most concerning false negatives (highest RS scores that were missed)
        worst_fn = false_negs.nlargest(5, 'RS_score')[['slide_id', 'RS_score', 'ensemble_prob_class1']]
        if len(worst_fn) > 0:
            print("\n  Most concerning missed cases (highest RS):")
            for _, row in worst_fn.iterrows():
                print(f"    Slide {row['slide_id']}: RS={row['RS_score']:.0f}, Pred_prob={row['ensemble_prob_class1']:.3f}")
    
    # False positive analysis
    false_pos_full = df[(df['true_label'] == 0) & (df['ensemble_pred'] == 1)]
    if len(false_pos_full) > 0:
        print(f"\nFalse Positives (incorrectly flagged as high-risk): {len(false_pos_full)}")
        print(f"  Mean RS score: {false_pos_full['RS_score'].mean():.1f}")
        print(f"  RS range: {false_pos_full['RS_score'].min():.0f} - {false_pos_full['RS_score'].max():.0f}")
    
    # Statistical test: Is accuracy different near threshold?
    near_threshold = df[df['RS_score'].between(20, 30)]
    far_from_threshold = df[(df['RS_score'] < 15) | (df['RS_score'] > 35)]
    
    if len(near_threshold) > 0 and len(far_from_threshold) > 0:
        near_acc = near_threshold['correct'].mean()
        far_acc = far_from_threshold['correct'].mean()
        
        # Chi-square test
        from scipy.stats import chi2_contingency
        contingency = [[near_threshold['correct'].sum(), len(near_threshold) - near_threshold['correct'].sum()],
                       [far_from_threshold['correct'].sum(), len(far_from_threshold) - far_from_threshold['correct'].sum()]]
        chi2, p_value, _, _ = chi2_contingency(contingency)
        
        print(f"\nThreshold proximity analysis:")
        print(f"  Accuracy near threshold (RS 20-30): {near_acc:.3f}")
        print(f"  Accuracy far from threshold: {far_acc:.3f}")
        print(f"  Statistical difference (p-value): {p_value:.4f}")
        if p_value < 0.05:
            print("  → Significant difference in accuracy near vs far from threshold")
    
    # Save detailed analysis
    analysis_summary = {
        'accuracy_by_rs_range': accuracy_by_rs.to_dict(),
        'total_misclassified': len(misclassified),
        'false_positives': len(false_pos),
        'false_negatives': len(false_neg),
        'borderline_accuracy': borderline_df['correct'].mean() if len(borderline_df) > 0 else None,
        'worst_accuracy_range': problem_ranges.index[0] if len(problem_ranges) > 0 else None
    }
    
    # Save to CSV for reference
    accuracy_by_rs.to_csv(f'{output_dir}/accuracy_by_rs_range.csv')
    print(f"\nDetailed accuracy table saved to {output_dir}/accuracy_by_rs_range.csv")
    
    return analysis_summary

# Run the analysis
if __name__ == "__main__":
    csv_file = "external_validation/ecu_updated/h-optimus_attention_20250914_231455_20250918_155601/ecu_all_predictions_detailed.csv"
    analyze_predictions_by_rs_score(csv_file)