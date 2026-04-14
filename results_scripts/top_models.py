import numpy as np
import pandas as pd

# Your 5-fold metric values for each model
# AUROC values
hoptimus_acmil_aurocs = [0.7668498168498169, 0.7053113553113554, 0.8179487179487179, 
                         0.7415750915750916, 0.7437728937728937]
hoptimus_abmil_aurocs = [0.724908424908425, 0.810989010989011, 0.7463369963369964, 
                         0.7880952380952381, 0.6961538461538461]

# Sensitivity values
hoptimus_acmil_sens = [0.03333333333333333, 0.10, 0.10, 0.03333333333333333, 0.8333333333333334]
hoptimus_abmil_sens = [0.5666666666666667, 0.4666666666666667, 0.03333333333333333, 0.6333333333333333, 0.7666666666666667]

# MCC values
hoptimus_acmil_mcc = [0.16956407262017123, 0.13265963173737344, 0.17553320637435493, 0.16956407262017123, 0.2745083758021893]
hoptimus_abmil_mcc = [0.21389701552834545, 0.3124852352007224, 0.10037432122170813, 0.3337442817169966, 0.2320187562342061]

# Balanced Accuracy values
hoptimus_acmil_bal_acc = [0.5166666666666667, 0.5362637362637362, 0.5417582417582417, 0.5166666666666667, 0.6968864468864469]
hoptimus_abmil_bal_acc = [0.6432234432234432, 0.6701465201465202, 0.513919413919414, 0.7122710622710622, 0.6663003663003664]

def bootstrap_analysis(metric1_folds, metric2_folds, metric_name, n_bootstrap=10000, alpha=0.05):
    """
    Perform bootstrap-based statistical analysis for small sample sizes
    """
    print(f"\n{'='*60}")
    print(f"{metric_name} - Bootstrap Analysis")
    print('='*60)
    
    # Calculate basic statistics
    mean1 = np.mean(metric1_folds)
    std1 = np.std(metric1_folds, ddof=1)
    mean2 = np.mean(metric2_folds)
    std2 = np.std(metric2_folds, ddof=1)
    observed_diff = mean1 - mean2
    
    print(f"\nDescriptive Statistics:")
    print(f"  ACMIL: {mean1:.4f} ± {std1:.4f}")
    print(f"  ABMIL: {mean2:.4f} ± {std2:.4f}")
    print(f"  Observed difference: {observed_diff:.4f}")
    
    # Bootstrap procedure
    np.random.seed(42)  # For reproducibility
    n_folds = len(metric1_folds)
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_folds, n_folds, replace=True)
        sample1 = [metric1_folds[i] for i in indices]
        sample2 = [metric2_folds[i] for i in indices]
        bootstrap_diffs.append(np.mean(sample1) - np.mean(sample2))
    
    # Calculate confidence intervals
    ci_lower = np.percentile(bootstrap_diffs, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
    
    # Calculate p-value using the bootstrap distribution
    # Two-tailed test: proportion of bootstrap samples more extreme than observed
    if observed_diff > 0:
        p_value = 2 * np.mean([d <= 0 for d in bootstrap_diffs])
    else:
        p_value = 2 * np.mean([d >= 0 for d in bootstrap_diffs])
    
    # Ensure p-value doesn't exceed 1
    p_value = min(p_value, 1.0)
    
    # Check if confidence interval contains zero
    contains_zero = (ci_lower <= 0 <= ci_upper)
    is_significant = not contains_zero
    
    # Calculate effect size using bootstrap standard error
    bootstrap_se = np.std(bootstrap_diffs)
    effect_size = observed_diff / bootstrap_se if bootstrap_se > 0 else 0
    
    print(f"\nBootstrap Results ({n_bootstrap:,} iterations):")
    print(f"  95% CI for difference: [{ci_lower:.4f}, {ci_upper:.4f}]")  # Fixed here!
    print(f"  Bootstrap p-value: {p_value:.4f}")
    print(f"  CI contains zero: {contains_zero}")
    print(f"  Effect size (d): {effect_size:.3f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if is_significant:
        print(f"  ✓ Statistically significant difference (p = {p_value:.4f} < {alpha})")
        if observed_diff > 0:
            print(f"    → ACMIL performs better than ABMIL")
        else:
            print(f"    → ABMIL performs better than ACMIL")
    else:
        print(f"  ✗ No significant difference (p = {p_value:.4f} ≥ {alpha})")
        print(f"    → Models perform similarly")
    
    return {
        'metric_name': metric_name,
        'mean1': mean1,
        'std1': std1,
        'mean2': mean2,
        'std2': std2,
        'observed_diff': observed_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'is_significant': is_significant,
        'effect_size': effect_size,
        'contains_zero': contains_zero
    }

# Run bootstrap analysis for all metrics
print("\n" + "="*60)
print("MODEL COMPARISON: HOPTIMUS ACMIL vs ABMIL")
print("Bootstrap-Based Statistical Testing (n=5 folds)")
print("="*60)

results = {}

# Analyze each metric
results['auroc'] = bootstrap_analysis(
    hoptimus_acmil_aurocs, 
    hoptimus_abmil_aurocs, 
    "AUROC"
)

results['sensitivity'] = bootstrap_analysis(
    hoptimus_acmil_sens, 
    hoptimus_abmil_sens, 
    "Sensitivity"
)

results['mcc'] = bootstrap_analysis(
    hoptimus_acmil_mcc, 
    hoptimus_abmil_mcc, 
    "Matthews Correlation Coefficient (MCC)"
)

results['balanced_accuracy'] = bootstrap_analysis(
    hoptimus_acmil_bal_acc, 
    hoptimus_abmil_bal_acc, 
    "Balanced Accuracy"
)

# Multiple Testing Correction (Bonferroni)
print("\n" + "="*60)
print("MULTIPLE TESTING CORRECTION")
print("="*60)

n_tests = len(results)
bonferroni_alpha = 0.05 / n_tests

print(f"\nBonferroni-corrected significance level: α = {bonferroni_alpha:.4f}")
print("\nAdjusted results:")

for metric_name, res in results.items():
    is_sig_bonferroni = res['p_value'] < bonferroni_alpha
    print(f"\n{res['metric_name']}:")
    print(f"  p-value: {res['p_value']:.4f}")
    print(f"  Significant (α=0.05): {res['is_significant']}")
    print(f"  Significant (Bonferroni α={bonferroni_alpha:.4f}): {is_sig_bonferroni}")

# Summary Table
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)

# Create DataFrame for better visualization
summary_data = []
for metric_name, res in results.items():
    summary_data.append({
        'Metric': res['metric_name'],
        'ACMIL Mean': f"{res['mean1']:.4f}",
        'ABMIL Mean': f"{res['mean2']:.4f}",
        'Difference': f"{res['observed_diff']:.4f}",
        '95% CI': f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]",
        'p-value': f"{res['p_value']:.4f}",
        'Significant': '✓' if res['is_significant'] else '✗'
    })

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

# Overall Conclusion
print("\n" + "="*60)
print("OVERALL CONCLUSION")
print("="*60)

significant_metrics = [name for name, res in results.items() if res['is_significant']]
significant_bonferroni = [name for name, res in results.items() if res['p_value'] < bonferroni_alpha]

if significant_metrics:
    print(f"\nSignificant differences found in: {', '.join(significant_metrics)}")
    if significant_bonferroni:
        print(f"After Bonferroni correction: {', '.join(significant_bonferroni)}")
    else:
        print("However, no differences remain significant after Bonferroni correction.")
else:
    print("\nNo significant differences found between ACMIL and ABMIL models.")

print("\nNote: With only 5 folds, statistical power is limited. Consider:")
print("  • Using 10-fold CV if computationally feasible")
print("  • Reporting effect sizes alongside p-values")
print("  • Interpreting results with appropriate caution")

# Export to CSV
results_for_csv = []
for metric_name, res in results.items():
    results_for_csv.append({
        'Metric': res['metric_name'],
        'ACMIL_Mean': res['mean1'],
        'ACMIL_Std': res['std1'],
        'ABMIL_Mean': res['mean2'],
        'ABMIL_Std': res['std2'],
        'Mean_Difference': res['observed_diff'],
        'CI_95_Lower': res['ci_lower'],
        'CI_95_Upper': res['ci_upper'],
        'Bootstrap_P_Value': res['p_value'],
        'Effect_Size': res['effect_size'],
        'Significant_0.05': res['is_significant'],
        'Significant_Bonferroni': res['p_value'] < bonferroni_alpha
    })

df_export = pd.DataFrame(results_for_csv)
df_export.to_csv('bootstrap_statistical_comparison.csv', index=False)
print(f"\n✓ Results exported to 'bootstrap_statistical_comparison.csv'")