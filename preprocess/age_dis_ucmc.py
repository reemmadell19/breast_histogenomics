"""
Standalone script for UCMC Age Distribution Analysis
This script loads, filters, and analyzes age distribution in the UCMC dataset
Creates separate figures for each plot and saves them
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def create_output_directory(base_path="/cs/student/projects1/aibh/2024/elnefary/"):
    """
    Create output directory for saving plots
    """
    output_dir = os.path.join(base_path, "age_distribution_plots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Using existing output directory: {output_dir}")
    return output_dir

def load_and_filter_ucmc_data(ucmc_path):
    """
    Load and filter UCMC data to exclude patients with missing actual RS scores and missing files
    Missing file: 'UCH_BRCA_RS_19(2)'
    """
    print("="*60)
    print("LOADING AND FILTERING UCMC DATA")
    print("="*60)
    
    # Load UCMC data
    ucmc_data = pd.read_csv(ucmc_path)
    print(f"Original UCMC dataset size: {len(ucmc_data)}")
    
    # Filter out patients without actual RS scores (NaN in RS column)
    filtered_data = ucmc_data[ucmc_data['RS'].notna()].copy()
    print(f"After removing missing RS scores: {len(filtered_data)}")
    
    # Filter out patients with missing data files
    missing_file_id = 'UCH_BRCA_RS_19(2)'
    filtered_data = filtered_data[filtered_data['slide'] != missing_file_id].copy()
    print(f"Removed missing file: {missing_file_id}")
    
    print(f"Final filtered UCMC dataset size: {len(filtered_data)}")
    print(f"Total excluded patients: {len(ucmc_data) - len(filtered_data)}")
    print("="*60 + "\n")
    
    return filtered_data

def plot_basic_age_distribution(ages, output_dir):
    """
    Plot 1: Basic age distribution histogram
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(ages, bins=25, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=ages.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ages.mean():.1f}')
    plt.axvline(x=ages.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {ages.median():.1f}')
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Age Distribution - UCMC Dataset', fontweight='bold', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = (f'n = {len(ages)}\n'
                 f'Mean: {ages.mean():.1f} years\n'
                 f'Std: {ages.std():.1f} years\n'
                 f'Min: {ages.min():.0f} years\n'
                 f'Max: {ages.max():.0f} years\n'
                 f'Q1: {ages.quantile(0.25):.0f} years\n'
                 f'Q3: {ages.quantile(0.75):.0f} years')
    
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    
    # Save the figure
    filepath = os.path.join(output_dir, 'age_distribution_basic.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    
    plt.show()

def plot_age_by_risk_category(age_data, age_col, output_dir):
    """
    Plot 2: Age distribution by risk category
    """
    if 'RS' not in age_data.columns:
        print("RS column not found, skipping risk category plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    age_data['risk_category'] = age_data['RS'].apply(
        lambda x: 'High Risk (RS≥25)' if x >= 25 else 'Low Risk (RS<25)')
    
    low_risk_ages = age_data[age_data['risk_category'] == 'Low Risk (RS<25)'][age_col]
    high_risk_ages = age_data[age_data['risk_category'] == 'High Risk (RS≥25)'][age_col]
    
    plt.hist([low_risk_ages, high_risk_ages], bins=20, alpha=0.6, 
            label=[f'Low Risk (n={len(low_risk_ages)})', f'High Risk (n={len(high_risk_ages)})'], 
            color=['skyblue', 'lightcoral'], edgecolor='black')
    
    # Add mean lines
    plt.axvline(x=low_risk_ages.mean(), color='blue', linestyle=':', linewidth=2, alpha=0.7, 
               label=f'Low Risk Mean: {low_risk_ages.mean():.1f}')
    plt.axvline(x=high_risk_ages.mean(), color='red', linestyle=':', linewidth=2, alpha=0.7,
               label=f'High Risk Mean: {high_risk_ages.mean():.1f}')
    
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Age Distribution by Risk Category', fontweight='bold', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    t_stat, p_value = stats.ttest_ind(low_risk_ages, high_risk_ages)
    risk_stats = (f'Low Risk: {len(low_risk_ages)} patients\n'
                 f'  Mean: {low_risk_ages.mean():.1f} ± {low_risk_ages.std():.1f}\n'
                 f'High Risk: {len(high_risk_ages)} patients\n'
                 f'  Mean: {high_risk_ages.mean():.1f} ± {high_risk_ages.std():.1f}\n'
                 f'Difference: {abs(high_risk_ages.mean() - low_risk_ages.mean()):.1f} years\n'
                 f'T-test p-value: {p_value:.4f}')
    
    plt.text(0.98, 0.98, risk_stats, transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    
    # Save the figure
    filepath = os.path.join(output_dir, 'age_distribution_by_risk.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    
    plt.show()

def plot_age_groups(ages, output_dir):
    """
    Plot 3: Age groups distribution
    """
    plt.figure(figsize=(10, 6))
    
    age_groups = pd.cut(ages, bins=[0, 40, 50, 60, 70, 100], 
                       labels=['<40', '40-49', '50-59', '60-69', '≥70'])
    age_group_counts = age_groups.value_counts().sort_index()
    
    bars = plt.bar(age_group_counts.index, age_group_counts.values, 
                   color='skyblue', edgecolor='black', alpha=0.7, width=0.6)
    
    plt.xlabel('Age Group (years)', fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.title('Distribution by Age Groups', fontweight='bold', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add count and percentage labels on bars
    for bar, count in zip(bars, age_group_counts.values):
        percentage = count/len(ages)*100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{count}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add a horizontal line for mean count
    mean_count = age_group_counts.mean()
    plt.axhline(y=mean_count, color='red', linestyle='--', alpha=0.5, 
               label=f'Mean count: {mean_count:.0f}')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the figure
    filepath = os.path.join(output_dir, 'age_groups_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    
    plt.show()
    
    return age_group_counts

def plot_cumulative_distribution(ages, output_dir):
    """
    Plot 4: Cumulative age distribution
    """
    plt.figure(figsize=(10, 6))
    
    sorted_ages = np.sort(ages)
    cumulative = np.arange(1, len(sorted_ages) + 1) / len(sorted_ages) * 100
    
    plt.plot(sorted_ages, cumulative, linewidth=2.5, color='darkblue')
    plt.fill_between(sorted_ages, cumulative, alpha=0.3, color='skyblue')
    
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Cumulative Percentage (%)', fontsize=12)
    plt.title('Cumulative Age Distribution', fontweight='bold', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add quartile lines and labels
    quartiles = [25, 50, 75]
    colors = ['green', 'orange', 'red']
    for q, color in zip(quartiles, colors):
        age_q = ages.quantile(q/100)
        plt.axvline(x=age_q, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
        plt.axhline(y=q, color=color, linestyle=':', alpha=0.4, linewidth=1)
        
        # Add label
        label = 'Q1' if q == 25 else ('Median' if q == 50 else 'Q3')
        plt.text(age_q + 0.5, 5, f'{label}\n{age_q:.0f} yrs', 
                ha='left', fontsize=9, color=color, fontweight='bold')
    
    # Add percentile information box
    percentile_text = (f'25th percentile: {ages.quantile(0.25):.0f} years\n'
                       f'50th percentile: {ages.quantile(0.50):.0f} years\n'
                       f'75th percentile: {ages.quantile(0.75):.0f} years\n'
                       f'90th percentile: {ages.quantile(0.90):.0f} years\n'
                       f'95th percentile: {ages.quantile(0.95):.0f} years')
    
    plt.text(0.02, 0.98, percentile_text, transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    
    # Save the figure
    filepath = os.path.join(output_dir, 'age_cumulative_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    
    plt.show()

def print_statistical_summary(ages, age_data, age_col, age_group_counts):
    """
    Print comprehensive statistical summary
    """
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY")
    print("="*60)
    
    print("\n--- Age Statistics ---")
    print(f"Total patients analyzed: {len(ages)}")
    print(f"Mean age: {ages.mean():.2f} years")
    print(f"Median age: {ages.median():.2f} years")
    print(f"Standard deviation: {ages.std():.2f} years")
    print(f"Age range: {ages.min():.0f} - {ages.max():.0f} years")
    print(f"Interquartile range: {ages.quantile(0.25):.0f} - {ages.quantile(0.75):.0f} years")
    print(f"95% CI for mean: [{ages.mean() - 1.96*ages.sem():.1f}, {ages.mean() + 1.96*ages.sem():.1f}]")
    
    print("\n--- Age Group Distribution ---")
    for group, count in age_group_counts.items():
        print(f"{group} years: {count} patients ({count/len(ages)*100:.1f}%)")
    
    if 'RS' in age_data.columns:
        age_data['risk_category'] = age_data['RS'].apply(
            lambda x: 'High Risk (RS≥25)' if x >= 25 else 'Low Risk (RS<25)')
        
        low_risk_ages = age_data[age_data['risk_category'] == 'Low Risk (RS<25)'][age_col]
        high_risk_ages = age_data[age_data['risk_category'] == 'High Risk (RS≥25)'][age_col]
        
        print("\n--- Age by Risk Category ---")
        print(f"Low Risk (RS<25) patients: {len(low_risk_ages)} ({len(low_risk_ages)/len(age_data)*100:.1f}%)")
        print(f"  Mean age: {low_risk_ages.mean():.2f} years")
        print(f"  Median age: {low_risk_ages.median():.2f} years")
        print(f"  Std dev: {low_risk_ages.std():.2f} years")
        
        print(f"\nHigh Risk (RS≥25) patients: {len(high_risk_ages)} ({len(high_risk_ages)/len(age_data)*100:.1f}%)")
        print(f"  Mean age: {high_risk_ages.mean():.2f} years")
        print(f"  Median age: {high_risk_ages.median():.2f} years")
        print(f"  Std dev: {high_risk_ages.std():.2f} years")
        
        # Statistical test for age difference between risk groups
        t_stat, p_value = stats.ttest_ind(low_risk_ages, high_risk_ages)
        print("\n--- Statistical Test ---")
        print("Independent t-test for age difference between risk groups:")
        print(f"  Mean difference: {abs(high_risk_ages.mean() - low_risk_ages.mean()):.2f} years")
        print(f"  T-statistic: {t_stat:.3f}")
        print(f"  P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("  Result: Significant difference in age between risk groups (p < 0.05)")
        else:
            print("  Result: No significant difference in age between risk groups (p ≥ 0.05)")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(low_risk_ages)-1)*low_risk_ages.std()**2 + 
                              (len(high_risk_ages)-1)*high_risk_ages.std()**2) / 
                             (len(low_risk_ages) + len(high_risk_ages) - 2))
        cohens_d = abs(high_risk_ages.mean() - low_risk_ages.mean()) / pooled_std
        print(f"  Cohen's d (effect size): {cohens_d:.3f}")
        
        if cohens_d < 0.2:
            effect = "negligible"
        elif cohens_d < 0.5:
            effect = "small"
        elif cohens_d < 0.8:
            effect = "medium"
        else:
            effect = "large"
        print(f"  Effect size interpretation: {effect}")

def analyze_age_distribution(ucmc_filtered, output_dir):
    """
    Comprehensive age distribution analysis for filtered UCMC dataset
    """
    print("UCMC AGE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Check if age column exists
    age_columns = ['age', 'Age', 'AGE', 'patient_age', 'Patient_Age']
    age_col = None
    
    for col in age_columns:
        if col in ucmc_filtered.columns:
            age_col = col
            break
    
    if age_col is None:
        print("ERROR: No age column found in UCMC data")
        print(f"Available columns: {ucmc_filtered.columns.tolist()}")
        return None
    
    # Filter out any missing age values
    age_data = ucmc_filtered[ucmc_filtered[age_col].notna()].copy()
    ages = age_data[age_col]
    
    print(f"Age column found: '{age_col}'")
    print(f"Patients with age data: {len(age_data)} out of {len(ucmc_filtered)}")
    print(f"Missing age values: {len(ucmc_filtered) - len(age_data)}")
    print("="*60 + "\n")
    
    print("Generating plots...")
    print("-"*40)
    
    # Generate all plots separately
    plot_basic_age_distribution(ages, output_dir)
    plot_age_by_risk_category(age_data, age_col, output_dir)
    age_group_counts = plot_age_groups(ages, output_dir)
    plot_cumulative_distribution(ages, output_dir)
    
    print("-"*40)
    print(f"All plots saved to: {output_dir}")
    
    # Print statistical summary
    print_statistical_summary(ages, age_data, age_col, age_group_counts)
    
    return age_data

def main():
    """
    Main execution function
    """
    # Define the path to UCMC data
    ucmc_path = "/cs/student/projects1/aibh/2024/elnefary/data/raw/UCMC/uch_brca_complete.csv"
    
    print("\n" + "="*60)
    print("UCMC AGE DISTRIBUTION ANALYSIS - STANDALONE SCRIPT")
    print("="*60 + "\n")
    
    # Create output directory for plots
    output_dir = create_output_directory()
    
    # Load and filter UCMC data
    ucmc_filtered = load_and_filter_ucmc_data(ucmc_path)
    
    # Analyze age distribution
    age_data = analyze_age_distribution(ucmc_filtered, output_dir)
    
    if age_data is not None:
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"\nAll plots have been saved to:")
        print(f"  {output_dir}")
        print("\nFiles created:")
        print("  - age_distribution_basic.png")
        print("  - age_distribution_by_risk.png")
        print("  - age_groups_distribution.png")
        print("  - age_cumulative_distribution.png")
    else:
        print("\nScript completed with warnings - check for missing age column.")

if __name__ == "__main__":
    main()