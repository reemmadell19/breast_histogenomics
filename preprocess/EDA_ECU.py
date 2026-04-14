"""
ECU Dataset Simple EDA Script
Follows the exact style and colors from the original comprehensive EDA script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set style for publication-quality plots - EXACT same as original
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)  # Same as original
plt.rcParams['font.size'] = 12

def create_output_directory(base_path="/cs/student/projects1/aibh/2024/elnefary/"):
    """
    Create output directory for saving plots
    """
    output_dir = os.path.join(base_path, "ecu_eda_plots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Using existing output directory: {output_dir}")
    return output_dir

def load_and_filter_ecu_data():
    """
    Load and filter ECU data following the exact pattern from original script
    """
    print("\n=== LOADING ECU DATASET ===")
    
    # Load ECU manifest
    ecu_manifest_path = "/cs/student/projects1/aibh/2024/elnefary/data/raw/ECU/ecu_manifest.csv"
    ecu_data = pd.read_csv(ecu_manifest_path)
    
    print(f"ECU manifest loaded successfully!")
    print(f"Shape: {ecu_data.shape}")
    print(f"Columns: {ecu_data.columns.tolist()}")
    
    # Check for RS scores
    if 'RS' in ecu_data.columns:
        print(f"RS scores available: {ecu_data['RS'].notna().sum()} out of {len(ecu_data)}")
    
    # Filter ECU data
    print("\n=== FILTERING ECU DATA ===")
    print(f"Original ECU dataset size: {len(ecu_data)}")
    
    if 'RS' in ecu_data.columns:
        filtered_data = ecu_data[ecu_data['RS'].notna()].copy()
        print(f"After removing missing RS scores: {len(filtered_data)}")
    else:
        filtered_data = ecu_data.copy()
        print("No RS column found - keeping all records")
    
    print(f"Final filtered ECU dataset size: {len(filtered_data)}")
    print(f"Total excluded patients: {len(ecu_data) - len(filtered_data)}")
    
    return filtered_data

def plot_age_distribution(ecu_data, output_dir):
    """
    Plot age distribution using the exact style from original
    """
    # Find age column
    age_col = None
    for col in ['age', 'Age', 'AGE', 'patient_age', 'Patient_Age']:
        if col in ecu_data.columns:
            age_col = col
            break
    
    if age_col is None:
        print("No age column found in ECU data")
        return
    
    ages = ecu_data[age_col].dropna()
    
    # Single plot for age distribution
    plt.figure(figsize=(12, 8))
    
    # Use lightgreen color as in ECU sections of original
    plt.hist(ages, bins=25, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(x=ages.mean(), color='blue', linestyle=':', linewidth=2, label=f'Mean: {ages.mean():.1f}')
    plt.axvline(x=ages.median(), color='green', linestyle=':', linewidth=2, label=f'Median: {ages.median():.1f}')
    
    plt.xlabel('Age (years)')
    plt.ylabel('Frequency')
    plt.title('ECU Dataset - Age Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box - same style as original
    stats_text = (f'n = {len(ages)}\n'
                 f'Mean: {ages.mean():.1f}\n'
                 f'Std: {ages.std():.1f}\n'
                 f'Min: {ages.min():.0f}\n'
                 f'Max: {ages.max():.0f}')
    
    plt.text(0.65, 0.95, stats_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'ecu_age_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.show()

def plot_risk_distribution_pie(ecu_data, output_dir):
    """
    Plot risk distribution pie chart using exact colors from original
    """
    if 'RS' not in ecu_data.columns:
        print("No RS column found")
        return
    
    # Create risk categories
    ecu_data['risk_category'] = ecu_data['RS'].apply(
        lambda x: 'High Risk' if x >= 25 else 'Low Risk')
    
    # Calculate statistics
    high_risk_count = (ecu_data['RS'] >= 25).sum()
    low_risk_count = len(ecu_data) - high_risk_count
    imbalance_ratio = low_risk_count / high_risk_count if high_risk_count > 0 else float('inf')
    
    # Single pie chart with exact colors from original (skyblue for low, lightcoral for high)
    plt.figure(figsize=(12, 8))
    
    counts = ecu_data['risk_category'].value_counts()
    colors = ['skyblue', 'lightcoral']  # Exact colors from original
    
    plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', 
           colors=colors, startangle=90)
    plt.title('ECU Class Distribution (n=212)', fontsize=14)
    
    # Add count labels
    plt.text(0, -1.2, f'Total patients: {len(ecu_data)}\nImbalance Ratio (Low:High): {imbalance_ratio:.2f}:1', 
            ha='center', fontsize=11)
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'ecu_risk_distribution_pie.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.show()
    
    # Print statistics - same format as original
    print("\n=== CLASS IMBALANCE ANALYSIS ===")
    print(f"\nECU:")
    print(f"  Low Risk (RS < 25): {low_risk_count} ({low_risk_count/len(ecu_data)*100:.1f}%)")
    print(f"  High Risk (RS ≥ 25): {high_risk_count} ({high_risk_count/len(ecu_data)*100:.1f}%)")
    print(f"  Imbalance Ratio (Low:High): {imbalance_ratio:.2f}:1")

def plot_race_distribution(ecu_data, output_dir):
    """
    Plot race distribution pie chart
    """
    print("\nCreating Race Distribution Pie Chart...")
    
    # Find race column - try multiple possible column names
    race_col = None
    for col in ['race', 'Race', 'RACE', 'ethnicity', 'Ethnicity', 'Race/Ethnicity']:
        if col in ecu_data.columns:
            race_col = col
            print(f"Found race column: '{race_col}'")
            break
    
    if race_col is None:
        print("WARNING: No race/ethnicity column found in data")
        print(f"Available columns: {ecu_data.columns.tolist()}")
        return
    
    # Get race data and drop missing values
    races = ecu_data[race_col].dropna()
    
    if len(races) == 0:
        print("WARNING: No valid race data found (all values are missing)")
        return
    
    print(f"Found {len(races)} patients with race data out of {len(ecu_data)} total")
    
    # Get value counts
    race_counts = races.value_counts()
    print(f"Original race categories found: {race_counts.index.tolist()}")
    
    # Fix label names - replace codes with full names
    # Handle both numeric codes and string variations
    def fix_race_label(x):
        x_str = str(x).strip()
        # Check for white variations
        if x_str in ['01 White', '01', '1', 'W', 'w', 'White', 'white', '1.0']:
            return 'White'
        # Check for black variations  
        elif x_str in ['02 Black', '02', '2', 'B', 'b', 'Black', 'black', '2.0']:
            return 'Black'
        # Check for other variations
        elif x_str.lower() in ['others', 'other', '3', '03', '3.0']:
            return 'Others'
        else:
            return 'Others'  # Group all unknowns as Others
    
    race_counts.index = race_counts.index.map(fix_race_label)
    
    # Combine duplicates after renaming
    race_counts = race_counts.groupby(race_counts.index).sum()
    print(f"Cleaned race categories: {race_counts.to_dict()}")
    
    # Create the pie chart
    plt.figure(figsize=(12, 8))
    
    # Define specific colors to match the desired style
    color_map = {
        'White': '#ff9999',  # Light salmon/pink color
        'Black': '#66b3ff',  # Light blue color  
        'Others': '#90ee90'  # Light green color
    }
    
    # Get colors in the right order
    colors = [color_map.get(race, '#dddddd') for race in race_counts.index]
    
    # Create pie chart with labels showing both name and count
    labels = [f'{race}\n({count} patients)' for race, count in zip(race_counts.index, race_counts.values)]
    
    plt.pie(race_counts.values, labels=labels, autopct='%1.1f%%',
           colors=colors, startangle=90, textprops={'fontsize': 11})
    plt.title('ECU Race/Ethnicity Distribution', fontsize=14, fontweight='bold')
    
    plt.text(0, -1.3, f'Total patients with race data: {len(races)}', 
            ha='center', fontsize=11)
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'ecu_race_distribution_pie.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.show()
    
    # Print statistics
    print(f"\nRace/Ethnicity Distribution:")
    for race, count in race_counts.items():
        print(f"  {race}: {count} ({count/len(races)*100:.1f}%)")

def plot_rs_distribution(ecu_data, output_dir):
    """
    Plot RS distribution histogram - matching original style exactly
    """
    if 'RS' not in ecu_data.columns:
        return
    
    scores = ecu_data['RS']
    
    plt.figure(figsize=(12, 8))
    
    # Use lightgreen for ECU as in original
    plt.hist(scores, bins=25, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(x=25, color='red', linestyle='--', linewidth=2, label='RS=25')
    plt.xlabel('Recurrence Score')
    plt.ylabel('Frequency')
    plt.title('ECU (n=' + str(len(ecu_data)) + ')', fontsize=12, fontweight='bold')
    
    # Add statistics text box - exact same position and style as original
    stats_text = f'Mean: {scores.mean():.1f}\nStd: {scores.std():.1f}\nMedian: {scores.median():.1f}'
    plt.text(0.65, 0.95, stats_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'ecu_rs_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.show()

def plot_boundary_analysis(ecu_data, output_dir):
    """
    Analyze RS distribution around clinical threshold - exact same as original
    """
    if 'RS' not in ecu_data.columns:
        return
    
    scores = ecu_data['RS']
    boundary_range = 10  # ±10 around RS=25
    
    # Define boundary regions
    below_boundary = scores[scores < (25 - boundary_range)]
    near_boundary = scores[(scores >= (25 - boundary_range)) & (scores <= (25 + boundary_range))]
    above_boundary = scores[scores > (25 + boundary_range)]
    
    plt.figure(figsize=(12, 8))
    
    plt.hist(scores, bins=30, alpha=0.5, color='lightgreen', edgecolor='black', label='All scores')
    plt.hist(near_boundary, bins=15, alpha=0.8, color='orange', edgecolor='black', 
            label=f'Boundary (15-35)')
    plt.axvline(x=25, color='red', linestyle='--', linewidth=2, label='RS=25')
    plt.axvspan(15, 35, alpha=0.2, color='yellow')
    
    plt.xlabel('Recurrence Score')
    plt.ylabel('Frequency')
    plt.title('ECU - Boundary Analysis', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics - exact same format
    stats_text = (f'Below 15: {len(below_boundary)} ({len(below_boundary)/len(scores)*100:.1f}%)\n'
                 f'15-35: {len(near_boundary)} ({len(near_boundary)/len(scores)*100:.1f}%)\n'
                 f'Above 35: {len(above_boundary)} ({len(above_boundary)/len(scores)*100:.1f}%)')
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'ecu_boundary_analysis.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.show()
    
    # Print boundary statistics - exact same format
    print("\n=== BOUNDARY REGION ANALYSIS (RS 15-35) ===")
    print(f"\nECU:")
    print(f"  Patients in boundary region (15-35): {len(near_boundary)} "
          f"({len(near_boundary)/len(scores)*100:.1f}%)")
    print(f"  Mean RS in boundary: {near_boundary.mean():.2f}")
    print(f"  Std RS in boundary: {near_boundary.std():.2f}")

def plot_dataset_characteristics_table(ecu_data, output_dir):
    """
    Create characteristics table - exact same format as original
    """
    if 'RS' in ecu_data.columns:
        scores = ecu_data['RS']
        
        char_dict = {
            'Dataset': 'ECU',
            'N': len(ecu_data),
            'Mean RS': f"{scores.mean():.1f}",
            'Std RS': f"{scores.std():.1f}",
            'Min RS': f"{scores.min():.0f}",
            'Max RS': f"{scores.max():.0f}",
            'Median RS': f"{scores.median():.1f}",
            'Low Risk (%)': f"{((scores < 25).sum()/len(scores)*100):.1f}",
            'High Risk (%)': f"{((scores >= 25).sum()/len(scores)*100):.1f}",
            'Missing RS': f"{scores.isna().sum()}"
        }
    else:
        char_dict = {
            'Dataset': 'ECU',
            'N': len(ecu_data),
            'Mean RS': 'N/A',
            'Std RS': 'N/A',
            'Min RS': 'N/A',
            'Max RS': 'N/A',
            'Median RS': 'N/A',
            'Low Risk (%)': 'N/A',
            'High Risk (%)': 'N/A',
            'Missing RS': 'N/A'
        }
    
    # Create DataFrame
    char_df = pd.DataFrame([char_dict])
    
    # Display table - exact same style
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=char_df.values,
                    colLabels=char_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.08, 0.06, 0.08, 0.08, 0.08, 0.08, 0.09, 0.1, 0.1, 0.09])
    
    # Style the table - exact same as original
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Header styling - exact same green color
    for i in range(len(char_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Dataset Characteristics Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'ecu_characteristics_table.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.show()
    
    # Print the table as well
    print("\n=== DATASET CHARACTERISTICS TABLE ===")
    print(char_df.to_string(index=False))

def main():
    """
    Main execution - following exact pattern from original
    """
    print("="*60)
    print("ECU DATASET EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Load and filter ECU data
    ecu_filtered = load_and_filter_ecu_data()
    
    # Generate all plots
    plot_age_distribution(ecu_filtered, output_dir)
    plot_risk_distribution_pie(ecu_filtered, output_dir)
    plot_race_distribution(ecu_filtered, output_dir)
    plot_rs_distribution(ecu_filtered, output_dir)
    plot_boundary_analysis(ecu_filtered, output_dir)
    plot_dataset_characteristics_table(ecu_filtered, output_dir)
    
    # Final summary - exact same format
    print("\n" + "="*60)
    print("FINAL DATASET SUMMARY")
    print("="*60)
    
    print(f"\nExternal Dataset (for testing):")
    print(f"  ECU: {len(ecu_filtered)} patients")
    
    if 'RS' in ecu_filtered.columns:
        rs_data = ecu_filtered['RS'].dropna()
        high_risk = (rs_data >= 25).sum()
        low_risk = (rs_data < 25).sum()
        print(f"\nRisk Distribution:")
        print(f"  Low Risk: {low_risk} ({low_risk/len(rs_data)*100:.1f}%)")
        print(f"  High Risk: {high_risk} ({high_risk/len(rs_data)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()