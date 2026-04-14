import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def plot_combined_rs_distribution():
    """
    Load datasets and plot the combined RS distribution
    """
    # Load BCR-NET data
    bcr_net_path = "/cs/student/projects1/aibh/2024/elnefary/data/raw/BCR_NET/Labels.xlsx"
    bcr_net_data = pd.read_excel(bcr_net_path)
    
    # Load UCMC data
    ucmc_path = "/cs/student/projects1/aibh/2024/elnefary/data/raw/UCMC/uch_brca_complete.csv"
    ucmc_data = pd.read_csv(ucmc_path)
    
    # Filter UCMC data (only patients with actual RS scores AND remove missing file)
    ucmc_filtered = ucmc_data[ucmc_data['RS'].notna()].copy()
    
    # Remove missing file: 'UCH_BRCA_RS_19(2)'
    missing_file_id = 'UCH_BRCA_RS_19(2)'
    ucmc_filtered = ucmc_filtered[ucmc_filtered['slide'] != missing_file_id].copy()
    
    print(f"UCMC: Removed {len(ucmc_data) - len(ucmc_filtered)} patients (missing RS or files)")
    print(f"UCMC final size: {len(ucmc_filtered)}")
    
    # Filter BCR-NET data (remove missing files)
    missing_ids = ['1', 'd00104-101', 'd00104-63']
    bcr_filtered = bcr_net_data[~bcr_net_data['Image ID'].astype(str).isin(missing_ids)].copy()
    
    # Extract RS scores
    ucmc_scores = ucmc_filtered['RS'].values
    bcr_scores = bcr_filtered['score'].values
    combined_scores = np.concatenate([ucmc_scores, bcr_scores])
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Individual dataset histograms
    axes[0,0].hist(ucmc_scores, bins=25, alpha=0.7, color='skyblue', 
                   edgecolor='black', label=f'UCMC (n={len(ucmc_scores)})')
    axes[0,0].axvline(x=25, color='red', linestyle='--', linewidth=2, 
                     label='Clinical Threshold')
    axes[0,0].set_xlabel('Recurrence Score')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('UCMC Dataset')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].hist(bcr_scores, bins=20, alpha=0.7, color='lightcoral', 
                   edgecolor='black', label=f'BCR-NET (n={len(bcr_scores)})')
    axes[0,1].axvline(x=25, color='red', linestyle='--', linewidth=2, 
                     label='Clinical Threshold')
    axes[0,1].set_xlabel('Recurrence Score')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('BCR-NET Dataset')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 2. Overlapping comparison
    axes[1,0].hist([ucmc_scores, bcr_scores], bins=30, alpha=0.6, 
                   label=['UCMC', 'BCR-NET'], color=['skyblue', 'lightcoral'], 
                   edgecolor='black')
    axes[1,0].axvline(x=25, color='red', linestyle='--', linewidth=2, 
                     label='Clinical Threshold (RS=25)')
    axes[1,0].set_xlabel('Recurrence Score')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Dataset Comparison (Overlapping)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 3. MAIN PLOT: Combined dataset histogram
    axes[1,1].hist(combined_scores, bins=35, alpha=0.8, color='mediumseagreen', 
                   edgecolor='black')
    axes[1,1].axvline(x=25, color='red', linestyle='--', linewidth=2, 
                     label='Clinical Threshold (RS=25)')
    axes[1,1].set_xlabel('Recurrence Score')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title(f'Combined Dataset: RS Distribution (n={len(combined_scores)})')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print combined dataset statistics
    print("="*60)
    print("COMBINED DATASET RS STATISTICS")
    print("="*60)
    print(f"Total samples: {len(combined_scores)}")
    print(f"UCMC contribution: {len(ucmc_scores)} ({len(ucmc_scores)/len(combined_scores)*100:.1f}%)")
    print(f"BCR-NET contribution: {len(bcr_scores)} ({len(bcr_scores)/len(combined_scores)*100:.1f}%)")
    
    print(f"\nRS Score Statistics:")
    print(f"  Mean: {np.mean(combined_scores):.2f}")
    print(f"  Median: {np.median(combined_scores):.2f}")
    print(f"  Std: {np.std(combined_scores):.2f}")
    print(f"  Min: {np.min(combined_scores):.1f}")
    print(f"  Max: {np.max(combined_scores):.1f}")
    print(f"  25th percentile: {np.percentile(combined_scores, 25):.2f}")
    print(f"  75th percentile: {np.percentile(combined_scores, 75):.2f}")
    
    # Class distribution
    high_risk_combined = (combined_scores >= 25).sum()
    low_risk_combined = len(combined_scores) - high_risk_combined
    print(f"\nClass Distribution:")
    print(f"  Low Risk (RS < 25): {low_risk_combined} ({low_risk_combined/len(combined_scores)*100:.1f}%)")
    print(f"  High Risk (RS ≥ 25): {high_risk_combined} ({high_risk_combined/len(combined_scores)*100:.1f}%)")
    print(f"  Imbalance Ratio (Low:High): {low_risk_combined/high_risk_combined:.2f}:1")
    
    return combined_scores, ucmc_scores, bcr_scores

if __name__ == "__main__":
    combined_scores, ucmc_scores, bcr_scores = plot_combined_rs_distribution()