# analyze_h5_files.py
"""
Analyze and visualize H5 files from CLAM preprocessing.
Shows patches, statistics, and coordinate distribution.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import pandas as pd
from typing import List, Tuple
import argparse
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class H5Analyzer:
    """Analyze and visualize H5 files from preprocessing."""
    
    def __init__(self, h5_dir: str):
        """
        Initialize analyzer.
        
        Args:
            h5_dir: Directory containing H5 files
        """
        self.h5_dir = Path(h5_dir)
        self.h5_files = sorted(list(self.h5_dir.glob('*.h5')))
        
        if not self.h5_files:
            print(f"No H5 files found in {h5_dir}")
            return
            
        print(f"Found {len(self.h5_files)} H5 files")
    
    def analyze_single_h5(self, h5_path: Path) -> dict:
        """
        Analyze a single H5 file.
        
        Args:
            h5_path: Path to H5 file
            
        Returns:
            Dictionary with statistics
        """
        stats = {'patient_id': h5_path.stem}
        
        try:
            with h5py.File(h5_path, 'r') as f:
                # Check keys
                stats['keys'] = list(f.keys())
                
                if 'bag' in f and 'coords' in f:
                    # Get patches and coords
                    patches = f['bag']
                    coords = f['coords'][:]
                    
                    # Basic stats
                    stats['n_patches'] = patches.shape[0]
                    stats['patch_shape'] = patches.shape[1:]
                    stats['dtype'] = str(patches.dtype)
                    
                    # Pixel statistics (sample first 10 patches)
                    sample_patches = patches[:min(10, len(patches))]
                    stats['pixel_min'] = float(np.min(sample_patches))
                    stats['pixel_max'] = float(np.max(sample_patches))
                    stats['pixel_mean'] = float(np.mean(sample_patches))
                    stats['pixel_std'] = float(np.std(sample_patches))
                    
                    # Coordinate statistics
                    stats['coord_x_min'] = int(np.min(coords[:, 0]))
                    stats['coord_x_max'] = int(np.max(coords[:, 0]))
                    stats['coord_y_min'] = int(np.min(coords[:, 1]))
                    stats['coord_y_max'] = int(np.max(coords[:, 1]))
                    stats['coord_x_range'] = stats['coord_x_max'] - stats['coord_x_min']
                    stats['coord_y_range'] = stats['coord_y_max'] - stats['coord_y_min']
                    
                    # Estimate coverage
                    unique_x = len(np.unique(coords[:, 0]))
                    unique_y = len(np.unique(coords[:, 1]))
                    stats['unique_x_positions'] = unique_x
                    stats['unique_y_positions'] = unique_y
                    
                    # File size
                    stats['file_size_mb'] = h5_path.stat().st_size / (1024 * 1024)
                    
                else:
                    stats['error'] = 'Missing bag or coords keys'
                    
        except Exception as e:
            stats['error'] = str(e)
        
        return stats
    
    def analyze_all(self) -> pd.DataFrame:
        """
        Analyze all H5 files.
        
        Returns:
            DataFrame with statistics for all files
        """
        all_stats = []
        
        print("\nAnalyzing H5 files...")
        for h5_path in self.h5_files:
            stats = self.analyze_single_h5(h5_path)
            all_stats.append(stats)
            
            if 'error' in stats:
                print(f"  {h5_path.stem}: ERROR - {stats['error']}")
            else:
                print(f"  {h5_path.stem}: {stats['n_patches']} patches")
        
        return pd.DataFrame(all_stats)
    
    def visualize_patches(self, h5_path: Path, n_patches: int = 16, 
                         save_path: str = None):
        """
        Visualize sample patches from an H5 file.
        
        Args:
            h5_path: Path to H5 file
            n_patches: Number of patches to visualize
            save_path: Optional path to save figure
        """
        with h5py.File(h5_path, 'r') as f:
            patches = f['bag']
            coords = f['coords'][:]
            
            # Sample patches evenly
            n_total = patches.shape[0]
            if n_total < n_patches:
                indices = np.arange(n_total)
            else:
                indices = np.linspace(0, n_total-1, n_patches, dtype=int)
            
            # Create grid plot
            n_cols = int(np.sqrt(n_patches))
            n_rows = int(np.ceil(n_patches / n_cols))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
            axes = axes.flatten() if n_patches > 1 else [axes]
            
            for i, idx in enumerate(indices):
                if i >= n_patches:
                    break
                    
                patch = patches[idx]
                coord = coords[idx]
                
                axes[i].imshow(patch)
                axes[i].set_title(f"Patch {idx}\n({coord[0]}, {coord[1]})", 
                                 fontsize=8)
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(len(indices), len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(f"Sample Patches from {h5_path.stem}\n"
                        f"Total: {n_total} patches", fontsize=14)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved patch visualization to {save_path}")
            else:
                plt.show()
    
    def visualize_coordinates(self, h5_path: Path, save_path: str = None):
        """
        Visualize patch coordinate distribution.
        
        Args:
            h5_path: Path to H5 file
            save_path: Optional path to save figure
        """
        with h5py.File(h5_path, 'r') as f:
            coords = f['coords'][:]
            n_patches = len(coords)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Scatter plot of coordinates
        ax = axes[0, 0]
        scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                           alpha=0.5, s=10, c=np.arange(len(coords)), 
                           cmap='viridis')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(f'Patch Locations (n={n_patches})')
        ax.invert_yaxis()  # Match image coordinates
        plt.colorbar(scatter, ax=ax, label='Patch index')
        
        # Add grid to show patch size
        patch_size = 224
        for x in range(coords[:, 0].min(), coords[:, 0].max() + patch_size, patch_size):
            ax.axvline(x, color='gray', alpha=0.1, linewidth=0.5)
        for y in range(coords[:, 1].min(), coords[:, 1].max() + patch_size, patch_size):
            ax.axhline(y, color='gray', alpha=0.1, linewidth=0.5)
        
        # 2. Coverage map / heatmap
        ax = axes[0, 1]
        # Create 2D histogram
        h, xedges, yedges = np.histogram2d(coords[:, 0], coords[:, 1], bins=20)
        im = ax.imshow(h.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                      origin='lower', cmap='hot', aspect='auto')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title('Patch Density Heatmap')
        plt.colorbar(im, ax=ax, label='Patch count')
        
        # 3. X distribution
        ax = axes[1, 0]
        ax.hist(coords[:, 0], bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Count')
        ax.set_title('X Coordinate Distribution')
        ax.axvline(coords[:, 0].mean(), color='red', linestyle='--', 
                  label=f'Mean: {coords[:, 0].mean():.0f}')
        ax.legend()
        
        # 4. Y distribution
        ax = axes[1, 1]
        ax.hist(coords[:, 1], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('Y coordinate')
        ax.set_ylabel('Count')
        ax.set_title('Y Coordinate Distribution')
        ax.axvline(coords[:, 1].mean(), color='red', linestyle='--',
                  label=f'Mean: {coords[:, 1].mean():.0f}')
        ax.legend()
        
        plt.suptitle(f'Coordinate Analysis: {h5_path.stem}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved coordinate visualization to {save_path}")
        else:
            plt.show()
    
    def print_summary_statistics(self, stats_df: pd.DataFrame):
        """
        Print summary statistics for all H5 files.
        
        Args:
            stats_df: DataFrame with statistics
        """
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        # Remove error cases for statistics
        valid_df = stats_df[~stats_df['n_patches'].isna()].copy()
        
        if len(valid_df) == 0:
            print("No valid H5 files found!")
            return
        
        print(f"Total patients: {len(stats_df)}")
        print(f"Valid H5 files: {len(valid_df)}")
        
        if len(stats_df) > len(valid_df):
            print(f"Files with errors: {len(stats_df) - len(valid_df)}")
            error_files = stats_df[stats_df['n_patches'].isna()]['patient_id'].tolist()
            print(f"  Error files: {', '.join(error_files[:5])}")
        
        print(f"\nPatch Statistics:")
        print(f"  Total patches: {valid_df['n_patches'].sum():,}")
        print(f"  Patches per patient:")
        print(f"    Mean: {valid_df['n_patches'].mean():.1f}")
        print(f"    Std:  {valid_df['n_patches'].std():.1f}")
        print(f"    Min:  {valid_df['n_patches'].min()}")
        print(f"    Max:  {valid_df['n_patches'].max()}")
        
        print(f"\nPixel Value Range:")
        print(f"  Min: {valid_df['pixel_min'].min():.1f}")
        print(f"  Max: {valid_df['pixel_max'].max():.1f}")
        print(f"  Mean: {valid_df['pixel_mean'].mean():.1f}")
        
        print(f"\nCoordinate Coverage:")
        print(f"  X range: {valid_df['coord_x_range'].mean():.0f} ± "
              f"{valid_df['coord_x_range'].std():.0f}")
        print(f"  Y range: {valid_df['coord_y_range'].mean():.0f} ± "
              f"{valid_df['coord_y_range'].std():.0f}")
        
        print(f"\nStorage:")
        total_size = valid_df['file_size_mb'].sum()
        print(f"  Total size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
        print(f"  Average file size: {valid_df['file_size_mb'].mean():.1f} MB")
        
        # Identify outliers
        print(f"\nOutliers:")
        threshold_low = valid_df['n_patches'].quantile(0.1)
        threshold_high = valid_df['n_patches'].quantile(0.9)
        
        low_patch_patients = valid_df[valid_df['n_patches'] < threshold_low]['patient_id'].tolist()
        high_patch_patients = valid_df[valid_df['n_patches'] > threshold_high]['patient_id'].tolist()
        
        if low_patch_patients:
            print(f"  Patients with very few patches (<{threshold_low:.0f}):")
            for p in low_patch_patients[:3]:
                n = valid_df[valid_df['patient_id'] == p]['n_patches'].iloc[0]
                print(f"    - {p}: {n} patches")
        
        if high_patch_patients:
            print(f"  Patients with many patches (>{threshold_high:.0f}):")
            for p in high_patch_patients[:3]:
                n = valid_df[valid_df['patient_id'] == p]['n_patches'].iloc[0]
                print(f"    - {p}: {n} patches")
    
    def create_distribution_plots(self, stats_df: pd.DataFrame, save_path: str = None):
        """
        Create distribution plots for all patients.
        
        Args:
            stats_df: DataFrame with statistics
            save_path: Optional path to save figure
        """
        valid_df = stats_df[~stats_df['n_patches'].isna()].copy()
        
        if len(valid_df) == 0:
            print("No valid data for plotting!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Patches per patient distribution
        ax = axes[0, 0]
        ax.hist(valid_df['n_patches'], bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(valid_df['n_patches'].mean(), color='red', linestyle='--',
                  label=f"Mean: {valid_df['n_patches'].mean():.0f}")
        ax.axvline(valid_df['n_patches'].median(), color='green', linestyle='--',
                  label=f"Median: {valid_df['n_patches'].median():.0f}")
        ax.set_xlabel('Number of Patches')
        ax.set_ylabel('Number of Patients')
        ax.set_title('Distribution of Patches per Patient')
        ax.legend()
        
        # 2. File size distribution
        ax = axes[0, 1]
        ax.hist(valid_df['file_size_mb'], bins=20, alpha=0.7, 
                color='orange', edgecolor='black')
        ax.set_xlabel('File Size (MB)')
        ax.set_ylabel('Number of Patients')
        ax.set_title('H5 File Size Distribution')
        
        # 3. Patches vs File Size
        ax = axes[1, 0]
        ax.scatter(valid_df['n_patches'], valid_df['file_size_mb'], alpha=0.6)
        ax.set_xlabel('Number of Patches')
        ax.set_ylabel('File Size (MB)')
        ax.set_title('Patches vs File Size')
        
        # Add trend line
        z = np.polyfit(valid_df['n_patches'], valid_df['file_size_mb'], 1)
        p = np.poly1d(z)
        ax.plot(valid_df['n_patches'].sort_values(), 
                p(valid_df['n_patches'].sort_values()), 
                "r--", alpha=0.8, label='Trend')
        ax.legend()
        
        # 4. Top patients by patch count
        ax = axes[1, 1]
        top_10 = valid_df.nlargest(10, 'n_patches')[['patient_id', 'n_patches']]
        ax.barh(range(len(top_10)), top_10['n_patches'].values)
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10['patient_id'].values, fontsize=8)
        ax.set_xlabel('Number of Patches')
        ax.set_title('Top 10 Patients by Patch Count')
        ax.invert_yaxis()
        
        plt.suptitle('ECU Preprocessing Statistics', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved distribution plots to {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze H5 files from preprocessing')
    parser.add_argument('--h5_dir', type=str, required=True,
                       help='Directory containing H5 files')
    parser.add_argument('--output_dir', type=str, default='h5_analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--sample_patients', type=int, default=3,
                       help='Number of patients to visualize in detail')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = H5Analyzer(args.h5_dir)
    
    if not analyzer.h5_files:
        return
    
    # Analyze all files
    print("\nAnalyzing H5 files...")
    stats_df = analyzer.analyze_all()
    
    # Save statistics
    stats_path = output_dir / 'h5_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"\nStatistics saved to {stats_path}")
    
    # Print summary
    analyzer.print_summary_statistics(stats_df)
    
    # Create distribution plots
    dist_plot_path = output_dir / 'distribution_plots.png'
    analyzer.create_distribution_plots(stats_df, dist_plot_path)
    
    # Visualize sample patients
    print(f"\nVisualizing {args.sample_patients} sample patients...")
    for i, h5_path in enumerate(analyzer.h5_files[:args.sample_patients]):
        print(f"\nVisualizing {h5_path.stem}...")
        
        # Visualize patches
        patch_viz_path = output_dir / f'{h5_path.stem}_patches.png'
        analyzer.visualize_patches(h5_path, n_patches=16, save_path=patch_viz_path)
        
        # Visualize coordinates
        coord_viz_path = output_dir / f'{h5_path.stem}_coordinates.png'
        analyzer.visualize_coordinates(h5_path, save_path=coord_viz_path)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()