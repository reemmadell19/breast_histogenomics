import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10

def visualize_patches_with_coordinates(file_path, num_samples=12):
    """
    Visualize sample patches from BCR-NET file with their coordinates
    """
    print(f"Loading file: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # Load data
        patches = f['bag'][:]  # Shape: (num_patches, 3, 224, 224)
        coords = f['coords'][:]  # Shape: (2, num_patches) - [x, y] coordinates
        label = f['label'][()]
        
        print(f"Total patches: {patches.shape[0]}")
        print(f"Coordinates shape: {coords.shape}")
        print(f"Label: {label}")
        
        # Convert patches from (num_patches, channels, height, width) to (num_patches, height, width, channels)
        patches = np.transpose(patches, (0, 2, 3, 1))
        
        # Extract x, y coordinates
        x_coords = coords[0, :]  # X coordinates
        y_coords = coords[1, :]  # Y coordinates
        
        print(f"X coordinate range: [{x_coords.min()}, {x_coords.max()}]")
        print(f"Y coordinate range: [{y_coords.min()}, {y_coords.max()}]")
        
        # Sample patches for visualization
        sample_indices = np.linspace(0, len(patches)-1, num_samples, dtype=int)
        
        # Create visualization
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, idx in enumerate(sample_indices):
            patch = patches[idx]
            x_coord = x_coords[idx]
            y_coord = y_coords[idx]
            
            # Display patch
            axes[i].imshow(patch)
            axes[i].set_title(f'Patch {idx}\nCoords: ({x_coord}, {y_coord})', fontsize=12)
            axes[i].axis('off')
            
            # Add coordinate text on the patch
            axes[i].text(5, 20, f'X: {x_coord}\nY: {y_coord}', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        fontsize=10, fontweight='bold')
        
        plt.suptitle(f'Sample Patches from {Path(file_path).name}\nTotal Patches: {len(patches)}, Label: {label}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return patches, coords, label

def visualize_coordinate_distribution(coords, file_name):
    """
    Visualize the spatial distribution of patch coordinates
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x_coords = coords[0, :]
    y_coords = coords[1, :]
    
    # 1. Scatter plot of coordinates
    scatter = axes[0].scatter(x_coords, y_coords, alpha=0.6, s=10)
    axes[0].set_xlabel('X Coordinate')
    axes[0].set_ylabel('Y Coordinate')
    axes[0].set_title(f'Spatial Distribution of Patches\n{file_name}')
    axes[0].grid(True, alpha=0.3)
    
    # 2. X coordinate histogram
    axes[1].hist(x_coords, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1].set_xlabel('X Coordinate')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('X Coordinate Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Y coordinate histogram
    axes[2].hist(y_coords, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[2].set_xlabel('Y Coordinate')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Y Coordinate Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print coordinate statistics
    print("\n=== COORDINATE STATISTICS ===")
    print(f"X coordinates: Min={x_coords.min()}, Max={x_coords.max()}, Mean={x_coords.mean():.1f}, Std={x_coords.std():.1f}")
    print(f"Y coordinates: Min={y_coords.min()}, Max={y_coords.max()}, Mean={y_coords.mean():.1f}, Std={y_coords.std():.1f}")

def create_coordinate_heatmap(coords, file_name):
    """
    Create a heatmap showing patch density across the slide
    """
    x_coords = coords[0, :]
    y_coords = coords[1, :]
    
    # Create 2D histogram for heatmap
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    hist, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=50)
    
    # Plot heatmap
    plt.imshow(hist.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
               cmap='viridis', aspect='auto')
    plt.colorbar(label='Patch Count')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Patch Density Heatmap\n{file_name}')
    
    # Add some sample patch locations
    sample_indices = np.random.choice(len(x_coords), 20, replace=False)
    sample_x = x_coords[sample_indices]
    sample_y = y_coords[sample_indices]
    plt.scatter(sample_x, sample_y, color='red', s=30, alpha=0.7, marker='x', label='Sample Patches')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def analyze_file_10():
    """
    Analyze specifically file 10.h5 from BCR-NET
    """
    file_path = "/cs/student/projects1/aibh/2024/elnefary/data/raw/BCR_NET/10.h5"
    
    print("="*60)
    print("BCR-NET FILE 10.h5 ANALYSIS WITH COORDINATES")
    print("="*60)
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return
    
    # Visualize patches with coordinates
    patches, coords, label = visualize_patches_with_coordinates(file_path, num_samples=12)
    
    # Visualize coordinate distribution
    visualize_coordinate_distribution(coords, "10.h5")
    
    # Create coordinate heatmap
    create_coordinate_heatmap(coords, "10.h5")
    
    # Additional analysis
    print("\n=== ADDITIONAL ANALYSIS ===")
    print(f"Patch dimensions: {patches[0].shape}")
    print(f"Data type: {patches.dtype}")
    print(f"Intensity range: [{patches.min()}, {patches.max()}]")
    print(f"Total tissue area covered: {len(patches)} patches of 224×224 pixels")
    
    # Calculate approximate slide coverage
    x_range = coords[0, :].max() - coords[0, :].min()
    y_range = coords[1, :].max() - coords[1, :].min()
    print(f"Slide coordinate range: X={x_range}, Y={y_range}")

if __name__ == "__main__":
    analyze_file_10()