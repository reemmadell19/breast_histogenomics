import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import tensorflow as tf
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

def analyze_bcr_net_patches(data_directory, num_samples=3):
    """
    Analyze BCR-NET patches: (num_patches, 3, 224, 224) - RGB patches
    """
    print("=== BCR-NET PATCH ANALYSIS ===")
    
    h5_files = list(Path(data_directory).glob("*.h5"))[:num_samples]
    
    all_intensities = []
    patch_info = []
    
    for file_path in h5_files:
        with h5py.File(file_path, 'r') as f:
            patches = f['bag'][:]  # Shape: (num_patches, 3, 224, 224)
            
            # Convert from (num_patches, channels, height, width) to (num_patches, height, width, channels)
            patches = np.transpose(patches, (0, 2, 3, 1))
            
            # Sample some patches for intensity analysis
            sample_patches = patches[:50]  # Sample 50 patches
            all_intensities.extend(sample_patches.flatten())
            
            patch_info.append({
                'dataset': 'BCR-NET',
                'patch_height': patches.shape[1],
                'patch_width': patches.shape[2], 
                'channels': patches.shape[3],
                'data_type': patches.dtype,
                'mean_intensity': np.mean(sample_patches),
                'std_intensity': np.std(sample_patches)
            })
            
            print(f"File: {file_path.name}")
            print(f"  Patches shape: {patches.shape}")
            print(f"  Patch size: {patches.shape[1]}×{patches.shape[2]}×{patches.shape[3]}")
            print(f"  Data type: {patches.dtype}")
            print(f"  Intensity range: [{np.min(sample_patches)}, {np.max(sample_patches)}]")
    
    return patch_info, all_intensities

def analyze_ucmc_patches(data_directory, num_samples=3):
    """
    Analyze UCMC patches from TFRecord files
    """
    print("\n=== UCMC PATCH ANALYSIS ===")
    
    tfrecord_files = list(Path(data_directory).glob("*.tfrecord*"))[:num_samples]
    
    all_intensities = []
    patch_info = []
    
    for file_path in tfrecord_files:
        dataset = tf.data.TFRecordDataset(str(file_path))
        
        patches_analyzed = 0
        
        for raw_record in dataset.take(50):  # Sample 50 patches
            try:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                
                features = example.features.feature
                
                # Get raw image data
                if 'image_raw' in features:
                    image_bytes = features['image_raw'].bytes_list.value[0]
                    
                    # Decode image (assuming it's stored as PNG/JPEG)
                    image = tf.image.decode_image(image_bytes, channels=3)
                    image_np = image.numpy()
                    
                    if patches_analyzed == 0:  # First patch - get dimensions
                        height, width, channels = image_np.shape
                        patch_info.append({
                            'dataset': 'UCMC',
                            'patch_height': height,
                            'patch_width': width,
                            'channels': channels,
                            'data_type': image_np.dtype,
                            'mean_intensity': 0,  # Will calculate later
                            'std_intensity': 0    # Will calculate later
                        })
                        print(f"File: {file_path.name}")
                        print(f"  Patch size: {height}×{width}×{channels}")
                        print(f"  Data type: {image_np.dtype}")
                    
                    all_intensities.extend(image_np.flatten())
                    patches_analyzed += 1
                    
            except Exception as e:
                continue
        
        if patches_analyzed > 0 and patch_info:
            # Update intensity stats
            sample_intensities = np.array(all_intensities[-patches_analyzed*height*width*channels:])
            patch_info[-1]['mean_intensity'] = np.mean(sample_intensities)
            patch_info[-1]['std_intensity'] = np.std(sample_intensities)
            print(f"  Intensity range: [{np.min(sample_intensities)}, {np.max(sample_intensities)}]")
            print(f"  Analyzed {patches_analyzed} patches")
    
    return patch_info, all_intensities

def visualize_patch_characteristics(bcr_info, bcr_intensities, ucmc_info, ucmc_intensities):
    """
    Create visualizations for patch characteristics
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Patch Dimensions Comparison
    if bcr_info and ucmc_info:
        datasets = ['BCR-NET', 'UCMC']
        heights = [bcr_info[0]['patch_height'], ucmc_info[0]['patch_height']]
        widths = [bcr_info[0]['patch_width'], ucmc_info[0]['patch_width']]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, heights, width, label='Height', alpha=0.7, color='skyblue')
        axes[0, 0].bar(x + width/2, widths, width, label='Width', alpha=0.7, color='lightcoral')
        axes[0, 0].set_xlabel('Dataset')
        axes[0, 0].set_ylabel('Pixels')
        axes[0, 0].set_title('Patch Dimensions')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(datasets)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Data Types
    if bcr_info and ucmc_info:
        data_types = [str(bcr_info[0]['data_type']), str(ucmc_info[0]['data_type'])]
        colors = ['lightcoral', 'skyblue']
        
        axes[0, 1].bar(datasets, [1, 1], color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Data Type')
        axes[0, 1].set_title('Data Types by Dataset')
        
        for i, (dataset, dtype) in enumerate(zip(datasets, data_types)):
            axes[0, 1].text(i, 0.5, dtype, ha='center', va='center', fontweight='bold')
        axes[0, 1].set_ylim(0, 1.2)
        axes[0, 1].set_yticks([])
    
    # 3. Pixel Intensity Distributions
    if bcr_intensities and ucmc_intensities:
        # Sample intensities for plotting (to avoid memory issues)
        bcr_sample = np.random.choice(bcr_intensities, min(10000, len(bcr_intensities)), replace=False)
        ucmc_sample = np.random.choice(ucmc_intensities, min(10000, len(ucmc_intensities)), replace=False)
        
        axes[0, 2].hist(bcr_sample, bins=50, alpha=0.7, label='BCR-NET', color='lightcoral', density=True)
        axes[0, 2].hist(ucmc_sample, bins=50, alpha=0.7, label='UCMC', color='skyblue', density=True)
        axes[0, 2].set_xlabel('Pixel Intensity')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Pixel Intensity Distributions')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Intensity Statistics
    if bcr_info and ucmc_info:
        stats = ['Mean', 'Std']
        bcr_stats = [bcr_info[0]['mean_intensity'], bcr_info[0]['std_intensity']]
        ucmc_stats = [ucmc_info[0]['mean_intensity'], ucmc_info[0]['std_intensity']]
        
        x = np.arange(len(stats))
        axes[1, 0].bar(x - width/2, bcr_stats, width, label='BCR-NET', alpha=0.7, color='lightcoral')
        axes[1, 0].bar(x + width/2, ucmc_stats, width, label='UCMC', alpha=0.7, color='skyblue')
        axes[1, 0].set_xlabel('Statistic')
        axes[1, 0].set_ylabel('Intensity Value')
        axes[1, 0].set_title('Intensity Statistics')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(stats)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Box plot comparison
    if bcr_intensities and ucmc_intensities:
        bcr_sample = np.random.choice(bcr_intensities, min(5000, len(bcr_intensities)), replace=False)
        ucmc_sample = np.random.choice(ucmc_intensities, min(5000, len(ucmc_intensities)), replace=False)
        
        data_for_box = [bcr_sample, ucmc_sample]
        bp = axes[1, 1].boxplot(data_for_box, labels=['BCR-NET', 'UCMC'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('skyblue')
        axes[1, 1].set_ylabel('Pixel Intensity')
        axes[1, 1].set_title('Intensity Distribution Comparison')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Summary table
    axes[1, 2].axis('tight')
    axes[1, 2].axis('off')
    
    if bcr_info and ucmc_info:
        table_data = [
            ['Dataset', 'BCR-NET', 'UCMC'],
            ['Dimensions', f"{bcr_info[0]['patch_height']}×{bcr_info[0]['patch_width']}×{bcr_info[0]['channels']}", 
             f"{ucmc_info[0]['patch_height']}×{ucmc_info[0]['patch_width']}×{ucmc_info[0]['channels']}"],
            ['Data Type', str(bcr_info[0]['data_type']), str(ucmc_info[0]['data_type'])],
            ['Mean Intensity', f"{bcr_info[0]['mean_intensity']:.1f}", f"{ucmc_info[0]['mean_intensity']:.1f}"],
            ['Std Intensity', f"{bcr_info[0]['std_intensity']:.1f}", f"{ucmc_info[0]['std_intensity']:.1f}"]
        ]
        
        table = axes[1, 2].table(cellText=table_data[1:], colLabels=table_data[0],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        axes[1, 2].set_title('Patch Characteristics Summary')
    
    plt.tight_layout()
    plt.show()

def print_preprocessing_summary(bcr_info, ucmc_info):
    """
    Print summary for thesis
    """
    print("\n" + "="*60)
    print("PATCH CHARACTERISTICS FOR THESIS")
    print("="*60)
    
    if bcr_info:
        print("\nBCR-NET Dataset:")
        print(f"  Patch dimensions: {bcr_info[0]['patch_height']}×{bcr_info[0]['patch_width']}×{bcr_info[0]['channels']} pixels")
        print(f"  Data type: {bcr_info[0]['data_type']}")
        print(f"  Mean pixel intensity: {bcr_info[0]['mean_intensity']:.1f}")
        print(f"  Standard deviation: {bcr_info[0]['std_intensity']:.1f}")
        print(f"  Storage format: HDF5 with keys ['bag', 'coords', 'label']")
    
    if ucmc_info:
        print("\nUCMC Dataset:")
        print(f"  Patch dimensions: {ucmc_info[0]['patch_height']}×{ucmc_info[0]['patch_width']}×{ucmc_info[0]['channels']} pixels")
        print(f"  Data type: {ucmc_info[0]['data_type']}")
        print(f"  Mean pixel intensity: {ucmc_info[0]['mean_intensity']:.1f}")
        print(f"  Standard deviation: {ucmc_info[0]['std_intensity']:.1f}")
        print(f"  Storage format: TFRecord with features ['image_raw', 'slide', 'loc_x', 'loc_y']")
    
    if bcr_info and ucmc_info:
        print("\nComparison:")
        if bcr_info[0]['patch_height'] == ucmc_info[0]['patch_height']:
            print("  ✓ Same patch dimensions across datasets")
        else:
            print("  ⚠ Different patch dimensions - preprocessing needed")
        
        intensity_diff = abs(bcr_info[0]['mean_intensity'] - ucmc_info[0]['mean_intensity'])
        if intensity_diff < 10:
            print("  ✓ Similar intensity ranges")
        else:
            print("  ⚠ Different intensity ranges - normalization recommended")

def visualize_sample_patches(bcr_net_dir, ucmc_dir):
    """
    Visualize sample patches from both datasets
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # BCR-NET samples
    print("Loading BCR-NET patch samples...")
    h5_files = list(Path(bcr_net_dir).glob("*.h5"))
    
    if h5_files:
        with h5py.File(h5_files[0], 'r') as f:
            patches = f['bag'][:]  # Shape: (num_patches, 3, 224, 224)
            # Convert to (num_patches, height, width, channels)
            patches = np.transpose(patches, (0, 2, 3, 1))
            
            # Show 5 sample patches
            for i in range(5):
                patch = patches[i * 100]  # Sample every 100th patch
                axes[0, i].imshow(patch)
                axes[0, i].set_title(f'BCR-NET Patch {i+1}\n224×224×3')
                axes[0, i].axis('off')
    
    # UCMC samples
    print("Loading UCMC patch samples...")
    tfrecord_files = list(Path(ucmc_dir).glob("*.tfrecord*"))
    
    if tfrecord_files:
        dataset = tf.data.TFRecordDataset(str(tfrecord_files[0]))
        
        patch_count = 0
        for raw_record in dataset:
            if patch_count >= 5:
                break
                
            try:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                features = example.features.feature
                
                if 'image_raw' in features:
                    image_bytes = features['image_raw'].bytes_list.value[0]
                    image = tf.image.decode_image(image_bytes, channels=3)
                    image_np = image.numpy()
                    
                    axes[1, patch_count].imshow(image_np)
                    axes[1, patch_count].set_title(f'UCMC Patch {patch_count+1}\n299×299×3')
                    axes[1, patch_count].axis('off')
                    patch_count += 1
                    
            except Exception as e:
                continue
    
    # Add dataset labels
    axes[0, 0].text(-0.1, 0.5, 'BCR-NET', rotation=90, va='center', ha='center',
                   transform=axes[0, 0].transAxes, fontsize=16, fontweight='bold')
    axes[1, 0].text(-0.1, 0.5, 'UCMC', rotation=90, va='center', ha='center',
                   transform=axes[1, 0].transAxes, fontsize=16, fontweight='bold')
    
    plt.suptitle('Sample Patches from Both Datasets', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()

def run_patch_visualization():
    """
    Main function with patch visualization added
    """
    bcr_net_dir = "/cs/student/projects1/aibh/2024/elnefary/data/raw/BCR_NET"
    ucmc_dir = "/cs/student/projects1/aibh/2024/elnefary/data/raw/UCMC"
    
    print("Analyzing patch characteristics for visualization...")
    
    # First show sample patches
    visualize_sample_patches(bcr_net_dir, ucmc_dir)
    
    # Then analyze characteristics
    bcr_info, bcr_intensities = analyze_bcr_net_patches(bcr_net_dir, num_samples=3)
    ucmc_info, ucmc_intensities = analyze_ucmc_patches(ucmc_dir, num_samples=3)
    
    # Create characteristic visualizations
    if bcr_info or ucmc_info:
        visualize_patch_characteristics(bcr_info, bcr_intensities, ucmc_info, ucmc_intensities)
        print_preprocessing_summary(bcr_info, ucmc_info)
    else:
        print("No patch data could be analyzed.")

if __name__ == "__main__":
    run_patch_visualization()