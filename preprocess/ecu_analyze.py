#!/usr/bin/env python3
"""
Test script to check ECU tfrecord format and image encoding.
Run this before feature extraction to understand the data structure.
"""

import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def test_tfrecord_structure(tfrecord_path):
    """Check the structure and keys of a tfrecord file."""
    print("\n" + "="*60)
    print("TFRECORD STRUCTURE TEST")
    print("="*60)
    print(f"File: {os.path.basename(tfrecord_path)}")
    print("-"*60)
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    for i, raw_record in enumerate(dataset.take(1)):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        feature_dict = example.features.feature
        print(f"\nFound {len(feature_dict)} keys:")
        
        for key in sorted(feature_dict.keys()):
            feature = feature_dict[key]
            
            if feature.HasField('bytes_list'):
                data_type = "bytes"
                count = len(feature.bytes_list.value)
                if count > 0 and key == 'image':
                    size = len(feature.bytes_list.value[0])
                    print(f"  '{key}': {data_type} (count: {count}, size: {size} bytes)")
                else:
                    print(f"  '{key}': {data_type} (count: {count})")
            elif feature.HasField('float_list'):
                data_type = "float"
                count = len(feature.float_list.value)
                print(f"  '{key}': {data_type} (count: {count})")
            elif feature.HasField('int64_list'):
                data_type = "int64"
                count = len(feature.int64_list.value)
                value = feature.int64_list.value[0] if count > 0 else None
                print(f"  '{key}': {data_type} (count: {count}, value: {value})")

def test_image_decoding(tfrecord_path, num_samples=3):
    """Test different image decoding methods."""
    print("\n" + "="*60)
    print("IMAGE DECODING TEST")
    print("="*60)
    
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'da_number': tf.io.FixedLenFeature([], tf.int64),
        'global_x': tf.io.FixedLenFeature([], tf.int64),
        'global_y': tf.io.FixedLenFeature([], tf.int64),
    }
    
    def _parse(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)
    
    dataset = tf.data.TFRecordDataset(tfrecord_path).map(_parse)
    
    successful_decodings = []
    
    for i, record in enumerate(dataset.take(num_samples)):
        print(f"\n--- Sample {i+1} ---")
        img_bytes = record['image'].numpy()
        print(f"DA number: {record['da_number'].numpy()}")
        print(f"Global position: ({record['global_x'].numpy()}, {record['global_y'].numpy()})")
        print(f"Raw bytes length: {len(img_bytes)} bytes")
        
        # Try different decodings
        img = None
        decode_method = None
        
        # 1. Try JPEG
        try:
            img = tf.io.decode_jpeg(img_bytes).numpy()
            decode_method = "JPEG"
            print(f"✓ JPEG decode successful")
        except Exception as e:
            print(f"✗ JPEG failed: {str(e)[:50]}")
        
        # 2. Try PNG if JPEG failed
        if img is None:
            try:
                img = tf.io.decode_png(img_bytes).numpy()
                decode_method = "PNG"
                print(f"✓ PNG decode successful")
            except Exception as e:
                print(f"✗ PNG failed: {str(e)[:50]}")
        
        # 3. Try raw decode if both failed
        if img is None:
            try:
                img = tf.io.decode_raw(img_bytes, tf.uint8).numpy()
                print(f"? Raw decode: shape={img.shape}")
                
                # Try common reshape patterns
                if img.shape[0] == 224*224*3:
                    img = img.reshape(224, 224, 3)
                    decode_method = "Raw (224x224x3)"
                    print(f"  Reshaped to 224x224x3")
                elif img.shape[0] == 224*224:
                    img = img.reshape(224, 224)
                    img = np.stack([img]*3, axis=-1)  # Convert grayscale to RGB
                    decode_method = "Raw (224x224 grayscale)"
                    print(f"  Reshaped to 224x224 grayscale → RGB")
                else:
                    decode_method = "Raw (unknown shape)"
                    print(f"  Could not determine reshape pattern")
            except Exception as e:
                print(f"✗ Raw decode failed: {str(e)[:50]}")
        
        # Print image statistics if successful
        if img is not None:
            print(f"\nImage successfully decoded using: {decode_method}")
            print(f"  Shape: {img.shape}")
            print(f"  Dtype: {img.dtype}")
            print(f"  Value range: [{img.min()}, {img.max()}]")
            print(f"  Mean: {img.mean():.2f}, Std: {img.std():.2f}")
            
            successful_decodings.append({
                'sample': i+1,
                'method': decode_method,
                'shape': img.shape,
                'image': img
            })
        else:
            print(f"\n✗ Failed to decode image")
    
    return successful_decodings

def visualize_samples(successful_decodings, save_path=None):
    """Visualize successfully decoded images."""
    if not successful_decodings:
        print("\nNo images to visualize")
        return
    
    print("\n" + "="*60)
    print("IMAGE VISUALIZATION")
    print("="*60)
    
    n_samples = min(len(successful_decodings), 6)
    fig, axes = plt.subplots(1, n_samples, figsize=(3*n_samples, 3))
    
    if n_samples == 1:
        axes = [axes]
    
    for i, data in enumerate(successful_decodings[:n_samples]):
        img = data['image']
        
        # Ensure image is in displayable format
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Sample {data['sample']}\n{data['method']}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def get_dataset_stats(tfrecord_dir):
    """Get statistics about the entire dataset."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    tfrecord_files = [f for f in os.listdir(tfrecord_dir) 
                     if f.endswith('.tfrecord') or f.endswith('.tfrecords')]
    
    print(f"Total tfrecord files: {len(tfrecord_files)}")
    
    if tfrecord_files:
        # Sample first file for patch count
        first_file = os.path.join(tfrecord_dir, tfrecord_files[0])
        dataset = tf.data.TFRecordDataset(first_file)
        patch_count = sum(1 for _ in dataset)
        print(f"Patches in first file ({tfrecord_files[0]}): {patch_count}")
        
        # File sizes
        total_size = 0
        for f in tfrecord_files[:5]:  # Show first 5
            size_mb = os.path.getsize(os.path.join(tfrecord_dir, f)) / (1024*1024)
            total_size += size_mb
            print(f"  {f}: {size_mb:.2f} MB")
        
        if len(tfrecord_files) > 5:
            print(f"  ... and {len(tfrecord_files)-5} more files")

def main():
    """Main testing function."""
    print("="*60)
    print("ECU TFRECORD FORMAT TESTER")
    print("="*60)
    
    # ============ UPDATE THIS PATH ============
    ECU_DIR = "data/raw/ECU"  # Directory containing ECU tfrecords
    # ==========================================
    
    if not os.path.exists(ECU_DIR):
        print(f"\n❌ Directory not found: {ECU_DIR}")
        print("Please update the ECU_DIR path in the script")
        return
    
    # Get list of tfrecord files
    tfrecord_files = [f for f in os.listdir(ECU_DIR) 
                     if f.endswith('.tfrecord') or f.endswith('.tfrecords')]
    
    if not tfrecord_files:
        print(f"\n❌ No tfrecord files found in {ECU_DIR}")
        return
    
    print(f"\nFound {len(tfrecord_files)} tfrecord files")
    print(f"Testing with: {tfrecord_files[0]}")
    
    test_file = os.path.join(ECU_DIR, tfrecord_files[0])
    
    # Run tests
    try:
        # 1. Test structure
        test_tfrecord_structure(test_file)
        
        # 2. Test image decoding
        successful = test_image_decoding(test_file, num_samples=3)
        
        # 3. Visualize samples
        if successful:
            visualize_samples(successful, save_path="ecu_sample_patches.png")
        
        # 4. Dataset statistics
        get_dataset_stats(ECU_DIR)
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        if successful:
            methods = set(d['method'] for d in successful)
            print(f"✓ Images can be decoded successfully")
            print(f"  Decoding method(s): {', '.join(methods)}")
            print(f"  Image shape: {successful[0]['shape']}")
            print(f"\n✓ Ready for feature extraction!")
        else:
            print("✗ Failed to decode images")
            print("  Check the encoding format of your patches")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()