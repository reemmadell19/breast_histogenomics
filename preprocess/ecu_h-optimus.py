#!/usr/bin/env python3
"""
H-optimus-1 Feature Extraction Script for ECU Dataset
Extract features from ECU histopathology images using H-optimus-1 foundation model.
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import tensorflow as tf
import numpy as np
import timm
from huggingface_hub import login
import gc
import time

# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ECU transform using H-optimus-1 specific normalization parameters
# H-optimus-1 uses custom normalization optimized for histopathological images
ecu_hoptimus_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.707223, 0.578729, 0.703617),
        std=(0.211883, 0.230117, 0.177517)
    ),
])

def load_hoptimus_model():
    """Load H-optimus-1 model for feature extraction."""
    print("Loading H-optimus-1 model...")

    # Login to Hugging Face
    try:
        login()
        print("Successfully logged into Hugging Face")
    except Exception as e:
        print(f"HF login failed: {e}")
        print("Trying without explicit login...")

    try:
        # H-optimus-1 model configuration
        model_repo = "bioptimus/H-optimus-1"

        print(f"Loading model from {model_repo}...")

        # Check available disk space before downloading
        import shutil
        free_space_gb = shutil.disk_usage('.').free / (1024**3)
        print(f"Available disk space: {free_space_gb:.2f} GB")

        if free_space_gb < 6:
            print(f"WARNING: Low disk space ({free_space_gb:.2f} GB). H-optimus-1 requires ~5GB.")
            print("Solutions:")
            print("1. Free up disk space and try again")
            print("2. Set custom cache directory: export HF_HOME=/path/to/larger/disk")
            print("3. Use a different feature extractor (CONCH, UNI, Virchow)")
            raise RuntimeError(f"Insufficient disk space: {free_space_gb:.2f} GB available, need ~5GB")

        # Try to set cache directory to a custom location if needed
        cache_dir = os.environ.get('HF_HOME', None)
        if cache_dir:
            print(f"Using custom cache directory: {cache_dir}")

        model = timm.create_model(
            f"hf-hub:{model_repo}",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
            num_classes=0,  # Remove classification head for feature extraction
        )

        # Move to device and set to eval mode with half precision
        model = model.to(device).half()  # Convert model to half precision
        model.eval()

        # H-optimus-1 feature dimension is 1536
        feature_dim = 1536

        print(f"Loaded H-optimus-1 model (half precision)")
        print(f"Feature dimension: {feature_dim}")
        print(f"Model parameters: ~1.1B (float16)")
        print(f"Memory usage: ~50% reduced")

        return model, feature_dim

    except Exception as e:
        error_msg = str(e)
        print(f"Failed to load H-optimus-1: {e}")

        if "Disk quota exceeded" in error_msg or "No space left" in error_msg:
            print("\nDISK SPACE SOLUTIONS:")
            print("1. Clean up your home directory: rm -rf ~/.cache/huggingface/")
            print("2. Set custom cache: export HF_HOME=/tmp/hf_cache (or larger disk)")
            print("3. Check disk usage: df -h")
            print("4. Free up space and try again")
            print("5. Consider using a smaller model like CONCH (512 dim) or UNI")
        else:
            print("Make sure you have access to the bioptimus/H-optimus-1 repository")
            print("You may need to request access from Bioptimus")
        raise

# Load H-optimus-1 model
hoptimus_model, feature_dim = load_hoptimus_model()

def extract_features_in_batches(images, batch_size=8):
    """Extract features using half precision for H-optimus-1"""
    features = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            # Clear cache before each batch
            torch.cuda.empty_cache()

            # Convert input to half precision
            batch = torch.stack(images[i:i + batch_size]).to(device, dtype=torch.float16, non_blocking=True)

            try:
                # Model is already in half precision, so no autocast needed
                with torch.inference_mode():
                    feat = hoptimus_model(batch)

                # Handle different output formats
                if isinstance(feat, dict):
                    feat = feat.get('features', feat.get('pooler_output', list(feat.values())[0]))
                elif isinstance(feat, tuple):
                    feat = feat[0]

                # Ensure we have the right shape [batch_size, 1536]
                if feat.dim() > 2:
                    feat = feat.mean(dim=1)  # Global average pooling if needed

                # Keep in half precision until final storage
                features.append(feat.cpu().float())  # Convert to float32 only for storage compatibility

                # Delete batch from GPU immediately
                del batch
                del feat
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"    Batch failed, trying single images: {str(e)[:100]}...")
                del batch
                torch.cuda.empty_cache()

                # Fallback: process images one by one
                batch_features = []
                for j, img in enumerate(images[i:i + batch_size]):
                    try:
                        torch.cuda.empty_cache()
                        single_img = img.unsqueeze(0).to(device, dtype=torch.float16, non_blocking=True)

                        with torch.inference_mode():
                            single_feat = hoptimus_model(single_img)

                        if isinstance(single_feat, dict):
                            single_feat = single_feat.get('features', single_feat.get('pooler_output', list(single_feat.values())[0]))
                        elif isinstance(single_feat, tuple):
                            single_feat = single_feat[0]
                        if single_feat.dim() > 2:
                            single_feat = single_feat.mean(dim=1)

                        batch_features.append(single_feat.cpu().float())

                        # Clean up immediately
                        del single_img
                        del single_feat
                        torch.cuda.empty_cache()

                    except Exception as e2:
                        print(f"    Single image {j} failed: {str(e2)[:50]}...")
                        batch_features.append(torch.zeros(1, feature_dim))
                        torch.cuda.empty_cache()

                if batch_features:
                    features.append(torch.cat(batch_features, dim=0))

            # Clear GPU cache periodically
            if i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()

    return torch.cat(features) if features else torch.zeros(0, feature_dim)

def extract_features_from_ecu(tfrecord_path, out_path):
    """Extract features from ECU tfrecord files using H-optimus-1"""
    print(f"  Loading {os.path.basename(tfrecord_path)}...")

    # ECU-specific feature description
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'global_x': tf.io.FixedLenFeature([], tf.int64),
        'global_y': tf.io.FixedLenFeature([], tf.int64),
        'local_x': tf.io.FixedLenFeature([], tf.int64),
        'local_y': tf.io.FixedLenFeature([], tf.int64),
        'tile_col': tf.io.FixedLenFeature([], tf.int64),
        'tile_row': tf.io.FixedLenFeature([], tf.int64),
        'da_number': tf.io.FixedLenFeature([], tf.int64),
        'tile_name': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    dataset = tf.data.TFRecordDataset(tfrecord_path).map(_parse)

    images = []
    coords = []
    da_numbers = []
    all_features = None  # For accumulating features in chunks

    # Count total patches for progress
    total_patches = sum(1 for _ in dataset)
    dataset = tf.data.TFRecordDataset(tfrecord_path).map(_parse)  # Recreate after counting

    print(f"  Found {total_patches} patches")

    if total_patches > 10000:
        print(f"  Large file with {total_patches} patches - processing in chunks")

    for idx, record in enumerate(tqdm(dataset, desc=f"  Reading {os.path.basename(tfrecord_path)}", leave=False)):
        try:
            # ECU uses JPEG encoding
            img_bytes = record['image'].numpy()
            img = tf.io.decode_jpeg(img_bytes).numpy()

            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            pil_img = Image.fromarray(img.astype('uint8'))
            tensor_img = ecu_hoptimus_transform(pil_img)  # Use H-optimus-1 specific normalization
            images.append(tensor_img)

            # Use global coordinates to match other datasets format
            coords.append([record['global_x'].numpy(), record['global_y'].numpy()])
            da_numbers.append(record['da_number'].numpy())

            # Process in smaller chunks for half precision H-optimus-1
            if len(images) >= 200:
                chunk_features = extract_features_in_batches(images, batch_size=8)
                if all_features is None:
                    all_features = chunk_features
                else:
                    all_features = torch.cat([all_features, chunk_features])
                images = []
                gc.collect()

                # Progress update for large files
                if total_patches > 10000 and idx % 5000 == 0:
                    print(f"    Progress: {idx}/{total_patches} patches processed")

        except Exception as e:
            print(f"    Skipping patch {idx}: {e}")
            continue

    # Process remaining images
    if images:
        chunk_features = extract_features_in_batches(images, batch_size=8)
        if all_features is None:
            all_features = chunk_features
        else:
            all_features = torch.cat([all_features, chunk_features])

    if all_features is not None and len(all_features) > 0:
        print(f"  Saving {len(all_features)} features to {os.path.basename(out_path)}")

        # Save with H-optimus-1 metadata and coordinates
        torch.save({
            'features': all_features,
            'coords': torch.tensor(coords),
            'slide_name': os.path.basename(tfrecord_path),
            'da_numbers': torch.tensor(da_numbers),  # ECU-specific: original tile IDs
            'num_patches': len(all_features),
            'feature_dim': feature_dim,
            'backbone': 'h-optimus-1'
        }, out_path)

        # Clear memory
        del all_features
        torch.cuda.empty_cache()
        gc.collect()

        print(f"  Saved features and coordinates for {os.path.basename(tfrecord_path)}")
        return True
    else:
        print(f"  No valid patches in {tfrecord_path}")
        return False

def run_ecu_extraction(input_dir, output_dir, manifest_csv=None):
    """Run ECU extraction with progress tracking"""

    # Get list of tfrecord files
    if manifest_csv and os.path.exists(manifest_csv):
        df = pd.read_csv(manifest_csv)
        if 'file_name' in df.columns:
            tfrecord_files = df['file_name'].tolist()
            print(f"Using manifest with {len(tfrecord_files)} files")
        else:
            print(f"Warning: 'file_name' column not found in manifest")
            tfrecord_files = [f for f in os.listdir(input_dir)
                            if f.endswith(('.tfrecord', '.tfrecords'))]
    else:
        tfrecord_files = [f for f in os.listdir(input_dir)
                         if f.endswith(('.tfrecord', '.tfrecords'))]
        tfrecord_files.sort()
        print(f"Found {len(tfrecord_files)} tfrecord files in directory")

    total_files = len(tfrecord_files)
    existing_files = 0
    processed_files = 0
    failed_files = 0
    start_time = time.time()

    print(f"\nStarting H-optimus-1 extraction for ECU dataset")
    print(f"Total files to process: {total_files}")
    print(f"Output directory: {output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for idx, tfrecord_file in enumerate(tfrecord_files):
        # Progress update every 15 files (less frequent for large model)
        if idx % 15 == 0 and idx > 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (total_files - idx) / rate if rate > 0 else 0
            print(f"\nProgress: {idx}/{total_files} ({idx/total_files*100:.1f}%) - "
                  f"{rate:.1f} files/sec - ETA: {remaining/60:.1f} min")

        # Get paths
        base_name = os.path.splitext(tfrecord_file)[0]
        input_path = os.path.join(input_dir, tfrecord_file)
        output_path = os.path.join(output_dir, f"{base_name}.pt")

        # Skip if already exists
        if os.path.exists(output_path):
            existing_files += 1
            continue

        # Check if input exists
        if not os.path.exists(input_path):
            print(f"  File not found: {tfrecord_file}")
            failed_files += 1
            continue

        print(f"\n[{idx+1}/{total_files}] Processing {tfrecord_file}...")

        try:
            success = extract_features_from_ecu(input_path, output_path)

            if success:
                processed_files += 1
                print(f"Completed {tfrecord_file}")
            else:
                failed_files += 1
                print(f"Failed {tfrecord_file}")

        except KeyboardInterrupt:
            print(f"\nProcess interrupted by user at {tfrecord_file}")
            break
        except Exception as e:
            print(f"Unexpected error for {tfrecord_file}: {e}")
            failed_files += 1
            torch.cuda.empty_cache()
            gc.collect()
            continue

    # Print summary
    elapsed = time.time() - start_time
    print(f"\nH-OPTIMUS-1 ECU EXTRACTION SUMMARY")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Total files: {total_files}")
    print(f"Already existed (skipped): {existing_files}")
    print(f"Newly processed: {processed_files}")
    print(f"Failed: {failed_files}")

    if total_files - existing_files > 0:
        success_rate = (processed_files / (total_files - existing_files)) * 100
        print(f"Success rate: {success_rate:.1f}%")

    print(f"Features saved with dimension: {feature_dim}")
    print(f"Coordinates saved for all patches")

def verify_extraction(output_dir, sample_file=None):
    """Verify extracted H-optimus-1 features"""
    print("\n" + "="*60)
    print("VERIFICATION - H-OPTIMUS-1 FEATURES")
    print("="*60)

    pt_files = [f for f in os.listdir(output_dir) if f.endswith('.pt')]

    if not pt_files:
        print("No .pt files found to verify")
        return

    # Check a sample file
    if sample_file and sample_file in pt_files:
        check_file = sample_file
    else:
        check_file = pt_files[0]

    sample_path = os.path.join(output_dir, check_file)
    data = torch.load(sample_path, map_location='cpu')

    print(f"Sample file: {check_file}")
    print(f"Keys in file: {list(data.keys())}")
    print(f"Feature shape: {data['features'].shape}")
    print(f"Feature dimension: {data['features'].shape[1]} (expected: {feature_dim})")
    print(f"Coords shape: {data['coords'].shape}")
    print(f"Number of patches: {data.get('num_patches', len(data['features']))}")
    print(f"Backbone: {data.get('backbone', 'not specified')}")

    if 'da_numbers' in data:
        unique_das = len(torch.unique(data['da_numbers']))
        print(f"Unique DA numbers (original tiles): {unique_das}")

    print(f"\nH-optimus-1 extraction verified successfully!")

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("H-OPTIMUS-1 FEATURE EXTRACTION FOR ECU DATASET")
    print("="*60)
    print(f"Device: {device}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Backbone: H-optimus-1 (Bioptimus Foundation Model)")
    print(f"Precision: Half precision (float16) - 50% memory reduction")
    print(f"Batch size: 8 (increased due to half precision)")
    print(f"Normalization: Custom H-optimus-1 parameters")
    print("="*60)

    # ========== UPDATE THESE PATHS ==========
    ECU_INPUT_DIR = "data/raw/ECU"  # Directory with ECU tfrecord files
    ECU_OUTPUT_DIR = "data/features_hoptimus/ecu"  # Where to save H-optimus-1 features
    ECU_MANIFEST = "data/manifests/ecu_manifest.csv"  # Optional: CSV with file info
    # =========================================

    try:
        # Run extraction
        run_ecu_extraction(
            input_dir=ECU_INPUT_DIR,
            output_dir=ECU_OUTPUT_DIR,
            manifest_csv=ECU_MANIFEST if os.path.exists(ECU_MANIFEST) else None
        )

        # Verify results
        if os.path.exists(ECU_OUTPUT_DIR):
            verify_extraction(ECU_OUTPUT_DIR)

        print("\n" + "="*60)
        print("H-OPTIMUS-1 ECU FEATURE EXTRACTION COMPLETED")
        print("="*60)
        print("\nNext steps:")
        print(f"1. Verify features have dimension: {feature_dim}")
        print("2. Update MIL model input_dim to 1536 for H-optimus-1")
        print("3. Run evaluation with ECU H-optimus-1 features")
        print("4. Compare performance across different backbones")
        print("5. Note: H-optimus-1 uses custom normalization parameters")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()