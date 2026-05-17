
import os
import torch
import pandas as pd
from tqdm import tqdm
from torchvision import models, transforms
from PIL import Image
import tensorflow as tf
import numpy as np
import gc
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transform for ECU (224x224 patches, no cropping needed)
ecu_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load pretrained ResNet50
print("Loading ResNet50...")
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()  # Remove classification layer
resnet.eval()
resnet = resnet.to(device)

# ResNet50 outputs 2048 features vs ResNet18's 512
FEATURE_DIM = 2048
print(f"ResNet50 feature dimension: {FEATURE_DIM}")
print(f"Device: {device}")

# -------------- Helper Functions --------------

def extract_features_in_batches(images, batch_size=12):
    """
    Extract ResNet50 features in smaller batches for memory efficiency.
    ResNet50 is larger than ResNet18, so we use smaller batch size.
    """
    features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i + batch_size]).to(device)
            feat = resnet(batch)
            features.append(feat.cpu())

            # Clear GPU memory more frequently for ResNet50
            if i % (batch_size * 3) == 0:
                torch.cuda.empty_cache()

    return torch.cat(features)

def extract_features_from_ecu_tfrecord(tfrecord_path, out_path):
    """
    Extract ResNet50 features from a single ECU tfrecord file.

    Args:
        tfrecord_path: Path to the tfrecord file
        out_path: Path to save the extracted features (.pt file)
    """

    # Define the feature description for ECU tfrecords
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

    # Count total patches for progress bar
    total_patches = sum(1 for _ in dataset)
    dataset = tf.data.TFRecordDataset(tfrecord_path).map(_parse)  # Recreate after counting

    print(f"  Found {total_patches} patches in {os.path.basename(tfrecord_path)}")

    if total_patches > 10000:
        print(f"  Large file with {total_patches} patches - processing in chunks")

    # Process each patch
    skipped_patches = 0
    for idx, record in enumerate(dataset):
        try:
            # Decode JPEG image
            img_bytes = record['image'].numpy()
            img = tf.io.decode_jpeg(img_bytes).numpy()

            # Convert to PIL and apply transforms
            pil_img = Image.fromarray(img.astype('uint8'))
            tensor_img = ecu_transform(pil_img)
            images.append(tensor_img)

            # Store coordinates (using global coordinates for consistency)
            coords.append([
                record['global_x'].numpy(),
                record['global_y'].numpy()
            ])

            # Store DA number (original tile ID)
            da_numbers.append(record['da_number'].numpy())

            # Process in chunks to avoid memory issues with ResNet50
            if len(images) >= 800:  # Smaller chunks for ResNet50
                chunk_features = extract_features_in_batches(images, batch_size=12)
                if all_features is None:
                    all_features = chunk_features
                else:
                    all_features = torch.cat([all_features, chunk_features])

                images = []  # Clear the list
                gc.collect()  # Force garbage collection

                # Progress update for large files
                if total_patches > 10000 and idx % 5000 == 0:
                    print(f"    Progress: {idx}/{total_patches} patches processed")

        except Exception as e:
            skipped_patches += 1
            if skipped_patches <= 5:  # Only print first few errors
                print(f"    Skipping patch {idx}: {e}")
            continue

    # Process remaining images
    if images:
        chunk_features = extract_features_in_batches(images, batch_size=12)
        if all_features is None:
            all_features = chunk_features
        else:
            all_features = torch.cat([all_features, chunk_features])

    # Save features if we have valid patches
    if all_features is not None and len(all_features) > 0:
        print(f"  Saving {len(all_features)} features to {os.path.basename(out_path)}")

        # Save in consistent format with metadata
        torch.save({
            'features': all_features,  # Shape: [N, 2048]
            'coords': torch.tensor(coords),  # Shape: [N, 2]
            'slide_name': os.path.basename(tfrecord_path),
            'da_numbers': torch.tensor(da_numbers),  # ECU-specific: original tile IDs
            'num_patches': len(all_features),
            'feature_dim': FEATURE_DIM,
            'backbone': 'resnet50'
        }, out_path)

        # Clear memory
        del all_features
        torch.cuda.empty_cache()
        gc.collect()

        if skipped_patches > 0:
            print(f"  Skipped {skipped_patches} patches due to errors")

        return len(coords)
    else:
        print(f"  No valid patches in {os.path.basename(tfrecord_path)}")
        return 0

def run_ecu_extraction(input_dir, output_dir, manifest_csv=None):
    """
    Run ResNet50 feature extraction for all ECU tfrecord files.

    Args:
        input_dir: Directory containing ECU tfrecord files
        output_dir: Directory to save extracted features
        manifest_csv: Optional CSV with file names and RS scores
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of tfrecord files
    if manifest_csv and os.path.exists(manifest_csv):
        # Use manifest if provided
        df = pd.read_csv(manifest_csv)
        if 'file_name' in df.columns:
            tfrecord_files = df['file_name'].tolist()
            print(f"Using manifest with {len(tfrecord_files)} files")
        else:
            print(f"Warning: 'file_name' column not found in manifest")
            tfrecord_files = [f for f in os.listdir(input_dir)
                            if f.endswith(('.tfrecord', '.tfrecords'))]
    else:
        # Scan directory for tfrecord files
        tfrecord_files = [f for f in os.listdir(input_dir)
                         if f.endswith(('.tfrecord', '.tfrecords'))]
        tfrecord_files.sort()  # Sort for consistent ordering
        print(f"Found {len(tfrecord_files)} tfrecord files in directory")

    # Processing statistics
    total_files = len(tfrecord_files)
    existing_files = 0
    processed_files = 0
    failed_files = 0
    total_patches = 0
    start_time = time.time()

    print("\n" + "="*60)
    print("ECU RESNET50 FEATURE EXTRACTION")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total files to process: {total_files}")
    print(f"Device: {device}")
    print(f"Feature dimension: {FEATURE_DIM}")
    print(f"Backbone: ResNet50")
    print("="*60 + "\n")

    # Process each tfrecord file
    for idx, tfrecord_file in enumerate(tfrecord_files):
        # Progress update every 25 files
        if idx % 25 == 0 and idx > 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (total_files - idx) / rate if rate > 0 else 0
            print(f"\nProgress: {idx}/{total_files} ({idx/total_files*100:.1f}%) - "
                  f"{rate:.1f} files/sec - ETA: {remaining/60:.1f} min")

        # Get paths
        base_name = os.path.splitext(tfrecord_file)[0]
        input_path = os.path.join(input_dir, tfrecord_file)
        output_path = os.path.join(output_dir, f"{base_name}.pt")

        # Skip if already processed
        if os.path.exists(output_path):
            existing_files += 1
            continue

        # Check if input exists
        if not os.path.exists(input_path):
            print(f"  File not found: {tfrecord_file}")
            failed_files += 1
            continue

        print(f"\n[{idx+1}/{total_files}] Processing {tfrecord_file}...")

        # Process the file
        try:
            num_patches = extract_features_from_ecu_tfrecord(input_path, output_path)
            if num_patches > 0:
                processed_files += 1
                total_patches += num_patches
                print(f"  Completed {tfrecord_file} ({num_patches} patches)")
            else:
                failed_files += 1

        except KeyboardInterrupt:
            print(f"\nProcess interrupted by user at {tfrecord_file}")
            break
        except Exception as e:
            print(f"  Failed processing {tfrecord_file}: {e}")
            failed_files += 1
            torch.cuda.empty_cache()
            gc.collect()
            continue

    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("RESNET50 EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Total files: {total_files}")
    print(f"Already existed (skipped): {existing_files}")
    print(f"Successfully processed: {processed_files}")
    print(f"Failed: {failed_files}")

    if processed_files > 0:
        avg_patches = total_patches / processed_files
        avg_time = elapsed / processed_files if processed_files > 0 else 0
        print(f"Total patches processed: {total_patches}")
        print(f"Average patches per slide: {avg_patches:.1f}")
        print(f"Average time per slide: {avg_time:.1f} seconds")

    if total_files - existing_files > 0:
        success_rate = (processed_files / (total_files - existing_files)) * 100
        print(f"Success rate: {success_rate:.1f}%")

    print(f"Features saved with dimension: {FEATURE_DIM}")
    print(f"Coordinates saved for all patches")
    print("="*60)

def verify_extraction(output_dir, sample_file=None):
    """
    Verify extracted ResNet50 features by loading and checking a sample.

    Args:
        output_dir: Directory containing extracted features
        sample_file: Optional specific file to check
    """
    print("\n" + "="*60)
    print("VERIFICATION - RESNET50 FEATURES")
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
    data = torch.load(sample_path)

    print(f"Sample file: {check_file}")
    print(f"Keys in file: {list(data.keys())}")
    print(f"Feature shape: {data['features'].shape}")
    print(f"Feature dimension: {data['features'].shape[1]} (should be {FEATURE_DIM})")
    print(f"Coords shape: {data['coords'].shape}")
    print(f"Number of patches: {data.get('num_patches', len(data['features']))}")
    print(f"Backbone: {data.get('backbone', 'not specified')}")

    if 'da_numbers' in data:
        unique_das = len(torch.unique(data['da_numbers']))
        print(f"Unique DA numbers (original tiles): {unique_das}")

    # Verify feature dimension
    if data['features'].shape[1] == FEATURE_DIM:
        print(f"\nFeature dimension verified: {FEATURE_DIM}")
    else:
        print(f"\nWarning: Feature dimension mismatch!")

    print(f"\nResNet50 extraction verified successfully!")

# -------------- Main --------------
if __name__ == "__main__":
    # ========== UPDATE THESE PATHS ==========
    ECU_INPUT_DIR = "data/raw/ECU"  # Directory with ECU tfrecord files
    ECU_OUTPUT_DIR = "data/features_resnet50/ecu"  # Where to save ResNet50 features
    ECU_MANIFEST = "data/manifests/ecu_manifest.csv"  # Optional: CSV with file info
    # =========================================

    print("="*60)
    print("ECU Dataset ResNet50 Feature Extraction")
    print("="*60)
    print(f"Backbone: ResNet50")
    print(f"Feature dimension: {FEATURE_DIM}")
    print(f"Device: {device}")
    print("="*60)

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
        print("ECU ResNet50 feature extraction completed")
        print("="*60)
        print("\nNext steps:")
        print("1. Update MIL model input_dim from 512 to 2048 for ResNet50 features")
        print("2. Run evaluation with these ResNet50 features")
        print("3. Compare performance between ResNet18 and ResNet50")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()