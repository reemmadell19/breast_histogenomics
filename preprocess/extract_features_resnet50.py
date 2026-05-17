
import os
import torch
import pandas as pd
from tqdm import tqdm
from torchvision import models, transforms
from PIL import Image
import h5py
import tensorflow as tf
import numpy as np
import gc
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image transforms
basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

ucmc_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load pretrained ResNet50
print("Loading ResNet50...")
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()  # Remove classification head
resnet.eval()
resnet = resnet.to(device)

# ResNet50 outputs 2048 features vs ResNet18's 512 features
feature_dim = 2048
print(f"ResNet50 feature dimension: {feature_dim}")

def extract_features_in_batches(images, batch_size=12):
    """Extract features in smaller batches to avoid memory issues with ResNet50"""
    features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i + batch_size]).to(device)
            feat = resnet(batch)
            features.append(feat.cpu())

            if i % (batch_size * 3) == 0:
                torch.cuda.empty_cache()

    return torch.cat(features)

def extract_features_from_bcrnet(h5_path, out_path):
    """Extract features from BCR-NET h5 files with coordinates"""
    print(f"   Loading {os.path.basename(h5_path)}...")

    try:
        with h5py.File(h5_path, 'r') as f:
            # Check file structure first
            if 'bag' not in f:
                print(f"   No 'bag' dataset in {h5_path}")
                return False

            patches = f['bag'][:]
            coords = f['coords'][:]
            print(f"   Found {len(patches)} patches, shape: {patches.shape}")
            print(f"   Coordinates shape: {coords.shape}")

            # Check if file is too large
            if len(patches) > 50000:
                print(f"   Large file with {len(patches)} patches - this may take a while")

    except Exception as e:
        print(f"   Error reading {h5_path}: {e}")
        return False

    images = []
    skipped_patches = 0

    print(f"   Processing patches...")

    for i, patch in enumerate(tqdm(patches, desc=f"  Processing {os.path.basename(h5_path)}", leave=False)):
        try:
            if patch.shape[0] == 3:
                patch = patch.transpose(1, 2, 0)  # convert [3, H, W] to [H, W, 3]

            img = Image.fromarray(patch.astype('uint8'))
            tensor_img = basic_transform(img)
            images.append(tensor_img)

            # Process in smaller chunks for ResNet50 to avoid memory buildup
            if len(images) >= 800:
                chunk_features = extract_features_in_batches(images, batch_size=12)
                if 'all_features' not in locals():
                    all_features = chunk_features
                else:
                    all_features = torch.cat([all_features, chunk_features])
                images = []  # Clear the list
                gc.collect()  # Force garbage collection

        except Exception as e:
            skipped_patches += 1
            if skipped_patches <= 5:  # Only print first few errors
                print(f"      Skipping patch {i}: {e}")
            continue

    # Process remaining images
    if images:
        chunk_features = extract_features_in_batches(images, batch_size=12)
        if 'all_features' not in locals():
            all_features = chunk_features
        else:
            all_features = torch.cat([all_features, chunk_features])

    if 'all_features' in locals() and len(all_features) > 0:
        print(f"   Saving {len(all_features)} features to {os.path.basename(out_path)}")

        # Save with ResNet50 metadata and coordinates
        torch.save({
            'features': all_features,
            'coords': torch.tensor(coords.T),  # transpose to match UCMC format
            'slide_name': os.path.basename(h5_path),
            'num_patches': len(all_features),
            'feature_dim': feature_dim,
            'backbone': 'resnet50'
        }, out_path)

        # Clear memory
        del all_features
        torch.cuda.empty_cache()
        gc.collect()

        if skipped_patches > 0:
            print(f"   Skipped {skipped_patches} patches due to errors")

        print(f"   Saved features and coordinates for {os.path.basename(h5_path)}")
        return True
    else:
        print(f"   No valid patches extracted from {h5_path}")
        return False

def extract_features_from_ucmc(tfrecord_path, out_path):
    """Extract features from UCMC tfrecord files using ResNet50"""
    print(f"   Loading {os.path.basename(tfrecord_path)}...")

    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'slide': tf.io.FixedLenFeature([], tf.string),
        'loc_x': tf.io.FixedLenFeature([], tf.int64),
        'loc_y': tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    dataset = tf.data.TFRecordDataset(tfrecord_path).map(_parse)

    images = []
    coords = []
    valid_patches = 0

    for record in tqdm(dataset, desc=f"  Reading {os.path.basename(tfrecord_path)}", leave=False):
        try:
            raw = record['image_raw'].numpy()
            img = tf.io.decode_png(raw).numpy()
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            pil_img = Image.fromarray(img.astype('uint8'))
            tensor_img = ucmc_transform(pil_img)
            images.append(tensor_img)
            coords.append([record['loc_x'].numpy(), record['loc_y'].numpy()])
            valid_patches += 1
        except Exception as e:
            print(f"      Skipping patch in {tfrecord_path}: {e}")
            continue

    if images:
        print(f"   Computing ResNet50 features for {valid_patches} patches...")
        features = extract_features_in_batches(images, batch_size=12)

        print(f"   Saving features to {os.path.basename(out_path)}")

        # Save with ResNet50 metadata and coordinates
        torch.save({
            'features': features,
            'coords': torch.tensor(coords),
            'slide_name': os.path.basename(tfrecord_path),
            'num_patches': len(features),
            'feature_dim': feature_dim,
            'backbone': 'resnet50'
        }, out_path)

        torch.cuda.empty_cache()
        gc.collect()
        print(f"   Saved features and coordinates for {os.path.basename(tfrecord_path)}")
        return True
    else:
        print(f"   No valid patches in {tfrecord_path}")
        return False

def run_extraction(manifest_path, out_dir):
    """Run extraction with better progress tracking and error handling"""
    df = pd.read_csv(manifest_path)

    total_files = len(df)
    existing_files = 0
    processed_files = 0
    failed_files = 0
    start_time = time.time()

    print(f"\nStarting ResNet50 extraction for {manifest_path}")
    print(f"Total files to process: {total_files}")
    print(f"Output directory: {out_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    for idx, row in enumerate(df.iterrows()):
        _, row = row
        fname = row['file_name']
        dataset = row['dataset']

        # Progress update every 25 files
        if idx % 25 == 0 and idx > 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            print(f"  Progress: {idx}/{total_files} ({idx/total_files*100:.1f}%) - {rate:.1f} files/sec")

        # Prepend correct base path
        if dataset == "UCMC":
            full_path = os.path.join("data/raw/UCMC", fname)
        elif dataset == "BCRNet":
            full_path = os.path.join("data/raw/BCR_NET", fname)
        else:
            print(f"Unknown dataset type for {fname}")
            failed_files += 1
            continue

        slide_id = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, slide_id + ".pt")

        # Check if file already exists - skip if it does
        if os.path.exists(out_path):
            existing_files += 1
            continue

        print(f"\n[{idx+1}/{total_files}] Processing {fname}...")

        try:
            success = False
            if fname.endswith(".tfrecords"):
                success = extract_features_from_ucmc(full_path, out_path)
            elif fname.endswith(".h5"):
                success = extract_features_from_bcrnet(full_path, out_path)
            else:
                print(f"Unsupported file format: {fname}")
                failed_files += 1
                continue

            if success:
                processed_files += 1
                print(f"Completed {fname}")
            else:
                failed_files += 1
                print(f"Failed {fname}")

        except KeyboardInterrupt:
            print(f"\nProcess interrupted by user at {fname}")
            break
        except Exception as e:
            print(f"Unexpected error for {fname}: {e}")
            failed_files += 1
            torch.cuda.empty_cache()
            gc.collect()
            continue

    # Print summary
    elapsed = time.time() - start_time
    print(f"\nRESNET50 EXTRACTION SUMMARY for {os.path.basename(manifest_path)}")
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Total files: {total_files}")
    print(f"Already existed (skipped): {existing_files}")
    print(f"Newly processed: {processed_files}")
    print(f"Failed: {failed_files}")
    print(f"Success rate: {processed_files}/{total_files - existing_files} new files")
    print(f"Features saved with dimension: {feature_dim}")
    print(f"Coordinates saved for both UCMC and BCR-NET datasets")

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("RESNET50 FEATURE EXTRACTION WITH COORDINATES")
    print("="*60)
    print(f"Device: {device}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Backbone: ResNet50")
    print("="*60)

    # Create output directories for ResNet50 features
    base_dirs = {
        "train": "data/features_resnet50/train",
        "val": "data/features_resnet50/val",
        "test": "data/features_resnet50/test"
    }

    # Create directories
    for dir_path in base_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    try:
        # Extract features for each split
        run_extraction("data/manifests/train_manifest.csv", base_dirs["train"])
        run_extraction("data/manifests/val_manifest.csv", base_dirs["val"])
        run_extraction("data/manifests/test_manifest.csv", base_dirs["test"])

        print("\n" + "="*60)
        print("RESNET50 FEATURE EXTRACTION COMPLETED")
        print("="*60)
        print("\nNext steps:")
        print("1. Update MIL model input_dim from 512 to 2048")
        print("2. Run training scripts with ResNet50 features")
        print("3. Compare performance with ResNet18 features")
        print("4. Use coordinates for interpretability analysis")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")