import os
import torch
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import h5py
import tensorflow as tf
import numpy as np
import timm
from huggingface_hub import login
import gc
import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# H-optimus-1 specific transforms - uses custom normalization parameters
# optimized for histopathological image statistics
hoptimus_transform = transforms.Compose([
    transforms.CenterCrop(224),
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
    """Extract features using full half precision for H-optimus-1"""
    features = []

    print(f"    Processing {len(images)} images in batches of {batch_size} (half precision)")

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

            # Progress indicator
            if (i // batch_size) % 10 == 0:
                print(f"    Processed {i + batch_size}/{len(images)} images")

    return torch.cat(features) if features else torch.zeros(0, feature_dim)

def extract_features_from_bcrnet(h5_path, out_path):
    """Extract features from BCR-NET h5 files with coordinates"""
    print(f"  Loading {os.path.basename(h5_path)}...")

    try:
        with h5py.File(h5_path, 'r') as f:
            if 'bag' not in f:
                print(f"  No 'bag' dataset in {h5_path}")
                return False

            patches = f['bag'][:]
            coords = f['coords'][:]
            print(f"  Found {len(patches)} patches, shape: {patches.shape}")
            print(f"  Coordinates shape: {coords.shape}")

            if len(patches) > 50000:
                print(f"  Large file with {len(patches)} patches - this may take a while")

    except Exception as e:
        print(f"  Error reading {h5_path}: {e}")
        return False

    images = []
    skipped_patches = 0

    print(f"  Processing patches...")

    for i, patch in enumerate(tqdm(patches, desc=f"  Processing {os.path.basename(h5_path)}", leave=False)):
        try:
            if patch.shape[0] == 3:
                patch = patch.transpose(1, 2, 0)

            img = Image.fromarray(patch.astype('uint8'))
            tensor_img = hoptimus_transform(img)
            images.append(tensor_img)

            # Process in smaller chunks for half precision H-optimus-1
            if len(images) >= 200:
                chunk_features = extract_features_in_batches(images, batch_size=8)
                if 'all_features' not in locals():
                    all_features = chunk_features
                else:
                    all_features = torch.cat([all_features, chunk_features])
                images = []
                gc.collect()

        except Exception as e:
            skipped_patches += 1
            if skipped_patches <= 5:
                print(f"    Skipping patch {i}: {e}")
            continue

    # Process remaining images
    if images:
        chunk_features = extract_features_in_batches(images, batch_size=8)
        if 'all_features' not in locals():
            all_features = chunk_features
        else:
            all_features = torch.cat([all_features, chunk_features])

    if 'all_features' in locals() and len(all_features) > 0:
        print(f"  Saving {len(all_features)} features to {os.path.basename(out_path)}")

        # Save with H-optimus-1 metadata and coordinates
        torch.save({
            'features': all_features,
            'coords': torch.tensor(coords.T),  # transpose to match UCMC format
            'slide_name': os.path.basename(h5_path),
            'num_patches': len(all_features),
            'feature_dim': feature_dim,
            'backbone': 'h-optimus-1'
        }, out_path)

        # Clear memory
        del all_features
        torch.cuda.empty_cache()
        gc.collect()

        if skipped_patches > 0:
            print(f"  Skipped {skipped_patches} patches due to errors")

        print(f"  Saved features and coordinates for {os.path.basename(h5_path)}")
        return True
    else:
        print(f"  No valid patches extracted from {h5_path}")
        return False

def extract_features_from_ucmc(tfrecord_path, out_path):
    """Extract features from UCMC tfrecord files using H-optimus-1"""
    print(f"  Loading {os.path.basename(tfrecord_path)}...")

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
            tensor_img = hoptimus_transform(pil_img)
            images.append(tensor_img)
            coords.append([record['loc_x'].numpy(), record['loc_y'].numpy()])
            valid_patches += 1
        except Exception as e:
            print(f"    Skipping patch in {tfrecord_path}: {e}")
            continue

    if images:
        print(f"  Computing H-optimus-1 features for {valid_patches} patches...")
        features = extract_features_in_batches(images, batch_size=8)

        print(f"  Saving features to {os.path.basename(out_path)}")

        # Save with H-optimus-1 metadata and coordinates
        torch.save({
            'features': features,
            'coords': torch.tensor(coords),
            'slide_name': os.path.basename(tfrecord_path),
            'num_patches': len(features),
            'feature_dim': feature_dim,
            'backbone': 'h-optimus-1'
        }, out_path)

        torch.cuda.empty_cache()
        gc.collect()
        print(f"  Saved features and coordinates for {os.path.basename(tfrecord_path)}")
        return True
    else:
        print(f"  No valid patches in {tfrecord_path}")
        return False

def run_extraction(manifest_path, out_dir):
    """Run extraction with better progress tracking and error handling"""
    df = pd.read_csv(manifest_path)

    total_files = len(df)
    existing_files = 0
    processed_files = 0
    failed_files = 0
    start_time = time.time()

    print(f"\nStarting H-optimus-1 extraction for {manifest_path}")
    print(f"Total files to process: {total_files}")
    print(f"Output directory: {out_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    for idx, row in enumerate(df.iterrows()):
        _, row = row
        fname = row['file_name']
        dataset = row['dataset']

        # Progress update every 15 files (less frequent for large model)
        if idx % 15 == 0 and idx > 0:
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
    print(f"\nH-OPTIMUS-1 EXTRACTION SUMMARY for {os.path.basename(manifest_path)}")
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
    print("H-OPTIMUS-1 FEATURE EXTRACTION WITH COORDINATES")
    print("="*60)
    print(f"Device: {device}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Backbone: H-optimus-1 (Bioptimus Foundation Model)")
    print(f"Precision: Half precision (float16) - 50% memory reduction")
    print(f"Batch size: 8 (increased due to half precision)")
    print("="*60)

    # Create output directories for H-optimus-1 features
    base_dirs = {
        "train": "data/features_hoptimus/train",
        "val": "data/features_hoptimus/val",
        "test": "data/features_hoptimus/test"
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
        print("H-OPTIMUS-1 FEATURE EXTRACTION COMPLETED")
        print("="*60)
        print("\nNext steps:")
        print(f"1. Generate H-optimus-1 manifests with feature dimension: {feature_dim}")
        print("2. Update MIL model input_dim to match H-optimus-1 features")
        print("3. Run training scripts with H-optimus-1 features")
        print("4. Compare performance with CONCH and ResNet features")
        print("5. Use coordinates for interpretability analysis")
        print("6. Note: H-optimus-1 uses custom normalization parameters")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")