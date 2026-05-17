
import os
import pandas as pd
from tqdm import tqdm
import timm
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.models as models
import torchvision.transforms as transforms
import warnings
from PIL import Image
import h5py
import tensorflow as tf
import numpy as np
from huggingface_hub import login


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image transforms
# ----------------------------------------------------
# 1) basic_transform: CenterCrop(224) + ToTensor & Normalize (for BCR-Net)
# 2) ucmc_transform: CenterCrop(224) + ToTensor & Normalize (for UCMC)
# Note: Both use CenterCrop(224) to preserve histopathology details
# ----------------------------------------------------
basic_transform = transforms.Compose([
    transforms.CenterCrop(224),
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

def load_uni_model(model_name='uni2-h'):
    """Load UNI model for feature extraction."""
    print(f"Loading {model_name} model...")

    # Login to Hugging Face
    login()

    if model_name == 'uni':
        model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            img_size=224,
            patch_size=16,
            num_classes=0
        )
        feature_dim = 1024

    elif model_name == 'uni2-h':
        timm_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        feature_dim = 1536

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print(f"Loaded {model_name} model")
    print(f"Feature dimension: {feature_dim}")
    return model, feature_dim

# Load UNI model (change model_name here if needed)
MODEL_NAME = 'uni2-h'  # Options: 'uni', 'uni2-h'
uni_model, feature_dim = load_uni_model(MODEL_NAME)

# -------------- Batch feature extraction --------------
def extract_features_in_batches(images, batch_size=16):
    """Extract features in batches to manage memory usage."""
    features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i + batch_size]).to(device)
            feat = uni_model(batch)
            features.append(feat.cpu())

            # Clear GPU cache periodically
            if i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()

    return torch.cat(features)

def extract_features_from_bcrnet(h5_path, out_path):
    """Extract features from BCR-Net H5 files using UNI with coordinates."""
    print(f"Processing BCR-Net file: {os.path.basename(h5_path)}")

    with h5py.File(h5_path, 'r') as f:
        patches = f['bag'][:]  # shape [N, 3, H, W] or [N, H, W, 3]
        coords = f['coords'][:]
        print(f"  Found {len(patches)} patches, shape: {patches.shape}")
        print(f"  Coordinates shape: {coords.shape}")

    images = []
    valid_patches = 0

    for i, patch in enumerate(tqdm(patches, desc=f"Reading {os.path.basename(h5_path)}")):
        try:
            if patch.shape[0] == 3:
                patch = patch.transpose(1, 2, 0)  # convert [3, H, W] to [H, W, 3]
            img = Image.fromarray(patch.astype('uint8'))
            tensor_img = basic_transform(img)
            images.append(tensor_img)
            valid_patches += 1
        except Exception as e:
            print(f"Skipping patch {i} in {h5_path}: {e}")
            continue

    if images:
        print(f"Extracted {valid_patches} valid patches, computing UNI features...")
        features = extract_features_in_batches(images)

        # Save with additional metadata and coordinates
        torch.save({
            'features': features,
            'coords': torch.tensor(coords.T),  # transpose to match UCMC format
            'slide_name': os.path.basename(h5_path),
            'num_patches': len(features),
            'feature_dim': feature_dim,
            'backbone': MODEL_NAME
        }, out_path)

        print(f"Saved features: {features.shape} to {out_path}")
        print(f"Saved features and coordinates for {os.path.basename(h5_path)}")
        torch.cuda.empty_cache()
    else:
        print(f"No valid patches extracted from {h5_path}")

def extract_features_from_ucmc(tfrecord_path, out_path):
    """Extract features from UCMC TFRecord files using UNI."""
    print(f"Processing UCMC file: {os.path.basename(tfrecord_path)}")

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

    for record in tqdm(dataset, desc=f"Reading {os.path.basename(tfrecord_path)}"):
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
            print(f"Skipping patch in {tfrecord_path}: {e}")
            continue

    if images:
        print(f"Extracted {valid_patches} valid patches, computing UNI features...")
        features = extract_features_in_batches(images)

        # Save with additional metadata and coordinates
        torch.save({
            'features': features,
            'coords': torch.tensor(coords),
            'slide_name': os.path.basename(tfrecord_path),
            'num_patches': len(features),
            'feature_dim': feature_dim,
            'backbone': MODEL_NAME
        }, out_path)

        print(f"Saved features: {features.shape} to {out_path}")
        print(f"Saved features and coordinates for {os.path.basename(tfrecord_path)}")
        torch.cuda.empty_cache()
    else:
        print(f"No valid patches in {tfrecord_path}")


def run_extraction(manifest_path, out_dir):
    """Run feature extraction for a given manifest."""
    print(f"\n{'='*60}")
    print(f"EXTRACTING FEATURES FROM: {manifest_path}")
    print(f"OUTPUT DIRECTORY: {out_dir}")
    print(f"{'='*60}")

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(manifest_path)
    print(f"Found {len(df)} files to process")

    successful = 0
    failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(manifest_path)}"):
        fname = row['file_name']
        dataset = row['dataset']

        # Prepend correct base path
        if dataset == "UCMC":
            full_path = os.path.join("data/raw/UCMC", fname)
        elif dataset == "BCRNet":
            full_path = os.path.join("data/raw/BCR_NET", fname)
        else:
            print(f"Unknown dataset type for {fname}")
            failed += 1
            continue

        slide_id = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, slide_id + ".pt")

        # Skip if already processed
        if os.path.exists(out_path):
            print(f"Skipping {fname} - already processed")
            successful += 1
            continue

        try:
            if fname.endswith(".tfrecords"):
                extract_features_from_ucmc(full_path, out_path)
            elif fname.endswith(".h5"):
                extract_features_from_bcrnet(full_path, out_path)
            else:
                print(f"Unsupported file format: {fname}")
                failed += 1
                continue

            successful += 1

        except Exception as e:
            print(f"Failed for {fname}: {e}")
            failed += 1
            torch.cuda.empty_cache()
            continue

    print(f"\nCompleted {os.path.basename(manifest_path)}: {successful} successful, {failed} failed")
    print(f"Coordinates saved for both UCMC and BCR-NET datasets")

# -------------- Main --------------
if __name__ == "__main__":
    print("="*60)
    print(f"UNI ({MODEL_NAME.upper()}) FEATURE EXTRACTION WITH COORDINATES")
    print("="*60)
    print(f"Device: {device}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Backbone: {MODEL_NAME}")
    print("="*60)

    # Create output directories for UNI features
    base_dirs = {
        "train": f"data/features_{MODEL_NAME}/train",
        "val": f"data/features_{MODEL_NAME}/val",
        "test": f"data/features_{MODEL_NAME}/test"
    }

    # Create directories
    for dir_path in base_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Run extraction for each split
    run_extraction("data/manifests/train_manifest.csv", base_dirs["train"])
    run_extraction("data/manifests/val_manifest.csv", base_dirs["val"])
    run_extraction("data/manifests/test_manifest.csv", base_dirs["test"])

    print("\n" + "="*60)
    print(f"UNI ({MODEL_NAME.upper()}) FEATURE EXTRACTION COMPLETED")
    print("Next steps: Use coordinates for attention visualization")