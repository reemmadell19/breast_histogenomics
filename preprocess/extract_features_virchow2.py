"""
Virchow-2 Feature Extraction Script with Coordinates
Extract features from histopathology images using Virchow-2 foundation model.
"""
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
from huggingface_hub import login, hf_hub_download

# Your HF token
HF_TOKEN = "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"

# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image transforms - CenterCrop(224) to preserve histopathology details
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

def load_virchow2_model():
    """Load Virchow-2 model for feature extraction."""
    print("Loading Virchow-2 model...")
    
    # Login to Hugging Face
    try:
        login(token=HF_TOKEN)
        print("✅ Successfully logged in to Hugging Face")
    except Exception as e:
        print(f"❌ HF login failed: {e}")
        raise
    
    try:
        # Load Virchow-2 with correct configuration from official docs
        from timm.layers import SwiGLUPacked
        
        print("📥 Loading Virchow-2 with official configuration...")
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2", 
            pretrained=True, 
            mlp_layer=SwiGLUPacked,  # Required for Virchow-2
            act_layer=torch.nn.SiLU   # Required activation function
        )
        
        model = model.eval()
        print(f"✅ Virchow-2 loaded successfully!")
        
        # According to docs: output is 1 x 261 x 1280
        # We'll use class token (index 0) which is 1280-dimensional
        feature_dim = 1280
        
    except Exception as e:
        print(f"❌ Failed to load Virchow-2: {e}")
        print("Please check:")
        print("1. You have access to paige-ai/Virchow2")
        print("2. Your timm version is >= 0.9.11")
        print("3. Your institutional email matches your HF account")
        raise RuntimeError("Could not load Virchow-2 model")
    
    # Move to device
    print(f"🚀 Moving model to {device}...")
    model = model.to(device)
    
    # Test the model with correct output extraction
    print(f"🧪 Testing model with dummy input...")
    try:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            output = model(dummy_input)  # Shape: [1, 261, 1280]
            
            # Extract class token (first token, index 0)
            class_token = output[:, 0]  # Shape: [1, 1280]
            
            print(f"✅ Model test successful!")
            print(f"📏 Full output shape: {output.shape}")
            print(f"📏 Class token shape: {class_token.shape}")
            print(f"🎯 CONFIRMED: Virchow-2 ready for extraction!")
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        raise
    
    print(f"\n" + "="*60)
    print(f"✅ VIRCHOW-2 MODEL LOADED SUCCESSFULLY")
    print(f"📏 Feature dimension: {feature_dim} (class token)")
    print(f"🏗️  Architecture: ViT-H/14 with SwiGLU")
    print(f"🎯 Ready for feature extraction!")
    print(f"="*60)
    
    return model, feature_dim

# Load Virchow-2 model
virchow2_model, feature_dim = load_virchow2_model()

# -------------- Batch feature extraction --------------
def extract_features_in_batches(images, batch_size=16):
    """Extract features in batches to manage memory usage - Virchow-2 specific."""
    features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i + batch_size]).to(device)
            
            # Virchow-2 outputs shape: [batch_size, 261, 1280]
            # We use the class token (index 0) as per official docs
            output = virchow2_model(batch)  # [batch_size, 261, 1280]
            class_tokens = output[:, 0]     # [batch_size, 1280] - extract class token
            
            features.append(class_tokens.cpu())
            
            # Clear GPU cache periodically
            if i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
                
    return torch.cat(features)

def extract_features_from_bcrnet(h5_path, out_path):
    """Extract features from BCR-Net H5 files using Virchow-2 with coordinates."""
    print(f"Processing BCR-Net file: {os.path.basename(h5_path)}")
    
    with h5py.File(h5_path, 'r') as f:
        patches = f['bag'][:]  # shape [N, 3, H, W] or [N, H, W, 3]
        coords = f['coords'][:]  # ← ADD THIS LINE
        print(f"  📊 Found {len(patches)} patches, shape: {patches.shape}")
        print(f"  📊 Coordinates shape: {coords.shape}")

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
        print(f"Extracted {valid_patches} valid patches, computing Virchow-2 features...")
        features = extract_features_in_batches(images)
        
        # Verify feature dimensions before saving
        expected_shape = (len(features), feature_dim)
        actual_shape = features.shape
        
        if actual_shape != expected_shape:
            print(f"⚠️  Warning: Feature shape mismatch!")
            print(f"   Expected: {expected_shape}")
            print(f"   Actual: {actual_shape}")
        else:
            print(f"✅ Feature dimensions verified: {actual_shape}")
        
        # Save with additional metadata AND coordinates
        torch.save({
            'features': features,
            'coords': torch.tensor(coords.T),  # ← ADD THIS LINE (transpose to match UCMC format)
            'slide_name': os.path.basename(h5_path),
            'num_patches': len(features),
            'feature_dim': feature_dim,
            'actual_feature_dim': features.shape[-1],
            'backbone': 'virchow2',
            'model_repo': 'paige-ai/Virchow2',
            'extraction_version': 'v1'
        }, out_path)
        
        print(f"Saved features: {features.shape} to {out_path}")
        print(f"✅ Saved features and coordinates for {os.path.basename(h5_path)}")
        torch.cuda.empty_cache()
    else:
        print(f"No valid patches extracted from {h5_path}")

def extract_features_from_ucmc(tfrecord_path, out_path):
    """Extract features from UCMC TFRecord files using Virchow-2."""
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
        print(f"Extracted {valid_patches} valid patches, computing Virchow-2 features...")
        features = extract_features_in_batches(images)
        
        # Verify feature dimensions before saving
        expected_shape = (len(features), feature_dim)
        actual_shape = features.shape
        
        if actual_shape != expected_shape:
            print(f"⚠️  Warning: Feature shape mismatch!")
            print(f"   Expected: {expected_shape}")
            print(f"   Actual: {actual_shape}")
        else:
            print(f"✅ Feature dimensions verified: {actual_shape}")
        
        # Save with additional metadata and coordinates
        torch.save({
            'features': features,
            'coords': torch.tensor(coords),
            'slide_name': os.path.basename(tfrecord_path),
            'num_patches': len(features),
            'feature_dim': feature_dim,
            'actual_feature_dim': features.shape[-1],
            'backbone': 'virchow2',
            'model_repo': 'paige-ai/Virchow2',
            'extraction_version': 'v1'
        }, out_path)
        
        print(f"Saved features: {features.shape} to {out_path}")
        print(f"✅ Saved features and coordinates for {os.path.basename(tfrecord_path)}")
        torch.cuda.empty_cache()
    else:
        print(f"No valid patches in {tfrecord_path}")

def run_extraction(manifest_path, out_dir):
    """Run feature extraction for a given manifest."""
    print(f"\n{'='*60}")
    print(f"EXTRACTING FEATURES FROM: {manifest_path}")
    print(f"OUTPUT DIRECTORY: {out_dir}")
    print(f"EXPECTED FEATURE DIMENSION: {feature_dim}")
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

        # Skip if already processed (check for Virchow-2 version)
        if os.path.exists(out_path):
            try:
                existing_data = torch.load(out_path, map_location='cpu')
                if existing_data.get('backbone') == 'virchow2' and existing_data.get('actual_feature_dim') == feature_dim:
                    print(f"Skipping {fname} - already processed with Virchow-2")
                    successful += 1
                    continue
                else:
                    print(f"Re-processing {fname} - updating to Virchow-2")
            except:
                print(f"Re-processing {fname} - previous file corrupted")

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
    print(f"📍 Coordinates saved for both UCMC and BCR-NET datasets")

# -------------- Main --------------
if __name__ == "__main__":
    print("="*60)
    print("VIRCHOW-2 FEATURE EXTRACTION WITH COORDINATES")
    print("="*60)
    print(f"Device: {device}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Backbone: Virchow-2")
    print(f"Developer: Paige AI")
    print(f"Expected output: 1280-dimensional features")
    print(f"Coordinates: ✅ UCMC and BCR-NET")
    print("="*60)
    
    # Create output directories for Virchow-2 features
    base_dirs = {
        "train": "data/features_virchow2/train",
        "val": "data/features_virchow2/val", 
        "test": "data/features_virchow2/test"
    }
    
    # Create directories
    for dir_path in base_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Run extraction for each split
    run_extraction("data/manifests/train_manifest.csv", base_dirs["train"])
    run_extraction("data/manifests/val_manifest.csv", base_dirs["val"])
    run_extraction("data/manifests/test_manifest.csv", base_dirs["test"])
    
    print("\n" + "="*60)
    print("VIRCHOW-2 FEATURE EXTRACTION COMPLETED")
    print("="*60)
    print("\nNext steps:")
    print(f"1. Generate Virchow-2 manifests")
    print(f"2. Update MIL model input_dim to {feature_dim}")
    print("3. Run training scripts with Virchow-2 features")
    print(f"4. Compare with H-Optimus-1 (AUC: 0.8855) and UNI-2H (AUC: 0.8521)")
    print("5. Use coordinates for interpretability analysis")
    print(f"\nFeatures saved to: data/features_virchow2/")
    
    # Final verification
    print("\n🔍 FINAL VERIFICATION:")
    sample_files = []
    for split_dir in base_dirs.values():
        if os.path.exists(split_dir):
            files = os.listdir(split_dir)
            if files:
                sample_files.append(os.path.join(split_dir, files[0]))
    
    if sample_files:
        for sample_file in sample_files[:2]:  # Check first 2 files
            try:
                data = torch.load(sample_file, map_location='cpu')
                actual_dim = data['features'].shape[-1]
                has_coords = 'coords' in data
                print(f"✅ {os.path.basename(sample_file)}: {actual_dim} dimensions, coords: {has_coords}")
                if actual_dim == 1280:
                    print(f"   🎯 SUCCESS: True Virchow-2 features!")
                else:
                    print(f"   ⚠️  Note: Expected 1280, got {actual_dim}")
            except Exception as e:
                print(f"❌ Could not verify {sample_file}: {e}")
    else:
        print("❌ No files found to verify")
    
    print(f"\n🏁 EXTRACTION SUMMARY:")
    print(f"📊 Expected performance target: > H-Optimus-1 AUC (0.8855)")
    print(f"⏱️  Estimated processing time: ~20 hours")
    print(f"🔬 Clinical validation: Virchow-2 has extensive clinical studies")
    print(f"📍 Coordinates: Ready for interpretability analysis")
    print(f"🎯 Next: Train and compare all foundation models!")