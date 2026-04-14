#!/usr/bin/env python3
"""
Virchow-2 Feature Extraction Script for ECU Dataset
Extract features from ECU histopathology images using Virchow-2 foundation model.
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

# Your HF token
HF_TOKEN = "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"

# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ECU transform (no cropping needed - patches are already 224x224)
ecu_transform = transforms.Compose([
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

def extract_features_from_ecu(tfrecord_path, out_path):
    """Extract features from ECU tfrecord files using Virchow-2."""
    print(f"  📁 Loading {os.path.basename(tfrecord_path)}...")
    
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
    
    print(f"  📊 Found {total_patches} patches")
    
    if total_patches > 10000:
        print(f"  ⚠️  Large file with {total_patches} patches - processing in chunks")

    for idx, record in enumerate(tqdm(dataset, desc=f"  Reading {os.path.basename(tfrecord_path)}", leave=False)):
        try:
            # ECU uses JPEG encoding
            img_bytes = record['image'].numpy()
            img = tf.io.decode_jpeg(img_bytes).numpy()
            
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            pil_img = Image.fromarray(img.astype('uint8'))
            tensor_img = ecu_transform(pil_img)  # ECU patches are already 224x224
            images.append(tensor_img)
            
            # Use global coordinates to match other datasets format
            coords.append([record['global_x'].numpy(), record['global_y'].numpy()])
            da_numbers.append(record['da_number'].numpy())
            
            # Process in chunks to avoid memory issues
            if len(images) >= 800:  # Process in chunks
                chunk_features = extract_features_in_batches(images, batch_size=16)
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
            print(f"    ⚠️  Skipping patch {idx}: {e}")
            continue

    # Process remaining images
    if images:
        chunk_features = extract_features_in_batches(images, batch_size=16)
        if all_features is None:
            all_features = chunk_features
        else:
            all_features = torch.cat([all_features, chunk_features])

    if all_features is not None and len(all_features) > 0:
        print(f"  💾 Saving {len(all_features)} features to {os.path.basename(out_path)}")
        
        # Verify feature dimensions before saving
        expected_shape = (len(all_features), feature_dim)
        actual_shape = all_features.shape
        
        if actual_shape != expected_shape:
            print(f"  ⚠️  Warning: Feature shape mismatch!")
            print(f"     Expected: {expected_shape}")
            print(f"     Actual: {actual_shape}")
        else:
            print(f"  ✅ Feature dimensions verified: {actual_shape}")
        
        # Save with Virchow-2 metadata and coordinates
        torch.save({
            'features': all_features,
            'coords': torch.tensor(coords),
            'slide_name': os.path.basename(tfrecord_path),
            'da_numbers': torch.tensor(da_numbers),  # ECU-specific: original tile IDs
            'num_patches': len(all_features),
            'feature_dim': feature_dim,
            'actual_feature_dim': all_features.shape[-1],
            'backbone': 'virchow2',
            'model_repo': 'paige-ai/Virchow2',
            'extraction_version': 'v1'
        }, out_path)
        
        # Clear memory
        del all_features
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"  ✅ Saved features and coordinates for {os.path.basename(tfrecord_path)}")
        return True
    else:
        print(f"  ❌ No valid patches in {tfrecord_path}")
        return False

def run_ecu_extraction(input_dir, output_dir, manifest_csv=None):
    """Run ECU extraction with progress tracking."""
    
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

    print(f"\n🚀 Starting Virchow-2 extraction for ECU dataset")
    print(f"📊 Total files to process: {total_files}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📏 Expected feature dimension: {feature_dim}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for idx, tfrecord_file in enumerate(tfrecord_files):
        # Progress update every 25 files
        if idx % 25 == 0 and idx > 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (total_files - idx) / rate if rate > 0 else 0
            print(f"\n📈 Progress: {idx}/{total_files} ({idx/total_files*100:.1f}%) - "
                  f"{rate:.1f} files/sec - ETA: {remaining/60:.1f} min")

        # Get paths
        base_name = os.path.splitext(tfrecord_file)[0]
        input_path = os.path.join(input_dir, tfrecord_file)
        output_path = os.path.join(output_dir, f"{base_name}.pt")

        # Skip if already exists (check for Virchow-2 version)
        if os.path.exists(output_path):
            try:
                existing_data = torch.load(output_path, map_location='cpu')
                if existing_data.get('backbone') == 'virchow2' and existing_data.get('actual_feature_dim') == feature_dim:
                    existing_files += 1
                    continue
                else:
                    print(f"  Re-processing {tfrecord_file} - updating to Virchow-2")
            except:
                print(f"  Re-processing {tfrecord_file} - previous file corrupted")

        # Check if input exists
        if not os.path.exists(input_path):
            print(f"  ❌ File not found: {tfrecord_file}")
            failed_files += 1
            continue

        print(f"\n🔄 [{idx+1}/{total_files}] Processing {tfrecord_file}...")
        
        try:
            success = extract_features_from_ecu(input_path, output_path)
            
            if success:
                processed_files += 1
                print(f"✅ Completed {tfrecord_file}")
            else:
                failed_files += 1
                print(f"❌ Failed {tfrecord_file}")
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Process interrupted by user at {tfrecord_file}")
            break
        except Exception as e:
            print(f"❌ Unexpected error for {tfrecord_file}: {e}")
            failed_files += 1
            torch.cuda.empty_cache()
            gc.collect()
            continue

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n📋 VIRCHOW-2 ECU EXTRACTION SUMMARY")
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
    print(f"📊 Total files: {total_files}")
    print(f"⏭️  Already existed (skipped): {existing_files}")
    print(f"✅ Newly processed: {processed_files}")
    print(f"❌ Failed: {failed_files}")
    
    if total_files - existing_files > 0:
        success_rate = (processed_files / (total_files - existing_files)) * 100
        print(f"📈 Success rate: {success_rate:.1f}%")
    
    print(f"💾 Features saved with dimension: {feature_dim}")
    print(f"📍 Coordinates saved for all patches")

def verify_extraction(output_dir, sample_file=None):
    """Verify extracted Virchow-2 features."""
    print(f"\n" + "="*60)
    print(f"VERIFICATION - VIRCHOW-2 FEATURES")
    print(f"="*60)
    
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
    print(f"Model repo: {data.get('model_repo', 'not specified')}")
    
    if 'da_numbers' in data:
        unique_das = len(torch.unique(data['da_numbers']))
        print(f"Unique DA numbers (original tiles): {unique_das}")
    
    if data['features'].shape[1] == 1280:
        print(f"\n🎯 SUCCESS: True Virchow-2 features!")
    else:
        print(f"\n⚠️  Note: Expected 1280, got {data['features'].shape[1]}")

# -------------- Main --------------
if __name__ == "__main__":
    print("="*60)
    print("VIRCHOW-2 FEATURE EXTRACTION FOR ECU DATASET")
    print("="*60)
    print(f"Device: {device}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Backbone: Virchow-2")
    print(f"Developer: Paige AI")
    print(f"Expected output: 1280-dimensional features")
    print("="*60)
    
    # ========== UPDATE THESE PATHS ==========
    ECU_INPUT_DIR = "data/raw/ECU"  # Directory with ECU tfrecord files
    ECU_OUTPUT_DIR = "data/features_virchow2/ecu"  # Where to save Virchow-2 features
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
        print("🎉 VIRCHOW-2 ECU FEATURE EXTRACTION COMPLETED!")
        print("="*60)
        print("\n📝 Next steps:")
        print(f"1. Verify features have dimension: {feature_dim}")
        print("2. Update MIL model input_dim to 1280 for Virchow-2")
        print("3. Run evaluation with ECU Virchow-2 features")
        print("4. Compare performance across different backbones")
        print("5. Use coordinates for interpretability analysis")
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()