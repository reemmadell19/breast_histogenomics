#!/usr/bin/env python3
"""
UNI Feature Extraction for ECU Dataset
Extract features from ECU histopathology images using UNI foundation models.
"""
# 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

import os
import pandas as pd
from tqdm import tqdm
import timm
from torchvision import transforms
import torch
import torch.nn as nn
import warnings
from PIL import Image
import tensorflow as tf
import numpy as np
from huggingface_hub import login
import gc
import time

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
    
    print(f"✅ Loaded {model_name} model")
    print(f"📊 Feature dimension: {feature_dim}")
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

def extract_features_from_ecu(tfrecord_path, out_path):
    """Extract features from ECU tfrecord files using UNI."""
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
        
        # Save with UNI metadata and coordinates
        torch.save({
            'features': all_features,
            'coords': torch.tensor(coords),
            'slide_name': os.path.basename(tfrecord_path),
            'da_numbers': torch.tensor(da_numbers),  # ECU-specific: original tile IDs
            'num_patches': len(all_features),
            'feature_dim': feature_dim,
            'backbone': MODEL_NAME
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

    print(f"\n🚀 Starting {MODEL_NAME.upper()} extraction for ECU dataset")
    print(f"📊 Total files to process: {total_files}")
    print(f"📁 Output directory: {output_dir}")
    
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

        # Skip if already exists
        if os.path.exists(output_path):
            existing_files += 1
            continue

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
    print(f"\n📋 {MODEL_NAME.upper()} ECU EXTRACTION SUMMARY")
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
    """Verify extracted UNI features."""
    print(f"\n" + "="*60)
    print(f"VERIFICATION - {MODEL_NAME.upper()} FEATURES")
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
    data = torch.load(sample_path)
    
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
    
    print(f"\n✅ {MODEL_NAME.upper()} extraction verified successfully!")

# -------------- Main --------------
if __name__ == "__main__":
    print("="*60)
    print(f"UNI ({MODEL_NAME.upper()}) FEATURE EXTRACTION FOR ECU DATASET")
    print("="*60)
    print(f"Device: {device}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Backbone: {MODEL_NAME}")
    print("="*60)
    
    # ========== UPDATE THESE PATHS ==========
    ECU_INPUT_DIR = "data/raw/ECU"  # Directory with ECU tfrecord files
    ECU_OUTPUT_DIR = f"data/features_{MODEL_NAME}/ecu"  # Where to save UNI features
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
        print(f"🎉 UNI ({MODEL_NAME.upper()}) ECU FEATURE EXTRACTION COMPLETED!")
        print("="*60)
        print("\n📝 Next steps:")
        print(f"1. Verify features have dimension: {feature_dim}")
        print("2. Update MIL model input_dim to match UNI features")
        print("3. Run evaluation with ECU UNI features")
        print("4. Compare performance across different backbones")
        print("5. Use coordinates for interpretability analysis")
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()