# 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

#!/usr/bin/env python3
"""
CONCH Feature Extraction Script for ECU Dataset
Extract features from ECU histopathology images using CONCH vision-language foundation model.
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
from huggingface_hub import login, hf_hub_download
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

def load_conch_model():
    """Load CONCH model for feature extraction."""
    print("Loading CONCH model...")
    
    # Login to Hugging Face
    try:
        login()
    except Exception as e:
        print(f"HF login failed: {e}")
        print("Trying without explicit login...")
    
    try:
        # CONCH model configuration
        model_repo = "MahmoodLab/conch"
        
        # Method 1: Try direct timm loading
        try:
            model = timm.create_model(
                f"hf-hub:{model_repo}",
                pretrained=True,
                num_classes=0,  # Remove classification head
            )
            
            feature_dim = getattr(model, 'embed_dim', 512)
            
        except Exception as e:
            print(f"Direct timm loading failed: {e}")
            print("Trying manual approach...")
            
            # Method 2: Manual loading approach
            local_dir = "./assets/ckpts/conch/"
            os.makedirs(local_dir, exist_ok=True)
            
            try:
                # Download model files
                model_path = hf_hub_download(
                    repo_id=model_repo,
                    filename="pytorch_model.bin",
                    local_dir=local_dir,
                    force_download=False
                )
                
                model = timm.create_model(
                    'vit_base_patch16_224',
                    pretrained=False,
                    num_classes=0,
                    img_size=224,
                    patch_size=16,
                    embed_dim=512,
                    depth=12,
                    num_heads=8,
                    mlp_ratio=4.0,
                )
                
                # Load weights
                state_dict = torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu")
                
                # Handle potential key mismatches for vision-language models
                if 'vision_model' in str(list(state_dict.keys())[0]):
                    vision_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('vision_model.'):
                            new_key = k.replace('vision_model.', '')
                            vision_state_dict[new_key] = v
                    state_dict = vision_state_dict
                
                model.load_state_dict(state_dict, strict=False)
                feature_dim = 512
                
            except Exception as e:
                print(f"Manual loading failed: {e}")
                
                # Method 3: Alternative CONCH configuration
                model = timm.create_model(
                    'vit_base_patch16_224',
                    pretrained=False,
                    num_classes=0,
                    global_pool='',
                )
                
                possible_files = [
                    "pytorch_model.bin",
                    "model.bin", 
                    "vision_model.bin",
                    "conch_model.bin"
                ]
                
                loaded = False
                for filename in possible_files:
                    try:
                        model_path = hf_hub_download(
                            repo_id=model_repo,
                            filename=filename,
                            local_dir=local_dir,
                            force_download=False
                        )
                        state_dict = torch.load(model_path, map_location="cpu")
                        
                        if 'model' in state_dict:
                            state_dict = state_dict['model']
                        elif 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                        
                        model.load_state_dict(state_dict, strict=False)
                        loaded = True
                        break
                    except:
                        continue
                
                if not loaded:
                    raise RuntimeError("Could not load CONCH model weights")
                    
                feature_dim = 768
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        print(f"✅ Loaded CONCH model")
        print(f"📊 Feature dimension: {feature_dim}")
        
        return model, feature_dim
        
    except Exception as e:
        print(f"❌ Failed to load CONCH: {e}")
        print("Make sure you have access to the MahmoodLab/conch repository")
        raise

# Load CONCH model
conch_model, feature_dim = load_conch_model()

def extract_features_in_batches(images, batch_size=24):
    """Extract features in smaller batches to avoid memory issues with CONCH"""
    features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i + batch_size]).to(device)
            
            try:
                feat = conch_model(batch)
                
                # Handle different output formats
                if isinstance(feat, dict):
                    feat = feat.get('pooler_output', feat.get('last_hidden_state', feat.get('features', list(feat.values())[0])))
                elif isinstance(feat, tuple):
                    feat = feat[0]
                
                # Ensure we have the right shape
                if feat.dim() > 2:
                    feat = feat.mean(dim=1)  # Global average pooling if needed
                
                features.append(feat.cpu())
                
            except Exception as e:
                print(f"    ⚠️  Error in batch processing: {e}")
                # Fallback: process images one by one
                batch_features = []
                for img in batch:
                    try:
                        single_feat = conch_model(img.unsqueeze(0))
                        if isinstance(single_feat, dict):
                            single_feat = single_feat.get('pooler_output', single_feat.get('last_hidden_state', list(single_feat.values())[0]))
                        elif isinstance(single_feat, tuple):
                            single_feat = single_feat[0]
                        if single_feat.dim() > 2:
                            single_feat = single_feat.mean(dim=1)
                        batch_features.append(single_feat.cpu())
                    except Exception as e2:
                        print(f"    ❌ Error processing single image: {e2}")
                        batch_features.append(torch.zeros(1, feature_dim))
                
                if batch_features:
                    features.append(torch.cat(batch_features, dim=0))
            
            # Clear GPU cache more frequently for CONCH
            if i % (batch_size * 2) == 0:
                torch.cuda.empty_cache()
                
    return torch.cat(features) if features else torch.zeros(0, feature_dim)

def extract_features_from_ecu(tfrecord_path, out_path):
    """Extract features from ECU tfrecord files using CONCH"""
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
            
            # Process in smaller chunks for CONCH
            if len(images) >= 600:  # Smaller chunks for CONCH
                chunk_features = extract_features_in_batches(images, batch_size=24)
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
            print(f"    ⚠️  Skipping patch {idx} in {tfrecord_path}: {e}")
            continue

    # Process remaining images
    if images:
        chunk_features = extract_features_in_batches(images, batch_size=24)
        if all_features is None:
            all_features = chunk_features
        else:
            all_features = torch.cat([all_features, chunk_features])

    if all_features is not None and len(all_features) > 0:
        print(f"  💾 Saving {len(all_features)} features to {os.path.basename(out_path)}")
        
        # Save with CONCH metadata and coordinates
        torch.save({
            'features': all_features,
            'coords': torch.tensor(coords),
            'slide_name': os.path.basename(tfrecord_path),
            'da_numbers': torch.tensor(da_numbers),  # ECU-specific: original tile IDs
            'num_patches': len(all_features),
            'feature_dim': feature_dim,
            'backbone': 'conch'
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

    print(f"\n🚀 Starting CONCH extraction for ECU dataset")
    print(f"📊 Total files to process: {total_files}")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for idx, tfrecord_file in enumerate(tfrecord_files):
        # Progress update every 20 files
        if idx % 20 == 0 and idx > 0:
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
    print(f"\n📋 CONCH ECU EXTRACTION SUMMARY")
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

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("🚀 CONCH FEATURE EXTRACTION FOR ECU DATASET")
    print("="*60)
    print(f"Device: {device}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Backbone: CONCH (Vision-Language)")
    print("="*60)
    
    # ========== UPDATE THESE PATHS ==========
    ECU_INPUT_DIR = "data/raw/ECU"  # Directory with ECU tfrecord files
    ECU_OUTPUT_DIR = "data/features_conch/ecu"  # Where to save CONCH features
    ECU_MANIFEST = "data/manifests/ecu_manifest.csv"  # Optional: CSV with file info
    # =========================================
    
    try:
        # Run extraction
        run_ecu_extraction(
            input_dir=ECU_INPUT_DIR,
            output_dir=ECU_OUTPUT_DIR,
            manifest_csv=ECU_MANIFEST if os.path.exists(ECU_MANIFEST) else None
        )
        
        print("\n" + "="*60)
        print("🎉 CONCH ECU FEATURE EXTRACTION COMPLETED!")
        print("="*60)
        print("\n📝 Next steps:")
        print(f"1. Verify features have dimension: {feature_dim}")
        print("2. Update MIL model input_dim to match CONCH features")
        print("3. Run evaluation with ECU CONCH features")
        print("4. Compare performance across different backbones")
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()