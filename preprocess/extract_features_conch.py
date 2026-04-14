# 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

#!/usr/bin/env python3
"""
Enhanced CONCH Feature Extraction Script with Skip Logic and Coordinates
Extract features from histopathology images using CONCH vision-language foundation model.
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
import gc
import time

# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image transforms - CenterCrop(224) to preserve histopathology details
# CONCH uses standard ImageNet normalization
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

def extract_features_in_batches(images, batch_size=24):  # Smaller batch size for CONCH
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

def extract_features_from_bcrnet(h5_path, out_path):
    """Extract features from BCR-NET h5 files with coordinates"""
    print(f"  📁 Loading {os.path.basename(h5_path)}...")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'bag' not in f:
                print(f"  ❌ No 'bag' dataset in {h5_path}")
                return False
            
            patches = f['bag'][:]
            coords = f['coords'][:]  # ← ADD THIS LINE
            print(f"  📊 Found {len(patches)} patches, shape: {patches.shape}")
            print(f"  📊 Coordinates shape: {coords.shape}")
            
            if len(patches) > 50000:
                print(f"  ⚠️  Large file with {len(patches)} patches - this may take a while")
    
    except Exception as e:
        print(f"  ❌ Error reading {h5_path}: {e}")
        return False

    images = []
    skipped_patches = 0
    
    print(f"  🔄 Processing patches...")
    
    for i, patch in enumerate(tqdm(patches, desc=f"  Processing {os.path.basename(h5_path)}", leave=False)):
        try:
            if patch.shape[0] == 3:
                patch = patch.transpose(1, 2, 0)
            
            img = Image.fromarray(patch.astype('uint8'))
            tensor_img = basic_transform(img)
            images.append(tensor_img)
            
            # Process in smaller chunks for CONCH
            if len(images) >= 600:  # Smaller chunks for CONCH
                chunk_features = extract_features_in_batches(images, batch_size=24)
                if 'all_features' not in locals():
                    all_features = chunk_features
                else:
                    all_features = torch.cat([all_features, chunk_features])
                images = []
                gc.collect()
                
        except Exception as e:
            skipped_patches += 1
            if skipped_patches <= 5:
                print(f"    ⚠️  Skipping patch {i}: {e}")
            continue

    # Process remaining images
    if images:
        chunk_features = extract_features_in_batches(images, batch_size=24)
        if 'all_features' not in locals():
            all_features = chunk_features
        else:
            all_features = torch.cat([all_features, chunk_features])

    if 'all_features' in locals() and len(all_features) > 0:
        print(f"  💾 Saving {len(all_features)} features to {os.path.basename(out_path)}")
        
        # Save with CONCH metadata AND coordinates
        torch.save({
            'features': all_features,
            'coords': torch.tensor(coords.T),  # ← ADD THIS LINE (transpose to match UCMC format)
            'slide_name': os.path.basename(h5_path),
            'num_patches': len(all_features),
            'feature_dim': feature_dim,
            'backbone': 'conch'
        }, out_path)
        
        # Clear memory
        del all_features
        torch.cuda.empty_cache()
        gc.collect()
        
        if skipped_patches > 0:
            print(f"  ⚠️  Skipped {skipped_patches} patches due to errors")
        
        print(f"  ✅ Saved features and coordinates for {os.path.basename(h5_path)}")
        return True
    else:
        print(f"  ❌ No valid patches extracted from {h5_path}")
        return False

def extract_features_from_ucmc(tfrecord_path, out_path):
    """Extract features from UCMC tfrecord files using CONCH"""
    print(f"  📁 Loading {os.path.basename(tfrecord_path)}...")
    
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
            print(f"    ⚠️  Skipping patch in {tfrecord_path}: {e}")
            continue

    if images:
        print(f"  🔄 Computing CONCH features for {valid_patches} patches...")
        features = extract_features_in_batches(images, batch_size=24)
        
        print(f"  💾 Saving features to {os.path.basename(out_path)}")
        
        # Save with CONCH metadata and coordinates
        torch.save({
            'features': features,
            'coords': torch.tensor(coords),
            'slide_name': os.path.basename(tfrecord_path),
            'num_patches': len(features),
            'feature_dim': feature_dim,
            'backbone': 'conch'
        }, out_path)
        
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  ✅ Saved features and coordinates for {os.path.basename(tfrecord_path)}")
        return True
    else:
        print(f"  ❌ No valid patches in {tfrecord_path}")
        return False

def run_extraction(manifest_path, out_dir):
    """Run extraction with better progress tracking and error handling"""
    df = pd.read_csv(manifest_path)
    
    total_files = len(df)
    existing_files = 0
    processed_files = 0
    failed_files = 0
    start_time = time.time()

    print(f"\n🚀 Starting CONCH extraction for {manifest_path}")
    print(f"📊 Total files to process: {total_files}")
    print(f"📁 Output directory: {out_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    for idx, row in enumerate(df.iterrows()):
        _, row = row
        fname = row['file_name']
        dataset = row['dataset']

        # Progress update every 20 files (frequent for CONCH)
        if idx % 20 == 0 and idx > 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            print(f"  📈 Progress: {idx}/{total_files} ({idx/total_files*100:.1f}%) - {rate:.1f} files/sec")

        # Prepend correct base path
        if dataset == "UCMC":
            full_path = os.path.join("data/raw/UCMC", fname)
        elif dataset == "BCRNet":
            full_path = os.path.join("data/raw/BCR_NET", fname)
        else:
            print(f"❌ Unknown dataset type for {fname}")
            failed_files += 1
            continue

        slide_id = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, slide_id + ".pt")

        # ✅ CHECK IF FILE ALREADY EXISTS - SKIP IF IT DOES
        if os.path.exists(out_path):
            existing_files += 1
            continue

        print(f"\n🔄 [{idx+1}/{total_files}] Processing {fname}...")
        
        try:
            success = False
            if fname.endswith(".tfrecords"):
                success = extract_features_from_ucmc(full_path, out_path)
            elif fname.endswith(".h5"):
                success = extract_features_from_bcrnet(full_path, out_path)
            else:
                print(f"❌ Unsupported file format: {fname}")
                failed_files += 1
                continue
            
            if success:
                processed_files += 1
                print(f"✅ Completed {fname}")
            else:
                failed_files += 1
                print(f"❌ Failed {fname}")
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Process interrupted by user at {fname}")
            break
        except Exception as e:
            print(f"❌ Unexpected error for {fname}: {e}")
            failed_files += 1
            torch.cuda.empty_cache()
            gc.collect()
            continue

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n📋 CONCH EXTRACTION SUMMARY for {os.path.basename(manifest_path)}")
    print(f"⏱️  Total time: {elapsed:.1f} seconds")
    print(f"📊 Total files: {total_files}")
    print(f"⏭️  Already existed (skipped): {existing_files}")
    print(f"✅ Newly processed: {processed_files}")
    print(f"❌ Failed: {failed_files}")
    print(f"📈 Success rate: {processed_files}/{total_files - existing_files} new files")
    print(f"💾 Features saved with dimension: {feature_dim}")
    print(f"📍 Coordinates saved for both UCMC and BCR-NET datasets")

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("🚀 CONCH FEATURE EXTRACTION WITH COORDINATES")
    print("="*60)
    print(f"Device: {device}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Backbone: CONCH (Vision-Language)")
    print(f"Coordinates: ✅ UCMC and BCR-NET")
    print("="*60)
    
    # Create output directories for CONCH features
    base_dirs = {
        "train": "data/features_conch/train",
        "val": "data/features_conch/val", 
        "test": "data/features_conch/test"
    }
    
    # Create directories
    for dir_path in base_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")
    
    try:
        # Extract features for each split
        run_extraction("data/manifests/train_manifest.csv", base_dirs["train"])
        run_extraction("data/manifests/val_manifest.csv", base_dirs["val"])
        run_extraction("data/manifests/test_manifest.csv", base_dirs["test"])
        
        print("\n" + "="*60)
        print("🎉 CONCH FEATURE EXTRACTION COMPLETED!")
        print("="*60)
        print("\n📝 Next steps:")
        print(f"1. Generate CONCH manifests with feature dimension: {feature_dim}")
        print("2. Update MIL model input_dim to match CONCH features")
        print("3. Run training scripts with CONCH features")
        print("4. Compare performance with ResNet18/50 features")
        print("5. Use coordinates for interpretability analysis")
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")