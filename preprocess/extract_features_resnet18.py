import os
import torch
import pandas as pd
from tqdm import tqdm
from torchvision import models, transforms
from PIL import Image
import h5py
import tensorflow as tf
import numpy as np

# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load pretrained ResNet18
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet.eval()
resnet = resnet.to(device)

# -------------- Batch feature extraction --------------
def extract_features_in_batches(images, batch_size=32):
    features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i + batch_size]).to(device)
            feat = resnet(batch)
            features.append(feat.cpu())
    return torch.cat(features)

def extract_features_from_bcrnet(h5_path, out_path):
    with h5py.File(h5_path, 'r') as f:
        patches = f['bag'][:]
        coords = f['coords'][:]  # ← ADD THIS

    images = []
    for i, patch in enumerate(tqdm(patches, desc=f"Reading {os.path.basename(h5_path)}")):
        try:
            if patch.shape[0] == 3:
                patch = patch.transpose(1, 2, 0)
            img = Image.fromarray(patch.astype('uint8'))
            tensor_img = basic_transform(img)
            images.append(tensor_img)
        except Exception as e:
            print(f"Skipping patch {i} in {h5_path}: {e}")
            continue

    if images:
        features = extract_features_in_batches(images)
        torch.save({
            'features': features,
            'coords': torch.tensor(coords.T),  # ← ADD THIS (transpose to match UCMC format)
            'slide_name': os.path.basename(h5_path)
        }, out_path)
        torch.cuda.empty_cache()
        
def extract_features_from_ucmc(tfrecord_path, out_path):
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
        except Exception as e:
            print(f"Skipping patch in {tfrecord_path}: {e}")
            continue

    if images:
        features = extract_features_in_batches(images)
        torch.save({
            'features': features,
            'coords': torch.tensor(coords),
            'slide_name': os.path.basename(tfrecord_path)
        }, out_path)
        torch.cuda.empty_cache()
    else:
        print(f"No valid patches in {tfrecord_path}")

def run_extraction(manifest_path, out_dir):
    df = pd.read_csv(manifest_path)
    
    # Count total files and existing files
    total_files = len(df)
    existing_files = 0
    processed_files = 0
    failed_files = 0

    print(f"Starting extraction for {manifest_path}")
    print(f"Total files to process: {total_files}")
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing: {os.path.basename(manifest_path)}"):
        fname = row['file_name']
        dataset = row['dataset']

        # Prepend correct base path
        if dataset == "UCMC":
            full_path = os.path.join("data/raw/UCMC", fname)
        elif dataset == "BCRNet":
            full_path = os.path.join("data/raw/BCR_NET", fname)
        else:
            print(f"Unknown dataset type for {fname}")
            continue

        slide_id = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, slide_id + ".pt")

        # ✅ CHECK IF FILE ALREADY EXISTS - SKIP IF IT DOES
        if os.path.exists(out_path):
            existing_files += 1
            # Uncomment the line below if you want to see each skipped file
            # print(f"⏭️  Skipping {fname} (already processed)")
            continue

        try:
            print(f"🔄 Processing {fname}...")
            if fname.endswith(".tfrecords"):
                extract_features_from_ucmc(full_path, out_path)
            elif fname.endswith(".h5"):
                extract_features_from_bcrnet(full_path, out_path)
            else:
                print(f"Unsupported file format: {fname}")
                failed_files += 1
                continue
            
            processed_files += 1
            print(f"✅ Completed {fname}")
            
        except Exception as e:
            print(f"❌ Failed for {fname}: {e}")
            failed_files += 1
            torch.cuda.empty_cache()
            continue

    # Print summary
    print(f"\n=== EXTRACTION SUMMARY for {os.path.basename(manifest_path)} ===")
    print(f"Total files: {total_files}")
    print(f"Already existed (skipped): {existing_files}")
    print(f"Newly processed: {processed_files}")
    print(f"Failed: {failed_files}")
    print(f"Success rate: {processed_files}/{total_files - existing_files} new files")

# -------------- Main --------------
if __name__ == "__main__":
    print("Starting ResNet18 feature extraction with skip logic...")
    
    # Extract features for each split
    run_extraction("data/manifests/train_manifest.csv", "data/features_resnet18/train")
    run_extraction("data/manifests/val_manifest.csv", "data/features_resnet18/val")
    run_extraction("data/manifests/test_manifest.csv", "data/features_resnet18/test")
    
    print("\n🎉 Feature extraction completed!")