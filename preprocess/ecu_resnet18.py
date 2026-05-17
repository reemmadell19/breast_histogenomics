import os
import torch
import pandas as pd
from tqdm import tqdm
from torchvision import models, transforms
from PIL import Image
import tensorflow as tf
import numpy as np

# Silence TensorFlow logs
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

# Load pretrained ResNet18
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()  # Remove classification layer
resnet.eval()
resnet = resnet.to(device)

# -------------- Helper Functions --------------

def extract_features_in_batches(images, batch_size=32):
    """Extract ResNet18 features in batches for memory efficiency."""
    features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i + batch_size]).to(device)
            feat = resnet(batch)
            features.append(feat.cpu())
    return torch.cat(features)

def extract_features_from_ecu_tfrecord(tfrecord_path, out_path):
    """
    Extract features from a single ECU tfrecord file.

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
    da_numbers = []  # Track original tile IDs

    # Process each patch in the tfrecord
    for record in dataset:
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

        except Exception as e:
            print(f"  Warning: Skipping patch in {os.path.basename(tfrecord_path)}: {e}")
            continue

    # Extract features if we have valid patches
    if images:
        features = extract_features_in_batches(images)

        # Save in consistent format with UCMC/BCR-NET
        torch.save({
            'features': features,  # Shape: [N, 512]
            'coords': torch.tensor(coords),  # Shape: [N, 2]
            'slide_name': os.path.basename(tfrecord_path),
            'da_numbers': torch.tensor(da_numbers),  # ECU-specific: original tile IDs
            'num_patches': len(images)
        }, out_path)

        torch.cuda.empty_cache()
        return len(images)
    else:
        print(f"  Warning: No valid patches in {os.path.basename(tfrecord_path)}")
        return 0

def run_ecu_extraction(input_dir, output_dir, manifest_csv=None):
    """
    Run feature extraction for all ECU tfrecord files.

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

    print("\n" + "="*60)
    print("ECU FEATURE EXTRACTION")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total files to process: {total_files}")
    print(f"Device: {device}")
    print("="*60 + "\n")

    # Process each tfrecord file
    for tfrecord_file in tqdm(tfrecord_files, desc="Processing ECU files"):
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

        # Process the file
        try:
            num_patches = extract_features_from_ecu_tfrecord(input_path, output_path)
            if num_patches > 0:
                processed_files += 1
                total_patches += num_patches
            else:
                failed_files += 1

        except Exception as e:
            print(f"  Failed processing {tfrecord_file}: {e}")
            failed_files += 1
            torch.cuda.empty_cache()
            continue

    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total files: {total_files}")
    print(f"Already existed (skipped): {existing_files}")
    print(f"Successfully processed: {processed_files}")
    print(f"Failed: {failed_files}")

    if processed_files > 0:
        avg_patches = total_patches / processed_files
        print(f"Total patches processed: {total_patches}")
        print(f"Average patches per slide: {avg_patches:.1f}")

    if total_files - existing_files > 0:
        success_rate = (processed_files / (total_files - existing_files)) * 100
        print(f"Success rate: {success_rate:.1f}%")

    print("="*60)

def verify_extraction(output_dir, sample_file=None):
    """
    Verify extracted features by loading and checking a sample.

    Args:
        output_dir: Directory containing extracted features
        sample_file: Optional specific file to check
    """
    print("\n" + "="*60)
    print("VERIFICATION")
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
    print(f"Coords shape: {data['coords'].shape}")
    print(f"Number of patches: {data.get('num_patches', len(data['features']))}")

    if 'da_numbers' in data:
        unique_das = len(torch.unique(data['da_numbers']))
        print(f"Unique DA numbers (original tiles): {unique_das}")

    print(f"\nExtraction verified successfully!")

# -------------- Main --------------
if __name__ == "__main__":
    # ========== UPDATE THESE PATHS ==========
    ECU_INPUT_DIR = "data/raw/ECU"  # Directory with ECU tfrecord files
    ECU_OUTPUT_DIR = "data/features_resnet18/ecu"  # Where to save features
    ECU_MANIFEST = "data/manifests/ecu_manifest.csv"  # Optional: CSV with file info
    # =========================================

    print("="*60)
    print("ECU Dataset ResNet18 Feature Extraction")
    print("="*60)

    # Run extraction
    run_ecu_extraction(
        input_dir=ECU_INPUT_DIR,
        output_dir=ECU_OUTPUT_DIR,
        manifest_csv=ECU_MANIFEST if os.path.exists(ECU_MANIFEST) else None
    )

    # Verify results
    if os.path.exists(ECU_OUTPUT_DIR):
        verify_extraction(ECU_OUTPUT_DIR)

    print("\nECU feature extraction completed!")