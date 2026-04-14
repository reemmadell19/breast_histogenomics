# generate_resnet50_manifests_simple.py
import pandas as pd
import os

# Define paths and locations
splits = ["train", "val", "test"]
base_manifest_names = {
    "train": "data/manifests/train_manifest.csv",
    "val":   "data/manifests/val_manifest.csv",
    "test":  "data/manifests/test_manifest.csv"
}

pt_dirs = {
    "train": "data/features_resnet50/train",
    "val":   "data/features_resnet50/val",
    "test":  "data/features_resnet50/test"
}

# Process each split
for split in splits:
    print(f"Processing {split} manifest...")
    
    # Load original manifest
    df = pd.read_csv(base_manifest_names[split])
    
    # Strip off either .tfrecords or .h5 to get slide_id
    df["slide_id"] = df["file_name"].str.replace(r"\.tfrecords$|\.h5$", "", regex=True)
    
    # Build new path to .pt file
    df["path"] = df["slide_id"].apply(lambda sid: os.path.join(pt_dirs[split], f"{sid}.pt"))
    
    # Select and reorder columns
    out_df = df[["slide_id", "path", "RS", "RSHigh"]]
    
    # Save new feature manifest alongside originals
    out_path = f"data/manifests/{split}_features_resnet50_updated.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

print("\nAll ResNet50 manifests converted successfully.")