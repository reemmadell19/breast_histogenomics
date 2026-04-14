# generate_h_optimus_manifests_simple.py
import pandas as pd
import os

# Define paths and locations
splits = ["train", "val", "test"]
MODEL_NAME = "h-optimus"

base_manifest_names = {
    "train": "data/manifests/train_manifest.csv",
    "val":   "data/manifests/val_manifest.csv",
    "test":  "data/manifests/test_manifest.csv"
}

pt_dirs = {
    "train": f"data/features_{MODEL_NAME}/train",
    "val":   f"data/features_{MODEL_NAME}/val",
    "test":  f"data/features_{MODEL_NAME}/test"
}

print(f"Generating {MODEL_NAME.upper()} feature manifests...")

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
    out_path = f"data/manifests/{split}_features_{MODEL_NAME}_updated.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

print(f"\nAll {MODEL_NAME.upper()} manifests converted successfully.")