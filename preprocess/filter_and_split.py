import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Paths
manifest_path = "data/manifests/unified_manifest.csv"
output_dir = "data/manifests"
data_dirs = ["data/raw/BCR_NET", "data/raw/UCMC"]
allowed_exts = [".h5", ".tfrecords"]

# Step 1: Load unified manifest
df = pd.read_csv(manifest_path)

# Step 2: Filter out UCMC rows with missing RS
df = df[~((df["dataset"] == "UCMC") & (df["RS"].isnull()))]

# Step 3: Keep only rows where file exists and has allowed extension
def file_exists(row):
    file_name = row["file_name"]
    ext = os.path.splitext(file_name)[1]
    if ext not in allowed_exts:
        return False
    for d in data_dirs:
        full_path = os.path.join(d, file_name)
        if os.path.exists(full_path):
            return True
    return False

df = df[df.apply(file_exists, axis=1)].copy()
print(f"Retained {len(df)} entries with valid, existing input files.")

# Step 4: Stratified split (train_val and test)
train_val_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["RSHigh"], random_state=42
)

# Step 5: Stratified split (train and val)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.2, stratify=train_val_df["RSHigh"], random_state=42
)

# Step 6: Save results
os.makedirs(output_dir, exist_ok=True)
train_df.to_csv(os.path.join(output_dir, "train_manifest.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val_manifest.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test_manifest.csv"), index=False)

print("Manifests saved:")
print(f"   Train: {len(train_df)}")
print(f"   Val:   {len(val_df)}")
print(f"   Test:  {len(test_df)}")
print("Filtering and splitting completed successfully!")