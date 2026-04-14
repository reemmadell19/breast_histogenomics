import pandas as pd
import json

# Load label data
labels_df = pd.read_csv("data/raw/UCMC/uch_brca_complete.csv")  # adjust path

# Load manifest.json
with open("data/raw/UCMC/manifest.json") as f:
    manifest_data = json.load(f)

rows = []

# Build manifest rows
for tfrecord_file, info in manifest_data.items():
    slide_id = tfrecord_file.replace(".tfrecords", "")
    
    # Find matching label row
    match = labels_df[labels_df["slide"] == slide_id]
    
    if not match.empty:
        RS = match["RS"].values[0]
        RSHigh = match["RSHigh"].values[0]
        rows.append({
            "file_name": tfrecord_file,
            "num_tiles": info["total"],
            "RS": RS,
            "RSHigh": RSHigh
        })
    else:
        print(f"Warning: No label found for {slide_id}")

# Create DataFrame
ucmc_manifest = pd.DataFrame(rows)

# Save to CSV
ucmc_manifest.to_csv("data/manifests/ucmc_manifest.csv", index=False)

print("UCMC manifest created: data/manifests/ucmc_manifest.csv")
import pandas as pd
import json

# Load label data
labels_df = pd.read_csv("data/raw/UCMC/uch_brca_complete.csv")  # adjust path

# Load manifest.json
with open("data/raw/UCMC/manifest.json") as f:
    manifest_data = json.load(f)

rows = []

# Build manifest rows
for tfrecord_file, info in manifest_data.items():
    slide_id = tfrecord_file.replace(".tfrecords", "")
    
    # Find matching label row
    match = labels_df[labels_df["slide"] == slide_id]
    
    if not match.empty:
        RS = match["RS"].values[0]
        RSHigh = match["RSHigh"].values[0]
        rows.append({
            "file_name": tfrecord_file,
            "num_tiles": info["total"],
            "RS": RS,
            "RSHigh": RSHigh
        })
    else:
        print(f"Warning: No label found for {slide_id}")

# Create DataFrame
ucmc_manifest = pd.DataFrame(rows)

# Save to CSV
ucmc_manifest.to_csv("data/manifests/ucmc_manifest.csv", index=False)

print("UCMC manifest created: data/manifests/ucmc_manifest.csv")
