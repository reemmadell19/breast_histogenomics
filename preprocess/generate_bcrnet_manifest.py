import pandas as pd
import h5py
import os

# Load the label sheet
labels_df = pd.read_excel("data/raw/BCR_NET/Labels.xlsx")  

# Keep only numeric IDs corresponding to your files
labels_df = labels_df[labels_df['Image ID'].astype(str).str.match(r'^\d+$')]

# Convert Image ID to int for matching
labels_df['Image ID'] = labels_df['Image ID'].astype(int)

rows = []
bcrnet_dir = "data/raw/BCR_NET/"

for i in range(2, 80):
    file_name = f"{i}.h5"
    file_path = os.path.join(bcrnet_dir, file_name)

    if os.path.exists(file_path):
        with h5py.File(file_path, 'r') as f:
            num_tiles = f['bag'].shape[0]

        label_row = labels_df[labels_df['Image ID'] == i]

        if not label_row.empty:
            RS = label_row['score'].values[0]
            RSHigh = 'H' if RS >= 26 else 'L'
        else:
            RS = None
            RSHigh = None
            print(f"Warning: No label for {file_name}")

        rows.append({
            "file_name": file_name,
            "num_tiles": num_tiles,
            "RS": RS,
            "RSHigh": RSHigh
        })
    else:
        print(f"Warning: Missing file {file_name}")

# Create and save manifest
bcrnet_manifest = pd.DataFrame(rows)
bcrnet_manifest.to_csv("data/manifests/bcrnet_manifest.csv", index=False)

print("BCR-NET manifest created: data/manifests/bcrnet_manifest.csv")
