# create_complete_bcrnet_manifest.py
import pandas as pd
import h5py
import os

def create_complete_bcrnet_manifest():
    # Load the label sheet
    labels_df = pd.read_excel("data/raw/BCR_NET/Labels.xlsx")
    print(f"Loaded {len(labels_df)} entries from Labels.xlsx")
    
    rows = []
    bcrnet_dir = "data/raw/BCR_NET/"
    
    # Get all .h5 files
    h5_files = [f for f in os.listdir(bcrnet_dir) if f.endswith('.h5')]
    print(f"Found {len(h5_files)} .h5 files")
    
    for file_name in sorted(h5_files):
        file_path = os.path.join(bcrnet_dir, file_name)
        
        print(f"Processing {file_name}")
        
        # Get number of tiles
        try:
            with h5py.File(file_path, 'r') as f:
                num_tiles = f['bag'].shape[0]
        except Exception as e:
            print(f"  Error reading {file_name}: {e}")
            continue
        
        # Extract the identifier for label matching (remove .h5 for matching only)
        identifier = file_name.replace('.h5', '')
        
        # Find matching label using the identifier without .h5
        label_row = labels_df[labels_df['Image ID'].astype(str) == identifier]
        
        if not label_row.empty:
            RS = label_row['score'].values[0]
            RSHigh = 'H' if RS >= 25 else 'L'  # Using 25 as threshold per your thesis
            print(f"  ✓ Found label: {identifier} -> RS={RS}, RSHigh={RSHigh}")
        else:
            RS = None
            RSHigh = None
            print(f"  ✗ No label found for {identifier}")
        
        # Keep the full filename with .h5 extension
        rows.append({
            "dataset": "BCRNet",
            "file_name": file_name,  # This keeps the .h5 extension
            "num_tiles": num_tiles,
            "RS": RS,
            "RSHigh": RSHigh,
            "split": ""
        })
    
    # Create manifest
    manifest_df = pd.DataFrame(rows)
    
    print(f"\n=== Summary ===")
    print(f"Total files: {len(manifest_df)}")
    print(f"Files with labels: {len(manifest_df[manifest_df['RS'].notna()])}")
    print(f"Files without labels: {len(manifest_df[manifest_df['RS'].isna()])}")
    
    if len(manifest_df[manifest_df['RS'].isna()]) > 0:
        print("\nFiles without labels:")
        for fname in manifest_df[manifest_df['RS'].isna()]['file_name'].tolist():
            print(f"  - {fname}")
    
    # Show RS distribution
    labeled_df = manifest_df[manifest_df['RS'].notna()]
    if not labeled_df.empty:
        high_risk = len(labeled_df[labeled_df['RSHigh'] == 'H'])
        low_risk = len(labeled_df[labeled_df['RSHigh'] == 'L'])
        print(f"\nRS Distribution:")
        print(f"  High risk (RS≥25): {high_risk} files")
        print(f"  Low risk (RS<25): {low_risk} files")
        print(f"  RS range: {labeled_df['RS'].min():.0f} - {labeled_df['RS'].max():.0f}")
    
    # Save manifest
    os.makedirs("data/manifests", exist_ok=True)
    manifest_df.to_csv("data/manifests/complete_bcrnet_manifest.csv", index=False)
    print(f"\nSaved to: data/manifests/complete_bcrnet_manifest.csv")
    
    return manifest_df

if __name__ == "__main__":
    manifest = create_complete_bcrnet_manifest()
    print("\nFirst 10 entries:")
    print(manifest[['file_name', 'RS', 'RSHigh']].head(10))