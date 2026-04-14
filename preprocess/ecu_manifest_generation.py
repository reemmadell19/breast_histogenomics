import pandas as pd
import os

# Define models and their feature dimensions
models = {
    "resnet18": 512,
    "resnet50": 2048,
    "conch": 512,  # or 768 depending on variant
    "uni2-h": 1536,
    "virchow2": 1280,
    "h-optimus": 1536
}

# Load existing ECU manifest with RS scores
ecu_manifest_path = "data/manifests/ecu_manifest.csv"
print(f"Loading ECU manifest from {ecu_manifest_path}")
base_df = pd.read_csv(ecu_manifest_path)

print(f"Loaded {len(base_df)} ECU samples")
print(f"Columns available: {list(base_df.columns)}")

# Check RS score distribution
if 'RS' in base_df.columns:
    rs_stats = base_df['RS'].describe()
    print(f"\nRS Score Statistics:")
    print(rs_stats)
    
    if 'RSHigh' in base_df.columns:
        # Count H and L values
        high_risk_count = (base_df['RSHigh'] == 'H').sum()
        low_risk_count = (base_df['RSHigh'] == 'L').sum()
        print(f"\nRisk Distribution:")
        print(f"  High risk (H): {high_risk_count}")
        print(f"  Low risk (L): {low_risk_count}")

print("\n" + "="*60)
print("Generating feature manifests for each model...")
print("="*60)

# Generate feature manifests for each model
for model_name in models.keys():
    print(f"\nProcessing {model_name}...")
    
    # Define feature directory based on model name
    if model_name == "uni2-h":
        feature_dir = "data/features_uni2-h/ecu"
    else:
        feature_dir = f"data/features_{model_name}/ecu"
    
    # Check if feature directory exists
    if not os.path.exists(feature_dir):
        print(f"  ⚠️  Feature directory {feature_dir} does not exist yet")
        print(f"     Run the {model_name} extraction script first")
        continue
    
    # Create a copy of base manifest
    df = base_df.copy()
    
    # Extract slide_id from file_name (remove .tfrecord or .tfrecords extension)
    df["slide_id"] = df["file_name"].str.replace(r"\.tfrecords?$", "", regex=True)
    
    # Build path to .pt files
    df["path"] = df["slide_id"].apply(lambda sid: os.path.join(feature_dir, f"{sid}.pt"))
    
    # Check which features actually exist
    df["exists"] = df["path"].apply(os.path.exists)
    existing_count = df["exists"].sum()
    
    print(f"  ✅ Found {existing_count}/{len(df)} extracted features")
    
    # Select columns for output manifest - keep RSHigh as H/L
    out_df = df[["slide_id", "path", "RS", "RSHigh"]]
    
    # Save feature manifest
    out_path = f"data/manifests/ecu_features_{model_name}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  💾 Saved {out_path}")
    
    # Report missing features if any
    if existing_count < len(df):
        missing_df = df[~df["exists"]]
        print(f"  ⚠️  Missing features for {len(missing_df)} slides:")
        # Show first few missing files
        for i, row in missing_df.head(5).iterrows():
            print(f"     - {row['file_name']}")
        if len(missing_df) > 5:
            print(f"     ... and {len(missing_df) - 5} more")

print("\n" + "="*60)
print("ECU MANIFEST GENERATION SUMMARY")
print("="*60)

# Summary table
summary_data = []
for model_name in models.keys():
    manifest_path = f"data/manifests/ecu_features_{model_name}.csv"
    if os.path.exists(manifest_path):
        df = pd.read_csv(manifest_path)
        feature_count = sum(1 for p in df["path"] if os.path.exists(p))
        percentage = (feature_count / len(df)) * 100
        summary_data.append({
            "Model": model_name,
            "Total": len(df),
            "Extracted": feature_count,
            "Missing": len(df) - feature_count,
            "Progress": f"{percentage:.1f}%"
        })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
else:
    print("No feature manifests generated yet")

print("="*60)
print("\nNext steps:")
print("1. Run feature extraction for any missing models")
print("2. Use these manifests for evaluation on ECU dataset")
print("3. Compare model performance across different backbones")
print("\nNote: RSHigh is preserved as 'H'/'L' format in manifests")