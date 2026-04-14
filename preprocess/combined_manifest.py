import pandas as pd
import os

def create_combined_manifest(train_csv_path, val_csv_path, output_path):
    """
    Combine train and validation CSV files into single manifest for CV
    """
    # Load train and val files
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    
    print(f"Train dataset: {len(train_df)} samples")
    print(f"Val dataset: {len(val_df)} samples")
    
    # Add split indicator (optional, for tracking)
    train_df['original_split'] = 'train'
    val_df['original_split'] = 'val'
    
    # Combine datasets
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    
    print(f"Combined dataset: {len(combined_df)} samples")
    
    # Save combined manifest
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    print(f"Combined manifest saved to: {output_path}")
    return combined_df

def create_all_combined_manifests():
    """
    Create combined manifests for all feature extractors
    """
    feature_extractors = ["resnet18", "resnet50", "conch", "uni2-h", "virchow2", "h-optimus"]
    
    print("="*60)
    print("CREATING COMBINED MANIFESTS FOR CROSS-VALIDATION")
    print("="*60)
    
    for feature in feature_extractors:
        print(f"\nProcessing {feature.upper()}...")
        
        train_csv = f"data/manifests/train_features_{feature}.csv"
        val_csv = f"data/manifests/val_features_{feature}.csv"
        output_csv = f"data/manifests/combined_features_{feature}.csv"
        
        # Check if files exist
        if not os.path.exists(train_csv):
            print(f"  Warning: {train_csv} not found - skipping")
            continue
        if not os.path.exists(val_csv):
            print(f"  Warning: {val_csv} not found - skipping")
            continue
        
        # Create combined manifest
        try:
            combined_df = create_combined_manifest(train_csv, val_csv, output_csv)
            
            # Verify RS column exists and show distribution
            if 'RS' in combined_df.columns:
                high_risk_count = (combined_df['RS'] >= 25).sum()
                low_risk_count = len(combined_df) - high_risk_count
                print(f"  Class distribution: {low_risk_count} low-risk, {high_risk_count} high-risk")
            else:
                print(f"  Warning: No 'RS' column found in {feature}")
                
        except Exception as e:
            print(f"  Error processing {feature}: {e}")
    
    print(f"\n{'='*60}")
    print("COMBINED MANIFEST CREATION COMPLETED")
    print(f"{'='*60}")
    print("Created files:")
    for feature in feature_extractors:
        output_csv = f"data/manifests/combined_features_{feature}.csv"
        if os.path.exists(output_csv):
            print(f"  ✓ {output_csv}")
        else:
            print(f"  ✗ {output_csv} (failed)")

if __name__ == "__main__":
    create_all_combined_manifests()