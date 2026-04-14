# add_missing_bcrnet_files.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def add_missing_bcrnet_files():
    # Load complete BCR-NET manifest
    complete_bcrnet_df = pd.read_csv("data/manifests/complete_bcrnet_manifest.csv")
    
    # Load existing manifests
    train_df = pd.read_csv("data/manifests/train_manifest.csv")
    val_df = pd.read_csv("data/manifests/val_manifest.csv") 
    test_df = pd.read_csv("data/manifests/test_manifest.csv")
    
    # Get all existing BCR-NET files across all splits
    existing_bcrnet_files = set()
    for df in [train_df, val_df, test_df]:
        bcrnet_files = df[df['dataset'] == 'BCRNet']['file_name'].tolist()
        existing_bcrnet_files.update(bcrnet_files)
    
    print(f"Found {len(existing_bcrnet_files)} existing BCR-NET files across all splits")
    print("Existing files:", sorted(existing_bcrnet_files))
    
    # Find NEW files (in complete manifest but not in existing splits)
    all_bcrnet_files = set(complete_bcrnet_df['file_name'].tolist())
    new_files = all_bcrnet_files - existing_bcrnet_files
    
    print(f"\nFound {len(new_files)} NEW BCR-NET files to add:")
    print("New files:", sorted(new_files))
    
    if len(new_files) == 0:
        print("No new files to add. All BCR-NET files are already in the splits.")
        return
    
    # Get the new files with their labels
    new_files_df = complete_bcrnet_df[complete_bcrnet_df['file_name'].isin(new_files)].copy()
    
    # Remove any files without labels
    labeled_new_df = new_files_df[new_files_df['RS'].notna()].copy()
    unlabeled_new_df = new_files_df[new_files_df['RS'].isna()]
    
    if len(unlabeled_new_df) > 0:
        print(f"Warning: {len(unlabeled_new_df)} new files without labels will be excluded:")
        print(unlabeled_new_df['file_name'].tolist())
    
    if len(labeled_new_df) == 0:
        print("No new labeled files to add.")
        return
    
    print(f"\nAdding {len(labeled_new_df)} new labeled BCR-NET files")
    
    # Calculate existing ratios from current splits
    total_existing = len(train_df) + len(val_df) + len(test_df)
    train_ratio = len(train_df) / total_existing
    val_ratio = len(val_df) / total_existing
    test_ratio = len(test_df) / total_existing
    
    print(f"Current split ratios:")
    print(f"  Train: {train_ratio:.3f} ({len(train_df)} files)")
    print(f"  Val: {val_ratio:.3f} ({len(val_df)} files)")
    print(f"  Test: {test_ratio:.3f} ({len(test_df)} files)")
    
    # Show RS distribution of new files
    high_risk = len(labeled_new_df[labeled_new_df['RSHigh'] == 'H'])
    low_risk = len(labeled_new_df[labeled_new_df['RSHigh'] == 'L'])
    print(f"\nNew files RS distribution: {high_risk} high-risk, {low_risk} low-risk")
    
    # Split new files using the same ratios
    np.random.seed(42)
    
    try:
        # First split: separate test set
        train_val_new, test_new = train_test_split(
            labeled_new_df, 
            test_size=test_ratio, 
            random_state=42,
            stratify=labeled_new_df['RSHigh']
        )
        
        # Second split: separate train and val
        relative_val_size = val_ratio / (train_ratio + val_ratio)
        train_new, val_new = train_test_split(
            train_val_new,
            test_size=relative_val_size,
            random_state=42,
            stratify=train_val_new['RSHigh']
        )
        
        print("✓ Used stratified splitting to maintain RS distribution")
        
    except ValueError as e:
        print(f"Stratified splitting failed ({e}), using random splitting")
        # Fall back to random splitting if stratification fails
        train_val_new, test_new = train_test_split(
            labeled_new_df, 
            test_size=test_ratio, 
            random_state=42
        )
        
        relative_val_size = val_ratio / (train_ratio + val_ratio)
        train_new, val_new = train_test_split(
            train_val_new,
            test_size=relative_val_size,
            random_state=42
        )
    
    # Update split labels for new files
    train_new = train_new.copy()
    val_new = val_new.copy()
    test_new = test_new.copy()
    
    train_new['split'] = 'train'
    val_new['split'] = 'val'
    test_new['split'] = 'test'
    
    print(f"\nNew files split:")
    print(f"  Train: {len(train_new)} files")
    print(f"  Val: {len(val_new)} files") 
    print(f"  Test: {len(test_new)} files")
    
    # Show RS distribution for each split of new files
    for split_name, split_df in [("Train", train_new), ("Val", val_new), ("Test", test_new)]:
        high_count = len(split_df[split_df['RSHigh'] == 'H'])
        low_count = len(split_df[split_df['RSHigh'] == 'L'])
        print(f"  {split_name}: {high_count} high-risk, {low_count} low-risk")
    
    # Create backups of original manifests
    print("\nCreating backups of original manifests...")
    train_df.to_csv("data/manifests/train_manifest_backup.csv", index=False)
    val_df.to_csv("data/manifests/val_manifest_backup.csv", index=False)
    test_df.to_csv("data/manifests/test_manifest_backup.csv", index=False)
    
    # Add new files to existing manifests (keeping all existing files)
    updated_train_df = pd.concat([train_df, train_new], ignore_index=True)
    updated_val_df = pd.concat([val_df, val_new], ignore_index=True)
    updated_test_df = pd.concat([test_df, test_new], ignore_index=True)
    
    # Save updated manifests
    updated_train_df.to_csv("data/manifests/train_manifest.csv", index=False)
    updated_val_df.to_csv("data/manifests/val_manifest.csv", index=False)
    updated_test_df.to_csv("data/manifests/test_manifest.csv", index=False)
    
    print(f"\nUpdated manifest sizes:")
    print(f"  Train: {len(train_df)} → {len(updated_train_df)} (+{len(train_new)})")
    print(f"  Val: {len(val_df)} → {len(updated_val_df)} (+{len(val_new)})")
    print(f"  Test: {len(test_df)} → {len(updated_test_df)} (+{len(test_new)})")
    
    # Show overall RS distribution after adding new files
    print(f"\nOverall RS distribution after adding new files:")
    for split_name, split_df in [("Train", updated_train_df), ("Val", updated_val_df), ("Test", updated_test_df)]:
        high_count = len(split_df[split_df['RSHigh'] == 'H'])
        low_count = len(split_df[split_df['RSHigh'] == 'L'])
        total = len(split_df)
        print(f"  {split_name}: {total} total ({high_count} high-risk, {low_count} low-risk)")
    
    print("\n✓ Original manifests backed up")
    print("✓ New files added to existing splits")
    print("✓ All existing files remain in their original splits")
    print("✓ Ready for feature extraction!")

if __name__ == "__main__":
    add_missing_bcrnet_files()