import os
import h5py
import numpy as np

h5_path = 'data/processed/ECU_preprocessed_9/385__20220311_101821.isyntax.h5'

with h5py.File(h5_path, 'r') as f:
    print(f"Keys: {list(f.keys())}")
    print(f"Bag shape: {f['bag'].shape}")
    print(f"Bag dtype: {f['bag'].dtype}")
    print(f"Bag compression: {f['bag'].compression}")
    print(f"Coords shape: {f['coords'].shape}")
    
    # Calculate actual size
    n_patches, h, w, c = f['bag'].shape
    uncompressed_size = n_patches * h * w * c / (1024**3)  # GB
    print(f"\nUncompressed size: {uncompressed_size:.2f} GB")
    print(f"File size on disk: {os.path.getsize(h5_path)/(1024**3):.2f} GB")