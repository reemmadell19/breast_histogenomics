# clam_preprocess_ecu_global_coords.py
"""
Fixed CLAM preprocessing with proper global coordinate assignment.
Each tile gets unique coordinates based on its position in the tile grid.
"""

import os
import sys
import numpy as np
import cv2
import h5py
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
import argparse
import gc
import re
import pickle

"""
 python preprocess/clam_preprocess_ecu.py \
    --source data/raw/ECU/ECU_raw_9 \
    --save_dir data/processed/ECU_preprocessed_9 \
    --tissue_thresh 0.5
"""
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

class GlobalCoordCLAMProcessor:
    """
    Preprocessing with proper global coordinate tracking.
    """
    
    def __init__(self,
                 patch_size: int = 224,
                 step_size: int = 224,
                 tissue_thresh_patch: float = 0.5,
                 min_tissue_in_tile: float = 0.20,
                 tile_size: int = 2000):
        self.patch_size = patch_size
        self.step_size = step_size
        self.tissue_thresh_patch = tissue_thresh_patch
        self.min_tissue_in_tile = min_tissue_in_tile
        self.tile_size = tile_size
    
    def get_tile_files(self, patient_dir: Path) -> List[Path]:
        """
        Get tile files, excluding Ss1.jpg and other non-tile images.
        """
        all_files = list(patient_dir.glob('*.jpg')) + list(patient_dir.glob('*.JPG'))
        
        # Exclude non-tile images
        exclude_patterns = ['ss1', 'ss2', 'param', 'thumbnail', 'overview', 'macro']
        
        tile_files = []
        for f in all_files:
            fname_lower = f.stem.lower()
            if any(pattern in fname_lower for pattern in exclude_patterns):
                continue
            tile_files.append(f)
        
        # Sort tiles numerically/alphabetically
        def extract_number(filepath):
            match = re.search(r'(\d+)', filepath.stem)
            if match:
                return int(match.group(1))
            return 0
        
        try:
            tile_files = sorted(tile_files, key=extract_number)
        except:
            tile_files = sorted(tile_files)
        
        return tile_files
    
    def estimate_tile_grid(self, patient_dir: Path, n_tiles: int) -> Tuple[int, int]:
        """
        Estimate tile grid dimensions from param.p or number of tiles.
        """
        param_path = patient_dir / 'param.p'
        
        if param_path.exists():
            try:
                with open(param_path, 'rb') as f:
                    params = pickle.load(f)
                
                # Get WSI dimensions
                wsi_width, wsi_height = params['slide_dimension']
                tile_size = params['cws_read_size'][0]
                
                # Calculate grid dimensions
                grid_width = (wsi_width + tile_size - 1) // tile_size
                grid_height = (wsi_height + tile_size - 1) // tile_size
                
                print(f"  WSI: {wsi_width}×{wsi_height}, Tile grid: {grid_width}×{grid_height}")
                return grid_width, grid_height
            except:
                pass
        
        # Fallback: Estimate square-ish grid from number of tiles
        grid_size = int(np.ceil(np.sqrt(n_tiles)))
        print(f"  Estimated grid: {grid_size}×{grid_size} (from {n_tiles} tiles)")
        return grid_size, grid_size
    
    def calculate_tile_offset(self, tile_index: int, grid_width: int) -> Tuple[int, int]:
        """
        Calculate the global offset for a tile based on its index in the grid.
        
        Args:
            tile_index: Index of the tile (0, 1, 2, ...)
            grid_width: Width of the tile grid
            
        Returns:
            (x_offset, y_offset) in pixels
        """
        # Calculate grid position
        grid_x = tile_index % grid_width
        grid_y = tile_index // grid_width
        
        # Convert to pixel coordinates
        x_offset = grid_x * self.tile_size
        y_offset = grid_y * self.tile_size
        
        return x_offset, y_offset
    
    def detect_tissue_improved(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Improved tissue detection for H&E slides.
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        rgb_std = np.std(img_rgb, axis=2)
        
        # Multiple criteria
        not_white = gray < 240
        has_saturation = saturation > 10
        has_variation = rgb_std > 5
        
        r = img_rgb[:, :, 0].astype(float)
        g = img_rgb[:, :, 1].astype(float)
        b = img_rgb[:, :, 2].astype(float)
        max_diff = np.maximum(np.maximum(np.abs(r - g), np.abs(r - b)), np.abs(g - b))
        has_color = max_diff > 10
        
        tissue_mask = not_white & (has_saturation | has_variation | has_color)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tissue_mask = cv2.morphologyEx(tissue_mask.astype(np.uint8) * 255, 
                                       cv2.MORPH_OPEN, kernel_open)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel_close)
        
        return tissue_mask
    
    def process_tile(self, tile_path: Path, tile_index: int, 
                    x_offset: int, y_offset: int) -> Tuple[List, List]:
        """
        Process a tile and return patches with GLOBAL coordinates.
        
        Args:
            tile_path: Path to tile image
            tile_index: Index of this tile
            x_offset: Global X offset for this tile
            y_offset: Global Y offset for this tile
        """
        img = cv2.imread(str(tile_path))
        if img is None:
            return [], []
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get tissue mask
        tissue_mask = self.detect_tissue_improved(img_rgb)
        tissue_pct = np.sum(tissue_mask > 0) / tissue_mask.size
        
        if tissue_pct < self.min_tissue_in_tile:
            print(f"    Tile {tile_index}: {tile_path.name} - Skip ({tissue_pct*100:.1f}% tissue)")
            return [], []
        
        print(f"    Tile {tile_index}: {tile_path.name} at ({x_offset}, {y_offset}) - {tissue_pct*100:.1f}% tissue")
        
        patches = []
        coords = []
        
        h, w = img_rgb.shape[:2]
        n_patches = 0
        
        for y in range(0, h - self.patch_size + 1, self.step_size):
            for x in range(0, w - self.patch_size + 1, self.step_size):
                patch_mask = tissue_mask[y:y+self.patch_size, x:x+self.patch_size]
                tissue_ratio = np.sum(patch_mask > 0) / patch_mask.size
                
                if tissue_ratio >= self.tissue_thresh_patch:
                    patch = img_rgb[y:y+self.patch_size, x:x+self.patch_size]
                    patches.append(patch)
                    
                    # GLOBAL coordinates = tile offset + local patch position
                    global_x = x_offset + x
                    global_y = y_offset + y
                    
                    coords.append([global_x, global_y])
                    n_patches += 1
        
        print(f"      → {n_patches} patches extracted")
        
        return patches, coords
    
    def process_patient_global_coords(self, patient_dir: Path, output_dir: Path) -> str:
        """
        Process patient with proper global coordinate tracking.
        """
        patient_id = patient_dir.name
        print(f"\nProcessing {patient_id}")
        
        # Get tile files
        tile_paths = self.get_tile_files(patient_dir)
        
        if not tile_paths:
            print(f"  No valid tile files found")
            return None
        
        print(f"  Found {len(tile_paths)} tiles")
        
        # Estimate tile grid dimensions
        grid_width, grid_height = self.estimate_tile_grid(patient_dir, len(tile_paths))
        
        # Create output H5 file
        output_dir.mkdir(parents=True, exist_ok=True)
        h5_path = output_dir / f"{patient_id}.h5"
        
        with h5py.File(h5_path, 'w') as f:
            bag_dset = f.create_dataset(
                'bag',
                shape=(0, self.patch_size, self.patch_size, 3),
                maxshape=(None, self.patch_size, self.patch_size, 3),
                dtype=np.uint8,
                compression='gzip',
                compression_opts=4,
                chunks=(1, self.patch_size, self.patch_size, 3)
            )
            
            coords_dset = f.create_dataset(
                'coords',
                shape=(0, 2),
                maxshape=(None, 2),
                dtype=np.int32
            )
        
        total_patches = 0
        
        # Process each tile with proper global coordinates
        for tile_idx, tile_path in enumerate(tile_paths):
            # Calculate this tile's global offset
            x_offset, y_offset = self.calculate_tile_offset(tile_idx, grid_width)
            
            # Process tile with global offset
            patches, coords = self.process_tile(tile_path, tile_idx, x_offset, y_offset)
            
            if patches:
                n_new = len(patches)
                
                with h5py.File(h5_path, 'a') as f:
                    f['bag'].resize((total_patches + n_new, 
                                    self.patch_size, self.patch_size, 3))
                    f['coords'].resize((total_patches + n_new, 2))
                    
                    f['bag'][total_patches:total_patches + n_new] = np.array(patches)
                    f['coords'][total_patches:total_patches + n_new] = np.array(coords)
                
                total_patches += n_new
                
                # Print coordinate range for this tile
                if coords:
                    coords_array = np.array(coords)
                    print(f"      Coord range: X[{coords_array[:, 0].min()}-{coords_array[:, 0].max()}], "
                          f"Y[{coords_array[:, 1].min()}-{coords_array[:, 1].max()}]")
            
            del patches, coords
            gc.collect()
        
        if total_patches == 0:
            print(f"  No valid patches found")
            if h5_path.exists():
                os.remove(h5_path)
            return None
        
        # Print final coordinate statistics
        with h5py.File(h5_path, 'r') as f:
            all_coords = f['coords'][:]
            print(f"\n  Final statistics:")
            print(f"    Total patches: {total_patches}")
            print(f"    Global coord range: X[{all_coords[:, 0].min()}-{all_coords[:, 0].max()}], "
                  f"Y[{all_coords[:, 1].min()}-{all_coords[:, 1].max()}]")
        
        return str(h5_path)


def main():
    parser = argparse.ArgumentParser(description='CLAM preprocessing with global coordinates')
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='ECU_preprocessed_global')
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--step_size', type=int, default=224)
    parser.add_argument('--tissue_thresh', type=float, default=0.1)
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir
    
    output_dir = Path(args.save_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    
    processor = GlobalCoordCLAMProcessor(
        patch_size=args.patch_size,
        step_size=args.step_size,
        tissue_thresh_patch=args.tissue_thresh
    )
    
    patient_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patients")
    
    manifest = []
    for i, patient_dir in enumerate(patient_dirs, 1):
        print(f"\n[{i}/{len(patient_dirs)}]")
        h5_path = processor.process_patient_global_coords(patient_dir, output_dir)
        
        if h5_path:
            manifest.append({
                'patient_id': patient_dir.name,
                'h5_path': h5_path
            })
        
        gc.collect()
    
    if manifest:
        manifest_df = pd.DataFrame(manifest)
        manifest_df.to_csv(output_dir / 'process_list.csv', index=False)
        print(f"\nProcessed {len(manifest)} patients")


if __name__ == "__main__":
    main()