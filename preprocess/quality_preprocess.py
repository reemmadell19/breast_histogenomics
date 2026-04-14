# clam_preprocess_ecu_global_coords_with_improved_quality.py
"""
Improved CLAM preprocessing with lenient quality filtering using RGB-based shadow detection.
Removes black/shadowed patches, green artifacts, and poor quality patches while preserving
rich morphological content.
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
 python preprocess/quality_preprocess.py \
    --source data/raw/ECU/ECU_raw_6 \
    --save_dir data/processed/ECU_preprocessed_6 \
    --tissue_thresh 0.6 \
    --blur_thresh 15.0 \
    --color_var_thresh 3.0
"""
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

class ImprovedQualityCLAMProcessor:
    """
    Preprocessing with improved quality filtering using RGB-based shadow detection.
    """
    
    def __init__(self,
                 patch_size: int = 224,
                 step_size: int = 224,
                 tissue_thresh_patch: float = 0.6,
                 min_tissue_in_tile: float = 0.6,
                 tile_size: int = 2000,
                 blur_thresh: float = 15.0,
                 color_var_thresh: float = 3.0,
                 max_patches_per_tile: int = None):
        self.patch_size = patch_size
        self.step_size = step_size
        self.tissue_thresh_patch = tissue_thresh_patch
        self.min_tissue_in_tile = min_tissue_in_tile
        self.tile_size = tile_size
        
        # Improved quality filtering parameters
        self.blur_thresh = blur_thresh
        self.color_var_thresh = color_var_thresh
        self.max_patches_per_tile = max_patches_per_tile
    
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
    
    def detect_black_shadows_rgb(self, patch: np.ndarray) -> bool:
        """
        RGB-based shadow and black patch detection.
        
        Args:
            patch: RGB patch of shape (H, W, 3)
            
        Returns:
            bool: True if patch is acceptable (not shadow/black), False if should be rejected
        """
        # Calculate mean intensity per channel
        r_mean = np.mean(patch[:, :, 0])
        g_mean = np.mean(patch[:, :, 1]) 
        b_mean = np.mean(patch[:, :, 2])
        
        overall_mean = (r_mean + g_mean + b_mean) / 3
        
        # 1. Pure black detection - very low overall intensity
        if overall_mean < 15:
            return False
        
        # 2. Shadow detection: low intensity AND low variation between channels
        channel_std = np.std([r_mean, g_mean, b_mean])
        
        # Shadows have very similar RGB values (no color information) and are dark
        if (overall_mean < 25) and (channel_std < 5):
            return False
        
        # 3. Check for extremely dark patches with no detail
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        if np.mean(gray) < 20 and np.max(gray) < 50:
            return False
        
        # 4. Check for patches that are too uniform (lack of texture/detail)
        if np.std(gray) < 3:  # Very low standard deviation = uniform
            return False
            
        return True
    
    def detect_green_artifacts(self, patch: np.ndarray) -> bool:
        """
        Strict green artifact detection - removes ANY green hints.
        
        Args:
            patch: RGB patch of shape (H, W, 3)
            
        Returns:
            bool: True if patch is acceptable, False if contains ANY green artifacts
        """
        r_mean = np.mean(patch[:, :, 0])
        g_mean = np.mean(patch[:, :, 1])
        b_mean = np.mean(patch[:, :, 2])
        
        # STRICT: Reject if green channel is higher than BOTH red and blue by ANY amount
        if g_mean > r_mean and g_mean > b_mean:
            green_dominance = g_mean - max(r_mean, b_mean)
            # Very strict threshold - reject even slight green dominance
            if green_dominance > 5:  # Much lower threshold
                return False
        
        # STRICT HSV-based green detection
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Expanded green hue range to catch more green variants (30-90 degrees)
        green_hue_mask = ((hue >= 30) & (hue <= 90) & (saturation > 30) & (value > 50))
        green_pixel_ratio = np.sum(green_hue_mask) / green_hue_mask.size
        
        # STRICT: Reject if more than 5% of pixels show ANY green characteristics
        if green_pixel_ratio > 0.05:
            return False
        
        # Additional check: Look for green-tinted areas even with low saturation
        # Check for pixels where green is noticeably higher than red/blue
        r = patch[:, :, 0].astype(float)
        g = patch[:, :, 1].astype(float)
        b = patch[:, :, 2].astype(float)
        
        # Pixel-wise green dominance check
        green_dominant_pixels = (g > r + 8) & (g > b + 8)  # Green higher by at least 8
        green_dominant_ratio = np.sum(green_dominant_pixels) / green_dominant_pixels.size
        
        # STRICT: Reject if more than 3% of pixels are green-dominant
        if green_dominant_ratio > 0.03:
            return False
        
        # Final check: Overall green bias in the patch
        total_intensity = r_mean + g_mean + b_mean
        if total_intensity > 0:
            green_ratio = g_mean / total_intensity
            # Normal tissue should have green ratio around 0.33, reject if significantly higher
            if green_ratio > 0.38:  # Allow only 5% deviation from perfect balance
                return False
            
        return True
    
    def detect_background_artifacts(self, patch: np.ndarray) -> bool:
        """
        Detect background and other artifacts.
        
        Args:
            patch: RGB patch of shape (H, W, 3)
            
        Returns:
            bool: True if patch is acceptable, False if contains artifacts
        """
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # 1. Too bright (likely background)
        if np.mean(gray) > 240:
            return False
        
        # 2. Too uniform (likely empty or artifact)
        if np.std(gray) < 5:
            return False
        
        # 3. Check for extreme values that indicate scanning artifacts
        if np.min(gray) == 0 and np.max(gray) == 255:  # Full dynamic range might be artifact
            # Check if it's just very high contrast tissue or actual artifact
            # If more than 10% pixels are pure white or pure black, likely artifact
            pure_white = np.sum(gray >= 250)
            pure_black = np.sum(gray <= 5)
            total_pixels = gray.size
            
            if (pure_white + pure_black) / total_pixels > 0.1:
                return False
        
        return True
    
    def calculate_patch_quality_improved(self, patch: np.ndarray) -> Tuple[float, bool, dict]:
        """
        Improved quality assessment with detailed filtering.
        
        Args:
            patch: RGB patch of shape (H, W, 3)
            
        Returns:
            quality_score: Combined quality score (higher is better)
            passes_filters: Boolean indicating if patch passes all quality filters
            filter_results: Dictionary with individual filter results for debugging
        """
        # Initialize filter results
        filter_results = {
            'passes_blur': False,
            'passes_color_var': False,
            'passes_shadow_check': False,
            'passes_green_check': False,
            'passes_background_check': False
        }
        
        # 1. Blur detection (variance of Laplacian)
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray_patch, cv2.CV_64F).var()
        filter_results['passes_blur'] = laplacian_var >= self.blur_thresh
        
        # 2. Color variance check (more lenient)
        color_std = np.std(patch.reshape(-1, 3), axis=0).mean()
        filter_results['passes_color_var'] = color_std >= self.color_var_thresh
        
        # 3. RGB-based shadow/black detection
        filter_results['passes_shadow_check'] = self.detect_black_shadows_rgb(patch)
        
        # 4. Green artifact detection
        filter_results['passes_green_check'] = self.detect_green_artifacts(patch)
        
        # 5. Background and artifact detection
        filter_results['passes_background_check'] = self.detect_background_artifacts(patch)
        
        # Overall pass/fail
        passes_filters = all(filter_results.values())
        
        # Calculate combined quality score (for ranking patches)
        quality_score = (laplacian_var * 0.01) + (color_std * 0.1)
        
        # Bonus points for good contrast and color diversity
        contrast_bonus = np.std(gray_patch) * 0.01
        color_bonus = np.mean(np.std(patch.reshape(-1, 3), axis=0)) * 0.05
        
        quality_score += contrast_bonus + color_bonus
        
        return quality_score, passes_filters, filter_results
    
    def process_tile(self, tile_path: Path, tile_index: int, 
                    x_offset: int, y_offset: int) -> Tuple[List, List, dict]:
        """
        Process a tile with improved quality filtering.
        """
        img = cv2.imread(str(tile_path))
        if img is None:
            return [], [], {}
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get tissue mask
        tissue_mask = self.detect_tissue_improved(img_rgb)
        tissue_pct = np.sum(tissue_mask > 0) / tissue_mask.size
        
        if tissue_pct < self.min_tissue_in_tile:
            print(f"    Tile {tile_index}: {tile_path.name} - Skip ({tissue_pct*100:.1f}% tissue)")
            return [], [], {}
        
        print(f"    Tile {tile_index}: {tile_path.name} at ({x_offset}, {y_offset}) - {tissue_pct*100:.1f}% tissue")
        
        # Collect candidate patches with quality scores
        candidate_patches = []
        candidate_coords = []
        candidate_scores = []
        
        # Detailed quality statistics
        quality_stats = {
            'total_candidates': 0,
            'passed_tissue': 0,
            'passed_blur': 0,
            'passed_color_var': 0,
            'passed_shadow_check': 0,
            'passed_green_check': 0,
            'passed_background_check': 0,
            'passed_all_filters': 0
        }
        
        h, w = img_rgb.shape[:2]
        
        for y in range(0, h - self.patch_size + 1, self.step_size):
            for x in range(0, w - self.patch_size + 1, self.step_size):
                quality_stats['total_candidates'] += 1
                
                # Check tissue threshold
                patch_mask = tissue_mask[y:y+self.patch_size, x:x+self.patch_size]
                tissue_ratio = np.sum(patch_mask > 0) / patch_mask.size
                
                if tissue_ratio >= self.tissue_thresh_patch:
                    quality_stats['passed_tissue'] += 1
                    
                    patch = img_rgb[y:y+self.patch_size, x:x+self.patch_size]
                    
                    # Calculate quality metrics with detailed results
                    quality_score, passes_filters, filter_results = self.calculate_patch_quality_improved(patch)
                    
                    # Update quality statistics
                    for filter_name, passed in filter_results.items():
                        if passed:
                            quality_stats[filter_name] += 1
                    
                    if passes_filters:
                        quality_stats['passed_all_filters'] += 1
                        
                        # Global coordinates = tile offset + local patch position
                        global_x = x_offset + x
                        global_y = y_offset + y
                        
                        candidate_patches.append(patch)
                        candidate_coords.append([global_x, global_y])
                        candidate_scores.append(quality_score)
        
        # Select final patches (either all or top N if max_patches_per_tile is set)
        if self.max_patches_per_tile and len(candidate_patches) > self.max_patches_per_tile:
            # Sort by quality score and take top patches
            indices = np.argsort(candidate_scores)[-self.max_patches_per_tile:]
            patches = [candidate_patches[i] for i in indices]
            coords = [candidate_coords[i] for i in indices]
            print(f"      → Selected top {len(patches)} patches from {len(candidate_patches)} candidates")
        else:
            patches = candidate_patches
            coords = candidate_coords
            print(f"      → {len(patches)} patches passed all quality filters")
        
        # Print detailed quality statistics
        total = quality_stats['total_candidates']
        print(f"      Quality pipeline: {total} total → "
              f"{quality_stats['passed_tissue']} tissue → "
              f"{quality_stats['passed_blur']} blur → "
              f"{quality_stats['passed_color_var']} color → "
              f"{quality_stats['passed_shadow_check']} shadow → "
              f"{quality_stats['passed_green_check']} green → "
              f"{quality_stats['passed_background_check']} background → "
              f"{quality_stats['passed_all_filters']} final")
        
        # Print coordinate range for this tile
        if coords:
            coords_array = np.array(coords)
            print(f"      Coord range: X[{coords_array[:, 0].min()}-{coords_array[:, 0].max()}], "
                  f"Y[{coords_array[:, 1].min()}-{coords_array[:, 1].max()}]")
        
        return patches, coords, quality_stats
    
    def process_patient_global_coords(self, patient_dir: Path, output_dir: Path) -> str:
        """
        Process patient with improved quality filtering and global coordinate tracking.
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
                compression_opts=9,
                chunks=(100, self.patch_size, self.patch_size, 3),
                shuffle=True,
                fletcher32=True
            )
            
            coords_dset = f.create_dataset(
                'coords',
                shape=(0, 2),
                maxshape=(None, 2),
                dtype=np.int32
            )
        
        total_patches = 0
        overall_quality_stats = {
            'total_candidates': 0,
            'passed_tissue': 0,
            'passed_blur': 0,
            'passed_color_var': 0,
            'passed_shadow_check': 0,
            'passed_green_check': 0,
            'passed_background_check': 0,
            'passed_all_filters': 0
        }
        
        # Process each tile with proper global coordinates
        for tile_idx, tile_path in enumerate(tile_paths):
            # Calculate this tile's global offset
            x_offset, y_offset = self.calculate_tile_offset(tile_idx, grid_width)
            
            # Process tile with global offset and improved quality filtering
            patches, coords, tile_quality_stats = self.process_tile(tile_path, tile_idx, x_offset, y_offset)
            
            # Update overall quality statistics
            for key in overall_quality_stats:
                overall_quality_stats[key] += tile_quality_stats.get(key, 0)
            
            if patches:
                n_new = len(patches)
                
                with h5py.File(h5_path, 'a') as f:
                    f['bag'].resize((total_patches + n_new, 
                                    self.patch_size, self.patch_size, 3))
                    f['coords'].resize((total_patches + n_new, 2))
                    
                    f['bag'][total_patches:total_patches + n_new] = np.array(patches)
                    f['coords'][total_patches:total_patches + n_new] = np.array(coords)
                
                total_patches += n_new
            
            del patches, coords
            gc.collect()
        
        if total_patches == 0:
            print(f"  No valid patches found")
            if h5_path.exists():
                os.remove(h5_path)
            return None
        
        # Print final comprehensive statistics
        with h5py.File(h5_path, 'r') as f:
            all_coords = f['coords'][:]
            print(f"\n  Final statistics:")
            print(f"    Total patches: {total_patches}")
            print(f"    Global coord range: X[{all_coords[:, 0].min()}-{all_coords[:, 0].max()}], "
                  f"Y[{all_coords[:, 1].min()}-{all_coords[:, 1].max()}]")
            
            # Print comprehensive quality filtering results
            print(f"\n  Improved quality filtering summary:")
            total = overall_quality_stats['total_candidates']
            if total > 0:
                print(f"    Total candidates: {total}")
                print(f"    Passed tissue threshold: {overall_quality_stats['passed_tissue']} ({overall_quality_stats['passed_tissue']/total*100:.1f}%)")
                print(f"    Passed blur filter: {overall_quality_stats['passed_blur']} ({overall_quality_stats['passed_blur']/total*100:.1f}%)")
                print(f"    Passed color variance filter: {overall_quality_stats['passed_color_var']} ({overall_quality_stats['passed_color_var']/total*100:.1f}%)")
                print(f"    Passed shadow/black filter: {overall_quality_stats['passed_shadow_check']} ({overall_quality_stats['passed_shadow_check']/total*100:.1f}%)")
                print(f"    Passed green artifact filter: {overall_quality_stats['passed_green_check']} ({overall_quality_stats['passed_green_check']/total*100:.1f}%)")
                print(f"    Passed background filter: {overall_quality_stats['passed_background_check']} ({overall_quality_stats['passed_background_check']/total*100:.1f}%)")
                print(f"    Final patches (all filters): {overall_quality_stats['passed_all_filters']} ({overall_quality_stats['passed_all_filters']/total*100:.1f}%)")
        
        return str(h5_path)


def main():
    parser = argparse.ArgumentParser(description='Improved CLAM preprocessing with lenient quality filtering')
    parser.add_argument('--source', type=str, required=True, help='Source directory with patient folders')
    parser.add_argument('--save_dir', type=str, default='ECU_preprocessed_improved', help='Output directory')
    parser.add_argument('--patch_size', type=int, default=224, help='Patch size in pixels')
    parser.add_argument('--step_size', type=int, default=224, help='Step size for patch extraction')
    parser.add_argument('--tissue_thresh', type=float, default=0.3, help='Tissue threshold for patches (more lenient)')
    
    # Improved quality filtering parameters (more lenient)
    parser.add_argument('--blur_thresh', type=float, default=15.0, help='Minimum Laplacian variance (more lenient)')
    parser.add_argument('--color_var_thresh', type=float, default=3.0, help='Minimum color variance (more lenient)')
    parser.add_argument('--max_patches_per_tile', type=int, default=20, help='Maximum patches per tile (default: 20 for optimal performance)')
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir
    
    output_dir = Path(args.save_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Improved quality filtering parameters:")
    print(f"  Tissue threshold: {args.tissue_thresh} (lenient)")
    print(f"  Blur threshold: {args.blur_thresh} (lenient)")
    print(f"  Color variance threshold: {args.color_var_thresh} (lenient)")
    print(f"  RGB-based shadow detection: enabled")
    print(f"  Green artifact detection: enabled")
    print(f"  Background artifact detection: enabled")
    print(f"  Max patches per tile: {args.max_patches_per_tile}")
    
    processor = ImprovedQualityCLAMProcessor(
        patch_size=args.patch_size,
        step_size=args.step_size,
        tissue_thresh_patch=args.tissue_thresh,
        blur_thresh=args.blur_thresh,
        color_var_thresh=args.color_var_thresh,
        max_patches_per_tile=args.max_patches_per_tile
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
        print(f"Manifest saved to: {output_dir / 'process_list.csv'}")


if __name__ == "__main__":
    main()