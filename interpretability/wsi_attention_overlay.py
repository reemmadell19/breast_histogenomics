# interpretability/wsi_attention_overlay.py

import os
import torch
import numpy as np
import tensorflow as tf
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd

from interpretability.unified_interpretability import UnifiedInterpretability

class WSIAttentionOverlay:
    """Create WSI visualization with attention overlay from TFRecords/H5 patches"""
    
    def __init__(self, patch_size: int = 256, display_size: int = 224):
        """
        Args:
            patch_size: Original patch size (256)
            display_size: Size patches were cropped to for model (224)
        """
        self.patch_size = patch_size
        self.display_size = display_size
        self.crop_margin = (patch_size - display_size) // 2
        
    def read_patches_from_tfrecord(self, tfrecord_path: str, 
                                   coordinates: np.ndarray,
                                   indices: Optional[np.ndarray] = None) -> Dict:
        """Read specific patches from TFRecord based on coordinates
        
        Args:
            tfrecord_path: Path to TFRecord file
            coordinates: Array of [x, y] coordinates
            indices: Optional indices of patches to load (for memory efficiency)
        """
        feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'slide': tf.io.FixedLenFeature([], tf.string),
            'loc_x': tf.io.FixedLenFeature([], tf.int64),
            'loc_y': tf.io.FixedLenFeature([], tf.int64),
        }
        
        def _parse(example_proto):
            return tf.io.parse_single_example(example_proto, feature_description)
        
        dataset = tf.data.TFRecordDataset(tfrecord_path).map(_parse)
        
        patches = {}
        patch_coords = {}
        
        for idx, record in enumerate(tqdm(dataset, desc="Loading patches")):
            loc_x = record['loc_x'].numpy()
            loc_y = record['loc_y'].numpy()
            
            # Check if this patch is in our coordinate list
            coord_match = np.where((coordinates[:, 0] == loc_x) & 
                                   (coordinates[:, 1] == loc_y))[0]
            
            if len(coord_match) > 0:
                patch_idx = coord_match[0]
                
                # Skip if we're only loading specific indices
                if indices is not None and patch_idx not in indices:
                    continue
                    
                raw = record['image_raw'].numpy()
                img = tf.io.decode_png(raw).numpy()
                
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
                
                patches[patch_idx] = img
                patch_coords[patch_idx] = (loc_x, loc_y)
        
        return {'patches': patches, 'coords': patch_coords}
    
    def read_patches_from_h5(self, h5_path: str,
                            coordinates: np.ndarray,
                            indices: Optional[np.ndarray] = None) -> Dict:
        """Read specific patches from H5 file based on coordinates"""
        patches = {}
        patch_coords = {}
        
        with h5py.File(h5_path, 'r') as f:
            all_patches = f['bag'][:]
            all_coords = f['coords'][:]  # Shape: (2, n_patches) based on your code
            
            # Transpose to match expected format
            all_coords = all_coords.T  # Now (n_patches, 2)
            
            for idx in range(len(all_patches)):
                loc_x, loc_y = all_coords[idx]
                
                # Check if this patch matches our coordinates
                coord_match = np.where((coordinates[:, 0] == loc_x) & 
                                       (coordinates[:, 1] == loc_y))[0]
                
                if len(coord_match) > 0:
                    patch_idx = coord_match[0]
                    
                    # Skip if we're only loading specific indices
                    if indices is not None and patch_idx not in indices:
                        continue
                    
                    patch = all_patches[idx]
                    if patch.shape[0] == 3:
                        patch = patch.transpose(1, 2, 0)
                    
                    patches[patch_idx] = patch.astype('uint8')
                    patch_coords[patch_idx] = (loc_x, loc_y)
        
        return {'patches': patches, 'coords': patch_coords}
    
    def create_attention_overlay(self,
                               attention_weights: np.ndarray,
                               coordinates: np.ndarray,
                               patch_source_path: str,
                               slide_id: str,
                               top_k: Optional[int] = None,
                               downsample_factor: int = 4,
                               alpha: float = 0.5,
                               colormap: str = 'jet') -> Tuple[np.ndarray, np.ndarray]:
        """Create WSI reconstruction with attention overlay
        
        Args:
            attention_weights: Attention weights for each patch
            coordinates: Patch coordinates
            patch_source_path: Path to TFRecord or H5 file with patches
            slide_id: Slide identifier
            top_k: Only load and display top-k attended patches (memory efficient)
            downsample_factor: Downsample factor for display
            alpha: Transparency for attention overlay
            colormap: Colormap for attention
        """
        
        # Modified: Load all patches by default
        if top_k is not None and top_k < len(attention_weights):
            top_indices = np.argsort(attention_weights)[-top_k:]
            print(f"  Loading top {top_k} patches for {slide_id}")
        else:
            top_indices = None  # This will load ALL patches
            print(f"  Loading all {len(attention_weights)} patches for {slide_id}")
            
        # Read patches based on file type
        if patch_source_path.endswith('.tfrecords'):
            patch_data = self.read_patches_from_tfrecord(
                patch_source_path, coordinates, top_indices)
        elif patch_source_path.endswith('.h5'):
            patch_data = self.read_patches_from_h5(
                patch_source_path, coordinates, top_indices)
        else:
            raise ValueError(f"Unsupported file format: {patch_source_path}")
        
        patches = patch_data['patches']
        patch_coords = patch_data['coords']
        
        if not patches:
            raise ValueError("No matching patches found!")
        
        print(f"Loaded {len(patches)} patches")
        
        # Calculate canvas dimensions
        all_coords = np.array(list(patch_coords.values()))
        min_x, min_y = all_coords.min(axis=0)
        max_x, max_y = all_coords.max(axis=0)
        
        # Add patch size to max coords
        canvas_width = (max_x - min_x + self.patch_size) // downsample_factor
        canvas_height = (max_y - min_y + self.patch_size) // downsample_factor
        
        print(f"Canvas size: {canvas_width} x {canvas_height}")
        
        # Create canvas for WSI reconstruction
        wsi_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # Create attention heatmap canvas
        attention_canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        
        # Place patches and attention values
        for idx, (patch_idx, (x, y)) in enumerate(patch_coords.items()):
            if patch_idx not in patches:
                continue
                
            patch = patches[patch_idx]
            
            # Apply center crop to match model input
            if self.crop_margin > 0:
                patch = patch[self.crop_margin:-self.crop_margin, 
                            self.crop_margin:-self.crop_margin]
            
            # Downsample patch
            if downsample_factor > 1:
                patch_pil = Image.fromarray(patch)
                new_size = (self.display_size // downsample_factor,
                           self.display_size // downsample_factor)
                patch_pil = patch_pil.resize(new_size, Image.LANCZOS)
                patch = np.array(patch_pil)
            
            # Calculate position on canvas
            x_pos = (x - min_x) // downsample_factor
            y_pos = (y - min_y) // downsample_factor
            
            patch_h, patch_w = patch.shape[:2]
            
            # Ensure we don't go out of bounds
            x_end = min(x_pos + patch_w, canvas_width)
            y_end = min(y_pos + patch_h, canvas_height)
            
            # Place patch on WSI canvas
            wsi_canvas[y_pos:y_end, x_pos:x_end] = patch[:y_end-y_pos, :x_end-x_pos]
            
            # Place attention value
            attention_value = attention_weights[patch_idx]
            attention_canvas[y_pos:y_end, x_pos:x_end] = attention_value
        
        # Smooth attention heatmap
        attention_canvas = gaussian_filter(attention_canvas, sigma=2)
        
        return wsi_canvas, attention_canvas
    
    def visualize_overlay(self,
                     wsi_canvas: np.ndarray,
                     attention_canvas: np.ndarray,
                     slide_id: str,
                     true_label: int,
                     pred_label: int,
                     pred_prob: float,
                     save_path: Optional[str] = None,
                     alpha: float = 0.3,  # Lower alpha for subtler overlay
                     colormap: str = 'RdBu_r',  # Blue-to-red colormap
                     show_colorbar: bool = True,
                     save_pdf: bool = False):
        """Create final visualization with WSI and attention overlay"""
        
        # Single figure with just the overlay
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Display WSI
        ax.imshow(wsi_canvas)
        
        # Normalize attention for better visualization
        if attention_canvas.max() > attention_canvas.min():
            attention_normalized = (attention_canvas - attention_canvas.min()) / \
                                (attention_canvas.max() - attention_canvas.min())
        else:
            attention_normalized = attention_canvas
        
        # Apply slight smoothing for cleaner appearance
        attention_smooth = gaussian_filter(attention_normalized, sigma=1.5)
        
        # Create custom colormap with transparency
        cmap = plt.get_cmap(colormap)
        
        # Only show attention where tissue is present (non-white areas)
        tissue_mask = np.any(wsi_canvas < 240, axis=2)  # Detect non-white regions
        attention_masked = np.ma.masked_where(~tissue_mask, attention_smooth)
        
        # Overlay attention
        im = ax.imshow(attention_masked, cmap=cmap, alpha=alpha, 
                    vmin=0, vmax=1, interpolation='bilinear')
        
        # Remove axes for cleaner look
        ax.axis('off')
        
        # Add title with prediction info
        risk_labels = ['Low Risk', 'High Risk']
        ax.set_title(
            f'{slide_id}\nTrue: {risk_labels[true_label]} | '
            f'Pred: {risk_labels[pred_label]} (prob: {pred_prob:.3f})',
            fontsize=12, pad=10
        )
        
        # Add colorbar on the side
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Attention Score', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            print(f"    Saved WSI overlay to: {save_path}")
            
            if save_pdf:
                pdf_path = save_path.replace('.png', '.pdf')
                plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
                        facecolor='white', edgecolor='none')
        
        plt.close(fig)
        return save_path