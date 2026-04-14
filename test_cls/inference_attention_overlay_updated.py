#!/usr/bin/env python3
"""
WSI Attention Overlay Visualization
Creates whole slide image reconstructions with attention heatmap overlays
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import h5py
from PIL import Image
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple, Optional, Union

# Project imports
from datasets.classification_mil_dataset import ClassificationMILDataset
from utils.mil_utils import mil_collate_fn
from utils.classification_evaluator import ClassificationEvaluator

# Model imports
from models.classification_model_updated import (
    MeanPoolingMILClassifier, MaxPoolingMILClassifier,
    AttentionMILClassifier, CLAMClassifier, ACMILClassifier
)

# Import interpretability module
try:
    from interpretability.unified_interpretability import UnifiedInterpretability
    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    INTERPRETABILITY_AVAILABLE = False
    print("Warning: Interpretability module not found. Interpretability analysis will be skipped.")


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
        """Read specific patches from TFRecord based on coordinates"""
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
        
        for idx, record in enumerate(tqdm(dataset, desc="Loading patches", leave=False)):
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
            all_coords = f['coords'][:]  # Shape: (2, n_patches)
            
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
    
    def apply_edge_blending(self, patch: np.ndarray, alpha: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
        """Apply edge blending mask to patch for smoother transitions"""
        h, w = patch.shape[:2]
        
        # Create edge blending mask
        mask = np.ones((h, w), dtype=np.float32)
        
        # Define edge width for blending (2-3 pixels)
        edge_width = 3
        
        # Apply gradual fade at edges
        for i in range(edge_width):
            fade_value = (i + 1) / edge_width * alpha
            mask[i, :] *= fade_value
            mask[-i-1, :] *= fade_value
            mask[:, i] *= fade_value
            mask[:, -i-1] *= fade_value
        
        return patch, mask
    
    def create_attention_overlay(self,
                               attention_weights: np.ndarray,
                               coordinates: np.ndarray,
                               patch_source_path: str,
                               slide_id: str,
                               top_k: Optional[int] = None,
                               downsample_factor: int = 4,
                               alpha: float = 0.3,
                               colormap: str = 'RdBu_r') -> Tuple[np.ndarray, np.ndarray]:
        """Create WSI reconstruction with attention overlay
        
        Args:
            attention_weights: Attention weights for each patch
            coordinates: Patch coordinates
            patch_source_path: Path to TFRecord or H5 file with patches
            slide_id: Slide identifier
            top_k: Only load top-k attended patches (None = load all)
            downsample_factor: Downsample factor for display
            alpha: Transparency for attention overlay
            colormap: Colormap for attention
        """
        
        # Determine which patches to load (default: load ALL)
        if top_k is not None and top_k < len(attention_weights):
            # Get indices of top-k attended patches
            top_indices = np.argsort(attention_weights)[-top_k:]
            print(f"  Loading top {top_k} patches for {slide_id}")
            print(f"  Attention range: {attention_weights[top_indices].min():.4f} - "
                  f"{attention_weights[top_indices].max():.4f}")
        else:
            top_indices = None
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
        
        print(f"  Successfully loaded {len(patches)} patches")
        
        # Calculate canvas dimensions
        all_coords = np.array(list(patch_coords.values()))
        min_x, min_y = all_coords.min(axis=0)
        max_x, max_y = all_coords.max(axis=0)
        
        # Add patch size to max coords
        canvas_width = (max_x - min_x + self.patch_size) // downsample_factor
        canvas_height = (max_y - min_y + self.patch_size) // downsample_factor
        
        print(f"  Canvas size: {canvas_width} x {canvas_height}")
        
        # Create canvas for WSI reconstruction (white background)
        wsi_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.float32) * 255
        
        # Create weight canvas for blending
        weight_canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        
        # Create attention heatmap canvas
        attention_canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        attention_weight_canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        
        # Place patches and attention values with edge blending
        for idx, (patch_idx, (x, y)) in enumerate(tqdm(patch_coords.items(), 
                                                       desc="  Placing patches", 
                                                       leave=False)):
            if patch_idx not in patches:
                continue
                
            patch = patches[patch_idx].astype(np.float32)
            
            # Apply center crop to match model input
            if self.crop_margin > 0:
                patch = patch[self.crop_margin:-self.crop_margin, 
                            self.crop_margin:-self.crop_margin]
            
            # Downsample patch if needed
            if downsample_factor > 1:
                patch_pil = Image.fromarray(patch.astype('uint8'))
                new_size = (self.display_size // downsample_factor,
                           self.display_size // downsample_factor)
                patch_pil = patch_pil.resize(new_size, Image.LANCZOS)
                patch = np.array(patch_pil, dtype=np.float32)
            
            # Apply edge blending
            patch, mask = self.apply_edge_blending(patch, alpha=0.95)
            
            # Calculate position on canvas
            x_pos = (x - min_x) // downsample_factor
            y_pos = (y - min_y) // downsample_factor
            
            patch_h, patch_w = patch.shape[:2]
            
            # Ensure we don't go out of bounds
            x_end = min(x_pos + patch_w, canvas_width)
            y_end = min(y_pos + patch_h, canvas_height)
            
            # Get the actual patch region size
            actual_h = y_end - y_pos
            actual_w = x_end - x_pos
            
            # Weighted blending for WSI canvas
            current_weight = weight_canvas[y_pos:y_end, x_pos:x_end]
            patch_mask = mask[:actual_h, :actual_w]
            
            # Update WSI canvas with weighted average
            for c in range(3):
                current_values = wsi_canvas[y_pos:y_end, x_pos:x_end, c]
                new_values = patch[:actual_h, :actual_w, c]
                
                # Weighted average based on existing and new weights
                total_weight = current_weight + patch_mask
                total_weight = np.where(total_weight > 0, total_weight, 1.0)
                
                wsi_canvas[y_pos:y_end, x_pos:x_end, c] = (
                    (current_values * current_weight + new_values * patch_mask) / 
                    total_weight
                )
            
            # Update weight canvas
            weight_canvas[y_pos:y_end, x_pos:x_end] = np.maximum(
                current_weight, patch_mask
            )
            
            # Place attention value with the same weighting
            attention_value = attention_weights[patch_idx]
            current_attention = attention_canvas[y_pos:y_end, x_pos:x_end]
            current_attention_weight = attention_weight_canvas[y_pos:y_end, x_pos:x_end]
            
            total_attention_weight = current_attention_weight + patch_mask
            total_attention_weight = np.where(total_attention_weight > 0, total_attention_weight, 1.0)
            
            attention_canvas[y_pos:y_end, x_pos:x_end] = (
                (current_attention * current_attention_weight + attention_value * patch_mask) / 
                total_attention_weight
            )
            
            attention_weight_canvas[y_pos:y_end, x_pos:x_end] = np.maximum(
                current_attention_weight, patch_mask
            )
        
        # Convert WSI canvas back to uint8
        wsi_canvas = np.clip(wsi_canvas, 0, 255).astype(np.uint8)
        
        # Smooth attention heatmap with smaller sigma for finer details
        attention_canvas = gaussian_filter(attention_canvas, sigma=1.5)
        
        return wsi_canvas, attention_canvas
    
    def visualize_overlay(self,
                         wsi_canvas: np.ndarray,
                         attention_canvas: np.ndarray,
                         slide_id: str,
                         true_label: int,
                         pred_label: int,
                         pred_prob: float,
                         save_path: Optional[str] = None,
                         alpha: float = 0.3,
                         colormap: str = 'RdBu_r',
                         show_colorbar: bool = True,
                         save_pdf: bool = False,
                         create_multi_panel: bool = False):
        """Create final visualization with WSI and attention overlay
        
        Args:
            create_multi_panel: If True, create 3-panel view. If False, single overlay only.
        """
        
        if create_multi_panel:
            # Original multi-panel visualization
            fig, axes = plt.subplots(1, 3, figsize=(20, 7))
            
            # 1. Original WSI reconstruction
            axes[0].imshow(wsi_canvas)
            axes[0].set_title(f'WSI Reconstruction - {slide_id}', fontsize=12)
            axes[0].axis('off')
            
            # 2. Attention heatmap only
            im = axes[1].imshow(attention_canvas, cmap=colormap, interpolation='bilinear')
            axes[1].set_title('Attention Heatmap', fontsize=12)
            axes[1].axis('off')
            if show_colorbar:
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            # 3. Overlay
            axes[2].imshow(wsi_canvas)
            
            # Normalize attention
            if attention_canvas.max() > attention_canvas.min():
                attention_normalized = (attention_canvas - attention_canvas.min()) / \
                                     (attention_canvas.max() - attention_canvas.min())
            else:
                attention_normalized = attention_canvas
            
            # Detect tissue regions (non-white areas)
            tissue_mask = np.any(wsi_canvas < 240, axis=2)
            
            # Apply attention only on tissue
            attention_masked = np.ma.masked_where(~tissue_mask, attention_normalized)
            
            im_overlay = axes[2].imshow(attention_masked, cmap=colormap, 
                                       alpha=alpha, interpolation='bilinear',
                                       vmin=0, vmax=1)
            axes[2].set_title('WSI with Attention Overlay', fontsize=12)
            axes[2].axis('off')
            
            # Add prediction info
            risk_labels = ['Low Risk', 'High Risk']
            fig.suptitle(
                f'True: {risk_labels[true_label]} | '
                f'Pred: {risk_labels[pred_label]} (prob: {pred_prob:.3f})',
                fontsize=14, fontweight='bold'
            )
            
            plt.tight_layout()
            
        else:
            # Single clean overlay visualization (like your reference image)
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
            attention_smooth = gaussian_filter(attention_normalized, sigma=1.2)
            
            # Detect tissue regions (non-white areas)
            # More sophisticated tissue detection
            gray_wsi = np.mean(wsi_canvas, axis=2)
            tissue_mask = gray_wsi < 245  # Slightly more inclusive threshold
            
            # Clean up tissue mask
            from scipy.ndimage import binary_erosion, binary_dilation
            tissue_mask = binary_dilation(binary_erosion(tissue_mask, iterations=1), iterations=1)
            
            # Apply attention only on tissue regions
            attention_masked = np.ma.masked_where(~tissue_mask, attention_smooth)
            
            # Create custom colormap with better transparency handling
            cmap = plt.get_cmap(colormap)
            
            # Overlay attention with custom normalization
            # Use percentile-based normalization for better contrast
            vmin = np.percentile(attention_smooth[tissue_mask], 5)
            vmax = np.percentile(attention_smooth[tissue_mask], 95)
            
            im = ax.imshow(attention_masked, cmap=cmap, alpha=alpha, 
                          vmin=vmin, vmax=vmax, interpolation='bilinear')
            
            # Remove axes for cleaner look
            ax.axis('off')
            
            # Add title with prediction info
            risk_labels = ['Low Risk', 'High Risk']
            title_color = 'green' if pred_label == true_label else 'red'
            ax.set_title(
                f'{slide_id}\n'
                f'True: {risk_labels[true_label]} | '
                f'Pred: {risk_labels[pred_label]} ({pred_prob:.2%})',
                fontsize=14, pad=15, color='black'
            )
            
            # Add colorbar on the side
            if show_colorbar:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Attention Score', rotation=270, labelpad=20, fontsize=12)
                cbar.ax.tick_params(labelsize=10)
            
            plt.tight_layout()
        
        if save_path:
            # Save as PNG with high quality
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"    Saved WSI overlay to: {save_path}")
            
            # Optionally save as PDF
            if save_pdf:
                pdf_path = save_path.replace('.png', '.pdf')
                plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                print(f"    Saved PDF to: {pdf_path}")
            
            # Also save high-resolution overlay only (no title/colorbar)
            fig2, ax2 = plt.subplots(figsize=(12, 10))
            ax2.imshow(wsi_canvas)
            ax2.imshow(attention_masked, cmap=colormap, alpha=alpha,
                      vmin=vmin, vmax=vmax, interpolation='bilinear')
            ax2.axis('off')
            
            overlay_only_path = save_path.replace('.png', '_overlay_only.png')
            plt.savefig(overlay_only_path, dpi=200, bbox_inches='tight', 
                       pad_inches=0, facecolor='white', edgecolor='none')
            print(f"    Saved high-res overlay to: {overlay_only_path}")
            
            plt.close(fig2)
        
        plt.close(fig)
        
        return save_path


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def load_config_from_folder(model_folder):
    """Load configuration from training folder"""
    config_path = os.path.join(model_folder, "config_used.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found in {model_folder}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def get_input_dim_from_model_name(model_name):
    """Get input dimensions for foundation models"""
    dim_map = {
        "resnet18": 512,
        "resnet50": 2048,
        "conch": 512,
        "uni2-h": 1536,
        "virchow2": 1280,
        "h-optimus": 1536
    }
    return dim_map.get(model_name, 512)


def create_model_from_config(config, device):
    """Recreate model architecture from config"""
    
    mil_architecture = config['mil_architecture']
    input_dim = get_input_dim_from_model_name(config['model_name'])
    n_classes = 2
    
    # Get architecture-specific params
    arch_params = config.get('architecture_specific_params', {})
    if not arch_params:
        arch_params = config  # Backward compatibility
    
    if mil_architecture == "mean":
        model = MeanPoolingMILClassifier(
            input_dim=input_dim,
            hidden_dim=arch_params.get('hidden_dim', 256),
            n_classes=n_classes,
            dropout=arch_params.get('dropout', 0.25)
        )
    
    elif mil_architecture == "attention":
        model = AttentionMILClassifier(
            input_dim=input_dim,
            hidden_dim=arch_params.get('hidden_dim', 256),
            attention_hidden_dim=arch_params.get('attention_hidden_dim', 128),
            n_classes=n_classes,
            dropout=arch_params.get('dropout', 0.25)
        )
    
    elif mil_architecture == "clam":
        model = CLAMClassifier(
            input_dim=input_dim,
            hidden_dim=arch_params.get('hidden_dim', 512),
            attention_hidden_dim=arch_params.get('attention_hidden_dim', 384),
            n_classes=n_classes,
            dropout=arch_params.get('dropout', 0.25),
            gate=arch_params.get('gate', True),
            instance_eval=arch_params.get('use_instance_learning', True),
            k_sample=arch_params.get('k_sample', 8),
            instance_loss_fn=arch_params.get('instance_loss_fn', 'svm')
        )
    
    elif mil_architecture == "acmil":
        model = ACMILClassifier(
            input_dim=input_dim,
            hidden_dim=arch_params.get('hidden_dim', 256),
            n_classes=n_classes,
            n_branches=arch_params.get('n_branches', 10),
            dropout=arch_params.get('dropout', 0.5),
            top_k=arch_params.get('top_k', 10),
            mask_ratio=arch_params.get('mask_ratio', 0.7),
            lambda_p=arch_params.get('lambda_p', 1.0),
            lambda_d=arch_params.get('lambda_d', 1.0),
            gate=arch_params.get('gate', True)
        )
    
    else:
        raise ValueError(f"Unknown MIL architecture: {mil_architecture}")
    
    return model.to(device)


def run_interpretability_analysis(model, test_loader, test_csv, device, mil_architecture, 
                                 output_dir, n_visualize=10, dataset_name="test", 
                                 save_pdf=True, create_wsi_overlay=True,
                                 patch_manifest_path=None, top_k_patches=None,
                                 downsample_factor=4, multi_panel=False):
    """Run interpretability analysis with optional WSI overlay
    
    Args:
        top_k_patches: Number of top patches to load (None = load all)
        multi_panel: Create 3-panel view vs single overlay
    """
    
    if not INTERPRETABILITY_AVAILABLE:
        print("Interpretability module not available. Skipping analysis.")
        return None
    
    # Create interpretability analyzer
    interp_dir = os.path.join(output_dir, f'interpretability_{dataset_name}')
    analyzer = UnifiedInterpretability(model_type=mil_architecture, save_dir=interp_dir)
    
    if mil_architecture == 'mean':
        print(f"Skipping interpretability for {mil_architecture} (no attention mechanism)")
        return None
    
    print(f"\nRunning interpretability analysis for {dataset_name} set...")
    print(f"  Saving visualizations as: PNG{' and PDF' if save_pdf else ''}")
    if create_wsi_overlay:
        patches_msg = "all patches" if top_k_patches is None else f"top {top_k_patches} patches"
        print(f"  Creating WSI overlays: Yes ({patches_msg})")
        print(f"  Visualization style: {'Multi-panel' if multi_panel else 'Single overlay'}")
    
    # Load patch manifest if provided
    patch_mapping = {}
    if create_wsi_overlay and patch_manifest_path and os.path.exists(patch_manifest_path):
        manifest_df = pd.read_csv(patch_manifest_path)
        for _, row in manifest_df.iterrows():
            slide_id = os.path.splitext(row['file_name'])[0]
            if row['dataset'] == "UCMC":
                full_path = os.path.join("data/raw/UCMC", row['file_name'])
            elif row['dataset'] == "BCRNet":
                full_path = os.path.join("data/raw/BCR_NET", row['file_name'])
            else:
                continue
            patch_mapping[slide_id] = full_path
        print(f"  Loaded patch mapping for {len(patch_mapping)} slides")
    
    # Load the test CSV to get slide IDs and paths
    test_df = pd.read_csv(test_csv)
    
    # Create WSI overlay generator if needed
    wsi_overlay_gen = WSIAttentionOverlay() if create_wsi_overlay else None
    
    all_metrics = []
    visualized_count = 0
    high_risk_visualized = 0
    low_risk_visualized = 0
    misclass_visualized = 0
    
    model.eval()
    with torch.no_grad():
        for idx, (features, label) in enumerate(tqdm(test_loader, desc="Interpretability")):
            # Handle batch format
            if isinstance(features, list):
                features = features[0]
            if isinstance(label, list):
                label = label[0]
            elif isinstance(label, torch.Tensor):
                label = label.item()
            
            # Get the corresponding row from the CSV
            csv_row = test_df.iloc[idx]
            feature_path = csv_row['path']
            slide_id = csv_row['slide_id']
            
            # Load the feature file to get coordinates
            if os.path.exists(feature_path):
                full_data = torch.load(feature_path, map_location='cpu')
                
                # Extract coordinates
                if 'coords' in full_data:
                    coordinates = full_data['coords'].numpy()
                else:
                    print(f"Warning: No coordinates found for {slide_id}")
                    continue
            else:
                print(f"Warning: Feature file not found: {feature_path}")
                continue
            
            # Get prediction
            features = features.to(device)
            logits = model(features)
            if logits.dim() == 2 and logits.shape[0] == 1:
                logits = logits.squeeze(0)
            
            probs = torch.softmax(logits, dim=0)
            pred_label = torch.argmax(logits).item()
            pred_prob = probs[1].item()  # Probability of high-risk
            
            # Decide whether to visualize
            is_misclassified = (pred_label != label)
            is_high_risk_correct = (pred_label == 1 and label == 1)
            is_low_risk_correct = (pred_label == 0 and label == 0)
            
            visualize = False
            if visualized_count < n_visualize:
                # Prioritize misclassifications
                if is_misclassified and misclass_visualized < n_visualize // 3:
                    visualize = True
                    misclass_visualized += 1
                # Then high-risk correct
                elif is_high_risk_correct and high_risk_visualized < n_visualize // 3:
                    visualize = True
                    high_risk_visualized += 1
                # Then low-risk correct
                elif is_low_risk_correct and low_risk_visualized < n_visualize // 3:
                    visualize = True
                    low_risk_visualized += 1
                # Fill remaining slots
                elif visualized_count < n_visualize:
                    visualize = True
                
                if visualize:
                    visualized_count += 1
            
            # Analyze slide with PDF saving option
            slide_metrics = analyzer.analyze_slide(
                model=model,
                features=features,
                coordinates=coordinates,
                slide_id=slide_id,
                true_label=int(label),
                pred_label=pred_label,
                pred_prob=pred_prob,
                visualize=visualize,
                save_pdf=save_pdf
            )
            
            # Add WSI overlay if visualizing
            if visualize and create_wsi_overlay and wsi_overlay_gen and slide_id in patch_mapping:
                try:
                    print(f"\n  Creating WSI overlay for {slide_id}...")
                    
                    # Extract attention weights
                    attention_weights = analyzer.extract_attention_weights(model, features)
                    
                    if attention_weights is not None:
                        # Create WSI reconstruction with overlay
                        wsi_canvas, attention_canvas = wsi_overlay_gen.create_attention_overlay(
                            attention_weights=attention_weights,
                            coordinates=coordinates,
                            patch_source_path=patch_mapping[slide_id],
                            slide_id=slide_id,
                            top_k=top_k_patches,  # None = load all patches
                            downsample_factor=downsample_factor,
                            alpha=0.3,
                            colormap='RdBu_r'
                        )
                        
                        # Save visualization
                        wsi_save_path = os.path.join(interp_dir, f'{slide_id}_wsi_overlay.png')
                        wsi_overlay_gen.visualize_overlay(
                            wsi_canvas=wsi_canvas,
                            attention_canvas=attention_canvas,
                            slide_id=slide_id,
                            true_label=int(label),
                            pred_label=pred_label,
                            pred_prob=pred_prob,
                            save_path=wsi_save_path,
                            save_pdf=save_pdf,
                            create_multi_panel=multi_panel
                        )
                        
                        slide_metrics['wsi_overlay_path'] = wsi_save_path
                        
                except Exception as e:
                    print(f"    Warning: Could not create WSI overlay for {slide_id}: {str(e)[:100]}")
            
            # Add RS score if available
            slide_metrics['RS_score'] = csv_row['RS']
            slide_metrics['RS_category'] = csv_row['RSHigh']
            
            all_metrics.append(slide_metrics)
    
    # Create summary DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save results and print summary
    if len(metrics_df) > 0:
        metrics_df.to_csv(os.path.join(interp_dir, f'{dataset_name}_interpretability_metrics.csv'), 
                         index=False)
        
        print(f"\n  Interpretability Summary:")
        print(f"    Samples analyzed: {len(metrics_df)}")
        print(f"    Visualizations saved: {visualized_count}")
        print(f"      Misclassified: {misclass_visualized}")
        print(f"      High-risk correct: {high_risk_visualized}")
        print(f"      Low-risk correct: {low_risk_visualized}")
        print(f"    Results saved to: {interp_dir}")
    
    return metrics_df


def test_individual_folds(model_folder, test_csv, output_dir=None, run_interpretability=False, 
                         analyze_all_folds=False, n_visualize=15, save_pdf=True,
                         create_wsi_overlay=False, patch_manifest_path=None,
                         top_k_patches=None, downsample_factor=4, multi_panel=False):
    """Test each fold individually without ensemble
    
    Args:
        top_k_patches: Number of top patches to load (None = load all)
        multi_panel: Create 3-panel view vs single overlay
    """
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load config
    config = load_config_from_folder(model_folder)
    print(f"\nLoaded configuration:")
    print(f"Model: {config['model_name']}")
    print(f"Architecture: {config['mil_architecture']}")
    
    # Create output directory
    if output_dir is None:
        model_name = os.path.basename(model_folder)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"test_results_cls_overlay/{model_name}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")
    
    # Load test dataset
    print(f"\nLoading test dataset: {test_csv}")
    test_dataset = ClassificationMILDataset(
        test_csv,
        label_column='RSHigh' if 'RSHigh' in pd.read_csv(test_csv).columns else 'RS',
        threshold=25.0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=mil_collate_fn,
        num_workers=0
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Find best fold from validation results
    best_fold_num = 1
    best_fold_auroc = -1
    
    for fold in range(1, 6):
        checkpoint_path = os.path.join(model_folder, f'best_model_fold_{fold}.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            val_auroc = checkpoint.get('best_auroc', 0)
            if val_auroc > best_fold_auroc:
                best_fold_auroc = val_auroc
                best_fold_num = fold
    
    print(f"\nBest fold: {best_fold_num} (Val AUROC: {best_fold_auroc:.4f})")
    
    # Run interpretability on best fold
    if run_interpretability:
        print(f"\n{'='*60}")
        print(f"Running interpretability analysis on best fold (Fold {best_fold_num})")
        print(f"{'='*60}")
        
        checkpoint_path = os.path.join(model_folder, f'best_model_fold_{best_fold_num}.pt')
        model = create_model_from_config(config, device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        interp_results = run_interpretability_analysis(
            model=model,
            test_loader=test_loader,
            test_csv=test_csv,
            device=device,
            mil_architecture=config['mil_architecture'],
            output_dir=output_dir,
            n_visualize=n_visualize,
            dataset_name=f"best_fold_{best_fold_num}",
            save_pdf=save_pdf,
            create_wsi_overlay=create_wsi_overlay,
            patch_manifest_path=patch_manifest_path,
            top_k_patches=top_k_patches,
            downsample_factor=downsample_factor,
            multi_panel=multi_panel
        )
    
    print(f"\n{'='*80}")
    print(f"All results saved to: {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test CV models with WSI attention overlay')
    parser.add_argument('--model-folder', type=str, required=True,
                      help='Path to folder containing fold models')
    parser.add_argument('--test-csv', type=str, required=True,
                      help='Path to test CSV file')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory (default: auto-generated)')
    parser.add_argument('--run-interpretability', action='store_true',
                      help='Run interpretability analysis on test set')
    parser.add_argument('--n-visualize', type=int, default=15,
                      help='Number of cases to visualize (default: 15)')
    parser.add_argument('--no-pdf', action='store_true',
                      help='Disable PDF saving for visualizations')
    parser.add_argument('--wsi-overlay', action='store_true',
                      help='Create WSI reconstruction with attention overlay')
    parser.add_argument('--patch-manifest', type=str, default=None,
                      help='Path to manifest CSV mapping slide_ids to patch files')
    parser.add_argument('--top-k-patches', type=int, default=None,
                      help='Number of top patches to load (default: None = load all)')
    parser.add_argument('--downsample-factor', type=int, default=4,
                      help='Downsampling factor for WSI reconstruction (default: 4)')
    parser.add_argument('--multi-panel', action='store_true',
                      help='Create 3-panel visualization instead of single overlay')
    
    args = parser.parse_args()
    
    # Determine whether to save PDFs
    save_pdf = not args.no_pdf
    
    test_individual_folds(
        model_folder=args.model_folder, 
        test_csv=args.test_csv, 
        output_dir=args.output_dir,
        run_interpretability=args.run_interpretability,
        n_visualize=args.n_visualize,
        save_pdf=save_pdf,
        create_wsi_overlay=args.wsi_overlay,
        patch_manifest_path=args.patch_manifest,
        top_k_patches=args.top_k_patches,
        downsample_factor=args.downsample_factor,
        multi_panel=args.multi_panel
    )