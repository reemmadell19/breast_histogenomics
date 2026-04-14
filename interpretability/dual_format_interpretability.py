import os
import torch
import numpy as np
from scipy.stats import entropy
from scipy.ndimage import label, gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict, List, Tuple, Optional
import pandas as pd
import json
import h5py
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

class DualFormatInterpretability:
    """Interpretability framework supporting both TFRecord and H5 patch formats"""
    
    def __init__(self, model_type: str, save_dir: str = 'results/interpretability', 
                 tfrecord_dir: str = None, h5_dir: str = None, patch_size: int = 224):
        self.model_type = model_type
        self.save_dir = save_dir
        self.tfrecord_dir = tfrecord_dir
        self.h5_dir = h5_dir
        self.patch_size = patch_size
        os.makedirs(save_dir, exist_ok=True)
        
        # Create subdirectories
        self.pdf_dir = os.path.join(save_dir, 'pdfs')
        self.patch_viz_dir = os.path.join(save_dir, 'patch_visualizations')
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.patch_viz_dir, exist_ok=True)
        
        # Cache for loaded data
        self.cache = {}
    def find_patch_file(self, slide_id: str) -> Tuple[Optional[str], str]:
        """Find patch file with enhanced debugging"""
        
        print(f"\n[DEBUG] Looking for patches for slide_id: '{slide_id}'")
        print(f"[DEBUG] Is numeric: {slide_id.strip().isdigit()}")
        print(f"[DEBUG] Contains UCH_BRCA_RS: {'UCH_BRCA_RS' in slide_id}")
        
        # For numeric-only IDs, check H5 files (BCR_NET)
        if slide_id.strip().isdigit():
            if self.h5_dir and os.path.exists(self.h5_dir):
                h5_path = os.path.join(self.h5_dir, f"{slide_id}.h5")
                print(f"[DEBUG] Checking H5 path: {h5_path}")
                print(f"[DEBUG] H5 exists: {os.path.exists(h5_path)}")
                if os.path.exists(h5_path):
                    print(f"✓ Found H5 (BCR_NET): {h5_path}")
                    return h5_path, 'h5'
                else:
                    # List available H5 files for debugging
                    h5_files = [f for f in os.listdir(self.h5_dir) if f.endswith('.h5')][:5]
                    print(f"[DEBUG] Sample H5 files in directory: {h5_files}")
        
        # For UCH_BRCA_RS IDs, check TFRecord files (UCMC)
        if 'UCH_BRCA_RS' in slide_id:
            if self.tfrecord_dir and os.path.exists(self.tfrecord_dir):
                tfrecord_path = os.path.join(self.tfrecord_dir, f"{slide_id}.tfrecords")
                print(f"[DEBUG] Checking TFRecord path: {tfrecord_path}")
                print(f"[DEBUG] TFRecord exists: {os.path.exists(tfrecord_path)}")
                if os.path.exists(tfrecord_path):
                    print(f"✓ Found TFRecord (UCMC): {tfrecord_path}")
                    return tfrecord_path, 'tfrecord'
        
        print(f"✗ Warning: No patch file found for '{slide_id}'")
        return None, 'none'
    def load_patches_from_h5(self, h5_path: str, max_patches: int = None) -> List[np.ndarray]:
        """Load patches from H5 file - Updated for BCR_NET format"""
        patches_list = []
        
        try:
            with h5py.File(h5_path, 'r') as f:
                # BCR_NET format has 'bag' key with CHW format
                if 'bag' in f:
                    patches_data = f['bag']
                    
                    n_patches = patches_data.shape[0]
                    n_to_load = min(n_patches, max_patches) if max_patches else n_patches
                    
                    for i in range(n_to_load):
                        patch = patches_data[i]  # Shape: (3, 224, 224)
                        
                        # Convert from CHW to HWC format for visualization
                        if patch.shape[0] == 3 and len(patch.shape) == 3:  # CHW format
                            patch = np.transpose(patch, (1, 2, 0))  # Convert to (224, 224, 3)
                        
                        # Ensure uint8
                        if patch.dtype != np.uint8:
                            if patch.max() <= 1.0:
                                patch = (patch * 255).astype(np.uint8)
                            else:
                                patch = patch.astype(np.uint8)
                        
                        patches_list.append(patch)
                    
                    print(f"Loaded {len(patches_list)} patches from H5 (BCR_NET format)")
                
                # Fallback to other possible keys if 'bag' not found
                elif 'patches' in f:
                    patches_data = f['patches']
                    n_patches = len(patches_data)
                    n_to_load = min(n_patches, max_patches) if max_patches else n_patches
                    
                    for i in range(n_to_load):
                        patch = patches_data[i]
                        if patch.shape[0] == 3 and len(patch.shape) == 3:
                            patch = np.transpose(patch, (1, 2, 0))
                        if patch.dtype != np.uint8:
                            patch = patch.astype(np.uint8)
                        patches_list.append(patch)
                    
                    print(f"Loaded {len(patches_list)} patches from H5")
                
                elif 'images' in f:
                    patches_data = f['images']
                    n_patches = len(patches_data)
                    n_to_load = min(n_patches, max_patches) if max_patches else n_patches
                    
                    for i in range(n_to_load):
                        patch = patches_data[i]
                        if patch.shape[-1] != 3 and patch.shape[0] == 3:
                            patch = np.transpose(patch, (1, 2, 0))
                        if patch.dtype != np.uint8:
                            patch = patch.astype(np.uint8)
                        patches_list.append(patch)
                    
                    print(f"Loaded {len(patches_list)} patches from H5")
                
                else:
                    print(f"No recognized patch data keys in H5 file. Keys found: {list(f.keys())}")
                    return []
                    
        except Exception as e:
            print(f"Error loading H5 file: {e}")
        
        return patches_list

    def parse_tfrecord_example(self, example_proto):
        """Parse a single TFRecord example"""
        feature_descriptions = [
            {'image': tf.io.FixedLenFeature([], tf.string)},
            {'image_raw': tf.io.FixedLenFeature([], tf.string)},
            {'image': tf.io.FixedLenFeature([], tf.string),
             'height': tf.io.FixedLenFeature([], tf.int64),
             'width': tf.io.FixedLenFeature([], tf.int64)}
        ]
        
        for desc in feature_descriptions:
            try:
                example = tf.io.parse_single_example(example_proto, desc)
                if 'image_raw' in example and 'image' not in example:
                    example['image'] = example['image_raw']
                return example
            except:
                continue
        
        raise ValueError("Could not parse TFRecord example")
    
    def load_patches_from_tfrecord(self, tfrecord_path: str, max_patches: int = None) -> List[np.ndarray]:
        """Load patches from TFRecord file"""
        patches_list = []
        
        try:
            dataset = tf.data.TFRecordDataset(tfrecord_path)
            
            for idx, raw_record in enumerate(dataset):
                if max_patches and idx >= max_patches:
                    break
                
                try:
                    example = self.parse_tfrecord_example(raw_record)
                    
                    if 'image' in example:
                        image = tf.io.decode_jpeg(example['image'], channels=3)
                        image = image.numpy()
                        patches_list.append(image)
                        
                except Exception as e:
                    continue
            
            print(f"Loaded {len(patches_list)} patches from TFRecord")
            
        except Exception as e:
            print(f"Error loading TFRecord: {e}")
        
        return patches_list
    
    def load_patches_by_index(self, slide_id: str, max_patches: int = None) -> List[np.ndarray]:
        """Load patches from either TFRecord or H5 format"""
        
        # Check cache first
        if slide_id in self.cache:
            return self.cache[slide_id]
        
        # Find the patch file
        patch_path, file_format = self.find_patch_file(slide_id)
        
        if patch_path is None:
            print(f"Warning: No patch file found for {slide_id}")
            return []
        
        print(f"Loading patches from {file_format.upper()}: {patch_path}")
        
        # Load based on format
        if file_format == 'h5':
            patches_list = self.load_patches_from_h5(patch_path, max_patches)
        elif file_format == 'tfrecord':
            patches_list = self.load_patches_from_tfrecord(patch_path, max_patches)
        else:
            patches_list = []
        
        # Cache if not too large
        if len(patches_list) < 10000:
            self.cache[slide_id] = patches_list
        
        return patches_list
    
    def extract_attention_weights(self, model, features: torch.Tensor) -> Optional[np.ndarray]:
        """Extract attention weights from MIL model"""
        model.eval()
        
        if self.model_type == 'mean':
            return None
            
        with torch.no_grad():
            if self.model_type == 'attention':
                if hasattr(model, 'get_attention_weights'):
                    attention_weights = model.get_attention_weights(features)
                else:
                    attention_scores = model.attention(features)
                    attention_weights = torch.softmax(attention_scores, dim=0)
                
            elif self.model_type == 'clam':
                if hasattr(model, 'get_attention_weights'):
                    attention_weights = model.get_attention_weights(features)
                else:
                    _, attention_weights = model(features, return_attention=True)
                
            elif self.model_type == 'acmil':
                if hasattr(model, 'get_attention_weights'):
                    attention_weights = model.get_attention_weights(features)
                else:
                    return None
            else:
                return None
        
        if attention_weights is not None:
            attention_weights = attention_weights.cpu().numpy()
            if attention_weights.ndim > 1:
                attention_weights = attention_weights.squeeze()
        
        return attention_weights
    
    def categorize_patches_by_attention(self, attention_weights: np.ndarray, 
                                       n_per_category: int = 5) -> Dict[str, np.ndarray]:
        """Categorize patches into high, medium, and low attention groups"""
        
        percentiles = np.percentile(attention_weights, [33, 67])
        
        low_mask = attention_weights <= percentiles[0]
        medium_mask = (attention_weights > percentiles[0]) & (attention_weights <= percentiles[1])
        high_mask = attention_weights > percentiles[1]
        
        low_indices = np.where(low_mask)[0]
        medium_indices = np.where(medium_mask)[0]
        high_indices = np.where(high_mask)[0]
        
        if len(low_indices) > 0:
            low_indices = low_indices[np.argsort(attention_weights[low_indices])][:n_per_category]
        if len(medium_indices) > 0:
            medium_sorted = medium_indices[np.argsort(attention_weights[medium_indices])]
            mid_point = len(medium_sorted) // 2
            start = max(0, mid_point - n_per_category // 2)
            end = min(len(medium_sorted), start + n_per_category)
            medium_indices = medium_sorted[start:end]
        if len(high_indices) > 0:
            high_indices = high_indices[np.argsort(attention_weights[high_indices])[-n_per_category:]]
        
        return {
            'high': high_indices,
            'medium': medium_indices,
            'low': low_indices
        }
    
    def visualize_attention_patches(self, 
                                   attention_weights: np.ndarray,
                                   coordinates: np.ndarray,
                                   slide_id: str,
                                   true_label: int,
                                   pred_label: int,
                                   pred_prob: float,
                                   n_patches_per_category: int = 5,
                                   save_pdf: bool = True) -> Optional[str]:
        """Visualize patches from high/medium/low attention regions"""
        
        # Load patches (automatically detects format)
        patches_list = self.load_patches_by_index(slide_id)
        
        if not patches_list:
            print(f"No patches found for {slide_id}")
            return None
        
        n_features = len(attention_weights)
        n_patches = len(patches_list)
        
        if n_patches < n_features:
            print(f"Warning: {n_patches} patches but {n_features} features")
        
        # Categorize patches
        patch_indices = self.categorize_patches_by_attention(
            attention_weights, n_patches_per_category
        )
        
        # Create figure
        fig = plt.figure(figsize=(20, 15))
        
        categories = [
            ('High Attention', patch_indices['high'], 'red'),
            ('Medium Attention', patch_indices['medium'], 'orange'),
            ('Low Attention', patch_indices['low'], 'blue')
        ]
        
        plot_idx = 1
        patches_found = False
        
        for cat_idx, (cat_name, indices, border_color) in enumerate(categories):
            for local_idx, patch_idx in enumerate(indices[:n_patches_per_category]):
                if patch_idx >= len(patches_list):
                    plot_idx += 1
                    continue
                
                patch_img = patches_list[patch_idx]
                attention_score = attention_weights[patch_idx]
                
                if coordinates is not None and patch_idx < len(coordinates):
                    coord_x, coord_y = coordinates[patch_idx]
                else:
                    coord_x, coord_y = 0, 0
                
                patches_found = True
                ax = plt.subplot(3, n_patches_per_category, plot_idx)
                
                ax.imshow(patch_img)
                ax.set_title(
                    f"Patch #{patch_idx}\n"
                    f"Att: {attention_score:.4f}\n"
                    f"Pos: ({int(coord_x)}, {int(coord_y)})",
                    fontsize=9,
                    color=border_color
                )
                ax.axis('off')
                
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
                    spine.set_visible(True)
                
                plot_idx += 1
        
        if not patches_found:
            plt.close()
            return None
        
        # Add category labels
        fig.text(0.02, 0.75, 'HIGH\nATTENTION', fontsize=12, fontweight='bold',
                rotation=0, va='center', ha='center', color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red"))
        fig.text(0.02, 0.5, 'MEDIUM\nATTENTION', fontsize=12, fontweight='bold',
                rotation=0, va='center', ha='center', color='orange',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="orange"))
        fig.text(0.02, 0.25, 'LOW\nATTENTION', fontsize=12, fontweight='bold',
                rotation=0, va='center', ha='center', color='blue',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="blue"))
        
        # Add title
        risk_label = ['Low Risk', 'High Risk']
        plt.suptitle(
            f'{self.model_type.upper()} - Patch Visualization - {slide_id}\n'
            f'True: {risk_label[true_label]}, Pred: {risk_label[pred_label]} (prob: {pred_prob:.3f})',
            fontsize=14, fontweight='bold', y=0.98
        )
        
        plt.tight_layout(rect=[0.05, 0, 1, 0.96])
        
        # Save
        base_path = os.path.join(self.patch_viz_dir, f'{slide_id}_attention_patches')
        png_path = f'{base_path}.png'
        pdf_path = f'{base_path}.pdf'
        
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        if save_pdf:
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            print(f"Saved: {pdf_path}")
        
        plt.close()
        
        return pdf_path if save_pdf else png_path
    
    def compute_metrics(self, attention_weights: np.ndarray) -> Dict:
        """Compute attention metrics"""
        if attention_weights is None:
            return {}
            
        metrics = {}
        
        metrics['entropy'] = float(entropy(attention_weights + 1e-8))
        metrics['normalized_entropy'] = metrics['entropy'] / np.log(len(attention_weights))
        metrics['gini'] = float(self.gini_coefficient(attention_weights))
        metrics['max_attention'] = float(np.max(attention_weights))
        metrics['top_5_mass'] = float(np.sum(np.sort(attention_weights)[-5:]))
        metrics['top_10_mass'] = float(np.sum(np.sort(attention_weights)[-10:]))
        metrics['top_20_mass'] = float(np.sum(np.sort(attention_weights)[-20:]))
        metrics['effective_size'] = float(np.exp(metrics['entropy']))
        metrics['effective_size_ratio'] = metrics['effective_size'] / len(attention_weights)
        metrics['mean_attention'] = float(np.mean(attention_weights))
        metrics['std_attention'] = float(np.std(attention_weights))
        metrics['n_patches'] = len(attention_weights)
        
        return metrics
    
    def gini_coefficient(self, x: np.ndarray) -> float:
        """Calculate Gini coefficient"""
        sorted_x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        if cumsum[-1] == 0:
            return 0.0
        return (2 * np.sum((np.arange(1, n+1)) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n
    
    def compute_spatial_coherence(self, attention_weights: np.ndarray, 
                                 coordinates: np.ndarray, 
                                 threshold_percentile: float = 75) -> Tuple[float, int]:
        """Measure spatial coherence of high-attention regions"""
        if coordinates is None or len(coordinates) == 0:
            return 0.0, 0
            
        threshold = np.percentile(attention_weights, threshold_percentile)
        high_attention_mask = attention_weights > threshold
        
        if np.sum(high_attention_mask) == 0:
            return 0.0, 0
        
        try:
            coords = np.array(coordinates)
            min_x, min_y = coords.min(axis=0)
            max_x, max_y = coords.max(axis=0)
            
            grid_size = self.patch_size
            grid_w = int((max_x - min_x) // grid_size + 1)
            grid_h = int((max_y - min_y) // grid_size + 1)
            
            if grid_w <= 0 or grid_h <= 0:
                return 0.0, 0
            
            spatial_grid = np.zeros((grid_h, grid_w))
            
            for i, (x, y) in enumerate(coordinates):
                if high_attention_mask[i]:
                    grid_x = int((x - min_x) // grid_size)
                    grid_y = int((y - min_y) // grid_size)
                    if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
                        spatial_grid[grid_y, grid_x] = 1
            
            labeled_array, num_clusters = label(spatial_grid)
            max_possible_clusters = np.sum(high_attention_mask)
            coherence = 1.0 - (num_clusters - 1) / max(max_possible_clusters - 1, 1)
            
            return float(coherence), num_clusters
        except:
            return 0.0, 0
    
    def analyze_slide(self, 
                     model,
                     features: torch.Tensor,
                     coordinates: np.ndarray,
                     slide_id: str,
                     true_label: int,
                     pred_label: int,
                     pred_prob: float,
                     visualize_patches: bool = False,
                     n_patches_per_category: int = 5) -> Dict:
        """Analyze slide with patch visualization only"""
        
        # Extract attention weights
        attention_weights = self.extract_attention_weights(model, features)
        
        if attention_weights is None:
            return {'slide_id': slide_id, 'error': 'No attention weights available'}
        
        # Compute metrics
        metrics = self.compute_metrics(attention_weights)
        
        # Add spatial coherence
        coherence, n_clusters = self.compute_spatial_coherence(attention_weights, coordinates)
        metrics['spatial_coherence'] = coherence
        metrics['n_clusters'] = n_clusters
        
        # Add metadata
        metrics['slide_id'] = slide_id
        metrics['true_label'] = true_label
        metrics['pred_label'] = pred_label
        metrics['pred_prob'] = pred_prob
        metrics['correct'] = true_label == pred_label
        
        # Create patch visualization if requested
        if visualize_patches:
            patch_viz_path = self.visualize_attention_patches(
                attention_weights, coordinates, slide_id,
                true_label, pred_label, pred_prob,
                n_patches_per_category=n_patches_per_category,
                save_pdf=True
            )
            if patch_viz_path:
                metrics['patch_visualization_path'] = patch_viz_path
        
        return 
    