import os
import torch
import numpy as np
from scipy.stats import entropy
from scipy.ndimage import label, gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
import json
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

class IndexBasedTFRecordInterpretability:
    """Complete interpretability framework using patch indices instead of coordinates"""
    
    def __init__(self, model_type: str, save_dir: str = 'results/interpretability', 
                 tfrecord_dir: str = None, patch_size: int = 224):
        self.model_type = model_type
        self.save_dir = save_dir
        self.tfrecord_dir = tfrecord_dir
        self.patch_size = patch_size
        os.makedirs(save_dir, exist_ok=True)
        
        # Create subdirectories
        self.pdf_dir = os.path.join(save_dir, 'pdfs')
        self.patch_viz_dir = os.path.join(save_dir, 'patch_visualizations')
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.patch_viz_dir, exist_ok=True)
        
        # Cache for TFRecord data
        self.tfrecord_cache = {}
    
    def find_tfrecord_path(self, slide_id: str) -> Optional[str]:
        """Find TFRecord file path from slide ID"""
        if self.tfrecord_dir is None:
            return None
        
        # Try different naming patterns
        patterns = [
            f"{slide_id}.tfrecords",
            f"{slide_id}.tfrecord",
            f"{slide_id}",
            f"{slide_id}_patches.tfrecords"
        ]
        
        for pattern in patterns:
            potential_path = os.path.join(self.tfrecord_dir, pattern)
            if os.path.exists(potential_path):
                return potential_path
        
        # Try to find any tfrecord containing the slide_id
        if os.path.exists(self.tfrecord_dir):
            for file in os.listdir(self.tfrecord_dir):
                if slide_id in file and ('.tfrecord' in file or '.tfrecords' in file):
                    return os.path.join(self.tfrecord_dir, file)
        
        return None
    
    def parse_tfrecord_example(self, example_proto):
        """Parse a single TFRecord example"""
        feature_descriptions = [
            # Most common format
            {
                'image': tf.io.FixedLenFeature([], tf.string),
            },
            # Alternative with image_raw
            {
                'image_raw': tf.io.FixedLenFeature([], tf.string),
            },
            # With additional fields
            {
                'image': tf.io.FixedLenFeature([], tf.string),
                'height': tf.io.FixedLenFeature([], tf.int64),
                'width': tf.io.FixedLenFeature([], tf.int64),
            }
        ]
        
        for desc in feature_descriptions:
            try:
                example = tf.io.parse_single_example(example_proto, desc)
                # Normalize to 'image' key
                if 'image_raw' in example and 'image' not in example:
                    example['image'] = example['image_raw']
                return example
            except:
                continue
        
        raise ValueError("Could not parse TFRecord example with any known format")
    
    def load_patches_from_tfrecord_by_index(self, slide_id: str, 
                                           max_patches: int = None) -> List[np.ndarray]:
        """Load patches from TFRecord file and return them as an ordered list"""
        
        # Check cache first
        if slide_id in self.tfrecord_cache:
            return self.tfrecord_cache[slide_id]
        
        tfrecord_path = self.find_tfrecord_path(slide_id)
        if tfrecord_path is None:
            print(f"Warning: TFRecord not found for {slide_id}")
            return []
        
        print(f"Loading patches from: {tfrecord_path}")
        
        patches_list = []
        
        try:
            # Read TFRecord file
            dataset = tf.data.TFRecordDataset(tfrecord_path)
            
            for idx, raw_record in enumerate(dataset):
                if max_patches and idx >= max_patches:
                    break
                
                try:
                    example = self.parse_tfrecord_example(raw_record)
                    
                    # Decode image
                    if 'image' in example:
                        image = tf.io.decode_jpeg(example['image'], channels=3)
                        image = image.numpy()
                        patches_list.append(image)
                        
                except Exception as e:
                    print(f"Error parsing record {idx}: {e}")
                    continue
            
            print(f"Loaded {len(patches_list)} patches from TFRecord")
            
            # Cache for future use if not too large
            if len(patches_list) < 10000:
                self.tfrecord_cache[slide_id] = patches_list
            
        except Exception as e:
            print(f"Error loading TFRecord for {slide_id}: {e}")
        
        return patches_list
    
    def extract_attention_weights(self, model, features: torch.Tensor) -> Optional[np.ndarray]:
        """Extract attention weights from any MIL model"""
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
    
    def compute_metrics(self, attention_weights: np.ndarray) -> Dict:
        """Compute all universal metrics"""
        if attention_weights is None:
            return {}
            
        metrics = {}
        
        # Distribution metrics
        metrics['entropy'] = float(entropy(attention_weights + 1e-8))
        metrics['normalized_entropy'] = metrics['entropy'] / np.log(len(attention_weights))
        metrics['gini'] = float(self.gini_coefficient(attention_weights))
        
        # Concentration metrics
        metrics['max_attention'] = float(np.max(attention_weights))
        metrics['top_5_mass'] = float(np.sum(np.sort(attention_weights)[-5:]))
        metrics['top_10_mass'] = float(np.sum(np.sort(attention_weights)[-10:]))
        metrics['top_20_mass'] = float(np.sum(np.sort(attention_weights)[-20:]))
        
        # Effective size
        metrics['effective_size'] = float(np.exp(metrics['entropy']))
        metrics['effective_size_ratio'] = metrics['effective_size'] / len(attention_weights)
        
        # Statistics
        metrics['mean_attention'] = float(np.mean(attention_weights))
        metrics['std_attention'] = float(np.std(attention_weights))
        metrics['n_patches'] = len(attention_weights)
        
        return metrics
    
    def gini_coefficient(self, x: np.ndarray) -> float:
        """Calculate Gini coefficient for attention inequality"""
        sorted_x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        if cumsum[-1] == 0:
            return 0.0
        return (2 * np.sum((np.arange(1, n+1)) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n
    
    def compute_spatial_coherence(self, attention_weights: np.ndarray, 
                                 coordinates: np.ndarray, 
                                 threshold_percentile: float = 75) -> Tuple[float, int]:
        """Measure if high-attention regions are spatially coherent"""
        # If no coordinates available, return default values
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
            
            # Count connected components
            labeled_array, num_clusters = label(spatial_grid)
            
            # Coherence is inverse of number of clusters (normalized)
            max_possible_clusters = np.sum(high_attention_mask)
            coherence = 1.0 - (num_clusters - 1) / max(max_possible_clusters - 1, 1)
            
            return float(coherence), num_clusters
        except Exception as e:
            print(f"Error computing spatial coherence: {e}")
            return 0.0, 0
    
    def generate_heatmap(self, attention_weights: np.ndarray, 
                        coordinates: np.ndarray) -> np.ndarray:
        """Generate spatial attention heatmap"""
        if coordinates is None or len(coordinates) == 0:
            return np.zeros((100, 100))
            
        try:
            coords = np.array(coordinates)
            
            max_x = coords[:, 0].max() + self.patch_size
            max_y = coords[:, 1].max() + self.patch_size
            
            scale_factor = 4
            heatmap_h = int(max_y // scale_factor)
            heatmap_w = int(max_x // scale_factor)
            heatmap = np.zeros((heatmap_h, heatmap_w))
            
            for weight, (x, y) in zip(attention_weights, coordinates):
                x_scaled = int(x // scale_factor)
                y_scaled = int(y // scale_factor)
                patch_size_scaled = self.patch_size // scale_factor
                
                y_end = min(y_scaled + patch_size_scaled, heatmap_h)
                x_end = min(x_scaled + patch_size_scaled, heatmap_w)
                
                heatmap[y_scaled:y_end, x_scaled:x_end] = np.maximum(
                    heatmap[y_scaled:y_end, x_scaled:x_end], 
                    weight
                )
            
            heatmap = gaussian_filter(heatmap, sigma=2)
            
            return heatmap
        except Exception as e:
            print(f"Error generating heatmap: {e}")
            return np.zeros((100, 100))
    
    def categorize_patches_by_attention(self, attention_weights: np.ndarray, 
                                       n_per_category: int = 5) -> Dict[str, np.ndarray]:
        """Categorize patches into high, medium, and low attention groups"""
        
        # Calculate percentiles for three groups
        percentiles = np.percentile(attention_weights, [33, 67])
        
        # Get indices for each category
        low_mask = attention_weights <= percentiles[0]
        medium_mask = (attention_weights > percentiles[0]) & (attention_weights <= percentiles[1])
        high_mask = attention_weights > percentiles[1]
        
        low_indices = np.where(low_mask)[0]
        medium_indices = np.where(medium_mask)[0]
        high_indices = np.where(high_mask)[0]
        
        # Sort and select samples
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
    
    def visualize_attention_patches_by_index(self, 
                                            attention_weights: np.ndarray,
                                            coordinates: np.ndarray,
                                            slide_id: str,
                                            true_label: int,
                                            pred_label: int,
                                            pred_prob: float,
                                            n_patches_per_category: int = 5,
                                            save_pdf: bool = True) -> Optional[str]:
        """Visualize patches using index-based matching"""
        
        # Load all patches from TFRecord as ordered list
        patches_list = self.load_patches_from_tfrecord_by_index(slide_id)
        
        if not patches_list:
            print(f"No patches found in TFRecord for {slide_id}")
            return None
        
        # Check if we have enough patches
        n_features = len(attention_weights)
        n_patches = len(patches_list)
        
        if n_patches < n_features:
            print(f"Warning: TFRecord has {n_patches} patches but {n_features} features.")
        elif n_patches > n_features:
            print(f"Note: TFRecord has {n_patches} patches but only {n_features} features.")
        
        # Categorize patches by attention
        patch_indices = self.categorize_patches_by_attention(
            attention_weights, n_patches_per_category
        )
        
        # Create figure
        fig = plt.figure(figsize=(20, 15))
        
        # Define categories
        categories = [
            ('High Attention', patch_indices['high'], 'red'),
            ('Medium Attention', patch_indices['medium'], 'orange'),
            ('Low Attention', patch_indices['low'], 'blue')
        ]
        
        plot_idx = 1
        patches_found = False
        
        for cat_idx, (cat_name, indices, border_color) in enumerate(categories):
            for local_idx, patch_idx in enumerate(indices[:n_patches_per_category]):
                # Check if we have a patch for this index
                if patch_idx >= len(patches_list):
                    print(f"Skipping patch {patch_idx} - beyond available patches")
                    plot_idx += 1
                    continue
                
                # Get the patch by index
                patch_img = patches_list[patch_idx]
                attention_score = attention_weights[patch_idx]
                
                # Get coordinates if available (for display purposes)
                if coordinates is not None and patch_idx < len(coordinates):
                    coord_x, coord_y = coordinates[patch_idx]
                else:
                    coord_x, coord_y = 0, 0
                
                patches_found = True
                ax = plt.subplot(3, n_patches_per_category, plot_idx)
                
                # Display patch
                ax.imshow(patch_img)
                ax.set_title(
                    f"Patch #{patch_idx}\n"
                    f"Att: {attention_score:.4f}\n"
                    f"Pos: ({int(coord_x)}, {int(coord_y)})",
                    fontsize=9,
                    color=border_color
                )
                ax.axis('off')
                
                # Add colored border
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
                    spine.set_visible(True)
                
                plot_idx += 1
        
        if not patches_found:
            plt.close()
            print(f"No patches could be displayed for {slide_id}")
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
            f'{self.model_type.upper()} - Index-Based Patch Visualization - {slide_id}\n'
            f'True: {risk_label[true_label]}, Pred: {risk_label[pred_label]} (prob: {pred_prob:.3f})\n'
            f'Using patch indices (not coordinate matching)',
            fontsize=14, fontweight='bold', y=0.98
        )
        
        plt.tight_layout(rect=[0.05, 0, 1, 0.96])
        
        # Save as both PNG and PDF
        base_path = os.path.join(self.patch_viz_dir, f'{slide_id}_attention_patches')
        png_path = f'{base_path}.png'
        pdf_path = f'{base_path}.pdf'
        
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        if save_pdf:
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            print(f"Saved patch visualization: {pdf_path}")
        
        plt.close()
        
        return pdf_path if save_pdf else png_path
    
    def visualize_attention_comprehensive(self, 
                                         attention_weights: np.ndarray,
                                         coordinates: np.ndarray,
                                         slide_id: str,
                                         true_label: int,
                                         pred_label: int,
                                         pred_prob: float,
                                         save_pdf: bool = True) -> Optional[str]:
        """Create comprehensive attention visualization with PDF output"""
        
        if attention_weights is None:
            print(f"No attention weights available for {self.model_type}")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Attention heatmap (if coordinates available)
        if coordinates is not None and len(coordinates) > 0:
            heatmap = self.generate_heatmap(attention_weights, coordinates)
            im = axes[0, 0].imshow(heatmap, cmap='jet', interpolation='bilinear', aspect='auto')
            axes[0, 0].set_title(f'Attention Heatmap - {slide_id}')
            axes[0, 0].axis('off')
            plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
        else:
            axes[0, 0].text(0.5, 0.5, 'Spatial heatmap not available\n(no coordinates)', 
                          ha='center', va='center')
            axes[0, 0].set_title(f'Attention Heatmap - {slide_id}')
            axes[0, 0].axis('off')
        
        # 2. Attention distribution with categories
        axes[0, 1].hist(attention_weights, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        
        # Add vertical lines for percentiles
        percentiles = np.percentile(attention_weights, [33, 67])
        axes[0, 1].axvline(percentiles[0], color='blue', linestyle='--', linewidth=2, label='Low/Medium')
        axes[0, 1].axvline(percentiles[1], color='red', linestyle='--', linewidth=2, label='Medium/High')
        axes[0, 1].axvline(np.mean(attention_weights), color='green', linestyle='-', linewidth=2, 
                          label=f'Mean: {np.mean(attention_weights):.4f}')
        
        axes[0, 1].set_xlabel('Attention Weight')
        axes[0, 1].set_ylabel('Number of Patches')
        axes[0, 1].set_title(f'Attention Distribution\nEntropy: {entropy(attention_weights):.3f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Top-20 patches bar plot
        top_k_indices = np.argsort(attention_weights)[-20:][::-1]
        top_k_weights = attention_weights[top_k_indices]
        
        # Color bars based on attention level
        colors = []
        for w in top_k_weights:
            if w > percentiles[1]:
                colors.append('red')
            elif w > percentiles[0]:
                colors.append('orange')
            else:
                colors.append('blue')
        
        axes[1, 0].bar(range(len(top_k_weights)), top_k_weights, color=colors, edgecolor='black')
        axes[1, 0].set_xlabel('Patch Rank')
        axes[1, 0].set_ylabel('Attention Weight')
        axes[1, 0].set_title(f'Top-20 Patches\nCumulative: {np.sum(top_k_weights):.3f}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Spatial scatter plot (if coordinates available)
        if coordinates is not None and len(coordinates) > 0:
            colors = np.zeros(len(attention_weights))
            colors[attention_weights <= percentiles[0]] = 0  # Low
            colors[(attention_weights > percentiles[0]) & (attention_weights <= percentiles[1])] = 1  # Medium
            colors[attention_weights > percentiles[1]] = 2  # High
            
            scatter = axes[1, 1].scatter(coordinates[:, 0], coordinates[:, 1], 
                                        c=colors, cmap='RdYlBu_r', 
                                        s=20, alpha=0.6)
            axes[1, 1].set_xlabel('X coordinate')
            axes[1, 1].set_ylabel('Y coordinate')
            axes[1, 1].set_title('Spatial Distribution by Attention Level')
            axes[1, 1].invert_yaxis()
            
            cbar = plt.colorbar(scatter, ax=axes[1, 1], fraction=0.046)
            cbar.set_ticks([0, 1, 2])
            cbar.set_ticklabels(['Low', 'Medium', 'High'])
        else:
            axes[1, 1].text(0.5, 0.5, 'Spatial plot not available\n(no coordinates)', 
                          ha='center', va='center')
            axes[1, 1].set_title('Spatial Distribution')
            axes[1, 1].axis('off')
        
        # Add prediction info
        risk_label = ['Low Risk', 'High Risk']
        plt.suptitle(
            f'{self.model_type.upper()} Attention Analysis\n'
            f'True: {risk_label[true_label]}, Pred: {risk_label[pred_label]} (prob: {pred_prob:.3f})',
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        
        # Save as both PNG and PDF
        base_path = os.path.join(self.save_dir, f'{slide_id}_attention_analysis')
        png_path = f'{base_path}.png'
        pdf_path = f'{base_path}.pdf'
        
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        if save_pdf:
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            print(f"Saved attention analysis: {pdf_path}")
        
        plt.close()
        
        return pdf_path if save_pdf else png_path
    
    def create_summary_pdf(self, all_metrics: List[Dict], dataset_name: str = 'dataset'):
        """Create a comprehensive PDF report with all visualizations"""
        
        pdf_path = os.path.join(self.pdf_dir, f'{dataset_name}_interpretability_report.pdf')
        
        with PdfPages(pdf_path) as pdf:
            # Page 1: Summary statistics
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle(f'Interpretability Report - {dataset_name.upper()}\n{self.model_type.upper()} Architecture', 
                        fontsize=16, fontweight='bold')
            
            metrics_df = pd.DataFrame(all_metrics)
            valid_metrics_df = metrics_df[~metrics_df.get('error', False).astype(bool)] if 'error' in metrics_df.columns else metrics_df
            
            if len(valid_metrics_df) > 0:
                summary_text = f"""
Dataset: {dataset_name}
Model Type: {self.model_type.upper()}
Total Samples: {len(metrics_df)}
Valid Samples: {len(valid_metrics_df)}

Attention Metrics Summary:
- Mean Entropy: {valid_metrics_df['entropy'].mean():.3f} ± {valid_metrics_df['entropy'].std():.3f}
- Mean Gini: {valid_metrics_df['gini'].mean():.3f} ± {valid_metrics_df['gini'].std():.3f}
- Mean Spatial Coherence: {valid_metrics_df['spatial_coherence'].mean() if 'spatial_coherence' in valid_metrics_df else 'N/A'}
- Mean Top-10 Mass: {valid_metrics_df['top_10_mass'].mean():.3f} ± {valid_metrics_df['top_10_mass'].std():.3f}

Performance Breakdown:
- Correct Predictions: {valid_metrics_df['correct'].sum() if 'correct' in valid_metrics_df else 'N/A'} / {len(valid_metrics_df)}
- Accuracy: {valid_metrics_df['correct'].mean() if 'correct' in valid_metrics_df else 'N/A'}
                """
                
                plt.text(0.1, 0.5, summary_text, fontsize=12, 
                        transform=plt.gca().transAxes, verticalalignment='center')
                plt.axis('off')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"Summary PDF report saved to: {pdf_path}")
        return pdf_path
    
    def analyze_slide(self, 
                     model,
                     features: torch.Tensor,
                     coordinates: np.ndarray,
                     slide_id: str,
                     true_label: int,
                     pred_label: int,
                     pred_prob: float,
                     visualize: bool = False,
                     visualize_patches: bool = False,
                     n_patches_per_category: int = 5) -> Dict:
        """Complete analysis using index-based patch matching"""
        
        # Extract attention weights
        attention_weights = self.extract_attention_weights(model, features)
        
        if attention_weights is None:
            return {'slide_id': slide_id, 'error': 'No attention weights available'}
        
        # Compute metrics
        metrics = self.compute_metrics(attention_weights)
        
        # Add spatial coherence (will handle missing coordinates gracefully)
        coherence, n_clusters = self.compute_spatial_coherence(attention_weights, coordinates)
        metrics['spatial_coherence'] = coherence
        metrics['n_clusters'] = n_clusters
        
        # Add slide metadata
        metrics['slide_id'] = slide_id
        metrics['true_label'] = true_label
        metrics['pred_label'] = pred_label
        metrics['pred_prob'] = pred_prob
        metrics['correct'] = true_label == pred_label
        
        # Visualize if requested
        if visualize:
            # Create comprehensive attention analysis
            viz_path = self.visualize_attention_comprehensive(
                attention_weights, coordinates, slide_id,
                true_label, pred_label, pred_prob, save_pdf=True
            )
            if viz_path:
                metrics['visualization_path'] = viz_path
            
            # Create patch visualization using indices
            if visualize_patches and self.tfrecord_dir:
                patch_viz_path = self.visualize_attention_patches_by_index(
                    attention_weights, coordinates, slide_id,
                    true_label, pred_label, pred_prob,
                    n_patches_per_category=n_patches_per_category,
                    save_pdf=True
                )
                if patch_viz_path:
                    metrics['patch_visualization_path'] = patch_viz_path
        
        return metrics