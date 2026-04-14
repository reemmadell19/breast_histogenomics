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

class UnifiedInterpretability:
    """Universal interpretability framework for all attention-based MIL models"""
    
    def __init__(self, model_type: str, save_dir: str = 'results/interpretability'):
        self.model_type = model_type
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def extract_attention_weights(self, model, features: torch.Tensor) -> Optional[np.ndarray]:
        """Extract attention weights from any MIL model during forward pass"""
        model.eval()
        
        if self.model_type == 'mean':
            # No attention weights available
            return None
            
        with torch.no_grad():
            if self.model_type == 'attention':
                # AttentionMILClassifier
                if hasattr(model, 'get_attention_weights'):
                    attention_weights = model.get_attention_weights(features)
                else:
                    # Manual extraction if method not available
                    attention_scores = model.attention(features)
                    attention_weights = torch.softmax(attention_scores, dim=0)
                
            elif self.model_type == 'clam':
                # CLAMClassifier
                if hasattr(model, 'get_attention_weights'):
                    attention_weights = model.get_attention_weights(features)
                else:
                    # Use forward with return_attention
                    _, attention_weights = model(features, return_attention=True)
                
            elif self.model_type == 'acmil':
                # ACMILClassifier - get averaged attention across branches
                if hasattr(model, 'get_attention_weights'):
                    attention_weights = model.get_attention_weights(features)
                else:
                    # Manual extraction would be complex, rely on method
                    return None
            else:
                return None
        
        # Convert to numpy and ensure proper shape
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
        
        # 1. Distribution metrics
        metrics['entropy'] = float(entropy(attention_weights + 1e-8))
        metrics['normalized_entropy'] = metrics['entropy'] / np.log(len(attention_weights))
        metrics['gini'] = float(self.gini_coefficient(attention_weights))
        
        # 2. Concentration metrics
        metrics['max_attention'] = float(np.max(attention_weights))
        metrics['top_5_mass'] = float(np.sum(np.sort(attention_weights)[-5:]))
        metrics['top_10_mass'] = float(np.sum(np.sort(attention_weights)[-10:]))
        metrics['top_20_mass'] = float(np.sum(np.sort(attention_weights)[-20:]))
        
        # 3. Effective size
        metrics['effective_size'] = float(np.exp(metrics['entropy']))
        metrics['effective_size_ratio'] = metrics['effective_size'] / len(attention_weights)
        
        # 4. Statistics
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
        threshold = np.percentile(attention_weights, threshold_percentile)
        high_attention_mask = attention_weights > threshold
        
        if np.sum(high_attention_mask) == 0:
            return 0.0, 0
        
        # Create spatial grid from coordinates
        coords = np.array(coordinates)
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        
        # Map to grid (assuming patch size of 224)
        grid_size = 224
        grid_w = (max_x - min_x) // grid_size + 1
        grid_h = (max_y - min_y) // grid_size + 1
        
        if grid_w <= 0 or grid_h <= 0:
            return 0.0, 0
        
        spatial_grid = np.zeros((int(grid_h), int(grid_w)))
        
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
    
    def identify_top_k_patches(self, attention_weights: np.ndarray, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Get indices and weights of top-k attended patches"""
        k = min(k, len(attention_weights))
        top_k_indices = np.argsort(attention_weights)[-k:][::-1]
        top_k_weights = attention_weights[top_k_indices]
        
        return top_k_indices, top_k_weights
    
    def generate_heatmap(self, attention_weights: np.ndarray, 
                        coordinates: np.ndarray, 
                        patch_size: int = 224) -> np.ndarray:
        """Generate spatial attention heatmap"""
        coords = np.array(coordinates)
        
        # Estimate WSI dimensions
        max_x = coords[:, 0].max() + patch_size
        max_y = coords[:, 1].max() + patch_size
        
        # Create heatmap at reduced resolution for efficiency
        scale_factor = 4
        heatmap_h = max_y // scale_factor
        heatmap_w = max_x // scale_factor
        heatmap = np.zeros((heatmap_h, heatmap_w))
        
        # Map attention to spatial locations
        for weight, (x, y) in zip(attention_weights, coordinates):
            x_scaled = x // scale_factor
            y_scaled = y // scale_factor
            patch_size_scaled = patch_size // scale_factor
            
            y_end = min(y_scaled + patch_size_scaled, heatmap_h)
            x_end = min(x_scaled + patch_size_scaled, heatmap_w)
            
            heatmap[y_scaled:y_end, x_scaled:x_end] = np.maximum(
            heatmap[y_scaled:y_end, x_scaled:x_end], 
            weight
            )
        
        # Smooth for better visualization
        heatmap = gaussian_filter(heatmap, sigma=2)
        
        return heatmap
    def visualize_attention(self, 
                        attention_weights: np.ndarray,
                        coordinates: np.ndarray,
                        slide_id: str,
                        true_label: int,
                        pred_label: int,
                        pred_prob: float,
                        save: bool = True,
                        save_pdf: bool = True) -> Optional[Dict[str, str]]:
        """Create comprehensive attention visualization with separate PDF exports"""
        
        if attention_weights is None:
            print(f"No attention weights available for {self.model_type}")
            return None
        
        saved_paths = {}
        
        # Main figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Attention heatmap
        heatmap = self.generate_heatmap(attention_weights, coordinates)
        im = axes[0, 0].imshow(heatmap, cmap='jet', interpolation='bilinear', aspect='auto')
        axes[0, 0].set_title(f'Attention Heatmap - {slide_id}')
        axes[0, 0].axis('off')
        plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
        
        # 2. Attention distribution
        axes[0, 1].hist(attention_weights, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 1].set_xlabel('Attention Weight')
        axes[0, 1].set_ylabel('Number of Patches')
        axes[0, 1].set_title(f'Attention Distribution\nEntropy: {entropy(attention_weights):.3f}')
        axes[0, 1].axvline(np.mean(attention_weights), color='red', 
                        linestyle='--', label=f'Mean: {np.mean(attention_weights):.4f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Top-20 patches bar plot
        top_k_indices, top_k_weights = self.identify_top_k_patches(attention_weights, k=20)
        axes[1, 0].bar(range(len(top_k_weights)), top_k_weights, color='coral')
        axes[1, 0].set_xlabel('Patch Rank')
        axes[1, 0].set_ylabel('Attention Weight')
        axes[1, 0].set_title(f'Top-{len(top_k_weights)} Patches\nCumulative: {np.sum(top_k_weights):.3f}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Spatial scatter plot with gradient from blue to red
        axes[1, 1].set_facecolor('white')
        
        # Create custom colormap from light blue to dark red
        import matplotlib.colors as mcolors
        colors = ['#ADD8E6', '#87CEEB', '#FFA07A', '#FA8072', '#DC143C', '#8B0000']
        n_bins = 100
        cmap = mcolors.LinearSegmentedColormap.from_list('blue_to_red', colors, N=n_bins)
        
        scatter = axes[1, 1].scatter(coordinates[:, 0], coordinates[:, 1], 
                                    c=attention_weights, cmap=cmap,
                                    s=10, alpha=0.8, edgecolors='none',
                                    vmin=attention_weights.min(), 
                                    vmax=attention_weights.max())
        
        axes[1, 1].set_xlabel('X coordinate')
        axes[1, 1].set_ylabel('Y coordinate')
        axes[1, 1].set_title('Spatial Distribution of Attention')
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(True, alpha=0.2, color='gray')
        plt.colorbar(scatter, ax=axes[1, 1], fraction=0.046)
        
        # Add prediction info
        risk_label = ['Low Risk', 'High Risk']
        plt.suptitle(
            f'{self.model_type.upper()} Attention Analysis\n'
            f'True: {risk_label[true_label]}, Pred: {risk_label[pred_label]} (prob: {pred_prob:.3f})',
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        
        if save:
            # Save complete figure
            png_path = os.path.join(self.save_dir, f'{slide_id}_attention.png')
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            saved_paths['png'] = png_path
            
            if save_pdf:
                pdf_path = os.path.join(self.save_dir, f'{slide_id}_attention.pdf')
                plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
                saved_paths['pdf'] = pdf_path
        
        plt.close()
        
        # Save individual figures if save_pdf is True
        if save and save_pdf:
            # Save attention heatmap separately
            fig_heatmap = plt.figure(figsize=(8, 6))
            heatmap = self.generate_heatmap(attention_weights, coordinates)
            im = plt.imshow(heatmap, cmap='jet', interpolation='bilinear', aspect='auto')
            plt.title(f'Attention Heatmap - {slide_id}\n'
                    f'True: {risk_label[true_label]}, Pred: {risk_label[pred_label]} (prob: {pred_prob:.3f})',
                    fontsize=12)
            plt.axis('off')
            plt.colorbar(im, fraction=0.046)
            
            heatmap_pdf_path = os.path.join(self.save_dir, f'{slide_id}_heatmap.pdf')
            plt.savefig(heatmap_pdf_path, format='pdf', bbox_inches='tight', dpi=300)
            saved_paths['heatmap_pdf'] = heatmap_pdf_path
            plt.close()
            
            # Save spatial distribution separately with clean gradient
            fig_spatial = plt.figure(figsize=(10, 8))
            ax = plt.gca()
            ax.set_facecolor('white')
            
            # Simple scatter with gradient colormap
            scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1],
                            c=attention_weights, cmap=cmap,
                            s=12, alpha=0.85, edgecolors='none',
                            vmin=attention_weights.min(),
                            vmax=attention_weights.max())
            
            ax.set_xlabel('X coordinate', fontsize=12)
            ax.set_ylabel('Y coordinate', fontsize=12)
            ax.set_title(f'Spatial Distribution of Attention - {slide_id}\n'
                        f'True: {risk_label[true_label]}, Pred: {risk_label[pred_label]} (prob: {pred_prob:.3f})',
                        fontsize=14)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, color='gray', linestyle='--')
            
            # Add colorbar with label
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Attention Weight', rotation=270, labelpad=20)
            
            spatial_pdf_path = os.path.join(self.save_dir, f'{slide_id}_spatial.pdf')
            plt.savefig(spatial_pdf_path, format='pdf', bbox_inches='tight', dpi=300)
            saved_paths['spatial_pdf'] = spatial_pdf_path
            plt.close()
        
        if not save:
            plt.show()
        
        return saved_paths
    def analyze_slide(self, 
                    model,
                    features: torch.Tensor,
                    coordinates: np.ndarray,
                    slide_id: str,
                    true_label: int,
                    pred_label: int,
                    pred_prob: float,
                    visualize: bool = False,
                    save_pdf: bool = True) -> Dict:
        """Complete analysis for a single slide
        
        Args:
            save_pdf: If True, save visualizations as both PNG and PDF
        """
        
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
        
        # Add slide metadata
        metrics['slide_id'] = slide_id
        metrics['true_label'] = true_label
        metrics['pred_label'] = pred_label
        metrics['pred_prob'] = pred_prob
        metrics['correct'] = true_label == pred_label
        
        # Visualize if requested
        if visualize:
            viz_paths = self.visualize_attention(
                attention_weights, coordinates, slide_id,
                true_label, pred_label, pred_prob, 
                save=True, 
                save_pdf=save_pdf  # Pass the save_pdf parameter
            )
            if viz_paths:
                metrics['visualization_png'] = viz_paths.get('png')
                metrics['visualization_pdf'] = viz_paths.get('pdf')
        
        return metrics