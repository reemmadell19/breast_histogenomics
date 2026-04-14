# models/regression_model.py - Fixed CLAM implementation
import torch
import torch.nn as nn

class MeanPoolingMIL(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, **kwargs):
        """
        Mean pooling MIL model for regression
        Args:
            input_dim: Feature dimension from feature extractor
            hidden_dim: Hidden layer dimension
            **kwargs: Accept additional arguments for compatibility
        """
        super(MeanPoolingMIL, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(hidden_dim, 1)  # Single output for regression
        )
    
    def forward(self, x):
        """
        x: Tensor of shape [N, D] (N patches per slide)
        Returns: slide-level regression output (scalar)
        """
        pooled = x.mean(dim=0)  # mean pool over patches → [D]
        output = self.regressor(pooled)  # → [1]
        return output.squeeze()  # Return scalar


class MaxPoolingMIL(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, **kwargs):
        """
        Max pooling MIL model for regression
        Args:
            input_dim: Feature dimension from feature extractor
            hidden_dim: Hidden layer dimension
            **kwargs: Accept additional arguments for compatibility
        """
        super(MaxPoolingMIL, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(hidden_dim, 1)  # Single output for regression
        )
    
    def forward(self, x):
        """
        x: Tensor of shape [N, D] (N patches per slide)
        Returns: slide-level regression output (scalar)
        """
        pooled, _ = x.max(dim=0)  # max pool over patches → [D]
        output = self.regressor(pooled)  # → [1]
        return output.squeeze()  # Return scalar


class AttentionMIL(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, attention_hidden_dim=64, **kwargs):
        """
        Simple Attention-based MIL model for regression
        Args:
            input_dim: Feature dimension from feature extractor
            hidden_dim: Hidden layer dimension for regression head
            attention_hidden_dim: Hidden dimension for attention mechanism
            **kwargs: Accept additional arguments for compatibility
        """
        super(AttentionMIL, self).__init__()
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1)
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """
        x: Tensor of shape [N, D] (N patches per slide)
        Returns: slide-level regression output (scalar)
        """
        # Compute attention weights
        attention_scores = self.attention(x)  # [N, 1]
        attention_weights = torch.softmax(attention_scores, dim=0)  # [N, 1]
        
        # Weighted aggregation
        pooled = torch.sum(attention_weights * x, dim=0)  # [D]
        
        # Regression prediction
        output = self.regressor(pooled)  # [1]
        return output.squeeze()  # Return scalar
    
    def get_attention_weights(self, x):
        """
        Get attention weights for interpretability
        Args:
            x: Tensor of shape [N, D] (N patches per slide)
        Returns:
            attention_weights: Tensor of shape [N, 1]
        """
        with torch.no_grad():
            attention_scores = self.attention(x)  # [N, 1]
            attention_weights = torch.softmax(attention_scores, dim=0)  # [N, 1]
        return attention_weights


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        """Gated Attention Network from original CLAM"""
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class CLAM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, attention_hidden_dim=256, 
                 dropout=0.25, gate=True, **kwargs):
        """
        CLAM adapted for regression from original CLAM-SB implementation
        
        Args:
            input_dim: Feature dimension from feature extractor (embed_dim in original)
            hidden_dim: Hidden layer dimension (size[1] in original)
            attention_hidden_dim: Attention network dimension (size[2] in original)  
            dropout: Dropout rate
            gate: Whether to use gated attention
            **kwargs: Accept additional arguments for compatibility
        """
        super(CLAM, self).__init__()
        
        self.gate = gate
        
        # Feature transformation layers (separate, not in Sequential)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Attention network
        if gate:
            self.attention_net = Attn_Net_Gated(L=hidden_dim, D=attention_hidden_dim, 
                                               dropout=dropout, n_classes=1)
        else:
            # Simple attention network  
            self.attention_net = nn.Sequential(
                nn.Linear(hidden_dim, attention_hidden_dim),
                nn.Tanh(),
                nn.Linear(attention_hidden_dim, 1)
            )
        
        # Regression head (instead of classifier for multiple classes)
        self.regressor = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass adapted for regression
        Args:
            x: Tensor of shape [N, D] (N patches per slide)
        Returns:
            slide-level regression output (scalar)
        """
        # Feature transformation
        h = self.fc1(x)  # [N, hidden_dim]
        h = self.relu(h)
        h = self.dropout(h)
        
        # Get attention scores
        if self.gate:
            # Gated attention returns both A and the input features
            A, _ = self.attention_net(h)  # A: [N, 1], we ignore the second output
        else:
            # Simple attention just returns scores
            A = self.attention_net(h)  # A: [N, 1]
        
        # Transpose and softmax attention (following original CLAM)
        A = torch.transpose(A, 1, 0)  # [1, N] 
        A = torch.softmax(A, dim=1)  # softmax over N patches
        
        # Weighted aggregation (matrix multiplication like original)
        M = torch.mm(A, h)  # [1, hidden_dim]
        
        # Regression prediction
        output = self.regressor(M)  # [1, 1]
        
        return output.squeeze()  # Return scalar
    
    def get_attention_weights(self, x):
        """
        Get attention weights for interpretability
        Args:
            x: Tensor of shape [N, D] (N patches per slide)
        Returns:
            attention_weights: Tensor of shape [N, 1]
        """
        with torch.no_grad():
            # Feature transformation
            h = self.fc1(x)  # [N, hidden_dim]
            h = self.relu(h)
            # Note: No dropout during inference
            
            # Get attention scores
            if self.gate:
                A, _ = self.attention_net(h)  # A: [N, 1]
            else:
                A = self.attention_net(h)  # A: [N, 1]
            
            # Transpose and softmax
            A = torch.transpose(A, 1, 0)  # [1, N]
            attention_weights = torch.softmax(A, dim=1)  # [1, N]
            
        return attention_weights.transpose(1, 0)  # Return [N, 1] for consistency
    
    # models/regression_model.py - Add TransMIL to existing models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Keep existing models (MeanPoolingMIL, MaxPoolingMIL, AttentionMIL, CLAM)...
# Adding TransMIL below:

class PPEG(nn.Module):
    """Pyramid Position Encoding Generator for TransMIL"""
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W):
        """
        x: [B, N, C] where N = H*W + 1 (includes cls_token)
        H, W: spatial dimensions
        """
        B, N, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        
        # Apply pyramid convolutions
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class NystromAttention(nn.Module):
    """Nystrom Attention for efficient self-attention in TransMIL"""
    def __init__(self, dim, num_heads=8, num_landmarks=256, kernel_size=0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_landmarks = num_landmarks
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # For simplicity, using basic scaling
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        
        # Standard attention (simplified Nystrom for stability)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class TransformerLayer(nn.Module):
    """Transformer layer for TransMIL"""
    def __init__(self, dim=512, num_heads=8, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NystromAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransMIL(nn.Module):
    """
    TransMIL adapted for regression following the pattern of your other models
    
    Args:
        input_dim: Feature dimension from feature extractor (512 for ResNet-18, varies for others)
        hidden_dim: Hidden layer dimension (not used in TransMIL but kept for consistency)
        proj_dim: Projection dimension for input features (default 512)
        num_heads: Number of attention heads
        num_layers: Number of transformer layers (2 in original)
        pos_enc: Type of positional encoding ('PPEG' or 'none')
        dropout: Dropout rate
    """
    def __init__(self, input_dim=512, hidden_dim=128, proj_dim=512, 
                 num_heads=8, num_layers=2, pos_enc='PPEG', dropout=0.1, **kwargs):
        super(TransMIL, self).__init__()
        
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.num_layers = num_layers
        self.pos_enc = pos_enc
        
        # Project input features to TransMIL dimension if needed
        if input_dim != proj_dim:
            self._fc1 = nn.Sequential(
                nn.Linear(input_dim, proj_dim),
                nn.ReLU()
            )
        else:
            self._fc1 = nn.Identity()
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, proj_dim))
        
        # Positional encoding
        if pos_enc == 'PPEG':
            self.pos_layer = PPEG(dim=proj_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(dim=proj_dim, num_heads=num_heads, drop=dropout)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(proj_dim)
        
        # Regression head (single output)
        self._fc2 = nn.Linear(proj_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass for regression
        Args:
            x: Tensor of shape [N, D] (N patches per slide)
        Returns:
            slide-level regression output (scalar)
        """
        # Handle batch dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, N, D]
        
        B = x.shape[0]
        
        # Project features if needed
        h = self._fc1(x)  # [B, N, proj_dim]
        
        # Padding for PPEG
        H_orig = h.shape[1]
        if self.pos_enc == 'PPEG':
            # Find smallest square that fits all patches
            _H = _W = int(np.ceil(np.sqrt(H_orig)))
            add_length = _H * _W - H_orig
            
            # Pad with repeated features if needed
            if add_length > 0:
                h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, H*W, proj_dim]
        else:
            _H = _W = 1  # Not used if no PPEG
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)  # [B, 1+H*W, proj_dim]
        
        # Apply transformer layers with PPEG in between
        for i, layer in enumerate(self.layers):
            h = layer(h)
            
            # Apply PPEG after first layer (following original TransMIL)
            if i == 0 and self.pos_enc == 'PPEG':
                h = self.pos_layer(h, _H, _W)
        
        # Extract class token
        h = self.norm(h)[:, 0]  # [B, proj_dim]
        
        # Regression prediction
        output = self._fc2(h)  # [B, 1]
        
        # Return scalar if batch size is 1
        if B == 1:
            return output.squeeze()
        return output.squeeze(-1)  # [B]
    
    def get_attention_weights(self, x):
        """
        Get attention weights for interpretability
        This returns the attention from the last transformer layer
        
        Args:
            x: Tensor of shape [N, D] (N patches per slide)
        Returns:
            attention_weights: Attention maps from last layer
        """
        with torch.no_grad():
            # Handle batch dimension
            if x.dim() == 2:
                x = x.unsqueeze(0)
            
            B = x.shape[0]
            
            # Project features
            h = self._fc1(x)
            
            # Padding for PPEG
            H_orig = h.shape[1]
            if self.pos_enc == 'PPEG':
                _H = _W = int(np.ceil(np.sqrt(H_orig)))
                add_length = _H * _W - H_orig
                if add_length > 0:
                    h = torch.cat([h, h[:, :add_length, :]], dim=1)
            
            # Add class token
            cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
            h = torch.cat((cls_tokens, h), dim=1)
            
            # Store attention weights
            attention_maps = []
            
            # Apply transformer layers
            for i, layer in enumerate(self.layers):
                # We need to modify to extract attention
                # For now, return uniform weights as placeholder
                h = layer(h)
                
                if i == 0 and self.pos_enc == 'PPEG':
                    h = self.pos_layer(h, _H, _W)
            
            # Return uniform attention as placeholder
            # In practice, you'd modify TransformerLayer to return attention
            N = H_orig
            attention_weights = torch.ones(N, 1) / N
            
        return attention_weights.to(x.device)
    
#########################################

# models/regression_model.py - Updated with ACMIL implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Keep all existing models (MeanPoolingMIL, MaxPoolingMIL, AttentionMIL, CLAM, TransMIL)...
# [Previous models remain unchanged]

# ============= ACMIL Implementation =============

class MultibranchAttention(nn.Module):
    """Multiple Branch Attention module for ACMIL"""
    def __init__(self, input_dim=512, hidden_dim=128, n_branches=5, dropout=0.25):
        super(MultibranchAttention, self).__init__()
        self.n_branches = n_branches
        
        # Create multiple attention branches
        self.attention_branches = nn.ModuleList()
        for _ in range(n_branches):
            branch = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            self.attention_branches.append(branch)
    
    def forward(self, x):
        """
        Args:
            x: [N, D] where N is number of patches
        Returns:
            attention_weights_list: List of attention weights from each branch
        """
        attention_weights_list = []
        
        for branch in self.attention_branches:
            # Calculate attention for this branch
            A = branch(x)  # [N, 1]
            A = torch.softmax(A, dim=0)  # Normalize
            attention_weights_list.append(A)
        
        return attention_weights_list


def stochastic_topk_masking(attention_weights, k=10, mask_ratio=0.6, training=True):
    """
    Stochastic Top-K Instance Masking (STKIM)
    
    Args:
        attention_weights: [N, 1] attention weights
        k: number of top instances to consider for masking
        mask_ratio: proportion of top-k instances to mask
        training: whether in training mode
    """
    if not training or mask_ratio == 0.0:
        return attention_weights
    
    N = attention_weights.shape[0]
    k = min(k, N)  # Ensure k doesn't exceed number of patches
    
    # Get top-k indices
    topk_values, topk_indices = torch.topk(attention_weights.squeeze(), k)
    
    # Randomly select instances to mask from top-k
    n_mask = int(k * mask_ratio)
    if n_mask > 0:
        # Random selection of which top-k instances to mask
        mask_indices = torch.randperm(k, device=attention_weights.device)[:n_mask]
        masked_topk_indices = topk_indices[mask_indices]
        
        # Create mask (1 for keep, 0 for mask)
        mask = torch.ones_like(attention_weights)
        mask[masked_topk_indices] = 0
        
        # Apply mask and renormalize
        masked_attention = attention_weights * mask
        # Renormalize to sum to 1
        masked_attention = masked_attention / (masked_attention.sum() + 1e-10)
        
        return masked_attention
    
    return attention_weights


class ACMIL(nn.Module):
    """
    Attention-Challenging Multiple Instance Learning for regression
    
    Args:
        input_dim: Feature dimension from feature extractor (512 for ResNet-18, etc.)
        hidden_dim: Hidden layer dimension for attention and regression
        n_branches: Number of attention branches (default: 5)
        n_masked_patch: Top-k parameter for STKIM (default: 10)
        mask_ratio: Proportion of top-k to mask (default: 0.6)
        dropout: Dropout rate
    """
    def __init__(self, input_dim=512, hidden_dim=128, n_branches=5, 
                 n_masked_patch=10, mask_ratio=0.6, dropout=0.25, **kwargs):
        super(ACMIL, self).__init__()
        
        self.n_branches = n_branches
        self.n_masked_patch = n_masked_patch
        self.mask_ratio = mask_ratio
        
        # Multiple Branch Attention (MBA)
        self.mba = MultibranchAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_branches=n_branches,
            dropout=dropout
        )
        
        # Branch-specific regressors
        self.branch_regressors = nn.ModuleList()
        for _ in range(n_branches):
            regressor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)  # Single output for RS score
            )
            self.branch_regressors.append(regressor)
        
        # Optional: Final aggregation layer
        self.final_aggregation = nn.Linear(n_branches, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_branch_outputs=False):
        """
        Forward pass for ACMIL regression
        
        Args:
            x: Tensor of shape [N, D] (N patches per slide)
            return_branch_outputs: If True, return individual branch predictions
        Returns:
            slide-level regression output (scalar)
            Optionally: (output, branch_predictions, attention_weights)
        """
        # Get attention weights from all branches
        attention_weights_list = self.mba(x)  # List of [N, 1] tensors
        
        branch_predictions = []
        branch_features = []
        
        for branch_idx in range(self.n_branches):
            # Get attention for this branch
            attention = attention_weights_list[branch_idx]
            
            # Apply STKIM during training
            if self.training:
                attention = stochastic_topk_masking(
                    attention, 
                    k=self.n_masked_patch, 
                    mask_ratio=self.mask_ratio, 
                    training=self.training
                )
            
            # Weighted feature aggregation
            weighted_features = torch.sum(attention * x, dim=0, keepdim=True)  # [1, D]
            branch_features.append(weighted_features)
            
            # Branch-specific regression
            branch_pred = self.branch_regressors[branch_idx](weighted_features)  # [1, 1]
            branch_predictions.append(branch_pred.squeeze())
        
        # Aggregate branch predictions
        branch_preds_tensor = torch.stack(branch_predictions)  # [n_branches]
        
        # Two options for final prediction:
        # Option 1: Simple average (as in original ACMIL)
        final_prediction = torch.mean(branch_preds_tensor)
        
        # Option 2: Learned weighted aggregation (optional)
        # final_prediction = self.final_aggregation(branch_preds_tensor.unsqueeze(0)).squeeze()
        
        if return_branch_outputs:
            return final_prediction, branch_predictions, attention_weights_list
        
        return final_prediction
    
    def get_attention_weights(self, x):
        """
        Get attention weights for interpretability
        Returns averaged attention across all branches
        
        Args:
            x: Tensor of shape [N, D] (N patches per slide)
        Returns:
            attention_weights: Averaged attention weights [N, 1]
            branch_attentions: Individual branch attention weights
        """
        with torch.no_grad():
            # Get attention weights from all branches
            attention_weights_list = self.mba(x)
            
            # Average across branches for main visualization
            avg_attention = torch.mean(torch.cat(attention_weights_list, dim=1), dim=1, keepdim=True)
            
        return avg_attention, attention_weights_list


class ACMIL_CLAM_Hybrid(nn.Module):
    """
    ACMIL with CLAM-style gated attention option
    Combines the multi-branch approach of ACMIL with CLAM's gated attention
    """
    def __init__(self, input_dim=512, hidden_dim=128, attention_hidden_dim=256,
                 n_branches=5, n_masked_patch=10, mask_ratio=0.6, 
                 dropout=0.25, gate=True, **kwargs):
        super(ACMIL_CLAM_Hybrid, self).__init__()
        
        self.n_branches = n_branches
        self.n_masked_patch = n_masked_patch
        self.mask_ratio = mask_ratio
        self.gate = gate
        
        # Feature transformation (shared across branches)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Multiple gated attention branches
        self.attention_branches = nn.ModuleList()
        for _ in range(n_branches):
            if gate:
                attn = Attn_Net_Gated(L=hidden_dim, D=attention_hidden_dim, 
                                    dropout=dropout, n_classes=1)
            else:
                attn = nn.Sequential(
                    nn.Linear(hidden_dim, attention_hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(attention_hidden_dim, 1)
                )
            self.attention_branches.append(attn)
        
        # Branch-specific regressors
        self.branch_regressors = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_branches)
        ])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_branch_outputs=False):
        """
        Forward pass combining ACMIL and CLAM approaches
        """
        # Feature transformation (shared)
        h = self.fc1(x)  # [N, hidden_dim]
        h = self.relu(h)
        h = self.dropout(h)
        
        branch_predictions = []
        attention_weights_list = []
        
        for branch_idx in range(self.n_branches):
            # Get attention scores for this branch
            if self.gate:
                A, _ = self.attention_branches[branch_idx](h)
            else:
                A = self.attention_branches[branch_idx](h)
            
            # Process attention (CLAM style)
            A = torch.transpose(A, 1, 0)  # [1, N]
            A = torch.softmax(A, dim=1)
            
            # Apply STKIM if enabled
            if self.training and self.mask_ratio > 0:
                A_t = A.transpose(1, 0)  # Back to [N, 1] for masking
                A_t = stochastic_topk_masking(
                    A_t, k=self.n_masked_patch, 
                    mask_ratio=self.mask_ratio, 
                    training=self.training
                )
                A = A_t.transpose(1, 0)  # Back to [1, N]
            
            attention_weights_list.append(A.transpose(1, 0))  # Store as [N, 1]
            
            # Weighted aggregation
            M = torch.mm(A, h)  # [1, hidden_dim]
            
            # Branch prediction
            branch_pred = self.branch_regressors[branch_idx](M).squeeze()
            branch_predictions.append(branch_pred)
        
        # Aggregate predictions
        final_prediction = torch.mean(torch.stack(branch_predictions))
        
        if return_branch_outputs:
            return final_prediction, branch_predictions, attention_weights_list
        
        return final_prediction
    
    def get_attention_weights(self, x):
        """Get attention weights for interpretability"""
        with torch.no_grad():
            h = self.fc1(x)
            h = self.relu(h)
            
            attention_weights_list = []
            
            for branch_idx in range(self.n_branches):
                if self.gate:
                    A, _ = self.attention_branches[branch_idx](h)
                else:
                    A = self.attention_branches[branch_idx](h)
                
                A = torch.transpose(A, 1, 0)
                A = torch.softmax(A, dim=1)
                attention_weights_list.append(A.transpose(1, 0))
            
            # Average across branches
            avg_attention = torch.mean(torch.cat(attention_weights_list, dim=1), dim=1, keepdim=True)
            
        return avg_attention, attention_weights_list


# ============= Model Factory Function =============

def get_regression_model(model_name, **kwargs):
    """
    Factory function to get regression models
    
    Args:
        model_name: Name of the model ('mean', 'max', 'attention', 'clam', 'transmil', 'acmil', 'acmil_clam')
        **kwargs: Model-specific arguments
    
    Returns:
        Model instance
    """
    models = {
        'mean': MeanPoolingMIL,
        'max': MaxPoolingMIL,
        'attention': AttentionMIL,
        'clam': CLAM,
        'transmil': TransMIL,
        'acmil': ACMIL,
        'acmil_clam': ACMIL_CLAM_Hybrid
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    
    return models[model_name](**kwargs)