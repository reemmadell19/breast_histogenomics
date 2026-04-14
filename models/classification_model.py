
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CLAMClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, attention_hidden_dim=256, 
                 n_classes=2, dropout=0.25, gate=True, **kwargs):
        """
        CLAM adapted for classification
        """
        super(CLAMClassifier, self).__init__()
        
        self.gate = gate
        self.n_classes = n_classes
        
        # Feature transformation layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Attention network
        if gate:
            self.attention_net = Attn_Net_Gated(L=hidden_dim, D=attention_hidden_dim, 
                                               dropout=dropout, n_classes=1)
        else:
            self.attention_net = nn.Sequential(
                nn.Linear(hidden_dim, attention_hidden_dim),
                nn.Tanh(),
                nn.Linear(attention_hidden_dim, 1)
            )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, n_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass for classification
        Args:
            x: Tensor of shape [N, D] (N patches per slide)
        Returns:
            logits of shape [n_classes]
        """
        # Feature transformation
        h = self.fc1(x)  # [N, hidden_dim]
        h = self.relu(h)
        h = self.dropout(h)
        
        # Get attention scores
        if self.gate:
            A, _ = self.attention_net(h)  # A: [N, 1]
        else:
            A = self.attention_net(h)  # A: [N, 1]
        
        # Transpose and softmax attention
        A = torch.transpose(A, 1, 0)  # [1, N] 
        A = torch.softmax(A, dim=1)  # softmax over N patches
        
        # Weighted aggregation
        M = torch.mm(A, h)  # [1, hidden_dim]
        
        # Classification
        logits = self.classifier(M)  # [1, n_classes]
        logits = logits.squeeze(0)  # [n_classes]
        
        if return_attention:
            return logits, A.transpose(1, 0)
        return logits
    
    def get_attention_weights(self, x):
        """Get attention weights for interpretability"""
        with torch.no_grad():
            h = self.fc1(x)
            h = self.relu(h)
            
            if self.gate:
                A, _ = self.attention_net(h)
            else:
                A = self.attention_net(h)
            
            A = torch.transpose(A, 1, 0)
            attention_weights = torch.softmax(A, dim=1)
            
        return attention_weights.transpose(1, 0)
    




    ######################




import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MeanPoolingMILClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, n_classes=2, dropout=0.1, **kwargs):
        """
        Mean pooling MIL model for classification
        Args:
            input_dim: Feature dimension from feature extractor
            hidden_dim: Hidden layer dimension
            n_classes: Number of output classes (2 for binary)
            dropout: Dropout rate
        """
        super(MeanPoolingMILClassifier, self).__init__()
        self.n_classes = n_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x, return_features=False):
        """
        x: Tensor of shape [N, D] (N patches per slide)
        Returns: logits of shape [n_classes] for binary classification
        """
        pooled = x.mean(dim=0)  # mean pool over patches → [D]
        logits = self.classifier(pooled)  # → [n_classes]
        
        if return_features:
            return logits, pooled
        return logits


class MaxPoolingMILClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, n_classes=2, dropout=0.1, **kwargs):
        """
        Max pooling MIL model for classification
        """
        super(MaxPoolingMILClassifier, self).__init__()
        self.n_classes = n_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x, return_features=False):
        """
        x: Tensor of shape [N, D] (N patches per slide)
        Returns: logits of shape [n_classes]
        """
        pooled, _ = x.max(dim=0)  # max pool over patches → [D]
        logits = self.classifier(pooled)  # → [n_classes]
        
        if return_features:
            return logits, pooled
        return logits


class AttentionMILClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, attention_hidden_dim=64, 
                 n_classes=2, dropout=0.1, **kwargs):
        """
        Simple Attention-based MIL model for classification
        """
        super(AttentionMILClassifier, self).__init__()
        self.n_classes = n_classes
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x, return_attention=False):
        """
        x: Tensor of shape [N, D] (N patches per slide)
        Returns: logits of shape [n_classes]
        """
        # Compute attention weights
        attention_scores = self.attention(x)  # [N, 1]
        attention_weights = torch.softmax(attention_scores, dim=0)  # [N, 1]
        
        # Weighted aggregation
        pooled = torch.sum(attention_weights * x, dim=0)  # [D]
        
        # Classification
        logits = self.classifier(pooled)  # [n_classes]
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def get_attention_weights(self, x):
        """Get attention weights for interpretability"""
        with torch.no_grad():
            attention_scores = self.attention(x)
            attention_weights = torch.softmax(attention_scores, dim=0)
        return attention_weights


# Reuse Attn_Net_Gated from regression model
class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        """Gated Attention Network from original CLAM"""
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
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


# ============= ACMIL Classification Models =============

class MultibranchAttentionClassifier(nn.Module):
    """Multiple Branch Attention module for ACMIL Classifier"""
    def __init__(self, input_dim=512, hidden_dim=128, n_branches=5, dropout=0.25):
        super(MultibranchAttentionClassifier, self).__init__()
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
            A = branch(x)  # [N, 1]
            A = torch.softmax(A, dim=0)  # Normalize
            attention_weights_list.append(A)
        
        return attention_weights_list


def stochastic_topk_masking(attention_weights, k=10, mask_ratio=0.6, training=True):
    """
    Stochastic Top-K Instance Masking (STKIM) for ACMIL
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
        mask_indices = torch.randperm(k, device=attention_weights.device)[:n_mask]
        masked_topk_indices = topk_indices[mask_indices]
        
        # Create mask (1 for keep, 0 for mask)
        mask = torch.ones_like(attention_weights)
        mask[masked_topk_indices] = 0
        
        # Apply mask and renormalize
        masked_attention = attention_weights * mask
        masked_attention = masked_attention / (masked_attention.sum() + 1e-10)
        
        return masked_attention
    
    return attention_weights


class ACMILClassifier(nn.Module):
    """
    Attention-Challenging Multiple Instance Learning for classification
    """
    def __init__(self, input_dim=512, hidden_dim=128, n_branches=5, 
                 n_masked_patch=10, mask_ratio=0.6, n_classes=2, 
                 dropout=0.25, **kwargs):
        super(ACMILClassifier, self).__init__()
        
        self.n_branches = n_branches
        self.n_masked_patch = n_masked_patch
        self.mask_ratio = mask_ratio
        self.n_classes = n_classes
        
        # Multiple Branch Attention (MBA)
        self.mba = MultibranchAttentionClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_branches=n_branches,
            dropout=dropout
        )
        
        # Branch-specific classifiers
        self.branch_classifiers = nn.ModuleList()
        for _ in range(n_branches):
            classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes)  # Output logits for classification
            )
            self.branch_classifiers.append(classifier)
        
        # Optional: Final aggregation layer
        self.final_aggregation = nn.Linear(n_branches * n_classes, n_classes)
        
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
        Forward pass for ACMIL classification
        
        Args:
            x: Tensor of shape [N, D] (N patches per slide)
            return_branch_outputs: If True, return individual branch predictions
        Returns:
            logits of shape [n_classes]
        """
        # Get attention weights from all branches
        attention_weights_list = self.mba(x)  # List of [N, 1] tensors
        
        branch_logits = []
        
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
            
            # Branch-specific classification
            branch_logit = self.branch_classifiers[branch_idx](weighted_features)  # [1, n_classes]
            branch_logits.append(branch_logit.squeeze(0))
        
        # Aggregate branch predictions
        branch_logits_tensor = torch.stack(branch_logits)  # [n_branches, n_classes]
        
        # Average logits across branches
        final_logits = torch.mean(branch_logits_tensor, dim=0)  # [n_classes]
        
        if return_branch_outputs:
            return final_logits, branch_logits, attention_weights_list
        
        return final_logits
    
    def get_attention_weights(self, x):
        """Get attention weights for interpretability"""
        with torch.no_grad():
            attention_weights_list = self.mba(x)
            # Average across branches for main visualization
            avg_attention = torch.mean(torch.cat(attention_weights_list, dim=1), dim=1, keepdim=True)
        return avg_attention, attention_weights_list


class ACMIL_CLAM_HybridClassifier(nn.Module):
    """
    ACMIL with CLAM-style gated attention for classification
    """
    def __init__(self, input_dim=512, hidden_dim=128, attention_hidden_dim=256,
                 n_branches=5, n_masked_patch=10, mask_ratio=0.6, 
                 n_classes=2, dropout=0.25, gate=True, **kwargs):
        super(ACMIL_CLAM_HybridClassifier, self).__init__()
        
        self.n_branches = n_branches
        self.n_masked_patch = n_masked_patch
        self.mask_ratio = mask_ratio
        self.gate = gate
        self.n_classes = n_classes
        
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
        
        # Branch-specific classifiers
        self.branch_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, n_classes) for _ in range(n_branches)
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
        Forward pass combining ACMIL and CLAM approaches for classification
        """
        # Feature transformation (shared)
        h = self.fc1(x)  # [N, hidden_dim]
        h = self.relu(h)
        h = self.dropout(h)
        
        branch_logits = []
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
            
            # Branch classification
            branch_logit = self.branch_classifiers[branch_idx](M).squeeze(0)  # [n_classes]
            branch_logits.append(branch_logit)
        
        # Aggregate logits
        final_logits = torch.mean(torch.stack(branch_logits), dim=0)  # [n_classes]
        
        if return_branch_outputs:
            return final_logits, branch_logits, attention_weights_list
        
        return final_logits
    
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

def get_classification_model(model_name, **kwargs):
    """
    Factory function to get classification models
    
    Args:
        model_name: Name of the model ('mean', 'max', 'attention', 'clam', 'acmil', 'acmil_clam')
        **kwargs: Model-specific arguments (must include n_classes)
    
    Returns:
        Model instance
    """
    models = {
        'mean': MeanPoolingMILClassifier,
        'max': MaxPoolingMILClassifier,
        'attention': AttentionMILClassifier,
        'clam': CLAMClassifier,
        'acmil': ACMILClassifier,
        'acmil_clam': ACMIL_CLAM_HybridClassifier
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    
    # Ensure n_classes is specified
    if 'n_classes' not in kwargs:
        kwargs['n_classes'] = 2  # Default to binary classification
    
    return models[model_name](**kwargs)
    def __init__(self, input_dim=512, hidden_dim=128, attention_hidden_dim=256, 
                 n_classes=2, dropout=0.25, gate=True, **kwargs):
        """
        CLAM adapted for classification
        """
        super(CLAMClassifier, self).__init__()
        
        self.gate = gate
        self.n_classes = n_classes
        
        # Feature transformation layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Attention network
        if gate:
            self.attention_net = Attn_Net_Gated(L=hidden_dim, D=attention_hidden_dim, 
                                               dropout=dropout, n_classes=1)
        else:
            self.attention_net = nn.Sequential(
                nn.Linear(hidden_dim, attention_hidden_dim),
                nn.Tanh(),
                nn.Linear(attention_hidden_dim, 1)
            )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, n_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass for classification
        Args:
            x: Tensor of shape [N, D] (N patches per slide)
        Returns:
            logits of shape [n_classes]
        """
        # Feature transformation
        h = self.fc1(x)  # [N, hidden_dim]
        h = self.relu(h)
        h = self.dropout(h)
        
        # Get attention scores
        if self.gate:
            A, _ = self.attention_net(h)  # A: [N, 1]
        else:
            A = self.attention_net(h)  # A: [N, 1]
        
        # Transpose and softmax attention
        A = torch.transpose(A, 1, 0)  # [1, N] 
        A = torch.softmax(A, dim=1)  # softmax over N patches
        
        # Weighted aggregation
        M = torch.mm(A, h)  # [1, hidden_dim]
        
        # Classification
        logits = self.classifier(M)  # [1, n_classes]
        logits = logits.squeeze(0)  # [n_classes]
        
        if return_attention:
            return logits, A.transpose(1, 0)
        return logits
    
    def get_attention_weights(self, x):
        """Get attention weights for interpretability"""
        with torch.no_grad():
            h = self.fc1(x)
            h = self.relu(h)
            
            if self.gate:
                A, _ = self.attention_net(h)
            else:
                A = self.attention_net(h)
            
            A = torch.transpose(A, 1, 0)
            attention_weights = torch.softmax(A, dim=1)
            
        return attention_weights.transpose(1, 0)
