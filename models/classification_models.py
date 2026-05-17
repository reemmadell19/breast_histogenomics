# models/classification_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import copy


class Attn_Net(nn.Module):
    """Attention Network without Gating (2 fc layers)"""
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x  # N x n_classes, N x L

class Attn_Net_Gated(nn.Module):
    """Attention Network with Sigmoid Gating (3 fc layers)"""
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
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

# ====== CLAM =======
class CLAMClassifier(nn.Module):
    """
    CLAM implementation for binary classification with instance-level learning
    """
    def __init__(self, input_dim=512, hidden_dim=512, attention_hidden_dim=256,
                 n_classes=2, dropout=0.25, gate=True, 
                 instance_eval=False, k_sample=8, instance_loss_fn='ce', **kwargs):
        super(CLAMClassifier, self).__init__()
        
        assert n_classes == 2, "This implementation is for binary classification"
        
        self.n_classes = n_classes
        self.gate = gate
        self.instance_eval = instance_eval
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        
        # Build network architecture
        fc = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        
        if gate:
            attention_net = Attn_Net_Gated(L=hidden_dim, D=attention_hidden_dim, 
                                          dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=hidden_dim, D=attention_hidden_dim, 
                                    dropout=dropout, n_classes=1)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        # Bag-level classifier
        self.classifier = nn.Linear(hidden_dim, n_classes)
        
        # Instance-level classifier for binary classification
        if self.instance_eval:
            self.instance_classifier = nn.Linear(hidden_dim, 2)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def compute_instance_loss(self, h, attention_scores):
        """
        Compute instance-level loss for binary classification
        """
        device = h.device
        attention_scores = attention_scores.squeeze()
        
        n_patches = len(h)
        k = min(self.k_sample, n_patches // 2)
        
        if k == 0:
            return torch.tensor(0.0, device=device)
        
        _, top_idx = torch.topk(attention_scores, k)
        _, bottom_idx = torch.topk(-attention_scores, k)
        
        top_features = h[top_idx]
        bottom_features = h[bottom_idx]
        
        top_pred = self.instance_classifier(top_features)
        bottom_pred = self.instance_classifier(bottom_features)
        
        if self.instance_loss_fn == 'svm':
            top_loss = torch.clamp(1.0 - (top_pred[:, 1] - top_pred[:, 0]), min=0).mean()
            bottom_loss = torch.clamp(1.0 - (bottom_pred[:, 0] - bottom_pred[:, 1]), min=0).mean()
            instance_loss = (top_loss + bottom_loss) / 2
        else:
            top_targets = torch.ones(k, dtype=torch.long, device=device)
            bottom_targets = torch.zeros(k, dtype=torch.long, device=device)
            instance_loss = (F.cross_entropy(top_pred, top_targets) + 
                           F.cross_entropy(bottom_pred, bottom_targets)) / 2
        
        return instance_loss
    
    def forward(self, x, return_instance_loss=False, return_attention=False):
        A, h = self.attention_net(x)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        logits = self.classifier(M).squeeze(0)
        
        instance_loss = None
        if return_instance_loss and self.instance_eval and self.training:
            instance_loss = self.compute_instance_loss(h, A)
        
        if return_instance_loss and instance_loss is not None:
            if return_attention:
                return logits, instance_loss, A.transpose(1, 0)
            return logits, instance_loss
        elif return_attention:
            return logits, A.transpose(1, 0)
        else:
            return logits
    
    def get_attention_weights(self, x):
        with torch.no_grad():
            A, h = self.attention_net(x)
            A = torch.transpose(A, 1, 0)
            attention_weights = F.softmax(A, dim=1)
        return attention_weights.transpose(1, 0)

# ====== Mean =======
class MeanPoolingMILClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, n_classes=2, dropout=0.1, **kwargs):
        super(MeanPoolingMILClassifier, self).__init__()
        self.n_classes = n_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x, return_features=False):
        pooled = x.mean(dim=0)
        logits = self.classifier(pooled)
        if return_features:
            return logits, pooled
        return logits
# ====== Max =======
class MaxPoolingMILClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, n_classes=2, dropout=0.1, **kwargs):
        super(MaxPoolingMILClassifier, self).__init__()
        self.n_classes = n_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x, return_features=False):
        pooled, _ = x.max(dim=0)
        logits = self.classifier(pooled)
        if return_features:
            return logits, pooled
        return logits
# ====== Attention =======
class AttentionMILClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, attention_hidden_dim=64, 
                 n_classes=2, dropout=0.1, **kwargs):
        super(AttentionMILClassifier, self).__init__()
        self.n_classes = n_classes
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x, return_attention=False):
        attention_scores = self.attention(x)
        attention_weights = torch.softmax(attention_scores, dim=0)
        pooled = torch.sum(attention_weights * x, dim=0)
        logits = self.classifier(pooled)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def get_attention_weights(self, x):
        with torch.no_grad():
            attention_scores = self.attention(x)
            attention_weights = torch.softmax(attention_scores, dim=0)
        return attention_weights
# ====== ACMIL=======
class ACMILClassifier(nn.Module):
    """
    Official ACMIL: ABMIL + MBA + STKIM
    Based on: "Attention-Challenging Multiple Instance Learning for WSI Classification"
    """
    def __init__(self, input_dim=512, hidden_dim=128, 
                 n_classes=2, n_branches=10, dropout=0.25,
                 top_k=10, mask_ratio=0.7, 
                 lambda_p=0.5,  # Weight for semantic regularization
                 lambda_d=0.1,  # Weight for diversity loss
                 gate=True,      # Use gated attention like ABMIL
                 **kwargs):
        super(ACMILClassifier, self).__init__()
        
        # Remove unused parameters from kwargs to avoid conflicts
        unused_params = ['n_clusters', 'n_masked_patch', 'momentum', 'temperature', 
                        'lambda_cluster', 'lambda_consistency']
        for param in unused_params:
            kwargs.pop(param, None)
        
        self.n_classes = n_classes
        self.n_branches = n_branches
        self.top_k = top_k
        self.mask_ratio = mask_ratio
        self.lambda_p = lambda_p
        self.lambda_d = lambda_d
        self.gate = gate
        
        # Feature projection (like ABMIL)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple attention branches (MBA)
        self.attention_branches = nn.ModuleList()
        for _ in range(n_branches):
            if gate:
                # Use existing Attn_Net_Gated
                attention = Attn_Net_Gated(L=512, D=hidden_dim, dropout=dropout, n_classes=1)
            else:
                # Use existing Attn_Net
                attention = Attn_Net(L=512, D=hidden_dim, dropout=dropout, n_classes=1)
            self.attention_branches.append(attention)
        
        # Branch-specific classifiers for semantic regularization
        self.branch_classifiers = nn.ModuleList([
            nn.Linear(512, n_classes)
            for _ in range(n_branches)
        ])
        
        # Final bag classifier
        self.bag_classifier = nn.Linear(512, n_classes)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # Only initialize new layers, not the pre-existing attention modules
        for m in [self.bag_classifier] + list(self.branch_classifiers):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, x, label=None):
        """
        Args:
            x: [N, D] patch features from a bag
            label: bag label for training
        Returns:
            logits: final prediction
            total_loss: combined loss if training
        """
        device = x.device
        
        # Feature extraction
        h = self.feature_extractor(x)  # [N, 512]
        
        # Process each attention branch
        branch_embeddings = []
        branch_attentions = []
        branch_logits = []
        
        for i, attention_branch in enumerate(self.attention_branches):
            # Compute attention
            A, h_out = attention_branch(h)  # Both gated and non-gated return (A, h)
            A = A.transpose(0, 1)  # [1, N]
            
            # Softmax normalization
            A = F.softmax(A, dim=1)  # [1, N]
            branch_attentions.append(A)
            
            # STKIM: Stochastic masking during training
            if self.training and self.top_k > 0:
                A_masked = A.clone()
                k = min(self.top_k, A.size(1))
                
                # Get top-k attention scores
                if k > 0:
                    _, top_idx = torch.topk(A_masked.squeeze(), k)
                    
                    # Randomly mask some top instances
                    n_mask = int(k * self.mask_ratio)
                    if n_mask > 0:
                        # Randomly select which top instances to mask
                        perm = torch.randperm(k, device=device)[:n_mask]
                        mask_idx = top_idx[perm]
                        A_masked[0, mask_idx] = 0
                        # Renormalize
                        sum_A = A_masked.sum()
                        if sum_A > 0:
                            A_masked = A_masked / sum_A
                        else:
                            # If all masked, use uniform distribution
                            A_masked = torch.ones_like(A_masked) / A_masked.size(1)
                
                # Use masked attention for aggregation
                z = torch.mm(A_masked, h)  # [1, 512]
            else:
                # No masking during inference
                z = torch.mm(A, h)  # [1, 512]
            
            branch_embeddings.append(z)
            
            # Branch-specific classification for semantic regularization
            if self.training and label is not None:
                branch_logit = self.branch_classifiers[i](z)
                branch_logits.append(branch_logit)
        
        # Average branch embeddings for final prediction
        avg_embedding = torch.mean(torch.stack(branch_embeddings), dim=0)  # [1, 512]
        logits = self.bag_classifier(avg_embedding).squeeze(0)  # [n_classes]
        
        if self.training and label is not None:
            if label.dim() == 0:
                label = label.unsqueeze(0)
            
            # 1. Bag classification loss
            L_bag = F.cross_entropy(logits.unsqueeze(0), label)
            
            # 2. Semantic regularization (branch losses)
            L_p = 0
            for branch_logit in branch_logits:
                L_p += F.cross_entropy(branch_logit, label)
            L_p = L_p / self.n_branches
            
            # 3. Diversity loss (encourages different attention patterns)
            L_d = 0
            num_pairs = 0
            for i in range(self.n_branches):
                for j in range(i+1, self.n_branches):
                    a_i = branch_attentions[i].squeeze()
                    a_j = branch_attentions[j].squeeze()
                    # Compute cosine similarity
                    cos_sim = F.cosine_similarity(a_i.unsqueeze(0), a_j.unsqueeze(0))
                    L_d += cos_sim
                    num_pairs += 1
            
            if num_pairs > 0:
                L_d = L_d / num_pairs
            
            # Total loss (note: diversity loss is subtracted to encourage diversity)
            total_loss = L_bag + self.lambda_p * L_p - self.lambda_d * L_d
            
            return logits, total_loss
        
        return logits
    
    def get_attention_weights(self, x):
        """Get averaged attention weights for visualization"""
        with torch.no_grad():
            h = self.feature_extractor(x)
            
            all_attentions = []
            for attention_branch in self.attention_branches:
                A, _ = attention_branch(h)
                A = A.transpose(0, 1)
                A = F.softmax(A, dim=1)
                all_attentions.append(A.transpose(0, 1))
            
            # Average attention across branches
            avg_attention = torch.mean(torch.stack(all_attentions), dim=0)
            
        return avg_attention

# ============= Model Factory Function =============

def get_classification_model(model_name, **kwargs):
    """Factory function to get classification models"""
    models = {
        'mean': MeanPoolingMILClassifier,
        'max': MaxPoolingMILClassifier,
        'attention': AttentionMILClassifier,
        'clam': CLAMClassifier,
        'acmil': ACMILClassifier,
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    
    if 'n_classes' not in kwargs:
        kwargs['n_classes'] = 2
    
    return models[model_name](**kwargs)