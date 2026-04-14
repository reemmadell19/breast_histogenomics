import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        alpha (float): Weighting factor for rare class (default: 1.0)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): Specifies the reduction to apply to the output
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Forward pass for Focal Loss.
        
        Args:
            inputs: Prediction logits [batch_size, num_classes]
            targets: Ground truth class indices [batch_size]
        
        Returns:
            Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute p_t
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with class-specific alpha weighting.
    
    Args:
        alpha (list or tensor): Class-specific weighting factors
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): Specifies the reduction to apply to the output
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Forward pass for Weighted Focal Loss.
        
        Args:
            inputs: Prediction logits [batch_size, num_classes]
            targets: Ground truth class indices [batch_size]
        
        Returns:
            Weighted focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Get alpha values for each target
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss