import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, output_dim=512):
        super().__init__()
        
        if backbone == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove classification head
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])  # remove fc

        # Optional projection layer
        if output_dim and output_dim != self.feature_dim:
            self.projector = nn.Linear(self.feature_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.projector = None
            self.output_dim = self.feature_dim

    def forward(self, x):
        # x: [B, C, H, W]
        feats = self.feature_extractor(x)  # [B, F, 1, 1]
        feats = feats.view(feats.size(0), -1)  # flatten to [B, F]
        if self.projector:
            feats = self.projector(feats)
        return feats
    
    def get_feature_dim(self):
        return self.output_dim  