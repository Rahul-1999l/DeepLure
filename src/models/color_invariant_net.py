import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ColorInvariantResNet50(nn.Module):
    """
    Proposed Color-Invariant Metric Learning Model.
    ResNet-50 backbone + Projection Head (2048 -> 512 -> embed_dim) + L2 Normalization.
    """
    def __init__(self, embed_dim=256, freeze_backbone=False, unfreeze_from_layer='layer4'):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)
        
        # Feature extractor up to avgpool (outputs 2048-d)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Configure backbone freezing
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif unfreeze_from_layer is not None:
            # Freeze early layers, unfreeze from specified layer
            layer_names = ['0', '1', '2', '3', '4', 'layer1', 'layer2', 'layer3', 'layer4']
            unfreeze = False
            for name, child in self.backbone.named_children():
                if name == unfreeze_from_layer:
                    unfreeze = True
                for param in child.parameters():
                    param.requires_grad = unfreeze

        # Projection Head: Linear(2048 -> 512) -> BatchNorm1d -> ReLU -> Dropout -> Linear(512 -> embed_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        features = self.backbone(x) # (B, 2048, 1, 1)
        features = features.squeeze(-1).squeeze(-1) # (B, 2048)
        embeddings = self.projection_head(features) # (B, embed_dim)
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
