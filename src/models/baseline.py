import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from src.data.augmentations import get_eval_transforms

class ResNet50FeatureExtractor(nn.Module):
    """
    Standard Pretrained ResNet-50 Feature Extractor.
    Extracts 2048-dimensional L2-normalized feature embeddings.
    """
    def __init__(self):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)
        # Remove final FC classification layer
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        
        # Standard PyTorch ImageNet preprocessing
        self.transform = get_eval_transforms()
        
    def forward(self, x):
        features = self.feature_extractor(x) # (B, 2048, 1, 1)
        features = features.squeeze(-1).squeeze(-1) # (B, 2048)
        # L2 normalize embeddings
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features

    def extract_image_embedding(self, image_path, device='cpu'):
        self.eval()
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = self.forward(tensor)
        return embedding.squeeze(0).cpu().numpy()
