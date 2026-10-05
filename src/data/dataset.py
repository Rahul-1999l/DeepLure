import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class SareeDataset(Dataset):
    """
    PyTorch Dataset for Saree Images.
    """
    def __init__(self, df, root_dir, transform=None, label_encoder=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Design IDs
        self.design_ids = self.df['design_id'].values
        
        if label_encoder is None:
            unique_ids = sorted(list(set(self.design_ids)))
            self.label_encoder = {did: idx for idx, did in enumerate(unique_ids)}
        else:
            self.label_encoder = label_encoder
            
        self.labels = [self.label_encoder[did] for did in self.design_ids]
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row['image_path']
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = img
            
        label = self.labels[idx]
        design_id = self.design_ids[idx]
        
        return {
            'image': img_tensor,
            'label': label,
            'design_id': design_id,
            'index': idx,
            'image_path': row['image_path']
        }
