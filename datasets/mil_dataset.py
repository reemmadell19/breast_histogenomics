# mil_dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd

# Binary label mapping
rshigh_to_label = {'L': 0, 'H': 1}

class MILDataset(Dataset):
    def __init__(self, csv_path, label_column='RSHigh', transform=None):
        self.df = pd.read_csv(csv_path)
        self.label_column = label_column
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_path = row['path']
        
        # Load dict and extract the actual feature tensor
        data = torch.load(feature_path)
        features = data['features']  # shape [N, 512]
        label = row[self.label_column]
        
        if self.label_column == 'RSHigh':
            label = rshigh_to_label[label]
        
        if self.transform:
            features = self.transform(features)
        
        return features, label