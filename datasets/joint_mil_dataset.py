# datasets/joint_mil_dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd

# Binary label mapping
rshigh_to_label = {'L': 0, 'H': 1}

class JointMILDataset(Dataset):
    """
    Dataset for joint classification and regression tasks.
    Returns both binary labels and continuous RS scores.
    """
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        
        # Verify required columns exist
        required_cols = ['path', 'RS', 'RSHigh']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_path = row['path']
        
        # Load dict and extract the actual feature tensor
        data = torch.load(feature_path)
        features = data['features']  # shape [N, 512]
        
        # Get both labels
        classification_label = rshigh_to_label[row['RSHigh']]  # Binary: 0 or 1
        regression_target = float(row['RS'])  # Continuous: 0-100
        
        if self.transform:
            features = self.transform(features)
        
        return {
            'features': features,
            'classification_label': classification_label,
            'regression_target': regression_target,
            'slide_id': row.get('slide_id', f'slide_{idx}')  # Optional slide ID
        }


def joint_mil_collate_fn(batch):
    """
    Collate function for joint MIL dataset.
    Handles variable number of patches per slide.
    """
    features_list = []
    classification_labels = []
    regression_targets = []
    slide_ids = []
    
    for item in batch:
        features_list.append(item['features'])
        classification_labels.append(item['classification_label'])
        regression_targets.append(item['regression_target'])
        slide_ids.append(item['slide_id'])
    
    # Convert to tensors
    classification_labels = torch.tensor(classification_labels, dtype=torch.long)
    regression_targets = torch.tensor(regression_targets, dtype=torch.float32)
    
    return {
        'features': features_list,  # List of tensors (variable sizes)
        'classification_labels': classification_labels,
        'regression_targets': regression_targets,
        'slide_ids': slide_ids
    }


# For backward compatibility with your existing dataset
class MILDataset(Dataset):
    """
    Original dataset class for classification only.
    Extended to optionally return regression targets.
    """
    def __init__(self, csv_path, label_column='RSHigh', return_regression=False, transform=None):
        self.df = pd.read_csv(csv_path)
        self.label_column = label_column
        self.return_regression = return_regression
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
        
        if self.return_regression:
            regression_target = float(row['RS'])
            return features, label, regression_target
        else:
            return features, label