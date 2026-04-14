
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np

# Binary label mapping
rshigh_to_label = {'L': 0, 'H': 1}

class ClassificationMILDataset(Dataset):
    """
    MIL Dataset for classification of breast cancer recurrence risk.
    Converts continuous RS scores to categorical labels if needed.
    """
    def __init__(self, csv_path, label_column='RSHigh', transform=None, threshold=25.0):
        """
        Args:
            csv_path: Path to CSV file with features
            label_column: Either 'RSHigh' (for L/H labels) or 'RS' (for continuous scores)
            transform: Optional data transforms
            threshold: RS threshold for binary classification (default 25.0)
        """
        self.df = pd.read_csv(csv_path)
        self.label_column = label_column
        self.transform = transform
        self.threshold = threshold
        
        # Handle different label formats
        if label_column == 'RSHigh' and label_column not in self.df.columns:
            # Create RSHigh from RS if not present
            if 'RS' in self.df.columns:
                self.df['RSHigh'] = self.df['RS'].apply(lambda x: 'H' if x >= threshold else 'L')
            else:
                raise ValueError(f"Neither {label_column} nor RS column found in CSV")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_path = row['path']
        
        # Load dict and extract the actual feature tensor
        data = torch.load(feature_path)
        features = data['features']  # shape [N, D]
        
        # Get label based on column type
        if self.label_column == 'RSHigh':
            label = rshigh_to_label[row[self.label_column]]
        elif self.label_column == 'RS':
            # Convert continuous RS to binary label
            rs_score = float(row['RS'])
            label = 1 if rs_score >= self.threshold else 0
        else:
            label = int(row[self.label_column])
        
        if self.transform:
            features = self.transform(features)
        
        return features, label


def create_classification_weighted_sampler(dataset, balance_classes=True):
    """
    Create weighted sampler for handling class imbalance in classification.
    
    Args:
        dataset: ClassificationMILDataset instance
        balance_classes: If True, use inverse frequency weighting
    """
    if not balance_classes:
        return None
    
    # Get all labels
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Calculate class weights (inverse frequency)
    n_samples = len(labels)
    n_classes = len(np.unique(labels))
    class_counts = np.bincount(labels)
    
    # Inverse frequency weights
    class_weights = n_samples / (n_classes * class_counts)
    
    # Assign weight to each sample
    sample_weights = class_weights[labels]
    
    print(f"Class distribution: {class_counts[0]} low-risk, {class_counts[1]} high-risk")
    print(f"Class weights: Low={class_weights[0]:.3f}, High={class_weights[1]:.3f}")
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


# ================================