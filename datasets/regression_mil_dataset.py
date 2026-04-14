# datasets/regression_mil_dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class RegressionMILDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        
        # Verify RS column exists
        if 'RS' not in self.df.columns:
            raise ValueError("Missing required 'RS' column for regression targets")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_path = row['path']
        
        # Load dict and extract the actual feature tensor
        data = torch.load(feature_path)
        features = data['features']  # shape [N, 512]
        
        # Get continuous RS score as regression target
        rs_score = float(row['RS'])  # Continuous: 0-100
        
        if self.transform:
            features = self.transform(features)
        
        return features, rs_score

def create_rs_weighted_sampler(dataset, boundary_focus=True, class_balance=True, 
                              threshold=25.0, boundary_range=10.0):
    """
    Create weighted sampler that handles both class imbalance and boundary emphasis.
    
    Args:
        dataset: RegressionMILDataset instance
        boundary_focus: If True, weight samples near threshold more heavily
        class_balance: If True, balance high-risk vs low-risk samples
        threshold: RS threshold for clinical decision (default 25.0)
        boundary_range: Range around threshold to emphasize (default 10.0)
    """
    rs_scores = np.array([dataset[i][1] for i in range(len(dataset))])
    weights = np.ones(len(rs_scores))
    
    # Apply class balancing weights
    if class_balance:
        binary_labels = (rs_scores >= threshold).astype(int)
        n_low = np.sum(binary_labels == 0)  # RS < 25
        n_high = np.sum(binary_labels == 1)  # RS >= 25
        
        # Inverse frequency weighting
        low_weight = 1.0 / (n_low / len(rs_scores))  # Weight for low-risk samples
        high_weight = 1.0 / (n_high / len(rs_scores))  # Weight for high-risk samples
        
        weights = np.where(binary_labels == 0, low_weight, high_weight)
        
        print(f"Class distribution: {n_low} low-risk, {n_high} high-risk")
        print(f"Class weights: Low={low_weight:.3f}, High={high_weight:.3f}")
    
    # Apply additional boundary weighting (multiplicative)
    if boundary_focus:
        distances = np.abs(rs_scores - threshold)
        boundary_multiplier = np.where(distances <= boundary_range, 2.0, 1.0)
        weights = weights * boundary_multiplier
        print(f"Boundary weighting applied: 2x for samples within ±{boundary_range} of RS={threshold}")
    
    from torch.utils.data import WeightedRandomSampler
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )