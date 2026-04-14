import torch
sample_path = "data/features_h-optimus/train/2.pt"  # Use an actual path
data = torch.load(sample_path)
print("Keys:", data.keys())
print("Features shape:", data['features'].shape if 'features' in data else "No features key")
print("Coords shape:", data['coords'].shape if 'coords' in data else "No coords key")
print("Sample coords:", data['coords'][:5] if 'coords' in data else "No coords")