import pandas as pd
import matplotlib.pyplot as plt

def plot_distribution(manifest_path, split_name):
    df = pd.read_csv(manifest_path)
    counts = df["RSHigh"].value_counts().sort_index()
    plt.bar(counts.index, counts.values)
    plt.title(f"RSHigh Distribution in {split_name}")
    plt.xlabel("RSHigh")
    plt.ylabel("Count")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 1, str(v), ha='center')
    plt.show()

# Paths
train_path = "data/manifests/train_manifest.csv"
val_path = "data/manifests/val_manifest.csv"
test_path = "data/manifests/test_manifest.csv"

# Plot each split
plot_distribution(train_path, "Train")
plot_distribution(val_path, "Val")
plot_distribution(test_path, "Test")
