import pandas as pd

# Load both manifests
ucmc_df = pd.read_csv("data/manifests/ucmc_manifest.csv")
bcrnet_df = pd.read_csv("data/manifests/bcrnet_manifest.csv")

# Add dataset column
ucmc_df["dataset"] = "UCMC"
bcrnet_df["dataset"] = "BCRNet"

# Combine
combined_df = pd.concat([ucmc_df, bcrnet_df], ignore_index=True)

# Reorder columns if needed
combined_df = combined_df[["dataset", "file_name", "num_tiles", "RS", "RSHigh"]]

# Save
combined_df.to_csv("data/manifests/unified_manifest.csv", index=False)

print("Unified manifest created: data/manifests/unified_manifest.csv")
