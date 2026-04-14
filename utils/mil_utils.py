# Missing collate function
def mil_collate_fn(batch):
    """
    Custom collate function for MIL data
    batch: list of (features, label) tuples
    """
    features_list = []
    labels_list = []
    
    for features, label in batch:
        features_list.append(features)
        labels_list.append(label)
    
    # For MIL, we typically process one bag at a time
    # So we'll return the first item if batch_size=1
    if len(batch) == 1:
        return features_list[0], labels_list[0]
    else:
        # For batch processing, you might want to pad or handle variable sizes
        return features_list, labels_list