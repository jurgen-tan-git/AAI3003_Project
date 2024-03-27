"""BERT feature extraction dataset.
"""

import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """Feature extraction dataset.

    Args:
        features (torch.Tensor): Text embeddings from the BERT model.
        labels (torch.Tensor): Encoded labels.
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
