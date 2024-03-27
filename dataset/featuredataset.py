"""BERT feature extraction dataset.
"""

import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """Feature extraction dataset.

    :param features: Extracted features from the text.
    :type features: torch.Tensor
    :param labels: Encoded labels.
    :type labels: torch.Tensor
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
