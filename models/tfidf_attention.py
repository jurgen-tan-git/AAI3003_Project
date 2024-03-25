import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class TfIdfAttention(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        embed_size: int,
        n_heads: int,
        dropout: float,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        embed_dim = n_heads * embed_size
        assert embed_dim == n_inputs, "n_inputs must be divisible by n_heads * embed_size"
        self.attention = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.fc = weight_norm(nn.Linear(n_heads, n_outputs))
        self.relu = nn.Mish()
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.fc,
            self.relu,
            self.dropout,
        )

    def forward(self, x):
        x, _ = self.attention(x, x, x)
        x = self.net(x)
        return x
