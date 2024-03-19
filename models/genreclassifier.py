import torch
import torch.nn.functional as F
from torch import nn


class GenreClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GenreClassifier, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MiddleChomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
        self.half_chomp = chomp_size // 2

    def forward(self, x):
        return x[:, :, self.half_chomp : -self.half_chomp].contiguous()


class GenreClassifierBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        padding,
        dropout,
        dilation,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = MiddleChomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = torch.nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = MiddleChomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class AttentionGenreClassifier(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_channels,
        kernel_size,
        dropout,
        use_attention: bool,
        embed_dim=768,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        layers = []
        num_levels = len(num_channels)
        self.use_attention = use_attention

        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_channels[0],
                dropout=dropout,
                batch_first=True,
            )

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                GenreClassifierBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        fc1 = nn.Linear(num_channels[-1], 512)
        fc2 = nn.Linear(512, num_outputs)
        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            fc1, nn.Dropout(dropout), nn.ReLU(), fc2, nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        if self.use_attention:
            x, _ = self.attention(x, x, x)
            x = x.permute(0, 2, 1)
        x = self.network(x)
        x = torch.flatten(x)
        x = self.fc(x)
        return x
