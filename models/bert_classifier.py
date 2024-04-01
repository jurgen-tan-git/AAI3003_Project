"""BERT-based classifier models.
"""

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from transformers import AutoModel


class GenreClassifierBlockBertMod(nn.Module):
    """GenreClassifier block with BERT.

    :param n_inputs: Number of input channels.
    :type n_inputs: int
    :param n_outputs: Number of output channels.
    :type n_outputs: int
    :param kernel_size: Kernel sizes.
    :type kernel_size: int | tuple[int]
    :param stride: Stride sizes.
    :type stride: int | tuple[int]
    :param padding: Padding sizes.
    :type padding: int | tuple[int]
    :param dropout: Dropout rate.
    :type dropout: float
    :param dilation: Dilation factor.
    :type dilation: int
    """

    def __init__(
        self,
        *args,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int],
        padding: int | tuple[int],
        dropout: float,
        dilation: int,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = weight_norm(
            nn.Conv2d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
            )
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv2d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
            )
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.relu2,
            self.dropout2,
        )

        self.downsample = (
            nn.Conv2d(n_inputs, n_outputs, 1, stride=stride)
            if n_inputs != n_outputs
            else None
        )

        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights of the model."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class AttentionGenreClassifierBertMod(nn.Module):
    """Modified GenreClassifier with BERT and attention.

    :param num_inputs: Number of input channels.
    :type num_inputs: int
    :param num_classes: Number of output classes.
    :type num_classes: int
    :param channels: List of channel sizes.
    :type channels: list[int]
    :param kernel_size: Kernel sizes.
    :type kernel_size: int | tuple[int]
    :param dropout: Dropout rate.
    :type dropout: float
    :param padding: Padding sizes.
    :type padding: int | tuple[int]
    :param use_attention: Enables attention block, defaults to True
    :type use_attention: bool, optional
    :param embed_dim: Embedding dimension size, defaults to 768
    :type embed_dim: int, optional
    :param max_length: Maximum input length, defaults to 512
    :type max_length: int, optional
    :param dilate: Enables dilated layers, defaults to True
    :type dilate: bool, optional
    """

    def __init__(
        self,
        num_inputs: int,
        num_classes: int,
        channels: list,
        kernel_size: int | tuple,
        dropout: float,
        use_attention: bool = True,
        embed_dim: int = 768,
        max_length: int = 512,
        padding: int | tuple = 0,
        dilate: bool = True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        layers = []
        num_levels = len(channels)
        self.use_attention = use_attention

        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=channels[0],
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.attention = None

        for i in range(num_levels):
            dilation_size = 2**i if dilate else 1
            in_channels = num_inputs if i == 0 else channels[i - 1]
            out_channels = channels[i]
            pad = padding[i] if isinstance(padding, tuple) else padding
            stride = (
                kernel_size[0] - 1
                if isinstance(kernel_size, tuple)
                else kernel_size - 1
            )
            layers += [
                GenreClassifierBlockBertMod(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    stride=stride,
                    dilation=dilation_size,
                    padding=pad,
                    dropout=dropout,
                    kernel_size=kernel_size,
                )
            ]

        fc1 = nn.LazyLinear(512)
        fc2 = nn.Linear(512, num_classes)
        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            fc1, nn.Dropout(dropout), nn.ReLU(), fc2, nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        if self.use_attention:
            x, _ = self.attention(x, x, x)
            if len(x.shape) > 3:
                x = x.permute(0, 1, 3, 2)
            else:
                x = x.permute(0, 2, 1)
                x = x.unsqueeze(1)
        print(x.shape)
        x = self.network(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BertWithAttentionClassifier(nn.Module):
    """BERT with an attention-based classifier.

    :param model_name: Model name or version to download.
    :type model_name: str
    :param num_classes: Number of output classes.
    :type num_classes: int
    :param max_length: Maximum input length.
    :type max_length: int
    :param attention_dim: Attention dimension size, defaults to 100
    :type attention_dim: int, optional
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        max_length: int,
        attention_dim: int = 100,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.bert = AutoModel.from_pretrained(
            model_name, max_position_embeddings=max_length
        )
        self.max_length = max_length
        self.net = AttentionGenreClassifierBertMod(
            num_inputs=1,
            num_classes=num_classes,
            channels=[32, 64, 128, 64, 32],
            kernel_size=3,
            dropout=0.2,
            use_attention=True,
            embed_dim=768,
            max_length=max_length,
            padding=0,
            dilate=False,
        )

    def forward(self, inputs):
        outputs = self.bert(**inputs)
        last_hidden_state = outputs.last_hidden_state
        print(last_hidden_state.shape)
        attention_output = self.net(last_hidden_state)
        print(attention_output.shape)
        return attention_output


class BertWithLinearClassifier(nn.Module):
    """BERT with a linear classifier.

    :param num_classes: Number of output classes.
    :type num_classes: int
    :param max_length: Maximum input length.
    :type max_length: int
    :param dropout: Dropout rate.
    :type dropout: float
    :param model_name: Model name or version to download, defaults to "bert-base-uncased"
    :type model_name: str, optional
    """

    def __init__(
        self,
        num_classes: int,
        max_length: int,
        dropout: float,
        model_name: str = "bert-base-uncased",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.bert = AutoModel.from_pretrained(
            model_name,
            max_position_embeddings=max_length,
        )
        self.fc1 = nn.Linear(512 * 768, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.net = nn.Sequential(
            self.fc1, self.relu, nn.Dropout(dropout), self.fc2, nn.Dropout(dropout)
        )

    def forward(self, inputs):
        outputs = self.bert(**inputs)
        last_hidden_state = outputs.last_hidden_state
        if len(last_hidden_state.shape) > 3:
            x = torch.flatten(last_hidden_state, 2)
        else:
            x = torch.flatten(last_hidden_state, 1)
        x = self.net(x)
        return x
