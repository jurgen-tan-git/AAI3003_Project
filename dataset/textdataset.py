import os

import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class ArticleDataset(Dataset):
    """
    A PyTorch dataset for loading and preprocessing article data.

    Args:
        articles_dir (str or os.PathLike): The directory path where the articles are stored.
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to tokenize the text.
        max_length (int): The maximum length of the tokenized input.

    Attributes:
        articles_dir (str or os.PathLike): The directory path where the articles are stored.
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to tokenize the text.
        max_length (int): The maximum length of the tokenized input.
        articles (list): A list of article texts.
        labels (list): A list of article labels.
        categories (list): A list of article categories.
        label_encoder (LabelEncoder): The label encoder used to encode the article labels.

    Methods:
        _init_dataset(): Initializes the dataset by loading and preprocessing the articles.
        __len__(): Returns the number of articles in the dataset.
        __getitem__(idx): Returns the tokenized input and label for a given index.

    """

    def __init__(
        self,
        articles_dir: str | os.PathLike[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        super().__init__()
        self.articles_dir = articles_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.articles = []
        self.labels = []
        self.categories = []
        self.label_encoder = LabelEncoder()
        self._init_dataset()

    def _init_dataset(self):
        """
        Initializes the dataset by loading and preprocessing the articles.
        """
        self.categories = os.listdir(self.articles_dir)
        for category in self.categories:
            for article in os.listdir(os.path.join(self.articles_dir, category)):
                with open(
                    os.path.join(self.articles_dir, category, article),
                    "r",
                    encoding="utf-8",
                ) as f:
                    text = f.read()
                    self.articles.append(text)
                    self.labels.append(category)
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        """
        Returns the number of articles in the dataset.
        """
        return len(self.articles)

    def __getitem__(self, idx):
        """
        Returns the tokenized input and label for a given index.

        Args:
            idx (int): The index of the article.

        Returns:
            dict: A dictionary containing the tokenized input and label.
        """
        text = self.articles[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            truncation_strategy="only_first",
        )
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "token_type_ids": (
                inputs.token_type_ids if "token_type_ids" in inputs else None
            ),
        }, torch.tensor(label)
