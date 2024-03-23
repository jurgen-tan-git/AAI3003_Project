import os

import pandas as pd
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
        labelled_csv (str or os.PathLike): The path to the labelled CSV file.
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
        labelled_csv: str | os.PathLike[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        super().__init__()
        self.articles_dir = articles_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labelled_csv = labelled_csv
        self.articles = []
        self.labels = []
        self.targets = []
        self.categories = []
        self.label_encoder = LabelEncoder()
        self._init_dataset()

    def _init_dataset(self):
        """
        Initializes the dataset by loading and preprocessing the articles.
        """
        df = pd.read_csv(self.labelled_csv)
        for root, _, files in os.walk(self.articles_dir):
            for file in files:
                if file.endswith(".txt"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        text = f.read()
                        df.loc[df["File"] == file, "Text"] = text

        for _, row in df.iterrows():
            targets = row[2:]
            labels = df.columns[2:][targets == 1]
            labels = list(map(lambda x: x.replace("-", " "), labels))
            text = row["Text"]
            if text:
                self.articles.append(text)
                self.labels.append(labels)
                self.targets.append(targets)
        self.categories = df.columns[2:].to_list()

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
            dict: A dictionary containing the tokenized input and the multilabel
                targets.
        """
        text = self.articles[idx]
        target = self.targets[idx]
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
        }, torch.tensor(target)
