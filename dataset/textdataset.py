"""This module contains the ArticleDataset class for loading and preprocessing article data.
"""

import os

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class ArticleDataset(Dataset):
    """
    A PyTorch dataset for loading and preprocessing article data.

    :param articles_dir: Directory containing the articles.
    :type articles_dir: str | os.PathLike[str]
    :param labelled_csv: Path to the labelled CSV file.
    :type labelled_csv: str | os.PathLike[str]
    :param tokenizer: Tokenizer for tokenizing the articles.
    :type tokenizer: PreTrainedTokenizerBase
    :param max_length: Maximum length of the input tokens.
    :type max_length: int
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
                self.targets.append(targets.to_list())
        self.categories = df.columns[2:].to_list()

    def __len__(self):
        """
        Returns the number of articles in the dataset.
        """
        return len(self.articles)

    def __getitem__(self, idx):
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
        res = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
        }

        if "token_type_ids" in inputs:
            res["token_type_ids"] = inputs.token_type_ids

        return res, torch.tensor(target)
