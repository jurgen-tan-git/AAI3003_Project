"""This module contains functions for loading and processing data for the transformers library.
"""

import os

import pandas as pd


def load_data(
    labelled_csv: str | os.PathLike,
    articles_dir: str | os.PathLike,
    use_original_text: bool = False,
) -> pd.DataFrame:
    """Loads data from a labelled CSV file and a directory of articles.

    :param labelled_csv: Path to the labelled CSV file.
    :type labelled_csv: str | os.PathLike
    :param articles_dir: Directory containing the articles.
    :type articles_dir: str | os.PathLike
    :param use_original_text: Whether to use the original text or not.
    :type use_original_text: bool, optional
    :return: A pandas DataFrame containing the labelled data.
    :rtype: pd.DataFrame
    """
    df = pd.read_csv(labelled_csv)
    for root, _, files in os.walk(articles_dir):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    text = f.read()
                    if use_original_text:
                        df.loc[df["File"] == file, "Text"] = text
                    df.loc[df["File"] == file, "fp"] = os.path.join(root, file)
    return df


def get_dict(df: pd.DataFrame) -> dict:
    """Generates a dataset dictionary for the transformers library.

    :param df: A pandas DataFrame containing the dataset.
    :type df: pd.DataFrame
    :return: A dictionary containing the text, binary_targets, and labels.
    :rtype: dict
    """
    dataset = {}
    for _, row in df.iterrows():
        targets = row[2:]
        labels = df.columns[2:][targets == 1]
        labels = list(map(lambda x: x.replace("-", " "), labels))
        if dataset.get("text") is None:
            dataset["text"] = [row["Text"]]
            dataset["binary_targets"] = [targets]
            dataset["labels"] = [labels]
        else:
            dataset["text"].append(row["Text"])
            dataset["binary_targets"].append(targets)
            dataset["labels"].append(labels)
    return dataset
