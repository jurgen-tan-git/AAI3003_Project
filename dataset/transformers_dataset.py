import os

import pandas as pd
from tqdm.auto import tqdm


def load_data(labelled_csv: str | os.PathLike, articles_dir: str | os.PathLike):
    df = pd.read_csv(labelled_csv)
    for root, _, files in tqdm(
        os.walk(articles_dir),
        desc="Loading articles",
        unit="file",
        position=0,
        leave=False,
    ):
        for file in tqdm(
            files, desc="Loading files", unit="file", position=1, leave=False
        ):
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    text = f.read()
                    df.loc[df["File"] == file, "Text"] = text
    return df


def get_dict(df):
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
