"""This module contains functions for fine-tuning a BART model on a multi-label
text classification task.
"""

import os

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from dataset.transformers_dataset import get_dict, load_data
from metrics.transformers_evaluate import compute_metrics

BATCH_SIZE = 2  # Batch size for training
NUM_LABELS = 8  # Number of labels in the dataset
NUM_FOLDS = 5  # Number of folds for cross-validation


def preprocess_dataset(ds: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """Preprocess the dataset by tokenizing the text and adding the labels.

    :param ds: Dataset to preprocess.
    :type ds: Dataset
    :param tokenizer: Tokenizer to use for tokenization.
    :type tokenizer: AutoTokenizer
    :return: Preprocessed dataset.
    :rtype: Dataset
    """

    def preprocess_function(sample: dict) -> dict:
        """Preprocess the sample by tokenizing the text and adding the labels.

        :param sample: Sample to preprocess.
        :type sample: dict
        :return: Preprocessed sample.
        :rtype: dict
        """
        example = tokenizer(sample["text"], padding=True, truncation=True)
        labels = sample["binary_targets"]
        example["labels"] = labels.float()
        return example

    tokenized_ds = ds.map(preprocess_function, batched=True)
    return tokenized_ds


def finetune(
    kf: KFold,
    tokenizer: AutoTokenizer,
    ds: Dataset,
    df: pd.DataFrame,
) -> None:
    """Fine-tune the BART model on the multi-label text classification task.

    :param kf: KFold object for cross-validation.
    :type kf: KFold
    :param tokenizer: Tokenizer to use for tokenization.
    :type tokenizer: AutoTokenizer
    :param ds: Dataset to use for training and evaluation.
    :type ds: Dataset
    :param df: DataFrame containing the labels.
    :type df: pd.DataFrame
    """    
    for fold, (train_idx, test_idx) in tqdm(
        enumerate(kf.split(ds["text"], ds["labels"])),
        total=NUM_FOLDS,
        leave=False,
        position=0,
    ):
        model = AutoModelForSequenceClassification.from_pretrained(
            "facebook/bart-large-mnli",
            problem_type="multi_label_classification",
            num_labels=NUM_LABELS,
            id2label={str(i): label for i, label in enumerate(df.columns[2:])},
            label2id={label: i for i, label in enumerate(df.columns[2:])},
            ignore_mismatched_sizes=True,
        )
        if not os.path.exists(f"./results/{fold}/"):
            os.makedirs(f"./results/{fold}/")
        training_args = TrainingArguments(
            output_dir=f"./results/{fold}/",
            num_train_epochs=2,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            bf16=True,
            learning_rate=2e-5,
            metric_for_best_model="f1",
        )
        train_ds = ds.select(train_idx)
        test_ds = ds.select(test_idx)

        train_ds = preprocess_dataset(train_ds, tokenizer)
        test_ds = preprocess_dataset(test_ds, tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()


def main():
    """The main function to fine-tune the BART model on the multi-label text classification task."""
    # Load the data
    df = load_data("multi_label_dataset.csv", "./articles")
    ds = Dataset.from_dict(get_dict(df))
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    ds.set_format(type="torch")
    finetune(kf, tokenizer, ds, df)


if __name__ == "__main__":
    main()
