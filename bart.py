"""This module contains functions for fine-tuning a BART model on a multi-label
text classification task.
"""

from functools import partial
from typing import Generator

import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    coverage_error,
    f1_score,
    label_ranking_average_precision_score,
    label_ranking_loss,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import pipeline

from dataset.textdataset import ArticleDataset
from dataset.transformers_dataset import get_dict, load_data
from metrics.auc import godbole_accuracy

LABELLED_CSV = "multi_label_dataset.csv"
ARTICLES_DIR = "./articles"
NUM_FOLDS = 5


def evaluate(y_true: np.ndarray, y_prob: np.ndarray):
    """Evaluate the model's performance on a multi-label classification task.

    :param y_true: True labels.
    :type y_true: np.ndarray
    :param y_prob: Predicted probabilities.
    :type y_prob: np.ndarray
    """
    best_thresh = 0
    max_f1 = 0.0
    for thresh in sorted(y_prob.flatten()):
        y_pred = (y_prob > thresh).astype(int)
        f1 = f1_score(y_true, y_pred, average="samples")
        if f1 > max_f1:
            max_f1 = f1
            best_thresh = thresh
    y_pred = y_prob > best_thresh
    acc = accuracy_score(y_true, y_pred)
    godbole_acc = godbole_accuracy(y_true, y_pred, "macro")
    godbole_chance_acc = godbole_accuracy(
        y_true, np.ones_like(y_true) * np.mean(y_true), "macro"
    )
    cov_error = coverage_error(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, average="micro")
    lrap = label_ranking_average_precision_score(y_true, y_prob)
    lrap_chance = label_ranking_average_precision_score(
        y_true, np.ones_like(y_true) * np.mean(y_true)
    )
    lrl = label_ranking_loss(y_true, y_prob)
    prec = precision_score(y_true, y_pred, average="micro")
    rec = recall_score(y_true, y_pred, average="micro")
    y_test_inv = 1 - y_true
    y_pred_inv = 1 - y_pred
    spec = recall_score(y_test_inv, y_pred_inv, average="micro")
    mlm = multilabel_confusion_matrix(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob, average="micro")
    ap = average_precision_score(y_true, y_prob, average="micro")
    ap_chance_level = average_precision_score(
        y_true, np.ones_like(y_true) * np.mean(y_true), average="micro"
    )
    fill_rate_pred = np.sum(y_pred) / y_pred.size
    fill_rate = np.sum(y_true) / y_true.size

    print(
        f"acc: {acc:.4f}",
        f"jaccard_index: {godbole_acc:.4f} / {godbole_chance_acc:.4f} (chance)",
        f"lrap: {lrap:.4f} / {lrap_chance:.4f} (chance)",
        f"f1: {f1:.4f}",
        f"lrl: {lrl:.4f}",
        f"rec: {rec:.4f}",
        f"prec: {prec:.4f}",
        f"spec: {spec: 4f}",
        f"cov_err: {cov_error:.4f}",
        f"auroc: {auroc:.4f}",
        f"ap: {ap:.4f} / {ap_chance_level:.4f} (chance)",
        f"fill_rate_pred: {fill_rate_pred:.4f} / {fill_rate:.4f} (true)",
        sep="\n",
        end="\n\n",
    )


def tokenize_text(instance, tokenizer):
    return tokenizer(instance["text"], truncation=True)


def get_scores(results: dict, mlb: MultiLabelBinarizer) -> Generator:
    for result in results:
        score = result["scores"]
        labels = result["labels"]
        scores = [
            score[labels.index(label)] if label in labels else 0
            for label in mlb.classes
        ]
        yield scores


def main():
    df = load_data(LABELLED_CSV, ARTICLES_DIR, use_original_text=True)
    classes = [x.replace("-", " ") for x in df.columns[2:-1].to_list()]
    dataset = Dataset.from_dict(
        get_dict(df),
        features=Features(
            {
                "text": Value("string"),
                "binary_targets": Sequence(Value("int32")),
                "targets": Sequence(ClassLabel(num_classes=8, names=list(range(8)))),
                "labels": Sequence(ClassLabel(names=classes)),
            }
        ),
    )
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        dtype=torch.bfloat16,
        device="cuda",
        fp16=True,
    )
    candidate_labels = list(map(lambda x: x.replace(" ", "-"), classes))
    results = classifier(
        dataset["text"], candidate_labels=candidate_labels, multi_label=True
    )
    mlb = MultiLabelBinarizer(classes=candidate_labels)
    sample_labels = df[df.columns[2:-1]].apply(
        lambda x: list(df.columns[2:-1][x == 1]), axis=1
    )
    # label_to_dashed_labels = {label: label.replace(" ", "-") for label in classes}
    # dashed_labels_to_labels = {v: k for k, v in label_to_dashed_labels.items()}

    mlb.fit(sample_labels)
    y_scores = np.array(list(get_scores(results, mlb)))
    y_true = np.array(dataset["binary_targets"])
    evaluate(y_true, y_scores)


if __name__ == "__main__":
    main()
