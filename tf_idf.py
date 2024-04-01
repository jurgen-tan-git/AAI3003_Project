"""This script runs various machine learning models with 5-fold cross validation
on tf-idf features and evaluates them using various metrics.
"""
import math
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             coverage_error, f1_score,
                             label_ranking_average_precision_score,
                             label_ranking_loss, multilabel_confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset.tfidf import TfIdfDataset
from dataset.transformers_dataset import load_data
from metrics.auc import godbole_accuracy, k_fold_roc_curve
from models.learner import (AccuracyCallback, F1Callback, Learner,
                            ModelProgressCallback, PlotGraphCallback,
                            SaveModelCallback)
from models.tfidf_attention import TfIdfDense

NUM_FOLDS = 5
NUM_HEADS = 12
BATCH_SIZE = 32
NUM_EPOCHS = 50
THRESHOLD = 0.85
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else DEVICE


def train_and_eval(
    mskf: MultilabelStratifiedKFold, model_class: Callable, X: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """Train and evaluate a model using MultilabelStratifiedKFold cross-validation.

    :param mskf: MultilabelStratifiedKFold object
    :type mskf: MultilabelStratifiedKFold
    :param model_class: Class of the model to be trained
    :type model_class: Callable
    :param X: Dataset of input features
    :type X: np.ndarray
    :param y: Dataset of target labels
    :type y: np.ndarray
    :return: Dictionary of metrics
    :rtype: dict[str, float]
    """
    print(f"Model type: {model_class.__name__}")

    # Initialize lists to store metrics
    accs, lraps, f1s, lrls, precs, recs, specs, cov_errs, aurocs, aps, godbole_accs, fill_rate_preds = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []
    )
    y_preds, y_probs, y_tests = [], [], []

    for i, (train_idx, test_idx) in enumerate(mskf.split(X, y)):
        # Split the dataset into training and testing sets
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Initialize and fit the model to training data
        model = model_class()
        model.fit(X_train, y_train)

        # Retrieve the predicted probabilities and labels
        y_prob = np.array(model.predict_proba(X_test))
        if not model_class.__name__ == "MLPClassifier":
            y_prob = y_prob[:, :, 1].reshape(len(y_test), -1)
        y_pred = model.predict(X_test)

        # Calculate metrics
        chance_level = y_test.sum() / y_test.size
        acc = accuracy_score(y_test, y_pred)
        godbole_acc = godbole_accuracy(y_test, y_pred, "macro")
        cov_error = coverage_error(y_test, y_prob)
        f1 = f1_score(y_test, y_pred, average="micro")
        lrap = label_ranking_average_precision_score(y_test, y_prob)
        lrl = label_ranking_loss(y_test, y_prob)
        prec = precision_score(y_test, y_pred, average="micro")
        rec = recall_score(y_test, y_pred, average="micro")
        y_test_inv = 1 - y_test
        y_pred_inv = 1 - y_pred
        spec = recall_score(y_test_inv, y_pred_inv, average="micro")
        mlm = multilabel_confusion_matrix(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_prob, average="micro")
        ap = average_precision_score(y_test, y_prob, average="micro")

        # Append metrics to lists
        accs.append(acc)
        godbole_accs.append(godbole_acc)
        lraps.append(lrap)
        f1s.append(f1)
        lrls.append(lrl)
        precs.append(prec)
        recs.append(rec)
        specs.append(spec)
        cov_errs.append(cov_error)
        y_preds.append(y_pred)
        y_probs.append(y_prob)
        y_tests.append(y_test)
        aurocs.append(auroc)
        aps.append(ap)
        fill_rate = np.sum(y_test) / y_test.size
        fill_rate_pred = np.sum(y_pred) / y_pred.size
        fill_rate_preds.append(fill_rate_pred)

        # Print metrics
        print(f"Fold: {i + 1}")
        print(
            f"acc: {acc:.4f}",
            f"jaccard_index: {godbole_acc:.4f} / {chance_level:.4f} (chance)",
            f"lrap: {lrap:.4f} / {chance_level:.4f} (chance)",
            f"f1: {f1:.4f}",
            f"lrl: {lrl:.4f}",
            f"rec: {rec:.4f}",
            f"prec: {prec:.4f}",
            f"spec: {spec: 4f}",
            f"cov_err: {cov_error:.4f}",
            f"auroc: {auroc:.4f}",
            f"ap: {ap:.4f}",
            f"fill_rate_pred: {fill_rate_pred:.4f} / {fill_rate:.4f} (true)",
            sep="\n",
            end="\n\n",
        )

    # Return metrics as a dictionary
    return {
        "acc": accs,
        "jaccard_index": godbole_accs,
        "f1": f1s,
        "rec": recs,
        "prec": precs,
        "spec": specs,
        "lrap": lraps,
        "lrl": lrls,
        "cov_err": cov_errs,
        "auroc": aurocs,
        "ap": aps,
        "fill_rate_preds": fill_rate_preds,
        "y_pred": y_preds,
        "y": y_tests,
        "y_prob": y_probs,
        "mlm": mlm,
    }


def train_and_eval_pytorch(
    mskf: MultilabelStratifiedKFold, model: partial, model_name: str, ds: TfIdfDataset
) -> dict[str, float]:
    """Train and evaluate a model using MultilabelStratifiedKFold cross-validation.

    :param mskf: MultilabelStratifiedKFold object
    :type mskf: MultilabelStratifiedKFold
    :param model: Class of the model to be trained
    :type model: partial
    :param model_name: Name of the model
    :type model_name: str
    :param ds: TfIdfDataset object
    :type ds: TfIdfDataset
    :return: Dictionary of metrics
    :rtype: dict[str, float]
    """
    print(f"Model type: {model_name}")
    # Initialize lists to store metrics
    (
        accs,
        lraps,
        f1s,
        lrls,
        precs,
        recs,
        specs,
        cov_errs,
        aurocs,
        aps,
        godbole_accs,
        fill_rate_preds,
    ) = ([], [], [], [], [], [], [], [], [], [], [], [])
    y_preds, y_probs, y_tests = [], [], []

    for i, (train_idx, test_idx) in enumerate(mskf.split(ds.X, ds.y)):
        # Split the dataset into training and testing sets
        train_ds = Subset(ds, train_idx)
        test_ds = Subset(ds, test_idx)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize and fit the model to training data
        model_instance = model()
        learner = Learner(
            model_instance,
            nn.BCEWithLogitsLoss(),
            torch.device(DEVICE),
            cbs=[
                F1Callback(),
                PlotGraphCallback(),
                SaveModelCallback(
                    model=model_instance,
                    strategy="best",
                    root_dir=f"ckpts/{model_name}/fold_{i}",
                    model_pth="model_best.pth",
                    metric="valid_loss",
                ),
                ModelProgressCallback(["accuracy", "f1"]),
            ],
            metrics=[
                F1Callback(multilabel=True),
                AccuracyCallback(threshold=THRESHOLD, multilabel=True),
            ],
        )
        learner.fit(train_dl, test_dl, NUM_EPOCHS)

        # Retrieve the predicted probabilities and labels
        _, y_prob, y_test = learner.evaluate(test_dl)
        best_thresh = 0
        max_f1 = 0.0
        for thresh in sorted(y_prob.flatten()):
            y_pred = (y_prob > thresh).astype(int)
            f1 = f1_score(y_test, y_pred, average="samples")
            if f1 > max_f1:
                max_f1 = f1
                best_thresh = thresh
        y_pred = (y_prob > best_thresh).astype(int)
        y_test = y_test.astype(int)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        chance_level = y_test.sum() / y_test.size
        godbole_acc = godbole_accuracy(y_test, y_pred, "macro")
        cov_error = coverage_error(y_test, y_prob)
        f1 = f1_score(y_test, y_pred, average="micro")
        lrap = label_ranking_average_precision_score(y_test, y_prob)
        lrl = label_ranking_loss(y_test, y_prob)
        prec = precision_score(y_test, y_pred, average="micro")
        rec = recall_score(y_test, y_pred, average="micro")
        y_test_inv = 1 - y_test
        y_pred_inv = 1 - y_pred
        spec = recall_score(y_test_inv, y_pred_inv, average="micro")
        mlm = multilabel_confusion_matrix(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_prob, average="micro")
        ap = average_precision_score(y_test, y_prob, average="micro")

        # Append metrics to lists
        accs.append(acc)
        godbole_accs.append(godbole_acc)
        lraps.append(lrap)
        f1s.append(f1)
        lrls.append(lrl)
        precs.append(prec)
        recs.append(rec)
        specs.append(spec)
        cov_errs.append(cov_error)
        y_preds.append(y_pred)
        y_probs.append(y_prob)
        y_tests.append(y_test)
        aurocs.append(auroc)
        aps.append(ap)
        fill_rate = np.sum(y_test) / y_test.size
        fill_rate_pred = np.sum(y_pred) / y_pred.size
        fill_rate_preds.append(fill_rate_pred)

        # Print metrics
        print(f"Fold: {i + 1}")
        print(
            f"acc: {acc:.4f}",
            f"jaccard_index: {godbole_acc:.4f} / {chance_level:.4f} (chance)",
            f"lrap: {lrap:.4f} / {chance_level:.4f} (chance)",
            f"f1: {f1:.4f}",
            f"lrl: {lrl:.4f}",
            f"rec: {rec:.4f}",
            f"prec: {prec:.4f}",
            f"spec: {spec: 4f}",
            f"cov_err: {cov_error:.4f}",
            f"auroc: {auroc:.4f}",
            f"ap: {ap:.4f}",
            f"fill_rate_pred: {fill_rate_pred:.4f} / {fill_rate:.4f} (true)",
            sep="\n",
            end="\n\n",
        )

    # Return metrics as a dictionary
    return {
        "acc": accs,
        "jaccard_index": godbole_accs,
        "f1": f1s,
        "rec": recs,
        "prec": precs,
        "spec": specs,
        "lrap": lraps,
        "lrl": lrls,
        "cov_err": cov_errs,
        "auroc": aurocs,
        "ap": aps,
        "fill_rate_preds": fill_rate_preds,
        "y_pred": y_preds,
        "y": y_tests,
        "y_prob": y_probs,
        "mlm": mlm,
    }


def main():
    """Run the main function to train and evaluate models on tf-idf features.
    """

    # Load the dataset and vectorize the text
    sns.set_theme("paper", "whitegrid")
    df = load_data("multi_label_dataset.csv", "articles", False)
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
    X = vectorizer.fit_transform(df["Text"])
    y = df[df.columns[2:-1]].to_numpy()

    # MultilabelStratifiedKFold cross-validation
    mskf = MultilabelStratifiedKFold(
        n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    results = {}
    for model in [RandomForestClassifier, KNeighborsClassifier, DecisionTreeClassifier, MLPClassifier]:
        res = train_and_eval(mskf, model, X, y)
        results[model.__name__] = res

    chance_level = y.sum() / y.size

    # Print metrics
    for model, res in results.items():
        print("\n")
        print("=" * 80)
        print(f"model: {model}")
        print(
            f"acc: {np.mean(res['acc']):.4f} +/- {np.std(res['acc']):.4f}",
            f"jaccard_index: {np.mean(res['jaccard_index']):.4f} +/- {np.std(res['jaccard_index']):.4f} / {chance_level:.4f} (chance level)",
            f"rec: {np.mean(res['rec']):.4f} +/- {np.std(res['rec']):.4f}",
            f"prec: {np.mean(res['prec']):.4f} +/- {np.std(res['prec']):.4f}",
            f"spec: {np.mean(res['spec']):.4f} +/- {np.std(res['spec']):.4f}",
            f"f1: {np.mean(res['f1']):.4f} +/- {np.std(res['f1']):.4f}",
            f"lrap: {np.mean(res['lrap']):.4f} +/- {np.std(res["lrap"]):.4f} / {chance_level:.4f} (chance level)",
            f"lrl: {np.mean(res['lrl']):.4f} +/- {np.std(res['lrl']):.4f}",
            f"cov_err: {np.mean(res['cov_err']):.4f} +/- {np.std(res['cov_err']):.4f}",
            f"auroc: {np.mean(res['auroc']):.4f} +/- {np.std(res['auroc']):.4f}",
            f"ap: {np.mean(res['ap']):.4f} +/- {np.std(res['ap']):.4f} / {chance_level:.4f} (chance level)",
            f"fill_rate_pred: {np.mean(res['fill_rate_preds']):.4f} +/- {np.std(res['fill_rate_preds']):.4f} / {np.sum(y) / y.size:.4f} (true fill rate)",
            "=" * 80,
            sep="\n",
            end="\n",
        )

    # Deep Learning training
    ds = TfIdfDataset("multi_label_dataset.csv", "articles", False)
    PADDED_SIZE = math.ceil(ds.X[0].shape[1] / NUM_HEADS) * NUM_HEADS
    ds.set_padded_shape((0, PADDED_SIZE))

    model_partial = partial(
        TfIdfDense,
        n_inputs=PADDED_SIZE,
        n_outputs=len(ds.df.columns[2:-1]),
        hidden_size=[32, 64, 32],
        dropout=0.2
    )
    res = train_and_eval_pytorch(mskf, model_partial, "TfIdfDense", ds)
    results["TfIdfDense"] = res

    # Print metrics
    for model, res in results.items():
        print("\n")
        print("=" * 80)
        print(f"model: {model}")
        print(
            f"acc: {np.mean(res['acc']):.4f} +/- {np.std(res['acc']):.4f}",
            f"jaccard_index: {np.mean(res['jaccard_index']):.4f} +/- {np.std(res['jaccard_index']):.4f} / {chance_level:.4f} (chance level)",
            f"rec: {np.mean(res['rec']):.4f} +/- {np.std(res['rec']):.4f}",
            f"prec: {np.mean(res['prec']):.4f} +/- {np.std(res['prec']):.4f}",
            f"spec: {np.mean(res['spec']):.4f} +/- {np.std(res['spec']):.4f}",
            f"f1: {np.mean(res['f1']):.4f} +/- {np.std(res['f1']):.4f}",
            f"lrap: {np.mean(res['lrap']):.4f} +/- {np.std(res["lrap"]):.4f} / {chance_level:.4f} (chance level)",
            f"lrl: {np.mean(res['lrl']):.4f} +/- {np.std(res['lrl']):.4f}",
            f"cov_err: {np.mean(res['cov_err']):.4f} +/- {np.std(res['cov_err']):.4f}",
            f"auroc: {np.mean(res['auroc']):.4f} +/- {np.std(res['auroc']):.4f}",
            f"ap: {np.mean(res['ap']):.4f} +/- {np.std(res['ap']):.4f} / {chance_level:.4f} (chance level)",
            f"fill_rate_pred: {np.mean(res['fill_rate_preds']):.4f} +/- {np.std(res['fill_rate_preds']):.4f} / {np.sum(y) / y.size:.4f} (true fill rate)",
            "=" * 80,
            sep="\n",
            end="\n",
        )

    best_model = max(results.items(), key=lambda x: x[1]["f1"])
    print(f"Best model: {best_model[0]}, F1: {np.mean(best_model[1]['f1']):.4f}")

    # Convert results to a DataFrame
    results_for_df = {}
    for model_name, result in results.items():
        if "model" not in results_for_df:
            results_for_df["model"] = []
        results_for_df["model"].append(model_name)
        for metric, values in result.items():
            if metric in ["y_pred", "y", "y_prob", "mlm"]:
                continue
            mean = np.mean(values)
            std = np.std(values)
            if metric not in results_for_df:
                results_for_df[metric] = []
            results_for_df[metric].append(mean)
            if f"{metric}_std" not in results_for_df:
                results_for_df[f"{metric}_std"] = []
            results_for_df[f"{metric}_std"].append(std)

    results_df = pd.DataFrame.from_dict(results_for_df)

    # Display results
    pd.set_option("display.max_columns", None)
    print(results_df)

    # Show ROC and PRC for each model over the folds.
    model_outputs = []
    for model_name, result in results.items():
        current_model_outputs = []
        for i in range(NUM_FOLDS):
            fold_results = {
                "y_pred": result["y_pred"][i],
                "y_prob": result["y_prob"][i],
                "y": result["y"][i],
            }
            current_model_outputs.append(fold_results)
        model_outputs.append(current_model_outputs)
    for i, model_output in enumerate(model_outputs):
        model_name = list(results.keys())[i]
        k_fold_roc_curve(
            model_output,
            model_name,
            len(df.columns[2:-1]),
            "weighted",
        )


if __name__ == "__main__":
    main()
