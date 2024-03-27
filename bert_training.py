"""This script trains a BERT-based classifier on the multi-label dataset of articles.
"""

import math
import os

import numpy as np
import seaborn as sns
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
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
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertTokenizer

from dataset.textdataset import ArticleDataset
from metrics.auc import godbole_accuracy, k_fold_roc_curve
from models.bert_classifier import BertWithLinearClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
BATCH_SIZE = 4  # Batch size for training
NUM_EPOCHS = 1  # Number of epochs to train
MAX_LENGTH = 512  # Maximum length of the input sequence
NUM_FOLDS = 5  # Number of folds for cross-validation
THRESHOLD = 0.5  # Threshold for binary classification
TRAIN = False  # Train the model or evaluate it


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    verbose: bool = False,
    criterion: nn.Module = nn.CrossEntropyLoss(),
) -> dict:
    """Test the given model on the provided data loader.

    Args:
        model (nn.Module): Model to be tested.
        test_loader (DataLoader): Data loader for the test set.
        verbose (bool, optional): Prints additional metrics. Defaults to False.
        criterion (nn.Module, optional): Loss function. Defaults to nn.CrossEntropyLoss().

    Returns:
        dict: Dictionary containing the evaluation metrics.
    """
    model.eval()
    y_true = []
    y_prob = []
    y_pred = []
    with torch.no_grad():
        iterator = tqdm(
            test_loader,
            desc="Test batches",
            position=1,
            leave=False,
            total=int(math.ceil(len(test_loader.dataset) / BATCH_SIZE)),
        )
        # For each batch
        for inputs, labels in iterator:
            squeeze_dim = 1 if len(inputs["input_ids"].shape) == 3 else 0
            inputs = {
                key: value.to(DEVICE).squeeze(squeeze_dim)
                for key, value in inputs.items()
                if key != "label"
            }
            labels = labels.float().to(DEVICE)
            outputs = model(inputs)
            predicted = outputs > THRESHOLD
            # Append the predictions and labels
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            y_prob.extend(outputs.cpu().numpy())

    # Convert the lists to numpy arrays
    y_test = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)

    loss = criterion(torch.tensor(y_prob), torch.tensor(y_test)).item()

    # Calculate classification metrics
    # Get the best threshold
    best_thresh = 0
    max_f1 = 0.0
    for thresh in sorted(y_prob.flatten()):
        y_pred = (y_prob > thresh).astype(int)
        f1 = f1_score(y_true, y_pred, average="samples")
        if f1 > max_f1:
            max_f1 = f1
            best_thresh = thresh
    y_pred = (y_prob > best_thresh).astype(int)

    # Calculate metrics
    chance_level = np.mean(y_test)
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
    fill_rate_pred = np.sum(y_pred) / y_pred.size
    fill_rate = np.sum(y_test) / y_test.size

    # Print metrics if verbose
    if verbose:
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
            f"ap: {ap:.4f} / {chance_level:.4f} (chance)",
            f"fill_rate_pred: {fill_rate_pred:.4f} / {fill_rate:.4f} (true)",
            sep="\n",
            end="\n\n",
        )

    return {
        "acc": acc,
        "jaccard_index": godbole_acc,
        "rec": rec,
        "prec": prec,
        "f1": f1,
        "spec": spec,
        "lrap": lrap,
        "lrl": lrl,
        "cov_err": cov_error,
        "auroc": auroc,
        "ap": ap,
        "fill_rate_preds": fill_rate_pred,
        "y_pred": y_pred,
        "y": y_test,
        "y_prob": y_prob,
        "mlm": mlm,
        "val_loss": loss,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler = None,
    num_epochs: int = 10,
) -> None:
    """
    Trains the given model using the provided data loaders, criterion, optimizer, and scheduler (optional).

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): The data loader for the training set.
        test_loader (DataLoader): The data loader for the test/validation set.
        criterion (nn.Module): The loss function used for training.
        optimizer (optim.Optimizer): The optimizer used for updating the model's parameters.
        scheduler (optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler (default: None).
        num_epochs (int, optional): The number of training epochs (default: 10).

    Returns:
        None
    """
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None
    iterator = tqdm(range(num_epochs), desc="Epochs", position=0, leave=True)
    for _ in iterator:
        model.train()
        for inputs, labels in tqdm(
            train_loader,
            desc="Train batches",
            leave=False,
            position=1,
            total=int(math.ceil(len(train_loader.dataset) / BATCH_SIZE)),
        ):
            running_loss = 0.0

            # Mixed Precision
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                optimizer.zero_grad()
                squeeze_dim = 1 if len(inputs["input_ids"].shape) == 3 else 0
                inputs = {
                    key: value.to(DEVICE).squeeze(squeeze_dim)
                    for key, value in inputs.items()
                    if key != "label"
                }
                labels = labels.float().to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * inputs["input_ids"].size(0)

        running_loss /= len(train_loader.dataset)
        metrics = test_model(model, test_loader, criterion=criterion)
        if scheduler is not None:
            scheduler.step(metrics["val_loss"])

        line = (
            f"Train Loss: {running_loss:.4f}, "
            + f"Val Loss: {metrics['val_loss']:.4f}, "
            + f"Val Acc: {metrics['acc']:4f}%"
        )
        iterator.write(line)
        iterator.set_postfix_str(f"LR: {scheduler.get_last_lr()[0]:.4e}")


def main(train: bool):
    """Runs the main training loop for the BERT-based classifier.

    Args:
        train (bool): Whether to train the model or evaluate it.
    """
    sns.set_theme("paper", "whitegrid")

    # Setup the dataset and cross-validation
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = ArticleDataset(
        "./articles", "./multi_label_dataset.csv", tokenizer, MAX_LENGTH
    )
    mskf = MultilabelStratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    results = {}
    for fold, (train_index, test_index) in enumerate(
        mskf.split(dataset.articles, dataset.targets)
    ):
        # Create the data loaders
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        test_dataset = torch.utils.data.Subset(dataset, test_index)
        if train:
            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
            )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4
        )
        # Initialize the model
        model = BertWithLinearClassifier(
            len(dataset.categories), MAX_LENGTH, 0.2, "bert-base-uncased"
        )
        model.to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        if train:
            # Train the model
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=np.sqrt(0.1)
            )
            train_model(
                model,
                train_loader,
                test_loader,
                criterion,
                optimizer,
                scheduler,
                num_epochs=NUM_EPOCHS,
            )
        else:
            # Load the model
            model.load_state_dict(torch.load(f"./ckpts/bert/{fold}/model.pth"))

        # Evaluate the model
        res = test_model(model, test_loader, True, criterion)
        for key, value in res.items():
            if key not in results:
                results[key] = []
            results[key].append(value)

        if train:
            # Save the model
            if not os.path.exists(f"./ckpts/bert/{fold}"):
                os.makedirs(f"./ckpts/bert/{fold}")
            torch.save(model.state_dict(), f"./ckpts/bert/{fold}/model.pth")

    y = np.concatenate(results["y"])
    chance_level = np.mean(y)
    for key, value in results.items():
        if key in ["y_pred", "y", "y_prob", "mlm"]:
            continue
        if key in ["jaccard_index", "lrap", "ap"]:
            print(
                f"{key}:",
                f"{np.mean(value):.4f} ± {np.std(value):.4f}",
                "/",
                f"{chance_level:.4f} (chance level)",
            )
        else:
            print(f"{key}: {np.mean(value):.4f} ± {np.std(value):.4f}")

    # Plot ROC and PRC
    model_outputs = []
    for i in range(NUM_FOLDS):
        fold_results = {
            "y_pred": results["y_pred"][i],
            "y": results["y"][i],
            "y_prob": results["y_prob"][i],
        }
        model_outputs.append(fold_results)

    fig = k_fold_roc_curve(
        model_outputs,
        "bert-base-uncased",
        8,
        average="weighted",
    )
    fig.savefig("./ckpts/bert/bert_roc_curve.png")


if __name__ == "__main__":
    main(TRAIN)
