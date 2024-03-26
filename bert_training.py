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
from models.bert_classifier import BertWithAttentionClassifier, BertWithLinearClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 1
MAX_LENGTH = 512
NUM_FOLDS = 5
THRESHOLD = 0.5
TRAIN = False


def test_model(model, test_loader, verbose=False, criterion=nn.CrossEntropyLoss()):
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
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            y_prob.extend(outputs.cpu().numpy())

    y_test = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)

    loss = criterion(torch.tensor(y_prob), torch.tensor(y_test)).item()

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    godbole_acc = godbole_accuracy(y_test, y_pred, "macro")
    godbole_chance_acc = godbole_accuracy(y_test, y_test, "macro")
    cov_error = coverage_error(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, average="micro")
    lrap = label_ranking_average_precision_score(y_test, y_prob)
    lrap_chance = label_ranking_average_precision_score(y_test, y_test)
    lrl = label_ranking_loss(y_test, y_prob)
    prec = precision_score(y_test, y_pred, average="micro")
    rec = recall_score(y_test, y_pred, average="micro")
    y_test_inv = 1 - y_test
    y_pred_inv = 1 - y_pred
    spec = recall_score(y_test_inv, y_pred_inv, average="micro")
    mlm = multilabel_confusion_matrix(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob, average="micro")
    ap = average_precision_score(y_test, y_prob, average="micro")
    ap_chance_level = average_precision_score(y_test, y_test, average="micro")
    fill_rate_pred = np.sum(y_pred) / y_pred.size
    fill_rate = np.sum(y_test) / y_test.size

    if verbose:
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

        iterator.write(
            f"Train Loss: {running_loss:.4f}, Val Loss: {metrics["val_loss"]:.4f}, Val Acc: {metrics["acc"]:4f}%"
        )
        iterator.set_postfix_str(f"LR: {scheduler.get_last_lr()[0]:.4e}")


def main(train: bool):
    sns.set_theme("paper", "whitegrid")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = ArticleDataset(
        "./articles", "./multi_label_dataset.csv", tokenizer, MAX_LENGTH
    )
    mskf = MultilabelStratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    results = {}
    # model = BertWithAttentionClassifier(
    #     "bert-base-uncased", len(dataset.categories), MAX_LENGTH, 100
    # )
    for fold, (train_index, test_index) in enumerate(
        mskf.split(dataset.articles, dataset.targets)
    ):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        test_dataset = torch.utils.data.Subset(dataset, test_index)
        if train:
            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
            )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4
        )
        model = BertWithLinearClassifier(
            len(dataset.categories), MAX_LENGTH, 0.2, "bert-base-uncased"
        )
        model.to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        if train:
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
            model.load_state_dict(torch.load(f"./ckpts/bert/{fold}/model.pth"))

        res = test_model(model, test_loader, True, criterion)
        for key, value in res.items():
            if key not in results:
                results[key] = []
            results[key].append(value)

        # Save the model
        if train:
            if not os.path.exists(f"./ckpts/bert/{fold}"):
                os.makedirs(f"./ckpts/bert/{fold}")
            torch.save(model.state_dict(), f"./ckpts/bert/{fold}/model.pth")

    y = np.concatenate(results["y"])
    for key, value in results.items():
        if key in ["y_pred", "y", "y_prob", "mlm"]:
            continue
        match(key):
            case "jaccard_index":
                print(f"{key}: {np.mean(value):.4f} ± {np.std(value):.4f} / {godbole_accuracy(y, np.ones_like(y), "macro"):.4f} (chance level)")
            case "lrap":
                print(f"{key}: {np.mean(value):.4f} ± {np.std(value):.4f} / {label_ranking_average_precision_score(y, np.ones_like(y) * np.mean(y)):.4f} (chance level)")
            case "ap":
                print(f"{key}: {np.mean(value):.4f} ± {np.std(value):.4f} / {average_precision_score(y, np.ones_like(y) * np.mean(y)):.4f} (chance level)")
            case _:
                print(f"{key}: {np.mean(value):.4f} ± {np.std(value):.4f}")

    model_outputs = []
    for i in range(NUM_FOLDS):
        fold_results = {
            "y_pred": results["y_pred"][i],
            "y": results["y"][i],
            "y_prob": results["y_prob"][i],
        }
        model_outputs.append(fold_results)

    fig = k_fold_roc_curve(model_outputs, "bert-base-uncased", 8, average="weighted", )
    fig.savefig("./ckpts/bert/bert_roc_curve.png")

if __name__ == "__main__":
    main(TRAIN)
