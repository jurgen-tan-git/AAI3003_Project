import math
import os

import numpy as np
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertTokenizer

from dataset.textdataset import ArticleDataset
from models.bert_classifier import BertWithLinearClassifier, BertWithAttentionClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 1
MAX_LENGTH = 512


def test_model(model, test_loader, verbose=False, criterion=nn.CrossEntropyLoss()):
    model.eval()
    correct = 0
    total = 0
    loss = 0
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
            predicted = outputs > 0.5
            if verbose:
                iterator.write(f"{predicted}, {labels}")
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item() * inputs["input_ids"].size(0)
    accuracy = 100 * correct / total
    loss /= total
    if verbose:
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Loss: {loss:.4f}")

    return accuracy, loss


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
            with torch.autocast(device_type=DEVICE, dtype=torch.float16):
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
        val_acc, val_loss = test_model(model, test_loader, criterion=criterion)
        if scheduler is not None:
            scheduler.step(val_loss)

        iterator.write(
            f"Train Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:3.2f}%"
        )
        iterator.set_postfix_str(f"LR: {scheduler.get_last_lr()[0]:.4e}")


def main():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = ArticleDataset(
        "./articles", "./multi_label_dataset.csv", tokenizer, MAX_LENGTH
    )
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=generator
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    model = BertWithAttentionClassifier(
        "bert-base-uncased", len(dataset.categories), MAX_LENGTH, 100
    )
    # model = BertWithLinearClassifier(
    #     len(dataset.categories), MAX_LENGTH, 0.2, "bert-base-uncased"
    # )
    model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
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

    test_model(model, test_loader, True, criterion)

    # Save the model
    if not os.path.exists("./ckpts"):
        os.makedirs("./ckpts")
    torch.save(model.state_dict(), "./ckpts/model.pth")


if __name__ == "__main__":
    main()
