import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from dataset.featuredataset import FeatureDataset
from models.genreclassifier import AttentionGenreClassifier, GenreClassifier

torch.manual_seed(33)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_EPOCHS = 100
USE_ATTENTION = True
torch.set_default_device(DEVICE)


def encode_labels(categories):
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encode the 'Category' column
    encoded_categories = label_encoder.fit_transform(categories)

    # Convert the encoded categories to PyTorch tensor
    categories_tensor = torch.tensor(encoded_categories, dtype=torch.long).to(DEVICE)

    return categories_tensor, label_encoder

# Prepare the dataset and data loaders
def prepare_data(features, labels, batch_size=4):
    dataset = FeatureDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


# Define training function
def train_model(
    model, train_loader, criterion, optimizer, scheduler=None, num_epochs=10
):
    model.train()
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            running_loss = 0.0
            with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                optimizer.zero_grad()
                inputs = inputs.unsqueeze(2)  # Add a channel dimension
                outputs = model(inputs)
                outputs = F.softmax(outputs, dim=1)
                loss = criterion(outputs, labels)

            # Backward and optimize
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step(loss)
            running_loss += loss.item() * inputs.size(0)
        running_loss /= len(train_loader.dataset)

        val_acc, val_loss = test_model(model, test_loader, criterion=criterion)

        print(
            f"Epoch {epoch+1},",
            f"train_loss: {running_loss:.4e},",
            f"val_loss: {val_loss:.4e},",
            f"val_acc: {val_acc:.4f},",
            f"LR: {scheduler.get_last_lr()[0]:.4e}",
        )


def test_model(model, test_loader, verbose=False, criterion=nn.CrossEntropyLoss()):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.unsqueeze(2)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            if verbose:
                print(predicted, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item() * inputs.size(0)
    accuracy = 100 * correct / total
    loss /= total
    if verbose:
        print("Accuracy on the test set: {:.2f}%".format(accuracy))

    return accuracy, loss


if __name__ == "__main__":
    df_features = pd.read_csv("features.csv")
    df_features = df_features.dropna()

    features = df_features["Features"].apply(
        lambda x: [float(i) for i in x.strip("[]").split()]
    )
    labels = df_features["Category"].values

    # Convert features and labels to tensors
    labels_tensor, _ = encode_labels(labels)
    features_tensor = torch.tensor(features, dtype=torch.float32)

    # Get class weights
    classes = labels_tensor.to("cpu").numpy()
    class_weights = np.bincount(classes, minlength=len(np.unique(classes)))
    # replace 0s with 1s to avoid division by zero
    class_weights[class_weights == 0] = 1
    class_weights = 1 / class_weights
    class_weights /= class_weights.sum()

    # Normalize features
    mean = torch.mean(features_tensor, dim=0)
    std = torch.std(features_tensor, dim=0)
    features_tensor = (features_tensor - mean) / std
    features_tensor = features_tensor.to("cuda")

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features_tensor.cpu().numpy(),
        labels_tensor.cpu().numpy(),
        test_size=0.2,
        random_state=33,
    )

    # Convert the numpy arrays back to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to("cuda")
    X_test = torch.tensor(X_test, dtype=torch.float32).to("cuda")
    y_train = torch.tensor(y_train, dtype=torch.long).to("cuda")
    y_test = torch.tensor(y_test, dtype=torch.long).to("cuda")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Define model parameters
    input_size = features_tensor.shape[1]
    num_outputs = len(labels_tensor.unique())  # Number of unique genres
    print(f"in: {input_size}, out: {num_outputs}")

    # Initialize model, criterion, and optimizer
    model = AttentionGenreClassifier(
        input_size,
        num_outputs,
        [2 ** (i + 6) for i in range(7)],
        3,
        0.2,
        use_attention=USE_ATTENTION,
    ).to(DEVICE)
    # model = GenreClassifier(input_size, num_outputs).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    )
    optimizer = optim.AdamW(model.parameters(), lr=3e-3)

    # Prepare data loaders for training and testing sets
    train_loader = prepare_data(X_train, y_train, batch_size=BATCH_SIZE)
    test_loader = prepare_data(X_test, y_test, batch_size=BATCH_SIZE)

    # Prepare the learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=1 / np.sqrt(10),
        patience=int(math.ceil(len(train_loader.dataset) / BATCH_SIZE)) * 16,
    )

    # Train the model
    train_model(
        model, train_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS
    )

    # Test the model
    test_model(model, test_loader, verbose=True)
