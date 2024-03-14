import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

torch.manual_seed(33)

def encode_labels(categories):
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encode the 'Category' column
    encoded_categories = label_encoder.fit_transform(categories)

    # Convert the encoded categories to PyTorch tensor
    categories_tensor = torch.tensor(encoded_categories, dtype=torch.long)

    return categories_tensor.to('cuda')

# Define the neural network architecture
import torch.nn.functional as F

class GenreClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GenreClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define a custom dataset class
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Prepare the dataset and data loaders
def prepare_data(features, labels, batch_size=4):
    dataset = FeatureDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader

# Define training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.unsqueeze(2)  # Add a channel dimension
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.unsqueeze(2)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy on the test set: {:.2f}%'.format(accuracy))

if __name__ == "__main__":
    df_features = pd.read_csv('features.csv')
    df_features =df_features.dropna()

    features = df_features['Features'].apply(
        lambda x: [float(i) for i in x.strip('[]').split()])
    labels = df_features['Category'].values

    # Convert features and labels to tensors
    labels_tensor = encode_labels(labels)
    features_tensor = torch.tensor(features, dtype=torch.float32)

    # Normalize features
    mean = torch.mean(features_tensor, dim=0)
    std = torch.std(features_tensor, dim=0)
    features_tensor = (features_tensor - mean) / std
    features_tensor = features_tensor.to('cuda')
    

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_tensor.cpu(
    ).numpy(), labels_tensor.cpu().numpy(), test_size=0.2, random_state=33)

    # Convert the numpy arrays back to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to('cuda')
    X_test = torch.tensor(X_test, dtype=torch.float32).to('cuda')
    y_train = torch.tensor(y_train, dtype=torch.long).to('cuda')
    y_test = torch.tensor(y_test, dtype=torch.long).to('cuda')
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Define model parameters
    input_size = features_tensor.shape[1]
    num_classes = len(labels_tensor.unique())  # Number of unique genres

    # Initialize model, criterion, and optimizer
    model = GenreClassifier(input_size, num_classes).to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Prepare data loaders for training and testing sets
    train_loader = prepare_data(X_train, y_train)
    test_loader = prepare_data(X_test, y_test)

    # Train the model
    train_model(model, train_loader, criterion, optimizer)

    # Test the model
    test_model(model, test_loader)
