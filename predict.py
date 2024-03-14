import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

def encode_labels(categories):
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encode the 'Category' column
    encoded_categories = label_encoder.fit_transform(categories)

    # Convert the encoded categories to PyTorch tensor
    categories_tensor = torch.tensor(encoded_categories, dtype=torch.long)

    return categories_tensor.to('cuda')

# Define the neural network architecture
class GenreClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GenreClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=192, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.fc1 = nn.Linear(32 , 128)  # Adjust the input size based on the output size of conv layers and pooling
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)  # Flatten the output of convolutional layers
        x = F.tanh(self.fc1(x))
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
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

# Define training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.unsqueeze(2)  # Add a channel dimension
            outputs = model(inputs)

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
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy on the test set: {:.2f}%'.format(accuracy))

if __name__ == "__main__":
    df_features = pd.read_csv('features.csv')

    features = df_features['Features'].apply(lambda x: [float(i) for i in x.strip('[]').split()])
    labels = df_features['Category'].values

    # Convert features and labels to tensors
    features_tensor = torch.tensor(features, dtype=torch.float32).to('cuda')
    labels_tensor = encode_labels(labels)

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_tensor.cpu().numpy(), labels_tensor.cpu().numpy(), test_size=0.2, random_state=0)

    # Convert the numpy arrays back to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to('cuda')
    X_test = torch.tensor(X_test, dtype=torch.float32).to('cuda')
    y_train = torch.tensor(y_train, dtype=torch.long).to('cuda')
    y_test = torch.tensor(y_test, dtype=torch.long).to('cuda')
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # Define model parameters
    input_size = features_tensor.shape[1]  # Input size is the number of features
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
