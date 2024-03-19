import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

torch.manual_seed(33)

class GenreDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', max_length=self.max_length, return_tensors='pt', truncation=True)
        output = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)  # Convert labels to float tensor
        }
        return output

def train_model(model, train_loader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

def test_model(model, test_loader):
    model.eval()
    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predicted_probs = torch.sigmoid(outputs)
            predicted_labels = (predicted_probs > 0.5).long()

            all_true_labels.extend(labels.cpu().detach().numpy())
            all_pred_labels.extend(predicted_labels.cpu().detach().numpy())

    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)

    # Accuracy
    accuracy = accuracy_score(all_true_labels, all_pred_labels)

    # Precision, Recall, F1-score
    precision = precision_score(all_true_labels, all_pred_labels, average='macro')
    recall = recall_score(all_true_labels, all_pred_labels, average='macro')
    f1 = f1_score(all_true_labels, all_pred_labels, average='macro')

    return accuracy, precision, recall, f1

    

if __name__ == "__main__":
    df = pd.read_csv('raw.csv')

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    texts = df['Text'].tolist()
    labels = df.drop(columns=['Title', 'Text']).values  # Extract labels

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=33)

    train_dataset = GenreDataset(X_train, y_train, tokenizer)
    test_dataset = GenreDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', problem_type="multi_label_classification", num_labels=len(train_dataset.labels[0])).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for multi-label classification

    train_model(model, train_loader, optimizer, criterion)
    accuracy, precision, recall, f1 = test_model(model, test_loader)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}')