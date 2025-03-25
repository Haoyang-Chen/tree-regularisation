from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report
from torch import nn
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.tree import DecisionTreeClassifier, plot_tree

from utils_EXAI import post_pruning


class CSVDataset(Dataset):
    def __init__(self, csv_file, target_column):
        self.data = pd.read_csv(csv_file)
        self.features = self.data.drop(columns=[target_column]).values
        self.target = self.data[target_column].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.target[idx], dtype=torch.float32)
        return x, y

class Net(nn.Module):
    def __init__(self, input_dim=5):
        super(Net, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.feed_forward(x)

def evaluate_metrics(y_true, y_pred):
    """
    Calculate and return precision, recall, F1-score, ROC-AUC, confusion matrix, and classification report.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred)

    return accuracy, precision, recall, f1, roc_auc, cm, cr

def print_metrics(accuracy, precision, recall, f1, roc_auc, cm, cr, dataset_name):
    """
    Prints the performance metrics in a structured format.
    """
    print(f"\n{'='*10} {dataset_name} Performance Metrics {'='*10}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    print("="*50)

import torch

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # Perform testing after training is complete
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.where(outputs > 0.5, 1, 0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy, precision, recall, f1, roc_auc, cm, cr = evaluate_metrics(all_labels, all_preds)
    print_metrics(accuracy, precision, recall, f1, roc_auc, cm, cr, "Test")

    return model

def get_nn_predictions(model, data_loader, device):
    model.eval()
    predictions = []
    features = []
    with torch.no_grad():
        for inputs, _ in data_loader:  # Ignore true labels, only need X
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.where(outputs > 0.5, 1, 0).cpu().numpy().flatten()
            predictions.extend(preds)
            features.extend(inputs.cpu().numpy())  # Store X values

    return features, predictions  # X remains the same, Y is replaced with NN predictions


if __name__ == '__main__':
    device = 'mps'

    dataset_dir = os.path.join('dataset', 'adult_income')
    train_csv = os.path.join(dataset_dir, 'train_data.csv')
    val_csv = os.path.join(dataset_dir, 'val_data.csv')
    test_csv = os.path.join(dataset_dir, 'test_data.csv')

    data_train_loader = DataLoader(CSVDataset(train_csv, 'income'), batch_size=32, shuffle=True)
    data_val_loader = DataLoader(CSVDataset(val_csv, 'income'), batch_size=32, shuffle=False)
    data_test_loader = DataLoader(CSVDataset(test_csv, 'income'), batch_size=32, shuffle=False)

    input_dim = data_train_loader.dataset[0][0].shape[0]
    model = Net(input_dim)
    num_positives = sum(data_train_loader.dataset.target)
    num_negatives = len(data_train_loader.dataset) - num_positives
    pos_weight = torch.tensor(num_negatives / num_positives)
    print(pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20

    trained_model = train_model(model, data_train_loader, data_val_loader, criterion, optimizer, num_epochs, device)

    X_train, y_train_nn = get_nn_predictions(trained_model, data_train_loader, device)

    ccp_alpha = post_pruning(X_train, y_train_nn)
    # Train the Decision Tree on (X_train, y_train_nn)
    dt_model = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha, min_samples_leaf=50)  # You can tune max_depth
    dt_model.fit(X_train, y_train_nn)

    plt.figure(figsize=(16, 10))  # Increase figure size

    # Plot the decision tree with larger font sizes
    plot_tree(
        dt_model,
        filled=True,
        feature_names=["age","edu","CapitalGain","CapitalLoss","hours"],
        class_names=[str(c) for c in dt_model.classes_],
        fontsize=12,  # Increase font size
        proportion=True  # Normalize box sizes
    )

    plt.title("Decision Tree Visualization")
    plt.show()


