import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

# Check if CUDA (GPU support) is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load data
data = pd.read_csv('data/train_logs.csv')  # replace with your file path
scores = pd.read_csv("data/train_scores.csv")["score"].to_numpy()  # replace with your scores list

# Encode categorical features
categorical_features = ['activity', 'down_event', 'up_event', 'text_change']
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Normalize numerical features
numerical_features = ['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']
scaler = MinMaxScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Group data by id and create sequences
grouped = data.groupby('id')
sequences = [group.drop(['id'], axis=1).values for _, group in grouped]

# Convert sequences to PyTorch tensors and pad
padded_sequences = pad_sequence([torch.tensor(seq[-500 * j:]).float() for seq in sequences], batch_first=True)

# Prepare target variable (scores)
target = torch.tensor(scores).float()

# Split data into training and test sets
train_idx, test_idx = train_test_split(range(len(padded_sequences)), test_size=0.2, random_state=42)
X_train, X_test = padded_sequences[train_idx], padded_sequences[test_idx]
y_train, y_test = target[train_idx], target[test_idx]

# Create DataLoader
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, shuffle=True, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# Define LSTM model using nn.Sequential
input_size = X_train.size(2)  # Number of features
hidden_size = 300
output_size = 1

model = nn.Sequential(
    nn.LSTM(input_size, hidden_size, batch_first=True),
    nn.Flatten(),
    nn.Linear(hidden_size, output_size)
)
model = model.to(device)


# Function to extract LSTM output
def lstm_output(x):
    lstm_out, _ = model[0](x)
    return lstm_out[:, -1, :]


# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train_model():
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        lstm_out = lstm_output(X_batch)
        y_pred = model[1:](lstm_out)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        print(f"loss: {loss}")


# Train the model
for epoch in range(20):
    print(f"epoch: {epoch}")
    train_model()


# Evaluate the model
def evaluate_model():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            lstm_out = lstm_output(X_batch)
            y_pred = model[1:](lstm_out)
            loss = criterion(y_pred.squeeze(), y_batch)
            total_loss += loss.item()
    return total_loss / len(test_loader)


test_loss = evaluate_model()
print(f'Test Loss: {test_loss}')
