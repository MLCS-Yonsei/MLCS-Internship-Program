#==========================================#
# Title:  Stock prices prediction with LSTM
# Author: Jaewoong Han
# Date:   2024-06-27
#==========================================#
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

seq_length = 60
epochs = 20
hidden_size = 50
num_layers = 2
output_size = 1
batch_size = 1
lr = 0.0001
ticker = "SBUX"

"""
Step1: Preprocess Datasets
"""
data = yf.download(ticker, start="2020-01-01", end="2023-12-31")
data = data['Close'].values.reshape(-1, 1)
input_size = data.shape[1]

scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

X, y = create_sequences(data_normalized, seq_length)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""
Step2: Define LSTM Network
"""
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, output_size)
    
    def forward(self, x):
        hidden_cell = (torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device),
                            torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))

        out, _ = self.lstm(x, hidden_cell)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.fc2(out)
        return out

"""
Step3: Define model, criterion, optimizer and train
"""
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
for epoch in range(epochs):
    train_loss = 0.0
    for sequences, targets in train_loader:
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    print(f'Epoch {str(epoch+1).zfill(2)}/{epochs}, Loss: {train_loss/len(train_loader):.7f}')

"""
Step4: Evaluate model
"""
model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for sequences, targets in test_loader:
        output = model(sequences)
        if output.size(0) == 1:
            predictions.append(output.item())
            actuals.append(targets.item())
        else:
            predictions.extend(output.squeeze().tolist())
            actuals.extend(targets.squeeze().tolist())


        
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()