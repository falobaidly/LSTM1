import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = "/home/fahad/LSTM/MSFT_1986-03-13_2025-02-04.csv"
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Use only the 'Close' price for prediction
data = df[['Close']].values

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Define look-back period
look_back = 60
X, Y = [], []

for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i - look_back:i, 0])  # Past 60 days as input
    Y.append(scaled_data[i, 0])  # Next day as output

X, Y = np.array(X), np.array(Y)

# Reshape data for PyTorch (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
Y_tensor = torch.from_numpy(Y).float()

# Split data into training (80%) and testing (20%)
train_size = int(len(X_tensor) * 0.8)
X_train, Y_train = X_tensor[:train_size], Y_tensor[:train_size]
X_test, Y_test = X_tensor[train_size:], Y_tensor[train_size:]

# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Cell state

        out, _ = self.lstm(x, (h0, c0))  # Forward pass through LSTM
        out = self.fc(out[:, -1, :])  # Get last output
        return out

# Instantiate model
model = LSTMModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    loss = criterion(outputs.squeeze(), Y_train)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(X_test).squeeze()

# Convert predictions back to original scale
predictions = predictions.numpy().reshape(-1, 1)
predictions = scaler.inverse_transform(predictions)

# Reverse scaling for actual values
Y_test_actual = Y_test.numpy().reshape(-1, 1)
Y_test_actual = scaler.inverse_transform(Y_test_actual)

# Plot results
plt.figure(figsize=(14, 6))
plt.plot(df.index[train_size + look_back:], Y_test_actual, label="Actual Price", color='blue')
plt.plot(df.index[train_size + look_back:], predictions, label="Predicted Price", color='red')
plt.title("Microsoft Stock Price Prediction (LSTM - PyTorch)")
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.ion()
plt.show()
