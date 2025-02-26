import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ✅ Force CPU usage to prevent GPU memory issues
device = torch.device("cpu")

# ✅ Load the dataset
file_path = "/home/fahad/LSTM/MSFT_1986-03-13_2025-02-04.csv"
df = pd.read_csv(file_path)

# ✅ Convert 'Date' column to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# ✅ Calculate percentage change (better scaling)
df['Close_pct'] = df['Close'].pct_change()
df.dropna(inplace=True)  # Remove NaN values from pct_change

# ✅ Use percentage change for normalization
scaler = MinMaxScaler(feature_range=(-1, 1))  # Allow negative values
scaled_data = scaler.fit_transform(df[['Close_pct']].values)

# ✅ Reduce Look-Back Period to Lower Memory Usage
look_back = 50  # Change from 200 to 50

X, Y = [], []

for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i - look_back:i, 0])  # Past 50 days as input
    Y.append(scaled_data[i, 0])  # Next day percentage change as output

X, Y = np.array(X), np.array(Y)

# ✅ Reshape data for PyTorch (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# ✅ Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float().to(device)
Y_tensor = torch.from_numpy(Y).float().to(device)

# ✅ Split data into training (80%) and testing (20%)
train_size = int(len(X_tensor) * 0.8)
X_train, Y_train = X_tensor[:train_size], Y_tensor[:train_size]
X_test, Y_test = X_tensor[train_size:], Y_tensor[train_size:]

# ✅ Define an Optimized LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):  # Reduced size
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # Hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # Cell state

        out, _ = self.lstm(x, (h0, c0))  # Forward pass through LSTM
        out = self.fc(out[:, -1, :])  # Get last output
        return out

# ✅ Instantiate the improved model
model = LSTMModel().to(device)

# ✅ Define a better loss function and optimizer
criterion = nn.L1Loss()  # Better for stock price prediction
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# ✅ Reduce Batch Size to Avoid Memory Overload
epochs = 50  # Reduce from 100 to 50
batch_size = 16  # Reduce from 32 to 16

for epoch in range(epochs):
    model.train()
    
    for i in range(0, len(X_train), batch_size):
        x_batch = X_train[i:i+batch_size]
        y_batch = Y_train[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# ✅ Make predictions
model.eval()
with torch.no_grad():
    predictions = model(X_test).squeeze()

# ✅ Convert percentage change predictions back to actual price
predictions = predictions.numpy().reshape(-1, 1)
predictions = scaler.inverse_transform(predictions)

# ✅ Reverse scaling for actual values
Y_test_actual = Y_test.numpy().reshape(-1, 1)
Y_test_actual = scaler.inverse_transform(Y_test_actual)

# ✅ Calculate actual predicted stock prices
predicted_prices = df['Close'].iloc[train_size+look_back-1] * (1 + np.cumsum(predictions))
actual_prices = df['Close'].iloc[train_size+look_back-1] * (1 + np.cumsum(Y_test_actual))

# ✅ Save predictions and actual data to a CSV file
comparison_df = pd.DataFrame({
    "Date": df.index[train_size + look_back:],  # Corresponding dates
    "Actual Price": actual_prices.flatten(),
    "Predicted Price": predicted_prices.flatten()
})

comparison_df.to_csv("stock_predictions_fixed.csv", index=False)
print("\n✅ Predictions saved to 'stock_predictions_fixed.csv'.")

# ✅ Plot results
plt.figure(figsize=(14, 6))
plt.plot(df.index[train_size + look_back:], actual_prices, label="Actual Price", color='blue')
plt.plot(df.index[train_size + look_back:], predicted_prices, label="Predicted Price", color='red')
plt.title("Microsoft Stock Price Prediction (LSTM - PyTorch) - Optimized")
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.savefig("stock_prediction_fixed.png")
plt.show()
