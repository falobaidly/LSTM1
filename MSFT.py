import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for TensorFlow

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

# ✅ Use CPU to prevent memory issues
device = torch.device("cpu")

# ✅ Load dataset
file_path = "/home/fahad/LSTM1/combined_data.csv"
df = pd.read_csv(file_path)

# ✅ Encode labels (Spam = 1, Ham = 0)
df['label'] = df['label'].astype(int)

# ✅ Tokenization and Vocabulary Building
def tokenize(text):
    return text.lower().split()

def build_vocab(texts, min_freq=2):  # Reduce vocab size
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {word: i+2 for i, (word, count) in enumerate(counter.items()) if count >= min_freq}
    vocab["<unk>"] = 0  # Unknown tokens
    vocab["<pad>"] = 1  # Padding
    return vocab

vocab = build_vocab(df['text'])
vocab_size = len(vocab) + 2  # Ensure correct vocab size

# ✅ Convert text data to sequences
def text_pipeline(text):
    return [vocab.get(token, vocab["<unk>"]) for token in tokenize(text)]

sequences = [text_pipeline(text) for text in df['text']]

# ✅ Pad sequences
max_len = 40  # Reduce max email length to lower memory usage
padded_sequences = pad_sequence([torch.tensor(seq[:max_len]) for seq in sequences], batch_first=True, padding_value=vocab["<pad>"])

# ✅ Convert to tensors
X = padded_sequences.to(device)
Y = torch.tensor(df['label'].values, dtype=torch.float32).to(device)

# ✅ Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ✅ Create DataLoader for batching
batch_size = 8  # Reduce batch size further to lower memory usage
train_data = TensorDataset(X_train, Y_train)
test_data = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# ✅ Define the LSTM model for text classification
class SpamLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=32, num_layers=1, output_size=1):  # Further reduced complexity
        super(SpamLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=1)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.clamp(x, 0, vocab_size - 1)  # Ensure indices are within range
        x = self.embedding(x)
        h0 = torch.zeros(1, x.size(0), 32).to(device)
        c0 = torch.zeros(1, x.size(0), 32).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# ✅ Initialize model
model = SpamLSTM(vocab_size=vocab_size).to(device)
criterion = nn.BCELoss()  # Binary classification loss
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# ✅ Train the LSTM model
def train_model():
    epochs = 3  # Further reduce epochs for efficiency
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.6f}")
    torch.save(model.state_dict(), "spam_lstm_model.pth")
    print("✅ Model saved as 'spam_lstm_model.pth'")

# ✅ Evaluate the model
def evaluate_model():
    model.eval()
    correct, total = 0, 0
    predictions, actual_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    print(f"\n✅ Model Accuracy: {accuracy:.4f}")
    results_df = pd.DataFrame({
        "Actual Label": actual_labels,
        "Predicted Label": predictions
    })
    results_df.to_csv("spam_predictions.csv", index=False)
    print("✅ Predictions saved to 'spam_predictions.csv'.")

# ✅ Run Training and Evaluation
if __name__ == "__main__":
    train_model()
    evaluate_model()
