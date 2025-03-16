import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import time

# ✅ Maximize CPU utilization
torch.set_num_threads(os.cpu_count())
device = torch.device("cpu")

# ✅ Load dataset with optimized reading
file_path = "/home/fahad/LSTM1/combined_data.csv"
df = pd.read_csv(file_path, usecols=['text', 'label'])  # Only load needed columns
df['label'] = df['label'].astype(int)

# ✅ More efficient vocabulary building
def tokenize(text):
    return text.lower().split()

def build_vocab(texts, min_freq=2, max_vocab=15000):
    all_tokens = [token for text in texts for token in tokenize(text)]
    counter = Counter(all_tokens)
    
    # Limit vocabulary size
    vocab = {word: i + 2 for i, (word, count) in enumerate(
        counter.most_common(max_vocab)) if count >= min_freq}
    vocab["<unk>"] = 0
    vocab["<pad>"] = 1
    return vocab

vocab = build_vocab(df['text'])
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# ✅ Optimize sequence processing with vectorized operations
def text_pipeline(text):
    return [vocab.get(token, vocab["<unk>"]) for token in tokenize(text)]

# ✅ Process all sequences at once for better efficiency
sequences = [text_pipeline(txt) for txt in df['text']]

# ✅ Optimize sequence length (speed vs accuracy tradeoff)
max_len = min(max(len(seq) for seq in sequences), 35)  # Slightly reduced

# ✅ More efficient padding
padded_sequences = []
for seq in sequences:
    # Truncate and pad in one operation
    padded_seq = seq[:max_len] + [vocab["<pad>"]] * (max_len - min(len(seq), max_len))
    padded_sequences.append(padded_seq)

X = torch.tensor(padded_sequences, dtype=torch.long).to(device)
Y = torch.tensor(df['label'].values, dtype=torch.float32).to(device)

# ✅ Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ✅ Optimize DataLoader for CPU
batch_size = 128  # Increased batch size
train_data = TensorDataset(X_train, Y_train)
test_data = TensorDataset(X_test, Y_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                         num_workers=0, pin_memory=False)  # No workers for CPU
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, 
                        num_workers=0, pin_memory=False)

# ✅ Optimized GRU Model (Replaced LSTM with GRU)
class OptimizedSpamGRU(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=32, num_layers=1, output_size=1, dropout=0.2):
        super(OptimizedSpamGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=1)
        
        # Replace LSTM with GRU
        self.gru = nn.GRU(
            embed_size, 
            hidden_size, 
            num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0 if num_layers == 1 else dropout
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.vocab_size = vocab_size
        
    def forward(self, x):
        # Ensure token indices are valid
        x = torch.clamp(x, 0, self.vocab_size - 1)
        
        # Get embeddings
        x = self.embedding(x)
        x = self.dropout(x)
        
        # Process with GRU instead of LSTM
        gru_out, _ = self.gru(x)
        
        # Use only the last output and apply dropout
        last_out = gru_out[:, -1, :]
        last_out = self.dropout(last_out)
        
        # Final classification
        out = self.fc(last_out)
        return self.sigmoid(out)

# ✅ Initialize Model
model = OptimizedSpamGRU(
    vocab_size=vocab_size,
    embed_size=32,     # Increased from 24 to 32 for GRU
    hidden_size=32,    # Increased from 24 to 32 for GRU
    num_layers=1,
    dropout=0.2
).to(device)

# Try to optimize with JIT
try:
    model = torch.jit.script(model)
    print("✅ Model optimized with JIT compilation")
except Exception as e:
    print(f"JIT compilation not available: {e}")

# ✅ Optimized training
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# ✅ Use learning rate scheduler for faster convergence
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1, verbose=True
)

# ✅ Train with performance tracking and early stopping
def train_model(epochs=3, patience=2):
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...")
    total_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        batch_count = 0
        
        for inputs, labels in train_loader:
            batch_start = time.time()
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            batch_time = time.time() - batch_start
            total_loss += loss.item()
            batch_count += 1
            
            if batch_count % 20 == 0:
                print(f"Batch {batch_count}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Time: {batch_time:.3f}s")
        
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")
        
        # Update learning rate based on performance
        scheduler.step(avg_loss)
        
        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "optimized_spam_gru.pth")  # Changed filename
            print(f"✅ Model improved, saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    total_time = time.time() - total_start_time
    print(f"✅ Training completed in {total_time:.2f} seconds")

# ✅ Evaluate Model
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    print("\nStarting evaluation...")
    eval_start = time.time()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            predicted = (outputs > 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    eval_time = time.time() - eval_start
    accuracy = correct / total
    
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    print(f"✅ Model Accuracy: {accuracy:.4f}")

    # Calculate metrics and create a simple confusion matrix
    true_positives = sum((p == 1 and l == 1) for p, l in zip(all_predictions, all_labels))
    false_positives = sum((p == 1 and l == 0) for p, l in zip(all_predictions, all_labels))
    true_negatives = sum((p == 0 and l == 0) for p, l in zip(all_predictions, all_labels))
    false_negatives = sum((p == 0 and l == 1) for p, l in zip(all_predictions, all_labels))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    results_df = pd.DataFrame({
        "Actual": all_labels,
        "Predicted": all_predictions
    })
    results_df.to_csv("optimized_spam_gru_predictions.csv", index=False)  # Changed filename
    print("✅ Predictions saved to 'optimized_spam_gru_predictions.csv'")

# ✅ Inference function for production
def predict_single(text, model_path="optimized_spam_gru.pth"):  # Changed filename
    # Load the saved model for inference
    loaded_model = OptimizedSpamGRU(vocab_size=vocab_size).to(device)  # Changed class name
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    
    # Process the text
    tokenized = text_pipeline(text)
    padded = tokenized[:max_len] + [vocab["<pad>"]] * (max_len - min(len(tokenized), max_len))
    input_tensor = torch.tensor([padded], dtype=torch.long).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = loaded_model(input_tensor).item()
    
    is_spam = output > 0.5
    return {"is_spam": bool(is_spam), "confidence": output}

# ✅ Run Training and Evaluation
if __name__ == "__main__":
    train_model(epochs=3)
    evaluate_model()
    
    # Example of predicting on a single text
    sample_text = "Congratulations! You've won a free gift card. Click here to claim your prize now!"
    result = predict_single(sample_text)
    print(f"\nSample spam check: {result}")