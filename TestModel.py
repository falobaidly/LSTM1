import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from MSFT import SpamLSTM, build_vocab, tokenize  # Import trained model and preprocessing functions

# ✅ Load the trained vocabulary
file_path = "/home/fahad/LSTM1/combined_data.csv"
df = pd.read_csv(file_path)
vocab = build_vocab(df['text'])  # Ensure consistent vocab
vocab_size = len(vocab) + 2  # Ensure correct vocab size

# ✅ Load the trained model
device = torch.device("cpu")
model = SpamLSTM(vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load("spam_lstm_model.pth", map_location=device))
model.eval()

# ✅ Function to preprocess text from CSV
def preprocess_text(texts):
    def text_pipeline(text):
        return [vocab.get(token, vocab["<unk>"]) for token in tokenize(text)]

    sequences = [text_pipeline(text) for text in texts]
    max_len = 40  # Must match training max_len
    padded_sequences = pad_sequence([torch.tensor(seq[:max_len]) for seq in sequences], batch_first=True, padding_value=vocab["<pad>"])
    return padded_sequences

# ✅ Load the test CSV file
test_file_path = "/home/fahad/LSTM1/sample_test_data.csv"
test_df = pd.read_csv(test_file_path)

# ✅ Check if the file contains a 'text' column
if 'text' not in test_df.columns:
    raise ValueError("The CSV file must contain a 'text' column with email content.")

# ✅ Preprocess the test emails
X_test_new = preprocess_text(test_df['text']).to(device)

# ✅ Run model inference
with torch.no_grad():
    outputs = model(X_test_new).squeeze()
    predictions = (outputs > 0.5).float()

# ✅ Add predictions to the DataFrame
test_df['Predicted Label'] = predictions.cpu().numpy()
test_df['Predicted Label'] = test_df['Predicted Label'].apply(lambda x: "Spam" if x == 1 else "Ham")

# ✅ Save results to a new CSV file
output_path = "/home/fahad/LSTM1/sample_predictions.csv"
test_df.to_csv(output_path, index=False)

print(f"✅ Predictions saved to '{output_path}'.")
print(test_df[['text', 'Predicted Label']].head())  # Display first few results
