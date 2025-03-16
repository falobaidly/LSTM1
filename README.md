# README

## Overview

This repository demonstrates a **spam detection** pipeline using PyTorch. It originally utilized an LSTM architecture but has been updated to use a **GRU** instead. The code covers all steps from loading the dataset to training, evaluation, and inference.

## Features

- Efficient CPU usage through:
  - `torch.set_num_threads(os.cpu_count())`
  - Vectorized tokenization and sequence processing
  - Optimized vocabulary creation and dataset loading
- Model replaced **LSTM** with **GRU** for potentially better performance or simpler gating mechanisms
- Learning rate scheduler (`ReduceLROnPlateau`) to aid faster convergence
- Early stopping mechanism based on the validation loss
- JIT scripting (where available) for runtime optimization

## Requirements

- Python 3.7 or later
- [PyTorch](https://pytorch.org/) (version 1.7+ recommended)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)

Install the required libraries via:
```bash
pip install torch pandas numpy scikit-learn
```

## File Structure

- **combined_data.csv**  
  A CSV file containing two columns:  
  1. `text` - The text message content.  
  2. `label` - The corresponding label (0 for not spam, 1 for spam).

- **optimized_spam_gru.py** (or similar name)  
  Main script containing:
  - Data reading and preprocessing  
  - Vocabulary building and tokenization  
  - Model definition (using a GRU)  
  - Training loop with early stopping  
  - Evaluation metrics and confusion matrix  
  - Single text inference function  

- **optimized_spam_gru.pth**  
  Model checkpoint saved when an improvement in validation loss is observed.

## Usage

1. **Modify the file paths**  
   - Make sure the `file_path` variable in the code is set to the correct path for `combined_data.csv`.
   - Adjust any file output paths (`optimized_spam_gru_predictions.csv`, `optimized_spam_gru.pth`) if desired.

2. **Run the script**
   ```bash
   python optimized_spam_gru.py
   ```
   This will:
   - Load and preprocess the data
   - Train the GRU model
   - Evaluate the model’s performance
   - Save model weights to `optimized_spam_gru.pth`
   - Generate a CSV file (`optimized_spam_gru_predictions.csv`) containing predictions for the test set

3. **Check training and evaluation logs**  
   - During training, the script logs:
     - Batch-level loss and processing time
     - Epoch-level average loss and total time
   - Final metrics reported:
     - Accuracy, Precision, Recall, F1 Score

4. **Perform Inference (optional)**  
   After training, you can test the model on a custom text by calling the `predict_single` function:
   ```python
   sample_text = "Congratulations! You've won a free gift card. Click here to claim your prize now!"
   result = predict_single(sample_text)
   print(result)  # e.g. {"is_spam": True, "confidence": 0.8673}
   ```

## Code Explanation

Below is the high-level workflow of the code:

1. **Imports and Setup**  
   - Uses `os.cpu_count()` to set the number of CPU threads for PyTorch.
   - Defines the device as CPU.

2. **Data Loading**  
   - Loads only necessary columns (`text` and `label`) from the CSV file.
   - Converts `label` to integer type.

3. **Vocabulary Building**  
   - Tokenizes text with a simple `.split()` (after lowercasing).
   - Counts token frequencies and builds a vocabulary dictionary.
   - Sets special tokens `<unk>` and `<pad>`.

4. **Text to Sequence**  
   - Converts each text message into a list of token indices.
   - Pads (or truncates) these lists to a maximum length (`max_len = 35` by default).

5. **Dataset Preparation**  
   - Splits data into train and test sets using `train_test_split`.
   - Packs the data into `TensorDataset` and `DataLoader`.

6. **Model Definition** (`OptimizedSpamGRU`)  
   - An embedding layer maps vocabulary indices to learned vector embeddings.
   - A single-layer or multi-layer **GRU** processes each sequence.
   - The final hidden state goes into a fully connected layer, followed by a sigmoid for binary classification.
   - Includes dropout for regularization.

7. **Training Routine** (`train_model`)  
   - Uses `Adam` with weight decay for optimization.
   - Clips gradients to prevent exploding gradients.
   - Employs a learning rate scheduler (`ReduceLROnPlateau`).
   - Implements early stopping based on loss improvements.

8. **Evaluation** (`evaluate_model`)  
   - Computes accuracy by comparing predicted labels to ground truth.
   - Calculates precision, recall, and F1 score.
   - Saves a CSV with actual and predicted labels.

9. **Inference** (`predict_single`)  
   - Loads the trained model state dictionary from `.pth` file.
   - Tokenizes and pads a single text input.
   - Returns a dictionary containing the spam detection result and confidence score.

## Notes and Tips

- If you have a very large dataset, consider increasing the `num_workers` argument in the `DataLoader`. For purely CPU-based operations, this might or might not help depending on your system.
- Hyperparameters such as `embed_size`, `hidden_size`, `batch_size`, or `max_len` can be adjusted to balance performance and resource usage.
- For reproducibility, consider setting manual seeds for random operations (`torch.manual_seed`, `numpy.random.seed`, etc.).
- If you prefer using an **LSTM** instead of a GRU, you can revert the `nn.GRU` section to `nn.LSTM` and rename references accordingly.

## Contributing

Feel free to open issues or pull requests if you have suggestions for improvements or encounter any problems using this code.

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute this software as you see fit.

---

**Happy coding!** If you have any further questions or issues, please open an issue in the repository or reach out directly.
