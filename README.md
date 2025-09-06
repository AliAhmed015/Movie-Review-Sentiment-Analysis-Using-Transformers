# Sentiment Analysis with a Transformer Model

This project was developed as part of the **Future AWS AI Scientist Nanodegree** and implements a sentiment analysis model using a custom-built Transformer architecture in PyTorch. It is designed to classify movie reviews from the IMDB dataset as either **positive** or **negative**, and is entirely implemented in a Jupyter Notebook for clarity and experimentation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Data Loading and Preprocessing](#data-loading-and-preprocessing)
- [Implementation Details](#implementation-details)
  - [Data Loading and Preprocessing](#data-loading-and-preprocessing-1)
  - [Custom PyTorch Dataset](#custom-pytorch-dataset)
  - [Transformer Model Architecture](#transformer-model-architecture)
- [Performance and Results](#performance-and-results)
- [Requirements](#requirements)
- [Suggestions for Future Work](#suggestions-for-future-work)
- [License](#license)

---

## Project Overview

The primary objective of this project is to build and train a sentiment classification model for text data using a **Transformer-based neural network**. The solution involves the following core components:

- **Data Loading and Exploration**: Load and visualize the IMDB movie reviews dataset to understand its structure and content.
- **Custom Dataset Class**: Build a PyTorch-compatible `IMDBDataset` class for clean and efficient data handling.
- **Transformer Model**: Implement a custom Transformer architecture (`DemoGPT`) from scratch for binary sentiment classification.
- **Training Loop**: Develop a complete training pipeline with evaluation, loss tracking, and accuracy measurement.
- **Evaluation**: Test the model on a separate dataset split and summarize the final accuracy.

---

## Key Features

- **Notebook-based implementation** for interactive experimentation and easier debugging.
- **Custom Transformer architecture** (`DemoGPT`) built from scratch without relying on Hugging Face or pretrained APIs.
- **Clean modular design**, separating data preprocessing, model building, training, and evaluation stages.
- **Accuracy function** and evaluation metrics implemented for binary classification.
- **Optional inference support** for testing on new data using model prediction cells.

---

### Data Loading and Preprocessing

- Dataset: IMDB Movie Reviews  
- Dataset download: The IMDB dataset can be downloaded from [here](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)  
- To download the dataset directly in a Jupyter Notebook or a Linux environment, run:

  ```bash
  !wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
  ```

---

## Implementation Details

### Data Loading and Preprocessing

- Dataset: IMDB Movie Reviews
- Tokenization using custom tokenizer or simple whitespace splitting
- Padding/truncating to fixed sequence length
- Vocabulary creation from training data
- Data is split into training, validation, and test sets

### Custom PyTorch Dataset

A `torch.utils.data.Dataset` class called `IMDBDataset` is implemented with:

- `__init__()` for loading texts and labels
- `__len__()` for dataset size
- `__getitem__()` for returning a single (input, label) pair
- Supports batch loading via `DataLoader`

### Transformer Model Architecture

The custom `DemoGPT` Transformer includes:

- Embedding layer
- Positional encoding
- Multi-head self-attention
- Feed-forward layers
- Layer normalization
- Output layer for binary classification

---

## Performance and Results

- Final test accuracy: **76.88%**
- Training and validation losses are plotted across epochs
- Overfitting is avoided using dropout and careful hyperparameter tuning
- Summary of classification metrics is presented in the notebook

---

## Requirements

Install the following dependencies (preferably in a virtual environment):

```bash
pip install torch torchvision numpy matplotlib pandas
```

---

# Suggestions for Future Work

Based on project feedback, the following enhancements could further improve the model's performance and functionality:

- **Increase Accuracy:** Experiment with different model configurations or training loop tweaks to achieve a test accuracy exceeding 90%.

- **Model Checkpointing:** Implement functionality to save the best model checkpoint during training.

- **Inference Interface:** Create a simple function or class that can load the saved model and perform inference on a batch of new inputs.

- **Advanced Evaluation:** Explore additional evaluation metrics beyond accuracy, such as F1-score, precision, and recall.

---

# License

This project is licensed under the MIT License.
