# Sentiment Analysis with Recurrent Neural Networks (RNN)

This project implements a **Recurrent Neural Network (RNN)** to classify customer reviews from Swiggy as either **Positive** (1) or **Negative** (0) based on their average rating.

## 📌 Overview
The model uses Natural Language Processing (NLP) techniques to clean text data, tokenize it into sequences, and train a SimpleRNN model to understand the sentiment behind the reviews.

## 🛠️ Tech Stack
* **Python** (Core Logic)
* **Pandas & NumPy** (Data Manipulation)
* **Scikit-Learn** (Data Splitting)
* **TensorFlow/Keras** (Deep Learning & RNN)

## 🏗️ Model Architecture
The model is built using a sequential architecture:
1. **Embedding Layer**: Converts word integers into dense vectors of fixed size ($output\_dim=16$).
2. **SimpleRNN Layer**: Processes the sequence data with 64 hidden units and `tanh` activation.
3. **Dense Layer**: A single neuron with a **Sigmoid** activation function to output a probability between 0 and 1.

## 🚀 Workflow
1. **Data Cleaning**: All text is converted to lowercase and special characters are removed using Regex.
2. **Labeling**: Ratings > 3.5 are marked as Positive (1), while lower ratings are marked as Negative (0).
3. **Tokenization & Padding**: 
    * The vocabulary is limited to the top 5,000 words.
    * Reviews are converted to integer sequences and padded/truncated to a length of 200.
4. **Training**: The data is split into Training (72%), Validation (8%), and Test (20%) sets.
5. **Evaluation**: The model uses `binary_crossentropy` as the loss function and `adam` as the optimizer.

