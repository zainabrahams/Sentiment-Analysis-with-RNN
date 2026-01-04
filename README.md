# Sentiment Analysis with Recurrent Neural Networks (RNN)

This project implements a **Recurrent Neural Network (RNN)** to classify customer reviews from Swiggy as either **Positive** (1) or **Negative** (0) based on their average rating.

## 📌 Overview
The model uses Natural Language Processing (NLP) techniques to clean text data, tokenize it into sequences, and train a SimpleRNN model to understand the sentiment behind the reviews.

Although the final model achieved a **strong test accuracy of 84%**, a major limitation lies in the dataset itself. Sentiment labels were not manually annotated; instead, they were inferred directly from numerical review ratings. This introduces significant noise, as a customer may write a positive review but choose not to assign a maximum rating.

In the original version of the model, reviews with ratings ≥ 3.5 were labelled as positive, resulting in a test accuracy of 72%. By refining the labelling strategy, classifying reviews as positive only if the rating was ≥ 4, and negative if the rating was ≤ 3, the test accuracy improved to 84%. This improvement highlights that the model’s earlier underperformance was largely due to inconsistent and noisy sentiment labels rather than weaknesses in the model architecture itself.

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

## Future improvements
Future work could focus on improving label quality by manually annotating review sentiment rather than inferring sentiment solely from numerical ratings. This would significantly reduce label noise and allow the model to learn more accurate linguistic patterns. Additionally, more advanced architectures such as LSTM or transformer-based models (e.g., BERT) could be explored to better capture contextual and semantic meaning in reviews. Incorporating techniques such as class weighting, stopword removal, and data augmentation may further improve robustness, particularly for short or sentiment-heavy reviews.