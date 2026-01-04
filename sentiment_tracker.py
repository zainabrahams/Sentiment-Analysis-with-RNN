import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding

#loading dataset, reading the csv into a pandas dataframe and accessing the column names
data = pd.read_csv('swiggy.csv')
print("Columns in the dataset: ")
print(data.columns.tolist())


#cleaning text (reviews)
data["Review"] = data["Review"].str.lower()
data["Review"] = data["Review"].replace(r'[^a-z0-9\s]', '', regex=True) #removes everything that ISN'T in the brackets
data['sentiment'] = data['Avg Rating'].apply(lambda x: 1 if x > 3.5 else 0) #creates sentiment column with 1 for ratings above 3.5
data = data.dropna() #removes rows that contain missing values

#tokenizing the reviews and separating the target labels for learning
max_features = 5000 #max number of words in tokenizer
max_length = 200 #length for each input sequence after padding
tokenizer = Tokenizer(num_words=max_features) #tokenizer keeps top 5000 words
tokenizer.fit_on_texts(data["Review"]) #looks at all the reviews/assigns unique integer to each word 
X = pad_sequences(tokenizer.texts_to_sequences(data["Review"]), maxlen=max_length) #ensures all sequences have same length 200 (adding or truncating), #converts each review into sequence of integers using tokenizers vocabulary
y = data['sentiment'].values #target labels = y for each review

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) #splits data into training and test 80/20

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train) #splits training data into training and validation sets 90/10
#validation set is used to tune the model during training (checking performance each epoch, test set isn't touched till the very end)

#building RNN Model
model = Sequential([Embedding(input_dim=max_features, output_dim=16, input_length=max_length), SimpleRNN(64, activation='tanh', return_sequences=False), Dense(1, activation='sigmoid')]) 
#Sequential is the type of model being built layers stacked one after the other
#embedding layer converts words into vectors, 
#simpleRNN layer reads sequence and summarises into one vector (64 number of hidden units, tahn is an activation function), 
#dense layer fully connected layer predicting output (output layer)

#simpleRNN stores the memory in the RNN
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #compiling the model, the loss function and optimization algorithm

#training the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), verbose=1)
score = model.evaluate(X_test, y_test, verbose=0)

print(f"Test accuracy: {score[1]:.2f}")

def sentiment_prediction(review_text):
    text = review_text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences (seq, maxlen=max_length)

    prediction = model.predict(padded)[0][0]
    return f"{'Positive' if prediction >= 0.5 else 'Negative'} (Probability was {prediction:.2f})"

print()
review = "The food was horrible"
print(f"Review: {review}")
print(sentiment_prediction(review))



    







