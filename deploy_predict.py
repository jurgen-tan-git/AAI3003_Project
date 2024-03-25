from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from joblib import load
import numpy as np

# Vectorize text using the created vectorizer
def vectorize_text(text):
    vectorizer = load('./weights/tfidf_weights.pkl')
    X = vectorizer.transform(text)
    return X

# Function to predict using the model
def predict_on_model(text):
    # Load the Random Forest model
    model = load('./weights/RandomForestClassifier_fold_5_weights.pkl')

    X = vectorize_text(text)

    # Make predictions
    return model.predict(X)

# Function to get the label
def get_label(prediction):
    labels = ['adulting-101','big-read','commentary','gen-y-speaks','gen-z-speaks','singapore','voices','world']
    flat_prediction = np.asarray(prediction).flatten()  # Convert to 1D array
    label_indices = np.nonzero(flat_prediction)[0]  # Find non-zero indices
    return [labels[i] for i in label_indices]
