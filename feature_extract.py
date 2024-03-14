import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import BertModel, BertTokenizer
import torch

# Download NLTK resources (if not already downloaded)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove stopwords
    tokens = word_tokenize(text)
    
    text = ' '.join([word for word in tokens if word not in stop_words])

    return text


def extract_bert_features(texts, model_name='bert-base-uncased'):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)

    # Tokenize and encode text
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

    # Pass input through BERT model
    with torch.no_grad():
        outputs = bert_model(**inputs)
        hidden_states = outputs.last_hidden_state

    # Extract features (average pooling)
    features = hidden_states.mean(dim=1)  # Average pooling across tokens

    return features

if __name__ == "__main__":
    categories = os.listdir('./articles')
    all_features = []

    for category in categories:
        files = os.listdir(f'./articles/{category}')
        for file in files:
            print(category,file)
            with open(f'./articles/{category}/{file}', 'r', encoding='utf-8') as f:
                article = f.read()
                preprocessed_text = preprocess_text(article)
                features = extract_bert_features(preprocessed_text)
                features = features.numpy()
                all_features.append((category, file[:-4], features, article))  # Store category, filename, and features

    # Create DataFrame from collected features
    df_features = pd.DataFrame(all_features, columns=['Category', 'Title', 'Features', 'Text'])
    print(df_features.head()) 
    df_features.to_csv('features.csv', index=False)
