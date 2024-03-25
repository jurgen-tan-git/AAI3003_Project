from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from joblib import load
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk

def remove_tags(text):
  remove = re.compile(r'')
  return re.sub(remove, '', text)

def special_char(text):
  reviews = ''
  for x in text:
    if x.isalnum():
      reviews = reviews + x
    else:
      reviews = reviews + ' '
  return reviews

def convert_lower(text):
   return text.lower()


def remove_stopwords(text):
  stop_words = set(stopwords.words('english'))
  words = word_tokenize(text)
  return [x for x in words if x not in stop_words]

def lemmatize_word(text):
  wordnet = WordNetLemmatizer()
  return " ".join([wordnet.lemmatize(word) for word in text])

# Preprocess text
def preprocess_text(text):
    nltk.download('stopwords')
    nltk.download('wordnet')
    text = remove_tags(text)
    text = special_char(text)
    text = convert_lower(text)
    text = remove_stopwords(text)
    text = lemmatize_word(text)
    return text

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
    flat_prediction = np.asarray(prediction).flatten()
    label_indices = np.nonzero(flat_prediction)[0]
    return [labels[i] for i in label_indices]

# Test the function
text = ['singapore singapore former transport minister iswaran monday march 25 handed fresh charge alleging obtained valuable thing public servant item said worth 18 956 94 total obtained man named lum kok seng according charge iswaran eight new charge said obtained item mr lum november 2021 november 2022 iswaran transport minister time said obtained consideration mr lum knew related contract work tanah merah mrt station cna asked corrupt practice investigation bureau investigating mr lum singaporean name currently listed judiciary hearing list identified mr david lum kok seng company website mr lum managing director singapore exchange listed lum chang holding subsidiary construction property development investment mr lum managing director since sept 1 1985 also director number subsidiary including lum chang building contractor lum chang property lum chang property investment lum chang asia pacific according lum chang website mr lum 40 year industry experience successfully led expansion group property development activity singapore malaysia united kingdom add mr lum market knowledge strategic business contact relentless entrepreneurial drive significantly contributed development group brother mr raymond lum kwan sung executive chairman lum chang holding since 1984 mr lum two son adrian lum wen hong kelvin lum wen sum also director company outside lum chang network mr david lum kok seng sits board director nanyang girl high school also shareholder lasalle college art chinese opera institute lum chang holding website list lum chang building contractor one subsidiary business construction alongside lum chang interior lum chang brandsbridge lum chang building contractor involved project singapore government two ongoing one land transport authority lta first tender awarded 2016 addition alteration work tanah merah station existing viaduct includes adding platform expected completed year concourse east west line station contract valued 325 million project awarded 2018 involves construction 1 95km section north south corridor tunnel ang mo kio ave 3 ang mo kio ave 9 according medium release contract worth 799 million lta awarded contract lum chang building contractor since 2019 authority said statement iswaran charge monday lum chang building contractor also previously designed constructed mrt station including downtown line bukit panjang mrt station tunnel apart lta lum chang building contractor done project singapore national water agency pub lum chang building contractor also accused workplace safety health act breach resulting death worker shajahan mohammad charged state court incident dec 15 2020 principal construction company blt geoworks employee worksite near junction upper changi road east koh sek lim road charge sheet state lum chang building contractor failed take measure ensure safety health employee leading mr shajahan death lum chang building contractor employee ong yen sheng appointed workplace safety health officer worksite also charged ong case set mention march 28 company slated april 16 cna report like visit cna asia']
prediction = predict_on_model(text)
label = get_label(prediction[0])
print(prediction)