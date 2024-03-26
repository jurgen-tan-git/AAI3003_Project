import streamlit as st
from summarise import summarize
from scraper import *
from deploy_predict import *

st.title('TLDR News Summariser')

# User input for the URL
url = st.text_input('Enter the URL of the news article you want to summarise:')
# Radio button to select the model
model = st.radio('Select the model to use:', ['RandomForest', 'DistilBert'])

if st.button('Summarise & Predict'):
    if url:
        # Scrape the article
        article = scrape_article(url)
        article_text = ' '.join(article)
        postprocess_text = preprocess_text(article_text)

        # Predict the category of the article using RandomForest
        if model == 'RandomForest':
            predictions = predict_on_randomforest([postprocess_text])
            labels = get_label(predictions[0])
            st.write('The article is classified under the following categories using RandomForest:', labels)

        elif model == 'DistilBert':
            prediction = predict_on_distilbert_model([postprocess_text]).to('cpu')
            labels = get_label(prediction[0])
            st.write('The article is classified under the following categories using DistilBert:', labels)

        else:
            st.write('Please select a model to use.')
        # Summarise the article
        article_text = summarize(article_text)
        if not article_text:
            st.write('Failed to scrape the article. Please enter a valid URL.')
            st.stop()
        else:
           # Display the summarised article
            st.write(article_text)

    else:
        st.write('Please enter a valid URL.')
