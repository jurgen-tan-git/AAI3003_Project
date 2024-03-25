import streamlit as st
from summarise import summarize
from scraper import *
from deploy_predict import *

st.title('TLDR News Summariser')

# User input for the URL
url = st.text_input('Enter the URL of the news article you want to summarise:')

if st.button('Summarise Article'):
    if url:
        # Scrape the article
        article = scrape_article(url)
        article_text = ' '.join(article)

        # Predict the category of the article
        postprocess_text = preprocess_text(article_text)
        predictions = predict_on_model([postprocess_text])
        print(predictions)
        labels = get_label(predictions[0])
        st.write('The article is classified under the following categories:', labels)
        
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
