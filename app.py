import streamlit as st
from summarise import summarize
from scraper import *
from deploy_predict import *

st.title('TLDR News Summariser For Today Online & Channel News Asia News Articles')

# User input for the URL
url = st.text_input('Enter the URL of the news article you want to summarise:')

if st.button('Summarise Article & Predict Category(-ies)'):
    if url:
        try:
            # Scrape the article
            article = scrape_article(url)
            article_text = ' '.join(article)
            
            # Predict the category of the article
            postprocess_text = preprocess_text(article_text)
            predictions = predict_on_model([postprocess_text])
            labels = get_label(predictions[0])
            st.write('The article is classified under the following categories:', labels)
            
            # Summarise the article
            article_summary = summarize(article_text)
            if article_summary:
                st.write(article_summary)
            else:
                st.write('Failed to generate a summary for the article. Please check the URL.')
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.write('Please enter a valid URL.')
