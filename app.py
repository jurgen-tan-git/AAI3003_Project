import streamlit as st
from summarise import summarize
from scraper import *
from deploy_predict import *
import multiprocessing

st.title('TLDR News Summariser')

# User input for the URL
url = st.text_input('Enter the URL of the news article you want to summarise:')

if st.button('Summarise Article'):
    if url:
        # Scrape the article
        article = scrape_article(url)
        article_text = ' '.join(article)

        # Predict the category of the article using multiprocessing
        predict_process = multiprocessing.Process(target=predict_on_model, args=(article,))
        predict_process.start()

        # Summarise the article
        article_text = summarize(article_text)
        print(article_text)
        if not article_text:
            st.write('Failed to scrape the article. Please enter a valid URL.')
            st.stop()
        else:
            # Wait for the prediction process to finish
            predict_process.join()

            # Display the summarised article
            st.write(article_text)

            # Get predictions and display the categories
            predictions = predict_process.get()
            labels = get_label(predictions)
            st.write('The article is classified under the following categories:', labels)

    else:
        st.write('Please enter a valid URL.')
