import streamlit as st
from summarise import summarize
from scraper import *

st.title('TLDR News Summariser')

# User input for the URL
url = st.text_input('Enter the URL of the news article you want to summarise:')

if st.button('Summarise Article'):
    if url:
        # Scrape the article
        article = scrape_article(url)
        print(article)
        article_text = ' '.join(article)
        print(article_text)

        # Summarise the article
        article_text = summarize(article_text)
        print(article_text)
        if not article_text:
            st.write('Failed to scrape the article. Please enter a valid URL.')
            st.stop()
        else:
            # result here
            st.write(article_text)
    else:
        st.write('Please enter a valid URL.')
