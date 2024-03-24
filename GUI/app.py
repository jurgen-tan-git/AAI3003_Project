import streamlit as st

st.title('TLDR News Summariser')

# User input for the URL
url = st.text_input('Enter the URL of the news article you want to summarise:')

if st.button('Summarise Article'):
    if url:
        # call the model here
        #summary = summarise(url)
        
        # result here
        st.write('summary')
    else:
        st.write('Please enter a valid URL.')
