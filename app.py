import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

# Specify file paths
vectorizer_path = r'C:\Users\tejasri\email spam classification\vectorizer.pkl'  # Update with the correct path
model_path = r'C:\Users\tejasri\email spam classification\model.pkl'  # Update with the correct path

# Load the vectorizer file
try:
    with open(vectorizer_path, 'rb') as file:
        tfidf = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: '{vectorizer_path}' file not found.")
except Exception as e:
    st.error(f"Error loading '{vectorizer_path}': {e}")

# Load the model file
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: '{model_path}' file not found.")
except Exception as e:
    st.error(f"Error loading '{model_path}': {e}")

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Streamlit app
st.title('Email Spam Classification')
input_sms = st.text_area('Enter Message')

if st.button('Predict'):
    if not input_sms:
        st.warning('Please enter a message.')
    else:
        # Preprocess the input text
        transformed_sms = preprocess_text(input_sms)
        st.write('Processed Text:', transformed_sms)

        # Vectorize the preprocessed text
        vector_input = tfidf.transform([transformed_sms])

        # Debugging information
        st.write('Vectorized Input:', vector_input)

        # Make prediction
        result = model.predict(vector_input)[0]  # Use [0] to get the actual value
        proba = model.predict_proba(vector_input)[0][1]  # Probability of being spam

        # Debugging information
        st.write('Model Prediction:', result)
        st.write('Model Probability:', proba)

        # Display the result with adjusted threshold
        threshold = 0.4  # You can experiment with different threshold values
        if proba <= threshold and result == 1:
            st.error(f'Spam (Probability: {proba:.2f})')
        else:
            st.success(f'Not Spam (Probability: {proba:.2f})')

        # Additional debugging information
        st.write('Model Parameters:', model.get_params())

        # Check feature importance
        