import streamlit as st
import pandas as pd
import joblib
import os

# Get the current folder path
base_path = os.path.dirname(__file__)

# Model paths
model_path = os.path.join(base_path, "fake_news_model.pkl")
vectorizer_path = os.path.join(base_path, "tfidf_vectorizer.pkl")

# Load model and vectorizer safely
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError:
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

# App Title
st.title("Fake News Detection System")

st.write("Enter a news article below and the model will predict whether it is **Fake** or **Real**.")

# Input form
with st.form("news_form"):
    
    news_text = st.text_area("Enter News Article Text")

    submit_button = st.form_submit_button("Predict")

# Prediction section
if submit_button:

    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    
    else:
        # Transform text using TF-IDF
        vectorized_text = vectorizer.transform([news_text])

        # Make prediction
        prediction = model.predict(vectorized_text)

        # Display result
        st.subheader("Prediction Result")

        if prediction[0] == 0:
            st.error("This news is predicted to be **Fake News**")
        else:
            st.success("This news is predicted to be **Real News**")
