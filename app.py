import streamlit as st
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Get current folder
base_path = os.path.dirname(__file__)

model_path = os.path.join(base_path, "fake_news_model.pkl")
vectorizer_path = os.path.join(base_path, "tfidf_vectorizer.pkl")

# Load model and vectorizer
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except:
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Title
st.title("📰 Fake News Detection App")

st.write(
    "This application uses Machine Learning to classify a news article as **Fake** or **Real**."
)

st.divider()

# Input section
st.subheader("Enter News Article")

news_text = st.text_area(
    "Paste a news article below:",
    height=200
)

# Prediction
if st.button("Analyze News"):

    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:

        vectorized_text = vectorizer.transform([news_text])

        prediction = model.predict(vectorized_text)
        probability = model.predict_proba(vectorized_text)

        fake_prob = probability[0][0]
        real_prob = probability[0][1]

        st.divider()
        st.subheader("Prediction Result")

        if prediction[0] == 0:
            st.error("🚨 This news is predicted to be **FAKE**")
        else:
            st.success("✅ This news is predicted to be **REAL**")

        st.subheader("Model Confidence")

        st.write(f"Fake News Probability: {fake_prob*100:.2f}%")
        st.progress(fake_prob)

        st.write(f"Real News Probability: {real_prob*100:.2f}%")
        st.progress(real_prob)

st.divider()

st.caption("Machine Learning Fake News Detection Project")
