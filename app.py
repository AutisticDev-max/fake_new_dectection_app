import streamlit as st
import joblib

# --- LOAD MODEL AND VECTORIZER ---
# Make sure these files are in your repo root
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- STREAMLIT APP ---
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("📰 Fake News Detection App")
st.write("Enter the news text below, and the model will tell you if it's Fake or Real.")

# Text input
news_text = st.text_area("News Text", height=200)

# Predict button
if st.button("Predict"):
    if news_text.strip():  # check user input
        # Transform text using TF-IDF
        vec = vectorizer.transform([news_text])
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec).max()  # confidence

        if prediction == 0:
            st.error(f"Prediction: Fake News ❌\nConfidence: {probability:.2%}")
        else:
            st.success(f"Prediction: Real News ✅\nConfidence: {probability:.2%}")
    else:
        st.warning("Please enter some text to predict.")

# Optional: example news button
if st.button("Try Example News"):
    example = "Donald Trump just tweeted something shocking again."
    vec = vectorizer.transform([example])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec).max()

    if prediction == 0:
        st.error(f"Example Prediction: Fake News ❌\nConfidence: {probability:.2%}")
    else:
        st.success(f"Example Prediction: Real News ✅\nConfidence: {probability:.2%}")
