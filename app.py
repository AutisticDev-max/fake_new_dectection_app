import streamlit as st
import joblib
import os

# --- Load model and vectorizer ---
base_path = os.path.dirname(__file__)

model_path = os.path.join(base_path, "fake_news_model.pkl")
vectorizer_path = os.path.join(base_path, "tfidf_vectorizer.pkl")

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError:
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Streamlit UI ---
st.title("FAKE NEWS DETECTION")
st.write("Enter a news article and the model will predict whether it is Fake or Real.")

with st.form("news_form"):
    st.subheader("Enter News Article")

    news_text = st.text_area("News Text")

    submit_button = st.form_submit_button("Predict")

if submit_button:

    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        input_data = vectorizer.transform([news_text])

        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)

        st.subheader("Prediction Result")

        if prediction[0] == 0:
            st.error("The news is predicted as: FAKE ❌")
        else:
            st.success("The news is predicted as: REAL ✅")

        st.subheader("Prediction Confidence")

        class_names = model.classes_

        for name, prob in zip(class_names, probabilities[0]):
            percent = prob * 100
            st.write(f"Class {name}: {percent:.2f}%")
            st.progress(prob)
