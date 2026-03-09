import streamlit as st
import joblib

# load saved model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("Fake News Detector")

st.write("Enter a news article to check if it is Fake or Real")

news_text = st.text_area("Enter News Text")

if st.button("Predict"):

    if news_text.strip() != "":
        
        vectorized_text = vectorizer.transform([news_text])
        prediction = model.predict(vectorized_text)

        if prediction[0] == 0:
            st.error("Fake News")
        else:
            st.success("Real News")

    else:
        st.warning("Please enter some text")