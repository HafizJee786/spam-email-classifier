import streamlit as st
import pickle
import os
import nltk
from nltk.corpus import stopwords
from transform import transform_text

# --- NLTK setup (Cloud friendly) ---
nltk.download('stopwords', quiet=True)

# --- Load model and vectorizer from repo root ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    tfidf = pickle.load(f)

# --- Streamlit UI ---
st.title("Spam Email Classifier")
st.write("Enter an email message to check if it's Spam or Not Spam.")

input_sms = st.text_area("Type your message here:")

if st.button("Predict"):
    try:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        prediction = model.predict(vector_input)[0]

        if prediction == 'spam':
            st.error("Spam Email 🚫")
        else:
            st.success("Not Spam Email ✅")
    except Exception as e:
        st.error(f"Error: {e}")
