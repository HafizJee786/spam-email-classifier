import streamlit as st
import pickle
import os
import re

# Simple transform function without NLTK
def transform_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = text.split()
    # Optional: remove common English stopwords manually
    stop_words = set([
        'the','and','is','in','to','of','for','on','with','a','an','this','that'
    ])
    y = [word for word in text if word not in stop_words]
    return " ".join(y)

# Load model + vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "spam_model.pkl"), "rb") as f:
    data = pickle.load(f)

model = data["model"]
tfidf = data["tfidf"]

# Streamlit UI
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
