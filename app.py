import streamlit as st
import pickle
import os
from transform import transform_text

# Ensure pickle files are loaded relative to app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
tfidf = pickle.load(open(vectorizer_path, "rb"))

# Streamlit UI
st.title("Spam Email Classifier")
st.write("Enter an email message to check if it's Spam or Not Spam.")

# Text input
input_sms = st.text_area("Type your message here:")

# Predict button
if st.button("Predict"):
    # Preprocess and vectorize
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    
    # Predict
    prediction = model.predict(vector_input)[0]

    # Show result
    if prediction == 'spam':
        st.error("Spam Email 🚫")
    else:
        st.success("Not Spam Email ✅")
