import streamlit as st
import pickle
from transform import transform_text

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit UI
st.title("Spam Email Classifier")
st.write("Enter any email message below to check if it is Spam or Not Spam.")

input_sms = st.text_area("Type your message here:")

if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("Spam Email 🚫")
    else:
        st.success("Not Spam Email ✅")


import nltk
nltk.download('stopwords')
