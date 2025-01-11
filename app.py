import streamlit as st 
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

import spacy
# Load Spacy language model
nlp = spacy.load("en_core_web_sm")

# Initialize PorterStemmer (if needed for stemming)
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    # 1. Lowercase the text
    text = text.lower()

    # 2. Tokenize using Spacy
    tokens = [token.text for token in nlp(text)]

    # 3. Removing special characters (only alphanumeric tokens)
    y = []
    for i in tokens:
        if i.isalnum():
            y.append(i)

    # 4. Removing stopwords and punctuation
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # 5. Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    # 6. Return the processed text as a single string
    return " ".join(y)

tfidf = pickle.load(open('Vectorizer1.pkl','rb'))
model = pickle.load(open('Model1.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")