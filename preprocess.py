import pandas as pd

import re
from nltk.corpus import stopwords

def get_text_from_csv(csv_filepath):
    return pd.read_csv(csv_filepath)

def base_preprocess(text, remove_stop_words=True):

    stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = text.replace("\n", " ").replace("\t", " ")

    # Remove email addresses
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\b', '', text)

    # Remove URLs
    text = re.sub(r'\b(http|https)://\S+|www\.\S+|\S+\.(com|org|net|edu)\b', '', text)

    # Remove special characters but retain punctuation and currency symbols
    text = re.sub(r'[^a-zA-Z0-9\s,.\'!?£€$()/]', '', text)

    # Remove empty parentheses
    text = re.sub(r'\(\s*\)', '', text)

    if remove_stop_words:
        text = " ".join([word for word in text.split() if word not in stop_words])

    # Strip extra whitespace
    text = " ".join(text.split())

    return text



