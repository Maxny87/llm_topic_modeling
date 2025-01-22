import pandas as pd
from sklearn.datasets import fetch_20newsgroups

import re
from nltk.corpus import stopwords


def get_text():
    """
    This dataset was fetched using scikit-learn.

    :return:
    """
    newsgroup_data = fetch_20newsgroups(subset='all')

    # a list where each element is a text
    text = newsgroup_data['data']
    return text


def get_preprocessed_text(remove_stop_words=True, save_dataset=False):
    data = get_text()
    clean_data = [preprocess_text(t, remove_stop_words) for t in data]
    clean_dataset = [str(doc) for doc in clean_data if isinstance(doc, str) and doc.strip() != '']

    if save_dataset:
        clean_df = pd.DataFrame(clean_dataset, columns=['text'])
        clean_df.to_csv('./data/clean_newsgroup20_final.csv', index=False)

    return clean_dataset


def get_preprocessed_data_from_csv(file_path):
    data = pd.read_csv(file_path)["text"].to_list()
    return data


def preprocess_text(text, remove_stop_words):
    stop_words = set(stopwords.words('english'))

    # removes the header for each document
    header_end = text.find('\n\n')
    if header_end != -1:
        text = text[header_end:].strip()

    # Removes new lines and tabs
    new_text = text.replace("\n", " ")
    new_text = new_text.replace("\t", " ")

    # removes all email addresses
    new_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\b', '', new_text)
    new_text = re.sub(r'<.*?>', '', new_text)  # remove text within angle brackets

    # removing reoccurring special characters from the text
    new_text = re.sub(r'[<>|]', '', new_text)
    new_text = new_text.replace('-', ' ')

    # remove special characters and symbols but keep punctuation and currency symbols and parenthesis
    new_text = re.sub(r'[^a-zA-Z0-9\s,.\'!?()/$]', '', new_text)

    # Remove URLs
    new_text = re.sub(r'\b(http|https)://\S+|www\.\S+|\S+\.(com|org|net|edu)\b', '', new_text)

    # removes << and >> if there is any
    new_text = new_text.replace("<<", "").replace(">>", "")

    # normalize repeated punctuation
    new_text = re.sub(r'([!?])\1+', r'\1', new_text)

    # remove extra periods
    new_text = re.sub(r'\.{2,}', ' ', new_text)

    # removes empty parentheses
    new_text = re.sub(r'\(\s*\)', '', new_text)

    # convert to lowercase
    new_text = new_text.lower()

    # optionally remove stop words
    if remove_stop_words:
        new_text = " ".join([word for word in new_text.split() if word not in stop_words])

    # strip any extra whitespace
    new_text = " ".join(new_text.split())

    return new_text
