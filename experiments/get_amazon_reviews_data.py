import pandas as pd

import re
from nltk.corpus import stopwords

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException


def get_text(file_path, fraction_to_sample, chunk_size, seed):
    """
    This dataset was downloaded from: https://amazon-reviews-2023.github.io/main.html

    We are using the books subset of reviews for the analysis

    :return:
    """

    DetectorFactory.seed = seed
    chunks = []

    for chunk in pd.read_json(file_path, lines=True, chunksize=chunk_size):
        sample = chunk.sample(frac=fraction_to_sample, random_state=seed)
        chunks.append(sample)

    reviews = pd.concat(chunks, ignore_index=True)

    # making sure we only use english reviews
    english_reviews = []

    for review in reviews['text']:
        try:
            if detect(review) == 'en':
                english_reviews.append(review)
        except LangDetectException:
            # Handle cases where language detection fails
            pass

    return english_reviews


def get_preprocessed_data(file_path, fraction_to_sample, chunk_size, seed, remove_stopwords=True, save_dataset=False, save_file_path=None):
    text = get_text(file_path, fraction_to_sample, chunk_size, seed)
    clean_data = [preprocess_amazon_data(t, remove_stopwords) for t in text]
    clean_dataset = [str(doc) for doc in clean_data if isinstance(doc, str) and doc.strip() != '']

    if save_dataset:
        df_english = pd.DataFrame(clean_dataset, columns=['text'])
        df_english.to_csv(f'{save_file_path}.csv', escapechar='\\', index=False)
        df_english.to_json(f'{save_file_path}.jsonl', orient='records', lines=True)

    return clean_data


def get_preprocessed_data_from_csv(file_path):
    data = pd.read_csv(file_path)["text"].to_list()
    return data


def preprocess_amazon_data(text, remove_stop_words):
    stop_words = set(stopwords.words('english'))

    # removes new lines and tabs
    new_text = text.replace("\n", " ").replace("\t", " ")

    # removes all email addresses
    new_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\b', '', new_text)

    # remove HTML tags
    new_text = re.sub(r'<[^>]+>', '', new_text)

    # remove URLs
    new_text = re.sub(r'\b(http|https)://\S+|www\.\S+|\S+\.(com|org|net|edu)\b', '', new_text)

    # remove special characters and symbols but keep punctuation and currency symbols and parenthesis
    new_text = re.sub(r'[^a-zA-Z0-9\s,.\'!?£€$()]', '', new_text)

    # normalize repeated punctuation
    new_text = re.sub(r'([!?])\1+', r'\1', new_text)

    # removes << and >> if there is any
    new_text = new_text.replace("<<", "").replace(">>", "")

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
