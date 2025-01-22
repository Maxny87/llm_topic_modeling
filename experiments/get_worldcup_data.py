import pandas as pd
import random
import os
import re
from nltk.corpus import stopwords


def get_text(seed, directory='D:\\topic_modeling_research\\data\\worldcup_data', n=None):
    """
    This is our own datasets collected from twitter using the tweepy library.

    parameters:
        seed is the random seed to use for reproducibility
        n is the number of tweets to get 
    """

    random.seed(seed)
    dataframes = []

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        data = pd.read_csv(filepath, dtype={'coordinates': str, 'place': str})
        dataframes.append(data)

    combined_data = pd.concat(dataframes, ignore_index=False)
    text_data = [row.text for index, row in combined_data.iterrows()]

    if n is None:
        return text_data
    else:
        return random.sample(population=text_data, k=n)


def get_preprocessed_text(seed, directory='D:\\topic_modeling_research\\data\\worldcup_data', n=None,
                          remove_stop_words=True, save_dataset=False):
    data = get_text(seed, directory=directory, n=n)
    clean_data = [preprocess_text(text, remove_stop_words) for text in data]
    clean_dataset = [str(doc) for doc in clean_data if isinstance(doc, str) and doc.strip() != '']

    if save_dataset:
        clean_df = pd.DataFrame(clean_dataset, columns=['text'])
        clean_df.to_csv('./data/clean_worldcup_final.csv', escapechar='\\', index=False)

    return clean_dataset


def get_preprocessed_data_from_csv(file_path):
    data = pd.read_csv(file_path)["text"].to_list()
    return data


def preprocess_text(text, remove_stop_words):
    stop_words = set(stopwords.words('english'))

    # Removes new lines and tabs
    new_text = text.replace("\n", " ")
    new_text = new_text.replace("\t", " ")

    # removes all email addresses
    new_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\b', '', new_text)

    # removes all handles
    new_text = re.sub(r'@\w+', '', new_text)

    # new_text = re.sub(r'#\w+', '', new_text) removes hashtags if wanted

    # Remove URLs
    new_text = re.sub(r'\b(http|https)://\S+|www\.\S+|\S+\.(com|org|net|edu)\b', '', new_text)

    # remove special characters and symbols but keep punctuation and currency symbols and parenthesis
    new_text = re.sub(r'[^a-zA-Z0-9\s,.\'!?£€$()/]', '', new_text)

    # convert to lowercase
    new_text = new_text.lower()

    # removes empty parentheses
    new_text = re.sub(r'\(\s*\)', '', new_text)

    # optionally remove stop words
    if remove_stop_words:
        new_text = " ".join([word for word in new_text.split() if word not in stop_words])

    # strip any extra whitespace
    new_text = " ".join(new_text.split())

    return new_text
