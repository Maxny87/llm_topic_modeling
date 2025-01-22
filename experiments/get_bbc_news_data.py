import os

import pandas
import pandas as pd

import re
from nltk.corpus import stopwords


def get_text(data_folder_path='D:\\topic_modeling_research\\data\\bbc'):
    """
    Dataset of 2225 bbc news articles from this link: http://mlg.ucd.ie/datasets/bbc.html

    :param data_folder_path:
    :return:
    """
    data = []

    for folder in os.listdir(data_folder_path):
        path = os.path.join(data_folder_path, folder)

        # checking to make sure it is a directory
        if os.path.isdir(path):
            for file in os.listdir(path):
                # this will be the txt file from the bbc news categories
                file_path = os.path.join(path, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    data.append(content)

    return data


def get_preprocessed_text(remove_stop_words=True, save_dataset=False):
    data = get_text()
    clean_data = [preprocess_doc(t, remove_stop_words) for t in data]
    clean_dataset = [str(doc) for doc in clean_data if isinstance(doc, str) and doc.strip() != '']

    if save_dataset:
        clean_df = pd.DataFrame(clean_dataset, columns=['text'])
        clean_df.to_csv('./data/clean_bbc_news_dataset_final.csv', escapechar='\\', index=False)

    return clean_dataset


def get_preprocessed_data_from_csv(file_path):
    data = pd.read_csv(file_path)["text"].to_list()
    return data


def preprocess_doc(text, remove_stop_words):
    stop_words = set(stopwords.words('english'))

    # add a period to the header
    first_newline_index = text.find('\n')
    text = text[:first_newline_index] + '.' + text[first_newline_index:]

    # Removes new lines and tabs
    new_text = text.replace("\n", " ")
    new_text = new_text.replace("\t", " ")

    # removes all email addresses
    new_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\b', '', new_text)

    # removes all handles
    new_text = re.sub(r'@\w+', '', new_text)

    # remove URLs
    new_text = re.sub(r'\b(http|https)://\S+|www\.\S+|\S+\.(com|org|net|edu)\b', '', new_text)

    # remove special characters and symbols but keep punctuation and currency symbols and parenthesis
    new_text = re.sub(r'[^a-zA-Z0-9\s,.\'!?£€$()]', '', new_text)

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
