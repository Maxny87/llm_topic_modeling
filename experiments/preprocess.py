import pandas as pd

import re

from nltk.corpus import stopwords

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import os

from sklearn.datasets import fetch_20newsgroups


def get_text_from_csv(csv_filepath):
    return pd.read_csv(csv_filepath)

def base_preprocess(text, remove_stop_words=True):
    stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = text.replace("\n", " ").replace("\t", " ")

    # remove email addresses
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\b', '', text)

    # remove URLs
    text = re.sub(r'\b(http|https)://\S+|www\.\S+|\S+\.(com|org|net|edu)\b', '', text)

    if remove_stop_words:
        # text = " ".join([word for word in text.split() if word not in stop_words])
        pattern = r'\b(?:' + '|'.join(stop_words) + r')\b\s*'
        text = re.sub(pattern, '', text)

    # strip extra whitespace
    text = " ".join(text.split())

    return text


def preprocess_amazon(text, remove_stop_words):
    new_text = base_preprocess(text, remove_stop_words)

    # remove HTML tags
    new_text = re.sub(r'<[^>]+>', '', new_text)

    # remove special characters and symbols but keep punctuation and currency symbols and parenthesis
    new_text = re.sub(r'[^a-zA-Z0-9\s,.\'!?£€$()]', '', new_text)

    # normalize repeated punctuation
    new_text = re.sub(r'([!?])\1+', r'\1', new_text)

    # removes << and >> if there is any
    new_text = new_text.replace("<<", "").replace(">>", "")

    # removes empty parentheses
    new_text = re.sub(r'\(\s*\)', '', new_text)

    # strip any extra whitespace
    new_text = " ".join(new_text.split())

    return new_text

def get_amazon_data(file_path, fraction_to_sample, chunk_size, seed, preprocessed, remove_stopwords=True,
                    save_dataset=False, save_file_path=None):
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
            # language detection fails
            pass

    if preprocessed:
        english_reviews = [preprocess_amazon(t, remove_stopwords) for t in english_reviews]
        english_reviews = [str(doc) for doc in english_reviews if isinstance(doc, str) and doc.strip() != '']

    if save_dataset:
        df_english = pd.DataFrame(english_reviews, columns=['text'])
        df_english.to_csv(f'{save_file_path}.csv', escapechar='\\', index=False)

    return english_reviews

def preprocess_arxiv(text, remove_stop_words):
    new_text = base_preprocess(text, remove_stop_words)

    replacements = {
        r'\\alpha': 'alpha', r'\\beta': 'beta', r'\\gamma': 'gamma', r'\\delta': 'delta',
        r'\\epsilon': 'epsilon', r'\\zeta': 'zeta', r'\\eta': 'eta', r'\\theta': 'theta',
        r'\\iota': 'iota', r'\\kappa': 'kappa', r'\\lambda': 'lambda', r'\\mu': 'mu',
        r'\\nu': 'nu', r'\\xi': 'xi', r'\\omicron': 'omicron', r'\\pi': 'pi',
        r'\\rho': 'rho', r'\\sigma': 'sigma', r'\\tau': 'tau', r'\\upsilon': 'upsilon',
        r'\\phi': 'phi', r'\\chi': 'chi', r'\\psi': 'psi', r'\\omega': 'omega',
        r'\\text': '', r'\\frac': 'frac', r'\\sum': 'sum', r'\\int': 'int',
        r'\\lim': 'lim', r'\\infty': 'infinity', r'\$': ''
    }

    for pattern, replacement in replacements.items():
        new_text = re.sub(pattern + r'\b', replacement, new_text)

    new_text = re.sub(r'\\[a-zA-Z]+\{(.*?)\}', r'\1', new_text)

    new_text = re.sub(r'\{|\}', '', new_text)

    # remove special characters and symbols but keep punctuation and math operators
    new_text = re.sub(r'[^a-zA-Z0-9\s,.\'!?()<>^_+\-=/|*%]', '', new_text)

    # removes empty parentheses
    new_text = re.sub(r'\(\s*\)', '', new_text)

    # strip any extra whitespace
    new_text = " ".join(new_text.split())

    return new_text

def get_arxiv_data(json_file_path, preprocessed, remove_stop_words=True, save_dataset=False, save_file_path=None):
    data = pd.read_json(json_file_path, lines=True)['abstract'].to_list()

    if preprocessed:
        data = [preprocess_arxiv(t, remove_stop_words) for t in data]
        data = [str(doc) for doc in data if isinstance(doc, str) and doc.strip() != '']

    if save_dataset:
        df = pd.DataFrame(data, columns=['text'])
        df.to_csv(f'{save_file_path}.csv', escapechar='\\', index=False)

    return data

def preprocess_bbc(text, remove_stop_words):
    # add a period to the header
    first_newline_index = text.find('\n')
    text = text[:first_newline_index] + '.' + text[first_newline_index:]

    new_text = base_preprocess(text, remove_stop_words)

    # removes all handles
    new_text = re.sub(r'@\w+', '', new_text)

    # remove special characters and symbols but keep punctuation and currency symbols and parenthesis
    new_text = re.sub(r'[^a-zA-Z0-9\s,.\'!?£€$()]', '', new_text)

    # removes empty parentheses
    new_text = re.sub(r'\(\s*\)', '', new_text)

    # strip any extra whitespace
    new_text = " ".join(new_text.split())

    return new_text

def get_bbc_data(data_folder_path, preprocessed, remove_stop_words=True, save_dataset=False, save_file_path=None):
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

    if preprocessed:
        data = [preprocess_bbc(t, remove_stop_words) for t in data]
        data = [str(doc) for doc in data if isinstance(doc, str) and doc.strip() != '']

    if save_dataset:
        df = pd.DataFrame(data, columns=['text'])
        df.to_csv(f'{save_file_path}.csv', escapechar='\\', index=False)

    return data

def preprocess_newsgroup20(text, remove_stop_words=True):
    # removes the header for each document
    header_end = text.find('\n\n')
    if header_end != -1:
        text = text[header_end:].strip()

    new_text = base_preprocess(text, remove_stop_words)

    new_text = re.sub(r'<.*?>', '', new_text)  # remove text within angle brackets

    # removing reoccurring special characters from the text
    new_text = re.sub(r'[<>|]', '', new_text)
    new_text = new_text.replace('-', ' ')

    # remove special characters and symbols but keep punctuation and currency symbols and parenthesis
    new_text = re.sub(r'[^a-zA-Z0-9\s,.\'!?()/$]', '', new_text)

    # removes << and >> if there is any
    new_text = new_text.replace("<<", "").replace(">>", "")

    # normalize repeated punctuation
    new_text = re.sub(r'([!?])\1+', r'\1', new_text)

    # remove extra periods
    new_text = re.sub(r'\.{2,}', ' ', new_text)

    # removes empty parentheses
    new_text = re.sub(r'\(\s*\)', '', new_text)

    # strip any extra whitespace
    new_text = " ".join(new_text.split())

    return new_text

def get_newsgroup20_data(preprocessed, remove_stop_words=True, save_dataset=False, save_file_path=None):
    data = fetch_20newsgroups(subset='all')

    if preprocessed:
        data = [preprocess_newsgroup20(t, remove_stop_words) for t in data]
        data = [str(doc) for doc in data if isinstance(doc, str) and doc.strip() != '']

    if save_dataset:
        df = pd.DataFrame(data, columns=['text'])
        df.to_csv(f'{save_file_path}.csv', escapechar='\\', index=False)

def preprocess_worldcup_tweets(text, remove_stop_words=True):
    new_text = base_preprocess(text, remove_stop_words)

    # removes all handles
    new_text = re.sub(r'@\w+', '', new_text)

    # remove special characters and symbols but keep punctuation and currency symbols and parenthesis
    new_text = re.sub(r'[^a-zA-Z0-9\s,.\'!?£€$()/]', '', new_text)

    # removes empty parentheses
    new_text = re.sub(r'\(\s*\)', '', new_text)

    # strip any extra whitespace
    new_text = " ".join(new_text.split())

    return new_text

def get_worldcup_data(directory, preprocess, remove_stop_words=True, save_dataset=False, save_file_path=None):
    dataframes = []

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        data = pd.read_csv(filepath, dtype={'coordinates': str, 'place': str})
        dataframes.append(data)

    combined_data = pd.concat(dataframes, ignore_index=False)
    text_data = [row.text for index, row in combined_data.iterrows()]

    if preprocess:
        text_data = [preprocess_worldcup_tweets(t, remove_stop_words) for t in text_data]
        text_data = [str(doc) for doc in text_data if isinstance(doc, str) and doc.strip() != '']

    if save_dataset:
        df = pd.DataFrame(text_data, columns=['text'])
        df.to_csv(f'{save_file_path}.csv', escapechar='\\', index=False)

    return text_data
