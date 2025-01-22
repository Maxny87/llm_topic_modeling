import pandas as pd

import re
from nltk.corpus import stopwords


def get_text(file_path):
    """
    This dataset was downloaded from kaggle: https://www.kaggle.com/datasets/Cornell-University/arxiv/data

    :param file_path:
    :param seed:
    :param chunk_size:
    :param fraction_to_sample:
    :return:
    """
    data = pd.read_json(file_path, lines=True)

    return data['abstract'].to_list()


def get_preprocessed_data(file_path, remove_stop_words=True, save_dataset=False):
    """
    Returns cleaned arxiv data

    :param remove_stop_words:
    :param fraction_to_sample:
    :param chunk_size:
    :param seed:
    :param file_path:
    :return:
    """

    data = get_text(file_path)
    clean_data = [preprocess_arxiv_data(t, remove_stop_words) for t in data]
    clean_dataset = [str(doc) for doc in clean_data if isinstance(doc, str) and doc.strip() != '']

    if save_dataset:
        clean_df = pd.DataFrame(clean_dataset, columns=['text'])
        clean_df.to_csv('./data/clean_arxiv_abstracts_dataset_final.csv', escapechar='\\', index=False)

    return clean_dataset


def get_preprocessed_data_from_csv(file_path):
    data = pd.read_csv(file_path)["text"].to_list()
    return data


def preprocess_arxiv_data(text, remove_stop_words):
    """
    Pass in text abstract to preprocess it

    :param remove_stop_words: Boolean flag to remove stop words.
    :param text: The text to preprocess.
    :return: The preprocessed text.
    """

    stop_words = set(stopwords.words('english'))

    # Removes new lines and tabs
    new_text = text.replace("\n", " ").replace("\t", " ")

    # Replace LaTeX commands with the appropriate representation
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

    # Remove non-essential LaTeX commands but keep the content inside curly braces
    new_text = re.sub(r'\\[a-zA-Z]+\{(.*?)\}', r'\1', new_text)

    # Remove any remaining curly braces
    new_text = re.sub(r'\{|\}', '', new_text)

    # Remove special characters and symbols but keep punctuation and math operators
    new_text = re.sub(r'[^a-zA-Z0-9\s,.\'!?()<>^_+\-=/|*%]', '', new_text)

    # Remove all email addresses
    new_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\b', '', new_text)

    # Remove URLs
    new_text = re.sub(r'\b(http|https)://\S+|www\.\S+|\S+\.(com|org|net|edu)\b', '', new_text)

    # removes empty parentheses
    new_text = re.sub(r'\(\s*\)', '', new_text)

    # Convert to lowercase
    new_text = new_text.lower()

    # Optionally remove stop words
    if remove_stop_words:
        new_text = " ".join([word for word in new_text.split() if word not in stop_words])

    # Strip any extra whitespace
    new_text = " ".join(new_text.split())

    return new_text
