import pandas as pd
import re
from nltk.corpus import stopwords
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import os
from sklearn.datasets import fetch_20newsgroups

stop_words = set(stopwords.words('english')) # stopwords from NLTK

def preprocess_amazon(text, remove_stop_words):
    """
    This function was used to preprocess the Amazon Reviews Electronics dataset for the paper

    params:
        text: the review you want to preprocess
        remove_stop_words: boolean on whether to remove stop words
    """

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

    return new_text # return the preprocessed doc

def get_amazon_data(file_path, fraction_to_sample, chunk_size, seed, preprocessed, remove_stopwords=True,
                    save_dataset=False, save_file_path=None):
    """
    This function is used to return the Amazon Electronic Reviews dataset for the paper

    Because of computation constraints, we used pandas to load the data into chunks and sample with a fixed seed from each chunk. We then
    concat all the samples to make the final dataset and then filter them to make sure they are all English.

    Params:
        file_path: path to the file containing the reviews - should be a JSON file
        fraction_to_sample: % of reviews to sample from each chunk
        chunk_size: how many reviews to sample from per chunk
        seed: random seed for chunking and sampling
        preprocessed: boolean on whether to preprocess the data
        remove_stopwords: boolean on whether to remove stop words
        save_dataset: boolean on whether to save the dataset
        save_file_path: path to save the dataset
    """

    DetectorFactory.seed = seed # seed for the chunk
    chunks = [] # holds a chunk of the dataset

    for chunk in pd.read_json(file_path, lines=True, chunksize=chunk_size): # cutting the dataset into chunks and reading in chunks and sampling per chunk for the reduced dataset
        sample = chunk.sample(frac=fraction_to_sample, random_state=seed)
        chunks.append(sample)

    reviews = pd.concat(chunks, ignore_index=True) # concat the reviews

    # making sure we only use english reviews
    english_reviews = []

    for review in reviews['text']:
        try:
            if detect(review) == 'en':
                english_reviews.append(review)
        except LangDetectException:
            # language detection fails
            pass

    if preprocessed: # if preprocessed  is supposed to be returned use the preprocess helper function
        english_reviews = [preprocess_amazon(t, remove_stopwords) for t in english_reviews]
        english_reviews = [str(doc) for doc in english_reviews if isinstance(doc, str) and doc.strip() != '']

    if save_dataset:
        df_english = pd.DataFrame(english_reviews, columns=['text'])
        df_english.to_csv(f'{save_file_path}.csv', escapechar='\\', index=False)

    return english_reviews

def preprocess_arxiv(text, remove_stop_words):
    """
    This function was used to preprocess the Arxiv Abstracts dataset for the paper

    params:
        text: the abstract you want to preprocess
        remove_stop_words: boolean on whether to remove stop words
    """

    # removes new lines and tabs
    new_text = text.replace("\n", " ").replace("\t", " ")

    # replace LaTeX commands with the appropriate representation
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

    for pattern, replacement in replacements.items(): # replacing them
        new_text = re.sub(pattern + r'\b', replacement, new_text)

    # remove non-essential LaTeX commands but keep the content inside curly braces
    new_text = re.sub(r'\\[a-zA-Z]+\{(.*?)\}', r'\1', new_text)

    # remove any remaining curly braces
    new_text = re.sub(r'\{|\}', '', new_text)

    # remove special characters and symbols but keep punctuation and math operators
    new_text = re.sub(r'[^a-zA-Z0-9\s,.\'!?()<>^_+\-=/|*%]', '', new_text)

    # remove all email addresses
    new_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\b', '', new_text)

    # remove URLs
    new_text = re.sub(r'\b(http|https)://\S+|www\.\S+|\S+\.(com|org|net|edu)\b', '', new_text)

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

def get_arxiv_data(json_file_path, preprocessed, remove_stop_words=True, save_dataset=False, save_file_path=None):
    """
    This function is used to return the Arxiv Abstracts dataset for the paper

    Params:
        file_path: path to the file containing the abstracts - should be a JSON file
        preprocessed: boolean on whether to preprocess the data
        remove_stopwords: boolean on whether to remove stop words
        save_dataset: boolean on whether to save the dataset
        save_file_path: path to save the dataset
    """

    data = pd.read_json(json_file_path, lines=True)['abstract'].to_list()

    if preprocessed:
        data = [preprocess_arxiv(t, remove_stop_words) for t in data]
        data = [str(doc) for doc in data if isinstance(doc, str) and doc.strip() != '']

    if save_dataset:
        df = pd.DataFrame(data, columns=['text'])
        df.to_csv(f'{save_file_path}.csv', escapechar='\\', index=False)

    return data

def preprocess_bbc(text, remove_stop_words):
    """
    This function was used to preprocess the BBC News dataset for the paper

    params:
        text: the news you want to preprocess
        remove_stop_words: boolean on whether to remove stop words
    """

    # add a period to the header
    first_newline_index = text.find('\n')
    text = text[:first_newline_index] + '.' + text[first_newline_index:]

    # removes new lines and tabs
    new_text = text.replace("\n", " ").replace("\t", " ")

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

def get_bbc_data(data_folder_path, preprocessed, remove_stop_words=True, save_dataset=False, save_file_path=None):
    """
    This function is used to return the BBC News dataset for the paper.

    The data is structured where each sector the news covers gets its own folder with txt files inside a parent directory. To get the data,
    we navigate to the data folder path, which is the parent folder containing the 5 sector subdirectories, and we iterate over each subdirectory. In each
    subdirectory, we open the txt file and save it as a string and append it to a list of strings.

    Params:
        data_folder_path: path to the folder containing the bbc news data and subdirectories
        preprocessed: boolean on whether to preprocess the data
        remove_stopwords: boolean on whether to remove stop words
        save_dataset: boolean on whether to save the dataset
        save_file_path: path to save the dataset
    """

    data = [] # list of strings

    for folder in os.listdir(data_folder_path): # iterating over folders in parent directory
        path = os.path.join(data_folder_path, folder)

        # checking to make sure it is a directory
        if os.path.isdir(path):
            for file in os.listdir(path): # for file in folder
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
    """
    This function was used to preprocess the Amazon Reviews Electronics dataset for the paper

    params:
        text: the newsgroup post you want to preprocess
        remove_stop_words: boolean on whether to remove stop words
    """

    # removes the header for each document - header is before the first double \n\n
    header_end = text.find('\n\n')
    if header_end != -1:
        text = text[header_end:].strip()

    # removes new lines and tabs
    new_text = text.replace("\n", " ").replace("\t", " ")

    # removes all email addresses
    new_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\b', '', new_text)

    # remove text within angle brackets
    new_text = re.sub(r'<.*?>', '', new_text)

    # removing reoccurring special characters from the text
    new_text = re.sub(r'[<>|]', '', new_text)
    new_text = new_text.replace('-', ' ')

    # remove special characters and symbols but keep punctuation and currency symbols and parenthesis
    new_text = re.sub(r'[^a-zA-Z0-9\s,.\'!?()/$]', '', new_text)

    # remove URLs
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

def get_newsgroup20_data(preprocessed, remove_stop_words=True, save_dataset=False, save_file_path=None):
    """
    This function is used to return the Newsgroup20 dataset for the paper. Since it just uses the Sklearn fetch function, there
    is no need to pass a file path.

    Params:
        preprocessed: boolean on whether to preprocess the data
        remove_stopwords: boolean on whether to remove stop words
        save_dataset: boolean on whether to save the dataset
        save_file_path: path to save the dataset
    """

    data = fetch_20newsgroups(subset='all')['data'] # retrieving data from sklearn

    if preprocessed:
        data = [preprocess_newsgroup20(t, remove_stop_words) for t in data]
        data = [str(doc) for doc in data if isinstance(doc, str) and doc.strip() != '']

    if save_dataset:
        df = pd.DataFrame(data, columns=['text'])
        df.to_csv(f'{save_file_path}.csv', escapechar='\\', index=False)

def preprocess_worldcup_tweets(text, remove_stop_words=True):
    """
    This function was used to preprocess the #Worldcup2022 Tweets dataset for the paper

    params:
        text: the tweet you want to preprocess
        remove_stop_words: boolean on whether to remove stop words
    """

    # removes new lines and tabs
    new_text = text.replace("\n", " ").replace("\t", " ")

    # removes all email addresses
    new_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\b', '', new_text)

    # removes all handles
    new_text = re.sub(r'@\w+', '', new_text)

    # new_text = re.sub(r'#\w+', '', new_text) removes hashtags if wanted

    # remove URLs
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

def get_worldcup_data(directory, preprocess, remove_stop_words=True, save_dataset=False, save_file_path=None):
    """
    This function is used to return the #WorldCup2022 Tweets dataset for the paper. The data is a folder of csv files. We iterate over each csf file in the directory
    and read in the data into a pandas dataframe and save the dataframe to a list. Then we concat each dataframe to a final dataframe of all the tweets and every feature for each tweet.
    We then extract just the text content from each tweet.

    Params:
        directory: path to the folder containing the csv files
        preprocessed: boolean on whether to preprocess the data
        remove_stopwords: boolean on whether to remove stop words
        save_dataset: boolean on whether to save the dataset
        save_file_path: path to save the dataset
    """

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
