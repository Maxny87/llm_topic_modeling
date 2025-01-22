import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams, word_tokenize


def preprocess_text(text):
    # lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Removes new lines and tabs
    new_text = text.replace("\n", " ")
    new_text = new_text.replace("\t", " ")

    # removes all email addresses
    new_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\b', '', new_text)

    # removes all handles
    new_text = re.sub(r'@\w+', '', new_text)

    # # removes @ symbols but not the text with them
    # new_text = re.sub('@', '', new_text)

    # Removes all text with hashtags
    new_text = re.sub(r'#\w+', '', new_text)

    # Remove URLs
    new_text = re.sub(r'http\S+|www.\S+|\S+.com', '', new_text)

    # removes << and >> if there is any
    new_text = new_text.replace("<<", "").replace(">>", "")

    # removing LaTeX symbols and everything between them
    new_text = re.sub(r'\$.*?\$', '', new_text)

    # removes special characters symbols like (@#), punctuation, and any non-alphanumeric characters that are not
    # whitespace (like tabs) will also remove emojis since they are considered special characters
    new_text = re.sub(r'[^a-zA-Z0-9\s]', '', new_text)

    # convert to lowercase
    new_text = new_text.lower()

    # strip and rejoin to ensure single spacing
    new_text = " ".join([word for word in new_text.split() if word not in stop_words])

    # """ The maketrans method takes three parameters: x, y, and z. It returns a translation table that maps each
    # character in x to the character at the same position in y. If z is given, it specifies the characters to be
    # deleted from the input string. The translate method takes one parameter: table. It returns a copy of the input
    # string where each character has been mapped according to the table. If a character is not found in the table,
    # it is left unchanged. """ Fastest way to remove punctuation custom_punctuation = string.punctuation.replace(
    # '-','') If we do not want to remove the hyphen in between words. replace method is used to create a new string
    # that is a copy of string.punctuation, but with the hyphen removed. translator = str.maketrans("", "",
    # custom_punctuation) new_text = new_text.translate(translator) cleaned_text = ' '.join( [lemmatizer.lemmatize(
    # word) for word in word_tokenize(new_text.lower()) if word not in stop_words])

    return new_text
