from sl_utils.logger import datapipeline_logger as logger, log_function_call
import os
import csv
import zipfile
import io
import re
import ast
import pandas as pd
from tqdm.auto import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import spacy

# attach tqdm to pandas
tqdm.pandas()

# Download necessary NLTK resources (if you haven't already)
nltk.download('stopwords')
nltk.download('wordnet')

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


@log_function_call(logger)
def checkdirectory():
    current_dir = os.getcwd()
    current_dir

    # check current directory contains the file README.md
    if os.path.exists("README.md"):
        print("The file README.md exists in the current directory")
    else:
        print("The file README.md does not exist in the current directory")
        print("You are in the directory: ", current_dir)
        print("Changing current directory to its parent directory")
        os.chdir(os.path.dirname(current_dir))
        print("You set a new current directory")
        current_dir = os.getcwd()
        if os.path.exists("README.md"):
            print("The file README.md exists in the current directory")
        else:
            RuntimeError("The file README.md does not exist in the"
                         " current directory, please"
                         " check the current directory")
            print("Current Directory =", current_dir)


@log_function_call(logger)
def save_dataframe_to_zip(df, zip_filename, csv_filename='data.csv'):
    logger.info(f"Saving Df to {zip_filename}")
    """Saves a pandas DataFrame to a zipped CSV file.

    Args:
        df: The pandas DataFrame to save.
        zip_filename: The name of the zip file to create.
        csv_filename: The name of the CSV file inside the zip archive.
    """
    # save the dataframe to a csv file
    df.to_csv(csv_filename, index=False)
    # Create an in-memory buffer for the CSV data
    csv_buffer = io.StringIO()
    # Save the DataFrame to the buffer as a CSV
    df.to_csv(csv_buffer,
              index=True,
              index_label="index",
              quoting=csv.QUOTE_NONNUMERIC
              )  # index=False to exclude the index
    # Create a zip file and add the CSV data to it
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(csv_filename, csv_buffer.getvalue())


@log_function_call(logger)
def separate_string(input_string):
    # Extract the contents of the brackets
    bracket_content = re.search(r'\((.*?)\)', input_string).group(1)
    # Extract the remaining text excluding the
    # brackets and the hyphen or minus sign
    remaining_text = re.sub(r'\(.*?\)| -', '', input_string).strip()
    return remaining_text, bracket_content


@log_function_call(logger)
def clean_text(text):
    """Cleans the input text."""
    # check that passed text is a string
    if not isinstance(text, str):
        return ""  # Return empty string for non-string input
    # Remove HTML tags (if any)
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenization and stop word removal
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Rejoin the cleaned words
    cleaned_text = ' '.join(words)
    # return the cleaned text
    return cleaned_text


# Sentiment analysis on the cleaned text:
def get_sentiment(text):
    # check that passed text is a string
    if isinstance(text, str):
        # turn into a TextBlob object
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    else:
        return None, None


# Functions to categorize the polarity and subjectivity scores
def categorize_polarity(polarity):
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'


def categorize_subjectivity(subjectivity):
    if subjectivity > 0.8:
        return 'highly subjective'
    elif subjectivity > 0.6:
        return 'subjective'
    elif subjectivity > 0.4:
        return 'neutral'
    elif subjectivity > 0.2:
        return 'objective'
    else:
        return 'higly objective'


# covert a string to a list
def string_to_list(location_str):
    """Safely converts a string representation of a list to a list."""
    try:
        return ast.literal_eval(location_str)
    except (ValueError, SyntaxError, TypeError):
        return []


# define media types
def classify_media(text):
    media_dict = {'video': ['video', 'watch', 'live',
                            'stream', 'youtube',
                            'vimeo', 'twitch'],
                  'audio': ['audio', 'listen', 'podcast', 'radio'],
                  'image': ['image', 'photo', 'picture', 'gif'],
                  'infographic': ['infographic'],
                  'poll': ['poll'],
                  'twitter': ['twitter', 'X', 'x', 'tweet', 'retweeted'],
                  'facebook': ['facebook', 'fb', 'like', 'share', 'comment'],
                  'instagram': ['instagram', 'ig', ],
                  'linkedin': ['linkedin', 'share'],
                  }
    # Handle NaN cases safely
    if pd.isna(text):
        return 'text'
    for key, value in media_dict.items():
        # Lowercase to ensure case insensitivity
        if any(word in text.lower() for word in value):
            return key
    # Default to 'text' if no media type is found
    return 'text'
