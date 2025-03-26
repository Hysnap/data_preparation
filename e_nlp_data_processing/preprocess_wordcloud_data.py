import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


def clean_text(text, stop_words, remove_stopwords=True, remove_punctuation=True):
    # Lowercase the text
    text = text.lower()

    # Remove non-alphabetic characters if remove_punctuation is True
    if remove_punctuation:
        text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize and optionally remove stopwords
    tokens = text.split()
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    return " ".join(tokens)


def preprocess_for_wordcloud(input_path,
                             output_path=None,
                             label_col="label",
                             text_col="cleaned_text",
                             title_col=None,
                             remove_stopwords=True,
                             remove_punctuation=True):
    """
    Preprocess the dataset to generate text for word cloud visualizations.

    Args:
        input_path (str): Path to the input CSV file.
        output_path (str, optional): Path to save the preprocessed DataFrame (as CSV).
        label_col (str): Column name identifying true (1) vs false (0) articles.
        text_col (str): Column name containing the cleaned body text.
        title_col (str): Column name containing the title.
        remove_stopwords (bool): Whether to remove stopwords from the text.
        remove_punctuation (bool): Whether to remove punctuation from the text.
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)

    stop_words = set(stopwords.words("english"))

    # Combine title and text into a unified 'text' field
    if title_col is None:
        '# rename text_col to text'
        df.rename(columns={text_col: 'text'}, inplace=True)
    else:
        print("Combining title and text...")
        df["text"] = df[title_col].fillna("") + " " + df[text_col].astype(str)

    # Apply cleaning
    print("Cleaning text...")
    df["text"] = df["text"].apply(lambda x: clean_text(x,
                                                       stop_words,
                                                       remove_stopwords,
                                                       remove_punctuation))

    # Optional: drop originals
    df = df[[label_col, "text"]]

    # Optional: save output
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to {output_path}")

    return df


# Example usage
if __name__ == "__main__":
    # Update these paths accordingly
    INPUT_FILE = "data/training_data.zip"
    OUTPUT_FILE = "sl_data_for_dashboard/preprocessed_wordcloud.zip"

    # Set options for preprocessing
    REMOVE_STOPWORDS = False
    REMOVE_PUNCTUATION = False

    df_processed = preprocess_for_wordcloud(INPUT_FILE,
                                            OUTPUT_FILE,
                                            remove_stopwords=REMOVE_STOPWORDS,
                                            remove_punctuation=REMOVE_PUNCTUATION)
    print("Preview of processed data:")
    print(df_processed.head())
