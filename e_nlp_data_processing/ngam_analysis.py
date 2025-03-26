import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import spacy


nltk.download('stopwords')
nltk.download('wordnet')

def generate_ngram_summary_csv(df, text_column="text", label_column="label",
                               output_csv="sl_data_for_dashboard/ngram_summary.csv",
                               ngram_range=(1, 3), stop_words="english", progress_fn=print):
    """
    Creates a CSV summary of n-grams (words/phrases) with counts per class for reuse.
    Args:
        df (pd.DataFrame): DataFrame with preprocessed text
        text_column (str): Column containing text
        label_column (str): Binary label column (1 = Real, 0 = Dubious)
        output_csv (str): Path to save summary CSV
        ngram_range (tuple): Range of n-gram sizes
        stop_words (str or list): Stop words to exclude
        progress_fn (function): Optional progress callback (print default)
    Returns:
        pd.DataFrame: Generated summary DataFrame
    """
    df[text_column] = df[text_column].astype(str).fillna("")

    lemmatizer = WordNetLemmatizer()
    nltk_stop_words = set(stopwords.words('english'))

    def preprocess_1gram(text):
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in nltk_stop_words]
        return ' '.join(words)

    if ngram_range[0] == 1:
        progress_fn("🔄 Preprocessing 1-grams: removing stop words and lemmatizing...")
        df[text_column] = df[text_column].apply(preprocess_1gram)

    vectorizers = {
        n: CountVectorizer(ngram_range=(n, n), stop_words=stop_words if n > 1 else None)
        for n in range(ngram_range[0], ngram_range[1] + 1)
    }

    def get_counts(sub_df, n):
        vec = vectorizers[n]
        X = vec.fit_transform(sub_df[text_column])
        return pd.Series(X.sum(axis=0).A1, index=vec.get_feature_names_out())

    summary_rows = []
    for n in vectorizers:
        progress_fn(f"🔍 Processing {n}-grams...")
        real_counts = get_counts(df[df[label_column] == 1], n)
        dubious_counts = get_counts(df[df[label_column] == 0], n)

        all_terms = set(real_counts.index) | set(dubious_counts.index)
        for term in all_terms:
            summary_rows.append({
                "term": term,
                "count_real": real_counts.get(term, 0),
                "count_dubious": dubious_counts.get(term, 0),
                "in_real": real_counts.get(term, 0) > 0,
                "in_dubious": dubious_counts.get(term, 0) > 0,
                "ngram_size": n,
                "is_phrase": n > 1
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df["total_count"] = summary_df["count_real"] + summary_df["count_dubious"]
    summary_df["relevance_score"] = (
        abs(summary_df["count_real"] - summary_df["count_dubious"]) /
        summary_df["total_count"]
    )

    # remove noise
    summary_df = summary_df[summary_df["total_count"] > 10]
    summary_df = summary_df[summary_df["relevance_score"] > 0.5]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    progress_fn(f"✅ N-gram summary saved to: {output_csv}")
    return summary_df

# end of ngam_analysis.py
# Path: sl_components/ngam_analysis.py

# === Example usage ===
if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv("sl_data_for_dashboard/preprocessed_wordcloud.zip")

    # Generate n-gram summary
    summary = generate_ngram_summary_csv(df, output_csv="sl_data_for_dashboard/ngram_summary.zip")
    print(summary.head())