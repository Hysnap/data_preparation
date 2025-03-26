import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sl_utils.logger import log_function_call, streamlit_logger
from sl_visualisations.common_visual_functions import (
    get_dataset_or_error,
    get_stopwords,
    get_cleaned_text,
    generate_wordcloud,
    display_wordcloud,
    )


# ----------------- WordCloud Visualisations -----------------
@log_function_call(streamlit_logger)
def plot_wordcloud_all():
    df = get_dataset_or_error(dataflag="text")
    if df is None:
        return

    all_text = get_cleaned_text(df)
    stop_words = get_stopwords()
    wordcloud = generate_wordcloud(all_text, stop_words)
    display_wordcloud(wordcloud)


@log_function_call(streamlit_logger)
def plot_wordcloud_false():
    df = get_dataset_or_error(dataflag="text")
    if df is None:
        return

    fake_text = get_cleaned_text(df, label=0)
    real_text = get_cleaned_text(df, label=1)
    stop_words = get_stopwords()

    unique_fake_words = " ".join([word for word in fake_text.split()
                                  if word not in real_text.split()])
    wordcloud = generate_wordcloud(unique_fake_words, stop_words)
    display_wordcloud(wordcloud)


@log_function_call(streamlit_logger)
def plot_wordcloud_true():
    df = get_dataset_or_error(dataflag="text")
    if df is None:
        return

    real_text = get_cleaned_text(df, label=1)
    fake_text = get_cleaned_text(df, label=0)
    stop_words = get_stopwords()

    unique_real_words = " ".join([word for word in real_text.split()
                                  if word not in fake_text.split()])
    wordcloud = generate_wordcloud(unique_real_words, stop_words)
    display_wordcloud(wordcloud)


@log_function_call(streamlit_logger)
def plot_wordcloud_common():
    df = get_dataset_or_error(dataflag="text")
    if df is None:
        return

    real_text = get_cleaned_text(df, label=1)
    fake_text = get_cleaned_text(df, label=0)
    stop_words = get_stopwords()

    common_words = " ".join([word for word in real_text.split()
                             if word in fake_text.split()])
    wordcloud = generate_wordcloud(common_words, stop_words)
    display_wordcloud(wordcloud)


@log_function_call(streamlit_logger)
def plot_wordcloud_dubious():
    df = get_dataset_or_error(dataflag="text")
    if df is None:
        return

    fake_text = get_cleaned_text(df, label=0)
    real_text = get_cleaned_text(df, label=1)
    stop_words = get_stopwords()

    common_words = " ".join([word for word in fake_text.split()
                             if word in real_text.split()])
    wordcloud = generate_wordcloud(common_words, stop_words)
    display_wordcloud(wordcloud)


# ----------------- N-Gram Comparison -----------------
@log_function_call(streamlit_logger)
def plot_ngram_comparison():
    df = get_dataset_or_error(dataflag="text")
    if df is None:
        return

    real_df = df[df["label"] == 1]
    dubious_df = df[df["label"] == 0]

    stop_words = get_stopwords()
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words=stop_words)

    real_matrix = vectorizer.fit_transform(real_df["title"].astype(str))
    real_ngrams = pd.DataFrame({
        "ngram": vectorizer.get_feature_names_out(),
        "count": real_matrix.sum(axis=0).A1
    }).sort_values("count", ascending=False)

    dubious_matrix = vectorizer.fit_transform(dubious_df["title"].astype(str))
    dubious_ngrams = pd.DataFrame({
        "ngram": vectorizer.get_feature_names_out(),
        "count": dubious_matrix.sum(axis=0).A1
    }).sort_values("count", ascending=False)

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    sns.barplot(data=real_ngrams.head(10), x="count",
                y="ngram", ax=ax[0], color="green")
    sns.barplot(data=dubious_ngrams.head(10), x="count",
                y="ngram", ax=ax[1], color="red")

    ax[0].set_title("Most Common N-Grams in Real Articles", fontsize=10)
    ax[1].set_title("Most Common N-Grams in Dubious Articles", fontsize=10)

    for a in ax:
        a.set_xlabel("Count", fontsize=8)
        a.set_ylabel("N-Gram", fontsize=8)
        a.tick_params(axis='x', labelsize=8)
        a.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    st.pyplot(fig)

# end of wordcloud_visuals.py
# Path: sl_visualisations/boxplot_visuals.py
