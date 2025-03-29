# ngram_summary_pipeline.py

import os
import gc
import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import re

# Load SpaCy model once globally
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
STOP_WORDS = nlp.Defaults.stop_words


def compute_dynamic_threshold(n, num_rows):
    """
    Returns a dynamic min_count threshold based on n-gram size and dataset size.
    """
    min_fraction = {
        1: 0.005,   # 0.5% for unigrams
        2: 0.003,   # 0.3% for bigrams
        3: 0.002    # 0.2% for trigrams
    }
    hard_floor = {
        1: 10,
        2: 5,
        3: 3
    }
    frac = min_fraction.get(n, 0.002)
    floor = hard_floor.get(n, 3)
    return max(int(num_rows * frac), floor)


def preprocess_texts_spacy(texts, batch_size=1000):
    """Preprocess text using SpaCy: preserve contractions, no lemmatization, remove non-word tokens."""
    # Normalize smart quotes to plain apostrophes
    texts = [text.replace("’", "'").replace("‘", "'") for text in texts]

    docs = nlp.pipe(texts, batch_size=batch_size)
    cleaned_texts = []

    for doc in docs:
        tokens = []
        for token in doc:
            text = token.text.lower()
            # keep words like "didn't", "shouldn't", "won't"
            if re.fullmatch(r"[a-zA-Z]+('[a-zA-Z]+)?", text):
                if not token.is_stop:  # optionally remove this line if you want to keep stopwords too
                    tokens.append(text)
        cleaned_texts.append(" ".join(tokens))

    return cleaned_texts


def safe_stats(array):
    values = array[array > 0]
    if len(values) == 0:
        return 0, 0.0, 0.0
    if len(values) == 1:
        return values[0], float(values[0]), 0.0
    return values.max(), values.mean(), values.std(ddof=0)


def compute_ngram_row(term, idx, X, is_real, is_dubious,
                      n, vocab_filter, min_count_threshold):
    col = X[:, idx].toarray().flatten()
    real_counts = col[is_real]
    dubious_counts = col[is_dubious]

    count_real = real_counts.sum()
    count_dubious = dubious_counts.sum()
    total_count = count_real + count_dubious

    if (n == 1 and total_count < min_count_threshold) or (n > 1 and total_count < 5):
        return None
    # if n > 1 and not any(base in vocab_filter for base in term.split()):
    #     return None

    if n == 1:
        vocab_filter.add(term)

    max_r, mean_r, std_r = safe_stats(real_counts)
    max_d, mean_d, std_d = safe_stats(dubious_counts)

    row = {
        "term": term,
        "ngram_size": n,
        "is_phrase": n > 1,
        "count_real": int(count_real),
        "count_dubious": int(count_dubious),
        "total_count": int(total_count),
        "in_real": count_real > 0,
        "in_dubious": count_dubious > 0,
        "max_real": int(max_r),
        "mean_real": round(mean_r, 4),
        "stdev_real": round(std_r, 4),
        "max_dubious": int(max_d),
        "mean_dubious": round(mean_d, 4),
        "stdev_dubious": round(std_d, 4),
        "relevance_score": round(abs(count_real - count_dubious) / total_count,
                                 4) if total_count else 0.0
    }

    return row


def process_ngram(df, n, vocab_filter,
                  text_column, label_column,
                  output_csv, min_count_threshold,
                  write_header, progress_fn, n_jobs=4):
    progress_fn(f"🔍 Processing {n}-grams...")

    if n == 1:
        df = df.copy()
        df.loc[:, "__cleaned"] = preprocess_texts_spacy(df[text_column])
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df["__cleaned"])
        terms = vectorizer.get_feature_names_out()
        term_indices = {term: i for i, term in enumerate(terms)}
    else:
        # Initial n-gram vectorizer
        vectorizer = CountVectorizer(ngram_range=(n, n), stop_words=None)
        X = vectorizer.fit_transform(df[text_column])
        terms = vectorizer.get_feature_names_out()

        # Stopword-based filtering
        max_stop_words = n - 1
        filtered_terms = [
            term for term in terms
            if sum(1 for word in term.split() if word in STOP_WORDS) <= max_stop_words
        ]

        progress_fn(f"🧹 Filtered {len(terms) - len(filtered_terms)} / {len(terms)} {n}-gram terms due to stopwords")

        # Use the full X matrix and just filter term indices
        term_indices = {
            term: vectorizer.vocabulary_[term]
            for term in filtered_terms if term in vectorizer.vocabulary_
        }

    # Label mask vectors
    is_real = df[label_column].values == 1
    is_dubious = df[label_column].values == 0

    # Compute stats
    dynamic_threshold = compute_dynamic_threshold(n, len(df))
    progress_fn(f"📊 Using dynamic threshold: {dynamic_threshold} for {n}-grams ({len(df)} documents)")

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_ngram_row)(
            term,
            idx,
            X,
            is_real,
            is_dubious,
            n,
            vocab_filter.copy(),
            dynamic_threshold
        )
        for term, idx in tqdm(term_indices.items(),
                              desc=f"{n}-gram terms",
                              leave=False)
    )

    summary_rows = [r for r in results if r]

    if n == 1:
        vocab_filter.update(r["term"] for r in summary_rows)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_csv,
                          mode='a',
                          index=False,
                          header=write_header)
        progress_fn(f"✅ Appended {len(summary_rows)} {n}-grams to {output_csv}")
    else:
        progress_fn(f"⚠️ No {n}-grams met the threshold.")



def generate_ngram_summary_csv(
    df,
    text_column="text",
    label_column="label",
    output_csv="sl_data_for_dashboard/ngram_summary.csv",
    ngram_range=(1, 3),
    progress_fn=print,
    min_count_threshold=25,
        n_jobs=4):
    df = df.copy()  # ensures df is not a view/slice
    df.loc[:, text_column] = df[text_column].astype(str).fillna("")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    if os.path.exists(output_csv):
        os.remove(output_csv)

    vocab_filter = set()
    write_header = True

    for n in range(ngram_range[0], ngram_range[1] + 1):
        process_ngram(df,
                      n,
                      vocab_filter,
                      text_column,
                      label_column,
                      output_csv,
                      min_count_threshold,
                      write_header,
                      progress_fn,
                      n_jobs=n_jobs)
        write_header = False
        gc.collect()

    progress_fn(f"📁 All n-grams saved to {output_csv}")


def run_full_ngram_pipeline(input_csv,
                            output_dir,
                            text_column="text",
                            label_column="label",
                            ngram_range=(1, 3),
                            min_count_threshold=100,
                            test=False,
                            n_jobs=4):
    def progress(msg):
        print(msg)

    print(f"📦 Loading data from {input_csv}...")
    df = pd.read_csv(input_csv, low_memory=False)

    # if test= True, create a sample of 10,000 rows
    if test:
        if df.shape[0] > 10000:
            df = df.sample(n=10000, random_state=42)
        else:
            df = df.copy()

    if "year" in df.columns:
        for year in sorted(df["year"].dropna().unique()):
            year = int(year)
            subset = df[df["year"] == year]
            if subset.empty:
                continue
            output_csv = os.path.join(output_dir, f"ngram_summary_{year}.csv")
            generate_ngram_summary_csv(
                df=subset,
                text_column=text_column,
                label_column=label_column,
                output_csv=output_csv,
                ngram_range=ngram_range,
                progress_fn=progress,
                min_count_threshold=min_count_threshold,
                n_jobs=n_jobs
            )
            progress(f"📁 Year {year} complete.")
    else:
        output_csv = os.path.join(output_dir, "ngram_summary.csv")
        generate_ngram_summary_csv(
            df=df,
            text_column=text_column,
            label_column=label_column,
            output_csv=output_csv,
            ngram_range=ngram_range,
            progress_fn=progress,
            min_count_threshold=min_count_threshold,
            n_jobs=n_jobs
        )

    print("🏁 N-gram pipeline complete.")

# add scripy run __main__

if __name__ == "__main__":
    run_full_ngram_pipeline(
        input_csv="data//preprocessed_wordcloud.zip",
        output_dir="sl_data_for_dashboard//wordclouds",
        text_column="text",
        label_column="label",
        ngram_range=(1, 3),
        n_jobs=4,
        test=True
        )
