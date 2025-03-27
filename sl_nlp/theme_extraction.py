# theme_extraction.py

import os
import pandas as pd
import spacy
from collections import Counter

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def extract_common_nouns(texts, progress_fn=print):
    """Extract common noun tokens from a list of texts."""
    progress_fn("🔍 Extracting common themes (nouns)...")
    counter = Counter()
    docs = nlp.pipe(texts, batch_size=500)

    for doc in docs:
        for token in doc:
            if token.pos_ == "NOUN" and token.is_alpha and not token.is_stop:
                counter[token.lemma_.lower()] += 1

    theme_df = pd.DataFrame(counter.items(), columns=["theme", "count"])
    theme_df = theme_df.sort_values(by="count", ascending=False).reset_index(drop=True)
    return theme_df


def save_themes_by_year(df, text_column="text", output_dir="themes", progress_fn=print):
    os.makedirs(output_dir, exist_ok=True)
    df[text_column] = df[text_column].astype(str).fillna("")

    if "year" in df.columns:
        for year in sorted(df["year"].dropna().unique()):
            year = int(year)
            subset = df[df["year"] == year]
            if not subset.empty:
                theme_df = extract_common_nouns(subset[text_column], progress_fn)
                out_path = os.path.join(output_dir, f"common_themes_{year}.csv")
                theme_df.to_csv(out_path, index=False)
                progress_fn(f"📁 Saved common themes for {year} to {out_path}")
    else:
        theme_df = extract_common_nouns(df[text_column], progress_fn)
        out_path = os.path.join(output_dir, "common_themes.csv")
        theme_df.to_csv(out_path, index=False)
        progress_fn(f"📁 Saved common themes to {out_path}")

    progress_fn("🏁 Theme extraction complete.")
