# wordcloud_generator.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))


def save_wordcloud(text,
                   filename,
                   output_dir,
                   stop_words=STOP_WORDS,
                   progress_fn=print):
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", stopwords=stop_words
    ).generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path)
    plt.close(fig)
    progress_fn(f"✅ Saved wordcloud: {filename}")


def load_or_generate_text(filepath, generator_fn, progress_fn):
    if os.path.exists(filepath):
        progress_fn(f"✅ Cached file found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    else:
        text = generator_fn()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        progress_fn(f"✅ Generated and cached: {filepath}")
        return text


def generate_wordcloud_images(df,
                              text_column="text",
                              label_column="label",
                              output_dir="wordclouds",
                              cache_dir="wordclouds/cache",
                              progress_fn=print):
    os.makedirs(cache_dir, exist_ok=True)
    df[text_column] = df[text_column].astype(str).fillna("")

    def generate_images(df_subset, year_suffix=""):
        # Prepare paths
        text_paths = {
            "all_text": os.path.join(cache_dir,
                                     f"all_text{year_suffix}.txt"),
            "true_text": os.path.join(cache_dir,
                                      f"true_text{year_suffix}.txt"),
            "false_text": os.path.join(cache_dir,
                                       f"false_text{year_suffix}.txt"),
        }

        # Create a DataFrame to store words
        # and their occurrences in each category
        word_data = pd.DataFrame(columns=["word",
                                          "all_text",
                                          "true_text",
                                          "false_text"])

        # Generate text for each category
        all_text = " ".join(df_subset[text_column])
        true_text = " ".join(df_subset[df_subset[label_column] == 1][text_column])
        false_text = " ".join(df_subset[df_subset[label_column] == 0][text_column])

        # Populate the DataFrame with word counts
        all_words = set(all_text.split())
        for word in all_words:
            word_data = word_data.append({
                "word": word,
                "all_text": all_text.split().count(word),
                "true_text": true_text.split().count(word),
                "false_text": false_text.split().count(word),
            }, ignore_index=True)

        # Save the word data to a CSV file for debugging or further analysis
        word_data_path = os.path.join(cache_dir, f"word_data{year_suffix}.csv")
        word_data.to_csv(word_data_path, index=False, encoding="utf-8")
        progress_fn(f"✅ Word data saved to: {word_data_path}")

        # Compute text differences using the word data
        true_only = " ".join(word_data[word_data["true_text"] > 0 & (word_data["false_text"] == 0)]["word"])
        false_only = " ".join(word_data[word_data["false_text"] > 0 & (word_data["true_text"] == 0)]["word"])
        common_text = " ".join(word_data[(word_data["true_text"] > 0) & (word_data["false_text"] > 0)]["word"])

        # Save intermediate .txt files
        with open(os.path.join(cache_dir, f"true_only{year_suffix}.txt"), "w", encoding="utf-8") as f:
            f.write(true_only)
        with open(os.path.join(cache_dir, f"false_only{year_suffix}.txt"), "w", encoding="utf-8") as f:
            f.write(false_only)
        with open(os.path.join(cache_dir, f"common_text{year_suffix}.txt"), "w", encoding="utf-8") as f:
            f.write(common_text)

        # Generate word cloud images
        progress_fn(f"🎨 Generating wordcloud images{year_suffix}...")
        save_wordcloud(all_text, f"wordcloud_all{year_suffix}.png",
                       output_dir, progress_fn=progress_fn)
        save_wordcloud(true_text, f"wordcloud_real{year_suffix}.png",
                       output_dir, progress_fn=progress_fn)
        save_wordcloud(false_text, f"wordcloud_dubious{year_suffix}.png",
                       output_dir, progress_fn=progress_fn)
        save_wordcloud(true_only, f"wordcloud_true_unique{year_suffix}.png",
                       output_dir, progress_fn=progress_fn)
        save_wordcloud(false_only, f"wordcloud_false_unique{year_suffix}.png",
                       output_dir, progress_fn=progress_fn)
        save_wordcloud(common_text, f"wordcloud_common{year_suffix}.png",
                       output_dir, progress_fn=progress_fn)

    if "year" in df.columns:
        for year in sorted(df["year"].dropna().unique()):
            year = int(year)
            subset = df[df["year"] == year]
            if not subset.empty:
                generate_images(subset, year_suffix=f"_{year}")
                progress_fn(f"📁 Word clouds for {year} complete.")
    else:
        generate_images(df)

    progress_fn("🏁 Word cloud generation complete.")
