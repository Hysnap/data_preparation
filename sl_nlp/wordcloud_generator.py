import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path
import random
from PIL import Image
import numpy as np

def green_colormap(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(120, 100%%, %d%%)" % random.randint(30, 60)  # green hue


def red_colormap(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(10, 100%%, %d%%)" % random.randint(30, 60)  # red/orange hue

STOP_WORDS = set()  # Set this to your actual stop words


def save_wordcloud(text,
                   filename,
                   output_dir,
                   stop_words=STOP_WORDS,
                   color_func=None,
                   mask=None,
                   progress_fn=print):
    if not isinstance(text, str) or not text.strip():
        progress_fn(f"⚠️ Skipping {filename} — no words to render.")
        return
    progress_fn(f"🧪 Generating cloud for {filename}, word count: {len(text.strip().split())}")
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=stop_words,
            color_func=color_func,
            mask=mask
        ).generate(text)
    except ValueError as e:
        progress_fn(f"❌ Failed to generate {filename}: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path)
    plt.close(fig)
    progress_fn(f"✅ Saved wordcloud: {filename}")


def generate_wordclouds_from_ngram_summaries(input_dir,
                                             word_column="word",
                                             output_dir="wordclouds",
                                             cache_dir="wordclouds/cache",
                                             mask_real_path=None,
                                             mask_dubious_path=None,
                                             mask_all_path=None,
                                             ngram_types=(1, 2, 3),
                                             progress_fn=print):
    def load_mask(path):
        return np.array(Image.open(path)) if path else None

    mask_real = load_mask(mask_real_path)
    mask_dubious = load_mask(mask_dubious_path)
    mask_all = load_mask(mask_all_path)

    os.makedirs(cache_dir, exist_ok=True)

    for csv_file in sorted(Path(input_dir).glob("ngram_summary_*.csv")):
        filename_parts = csv_file.stem.split("_")
        year = filename_parts[-1]
        df = pd.read_csv(csv_file)

        if df.empty or "ngram_size" not in df.columns or word_column not in df.columns:
            progress_fn(f"⚠️ Skipped file {csv_file.name} (missing data or required columns)")
            continue

        df[word_column] = df[word_column].astype(str).fillna("").str.strip()

        for n in ngram_types:
            df_ngram = df[df["ngram_size"] == n]
            if df_ngram.empty:
                progress_fn(f"⚠️ No {n}-grams found for {year}")
                continue

            suffix = f"_{n}gram_{year}"

            # Subset DataFrames
            real_df = df_ngram[df_ngram["in_real"] == True]
            dubious_df = df_ngram[df_ngram["in_dubious"] == True]
            common_df = df_ngram[(df_ngram["in_real"]) & (df_ngram["in_dubious"])]
            real_only_df = df_ngram[(df_ngram["in_real"]) & (~df_ngram["in_dubious"])]
            dubious_only_df = df_ngram[(df_ngram["in_dubious"]) & (~df_ngram["in_real"])]

            def build_text(df_sub, count_col):
                return " ".join([
                    (str(word) + " ") * int(count)
                    for word, count in zip(df_sub[word_column], df_sub[count_col])
                    if count > 0
                ])

            all_text = build_text(df_ngram, "count_real") + build_text(df_ngram, "count_dubious")
            real_text = build_text(real_df, "count_real")
            dubious_text = build_text(dubious_df, "count_dubious")
            common_text = build_text(common_df, "count_real")
            real_only_text = build_text(real_only_df, "count_real")
            dubious_only_text = build_text(dubious_only_df, "count_dubious")

            text_dict = {
                "all_text": all_text,
                "real_text": real_text,
                "dubious_text": dubious_text,
                "common_text": common_text,
                "real_only": real_only_text,
                "dubious_only": dubious_only_text,
            }

            for name, text in text_dict.items():
                cache_path = os.path.join(cache_dir, f"{name}{suffix}.txt")
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
                progress_fn(f"💾 Cached: {name}{suffix}.txt")

            progress_fn(f"🎨 Generating {n}-gram wordclouds for {year}...")

            save_wordcloud(all_text, f"wordcloud_all{suffix}.png", output_dir,
                           mask=mask_all, progress_fn=progress_fn)
            save_wordcloud(real_text, f"wordcloud_real{suffix}.png", output_dir,
                           mask=mask_real, color_func=green_colormap, progress_fn=progress_fn)
            save_wordcloud(dubious_text, f"wordcloud_dubious{suffix}.png", output_dir,
                           mask=mask_dubious, color_func=red_colormap, progress_fn=progress_fn)
            save_wordcloud(real_only_text, f"wordcloud_real_unique{suffix}.png", output_dir,
                           mask=mask_real, color_func=green_colormap, progress_fn=progress_fn)
            save_wordcloud(dubious_only_text, f"wordcloud_dubious_unique{suffix}.png", output_dir,
                           mask=mask_dubious, color_func=red_colormap, progress_fn=progress_fn)
            save_wordcloud(common_text, f"wordcloud_common{suffix}.png", output_dir,
                           mask=mask_all, progress_fn=progress_fn)

    progress_fn(f"📁 Wordcloud generation complete.")



if __name__ == "__main__":
    generate_wordclouds_from_ngram_summaries(
        input_dir="sl_data_for_dashboard\\wordclouds",
        word_column="term",
        output_dir="wordclouds",
        cache_dir="sl_data_for_dashboard/wordclouds/cache",
        mask_real_path="sillouettes/mask_real.png",
        mask_dubious_path="sillouettes/mask_dubious.png",
        mask_all_path="sillouettes/mask_combined.png",
        ngram_types=(1, 2, 3),  # or (2,) if just 2-grams
        progress_fn=print
    )