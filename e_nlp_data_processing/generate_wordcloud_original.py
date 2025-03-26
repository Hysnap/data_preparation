import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")


def generate_wordcloud_images(df, text_column="text", label_column="label",
                              output_dir="wordclouds", progress_fn=print):
    """
    Generates and saves word cloud images with progress tracking.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame with 'text' and 'label'
        text_column (str): Name of the column with cleaned, lowercase text
        label_column (str): Binary label column (1=True, 0=False)
        output_dir (str): Folder where images will be saved
        progress_fn (function): Function to report progress (default: print)
    """

    os.makedirs(output_dir, exist_ok=True)
    stop_words = set(stopwords.words("english"))

    def save_wordcloud(text, filename):
        wordcloud = WordCloud(width=800, height=400,
                              background_color="white",
                              stopwords=stop_words).generate(text)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout()
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)
        progress_fn(f"✅ Saved wordcloud: {filename}")

    progress_fn("📦 Starting word cloud generation...")

    df[text_column] = df[text_column].astype(str).fillna("")

    # Paths for cached files
    text_variations = {
        "all_text": os.path.join(output_dir, "all_text.txt"),
        "true_text": os.path.join(output_dir, "true_text.txt"),
        "false_text": os.path.join(output_dir, "false_text.txt"),
        "true_only": os.path.join(output_dir, "true_only.txt"),
        "false_only": os.path.join(output_dir, "false_only.txt"),
        "common_text": os.path.join(output_dir, "common_text.txt"),
    }

    text_data = {}

    def load_or_generate(key, generate_fn):
        path = text_variations[key]
        if os.path.exists(path):
            progress_fn(f"✅ Cached file found for {key}. Loading...")
            with open(path, "r") as f:
                text_data[key] = f.read()
        else:
            progress_fn(f"⚙️ Generating {key}...")
            text_data[key] = generate_fn()
            with open(path, "w") as f:
                f.write(text_data[key])
            progress_fn(f"✅ {key} saved to cache.")

    # Generate or load all_text
    load_or_generate("all_text", lambda: " ".join(df[text_column]))

    # Generate or load true_text
    load_or_generate("true_text", lambda: " ".join(df[df[label_column] == 1][text_column]))

    # Generate or load false_text
    load_or_generate("false_text", lambda: " ".join(df[df[label_column] == 0][text_column]))

    # Generate or load true_only
    load_or_generate("true_only", lambda: " ".join(
        [word for word in text_data["true_text"].split() if word not in text_data["false_text"].split()]))

    # Generate or load false_only
    load_or_generate("false_only", lambda: " ".join(
        [word for word in text_data["false_text"].split() if word not in text_data["true_text"].split()]))

    # Generate or load common_text
    load_or_generate("common_text", lambda: " ".join(
        [word for word in text_data["all_text"].split()
         if word not in text_data["true_only"].split() and word not in text_data["false_only"].split()]))

    # Assign variables for further processing
    all_text = text_data["all_text"]
    true_text = text_data["true_text"]
    false_text = text_data["false_text"]
    true_only = text_data["true_only"]
    false_only = text_data["false_only"]
    common_text = text_data["common_text"]

    progress_fn("🎨 Generating wordcloud images...")
    save_wordcloud(all_text, "wordcloud_all.png")
    save_wordcloud(true_text, "wordcloud_real.png")
    save_wordcloud(false_text, "wordcloud_dubious.png")
    save_wordcloud(true_only, "wordcloud_true_unique.png")
    save_wordcloud(false_only, "wordcloud_false_unique.png")
    save_wordcloud(common_text, "wordcloud_common.png")

    progress_fn("🏁 Word cloud generation complete!")


# === Example usage ===
if __name__ == "__main__":
    csv_path = "sl_data_for_dashboard/preprocessed_wordcloud.zip"
    df = pd.read_csv(csv_path)
    generate_wordcloud_images(df)

# End of script
# PATH: sl_visualisations/generate_wordcloud.py
