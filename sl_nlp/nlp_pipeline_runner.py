# nlp_pipeline_runner.py

import os
import pandas as pd
from tqdm.auto import tqdm
from ngram_summary_pipeline import run_full_ngram_pipeline
from wordcloud_generator import generate_wordclouds_from_ngram_summaries
from theme_extraction import save_themes_by_year
import subprocess
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

tqdm.pandas(desc="Processing")


def run_full_nlp_pipeline(input_csv,
                          output_dir="sl_data_for_dashboard",
                          text_column="text",
                          label_column="label",
                          ngram_range=(1, 3),
                          min_count_threshold=25,
                          n_jobs=4):
    """
    Runs the full NLP pipeline:
    - N-gram summary generation
    - Word cloud image generation
    - Common theme extraction

    Args:
        input_csv (str): Path to input CSV file
        output_dir (str): Output directory for all results
        text_column (str): Name of text column
        label_column (str): Name of label column
        ngram_range (tuple): Range of n-gram sizes
        min_count_threshold (int): Minimum frequency threshold for n-grams
        n_jobs (int): Number of parallel jobs
    """
    def progress(msg):
        print(msg)

    print("\n🚀 Starting full NLP processing pipeline\n")

    df = pd.read_csv(input_csv, low_memory=False)

    # === Run N-gram summary ===
    run_full_ngram_pipeline(
        input_csv=input_csv,
        output_dir=output_dir,
        text_column=text_column,
        label_column=label_column,
        ngram_range=ngram_range,
        min_count_threshold=min_count_threshold,
        n_jobs=n_jobs
    )

    # === Generate Word Clouds ===
    generate_wordclouds_from_ngram_summaries(
        input_dir="sl_data_for_dashboard",
        word_column="term",
        output_dir="wordclouds",
        cache_dir="sl_data_for_dashboard/wordclouds/cache",
        progress_fn=print
    )

    # === Extract Common Themes ===
    save_themes_by_year(
        df=df,
        text_column=text_column,
        output_dir=os.path.join(output_dir, "themes"),
        progress_fn=progress
    )

    print("\n✅ All pipeline steps complete.\n")


if __name__ == "__main__":
    run_full_nlp_pipeline(
        input_csv="sl_data_for_dashboard/preprocessed_wordcloud.zip",
        output_dir="sl_data_for_dashboard",
        min_count_threshold=25,
        ngram_range=(1, 3),
        n_jobs=4
    )
