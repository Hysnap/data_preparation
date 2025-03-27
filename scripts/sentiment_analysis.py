from sl_utils.logger import datapipeline_logger as logger
from sl_utils.utils import save_dataframe_to_zip
from scripts.pipeline_code import prepare_combined_df
from scripts.sentiment_analysis import apply_sentiment
import pandas as pd


def run_sentiment_analysis(df, flags, save_to_disk):
    if flags.get("useprelocationdata"):
        try:
            df = pd.read_csv('data/combined_data_pre_location.zip', low_memory=False)
            return prepare_combined_df(df)
        except FileNotFoundError:
            logger.warning("Pre-location data not found. Regenerating...")

    df = apply_sentiment(df)

    if save_to_disk:
        save_dataframe_to_zip(df, 'data/combined_data_pre_location.zip', 'combined_data_pre_location.csv')

    return df