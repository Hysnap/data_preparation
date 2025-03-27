import pandas as pd
from sl_utils.logger import datapipeline_logger as logger
from sl_utils.utils import save_dataframe_to_zip
from scripts.TRANSFORM import classify_and_combine
from scripts.pipeline_code import prepare_combined_df
from scripts.pipeline_code import process_nlp


def run_nlp_processing(raw_data, flags, save_to_disk):
    if flags.get("usepostnlpdata"):
        try:
            df = pd.read_csv('data/combined_data_postnlp.zip',
                             low_memory=False)
            return prepare_combined_df(df)
        except FileNotFoundError:
            logger.warning("Post NLP data not found. Regenerating...")
    fake_df, true_df, test_df = raw_data

    if flags.get("useprecombineddata"):
        df = pd.read_csv('data/combined_pre_clean.zip',
                         low_memory=False)
    else:
        df = classify_and_combine(true_df,
                                  fake_df,
                                  test_df)

    df = prepare_combined_df(df)
    df = process_nlp(df)

    if save_to_disk:
        save_dataframe_to_zip(df,
                              'data/combined_data_postnlp.zip',
                              'combined_data_postnlp.csv')

    return df