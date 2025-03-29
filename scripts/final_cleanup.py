from sl_utils.logger import datapipeline_logger as logger
from sl_utils.utils import save_dataframe_to_zip

def finalize_data(df, save_to_disk):
    df['subject'] = df['subject'].fillna('Unknown')
    df = df.drop(['nlp_textloc',
                  'source',
                  'location',
                  'nlp_location'], axis=1)
    df = df.dropna(subset=['cleaned_text'])

    if save_to_disk:
        save_dataframe_to_zip(df,
                              'data/combined_data_step1.zip',
                              'combined_data_step1.csv')

    return df