from sl_utils.utils import save_dataframe_to_zip, nlp
from sl_utils.logger import datapipeline_logger as logger, log_function_call
import pandas as pd


@log_function_call(logger)
def common_themes(findcommonthemes, combined_df):
    logger.info("Finding common themes")
    # using NLP produce a li    st of common themes
    # create a list of common themes
    common_themes = []
    # iterate over the nlp_text column
    for text in combined_df['nlp_text']:
    # create a doc object
    doc = nlp(text)
    # iterate over the entities in the doc
    for ent in doc.ents:
    # if the entity is a common noun
    if ent.label_ == 'NOUN':
        # append the entity to the common_themes list
        common_themes.append(ent.text)
    # create a dataframe of common themes
    df_common_themes = pd.DataFrame(common_themes, columns=['theme'])
    # create a count of the number of times each theme appears
    df_common_themes = df_common_themes['theme'].value_counts().reset_index()
    # rename the columns
    df_common_themes.columns = ['theme', 'count']
    # save the common themes data to a csv file
    save_dataframe_to_zip(df_common_themes,
                          'data/common_themes.zip',
                          'common_themes.csv')