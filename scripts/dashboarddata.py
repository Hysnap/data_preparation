import sys
import os

# Add the parent directory of sl_utils to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sl_utils.logger import datapipeline_logger as logger
from sl_utils.utils import save_dataframe_to_zip
import pandas as pd
import tqdm.auto as tqdm

tqdm.tqdm.pandas()


def run_dashboard_data_generation(df):
    """
    Run dashboard data generation
    """

    # check for any blank values
    logger.debug(df.isnull().sum())
    logger.debug(df.head(5))

    # rename index to article_id if column article_id does not exist
    if 'article_id' not in df.columns:
        df.index.name = 'index'
        df['article_id'] = df.index

    # summarise data to minimise rows and memory requirement
    logger.info("Summarising data...")

    """
     measures used in visualisations

        1. Number of articles
        3. overall polarity value
        4. overall subjectivity value
        5. article polarity value
        6. article subjectivity value
        8. title polarity value
        9. title subjectivity value
        11. contradiction subjectivity value
        12. contradiction polarity value
        13. polarity variance
        14. subjectivity variance
        15. count of locations
        16. title_length
        17. text_length

        Grouped by
        1. Source_name
        2. Day of the week
        3. Day label
        4. Month
        5. Year
        6. media_type
        7. label
        8. overall sentiment value
        9. article sentiment value
        10. title sentiment value
        11. subject
    """

    # add count_of_locations
    df['count_of_locations'] = (
        df['locationsfromarticle'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0))

    # round off the values to 2 decimal places
    decimalround = 2
    df['overall_polarity_value'] = (
        df['overall_polarity'].round(decimalround))
    df['overall_subjectivity_value'] = (
        df['overall_subjectivity'].round(decimalround))
    df['article_polarity_value'] = (
        df['article_polarity'].round(decimalround))
    df['article_subjectivity_value'] = (
        df['article_subjectivity'].round(decimalround))
    df['title_polarity_value'] = (
        df['title_polarity'].round(decimalround))
    df['title_subjectivity_value'] = (
        df['title_subjectivity'].round(decimalround))
    df['contradiction_subjectivity_value'] = (
        df['contradiction_subjectivity'].round(decimalround))
    df['contradiction_polarity_value'] = (
        df['contradiction_polarity'].round(decimalround))
    df['polarity_variations_value'] = (
        df['polarity_variations'].round(decimalround))
    df['subjectivity_variations_value'] = (
        df['subjectivity_variations'].round(decimalround))
        
    # round text_length nearest 100
    #set variable to control rounding
    textround = -1
    df['text_length_value'] = (
        df['text_length'].apply(lambda x: round(x, textround)))

    # drop unnecessary columns
    df = df.drop([
            'title',
            'file',
            'day',
            'date_clean',
            'week_of_year',
            'is_weekend',
            'is_weekday',
            'week_of_year_sin',
            'week_of_year_cos',
            'month_sin',
            'month_cos',
            'holiday',
            'day_of_month_sine',
            'day_of_month_cos',
            'cleaned_text',
            'nlp_text',
            'nlp_title',
            'locationsfromarticle',
            ], axis=1)

    # Change dtype of suitable columns to category
    df['source_name'] = (
        df['source_name'].astype('category'))
    df['day_of_week'] = (
        df['day_of_week'].astype('category'))
    df['day_label'] = (
        df['day_label'].astype('category'))
    df['month'] = (
        df['month'].astype('category'))
    df['year'] = (
        df['year'].astype('category'))
    df['media_type'] = (
        df['media_type'].astype('category'))
    df['label'] = (
        df['label'].astype('category'))
    df['sentiment_overall'] = (
        df['sentiment_overall'].astype('category'))
    df['sentiment_article'] = (
        df['sentiment_article'].astype('category'))
    df['sentiment_title'] = (
        df['sentiment_title'].astype('category'))


    # group by Source_name, day_of_week, day_label, month,
    # year, media_type, label
    # overall_sentiment_value, overall_polarity_value,
    # overall_subjectivity_value
    # article_polarity_value, article_subjectivity_value,
    # article_sentiment_value
    # title_polarity_value, title_subjectivity_value, title_sentiment_value
    # contradiction_subjectivity_value, contradiction_polarity_value
    # polarity_variance, subjectivity_variance, count of locations
    # title_length, text_length
    # count of articles


    # ensure only observerd values are used
    DashBoardData = df.groupby(by=[
        'source_name',
        'subject',
        'day_of_week',
        'day_label',
        'month',
        'year',
        'media_type',
        'label',
        'sentiment_overall',
        'overall_polarity_value',
        'overall_subjectivity_value',
        'article_polarity_value',
        'article_subjectivity_value',
        'sentiment_article',
        'title_polarity_value',
        'title_subjectivity_value',
        'sentiment_title',
        'contradiction_subjectivity_value',
        'contradiction_polarity_value',
        'polarity_variations_value',
        'subjectivity_variations_value',
        'count_of_locations',
        'title_length',
        'text_length_value'],
        observed=True
        ).agg({
            'article_id': 'count'}).reset_index()

    print(DashBoardData.head(5))
    print(DashBoardData.info())

    # print DashBoardData memory usage
    logger.info("DashBoardData memory usage:")
    logger.info(DashBoardData.memory_usage(deep=True))

    # open DashBoardData in Data Wrangler
    # save DashBoardData to disk
    save_dataframe_to_zip(DashBoardData,
                          'data/dashboard_data.zip',
                          'dashboard_data.csv')
    logger.info("Dashboard data saved as dashboard_data.csv in data folder")

    

if __name__ == "__main__":
    # if df is null, then load combined_data.zip
    df = pd.read_csv("data//combined_data.zip", low_memory=False)
    run_dashboard_data_generation(df)

# This script is used to generate the dashboard data.
# It is called from the main script ROD_data_prep.py
# . The script imports the necessary functions and libraries to
# generate the dashboard data.
# The run_dashboard_data_generation function is called with the dataframe
# as an argument to generate the dashboard data.
# The function checks for any blank values in the dataframe and logs the
# results.
# The script is used to generate the dashboard data for the ROD project.

