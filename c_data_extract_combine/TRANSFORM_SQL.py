    """
    description: This script contains the data cleaning and feature extraction
    functions for the combined data. The functions in this script are used to
    classify the media type of the title and article text, extract date
    information from the date column, and create date features for use in
    future models. The script also removes rows where the title or text is
    identical, empty, or whitespace, and drops unnecessary columns from the
    combined dataframe. The cleaned and transformed data is then saved to a zip
    file for further processing.

    Functions:
    - classify_and_combine: This function adds label flags to the true and fake
    dataframes, concatenates the true, fake, and test dataframes, and classifies
    the media type of the title and article text. The function also extracts date
    information from the date column, creates date features for use in future
    models, and saves the cleaned and transformed data to a zip file.  
    """

import pandas as pd
import numpy as np
import holidays
from sl_utils.logger import datapipeline_logger as logger
from sl_utils.logger import log_function_call
from c_data_extract_combine.ETL import classify_media
from c_data_extract_combine.ETL import save_dataframe_to_zip


@log_function_call(logger)
# Data cleaning and feature extraction
def classify_and_combine(true_df, fake_df, test_data_df):
    # add a column to each dataframe to
    logger.info("Add label flags to true_df and fake_df.")
    # indicate whether the news is fake or true
    true_df["label"] = 1
    fake_df["label"] = 0
    # provide a count of rows in each dataframe
    logger.info("Add count column to true_df and fake_df.")
    true_df["count"] = true_df.index
    fake_df["count"] = fake_df.index
    test_data_df["count"] = test_data_df.index
    # log row count of files
    logger.info(f"Number of rows in true_df:  {true_df.shape[0]}")
    logger.info(f"Number of rows in fake_df:  {fake_df.shape[0]}")
    logger.info(f"Number of rows in test_data_df:  {test_data_df.shape[0]}")
    # add a column to each dataframe to indicate the source
    logger.info("Add source column to true_df and fake_df.")
    true_df["file"] = "true"
    fake_df["file"] = "fake"
    test_data_df["file"] = "test_data"
    # combine the three dataframes
    logger.info("Concatenate true_df, fake_df, and test_data_df.")
    combined_df = pd.concat([true_df, fake_df, test_data_df],
                            ignore_index=True)
    combined_pre_clean = combined_df.copy()
    # count the number of rows in the combined dataframe
    logger.info("Count the number of rows in the combined dataframe.")
    logger.info("Number of rows in combined"
                f" dataframe: {combined_df.shape[0]}")
    # remove rows where text is identical
    logger.info("Remove rows where text is identical.")
    combined_pre_clean.drop_duplicates(subset='text', inplace=True)
    # log the number of rows in the combined dataframe
    logger.info("Number of rows in combined dataframe"
                " after removing"
                f" identical text: {combined_pre_clean.shape[0]}")
    logger.info("Drop rows where title or text is empty.")
    # if title is empty or fill with first sentence from text
    combined_pre_clean['title'] = (
        combined_pre_clean['title'].fillna(
            combined_pre_clean['text'].str.split('.').str[0]))
    # remove any rows where the title or text is empty
    combined_pre_clean = combined_pre_clean.dropna(subset=['title', 'text'])
    # log the number of rows in the combined dataframe
    logger.info("Number of rows in combined "
                "dataframe after removing"
                f" empty title or text: {combined_pre_clean.shape[0]}")
    logger.info("Drop rows where title or text is whitespace.")
    # remove any rows where the title or text is whitespace
    combined_pre_clean = (
        combined_pre_clean[combined_pre_clean['title'].
                           str.strip().astype(bool)])
    combined_pre_clean = (
        combined_pre_clean[combined_pre_clean['text'].
                           str.strip().astype(bool)])
    # log the number of rows in the combined dataframe
    logger.info("Number of rows in combined dataframe"
                " after removing whitespace title"
                f" or text: {combined_pre_clean.shape[0]}")
    logger.info("Title and text column cleansing")
    # rename the text colum "article_text"
    combined_pre_clean.rename(columns={"text": "article_text"}, inplace=True)
    combined_pre_clean['title'] = (
        combined_pre_clean['title'].str.replace(r'[^\w\s]', ''))
    combined_pre_clean['article_text'] = (
        combined_pre_clean['article_text'].str.replace(r'[^\w\s]', ''))
    # remove all leading and trailing spaces from title
    combined_pre_clean['title'] = combined_pre_clean['title'].str.strip()
    # remove all leading and trailing spaces from article_text
    combined_pre_clean['article_text'] = (
        combined_pre_clean['article_text'].str.strip())
    logger.info("Title and text column cleansing completed")
    logger.info("Classify media type of title and article_text")
    # classify the media type of the title
    combined_pre_clean['media_type_title'] = (
        combined_pre_clean['title'].apply(classify_media))
    combined_pre_clean['media_type_article'] = (
        combined_pre_clean['article_text'].apply(classify_media))
    combined_pre_clean['media_type'] = combined_pre_clean.apply(
        lambda row: row['media_type_title']
        if row['media_type_title'] != 'text'
        else row['media_type_article'],
        axis=1
    )
    logger.info("Classify media type of title and article_text completed")

    # extract date information from date column
    # Month mapping dictionary
    month_map = {
        'january': '1', 'jan': '1',
        'february': '2', 'feb': '2',
        'march': '3', 'mar': '3',
        'april': '4', 'apr': '4',
        'may': '5',
        'june': '6', 'jun': '6',
        'july': '7', 'jul': '7',
        'august': '8', 'aug': '8',
        'september': '9', 'sep': '9',
        'october': '10', 'oct': '10',
        'november': '11', 'nov': '11',
        'december': '12', 'dec': '12'
    }
    logger.info("Extract date information from date column")
    # set all blank date values to '1901-01-01 00:000:00'
    combined_pre_clean['date'] = (
        combined_pre_clean['date'].fillna('1901-02-01 00:00:00'))
    # Remove punctuation & trim spaces
    combined_pre_clean['date'] = (
        combined_pre_clean['date'].str.replace(r'[^\w\s]',
                                               '',
                                               regex=True).str.strip())
    # Extract month, day, and year
    combined_pre_clean[['month', 'day', 'year']] = (
        combined_pre_clean['date'].str.extract(r'(\w+)\s+(\d+),?\s*(\d+)?'))
    # Convert month names to numbers
    combined_pre_clean['month'] = (
        combined_pre_clean['month'].str.lower().map(month_map))
    # Ensure numeric values and handle missing years (e.g., "23" → "2023")
    combined_pre_clean['year'] = (
        combined_pre_clean['year'].
        fillna(pd.to_datetime('today').year).astype(str))
    combined_pre_clean['year'] = (
        combined_pre_clean['year'].
        apply(lambda x: '20' + x if len(x) == 2 else x))
    # Ensure month, day, and year are strings and
    # fill NaN values with empty strings
    combined_pre_clean[['year', 'month', 'day']] = (
        combined_pre_clean[['year', 'month', 'day']].astype(str).fillna(''))
    # Ensure we only concatenate if all components are present
    combined_pre_clean['date_str'] = combined_pre_clean.apply(
        lambda row: f"{row['year']}-{row['month']}-{row['day']}"
        if row['year'] and row['month'] and row['day']
        else None, axis=1
    )
    # Convert to datetime, forcing errors to NaT
    combined_pre_clean['date_clean'] = (
        pd.to_datetime(combined_pre_clean['date_str'],
                       errors='coerce'))
    # log rows following date extraction
    logger.info("Number of rows with NA dates:"
                f" {combined_pre_clean['date_clean'].isna().sum()}")
    # if date is NA try and find a date in the article_text
    combined_pre_clean['date_clean'] = (
        combined_pre_clean['date_clean']
        .fillna(combined_pre_clean['article_text'].
                str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    )
    # log rows following date extraction
    logger.info("Number of rows with NA dates post articletext:"
                f" {combined_pre_clean['date_clean'].isna().sum()}")
    # if date is NA try and find a date in the title
    combined_pre_clean['date_clean'] = (
        combined_pre_clean['date_clean']
        .fillna(combined_pre_clean['title'].
                str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    )
    # log rows following date extraction
    logger.info("Number of rows with NA dates post title:"
                f" {combined_pre_clean['date_clean'].isna().sum()}")
    # use NLP to extract dates from article_text
    date_formats = [
        '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
        '%Y.%m.%d', '%d.%m.%Y', '%m.%d.%Y', '%Y %m %d', '%d %m %Y', '%m %d %Y'
    ]

    def try_parsing_date(text):
        for fmt in date_formats:
            try:
                return pd.to_datetime(text, format=fmt, errors='coerce')
            except ValueError:
                continue
        return None

    combined_pre_clean['date_clean'] = (
        combined_pre_clean['date_clean'].
        fillna(combined_pre_clean['article_text'].apply(
            lambda x: try_parsing_date(' '.join(
                [word for word in x.split() if word.isdigit()]))))
    )
    combined_pre_clean['date_clean'] = (
        combined_pre_clean['date_clean'].
        fillna(combined_pre_clean['title'].apply(
            lambda x: try_parsing_date(' '.join(
                [word for word in x.split() if word.isdigit()]))))
    )
    # log count of rows with NA dates
    logger.info("Number of rows with NA dates post NLP:"
                f" {combined_pre_clean['date_clean'].isna().sum()}")
    # force all Na dates to be 2000-02-01
    combined_pre_clean['date_clean'] = (
        combined_pre_clean['date_clean'].
        fillna(pd.to_datetime('2000-02-01')))
    # set missing month, day and year values
    combined_pre_clean[['month', 'day', 'year']] = (
        combined_pre_clean['date'].str.extract(r'(\w+)\s+(\d+),?\s*(\d+)?'))
    # Drop rows where date conversion failed
    combined_pre_clean = combined_pre_clean.dropna(subset=['date_clean'])
    # log rows following date extraction
    logger.info("Number of rows in combined dataframe"
                f" after date extraction: {combined_pre_clean.shape[0]}")
    # Convert 'date_clean' to datetime, handling errors
    combined_pre_clean['date_clean'] = (
        pd.to_datetime(combined_pre_clean['date_clean'], errors='coerce'))
    # Now apply .dt.strftime safely
    combined_pre_clean['date_clean'] = (
        combined_pre_clean['date_clean'].dt.strftime('%Y-%m-%d'))
    # ensure date_clean is in datetime format
    combined_pre_clean['date_clean'] = (
        pd.to_datetime(combined_pre_clean['date_clean']))
    # find earliest date and latest date then
    # find all Us_holidays in period
    min_date = combined_pre_clean['date_clean'].min()
    max_date = combined_pre_clean['date_clean'].max()
    min_year = min_date.year
    max_year = max_date.year
    us_holidays = holidays.US(years=range(min_year, max_year+1))
    logger.info("create date features for ML models")
    # Extract Features for use in future models
    combined_pre_clean['day_of_week'] = (
        combined_pre_clean['date_clean'].dt.day_name())
    combined_pre_clean['week_of_year'] = (
        combined_pre_clean['date_clean'].dt.isocalendar().week)
    combined_pre_clean['is_weekend'] = (
        combined_pre_clean['day_of_week'].
        isin(['Saturday', 'Sunday']).astype(int))
    combined_pre_clean['is_weekday'] = (
        (~combined_pre_clean['day_of_week'].
         isin(['Saturday', 'Sunday'])).astype(int))
    combined_pre_clean['week_of_year_sin'] = (
        np.sin(2 * np.pi * combined_pre_clean['week_of_year'] / 52))
    combined_pre_clean['week_of_year_cos'] = (
        np.cos(2 * np.pi * combined_pre_clean['week_of_year'] / 52))
    combined_pre_clean['holiday'] = (
        combined_pre_clean['date_clean'].isin(us_holidays).astype(int))
    combined_pre_clean['day_of_month_sine'] = (
        np.sin(2 * np.pi * combined_pre_clean['date_clean'].dt.day / 31))
    combined_pre_clean['day_of_month_cos'] = (
        np.cos(2 * np.pi * combined_pre_clean['date_clean'].dt.day / 31))
    combined_pre_clean['month_sin'] = (
        np.sin(2 * np.pi * combined_pre_clean['date_clean'].dt.month / 12))
    combined_pre_clean['month_cos'] = (
        np.cos(2 * np.pi * combined_pre_clean['date_clean'].dt.month / 12))
    # create day label set it to holiday name if holiday,
    # else set it to day of week
    combined_pre_clean['day_label'] = combined_pre_clean['date_clean'].apply(
        lambda x: us_holidays.get(x) if x in us_holidays else x.day_name()
    )
    logger.info("Date information extracted from date column")
    # drop unnecessary and blank columns
    logger.info("Drop unnecessary and blank columns")
    combined_pre_clean.drop(['Column1',
                             'subject2',
                             'count',
                             'date_str',
                             'media_type_title',
                             'media_type_article',
                             'date'
                             ],
                            axis=1, inplace=True)
    # log the number of rows in the combined dataframe
    logger.info("Number of rows in combined dataframe"
                f" after dropping columns: {combined_pre_clean.shape[0]}")
    # export combined_pre_clean to csv
    save_dataframe_to_zip(combined_pre_clean, 'data/combined_pre_clean.zip')
    logger.info("Combined data saved to combined_pre_clean.zip")
    # drop combined_pre_clean datafile
    return combined_pre_clean

# Path: c_data_extract_combine/TRANSFORM.py
# end of TRANSFORM.py
