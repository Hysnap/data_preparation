"""
Description: This module contains functions that load
and process data for the dashboard.
    The data is loaded from the data folder and processed
    to generate the required dataframes.
    The processed dataframes are then saved to the data
    folder for use in the dashboard.
    The functions in this module are used to load and process
    the data for the dashboard.
    The processed data is saved to the data folder for use in the dashboard.

    Functions:
    - mapdata: Loads the data required for the map visualization.
    - dashboarddata: Loads the data required for the dashboard visualization.
    - wordcountdata: Loads the data required for the wordcloud visualization.

    Returns:
        _type_: _description_
    """

import pandas as pd
from sl_utils.logger import log_function_call, datapipeline_logger as logger
import os


@log_function_call(logger)
def mapdata():
    # Check if articlesformap.csv exists
    file_path = "sl_data_for_dashboard//articlesformap.csv"
    if os.path.exists(file_path):
        logger.debug(f"{file_path} exists. Loading the file.")
        articles = pd.read_csv(file_path)
    else:
        logger.debug(f"{file_path} does not exist. Running generation script.")
        # load combined_data.zip file
        df = pd.read_csv(
            "sl_data_for_dashboard//dashboard_data.zip",
            usecols={
                    'index',
                    'title',
                    'label',
                    'month',
                    'day',
                    'year',
                    'date_clean',
                    'article_id',
                                },
            dtype={
                    'index': 'int64',
                    'title': 'object',
                    'label': 'int64',
                    'month': 'float64',
                    'day': 'float64',
                    'year': 'int64',
                    'date': 'object',
                    'article_id': 'int64',
                    },
            compression='zip')
        logger.debug("The shape of the data is: ", df.shape)
        logger.debug(df.info())
        # filter out rows with dates earlier than 2014-01-01
        df['date'] = pd.to_datetime(df['date_clean'])
        df = df[df['date'] >= '2014-01-01']
        logger.debug("The shape of the data is: ", df.shape)

        # load locationsfromarticle.zip file
        locationsfromarticle = (
            pd.read_csv("sl_data_for_dashboard//locationsfromarticle.zip",
                        usecols={
                            'article_id',
                            'location',
                        },
                        dtype={
                            'article_id': 'int64',
                            'location': 'object',
                        },
                        compression='zip')
                        )
        logger.debug(locationsfromarticle.info())
        # load unique_locations.csv file
        locations = pd.read_csv("sl_data_for_dashboard//unique_locations.csv",
                                dtype={
                                    'location': 'object',
                                    'latitude': 'float64',
                                    'longitude': 'float64',
                                    'state': 'object',
                                    'country': 'object',
                                    'continent': 'object',
                                    'subcontinent': 'object',
                                    'ignore': 'int64'
                                    },
                                )
        logger.debug(locations.info())
        # set all null subcontinent values to continent value
        locations['subcontinent'] = (
            locations['subcontinent'].fillna(locations['continent']))
        # set null country values to subcontinent value
        locations['country'] = (
            locations['country'].fillna(locations['subcontinent']))
        # set null state values to country value
        locations['state'] = (
            locations['state'].fillna(locations['country']))

        # match locations to locationsfromarticle
        locations['location'] = (
            locations['location'].str.lower())
        locationsfromarticle['location'] = (
            locationsfromarticle['location'].str.lower())
        locationsmerged = (
            locations.merge(locationsfromarticle, on='location', how='left'))
        logger.debug(locationsmerged.info())
        del locations, locationsfromarticle

        # drop all rows with ignore = 1
        locationsmerged = locationsmerged[locationsmerged['ignore'] != 1]
        locationsmerged = locationsmerged.drop(columns=['ignore'])
        logger.debug(locationsmerged.head())
        logger.debug(locationsmerged.info())

        # merge locationsmerged with df only keep rows with a match
        locationgraphdf = df.merge(locationsmerged,
                                   on='article_id',
                                   how='left')
        logger.debug(locationgraphdf.info())
        logger.debug(locationgraphdf.head())

        # create a new dataframe with the number of fake articles
        # per country, continent, and subcontinent
        fakearticles = locationgraphdf[locationgraphdf['label'] == 1]
        fakearticles = fakearticles.groupby([
            'year',
            'month',
            'day',
            'date',
            'state',
            'country',
            'continent',
            'subcontinent']).size().reset_index(name='fake_count')

        # create a new dataframe with the number of real articles
        # per country, continent, and subcontinent
        realarticles = locationgraphdf[locationgraphdf['label'] == 0]
        realarticles = realarticles.groupby([
            'year', 'month', 'day', 'date',
            'state', 'country', 'continent',
            'subcontinent']).size().reset_index(name='real_count')

        # merge fake and real articles dataframes
        articles = pd.merge(fakearticles, realarticles, on=['year',
                                                            'month',
                                                            'day',
                                                            'date',
                                                            'state',
                                                            'country',
                                                            'continent',
                                                            'subcontinent'],
                            how='outer').fillna(0)
        articles = articles.sort_values(by=['year',
                                            'month',
                                            'day',
                                            'date',
                                            'state',
                                            'country',
                                            'continent',
                                            'subcontinent'])
        logger.debug(articles.head())

        # save articles dataframe as data//articlesformap.csv
        articles.to_csv(file_path, index=False)

    return articles


@log_function_call(logger)
def dashboarddata():
    df = pd.read_csv("sl_data_for_dashboard//dashboard_data.zip",
                     usecols=None,  # Import all columns
                     dtype={
                        'index': int,
                        'title': str,
                        'subject': str,
                        'label': int,
                        'media_type': str,
                        'month': int,
                        'day': int,
                        'year': int,
                        'day_of_week': str,
                        'week_of_year': int,
                        'is_weekend': int,
                        'is_weekday': int,
                        'holiday': int,
                        'day_label': str,
                        'article_id': int,
                        'source_name': str,
                        'title_length': int,
                        'text_length': int,
                        'article_polarity': float,
                        'article_subjectivity': float,
                        'title_polarity': float,
                        'title_subjectivity': float,
                        'overall_polarity': float,
                        'overall_subjectivity': float,
                        'contradiction_polarity': float,
                        'contradiction_subjectivity': float,
                        'polarity_variations': float,
                        'subjectivity_variations': float,
                        'sentiment_article': str,
                        'sentiment_title': str,
                        'sentiment_overall': str,
                        'unique_location_count': int
                     },
                     compression='zip',
                     # Ensure date_clean is imported as a date
                     parse_dates=['date_clean']
                     )
    logger.debug("The shape of the data is: ", df.shape)
    logger.debug(df.head())

    # load data in to data_clean st.session_state
    return df


@log_function_call(logger)
def wordcountdata():
    # Use low_memory=False to avoid dtype inference issues
    df = pd.read_csv(
        "sl_data_for_dashboard//preprocessed_wordcloud.zip",
        dtype={
            'label': 'int8',
            'text': 'object',
        },
        compression='zip',
        low_memory=False
        )
    logger.debug("The shape of the data is: ", df.shape)
    logger.debug(df.head())

    # load data in to wordcountdata
    return df


# Path: f_dashboard/data_prep.py
# end of file
