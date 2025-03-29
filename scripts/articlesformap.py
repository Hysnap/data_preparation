from sl_utils.logger import datapipeline_logger as logger
import os
import pandas as pd
from sl_utils.utils import checkdirectory, save_dataframe_to_zip


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

if __name__ == "__main__":
    checkdirectory()
    mapdata()
    logger.info("Map data generation completed.")
    logger.info("Map data saved as articlesformap.csv in data folder")