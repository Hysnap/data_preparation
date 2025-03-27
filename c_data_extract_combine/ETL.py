# Description: This script loads the data from the csv files, cleans the data,
# extracts the location, source, and removes the text, performs sentiment analysis,
# calculates contradictions and variations, categorizes sentiments, appends NLP locations
# to text, extracts locations from articles, and saves the combined data to a csv file.
# The script also extracts the location, source, and removes the text, splits the locationsfromarticle
# into a separate dataframe, summarizes the locationsfromarticle data by article_id and location,
# creates a dataframe of unique locations, and filters out locations marked to ignore.
# The script also extracts latitude, longitude, and address, extracts continent, country, and state,
# searches for a location in multiple columns of worldcities_df, and saves the common themes data to a csv file.
# The script also generates a list of common themes, creates a dataframe of common themes, and saves the common themes data to a csv file.
# The script also drops unnecessary columns, drops rows with empty cleaned_text, and exports the data to a csv file.
# The script also sets month, day, and year for null values based off date_clean, and exports the data to a csv file.
# The script also drops unnecessary columns, and exports the data to a csv file.

import pandas as pd
import re
from geopy.geocoders import GoogleV3
import nltk
import spacy
import os
import tqdm
import zipfile
from flashtext import KeywordProcessor
from sl_utils.logger import datapipeline_logger as logger
from sl_utils.logger import log_function_call
from sl_utils.utils import (
    checkdirectory,
    save_dataframe_to_zip,
    separate_string,
    clean_text,
    get_sentiment,
    categorize_polarity,
    categorize_subjectivity,
    string_to_list,
    get_geolocation_info,
    extract_geolocation_details,
    extract_locations,
                            )
from c_data_extract_combine.TRANSFORM import classify_and_combine


# attach tqdm to pandas
tqdm.pandas

# Download necessary NLTK resources (if you haven't already)
nltk.download('stopwords')
nltk.download('wordnet')

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize the geolocator
# Load the API key from an environment variable
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError("Google API key not found. Please set "
                       "the GOOGLE_API_KEY environment variable.")

geolocator = GoogleV3(api_key=api_key)


@log_function_call(logger)
# load the data from the csv files
def dataload():
    logger.info("Loading fake news data...")
    fake_df = pd.read_csv("2_source_data/fake.csv.zip")
    logger.info("Loading true news data...")
    true_df = pd.read_csv("2_source_data/true.csv.zip")
    logger.info("Loading test data...")
    test_data_df = pd.read_csv("2_source_data/testdata.csv.zip")
    logger.info("Loading combined misinformation data...")
    comb_misinfo_df = pd.read_csv("2_source_data/combined_misinfo.zip")
    logger.info("Appending combined misinformation data to test data...")
    test_data_df = pd.concat([test_data_df,
                              comb_misinfo_df],
                              ignore_index=True)
    logger.info("Dropping combined misinformation data from memory...")
    del comb_misinfo_df
    logger.info("Data loading completed.")
    return fake_df, true_df, test_data_df


@log_function_call(logger)
# Function to extract the location, source, and remove the text
def extract_source_and_clean(text):
    # Define the regex pattern to match the source
    # (location + source in parentheses + hyphen)
    pattern = r'^[A-Za-z\s,/.]+ \([A-Za-z]+\) -'
    match = re.match(pattern, text)
    # if there is data in match then extract the source
    if match:
        # Extract the matched portion (location + source + hyphen)
        source = match.group(0).strip()
        # Remove the matched portion from the original
        # text to get the cleaned text
        cleaned_text = text.replace(source, '').strip()
        return source, cleaned_text
    else:
        return '', text


@log_function_call(logger)
# Function to load and clean the data
def data_pipeline(useprecombineddata=False, usepostnlpdata=False):
    logger.info("Starting data pipeline...")
    if usepostnlpdata is True:
        useprecombineddata = True
    # Pipeline to transform the data
    if useprecombineddata is False:
        logger.info("Loading data...")
        fake_df, true_df, test_data_df = dataload()

    if usepostnlpdata is False:
        logger.info("Cleaning data...")
        logger.info("Combining dataframes...")
        if useprecombineddata:
            combined_df = pd.read_csv('data/combined_pre_clean.zip')
        else:
            combined_df = classify_and_combine(true_df, fake_df, test_data_df)
        logger.info(combined_df.info())
        logger.info("Checking for duplicated columns...")
        logger.info("Duplicated columns:"
                    f" {combined_df.columns[combined_df.
                                            columns.duplicated()]}")
        logger.info("Resetting index...")
        combined_df.reset_index(drop=True, inplace=True)
        logger.info("Renaming index to article_id...")
        combined_df['article_id'] = combined_df.index
        combined_df.index.name = 'index'
        logger.info("Extracting source and cleaning text...")
        combined_df[['source',
                    'cleaned_text']] = combined_df['article_text'].apply(
            lambda x: pd.Series(extract_source_and_clean(x))
        )

        logger.info("Replacing empty sources with 'UNKNOWN (Unknown)'...")
        combined_df['source'].replace('', 'UNKNOWN (Unknown)', inplace=True)
        combined_df['source'].fillna('UNKNOWN (Unknown)', inplace=True)

        logger.info("Removing excess spaces from all columns...")
        combined_df = (
            combined_df.
            apply(lambda x: x.str.strip() if x.dtype == "object" else x))

        logger.info("Splitting source into location and source name...")
        combined_df[['location', 'source_name']] = combined_df['source'].apply(
            lambda x: pd.Series(separate_string(x))
        )
        combined_df['location'] = combined_df['location'].str.strip()
        combined_df['source_name'] = combined_df['source_name'].str.strip()

        logger.info("Processing locations...")
        combined_df['location'] = combined_df['location'].str.split('/')
        combined_df['location'] = (
            combined_df['location'].apply(lambda x: [i.strip() for i in x]))
        combined_df['location'] = combined_df['location'].fillna('UNKNOWN')
        combined_df['source_name'] = (
            combined_df['source_name'].fillna('Unknown'))

        logger.info("Calculating text and title lengths...")
        combined_df['title_length'] = combined_df['title'].apply(len)
        combined_df['text_length'] = combined_df['cleaned_text'].apply(len)

        logger.info("Cleaning text for NLP...")
        combined_df['nlp_text'] = combined_df['cleaned_text'].apply(clean_text)
        combined_df['nlp_title'] = combined_df['title'].apply(clean_text)
        combined_df['nlp_location'] = (
            combined_df['location'].
            apply(lambda x: [clean_text(i) for i in x]))
        logger.info("Saving post NLP data to CSV...")
        save_dataframe_to_zip(combined_df,  'data/combined_data_postnlp.zip',
                              'combined_data_postnlp.csv')
    else:
        try:
            logger.info("Loading post nlp data...")
            combined_df = pd.read_csv('data/combined_data_postnlp.zip')

            # Check data has loaded correctly
            logger.info(combined_df.info())
            logger.info("Checking for duplicated columns...")
            logger.info("Duplicated columns:"
                        f" {combined_df.columns[combined_df.
                            columns.duplicated()]}")
            logger.info("Resetting index...")
            combined_df.reset_index(drop=True, inplace=True)
            logger.info("Renaming index to article_id...")
            combined_df['article_id'] = combined_df.index
            combined_df.index.name = 'index'
        except (FileNotFoundError, ImportError):
            logger.error("Post NLP data not found. Generating new data...")
            usepostnlpdata = False
            combined_df = data_pipeline()

    logger.info("Performing sentiment analysis...")
    combined_df[['article_polarity', 'article_subjectivity']] = (
        combined_df['nlp_text'].apply(
            lambda x: pd.Series(get_sentiment(x))
            ))
    combined_df[['title_polarity', 'title_subjectivity']] = (
        combined_df['nlp_title'].apply(
            lambda x: pd.Series(get_sentiment(x))
            ))
    combined_df['overall_polarity'] = (
        (combined_df['article_polarity'] + combined_df['title_polarity']) / 2)
    combined_df['overall_subjectivity'] = (
        (combined_df['article_subjectivity'] +
         combined_df['title_subjectivity']) / 2)

    logger.info("Calculating contradictions and variations...")
    combined_df['contradiction_polarity'] = (
        combined_df['article_polarity'] - combined_df['title_polarity'])
    combined_df['contradiction_subjectivity'] = (
        combined_df['article_subjectivity'] -
        combined_df['title_subjectivity'])
    combined_df['polarity_variations'] = combined_df.apply(
        lambda row: row['contradiction_polarity'] / row['title_polarity']
        if row['title_polarity'] != 0 else 0, axis=1)
    combined_df['subjectivity_variations'] = (combined_df.apply(
        lambda row: row['contradiction_subjectivity'] /
        row['title_subjectivity']
        if row['title_subjectivity'] != 0 else 0, axis=1))

    logger.info("Categorizing sentiments...")
    combined_df['sentiment_article'] = (
        combined_df['article_polarity'].apply(categorize_polarity) +
        " " + combined_df['article_subjectivity'].
        apply(categorize_subjectivity))
    combined_df['sentiment_title'] = (
        combined_df['title_polarity'].apply(categorize_polarity) +
        " " + combined_df['title_subjectivity'].apply(categorize_subjectivity))
    combined_df['sentiment_overall'] = (
        combined_df['overall_polarity'].apply(categorize_polarity) +
        " " + combined_df['overall_subjectivity'].
        apply(categorize_subjectivity))

    logger.info("Appending NLP locations to text...")
    combined_df['nlp_textloc'] = (
        combined_df['nlp_text'] + ' '
        + combined_df['nlp_location'].apply(lambda x: ' '.join(x)))

    # Load NLP model globally to avoid repeated loading
    nlp = spacy.load("en_core_web_sm")

    # Load unique locations dataset once
    df_unique_locations = pd.read_csv("data/unique_locations.csv")

    # Filter out locations marked to ignore
    df_unique_locations = (
        df_unique_locations[df_unique_locations["ignore"] != 1])

    # Convert locations to a set for fast lookup
    unique_locations_set = set(df_unique_locations["location"].str.lower())

    # Initialize FlashText KeywordProcessor for fast extraction
    keyword_processor = KeywordProcessor()
    for loc in unique_locations_set:
        keyword_processor.add_keyword(loc)
    # use nlp to extract locations from the text

    logger.info("Extracting locations from articles...")
    combined_df['locationsfromarticle'] = (
        combined_df['nlp_textloc']
        .progressapply(extract_locations(keyword_processor,
                                         nlp)))

    logger.info("Saving combined data post location to CSV...")
    save_dataframe_to_zip(combined_df,
                          'data/combined_data_step2.zip',
                          'combined_data_step2.csv')

    logger.info("filling nulls with default values...")
    combined_df['subject'] = combined_df['subject'].fillna('Unknown')

    logger.info("Dropping unnecessary columns...")
    combined_df.drop([
                      'nlp_textloc',
                      'source',
                      'location',
                      'nlp_location'], axis=1, inplace=True)

    logger.info("Dropping rows with empty cleaned_text...")
    combined_df = combined_df.dropna(subset=['cleaned_text'])
    logger.info("Number of rows in combined dataframe"
                f" after dropping empty cleaned_text: {combined_df.shape[0]}")

    logger.info("Saving combined data to CSV...")
    save_dataframe_to_zip(combined_df,
                          'data/combined_data_step1.zip',
                          'combined_data_step1.csv')

    logger.info("Data pipeline completed.")
    return combined_df


@log_function_call(logger)
def find_location_match(location, worldcities_df):
    """
    Search for a location in multiple columns of worldcities_df.

    Args:
        location (str): The location to search for.
        worldcities_df (DataFrame): The dataframe containing city data.

    Returns:
        dict: A dictionary containing the matched value, the column it was
        found in, latitude, longitude, and country.
              Returns None if no match is found.

    Example usage:
        location_to_search = "New York"
        result = find_location_match(location_to_search, worldcities_df)
        if result:
            print("Match found:", result)
        else:
            print("No match found.")
    """
    search_columns = ['city', 'city_ascii', 'country',
                      'iso2', 'iso3', 'admin_name']

    # Convert input to string and lowercase for case-insensitive comparison
    location = str(location).strip().lower()

    for col in search_columns:
        # Find rows where the location matches the column
        match = worldcities_df[worldcities_df[col]
                               .astype(str)
                               .str.strip()
                               .str.lower() == location]

        if not match.empty:
            # Extract the first match found
            result = {
                'matched_value': match.iloc[0][col],
                'matched_column': col,
                'latitude': match.iloc[0]['lat'],
                'longitude': match.iloc[0]['lng'],
                'country': match.iloc[0]['country']
            }
            return result
    return None  # Return None if no match is found


# Extract continent, country, and state
@log_function_call(logger)
def extract_geolocation_details(df_unique_locations, worldcities_df):
    """
    Extract the continent, country, and state from an address.
    merge the worldcities data with the unique
    locations data on location to city
     Drop the temporary 'geolocation_info' and 'address' columns
     """
    logger.info("Using worldcities data...")
    # add a column which classifies the location
    # as a city, country or other
    zip_file_path = '2_source_data/simplemaps_worldcities_basicv1.77.zip'
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        with z.open('worldcities.csv') as f:
            worldcities_df = pd.read_csv(f)
    print(worldcities_df.head(5))
    # change city, city_ascii, country, iso2, iso3, admin_name to lowercase
    worldcities_df['city'] = worldcities_df['city'].str.lower()
    worldcities_df['city_ascii'] = worldcities_df['city_ascii'].str.lower()
    worldcities_df['country'] = worldcities_df['country'].str.lower()
    worldcities_df['iso2'] = worldcities_df['iso2'].str.lower()
    worldcities_df['iso3'] = worldcities_df['iso3'].str.lower()
    worldcities_df['admin_name'] = worldcities_df['admin_name'].str.lower()
    print(worldcities_df.head(5))
    # change the location column to lowercase
    df_unique_locations['location'] = (
        df_unique_locations['location'].str.lower())
    # update unique_locations country, state, city, latitude, longitude
    # with the information from worldcities data
    df_unique_locations['geolocation_info'] = (
        df_unique_locations['location']
        .apply(lambda x: find_location_match(x, worldcities_df))
        )
    # Extract latitude, longitude, and address
    df_unique_locations['latitude'] = (
        df_unique_locations['geolocation_info']
        .apply(lambda x: x['latitude'] if x else None))
    df_unique_locations['longitude'] = (
        df_unique_locations['geolocation_info']
        .apply(lambda x: x['longitude'] if x else None))
    df_unique_locations['address'] = (
        df_unique_locations['geolocation_info']
        .apply(lambda x: x['matched_value'] if x else None))
    df_unique_locations.drop(columns=['geolocation_info', 'address'],
                                inplace=True)
    # update df_locations with the information from df_unique_locations
    df_locations = pd.merge(df_locations,
                            df_unique_locations,
                            on='location',
                            how='left')
    # save the locationsfromarticle data to a csv file

    save_dataframe_to_zip(df_locations,
                        'data/locationsfromarticle.zip',
                        'locationsfromarticle.csv')
    # save the unique locations data to a csv file
    save_dataframe_to_zip(df_unique_locations,
                        'data/unique_locations.zip',
                        'unique_locations.csv')