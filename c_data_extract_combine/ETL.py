"""
Description: This script loads the data from the csv files, cleans the data,
extracts the location, source, and removes the text,
performs sentiment analysis,
calculates contradictions and variations, categorizes sentiments,
appends NLP locations
to text, extracts locations from articles, and saves the
combined data to a csv file.
The script also extracts the location, source, and removes the text,
splits the locationsfromarticle
into a separate dataframe, summarizes the locationsfromarticle
data by article_id and location,
creates a dataframe of unique locations, and filters out
locations marked to ignore.
The script also extracts latitude, longitude, and address,
extracts continent, country, and state,
searches for a location in multiple columns of worldcities_df,
and saves the common themes data to a csv file.
The script also generates a list of common themes, creates a
dataframe of common themes, and saves the common themes data to a csv file.
The script also drops unnecessary columns, drops rows with
empty cleaned_text, and exports the data to a csv file.
The script also sets month, day, and year for null values
based off date_clean, and exports the data to a csv file.
The script also drops unnecessary columns, and exports the
data to a csv file.
"""

import pandas as pd
import re
import nltk
import spacy
from tqdm.auto import tqdm
from sl_utils.logger import datapipeline_logger as logger, log_function_call
from sl_utils.utils import (
    save_dataframe_to_zip,
    separate_string,
    clean_text,
    get_sentiment,
    categorize_polarity,
    categorize_subjectivity,
    string_to_list
                            )
from c_data_extract_combine.TRANSFORM import classify_and_combine
from sl_utils.geo_utils import (
    build_world_location_set,
    init_keyword_processor,
    extract_locations_column,
    enrich_unique_locations,
    get_geolocation_info,
    extract_geolocation_details
)

# attach tqdm to pandas
tqdm.pandas()

# Download necessary NLTK resources (if you haven't already)
nltk.download('stopwords')
nltk.download('wordnet')

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


@log_function_call(logger)
# load the data from the csv files
def dataload():
    logger.info("Loading fake news data...")
    fake_df = pd.read_csv("b_source_data/fake.csv.zip")
    logger.info("Loading true news data...")
    true_df = pd.read_csv("b_source_data/true.csv.zip")
    logger.info("Loading test data...")
    test_data_df = pd.read_csv("b_source_data/testdata.csv.zip")
    logger.info("Loading combined misinformation data...")
    comb_misinfo_df = pd.read_csv("b_source_data/combined_misinfo.zip")
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
def data_pipeline(useprecombineddata=False,
                  usepostnlpdata=False,
                  useprelocationdata=False,
                  useprelocationenricheddata=False):
    logger.info("Starting data pipeline...")
    if useprelocationenricheddata is True:
        useprelocationdata = True
        useprecombineddata = True
        usepostnlpdata = True
    elif useprelocationdata is True:
        useprecombineddata = True
        usepostnlpdata = True
    elif usepostnlpdata is True:
        useprecombineddata = True
        useprelocationdata = False
    # Pipeline to transform the data
    elif useprecombineddata is False:
        logger.info("Loading data...")
        fake_df, true_df, test_data_df = dataload()
    else:
        logger.error("Invalid data pipeline configuration.")
        return

    if usepostnlpdata is False:
        logger.info("Cleaning data...")
        logger.info("Combining dataframes...")
        if useprecombineddata:
            combined_df = pd.read_csv('data/combined_pre_clean.zip',
                                      low_memory=False)
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
    elif usepostnlpdata is True:
        try:
            logger.info("Loading post nlp data...")
            combined_df = pd.read_csv('data/combined_data_postnlp.zip',
                                      low_memory=False)

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
    else:
        logger.error("Invalid data pipeline configuration.")
        return

    if useprelocationdata is False:
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
            (combined_df['article_polarity'] +
             combined_df['title_polarity']) / 2)
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
            " " + combined_df['title_subjectivity'].apply(
                categorize_subjectivity))
        combined_df['sentiment_overall'] = (
            combined_df['overall_polarity'].apply(categorize_polarity) +
            " " + combined_df['overall_subjectivity'].
            apply(categorize_subjectivity))
        logger.info("Appending NLP locations to text...")
        combined_df['nlp_textloc'] = (
            combined_df['nlp_text'] + ' '
            + combined_df['nlp_location'].apply(lambda x: ' '.join(x)))
        # use savetozip function to save the combined data to a zip file
        logger.info("Saving combined data to CSV...")
        save_dataframe_to_zip(combined_df,
                              'data/combined_data_pre_location.zip',
                              'combined_data_pre_location.csv')
    elif useprelocationdata is True:
        if useprelocationenricheddata is True:
            try:
                logger.info("Loading pre location enriched data...")
                combined_df = pd.read_csv(
                    'data/combined_data_pre_location_enriched.zip',
                    low_memory=False)

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
                logger.error("Pre location enriched data not"
                             " found. Generating new data...")
                useprelocationenricheddata = False
        try:
            logger.info("Loading pre location data...")
            combined_df = pd.read_csv(
                'data/combined_data_pre_location.zip',
                low_memory=False)

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
            logger.error("Pre location data not found. Generating new data...")
            useprelocationdata = False
            combined_df = data_pipeline()
    else:
        logger.error("Invalid data pipeline configuration.")

    # use generate_unique_loc function to split locationsfromarticle
    # into separate dataframe
    # create a referenced list based of locationsfromarticle with an entry
    # for each location in the list
    # and a reference to the index of the article
    # Load geo datasets
    worldcities_df = (
        pd.read_csv("b_source_data/simplemaps_worldcities_basicv1.77.zip"))
    usstates_df = pd.read_csv("b_source_data/USStates.csv")
    countrycontinent_df = pd.read_csv("b_source_data/countryContinent.csv")

    # Generate unique location data
    logger.info("Generating unique location data...")
    (df_locations,
     df_locations_sum,
     df_unique_locations) = generate_unique_loc(combined_df,
                                                worldcities_df,
                                                usstates_df,
                                                countrycontinent_df,
                                                useprelocationenricheddata
                                                )

    # save the unique locations data to a csv file
    logger.info("Saving unique locations data to CSV...")
    save_dataframe_to_zip(df_unique_locations,
                          'data/unique_locations.zip',
                          'unique_locations.csv')

    # save the locationsfromarticle data to a csv file
    logger.info("Saving locations from article data to CSV...")
    save_dataframe_to_zip(df_locations,
                          'data/locations_from_article.zip',
                          'locations_from_article.csv')

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
def generate_unique_loc(combined_df,
                        worldcities_df,
                        usstates_df,
                        countrycontinent_df,
                        usegeoapi=False,
                        useprelocationenricheddata=False):
    logger.info("Generating location insights from articles...")

    logger.info("Not Loading pre location enriched data...")
    # Build location keyword set and keyword processor
    location_set = build_world_location_set(worldcities_df,
                                            usstates_df,
                                            countrycontinent_df
                                            )
    keyword_processor = init_keyword_processor(location_set)

    # Extract locations using NLP + FlashText
    if useprelocationenricheddata:
        logger.info("Loading pre location enriched data...")
        try:
            combined_df = (
                pd.read_csv('data/combined_data_pre_location_enriched.zip',
                            low_memory=False))
        except (FileNotFoundError, ImportError):
            logger.error("Pre location enriched data not found."
                         " Generating new data...")
            combined_df = extract_locations_column(
                combined_df,
                'nlp_textloc',
                keyword_processor,
                nlp=nlp,
                )
    else:
        combined_df = extract_locations_column(
            combined_df,
            'nlp_textloc',
            keyword_processor,
            nlp=nlp,
            )

    # Create exploded location mapping
    df_locations = combined_df[['article_id', 'locationsfromarticle']].copy()
    df_locations['locationsfromarticle'] = (
        df_locations['locationsfromarticle'].apply(string_to_list))
    df_locations = (
        df_locations.explode('locationsfromarticle').rename(
            columns={'locationsfromarticle': 'location'}))

    # Summarize location frequency per article
    df_locations_sum = (
        df_locations.groupby(['article_id', 'location'])
        .size()
        .reset_index(name='count')
    )

    # Build unique location list and assign ignore flag
    unique_locations_df = (
        pd.DataFrame({'location': df_locations['location'].dropna().unique()}))
    known_cities = (
        worldcities_df['city_ascii'].str.lower().dropna().unique())
    unique_locations_df['ignore'] = (
        unique_locations_df['location'].apply(
            lambda x: 0 if x in known_cities else 1
        ))

    # save file before enriching
    save_dataframe_to_zip(combined_df,
                          'data/ccombined_data_pre_location_enriched.zip',
                          'combined_data_pre_location_enriched.csv')

    # Enrich location data from city/state/country lookups
    df_unique_locations = enrich_unique_locations(
        unique_locations_df,
        worldcities_df,
        usstates_df,
        countrycontinent_df
    )

    # Fallback to API geolocation for unknowns
    if usegeoapi:
        missing = df_unique_locations[
            df_unique_locations['latitude'].isnull() &
            (df_unique_locations['ignore'] != 1)
        ].copy()

        if not missing.empty:
            logger.info(f"Using geolocation API for {len(missing)} "
                        "unmatched locations...")
            missing['geolocation_info'] = (
                missing['location'].apply(get_geolocation_info))
            missing['latitude'] = (
                missing['geolocation_info'].apply(lambda x: x['latitude']))
            missing['longitude'] = (
                missing['geolocation_info'].apply(lambda x: x['longitude']))
            missing['address'] = (
                missing['geolocation_info'].apply(lambda x: x['address']))
            missing[['continent', 'country', 'state']] = (
                missing['address'].apply(
                    lambda x: pd.Series(extract_geolocation_details(x))
                    ))
            df_unique_locations.update(missing)

    return df_locations, df_locations_sum, df_unique_locations
