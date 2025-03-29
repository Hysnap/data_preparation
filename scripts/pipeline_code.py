# Modular refactored version of the full data pipeline

import pandas as pd
import re
import spacy
from sl_utils.geo_utils import (
    build_world_location_set,
    init_keyword_processor,
    extract_locations_column,
    enrich_unique_locations,
    get_geolocation_info,
    extract_geolocation_details
)
from sl_utils.logger import datapipeline_logger as logger
from sl_utils.utils import (
    separate_string,
    clean_text,
    get_sentiment,
    categorize_polarity,
    categorize_subjectivity,
    save_dataframe_to_zip,
    string_to_list
)

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


def prepare_combined_df(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preparing combined dataframe...")

    df = df.reset_index(drop=True)
    df['article_id'] = df.index
    df.index.name = 'index'

    df[['source', 'cleaned_text']] = df['article_text'].apply(
        lambda x: pd.Series(extract_source_and_clean(x))
    )

    df['source'] = df['source'].replace('', 'UNKNOWN (Unknown)')
    df['source'] = df['source'].fillna('UNKNOWN (Unknown)')

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    df[['location', 'source_name']] = df['source'].apply(
        lambda x: pd.Series(separate_string(x))
    )

    df['location'] = df['location'].str.strip()
    df['source_name'] = df['source_name'].str.strip()
    df['location'] = df['location'].str.split('/').apply(
        lambda parts: [part.strip() for part in parts] if isinstance(parts, list) else ['UNKNOWN']
    )
    df['location'] = df['location'].fillna('UNKNOWN')
    df['source_name'] = df['source_name'].fillna('Unknown')

    df['title_length'] = df['title'].apply(len)
    df['text_length'] = df['cleaned_text'].apply(len)

    return df


def process_nlp(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Processing NLP text...")
    df['nlp_text'] = df['cleaned_text'].apply(clean_text)
    df['nlp_title'] = df['title'].apply(clean_text)
    df['nlp_location'] = df['location'].apply(
        lambda locations: [clean_text(loc) for loc in locations]
    )
    df['nlp_textloc'] = df['nlp_text'] + ' ' + df['nlp_location'].apply(lambda x: ' '.join(x))
    return df


def apply_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Applying sentiment analysis...")

    df[['article_polarity', 'article_subjectivity']] = df['nlp_text'].apply(
        lambda x: pd.Series(get_sentiment(x))
    )
    df[['title_polarity', 'title_subjectivity']] = df['nlp_title'].apply(
        lambda x: pd.Series(get_sentiment(x))
    )

    df['overall_polarity'] = (df['article_polarity'] + df['title_polarity']) / 2
    df['overall_subjectivity'] = (df['article_subjectivity'] + df['title_subjectivity']) / 2
    df['contradiction_polarity'] = df['article_polarity'] - df['title_polarity']
    df['contradiction_subjectivity'] = df['article_subjectivity'] - df['title_subjectivity']

    df['polarity_variations'] = df.apply(
        lambda row: row['contradiction_polarity'] / row['title_polarity'] if row['title_polarity'] != 0 else 0,
        axis=1
    )
    df['subjectivity_variations'] = df.apply(
        lambda row: row['contradiction_subjectivity'] / row['title_subjectivity'] if row['title_subjectivity'] != 0 else 0,
        axis=1
    )

    df['sentiment_article'] = df['article_polarity'].apply(categorize_polarity) + ' ' + df['article_subjectivity'].apply(categorize_subjectivity)
    df['sentiment_title'] = df['title_polarity'].apply(categorize_polarity) + ' ' + df['title_subjectivity'].apply(categorize_subjectivity)
    df['sentiment_overall'] = df['overall_polarity'].apply(categorize_polarity) + ' ' + df['overall_subjectivity'].apply(categorize_subjectivity)

    return df


def finalize_data(df: pd.DataFrame, save_to_disk: bool = False) -> pd.DataFrame:
    logger.info("Finalizing dataframe...")

    df['subject'] = df['subject'].fillna('Unknown')

    df.drop(['nlp_textloc', 'source', 'location', 'nlp_location'], axis=1, inplace=True, errors='ignore')
    df = df.dropna(subset=['cleaned_text'])

    if save_to_disk:
        save_dataframe_to_zip(df, 'data/combined_data_step1.zip', 'combined_data_step1.csv')
        save_dataframe_to_zip(df, 'data/combined_data_step2.zip', 'combined_data_step2.csv')

    return df


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
                valid_locations_set=location_set
                )
    else:
        combined_df = extract_locations_column(
            combined_df,
            'nlp_textloc',
            keyword_processor,
            nlp=nlp,
            valid_locations_set=location_set
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
