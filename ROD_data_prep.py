"""
script to run ETL, Transform and other data prep tasks for ROD data
     ↓
"""
import pandas as pd
from scripts.common_themes import common_themes
from sl_utils.logger import datapipeline_logger as logger
from sl_utils.utils import (
     checkdirectory,
     save_dataframe_to_zip,
     get_geolocation_info,
     extract_geolocation_details,
     find_location_match,
     )
from c_data_extract_combine.ETL import (
     string_to_list,
     data_pipeline
     )
from d_transform.ML_model2 import run_pipeline
from sl_nlp.Ngram_summary_pipeline import run_full_ngram_pipeline

# Generate Combined Data and save as a csv file
useprecombineddata = False
usepostnlpdata = True
usesavedfile = False
usegeoapi = False
useworldcitiesdata = False
findcommonthemes = False

checkdirectory()

logger.info("Starting data pipeline...")
if usesavedfile is True:
     try:
          logger.info("Loading saved data...")
          try:
               combined_df = pd.read_csv('data/combined_data_step1.zip')
          except (FileNotFoundError, ImportError):
               logger.info("Saved data not found. Generating new data...")
               combined_df = data_pipeline(useprecombineddata=True,
                                           usepostnlpdata=True)
     except Exception as e:
          logger.error(f"Error loading saved data: {e}")
          logger.info("Generating new data...")
          combined_df = data_pipeline(useprecombineddata=True,
                                      usepostnlpdata=False)
     else:
          combined_df = data_pipeline(useprecombineddata, usepostnlpdata)

# check for any blank values
logger.debug(combined_df.isnull().sum())
logger.debug(combined_df.head(5))
# rename index to article_id if column article_id does not exist
if 'article_id' not in combined_df.columns:
     combined_df.index.name = 'index'
     combined_df['article_id'] = combined_df.index
# split locationsfromarticle into separate datafram
# create a referenced list based of locationsfromarticle with an
# entry for each location in the list
# and a reference to the index of the article
logger.info("Splitting locationsfromarticle into separate dataframe...")
df_locations = combined_df[['article_id',
                              'locationsfromarticle']].copy()
logger.debug(df_locations.head(5))
df_locations['locationsfromarticle'] = (
     df_locations['locationsfromarticle'].apply(string_to_list))
df_locations = (
     df_locations.explode('locationsfromarticle')
     .rename(columns={'locationsfromarticle': 'location'})
)
logger.debug(df_locations.head(10))
# summarise the locationsfromarticle data by article_id and location adding
# a count of the number of times the location appears in the article
logger.info("Summarizing locationsfromarticle data...")
df_locations_sum = (
     df_locations.groupby(['article_id',
                           'location'])
     .size()
     .reset_index(name='count'))
logger.debug(df_locations_sum.head(10))
# create a dataframe of unique locations
df_unique_locations = (
     pd.DataFrame({'location': df_locations['location'].unique()}))
logger.debug(df_unique_locations.shape)

if usegeoapi:
     logger.info("Using geolocation API...")
     # generate a list of rows with missing data and without the ignore flag = 1
     missing_geolocation_info = (
          df_unique_locations[
               df_unique_locations['geolocation_info'].isnull()
          ].query('ignore != 1')
          )
     # check the number of missing geolocation_info
     print(missing_geolocation_info.shape)
     # Apply geolocation info extraction
     missing_geolocation_info['geolocation_info'] = (
          missing_geolocation_info['location']
          .apply(get_geolocation_info))
     # Extract latitude, longitude, and address
     missing_geolocation_info['latitude'] = (
          missing_geolocation_info['geolocation_info']
          .apply(lambda x: x['latitude']))
     missing_geolocation_info['longitude'] = (
          missing_geolocation_info['geolocation_info']
          .apply(lambda x: x['longitude']))
     missing_geolocation_info['address'] = (
          missing_geolocation_info['geolocation_info']
          .apply(lambda x: x['address']))
     # Extract continent, country, and state
     missing_geolocation_info[['continent', 'country', 'state']] = (
          missing_geolocation_info['address'].apply(
               lambda x: pd.Series(extract_geolocation_details(x))
               ))
else:
     if useworldcitiesdata:
          logger.info("Using world cities data...")
          # load world cities data
          worldcities = pd.read_csv('data/worldcities.csv')
          # merge world cities data with unique locations data
          df_unique_locations = find_location_match(df_unique_locations,
                                                    worldcities)
          logger.debug(df_unique_locations.shape)

logger.info("Data pipeline completed.")
logger.info("Data cleaning and feature extraction completed.")

common_themes(findcommonthemes, combined_df)

# if no blank or null values in title, article_text, date, label,
# subject then export to csv
# set month, day and year for null values based off date_clean

# drop unnecessary columns
combined_df.drop(['article_text',
                  'subject',
                  ], axis=1, inplace=True)

if combined_df.isnull().sum().sum() == 0:
     logger.info("No missing values in the data")
     save_dataframe_to_zip(combined_df,
                           'data/combined_data.zip',
                           'combined_data.csv')
     logger.info("Data cleaned and saved as combined_data.csv in data folder")
else:
     logger.info("There are still missing values in the data")
     save_dataframe_to_zip(combined_df,
                           'data/combined_data.zip',
                           'combined_data.csv')
     logger.info("Data cleaned and saved as combined_data.csv in data folder")
     logger.info("Data cleaning and feature extraction completed.")

# regenerate MachineLearning Models and Data
run_pipeline(data_path="sl_data_for_dashboard/training_data.zip",
             model_type="classification")

# run ngram and wordcloud analysis and image creation
run_full_ngram_pipeline(input_csv="sl_data_for_dashboard/training_data.zip",
                        output_csv="sl_data_for_dashboard/ngram_data.zip",
                        output_image="wordclouds/ngram_image.png")
