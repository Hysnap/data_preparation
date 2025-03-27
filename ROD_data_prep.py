"""
script to run ETL, Transform and other data prep tasks for ROD data
     ↓
"""
import pandas as pd
from sl_utils.logger import datapipeline_logger as logger
from sl_utils.utils import (
     checkdirectory,
     save_dataframe_to_zip,
     )
from New_Pipeline_structure import (
     data_pipeline,
     )
from d_transform.ML_model2 import run_pipeline
from sl_nlp.ngram_summary_pipeline import run_full_ngram_pipeline


# Generate Combined Data and save as a csv file
useprecombineddata = False
usepostnlpdata = False
useprelocationdata = False
useprelocationenricheddata = False
usesavedfile = True
usegeoapi = True
useworldcitiesdata = False
findcommonthemes = False
runMLmodelgen = False
run_ngam_gen = False

checkdirectory()

logger.info("Starting data pipeline...")
combined_df = data_pipeline(useprecombineddata,
                            usepostnlpdata,
                            useprelocationdata,
                            useprelocationenricheddata)

# check for any blank values
logger.debug(combined_df.isnull().sum())
logger.debug(combined_df.head(5))

# rename index to article_id if column article_id does not exist
if 'article_id' not in combined_df.columns:
    combined_df.index.name = 'index'
    combined_df['article_id'] = combined_df.index

logger.info("Data pipeline completed.")
logger.info("Data cleaning and feature extraction completed.")

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
# drop all dataframes from memory
del combined_df

if runMLmodelgen:
    # check if combined_data.zip exists
    try:
        combined_df = pd.read_csv('data/combined_data.zip')
    except (FileNotFoundError, ImportError):
        logger.error("Combined data not found. Please run the data pipeline.")
    else:
        logger.info("Running ML model...")
        run_pipeline(data_path="data/combined_data.zip",
                     model_type="classification")


if run_ngam_gen:
    # check combined_df has data
    try:
        combined_df = pd.read_csv('data/combined_data.zip')
    except (FileNotFoundError, ImportError):
        logger.error("Combined data not found. Please run the data pipeline.")
    else:
        logger.info("Running ngram and wordcloud analysis...")
    # run ngram and wordcloud analysis and image creation
    run_full_ngram_pipeline(input_csv="data/combined_data.zip",
                            output_csv="data/ngram_data.zip",
                            output_image="wordclouds/ngram_image.png")
