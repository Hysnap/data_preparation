﻿"""
script to run ETL, Transform and other data prep tasks for ROD data

    flags = {
    "useprecombineddata": False,
    "usepostnlpdata": False,
    "useprelocationdata": False,
    "useprelocationenricheddata": False
        }

    df = data_pipeline(flags, save_to_disk=False)

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
from scripts.dashboarddata import run_dashboard_data_generation



# Generate Combined Data and save as a csv file
flags = {
    "run_pipeline": True,
    "useprecombineddata": False,
    "usepostnlpdata": False,
    "useprelocationdata": False,
    "useprelocationenricheddata": True,
    "usesavedfile": False,
    "usegeoapi": False,
    "useworldcitiesdata": False,
    "producedashboarddata": True,
    "findcommonthemes": False,
    "runMLmodelgen": False,
    "run_ngam_gen": False,
    }

checkdirectory()

if flags.get("run_pipeline"):
    logger.info("Starting data pipeline...")
    combined_df = data_pipeline(flags, save_to_disk=True)

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
    combined_df = combined_df.drop(['article_text',
                                    ], axis=1)

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
else:
    logger.info("Data pipeline not run.")
    logger.info("Please set run_pipeline flag to True to run the data pipeline.")

if flags.get("producedashboarddata"):
    logger.info("Producing dashboard data...")
    # check if combined_data.zip exists
    try:
        combined_df = pd.read_csv('data/combined_data.zip')
    except (FileNotFoundError, ImportError):
        logger.error("Combined data not found. Please run the data pipeline.")
    else:
        logger.info("Producing dashboard data...")
        # run dashboard data generation
        run_dashboard_data_generation(combined_df)
        logger.info("Dashboard data generation completed.")


if flags.get("runMLmodelgen"):
    logger.info("Running ML model...")
    # check if combined_data.zip exists
    try:
        combined_df = pd.read_csv('data/combined_data.zip')
    except (FileNotFoundError, ImportError):
        logger.error("Combined data not found. Please run the data pipeline.")
    else:
        logger.info("Running ML model...")
        run_pipeline(data_path="data/combined_data.zip",
                     model_type="classification")


if flags.get("run_ngam_gen"):
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
