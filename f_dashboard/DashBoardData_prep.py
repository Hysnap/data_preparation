# create function which recieves a file and summarises the data for use in sl_data_for_dashboard
from sl_utils.logger import datapipeline_logger as logger, log_function_call
from sl_utils.utils import (
    checkdirectory,
    save_dataframe_to_zip,
    string_to_list,
    )
)

@log_function_call
def DashBoardData_prep(db):
    # if db is None load combined_data.zip
    # else load db

    if db is None:
        logger.info("Loading saved data...")
        try:
            combined_df = pd.read_csv('data/combined_data.zip')
        except (FileNotFoundError, ImportError):
            logger.info("Saved data not found. Please rerun data pipeline...")
            return None
    else:
        combined_df = db

    # check for any blank values
    logger.debug(combined_df.isnull().sum())
    logger.debug(combined_df.head(5))

    #prepare and summarise data for dashboard

    # drop unique identifier columns - title, article_id, 
    combined_df.drop(columns=['title', 'article_id'], inplace=True)

    # summarise by all columns and add count of articles - only include rows for categorical columns where values exist
    combined_df_summary = combined_df.groupby(combined_df.columns.tolist(), as_index=False).size().reset_index(name='count')
    