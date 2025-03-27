from sl_utils.logger import datapipeline_logger as logger
from sl_utils.utils import save_dataframe_to_zip
from scripts.pipeline_code import prepare_combined_df
from scripts.location_enrichment import generate_unique_loc
import pandas as pd


def run_location_enrichment(df, flags, save_to_disk):
    if flags.get("useprelocationenricheddata"):
        try:
            df = pd.read_csv('data/combined_data_pre_location_enriched.zip', low_memory=False)
            return prepare_combined_df(df)
        except FileNotFoundError:
            logger.warning("Pre-location enriched data not found. Regenerating...")

    worldcities_df = pd.read_csv("b_source_data/simplemaps_worldcities_basicv1.77.zip")
    usstates_df = pd.read_csv("b_source_data/USStates.csv")
    countrycontinent_df = pd.read_csv("b_source_data/countryContinent.csv")

    df_locations, df_locations_sum, df_unique_locations = generate_unique_loc(
        df, worldcities_df, usstates_df, countrycontinent_df, flags.get("useprelocationenricheddata")
    )

    if save_to_disk:
        save_dataframe_to_zip(df, 'data/combined_data_pre_location_enriched.zip', 'combined_data_pre_location_enriched.csv')
        save_dataframe_to_zip(df_unique_locations, 'data/unique_locations.zip', 'unique_locations.csv')
        save_dataframe_to_zip(df_locations, 'data/locations_from_article.zip', 'locations_from_article.csv')

    return df