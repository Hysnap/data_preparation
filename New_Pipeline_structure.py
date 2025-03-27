from sl_utils.logger import datapipeline_logger as logger
from scripts.dependency_resolver import resolve_flag_dependencies
from scripts.nlp_processing import run_nlp_processing
from scripts.raw_data_loader import load_raw_data
from scripts.sentiment_analysis import run_sentiment_analysis
from scripts.location_enrichment import run_location_enrichment
from scripts.final_cleanup import finalize_data
import pandas as pd


def data_pipeline(flags: dict, save_to_disk: bool = False) -> pd.DataFrame:
    """
    Run the data pipeline.

    flags = {
    "useprecombineddata": False,
    "usepostnlpdata": False,
    "useprelocationdata": False,
    "useprelocationenricheddata": False
        }

    df = data_pipeline(flags, save_to_disk=False)  # Keeps everything in memory

    Args:
        flags (dict): _description_
        save_to_disk (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    logger.info("Starting data pipeline...")

    # Resolve dependencies
    resolved_flags = resolve_flag_dependencies(flags)

    # Load base data
    raw_data = load_raw_data(resolved_flags)

    # Stage 1: NLP Processing
    combined_df = run_nlp_processing(raw_data,
                                     resolved_flags,
                                     save_to_disk)

    # Stage 2: Sentiment
    combined_df = run_sentiment_analysis(combined_df,
                                         resolved_flags,
                                         save_to_disk)

    # Stage 3: Location enrichment
    combined_df = run_location_enrichment(combined_df,
                                          resolved_flags,
                                          save_to_disk)

    # Final cleanup
    combined_df = finalize_data(combined_df,
                                save_to_disk)

    logger.info("Data pipeline completed.")
    return combined_df
