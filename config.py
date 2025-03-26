# config.py - Stores global constants
import os
from pathlib import Path
# Base directory
BASE_DIR = Path(os.getcwd())

# Placeholder values
DIRECTORIES = {  # "directory_name": "directory_path"
    "BASE_DIR": Path(os.getcwd()),
    "data_dir": os.path.join("data"),
    "requirements_dir": os.path.join("a_requirement_gathering"),
    "source_dir": os.path.join("b_source_data"),
    "data_extract_combined_dir": os.path.join("c_data_extract_combine"),
    "transform_dir": os.path.join("d_transform"),
    "nlp_data_processing_dir": os.path.join("e_nlp_data_processing"),
    "dashboard_dir": os.path.join("f_dashboard"),
    "future_work_dir": os.path.join("j_future_developments"),
    "sl_logs_dir": os.path.join("sl_logs"),
    "pl_logs_dir": os.path.join("z_logs"),
    "app_pages_dir": os.path.join("sl_app_pages"),
    "utils_dir": os.path.join("sl_utils"),
    "components_dir": os.path.join("sl_components"),
    "reference_dir": os.path.join("sl_reference_files"),
    "visualisation_dir": os.path.join("sl_visualisations"),
}

# File paths
FILENAMES = {  # "directory" : {"file_name": "file_path"}
    "source_dir": {
        "combined_misinfo_fname": "combined_misinfo.zip",
        "countryContinent_fname": "countryContinent.csv",
        "fake_news_sources_fname": "fake.csv.zip",
        "true_news_sources_fname": "true.csv.zip",
        "USstates_fname": "USstates.csv",
        "worldcities_fname": "simplemaps_worldcities_basicv1.73.zip",
    },
    "data_dir": {
        "combined_data_fname": "combined_data.zip",
        "combined_data_cleaned_fname": "combined_data_cleaned.zip",
        "combined_data_postnlp_fname": "combined_data_postnlp.zip",
        "combined_data_step1_fname": "combined_data_step1.zip",
        "combined_data_step2_fname": "combined_data_step2.zip",
        "combined_pre_clean_fname": "combined_pre_clean.zip",
        "locationsfromarticles_fname": "locationsfromarticles.zip",
        "MachineLearning_results_fname": "MachineLearning_results.csv",
        "unique_locations_fname": "unique_locations.csv",
        "article_locations_fname": "articleformapc.csv",
    },
    "dashboard_dir": {
        "dashboard_fname": "Real_or_Dubious_news.pbix"
    },
    "transform_dir": {
        "ML_model_fname": "ML_models.py"},
    "reference_dir": {
        "page_settings_fname": "page_settings.json",
        "last_modified_fname": "last_modified_dates.json",
        "admin_text_fname": "admin_text.json",
        "admin_credentials_fname": "admin_credentials.json",
    },
    "sl_logs_dir": {
        "log_fname": "sl_app.log",
    },
    "pl_logs_dir": {
        "datapipeline_log_fname": "data_pipeline.log",
    },
}


# Data remappings
DATA_REMAPPINGS = {"label_remapping": {0: "True News", 1: "Fake News"}
                   }

# category filter definitions
FILTER_DEF = {  # "filter_name": {"column_name": "value"}
    "locations_ftr": {
        "ignore": 1
    },
    "fake_news_ftr": {
        "label": 1
    },
    "true_news_ftr": {
        "label": 0
    },
    "validdates_ftr": {
        "date": "> 2001-01-01"
    }, }

SECURITY = {  # "security_variable": "security_value"
    "is_admin": False,
    "is_authenticated": False,
    "username": "",
    "password": "",
}
