"""
Configuration file for the package.
"""
import os
import streamlit as st
from sl_utils.logger import log_function_call, streamlit_logger as logger
from sl_utils.version import get_git_version
from sl_utils.global_variables import initialize_session_state


@log_function_call(logger)
def setup_package():
    """
    Function to setup the package.
    """
    # log current file and path
    logger.info("Setting up the package")
    # Define the directory
    MODULE_DIR = os.path.dirname(__file__)
    logger.info(f"Module directory: {MODULE_DIR}")
    # Define the name of the package
    # packagename = "Real Or Dubious Analysis Dashboard"
    # Define the version of the package
    try:
        version = get_git_version(MODULE_DIR)
        logger.info(f"Version: {version}")
    except Exception as e:
        logger.critical(f"App setup crashed: {e}", exc_info=True)
        st.error(f"App setup failed. Please check logs. {e}")
        raise SystemExit("App setup failed. Exiting.")
    # Define the description of the package
    # packagedescription = "A package to clean and dedupe data"
    # Define the author of the package
    # author = "Paul Golder"
    # Define the email address of the package author
    # author_email = "PGOLDER1972@gmail.com"
    # Define the dependencies of the package
    # requirements = [
    #     "setuptools",
    #     "numpy",
    #     "scipy",
    #     "pandas",
    #     "matplotlib",
    #     "seaborn",
    #     "plotly",
    #     "patsy",
    #     "statsmodels",
    #     "streamlit",
    #     "pytest",
    #     "bcrypt"
    # ]
    # Define the entry points of the package
    # packageentry_points = {"Real_Or_Dubioys_Dashboard": ["main = main:main"]}
    # Define the package data
    # package_data = {"data": ["data/ROD_Streamlit_Data.csv"]}
    # Define the package classifiers
    # classifiers = [
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ]
    # Define the package keywords
    # packagekeywords = ["data",
    #                    "cleaning",
    #                    "deduplication",
    #                    "Real or Dubious News analysis"]
    # Define the package URL
    # packageurl = ""
    # initialize variables
    try:
        initialize_session_state()
        logger.info("Package setup complete")
    except Exception as e:
        logger.critical(f"App setup crashed: {e}", exc_info=True)
        st.error(f"App setup failed. Please check logs. {e}")
        raise SystemExit("App setup failed. Exiting.")
    return
