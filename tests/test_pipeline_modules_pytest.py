"""
_summary_
This script is a test script for the pipeline modules. It tests the functions in the pipeline modules using pytest.
The functions tested are:
- resolve_flag_dependencies
- load_raw_data
- run_nlp_processing
- run_sentiment_analysis
- run_location_enrichment
- finalize_data

to run the test, run the following command in the terminal:

pytest tests/test_pipeline_modules_pytest.py

The test will run and display the results in the terminal.

"""

# test_pipeline_modules_pytest.py

import pytest
import pandas as pd
from unittest.mock import patch
from pandas.testing import assert_frame_equal

from scripts.dependency_resolver import resolve_flag_dependencies
from scripts.raw_data_loader import load_raw_data
from scripts.nlp_processing import run_nlp_processing
from scripts.sentiment_analysis import run_sentiment_analysis
from scripts.location_enrichment import run_location_enrichment
from scripts.final_cleanup import finalize_data

# -----------------------------
# Dependency Resolver Tests
# -----------------------------

def test_resolve_dependencies():
    flags = {
        "useprecombineddata": False,
        "usepostnlpdata": False,
        "useprelocationdata": False,
        "useprelocationenricheddata": True
    }
    result = resolve_flag_dependencies(flags)
    assert result['useprelocationdata'] is True
    assert result['usepostnlpdata'] is True
    assert result['useprecombineddata'] is True


# -----------------------------
# Raw Data Loader
# -----------------------------

@patch("scripts.raw_data_loader.dataload")
def test_load_raw_data(mock_dataload):
    mock_dataload.return_value = ('fake', 'true', 'test')
    flags = {"useprecombineddata": False}
    result = load_raw_data(flags)
    assert result == ('fake', 'true', 'test')


# -----------------------------
# NLP Processing
# -----------------------------

@patch("scripts.nlp_processing.pd.read_csv")
@patch("scripts.nlp_processing.prepare_combined_df")
@patch("scripts.nlp_processing.process_nlp")
def test_run_nlp_processing_from_file(mock_process_nlp, mock_prepare_df, mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame({'article_text': ['text'], 'title': ['title']})
    mock_prepare_df.return_value = pd.DataFrame({'cleaned_text': ['clean'], 'title': ['title'], 'location': [['l1']]})
    mock_process_nlp.return_value = pd.DataFrame({'nlp_text': ['n'], 'nlp_title': ['n'], 'nlp_location': [['l1']], 'nlp_textloc': ['n l1']})
    flags = {"usepostnlpdata": True, "useprecombineddata": True}
    result = run_nlp_processing((None, None, None), flags, save_to_disk=False)
    assert 'nlp_text' in result.columns


# -----------------------------
# Sentiment Analysis
# -----------------------------

@patch("scripts.sentiment_analysis.pd.read_csv")
@patch("scripts.sentiment_analysis.prepare_combined_df")
@patch("scripts.sentiment_analysis.apply_sentiment")
def test_run_sentiment_analysis(mock_apply, mock_prepare, mock_read):
    mock_read.return_value = pd.DataFrame({'nlp_text': ['test'], 'nlp_title': ['title']})
    mock_prepare.return_value = mock_read.return_value
    mock_apply.return_value = pd.DataFrame({'overall_polarity': [0.5]})
    flags = {"useprelocationdata": True}
    result = run_sentiment_analysis(pd.DataFrame(), flags, save_to_disk=False)
    assert 'overall_polarity' in result.columns


# -----------------------------
# Location Enrichment
# -----------------------------

@patch("scripts.location_enrichment.pd.read_csv")
@patch("scripts.location_enrichment.prepare_combined_df")
@patch("scripts.location_enrichment.generate_unique_loc")
def test_run_location_enrichment(mock_generate, mock_prepare, mock_read):
    mock_read.return_value = pd.DataFrame({'some': ['data']})
    mock_prepare.return_value = mock_read.return_value
    mock_generate.return_value = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame({'latitude': [0.0]}))
    flags = {"useprelocationenricheddata": True}
    result = run_location_enrichment(pd.DataFrame(), flags, save_to_disk=False)
    assert isinstance(result, pd.DataFrame)


# -----------------------------
# Final Cleanup
# -----------------------------

def test_finalize_data():
    df = pd.DataFrame({
        'subject': [None, 'Politics'],
        'cleaned_text': ['some text', None],
        'nlp_textloc': ['abc', 'def'],
        'source': ['X', 'Y'],
        'location': ['US', 'UK'],
        'nlp_location': [['a'], ['b']]
    })
    cleaned_df = finalize_data(df, save_to_disk=False)
    assert 'nlp_textloc' not in cleaned_df.columns
    assert cleaned_df['subject'].notna().all()
    assert cleaned_df['cleaned_text'].notna().all()
