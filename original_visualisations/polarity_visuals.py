import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sl_components.filters import filter_by_date
from sl_utils.logger import log_function_call, streamlit_logger
from matplotlib.ticker import FuncFormatter


@log_function_call(streamlit_logger)
def plot_article_vs_title_polarity(target_label="Article vs Title Polarity",
                                   pageref_label="polarity_scatter"):
    """
    Plots scatter graph of article_polarity vs title_polarity
    with label color coding and date filtering.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Ensure the date column is in datetime format
    df["date_clean"] = pd.to_datetime(df["date_clean"])

    # Retrieve min/max date for filtering
    min_date = df["date_clean"].min().date()
    max_date = df["date_clean"].max().date()

    # Date selection slider
    show_slider = st.checkbox("Show Date Slider",
                              value=False,
                              key=f"{pageref_label}_slider")

    if show_slider:
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key=pageref_label
        )
    else:
        start_date, end_date = min_date, max_date

    # Filter data using the existing filter method
    filtered_df = filter_by_date(df, pd.to_datetime(start_date),
                                 pd.to_datetime(end_date), "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return

    my_pal = {0: "green", 1: "red"}
    # Create plot
    fig, ax = plt.subplots(figsize=(3, 3))

    sns.scatterplot(
        data=filtered_df,
        x="title_polarity",
        y="article_polarity",
        hue="label",  # Color by label (Real = 1, Dubious = 0)
        palette=my_pal,
        alpha=0.7,
        ax=ax
        )

    ax.set_title("Plot of Article Polarity vs Title Polarity",
                 fontsize=10)
    ax.set_xlabel("Title Polarity",
                  fontsize=8)
    ax.set_ylabel("Article Polarity",
                  fontsize=8)
    ax.legend(title="Label",
                labels=["Dubious (0)", "Real (1)"],
                fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_vs_title_polarity(
    target_label="Article vs Title polarity",
        pageref_label="polarity_scatter"):
    """
    Plots scatter graph of text vs title polarity scores
    with label color coding and date filtering.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Ensure the date column is in datetime format
    df["date_clean"] = pd.to_datetime(df["date_clean"])

    # Retrieve min/max date for filtering
    min_date = df["date_clean"].min().date()
    max_date = df["date_clean"].max().date()

    # Date selection slider
    show_slider = st.checkbox("Show Date Slider",
                              value=False,
                              key=f"{pageref_label}_slider")

    if show_slider:
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key=pageref_label
        )
    else:
        start_date, end_date = min_date, max_date

    # Filter data using the existing filter method
    filtered_df = filter_by_date(df,
                                 pd.to_datetime(start_date),
                                 pd.to_datetime(end_date),
                                 "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return
    my_pal = {0: "green", 1: "red"}
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=filtered_df,
        y="title_polarity",
        x="article_polarity",
        hue="label",
        palette=my_pal,
        alpha=0.7
    )

    ax.set_title("Scatter Plot of Polarities Articles vs Titles", fontsize=10)
    ax.set_ylabel("Title Polarity", fontsize=8)
    ax.set_xlabel("Article Polarity", fontsize=8)
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_polarity_contrad_variations(
    target_label="Polarity Contradiction vs Variations",
        pageref_label="Pol_con_var_scatter"):
    """
    Plots scatter graph of polarity contradictions vs variations
    with label color coding and date filtering.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Ensure the date column is in datetime format
    df["date_clean"] = pd.to_datetime(df["date_clean"])

    # Retrieve min/max date for filtering
    min_date = df["date_clean"].min().date()
    max_date = df["date_clean"].max().date()

    show_slider = st.checkbox("Show Date Slider",
                              value=False,
                              key=f"{pageref_label}_slider")

    if show_slider:
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key=pageref_label
        )
    else:
        start_date, end_date = min_date, max_date

    # Filter data using the existing filter method
    filtered_df = filter_by_date(df,
                                 pd.to_datetime(start_date),
                                 pd.to_datetime(end_date),
                                 "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return
    my_pal = {1: "red", 0: "green"}
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=filtered_df,
        y="contradiction_polarity",
        x="polarity_variations",
        hue="label",
        palette=my_pal,
        alpha=0.7
    )

    ax.set_title("Scatter Plot of Polarity Variations and Contradictions",
                 fontsize=10)
    ax.set_ylabel("Polarity\nContradictions", fontsize=8)
    ax.set_xlabel("Polarity\nVariations", fontsize=8)
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)
