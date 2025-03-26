import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sl_components.filters import filter_by_date
from sl_utils.logger import log_function_call, streamlit_logger
from matplotlib.ticker import FuncFormatter


@log_function_call(streamlit_logger)
def plot_article_vs_title_subjectivity(
    Target_label="Article vs Title subjectivity",
    pageref_label="subjectivity_scatter"):
    """
    Plots scatter graph of text vs title subjectivity scores
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
    # Create plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=filtered_df,
        y="title_subjectivity",
        x="article_subjectivity",
        hue="label",
        palette=my_pal,
        alpha=0.7,
        ax=ax
    )

    ax.set_title("Scatter Plot of Subjectivity Articles vs Titles",
                 fontsize=10)
    ax.set_ylabel("Title Subjectivity",
                  fontsize=8)
    ax.set_xlabel("Article Subjectivity",
                  fontsize=8)
    ax.legend(title="Label",
                labels=["Dubious (0)", "Real (1)"],
                fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_subjectivity_vs_polarity(
    target_label="Article Subjectivity vs Polarity",
    pageref_label="Article_S_V_P_scatter"):
    """
    Plots scatter graph of polarity vs article subjectivity scores
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
    # Option to show/hide date slide
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
    # Create plot
    fig, ax = plt.subplots(figsize=(3, 3))

    sns.scatterplot(
        data=filtered_df,
        y="article_subjectivity",
        x="article_polarity",
        hue="label",
        palette=my_pal,
        alpha=0.7,
        ax=ax
    )

    ax.set_title("Scatter Plot of Article Subjectivity vs Polarity",
                 fontsize=10)
    ax.set_ylabel("Article Subjectivity",
                  fontsize=8)
    ax.set_xlabel("Article Polarity",
                  fontsize=8)
    ax.legend(title="Label",
              labels=["Dubious (0)", "Real (1)"],
              fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_title_subjectivity_vs_polarity(
    target_label="Title Subjectivity vs Polarity",
    pageref_label="Title_S_V_P_scatter"):
    """
    Plots scatter graph of polarity vs title subjectivity scores
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
    # Option to show/hide date slide
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
    # Create plot
    fig, ax = plt.subplots(figsize=(3, 3))

    sns.scatterplot(
        data=filtered_df,
        y="title_subjectivity",
        x="title_polarity",
        hue="label",
        palette=my_pal,
        alpha=0.7,
        ax=ax
    )

    ax.set_title("Scatter Plot of Title Subjectivity vs Polarity",
                 fontsize=10)
    ax.set_ylabel("Title Subjectivity",
                  fontsize=8)
    ax.set_xlabel("Title Polarity",
                  fontsize=8)
    ax.legend(title="Label",
              labels=["Dubious (0)", "Real (1)"],
              fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_vs_title_subjectivity_scat(
    Target_label="Article vs Title subjectivity",
    pageref_label="subjectivity_scatter"):
    """
    Plots scatter graph of text vs title subjectivity scores
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
    show_slider = st.checkbox(
        "Show Date Slider",
        value=False,
        key=f"{pageref_label}_slider"
    )

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

    # Ensure `label` column exists
    if "label" not in filtered_df:
        st.error("The dataset is missing the 'label' column.")
        return

    # Convert `label` to string for correct hue mapping
    filtered_df["label"] = filtered_df["label"].astype(str)

    # Drop rows where both subjectivity values are missing
    filtered_df.dropna(subset=["title_subjectivity",
                               "article_subjectivity"],
                       how="all", inplace=True)

    # Define color palette for labels
    my_pal = {0: "green", 1: "red"}

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6))

    sns.scatterplot(
        data=filtered_df,
        y="title_subjectivity",
        x="article_subjectivity",
        hue="label",
        palette=my_pal,
        alpha=0.7,
        ax=ax
    )

    # Customize plot
    ax.set_title("Scatter Plot of Subjectivity: Articles vs Titles", fontsize=12)
    ax.set_ylabel("Title Subjectivity", fontsize=10)
    ax.set_xlabel("Article Subjectivity", fontsize=10)
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_subjectivity_contrad_variations(
    target_label="Subjectivity Contradiction vs Variations",
        pageref_label="Sub_con_var_scatter"):
    """
    Plots scatter graph of subjectivity contradictions vs variations
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
        y="contradiction_subjectivity",
        x="subjectivity_variations",
        hue="label",
        palette=my_pal,
        alpha=0.7
    )

    ax.set_title("Scatter Plot of Subjectivity Variations and Contradictions",
                 fontsize=10)
    ax.set_ylabel("Subjectivity Contradictions", fontsize=8)
    ax.set_xlabel("Subjectivity Variations", fontsize=8)
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)

