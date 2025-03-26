import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sl_components.filters import filter_by_date
from sl_utils.logger import log_function_call, streamlit_logger
from matplotlib.ticker import FuncFormatter


@log_function_call(streamlit_logger)
def plot_article_vs_title_characters(
    target_label="Article vs Title character",
        pageref_label="char_scatter"):
    """
    Plots scatter graph of text_length vs title_length
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
        y="title_length",
        x="text_length",
        hue="label",
        palette=my_pal,
        alpha=0.7,
        ax=ax
        )
    ax.set_ylim(0, 2000)

    ax.set_title("Scatter Plot of Character Counts Articles vs Titles",
                 fontsize=10)
    ax.set_ylabel("Title Character Count", fontsize=8)
    ax.set_xlabel("Article Character Count", fontsize=8)
    ax.legend(title="Label", labels=["Dubious (0)", "Real (1)"], fontsize=10)

    # Display visualization in Streamlit
    st.pyplot(fig)


# Boxplot using data_clean to show the distribution o
# article text_count scores by label
@log_function_call(streamlit_logger)
def plot_article_text_count_distribution(
    target_label="Article Text Count Distribution",
        pageref_label="article_text_count_distribution"):
    """
    Plots a boxplot of the distribution of article text_count scores by label
    (Real=1, Dubious=0), with color coding.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Check if required columns exist
    if "text_length" not in df.columns or "label" not in df.columns:
        st.error("Dataset missing required columns: 'text_length' or 'label'.")
        return

    # Create boxplot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.boxplot(
        data=df,
        x="label",
        y="text_length",
        palette={1: "green", 0: "red"}
    )

    ax.set_title("Article Text Count Distribution by Label", fontsize=10)
    ax.set_xlabel("Label", fontsize=8)
    ax.set_ylabel("Text Count", fontsize=8)
    ax.set_xticklabels(["Dubious (0)", "Real (1)"])
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_hex_subjectivity():
    """
    Plots hexbin heatmap showing the weighted percentage of Real vs. Fake articles.
    - Uses `weights` to normalize for proportion.
    - Side histograms show volume distribution.
    """

    # Load dataset
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
        "Show Date Slider for Hex Plot",
        value=False,
        key="hex_slider"
    )

    if show_slider:
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key="hex_date"
        )
    else:
        start_date, end_date = min_date, max_date

    # Filter data
    filtered_df = filter_by_date(df, pd.to_datetime(start_date), pd.to_datetime(end_date), "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Ensure correct data types
    filtered_df["label"] = filtered_df["label"].astype(int)

    # Handle null and near-null values
    threshold = 0.001
    filtered_df.loc[filtered_df["title_subjectivity"].abs() < threshold, "title_subjectivity"] = None
    filtered_df.loc[filtered_df["article_subjectivity"].abs() < threshold, "article_subjectivity"] = None
    filtered_df.dropna(subset=["title_subjectivity", "article_subjectivity"], how="all", inplace=True)

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    grid = sns.jointplot(
        data=filtered_df,
        x="article_subjectivity",
        y="title_subjectivity",
        kind="hex",
        cmap="coolwarm",
        gridsize=30,
        marginal_ticks=True
    )

    # Calculate weights for each hexbin (percent of Real articles)
    x = filtered_df["article_subjectivity"]
    y = filtered_df["title_subjectivity"]
    c = filtered_df["label"]  # 0 for Fake, 1 for Real

    hexbin = grid.ax_joint.hexbin(
        x, y, C=c, reduce_C_function=np.mean, gridsize=30, cmap="coolwarm"
    )

    # Color bar
    cbar = fig.colorbar(hexbin, ax=grid.ax_joint, orientation="vertical")
    cbar.set_label("Proportion of Real Articles", fontsize=10)

    # Side histograms (volume distributions)
    sns.histplot(x, bins=30,
                 ax=grid.ax_marg_x, color="gray", kde=True)
    sns.histplot(y, bins=30,
                 ax=grid.ax_marg_y, color="gray", kde=True)

    # Titles and labels
    grid.ax_joint.set_title("Weighted Percentage of Real vs. Fake Articles",
                            fontsize=12)
    grid.ax_joint.set_xlabel("Article Subjectivity",
                             fontsize=10)
    grid.ax_joint.set_ylabel("Title Subjectivity",
                             fontsize=10)

    # Display
    st.pyplot(grid.fig)


@log_function_call(streamlit_logger)
def plot_hex_charcounts():
    """
    Plots hexbin heatmap showing the weighted percentage of Real vs. Fake articles.
    - Uses `weights` to normalize for proportion.
    - Side histograms show volume distribution.
    """

    # Load dataset
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
        "Show Date Slider for Hex Plot",
        value=False,
        key="hex_slider"
    )

    if show_slider:
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key="hex_date"
        )
    else:
        start_date, end_date = min_date, max_date

    # Filter data
    filtered_df = filter_by_date(df,
                                 pd.to_datetime(start_date),
                                 pd.to_datetime(end_date),
                                 "date_clean")

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return

    # Ensure correct data types
    filtered_df["label"] = filtered_df["label"].astype(int)

    # Handle null and near-null values
    threshold = 10
    filtered_df.loc[filtered_df["title_length"].abs() < threshold,
                    "title_length"] = None
    filtered_df.loc[filtered_df["text_length"].abs() < threshold,
                    "text_length"] = None
    filtered_df.dropna(subset=["title_length",
                               "text_length"],
                       how="all", inplace=True)

    # Filter lengths
    filtered_df = filtered_df[(filtered_df["title_length"] < 1000) &
                              (filtered_df["text_length"] < 25000)]

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    grid = sns.jointplot(
        data=filtered_df,
        x="text_length",
        y="title_length",
        kind="hex",
        cmap="coolwarm",
        gridsize=10,
        marginal_ticks=True
    )


    # Calculate weights for each hexbin (percent of Real articles)
    x = filtered_df["text_length"]
    y = filtered_df["title_length"]
    c = filtered_df["label"]  # 0 for Fake, 1 for Real

    hexbin = grid.ax_joint.hexbin(
        x, y, C=c, reduce_C_function=np.mean,
        gridsize=50, cmap="coolwarm"
    )

    # Color bar
    cbar = fig.colorbar(hexbin,
                        ax=grid.ax_joint,
                        orientation="horizontal")
    cbar.set_label("Proportion of Real Articles",
                   fontsize=10)

    # Side histograms (volume distributions)
    sns.histplot(x, bins=25,
                 ax=grid.ax_marg_x,
                 color="gray",
                 kde=True)
    sns.histplot(y, bins=25,
                 ax=grid.ax_marg_y,
                 color="gray",
                 kde=True)


    # Titles and labels
    grid.ax_joint.set_title("Weighted Percentage of Real vs. Fake Articles",
                            fontsize=8)
    grid.ax_joint.set_xlabel("Article Size in Chars",
                             fontsize=6)
    grid.ax_joint.set_ylabel("Title Size in Chars",
                             fontsize=6)
    grid.ax_joint.set_ylim(0, 1000)
    grid.ax_joint.set_xlim(0, 25000)

    # Display
    st.pyplot(grid.figure)
