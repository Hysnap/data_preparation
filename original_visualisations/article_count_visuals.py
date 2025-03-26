import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sl_components.filters import filter_by_date
from sl_utils.logger import log_function_call, streamlit_logger
from matplotlib.ticker import FuncFormatter


@log_function_call(streamlit_logger)
def plot_article_count_by_subject(target_label="Article Count by Subject",
                                  pageref_label="article_subject_count"):
    """
    Plots a bar chart of the count of articles by subject,
    split by Label (Real=1, Dubious=0), with color coding.
    Allows option to show as percentage split or stacked count.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Check if required columns exist
    if "subject" not in df.columns or "label" not in df.columns:
        st.error("Dataset missing required columns: 'subject' or 'label'.")
        return

    # User option to select display type
    display_type = st.radio(
        "Select Display Type",
        options=["Count", "Percentage"],
        index=0,
        key=pageref_label
    )

    # Aggregate count of articles per subject split by label
    article_counts = df.groupby(["subject",
                                 "label"]).size().reset_index(name="count")

    if display_type == "Percentage":
        # Calculate percentage split
        total_counts = (
            article_counts.groupby("subject")["count"].transform("sum"))
        article_counts["percentage"] = (
            (article_counts["count"] / total_counts) * 100)
        y_value = "percentage"
        y_label = "Percentage of Articles"
    else:
        y_value = "count"
        y_label = "Count of Articles"
    my_pal = {0: "green", 1: "red"}
    # Create bar plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.barplot(
        data=article_counts,
        x="subject",
        y=y_value,
        hue="label",
        palette=my_pal,
        alpha=0.7
    )

    ax.set_title("Article Count by Subject (Real vs Dubious)", fontsize=10)
    ax.set_xlabel("Subject", fontsize=6)
    ax.set_ylabel(y_label, fontsize=6)
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles,
              title="Label",
              labels=["Dubious (0)", "Real (1)"],
              fontsize=10)
    plt.xticks(rotation=45)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_count_by_source(target_label="Article Count by Source",
                                 pageref_label="article_source_count"):
    """
    Plots a bar chart of the count of articles by source,
    split by Label (Real=1, Dubious=0), with color coding.
    Allows option to show as percentage split or stacked count.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Check if required columns exist
    if "source_name" not in df.columns or "label" not in df.columns:
        st.error("Dataset missing required columns: 'source_name' or 'label'.")
        return

    # User option to select display type
    display_type = st.radio(
        "Select Display Type",
        options=["Count", "Percentage"],
        index=0,
        key=pageref_label
    )

    # Aggregate count of articles per source split by label
    article_counts = df.groupby(["source_name",
                                 "label"]).size().reset_index(name="count")

    if display_type == "Percentage":
        # Calculate percentage split
        total_counts = (
            article_counts.groupby("source_name")["count"].transform("sum"))
        article_counts["percentage"] = (
            (article_counts["count"] / total_counts) * 100)
        y_value = "percentage"
        y_label = "Percentage of Articles"
    else:
        y_value = "count"
        y_label = "Count of Articles"
    my_pal = {0: "green", 1: "red"}
    # Create bar plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.barplot(
        data=article_counts,
        x="source_name",
        y=y_value,
        hue="label",
        palette=my_pal,
        alpha=0.7
    )

    ax.set_title("Article Count by Source (Real vs Dubious)", fontsize=10)
    ax.set_xlabel("Source", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    handles, _ = ax.get_legend_handles_labels()
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    ax.legend(handles=handles, title="Label",
              labels=["Dubious (0)", "Real (1)"], fontsize=8)
    plt.xticks(rotation=45)

    # Display visualization in Streamlit
    st.pyplot(fig)




@log_function_call(streamlit_logger)
def plot_article_count_by_media(target_label="Article Count by media",
                                pageref_label="article_media_count"):
    """
    Plots a bar chart of the count of articles by media,
    split by Label (Real=1, Dubious=0), with color coding.
    Allows option to show as percentage split or stacked count.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Check if required columns exist
    if "media_type" not in df.columns or "label" not in df.columns:
        st.error("Dataset missing required columns: 'media_type' or 'label'.")
        return

    # User option to select display type
    display_type = st.radio(
        "Select Display Type",
        options=["Count", "Percentage"],
        index=0,
        key=pageref_label
    )

    # Aggregate count of articles per media type split by label
    article_counts = df.groupby(["media_type",
                                 "label"]).size().reset_index(name="count")

    if display_type == "Percentage":
        # Calculate percentage split
        total_counts = (
            article_counts.groupby("media_type")["count"].transform("sum"))
        article_counts["percentage"] = (
            (article_counts["count"] / total_counts) * 100)
        y_value = "percentage"
        y_label = "Percentage of Articles"
    else:
        y_value = "count"
        y_label = "Count of Articles"

    # Sort media types by total count in descending order
    sorted_media_types = article_counts.groupby("media_type")["count"].sum().sort_values(ascending=False).index

    my_pal = {1: "green", 0: "red"}
    # Create bar plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.barplot(
        data=article_counts,
        x="media_type",
        y=y_value,
        hue="label",
        palette=my_pal,
        alpha=0.7,
        order=sorted_media_types
    )

    ax.set_title("Article Count by Media (Real vs Dubious)", fontsize=10)
    ax.set_xlabel("Media Type", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles,
              title="Label",
              labels=["Dubious (0)", "Real (1)"], fontsize=10)
    plt.xticks(rotation=45)

    # Display visualization in Streamlit
    st.pyplot(fig)


@log_function_call(streamlit_logger)
def plot_article_count_by_day_label(target_label="Article Count by Day Label",
                                    pageref_label="article_day_count"):
    """
    Plots a bar chart of the count of articles by source,
    split by Label (Real=1, Dubious=0), with color coding.
    Allows option to show as percentage split or stacked count.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Check if required columns exist
    if "day_label" not in df.columns or "label" not in df.columns:
        st.error("Dataset missing required columns: 'day_label' or 'label'.")
        return

    # User option to select display type
    display_type = st.radio(
        "Select Display Type",
        options=["Count", "Percentage"],
        index=0,
        key=pageref_label
    )

    # Aggregate count of articles per day_label split by label
    article_counts = df.groupby(["day_label",
                                 "label"]).size().reset_index(name="count")

    if display_type == "Percentage":
        # Calculate percentage split
        total_counts = (
            article_counts.groupby("day_label")["count"].transform("sum"))
        article_counts["percentage"] = (
            (article_counts["count"] / total_counts) * 100)
        y_value = "percentage"
        y_label = "Percentage of Articles"
    else:
        y_value = "count"
        y_label = "Count of Articles"
    my_pal = {0: "green", 1: "red"}
    # Create bar plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.barplot(
        data=article_counts,
        x="day_label",
        y=y_value,
        hue="label",
        palette=my_pal,
        alpha=0.7,
        order=article_counts["day_label"].unique()[::-1]
    )

    ax.set_title("Article Count by Day (Real vs Dubious)", fontsize=10)
    ax.set_xlabel("Day Label", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    handles, _ = ax.get_legend_handles_labels()
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(handles=handles,
              title="Label",
              labels=["Dubious (0)", "Real (1)"],
              fontsize=8)
    plt.xticks(rotation=45)

    # Split y-axis

    def y_formatter(y, pos):
        if y < 10000:
            return f'{y}'
        elif 10000 <= y < 22000:
            return ''
        elif 22000 <= y < 30000:
            return f'{y - 12000}'
        elif 30000 <= y < 50000:
            return ''
        else:
            return f'{y - 22000}'

    ax.set_yscale('linear')
    ax.set_yticks([0, 10000, 22000, 30000, 50000])
    ax.get_yaxis().set_major_formatter(FuncFormatter(y_formatter))

    # Display visualization in Streamlit
    st.pyplot(fig)


# graph using data_clean to show count of Real and Dubious articles by day
@log_function_call(streamlit_logger)
def plot_article_count_by_day(target_label="Article Count by Day",
                              pageref_label="article_day_count2"):
    """
    Plots a line graph of the count of articles by day,
    split by Label (Real=1, Dubious=0), with color coding.
    Allows option to show as percentage split or stacked count.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Check if required columns exist
    if "date_clean" not in df.columns or "label" not in df.columns:
        st.error("Dataset missing required columns: 'date_clean' or 'label'.")
        return
    df = df[df["date_clean"] >= '2015-01-01']
    # User option to select display type
    display_type = st.radio(
        "Select Display Type",
        options=["Count", "Percentage"],
        index=0,
        key=pageref_label
    )

    # Aggregate count of articles per day split by label
    article_counts = df.groupby(["date_clean",
                                 "label"]).size().reset_index(name="count")

    if display_type == "Percentage":
        # Calculate percentage split
        total_counts = (
            article_counts.groupby("date_clean")["count"].transform("sum"))
        article_counts["percentage"] = (
            (article_counts["count"] / total_counts) * 100)
        y_value = "percentage"
        y_label = "Percentage of Articles"
    else:
        y_value = "count"
        y_label = "Count of Articles"
    my_pal = {0: "green", 1: "red"}
    # Create line plot
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(
        data=article_counts,
        x="date_clean",
        y=y_value,
        hue="label",
        palette=my_pal,
        alpha=0.7
    )

    ax.set_title("Article Count by Day (Real vs Dubious)", fontsize=10)
    ax.set_xlabel("Day", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles,
              title="Label",
              labels=["Dubious (0)", "Real (1)"],
              fontsize=10)
    plt.xticks(rotation=45)

    # Display visualization in Streamlit
    st.pyplot(fig)


# graph using data_clean to show count of Real and Dubious articles
# by number of locations mentioned in the article
@log_function_call(streamlit_logger)
def plot_article_count_by_location(target_label="Article Count by Location",
                                   pageref_label="article_location_count"):
    """
    Plots a bar chart of the count of articles by number of locations
    mentioned in the article,
    split by Label (Real=1, Dubious=0), with color coding.
    Allows option to show as percentage split or stacked count.
    """

    # Load dataset from session state
    df = st.session_state.get("data_clean", None)

    if df is None:
        st.error("No data found. Please upload a dataset.")
        return

    # Check if required columns exist
    if "unique_location_count" not in df.columns or "label" not in df.columns:
        st.error("Dataset missing required columns:"
                 " 'unique_location_count' or 'label'.")
        return

    # User option to select display type
    display_type = st.radio(
        "Select Display Type",
        options=["Count", "Percentage"],
        index=0,
        key=pageref_label
    )

    # Aggregate count of articles per location count split by label
    article_counts = df.groupby(["unique_location_count",
                                 "label"]).size().reset_index(name="count")

    if display_type == "Percentage":
        # Calculate percentage split
        total_counts = (
            article_counts.groupby("unique_location_count")
            ["count"].transform("sum"))
        article_counts["percentage"] = (
            (article_counts["count"] / total_counts) * 100)
        y_value = "percentage"
        y_label = "Percentage of Articles"
    else:
        y_value = "count"
        y_label = "Count of Articles"
    my_pal = {0: "green", 1: "red"}
    # Create bar plot
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.barplot(
        data=article_counts,
        x="unique_location_count",
        y=y_value,
        hue="label",
        palette=my_pal,
        alpha=0.7
    )

    ax.set_title("Article Count by Location Count (Real vs Dubious)",
                 fontsize=10)
    ax.set_xlabel("Location Count", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles,
              title="Label",
              labels=["Dubious (0)", "Real (1)"],
              fontsize=10)
    plt.xticks(rotation=45)

    # Show every 5th x-axis label
    for index, label in enumerate(ax.get_xticklabels()):
        if index % 5 != 0:
            label.set_visible(False)

    # Display visualization in Streamlit
    st.pyplot(fig)