# -*- coding: utf-8 -*-
"""
@author: mizho
@copyright: Qualcomm (C) 2023-2024
@version: 1.8 (Added interactive HTML plot for individual use cases, fluctuation summary in HTML)

Parse AU45A 'Adjusted' across versions whether:
  A) columns are repeated per-version blocks (grouped layout), or
  B) versions are separate columns (wide format).

This script reads an Excel file, detects its layout (grouped or wide),
extracts specified metric(s) data for a specified use case across different
versions, and then plots the trend(s) and saves the processed data to a CSV.


Usage:
  # Plot a single metric (backward compatible)
  python aggregate_caselevel_trend.py "Caselevel_Trendency.xlsx" --sheet 0 --usecase AU45A --metric Adjusted

  # Plot multiple metrics
  python aggregate_caselevel_trend.py "Caselevel_Trendency.xlsx" --sheet 0 --usecase AU45A --metric Adjusted CX Power

  # Plot multiple metrics & HTML for a single use case
  python aggregate_caselevel_trend.py "Caselevel_Trendency.xlsx" --sheet 0 --usecase AU45A --metric Adjusted CX Power --html-output


  # MODIFICATION: Plot for a predefined list of China-related use cases
  # Will also generate an aggregated plot/data for all China use cases
  # AND perform fluctuation analysis on 'Adjusted' metric if requested.
  # New: Interactive HTML plot for aggregate data including fluctuation summary.
  python aggregate_caselevel_trend.py "Caselevel_Trendency.xlsx" --sheet 0 --usecase China --metric Adjusted CX Power --html-output --analyze-fluctuation

Outputs:
  - For individual use cases (e.g., AU45A):
    - AU45A_Adjusted_trend.png (static plot)
    - AU45A_Adjusted_trend.html (interactive HTML plot if --html-output is used)
    - AU45A_Adjusted_trend_data.csv
  - For 'China' usecase (aggregate functionality):
    - Individual plots/CSVs for each usecase (e.g., RMSG01-5G_trend.png)
    - aggregate_China_Adjusted_trend.png (Static Plot, only if --no-html-fallback is not used with --html-output)
    - aggregate_China_Adjusted_trend.html (Interactive Plot, if --html-output is used, now includes fluctuation summary)
    - aggregate_China_Adjusted_data_long_format.csv (Combined data for all China use cases + all metrics, in long format)
    - aggregate_China_Adjusted_fluctuation.csv (Analysis of Adjusted metric's fluctuation, still saved separately)
  (These files will be located in a timestamped folder like output_YYYYMMDD_HHMMSS)
"""

from __future__ import annotations
import re
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # For color cycle
import seaborn as sns # For static aggregate plotting (fallback)
import numpy as np # For statistical calculations like median

# MODIFICATION: Import plotly
try:
    import plotly.express as px
    import plotly.io as pio
    PIO_AVAILABLE = True
    # MODIFICATION: Set a default template for clean HTML output
    pio.templates.default = "plotly_white"
except ImportError:
    PIO_AVAILABLE = False
    logging.warning("Plotly not found. Interactive HTML output will not be available. Please install it: pip install plotly pandas")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------- Constants ----------
# Regex to extract 'raw' version labels from complex strings in 'wide' layout.
# Matches 'M' followed by 2-3 digits, optionally followed by '-', '_', more digits,
# and/or letters (e.g., M800, M851-27, M850E).
# It tries to capture the most version-like part.
VERSION_EXTRACT_PATTERN = re.compile(
    r"M\d{2,3}(?:[_-]\d+)?(?:[A-Z]+)?",  # MXXX, MXXX-YY, MXXX_YY, MXXXE
    re.IGNORECASE
)

# Common header keywords
USECASE_KEYWORD = "usecase"
DEFAULT_METRIC = "Adjusted"
DEFAULT_USECASE = "AU45A"

# MODIFICATION: Predefined list of use cases for "China" parameter
CHINA_USECASES = [
    "RMSG01-5G",
    "DOUYIN02-5G",
    "AU45A",
    "VS12W",
    "DOUYIN01W",
    "GP01W",
    "HOK01",
    "QQS01-5G",
    "TNEWS01W",
    "WCHT02W",
    "PMWCHT02W",
    "WB01-5G",
    "VOWCHT01-5G",
    "VIWCHT01W",
    "APPLNCH03A",
    # Add more China-related use cases here as needed
]

# Plotting constants
PLOT_WIDTH_PER_VERSION = 1.2
PLOT_HEIGHT = 9
PLOT_DPI = 160

# Define a color cycle for multiple plots (for static plots)
COLOR_CYCLE = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values()) # More colors

# ---------- Helper Functions ----------
def _clean_str(value: Union[str, float]) -> str:
    """Helper to clean and convert to string, handling NaNs."""
    return str(value).strip() if pd.notna(value) else ""

def _extract_version_from_wide_col_name(column_name: str) -> Optional[str]:
    """
    Extracts a version label from a compound column name used in 'wide' layout.
    Examples: "M800 | Adjusted" -> "M800", "M850E (Final) Adjusted" -> "M850E"
    """
    column_name_upper = column_name.upper()
    match = VERSION_EXTRACT_PATTERN.search(column_name_upper)
    if match:
        return match.group(0)  # Return the full matched string
    return None  # If no version pattern is found

# MODIFICATION: Helper function to sort versions naturally
def natural_sort_key(s):
    """
    Returns a key for natural sorting (e.g., M800, M801, M810 instead of M800, M810, M801).
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


# ---------- Layout Detection ----------
def detect_layout(df: pd.DataFrame) -> str:
    """
    Detects if the Excel file layout is 'grouped' (repeated column headers) or 'wide' (wide format).
    - 'grouped': A row contains two or more 'Usecase' keywords, indicating horizontal grouping.
    - 'wide': Only one 'Usecase' column, other columns contain version-specific data.

    Args:
        df (pd.DataFrame): The raw DataFrame loaded from Excel (without header).

    Returns:
        str: 'grouped' or 'wide'.
    """
    usecase_hits: List[Tuple[int, int]] = []  # (row_index, count_of_usecase_in_row)

    # Search for 'Usecase' keyword in the first few rows (e.g., up to 10 rows)
    # to efficiently detect the header structure.
    for r in range(min(df.shape[0], 10)):
        # Convert row values to string and lower for case-insensitive matching
        row_values_str_lower = [_clean_str(val).lower() for val in df.iloc[r].tolist()]
        count = sum(1 for val in row_values_str_lower if val == USECASE_KEYWORD)
        if count > 0:
            usecase_hits.append((r, count))

    if not usecase_hits:
        logger.warning(
            f"'{USECASE_KEYWORD.capitalize()}' keyword not found in header rows. "
            "Assuming 'wide' layout as a fallback."
        )
        return "wide"

    # If any row has multiple 'Usecase' occurrences, it's a grouped layout.
    max_count_in_a_row = max(hit[1] for hit in usecase_hits)
    return "grouped" if max_count_in_a_row >= 2 else "wide"

# ---------- Parser: grouped layout ----------
def parse_grouped(df: pd.DataFrame, usecase: str, metrics: List[str]) -> pd.DataFrame:
    """
    Parses data from a 'grouped' layout where columns are repeated per-version blocks.

    Args:
        df (pd.DataFrame): The raw DataFrame loaded from Excel.
        usecase (str): The target use case to filter by (e.g., 'AU45A').
        metrics (List[str]): The target metric column names (e.g., ['Adjusted', 'CX']).

    Returns:
        pd.DataFrame: A DataFrame with 'Version' and specified metric columns.
    """
    header_idx: Optional[int] = None
    # Find the row containing multiple 'Usecase' headers
    for r in range(df.shape[0]):
        row_values_lower = [_clean_str(v).lower() for v in df.iloc[r].tolist()]
        if row_values_lower.count(USECASE_KEYWORD) >= 2:
            header_idx = r
            break

    if header_idx is None:
        raise RuntimeError(f"Grouped layout parsing failed: Multiple '{USECASE_KEYWORD.capitalize()}' headers not detected.")

    header_row_values = df.iloc[header_idx].tolist()
    # Find all starting column indices for 'Usecase'
    starts = [i for i, v in enumerate(header_row_values) if _clean_str(v).lower() == USECASE_KEYWORD]

    processed_versions_order_grouped: List[str] = []
    # MODIFICATION: Use a list of dictionaries to collect data per version
    # This helps in merging different metrics for the same version
    collected_data_per_version: Dict[str, Dict[str, float]] = {}
    target_usecase_lower = usecase.lower()
    # target_metric_lower = metric.lower()
    target_metrics_lower = [m.lower() for m in metrics]

    # Iterate through each version block
    for i, s_col_idx in enumerate(starts):
        e_col_idx = starts[i + 1] if i + 1 < len(starts) else len(header_row_values)

        # Look upwards from the 'Usecase' column to find the version title
        version_raw_title = ""
        for r_search in range(header_idx - 1, -1, -1):
            cell_val = _clean_str(df.iloc[r_search, s_col_idx])
            if cell_val:  # Found a non-empty cell as version title
                version_raw_title = cell_val
                break

        # Use the raw title as the version label.
        version_label = version_raw_title if version_raw_title else f"Unknown Version {i+1}"
        if version_label not in processed_versions_order_grouped:
            processed_versions_order_grouped.append(version_label)

        # Extract data for the current block
        current_block_data = df.iloc[header_idx + 1:].iloc[:, s_col_idx:e_col_idx].copy()
        current_block_data.columns = [_clean_str(col) for col in header_row_values[s_col_idx:e_col_idx]]

        # Find the actual case-insensitive column names for Usecase and Metrics
        actual_usecase_col_name = next(
            (c for c in current_block_data.columns if c.lower() == USECASE_KEYWORD), None
        )
        
        actual_metric_col_names = {}
        for tm in target_metrics_lower:
            actual_metric_col_name = next(
                (c for c in current_block_data.columns if c.lower() == tm), None
            )
            if actual_metric_col_name:
                actual_metric_col_names[tm] = actual_metric_col_name

        if not actual_usecase_col_name or not actual_metric_col_names:
            missing_metrics = [m for m in target_metrics_lower if m not in actual_metric_col_names]
            if missing_metrics:
                logger.debug( # Changed to debug to avoid excessive warnings if many metrics are not found in every block
                    f"Skipping block for version '{version_label}': "
                    f"'{USECASE_KEYWORD.capitalize()}' or requested metrics {metrics} "
                    f"({', '.join(missing_metrics)}) column(s) missing."
                )
            continue

        # Drop rows where all values are NaN in the relevant columns of the current block
        cols_to_check_nan = [actual_usecase_col_name] + list(actual_metric_col_names.values())
        current_block_data = current_block_data.dropna(how="all", subset=cols_to_check_nan)

        # Filter for the target use case
        current_block_data = current_block_data[
            current_block_data[actual_usecase_col_name].astype(str).str.strip().str.lower() == target_usecase_lower
        ]

        if current_block_data.empty:
            continue
        
        # MODIFICATION: Process each metric for the current version
        for metric_original_case in metrics:
            metric_lower = metric_original_case.lower()
            if metric_lower in actual_metric_col_names:
                actual_col = actual_metric_col_names[metric_lower]
                # Convert to numeric, handle errors, and get the mean
                val = pd.to_numeric(current_block_data[actual_col], errors="coerce").mean()
                if pd.notna(val):
                    if version_label not in collected_data_per_version:
                        collected_data_per_version[version_label] = {}
                    collected_data_per_version[version_label][metric_original_case] = val

    if not collected_data_per_version:
        logger.warning(f"No data found for '{usecase}' and metrics {metrics} in grouped layout.")
        return pd.DataFrame(columns=["Version"] + metrics) # Return with all metric columns

    # MODIFICATION: Convert collected data to DataFrame
    tidy_agg = pd.DataFrame.from_dict(collected_data_per_version, orient='index')
    tidy_agg.index.name = "Version"
    tidy_agg = tidy_agg.reset_index()

    # Reorder the DataFrame according to the detection order from Excel
    filtered_order = [v for v in processed_versions_order_grouped if v in tidy_agg["Version"].values]
    tidy_agg["Version"] = pd.Categorical(tidy_agg["Version"], categories=filtered_order, ordered=True)
    tidy_agg = tidy_agg.sort_values("Version").reset_index(drop=True)

    return tidy_agg

# ---------- Parser: wide layout ----------
def _build_compound_header(df: pd.DataFrame, header_rows: List[int]) -> List[str]:
    """
    Combines multiple header rows into a single list of compound column names.
    e.g., ["M800", "Adjusted"] -> "M800|Adjusted"
    """
    compound_headers: List[str] = []
    for c in range(df.shape[1]):
        parts: List[str] = []
        for r in header_rows:
            cell_val = _clean_str(df.iloc[r, c])
            if cell_val:
                parts.append(cell_val)
        compound_headers.append("|".join(parts) if parts else "")
    return compound_headers

# MODIFICATION: metric parameter is now a list of strings
def parse_wide(df: pd.DataFrame, usecase: str, metrics: List[str]) -> pd.DataFrame:
    """
    Parses data from a 'wide' layout where versions are separate columns.

    Args:
        df (pd.DataFrame): The raw DataFrame loaded from Excel.
        usecase (str): The target use case to filter by.
        metrics (List[str]): The target metric column names.

    Returns:
        pd.DataFrame: A DataFrame with 'Version' and specified metric columns.
    """
    usecase_header_row_idx: Optional[int] = None
    # Search for the header row containing 'Usecase'
    for r in range(min(df.shape[0], 6)):  # Search within the first 6 rows
        if any(_clean_str(v).lower() == USECASE_KEYWORD for v in df.iloc[r].tolist()):
            usecase_header_row_idx = r
            break

    if usecase_header_row_idx is None:
        raise RuntimeError(
            f"Wide layout parsing failed: '{USECASE_KEYWORD.capitalize()}' header row not found within initial rows."
        )

    # Build compound header names by merging up to 3 rows (current + 2 above)
    header_rows_indices = list(range(max(0, usecase_header_row_idx - 2), usecase_header_row_idx + 1))
    cols_compound_names = _build_compound_header(df, header_rows_indices)
    print(cols_compound_names)

    # Create a DataFrame from the data part with the new compound headers
    data_start_row_idx = usecase_header_row_idx + 1
    data_df = df.iloc[data_start_row_idx:].copy()
    data_df.columns = cols_compound_names

    # Identify the actual 'Usecase' column (case-insensitive)
    usecase_col_candidates = [
        c for c in data_df.columns
        if _clean_str(c).lower().endswith(USECASE_KEYWORD) or _clean_str(c).lower() == USECASE_KEYWORD
    ]

    if not usecase_col_candidates:
        raise RuntimeError(f"Wide layout parsing failed: '{USECASE_KEYWORD.capitalize()}' column not found in processed data frame.")

    actual_usecase_col_name = usecase_col_candidates[0]
    data_df.rename(columns={actual_usecase_col_name: USECASE_KEYWORD.capitalize()}, inplace=True)

    # Filter for the target use case
    data_df = data_df[
        data_df[USECASE_KEYWORD.capitalize()].astype(str).str.strip().str.lower() == usecase.lower()
    ]

    if data_df.empty:
        logger.warning(f"No data found for '{usecase}' in wide layout.")
        return pd.DataFrame(columns=["Version"] + metrics)

    processed_versions_order_wide: List[str] = []
    # MODIFICATION: Use a dictionary to collect data per version
    collected_data_per_version: Dict[str, Dict[str, float]] = {}
    target_metrics_lower = [m.lower() for m in metrics]

    # Iterate through columns to find version-metric data
    for col_name in data_df.columns:
        if _clean_str(col_name).lower() == USECASE_KEYWORD:
            continue

        extracted_version = _extract_version_from_wide_col_name(col_name)
        if not extracted_version:
            continue # Not a version column

        if extracted_version not in processed_versions_order_wide:
            processed_versions_order_wide.append(extracted_version)

        # MODIFICATION: Check for all target metrics within the column name
        found_metric_for_col = None
        for metric_original_case in metrics:
            metric_lower = metric_original_case.lower()
            # Special handling for "Adjusted" and "Adj" if 'Adjusted' is a target metric
            is_adjusted_alias = (metric_original_case == DEFAULT_METRIC and "adj" in col_name.lower() and metric_lower not in col_name.lower())
            
            if metric_lower in col_name.lower() or is_adjusted_alias:
                # Prioritize exact match or user-specified metric
                # If 'Adjusted' and 'Adj' are both metrics, this might pick 'Adjusted' if "Adjusted" is in the string,
                # but it ensures at least one of the requested metrics is found.
                found_metric_for_col = metric_original_case
                break
        
        if found_metric_for_col:
            # Convert values to numeric and take the mean for this version and metric
            val = pd.to_numeric(data_df[col_name], errors="coerce").mean()
            if pd.notna(val):
                if extracted_version not in collected_data_per_version:
                    collected_data_per_version[extracted_version] = {}
                collected_data_per_version[extracted_version][found_metric_for_col] = val

    if not collected_data_per_version:
        logger.warning(f"No valid metric columns with version information found for metrics {metrics} in wide layout.")
        return pd.DataFrame(columns=["Version"] + metrics)

    # MODIFICATION: Convert collected data to DataFrame
    tidy_agg = pd.DataFrame.from_dict(collected_data_per_version, orient='index')
    tidy_agg.index.name = "Version"
    tidy_agg = tidy_agg.reset_index()

    # Reorder the DataFrame according to the detection order from Excel
    # Filter only versions that actually have data
    filtered_order = [v for v in processed_versions_order_wide if v in tidy_agg["Version"].values]
    tidy_agg["Version"] = pd.Categorical(tidy_agg["Version"], categories=filtered_order, ordered=True)
    tidy_agg = tidy_agg.sort_values("Version").reset_index(drop=True)

    return tidy_agg

# ---------- Plotting Functions ----------
# MODIFICATION: metric_names is now a list
def plot_trend(au_data: pd.DataFrame, out_png: Path, title: str, metric_names: List[str]):
    """
    Generates and saves a trend plot from the processed data.

    Args:
        au_data (pd.DataFrame): DataFrame containing 'Version' and the specified metric columns.
        out_png (Path): Path to save the output PNG image.
        title (str): Title for the plot.
        metric_names (List[str]): The actual names of the metric columns in au_data to plot.
    """
    if au_data.empty:
        logger.warning("No data found to plot. Skipping plot generation.")
        return

    au_data["Version"] = au_data["Version"].astype(str)

    # Dynamic plot width to accommodate more data points/longer labels
    fig_width = max(PLOT_HEIGHT, len(au_data) * PLOT_WIDTH_PER_VERSION)
    plt.figure(figsize=(fig_width, PLOT_HEIGHT))

    # MODIFICATION: Iterate over metric_names to plot multiple lines
    for i, metric_name in enumerate(metric_names):
        if metric_name in au_data.columns:
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)] # Cycle through predefined colors
            plt.plot(au_data["Version"], au_data[metric_name], marker="o", 
                     color=color, linewidth=2, label=metric_name)

            # Display values on each data point for the current metric
            for x, y in zip(au_data["Version"], au_data[metric_name]):
                # Only display if not NaN, and avoid overlapping too much if many metrics
                if pd.notna(y):
                    # Offset text slightly for different metrics if needed, e.g., by color or position
                    # For simplicity, we just put it on the point. For many lines, this might clutter.
                    plt.text(x, y, f"{y:.2f}", ha="center", va="bottom", 
                             fontsize=8, color=color, alpha=0.8)
        else:
            logger.warning(f"Metric '{metric_name}' not found in the processed data. Skipping plot for this metric.")

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)

    plt.title(title, fontsize=14)
    plt.xlabel("Version", fontsize=12)
    # MODIFICATION: If multiple metrics, Y-axis label is general
    plt.ylabel("Value", fontsize=12) 
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Metric") # MODIFICATION: Changed legend title from 'Rail' to 'Metric'
    plt.tight_layout()

    try:
        plt.savefig(out_png, dpi=PLOT_DPI, bbox_inches='tight')
        logger.info(f"Plot saved to: {out_png}")
    except Exception as e:
        logger.error(f"Error saving plot to {out_png}: {e}")
    finally:
        plt.close() # Always close the plot to free memory

# MODIFICATION: New function for static aggregate plotting (fallback)
def plot_aggregate_trend_static(
    all_data: pd.DataFrame, 
    out_png: Path, 
    title: str, 
    metric_names: List[str], 
    usecase_names: List[str]
):
    """
    Generates and saves a comprehensive STATIC trend plot for multiple use cases and metrics.
    Uses seaborn for better visualization of grouped data.

    Args:
        all_data (pd.DataFrame): Combined DataFrame containing 'Version', 'Usecase', and metric columns.
        out_png (Path): Path to save the output PNG image.
        title (str): Title for the plot.
        metric_names (List[str]): The actual names of the metric columns.
        usecase_names (List[str]): The list of use case names included.
    """
    if all_data.empty:
        logger.warning("No aggregate data found to plot. Skipping aggregate plot generation.")
        return

    # Ensure 'Version' is string for plotting
    all_data["Version"] = all_data["Version"].astype(str)

    # all_data is already in long format (Version, Usecase, Metric, Value)
    # So we just need to ensure NaNs are handled before plotting.
    df_plot = all_data.dropna(subset=["Value"]).copy() # Renamed to df_plot for consistency

    if df_plot.empty:
        logger.warning("No valid data points after dropping NaNs for aggregate plot. Skipping.")
        return
    
    # Set a fixed height for each subplot and an aspect ratio
    # Use col_wrap to control how many subplots appear per row
    subplot_height = 3.5 # Height of each individual subplot in inches
    subplot_aspect = 1.8 # Aspect ratio (width/height) of each individual subplot
    num_cols_wrap = 4   # Number of subplots per row

    # Create the relational plot
    g = sns.relplot(
        data=df_plot,
        x="Version",
        y="Value",
        hue="Metric",
        col="Usecase", # Separate plots for each Usecase
        col_wrap=num_cols_wrap, # MODIFICATION: Wrap columns to create a grid
        kind="line", # Line plot
        marker="o", # Add markers
        height=subplot_height, # MODIFICATION: Fixed height for each subplot
        aspect=subplot_aspect, # MODIFICATION: Fixed aspect ratio for each subplot
        facet_kws={'sharey': False, 'sharex': True}, # Share x-axis (versions) but allow different y-scales
        palette=COLOR_CYCLE # Use the predefined color cycle
    )

    # Add titles and labels for each subplot
    g.set_axis_labels("Version", "Value")
    g.set_titles(col_template="{col_name}") # Set subplot title to Usecase name

    # Rotate x-axis labels for better readability and add grid
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle="--", alpha=0.6)
        
        # Add value labels on points within each subplot
        for line in ax.lines:
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            color = line.get_color()
            
            for x, y in zip(xdata, ydata):
                if pd.notna(y):
                    ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=7, color=color, alpha=0.8)

    g.fig.suptitle(title, y=1.02, fontsize=16) # Overall title for the figure
    g.fig.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make space for suptitle

    try:
        plt.savefig(out_png, dpi=PLOT_DPI, bbox_inches='tight')
        logger.info(f"Static aggregate plot saved to: {out_png}")
    except Exception as e:
        logger.error(f"Error saving static aggregate plot to {out_png}: {e}")
    finally:
        plt.close() # Always close the plot to free memory

# MODIFICATION: New function for common axis formatting for Plotly
def format_plotly_xaxis_tick(axis):
    # Only apply if the axis values are numeric-like, otherwise, leave as is.
    try:
        # Convert tickvals to string first for robust numeric conversion
        if all(pd.to_numeric([str(c) for c in axis.tickvals], errors='coerce').notna().all()):
            axis.tickformat = ".0f"  # Format as integer
    except:
        pass # Do nothing if conversion fails

# MODIFICATION: New function for interactive HTML aggregate plotting
def plot_aggregate_trend_interactive(
    all_data: pd.DataFrame, 
    out_html: Path, 
    title: str, 
    metric_names: List[str], 
    usecase_names: List[str],
    analyze_fluctuation_flag: bool # NEW: Pass the analyze_fluctuation flag
):
    """
    Generates and saves a comprehensive INTERACTIVE HTML trend plot for multiple use cases and metrics.
    Uses plotly.express for interactive visualization.

    Args:
        all_data (pd.DataFrame): Combined DataFrame containing 'Version', 'Usecase', 'Metric', and 'Value' columns.
        out_html (Path): Path to save the output HTML file.
        title (str): Title for the plot.
        metric_names (List[str]): The actual names of the metric columns.
        usecase_names (List[str]): The list of use case names included.
        analyze_fluctuation_flag (bool): Whether to perform and display fluctuation analysis.
    """
    if not PIO_AVAILABLE:
        logger.error("Plotly is not installed. Cannot generate interactive HTML plot.")
        return

    if all_data.empty:
        logger.warning("No aggregate data found to plot. Skipping interactive HTML plot generation.")
        return
    
    # Drop rows where 'Value' is NaN
    df_plot = all_data.dropna(subset=["Value"]).copy()

    if df_plot.empty:
        logger.warning("No valid data points after dropping NaNs for interactive plot. Skipping.")
        return
    
    # Ensure Version is sorted correctly in the plot
    all_unique_versions = sorted(df_plot['Version'].unique().tolist(), key=natural_sort_key)
    df_plot["Version"] = pd.Categorical(df_plot["Version"], categories=all_unique_versions, ordered=True)
    df_plot = df_plot.sort_values(by=["Usecase", "Version", "Metric"]).reset_index(drop=True)

    # Calculate number of columns for facet grid to ensure reasonable layout
    num_usecases = len(df_plot["Usecase"].unique())
    if num_usecases <= 3:
        num_cols_facet = num_usecases
    elif num_usecases <= 8:
        num_cols_facet = 3
    else:
        num_cols_facet = 4 # Or 5 for many subplots

    fig = px.line(
        df_plot,
        x="Version",
        y="Value",
        color="Metric",     # Lines colored by Metric
        line_group="Metric",# Group lines by metric within each facet
        facet_col="Usecase",# Separate columns for each Usecase
        facet_col_wrap=num_cols_facet, # Wrap columns for grid layout
        title=title,
        labels={"Value": "Value", "Version": "Version"}, # Customize axis labels
        hover_data={"Usecase": False, "Metric": True, "Value": ":.2f", "Version": True} # Show data on hover
    )

    fig.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>", font_size=12))

    fig.update_layout(
        autosize=True,       
        hovermode="x unified", # Shows all traces at a specific x-point
        font=dict(size=10),
        title_x=0.5, # Center the main title
        margin=dict(t=50, b=50, l=40, r=40), # Smaller margins
        legend_title_text="Metric",
    )

    fig.update_xaxes(
        tickangle=45,
        categoryorder="array",
        categoryarray=all_unique_versions)
    # 增加纵坐标刻度
    fig.update_yaxes(
        # 方式1: 尝试生成大约10个刻度。你可以根据需要调整这个数字。
        nticks=4, 
        # 方式2: 如果想更精确控制间隔，可以取消注释并设置dtick。
        # 例如，如果你的值通常是整数，可以设置为1；如果是小数，可以设置为0.1等。
        # dtick=0.5, 
        # 方式3: 如果想手动指定刻度，提供一个刻度值的列表
        # tickvals=[0, 0.25, 0.5, 0.75, 1.0],
        
        # 也可以调整刻度标签的格式
        tickformat=".2f", # 格式化为两位小数，根据你的数据类型调整
        # 调整刻度标签字体大小，防止重叠
        tickfont=dict(size=9)
    )


    # NEW: Generate fluctuation report HTML
    fluctuation_report_html = ""
    if analyze_fluctuation_flag:
        fluctuation_report_html = _generate_fluctuation_report_html(df_plot, metric_names)

    try:
        # Get the HTML for the plot only, without full HTML boilerplate
        plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

        # Combine report HTML and plot HTML into a single full HTML page
        final_html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 30px; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary-paragraph {{ line-height: 1.6; }}
                .plot-container {{ width: 100%; height: auto; min-height: 600px; /* Ensure enough height for the plot */ }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            {fluctuation_report_html}
            <h2>Trend Plot</h2>
            <div class="plot-container">
                {plot_html}
            </div>
        </body>
        </html>
        """
        
        # Write the final combined HTML to file
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(final_html_content)

        logger.info(f"Interactive HTML plot with fluctuation summary saved to: {out_html}")
    except Exception as e:
        logger.error(f"Error saving interactive HTML plot to {out_html}: {e}")


# MODIFICATION: Add a Plotly version of plot_trend for single use case
def plot_trend_interactive(au_data: pd.DataFrame, out_html: Path, title: str, metric_names: List[str]):
    """
    Generates and saves an interactive HTML trend plot from the processed data for a single use case.
    """
    if not PIO_AVAILABLE:
        logger.error("Plotly is not installed. Cannot generate interactive HTML plot for single case.")
        return

    if au_data.empty:
        logger.warning("No data found to plot for single case. Skipping interactive HTML plot generation.")
        return

    # Melt the DataFrame to long format for Plotly Express
    df_plot_long = au_data.melt(
        id_vars=["Version"],
        value_vars=metric_names,
        var_name="Metric",
        value_name="Value"
    )
    df_plot_long = df_plot_long.dropna(subset=["Value"])

    if df_plot_long.empty:
        logger.warning("No valid data points after melting and dropping NaNs for single case interactive plot. Skipping.")
        return

    # Ensure Version is sorted correctly
    all_unique_versions = sorted(df_plot_long['Version'].unique().tolist(), key=natural_sort_key)
    df_plot_long["Version"] = pd.Categorical(df_plot_long["Version"], categories=all_unique_versions, ordered=True)
    df_plot_long = df_plot_long.sort_values(by=["Version", "Metric"]).reset_index(drop=True)

    fig = px.line(
        df_plot_long,
        x="Version",
        y="Value",
        color="Metric",
        line_group="Metric",
        title=title,
        labels={"Value": "Value", "Version": "Version"},
        hover_data={"Metric": True, "Value": ":.2f", "Version": True}
    )

    fig.update_layout(
        autosize=True,
        hovermode="x unified",
        font=dict(size=10),
        title_x=0.5,
        margin=dict(t=80, b=50, l=40, r=40),
        legend_title_text="Metric"
    )

    fig.update_xaxes(tickangle=45)
    
    # Use the same named function for axis formatting
    fig.for_each_xaxis(format_plotly_xaxis_tick)

    try:
        pio.write_html(fig, file=str(out_html), auto_open=False, full_html=True)
        logger.info(f"Interactive HTML plot saved to: {out_html}")
    except Exception as e:
        logger.error(f"Error saving interactive HTML plot to {out_html}: {e}")

# MODIFICATION: Renamed and modified function to return DataFrame
def _calculate_fluctuations(df_long: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """
    Calculates fluctuation (mean, std dev, CV) for a specific metric across use cases.
    Returns a DataFrame of fluctuation data.
    """
    df_filtered = df_long[df_long["Metric"] == metric_name].copy()

    if df_filtered.empty:
        return pd.DataFrame(columns=["Usecase", "Mean", "Standard_Deviation", "Coefficient_of_Variation_CV"])

    fluctuation_data = df_filtered.groupby("Usecase")["Value"].agg(
        Mean="mean",
        Standard_Deviation="std"
    ).reset_index()

    # Calculate Coefficient of Variation (CV)
    fluctuation_data["Coefficient_of_Variation_CV"] = fluctuation_data.apply(
        lambda row: row["Standard_Deviation"] / row["Mean"] if row["Mean"] != 0 and pd.notna(row["Mean"]) else float('nan'),
        axis=1
    )
    
    return fluctuation_data.sort_values(by="Coefficient_of_Variation_CV", ascending=False) # Sort for ranking


def _generate_fluctuation_report_html(combined_df_long: pd.DataFrame, metrics_to_analyze: List[str]) -> str:
    """
    Generates an HTML string containing fluctuation statistical summaries and textual conclusions (English version).
    "Metric" is changed to "Rail". Chart placeholders are placed after the conclusion.
    Detailed fluctuation data tables are hidden by default and can be manually expanded.
    Corrected for _calculate_fluctuations returning data sorted in descending order.
    """
    report_parts = []
    report_parts.append("<!DOCTYPE html>\n<html>\n<head>\n<title>Volatility Analysis Report</title>")
    report_parts.append("""
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; margin: 20px; background-color: #f4f4f4; }
        h1, h2, h3 { color: #0056b3; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 1em; background-color: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .summary-paragraph { background-color: #e6f7ff; border-left: 5px solid #8ccbff; padding: 10px; margin-bottom: 1em; }
        .chart-placeholder { background-color: #ffe6e6; border-left: 5px solid #ff8c8c; padding: 10px; margin-bottom: 1em; }
        .collapsible-content {
            display: none; /* Hidden by default */
            padding: 0 18px;
            overflow: hidden;
            background-color: #f1f1f1;
            border: 1px solid #ddd;
            border-top: none;
            margin-bottom: 1em;
            box-shadow: 0 0 5px rgba(0,0,0,0.05);
        }
        .collapsible-button {
            background-color: #007bff; /* Blue button */
            color: white;
            padding: 10px 15px;
            border: none;
            cursor: pointer;
            width: auto; /* Adjust width based on content */
            text-align: left;
            outline: none;
            font-size: 16px;
            transition: 0.3s;
            border-radius: 5px;
            margin-top: 1em;
            margin-bottom: 1em;
        }
        .collapsible-button:hover {
            background-color: #0056b3; /* Darker on hover */
        }
    </style>
    </head>
    <body>
    """)
    report_parts.append("<h1>Volatility Analysis Report</h1>") # Main title

    overall_cv_data: Dict[str, Dict[str, float]] = {}  # Stores average coefficient of variation for each Rail
    detailed_fluctuations_html = []  # Stores detailed tables for each Rail

    # Stores fluct_df calculated for each Rail to reuse in the third phase, avoiding recalculation
    rail_fluctuation_dfs: Dict[str, pd.DataFrame] = {}

    # Phase 1: Calculate overall CV data and collect HTML for detailed tables (not yet appended to report_parts)
    for metric in metrics_to_analyze: # 'metric' now represents the Rail name
        fluct_df = _calculate_fluctuations(combined_df_long, metric)
        rail_fluctuation_dfs[metric] = fluct_df # Store for later use

        if fluct_df.empty:
            detailed_fluctuations_html.append(f"<p>No volatility data available for Rail <b>{metric}</b>.</p>")
            continue
        
        # Calculate overall statistics for the Coefficient of Variation for this Rail
        mean_cv = fluct_df["Coefficient_of_Variation_CV"].mean()
        median_cv = fluct_df["Coefficient_of_Variation_CV"].median()
        std_cv = fluct_df["Coefficient_of_Variation_CV"].std()
        
        overall_cv_data[metric] = {
            "Mean_CV": mean_cv,
            "Median_CV": median_cv,
            "Std_Dev_of_CV": std_cv
        }

        # Generate the HTML part for the detailed table, store it in detailed_fluctuations_html list
        detailed_fluctuations_html.append(f"<h3>Rail: {metric}</h3>")
        detailed_fluctuations_html.append("<p><i>Ranked by Coefficient of Variation (CV) across all use cases. Lower CV indicates higher stability. (Table is sorted by CV in descending order, higher volatility first)</i></p>")
        
        table_html = "<table><thead><tr><th>Usecase</th><th>Mean</th><th>Standard Deviation</th><th>Coefficient of Variation (CV)</th></tr></thead><tbody>"
        for _, row in fluct_df.round(4).iterrows():
            table_html += f"<tr><td>{row['Usecase']}</td><td>{row['Mean']}</td><td>{row['Standard_Deviation']}</td><td>{row['Coefficient_of_Variation_CV']}</td></tr>"
        table_html += "</tbody></table>"
        detailed_fluctuations_html.append(table_html)

    # Phase 2: Generate overall summary
    if not overall_cv_data:
        report_parts.append("<p class='summary-paragraph'>No volatility data found for any Rail.</p>")
        report_parts.append("</body></html>") # Close HTML if no data
        return "".join(report_parts)

    report_parts.append("<h3>Overall Volatility Summary by Rail (Average Coefficient of Variation per Usecase)</h3>")
    summary_table_html = "<table><thead><tr><th>Rail</th><th>Average CV</th><th>Median CV</th><th>Std Dev of CVs</th></tr></thead><tbody>"
    
    ranked_metrics = sorted(overall_cv_data.items(), key=lambda item: item[1]["Mean_CV"]) # Sort by Average CV
    
    for metric, stats in ranked_metrics:
        summary_table_html += f"<tr><td>{metric}</td><td>{stats['Mean_CV']:.4f}</td><td>{stats['Median_CV']:.4f}</td><td>{stats['Std_Dev_of_CV']:.4f}</td></tr>"
    summary_table_html += "</tbody></table>"
    report_parts.append(summary_table_html)

    # Phase 3: Generate textual conclusions
    report_parts.append("<h3>Volatility Conclusion</h3>")
    conclusion_text = "<p class='summary-paragraph'>"

    if ranked_metrics:
        most_stable_metric = ranked_metrics[0][0]
        least_stable_metric = ranked_metrics[-1][0]
        
        conclusion_text += f"Among the analyzed Rails, <b>{most_stable_metric}</b> appears to be the most stable on average across all use cases (lowest average CV)."
        conclusion_text += f"Conversely, <b>{least_stable_metric}</b> exhibits the highest average volatility (highest average CV).<br><br>"
        
        # If DEFAULT_METRIC (e.g., 'Adjusted') is analyzed, provide specific conclusions
        if DEFAULT_METRIC in metrics_to_analyze:
            adjusted_fluct_df = rail_fluctuation_dfs.get(DEFAULT_METRIC) # Get from stored data
            if adjusted_fluct_df is not None and not adjusted_fluct_df.empty:
                # If _calculate_fluctuations returns DataFrame sorted by CV in descending order:
                # iloc[0] is the highest CV (most volatile)
                # iloc[-1] is the lowest CV (most stable)
                most_volatile_usecase_for_default = adjusted_fluct_df.iloc[0]['Usecase']
                most_stable_usecase_for_default = adjusted_fluct_df.iloc[-1]['Usecase']

                conclusion_text += f"For the <b>'{DEFAULT_METRIC}'</b> Rail, <b>{most_volatile_usecase_for_default}</b> shows the highest volatility, while <b>{most_stable_usecase_for_default}</b> demonstrates the highest stability across versions.<br><br>"
            else:
                conclusion_text += f"No specific volatility analysis available for the default Rail '{DEFAULT_METRIC}'.<br><br>"

        conclusion_text += "Users should investigate use cases and Rails with higher Coefficients of Variation (CV) to understand the causes of instability."

        # Conclusion for each Rail
        conclusion_text += "<br><br>Below is a summary of the most and least volatile use cases within each Rail:<ul>"
        for metric in metrics_to_analyze:
            # print(f"Checking for specific use case conclusions for Rail: '{metric}'") # Debug line
            fluct_df = rail_fluctuation_dfs.get(metric) # Get from stored data
            if fluct_df is not None and not fluct_df.empty:
                # _calculate_fluctuations returns DataFrame sorted by CV in descending order
                # iloc[0] is the highest CV (most volatile)
                # iloc[-1] is the lowest CV (most stable)
                most_volatile_usecase = fluct_df.iloc[0]['Usecase']
                least_volatile_usecase = fluct_df.iloc[-1]['Usecase'] # This is the most stable use case

                # Corrected wording:
                conclusion_text += f"<li>For Rail <b>'{metric}'</b>, use case <b>{most_volatile_usecase}</b> exhibits the highest volatility, whereas use case <b>{least_volatile_usecase}</b> shows the highest stability across versions.</li>"
            else:
                conclusion_text += f"<li>No specific use case volatility data available for Rail <b>'{metric}'</b>.</li>"
        conclusion_text += "</ul>"

    else:
        conclusion_text += "No volatility data was generated for any Rail."

    conclusion_text += "</p>"
    report_parts.append(conclusion_text)
    
    # # Phase 4: Add chart placeholders
    # report_parts.append("<h3>Volatility Visualization Charts</h3>")
    # report_parts.append("<p class='chart-placeholder'>Visual charts on volatility, such as CV distribution plots for each Rail or trend charts for key use cases, can be inserted here.</p>")

    # Phase 5: Add collapsible detailed data section
    report_parts.append("""
    <button type="button" class="collapsible-button" onclick="toggleDetails()">Show/Hide Detailed Volatility Data</button>
    <div class="collapsible-content" id="detailedDataContent">
    """)
    report_parts.append("<h3>Detailed Volatility Data per Rail</h3>")
    report_parts.extend(detailed_fluctuations_html)
    report_parts.append("</div>") # Close collapsible-content div

    # JavaScript to toggle the collapsible content
    report_parts.append("""
    <script>
    function toggleDetails() {
        var content = document.getElementById("detailedDataContent");
        if (content.style.display === "block" || content.style.display === "") {
            content.style.display = "none";
        } else {
            content.style.display = "block";
        }
    }
    </script>
    """)
    report_parts.append("</body>\n</html>")

    return "".join(report_parts)


# MODIFICATION: Renamed and modified function to return DataFrame
def analyze_fluctuations(df: pd.DataFrame, out_path: Path, metric_name: str):
    """
    Analyzes fluctuation (mean, std dev, CV) for a specific metric across use cases.
    Saves the results to CSV and prints top 10. This is the version called for external CSV.
    """
    logger.info(f"Analyzing fluctuation for metric '{metric_name}'...")

    fluctuation_data = _calculate_fluctuations(df, metric_name)

    if fluctuation_data.empty:
        logger.warning(f"No data for metric '{metric_name}' found for fluctuation analysis.")
        return

    try:
        fluctuation_data.round(4).to_csv(out_path, index=False)
        logger.info(f"Fluctuation analysis for '{metric_name}' saved to: {out_path}")
        logger.info("\n--- Fluctuation Analysis Results (Top 10 by CV) ---")
        logger.info(fluctuation_data.head(10).round(4).to_string(index=False)) # Display top 10
        logger.info("-----------------------------------------------------")

    except Exception as e:
        logger.error(f"Error saving fluctuation analysis to {out_path}: {e}")


# ---------- Main Execution ----------
def main():
    """
    Main function to parse arguments, read Excel, process data, and generate outputs.
    """
    ap = argparse.ArgumentParser(
        description="Analyze and plot trend for specific usecase(s)/metric(s) from Excel." # MODIFICATION: Updated description
    )
    ap.add_argument(
        "xlsx", 
        type=str, 
        help="Path to the Excel file."
    )
    ap.add_argument(
        "--sheet",
        type=str,
        default="0",
        help="Sheet index (0-based) or sheet name. Default is the first sheet (0)."
    )
    ap.add_argument(
        "--usecase",
        type=str,
        default=DEFAULT_USECASE,
        help=f"Target use case (e.g., 'AU45A'). Case-insensitive matching. "
             f"Special value 'China' will process a predefined list of China-related use cases and also generate an aggregate plot. " # MODIFICATION: Added explanation for 'China'
             f"Default: {DEFAULT_USECASE}."
    )
    ap.add_argument(
        "--metric",
        type=str,
        nargs="+", # MODIFICATION: Allow one or more arguments for --metric
        default=[DEFAULT_METRIC], # MODIFICATION: Default is now a list
        help=f"Target metric column name(s) (e.g., 'Adjusted', 'CX', 'Power'). Case-insensitive matching. "
             f"Provide multiple names separated by spaces. Default: {DEFAULT_METRIC}." # MODIFICATION: Updated help
    )
    # MODIFICATION: New argument for fluctuation analysis
    ap.add_argument(
        "--analyze-fluctuation",
        action="store_true", # This makes it a boolean flag
        help=f"If set, performs fluctuation analysis (mean, std dev, CV) for the '{DEFAULT_METRIC}' metric across versions for each use case when 'China' usecase is selected. In HTML, summary for all metrics will be shown."
    )
    # MODIFICATION: New argument for HTML output
    ap.add_argument(
        "--html-output",
        action="store_true",
        help="If set, generates an interactive HTML plot for the aggregate data (requires Plotly). "
             "Otherwise, a static PNG plot is generated using Seaborn."
    )
    # MODIFICATION: New argument to disable static fallback
    ap.add_argument(
        "--no-static-fallback",
        action="store_true",
        help="If --html-output is enabled and Plotly is unavailable, do NOT fallback to static PNG plot."
    )

    args = ap.parse_args()

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        logger.error(f"Error: Excel file not found at '{xlsx_path}'")
        sys.exit(1)

    sheet_identifier: Union[int, str] = int(args.sheet) if args.sheet.isdigit() else args.sheet

    is_china_special_case = False
    usecases_to_process: List[str]
    if args.usecase.lower() == "china":
        is_china_special_case = True
        usecases_to_process = CHINA_USECASES
        logger.info(f"Special 'China' usecase selected. Processing predefined use cases: {', '.join(CHINA_USECASES)}")
    else:
        usecases_to_process = [args.usecase]
        logger.info(f"Target Usecase: '{args.usecase}'")
    
    logger.info(f"Processing Excel file: '{xlsx_path}'")
    logger.info(f"Target sheet: '{sheet_identifier}'")
    logger.info(f"Target Metrics: '{', '.join(args.metric)}'") # MODIFICATION: Log all metrics

    try:
        # Read a limited number of rows without a header to detect layout later
        # Reading more rows than strictly necessary for header detection to ensure
        # enough context for multi-line headers, e.g., 50 rows.
        df_raw = pd.read_excel(xlsx_path, sheet_name=sheet_identifier, header=None, nrows=50)
    except Exception as e:
        logger.error(f"Error reading Excel file or sheet '{sheet_identifier}': {e}")
        sys.exit(1)

    layout = detect_layout(df_raw)
    logger.info(f"Detected layout: '{layout}'")

    # MODIFICATION: AGGREGATE PLOT - List to collect all processed dataframes
    all_processed_dfs: List[pd.DataFrame] = []

    # MODIFICATION: Loop through each use case to process
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Base output directory will now be determined once and reused for all outputs
    output_dir_prefix = "aggregate_output" if is_china_special_case else "output"
    output_dir_name = f"{output_dir_prefix}_{current_time_str}"
    output_base_dir = xlsx_path.parent / output_dir_name
    output_base_dir.mkdir(parents=True, exist_ok=True) 

    for current_usecase in usecases_to_process:
        logger.info(f"\n--- Processing data for use case: '{current_usecase}' ---")
        au_data_frame: pd.DataFrame
        try:
            if layout == "grouped":
                # Pass the raw_df for grouped layout as it needs to re-read from start
                au_data_frame = parse_grouped(
                    pd.read_excel(xlsx_path, sheet_name=sheet_identifier, header=None), # Re-read full data for grouped
                    usecase=current_usecase, 
                    metrics=args.metric
                )
            else:  # wide
                au_data_frame = parse_wide(df_raw, usecase=current_usecase, metrics=args.metric)
        except RuntimeError as e:
            logger.error(f"Data parsing error for '{current_usecase}': {e}")
            continue # Continue to the next use case if one fails
        except Exception as e:
            logger.error(f"An unexpected error occurred during parsing for '{current_usecase}': {e}", exc_info=True) # exc_info to print traceback
            continue # Continue to the next use case

        if not au_data_frame.empty:
            logger.info("\n--- Processed Versions and Metric values ---")
            # MODIFICATION: Display all metric columns found in the data frame
            display_cols = ["Version"] + [m for m in args.metric if m in au_data_frame.columns]
            if not display_cols: # Should not happen if au_data_frame is not empty and metrics were found
                logger.info("No relevant metric columns found for display.")
            else:
                logger.info(au_data_frame[display_cols].to_string(index=False))
            logger.info("---------------------------------------------")
            
            # MODIFICATION: AGGREGATE PLOT - Add a 'Usecase' column and append to list
            # Melt the data frame to long format before appending to all_processed_dfs
            # This is crucial for fluctuation analysis as well, as it standardizes the metric values
            # for all use cases into a single 'Value' column
            df_long_current_usecase = au_data_frame.melt(
                id_vars=["Version"],
                value_vars=[m for m in args.metric if m in au_data_frame.columns],
                var_name="Metric",
                value_name="Value"
            )
            df_long_current_usecase["Usecase"] = current_usecase
            all_processed_dfs.append(df_long_current_usecase)

        else:
            logger.info(f"No data processed for the specified Usecase ('{current_usecase}') and Metrics ('{', '.join(args.metric)}').")
            continue # Skip saving/plotting individual if no data

        # MODIFICATION: Output folder and file names now include the specific usecase
        metrics_str_for_filename = "_".join([m.lower().replace(" ", "") for m in args.metric])
        # MODIFICATION: Base name includes current_usecase
        output_base_name = f"{current_usecase}_{metrics_str_for_filename}_trend"
        out_csv_path = output_base_dir / f"{output_base_name}_data.csv" # Removed timestamp from individual filenames
        out_png_path = output_base_dir / f"{output_base_name}.png" # Removed timestamp from individual filenames
        out_html_path_individual = output_base_dir / f"{output_base_name}.html" # New path for individual HTML

        try:
            # For individual CSV, we want the wide format (Version | Metric1 | Metric2)
            # So, we convert df_long_current_usecase back to wide for individual save
            au_data_frame_for_save = df_long_current_usecase.pivot(
                index='Version', columns='Metric', values='Value'
            ).reset_index()
            au_data_frame_for_save.columns.name = None # Remove the 'Metric' name from columns index
            au_data_frame_for_save.to_csv(out_csv_path, index=False)
            logger.info(f"Individual data saved to: {out_csv_path}")
        except Exception as e:
            logger.error(f"Error saving individual data to CSV for '{current_usecase}': {e}")

        # MODIFICATION: Pass the list of metric names to plot_trend
        # Filter for metrics that actually have data in the dataframe
        metrics_to_plot = [m for m in args.metric if m in au_data_frame.columns]
        if metrics_to_plot:
            # MODIFICATION: Add logic for individual HTML/PNG plotting based on --html-output
            if args.html_output and PIO_AVAILABLE:
                # For individual plots, we currently don't include fluctuation analysis in HTML
                plot_trend_interactive(au_data_frame, out_html_path_individual,
                                       title=f"{current_usecase} Trend by Version ({', '.join(metrics_to_plot)})",
                                       metric_names=metrics_to_plot)
            elif not args.html_output or (args.html_output and not PIO_AVAILABLE and not args.no_static_fallback):
                # Fallback to static PNG if HTML not requested or Plotly not available (and fallback not disabled)
                if args.html_output and not PIO_AVAILABLE:
                    logger.warning(f"Plotly not available for individual plot '{current_usecase}', falling back to static PNG.")
                plot_trend(au_data_frame, out_png_path, 
                           title=f"{current_usecase} Trend by Version ({', '.join(metrics_to_plot)})",
                           metric_names=metrics_to_plot)
            else: # args.html_output and not PIO_AVAILABLE and args.no_static_fallback
                logger.warning(f"Plotly not available and static fallback disabled for individual plot '{current_usecase}'. No plot generated.")
        else:
            logger.warning(f"No specified metrics found in the processed data for '{current_usecase}' to plot individually.")

    # MODIFICATION: AGGREGATE PLOT - Process collected data after all use cases
    if is_china_special_case and all_processed_dfs:
        logger.info("\n--- Generating aggregate plot and data for all China use cases ---")
        # combined_df is now already in long format due to previous modification
        combined_df_long = pd.concat(all_processed_dfs, ignore_index=True)
        
        # Ensure 'Version' is a categorical type for consistent ordering across facets
        # Use the union of all unique versions found across all processed DFs, and apply natural sort
        all_unique_versions = sorted(combined_df_long['Version'].unique().tolist(), key=natural_sort_key)
        combined_df_long["Version"] = pd.Categorical(combined_df_long["Version"], categories=all_unique_versions, ordered=True)
        combined_df_long = combined_df_long.sort_values(by=["Usecase", "Version", "Metric"]).reset_index(drop=True)

        # Save aggregate data (still in long format, which is good for analysis and plotting)
        agg_csv_path = output_base_dir / f"aggregate_China_{metrics_str_for_filename}_data_long_format.csv" # Changed filename to indicate long format
        try:
            combined_df_long.to_csv(agg_csv_path, index=False)
            logger.info(f"Aggregate data (long format) saved to: {agg_csv_path}")
        except Exception as e:
            logger.error(f"Error saving aggregate data to CSV: {e}")

        # Determine plotting method (interactive vs. static)
        metrics_to_plot_agg = [m for m in args.metric if m in combined_df_long["Metric"].unique()] # Check from 'Metric' column
        if metrics_to_plot_agg:
            if args.html_output and PIO_AVAILABLE:
                agg_html_path = output_base_dir / f"aggregate_China_{metrics_str_for_filename}.html"
                plot_aggregate_trend_interactive(
                    combined_df_long, # Pass the long format dataframe directly
                    agg_html_path,
                    title=f"Trend for All China Use Cases ({', '.join(metrics_to_plot_agg)})",
                    metric_names=metrics_to_plot_agg,
                    usecase_names=usecases_to_process,
                    analyze_fluctuation_flag=args.analyze_fluctuation # Pass the flag here
                )
            elif not args.html_output or (args.html_output and not PIO_AVAILABLE and not args.no_static_fallback):
                # Fallback to static PNG if HTML not requested or Plotly not available (and fallback not disabled)
                if args.html_output and not PIO_AVAILABLE:
                    logger.warning("Plotly not available, falling back to static PNG plot.")
                agg_png_path = output_base_dir / f"aggregate_China_{metrics_str_for_filename}.png"
                plot_aggregate_trend_static(
                    combined_df_long, # Pass the long format dataframe directly
                    agg_png_path,
                    title=f"Trend for All China Use Cases ({', '.join(metrics_to_plot_agg)})",
                    metric_names=metrics_to_plot_agg,
                    usecase_names=usecases_to_process
                )
            else: # args.html_output and not PIO_AVAILABLE and args.no_static_fallback
                 logger.warning("Plotly not available and static fallback disabled. No aggregate plot generated.")
        else:
            logger.warning("No specified metrics found in the combined data to plot for the aggregate view.")
        
        # MODIFICATION: Perform fluctuation analysis if requested (for CSV output, still only for DEFAULT_METRIC)
        if args.analyze_fluctuation:
            if DEFAULT_METRIC in combined_df_long["Metric"].unique(): # Check if 'Adjusted' is among the collected metrics
                fluctuation_output_path = output_base_dir / f"aggregate_China_{DEFAULT_METRIC}_fluctuation.csv"
                # This call is only for the CSV output, the HTML report uses _generate_fluctuation_report_html instead.
                analyze_fluctuations(combined_df_long, fluctuation_output_path, DEFAULT_METRIC)
            else:
                logger.warning(f"Fluctuation analysis requested for '{DEFAULT_METRIC}', but this metric was not collected. "
                               f"Please ensure '--metric {DEFAULT_METRIC}' is included in command line arguments.")

    elif is_china_special_case:
        logger.warning("No data collected for any China use case. Skipping aggregate output and fluctuation analysis.")


if __name__ == "__main__":
    main()