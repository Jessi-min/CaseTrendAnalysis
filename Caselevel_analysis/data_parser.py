#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data parsing module for the case trend analysis tool.
Contains functions to parse data from Excel files in different layouts.
"""

import logging
from typing import List, Dict, Optional
import pandas as pd
from utils import clean_str, extract_version_from_wide_col_name, build_compound_header
from config import USECASE_KEYWORD

# Configure logging
logger = logging.getLogger(__name__)

def parse_grouped(df: pd.DataFrame, usecase: str, metrics: List[str]) -> pd.DataFrame:
    """
    Parses data from a 'grouped' layout where columns are repeated per-version blocks.

    Args:
        df: The raw DataFrame loaded from Excel
        usecase: The target use case to filter by
        metrics: The target metric column names

    Returns:
        DataFrame with 'Version' and specified metric columns
    """
    header_idx = None
    # Find the row containing multiple 'Usecase' headers
    for r in range(df.shape[0]):
        row_values_lower = [clean_str(v).lower() for v in df.iloc[r].tolist()]
        if row_values_lower.count(USECASE_KEYWORD) >= 2:
            header_idx = r
            break

    if header_idx is None:
        raise RuntimeError(f"Grouped layout parsing failed: Multiple '{USECASE_KEYWORD.capitalize()}' headers not detected.")

    header_row_values = df.iloc[header_idx].tolist()
    # Find all starting column indices for 'Usecase'
    starts = [i for i, v in enumerate(header_row_values) if clean_str(v).lower() == USECASE_KEYWORD]

    processed_versions_order = []
    collected_data_per_version: Dict[str, Dict[str, float]] = {}
    target_usecase_lower = usecase.lower()
    target_metrics_lower = [m.lower() for m in metrics]

    # Iterate through each version block
    for i, s_col_idx in enumerate(starts):
        e_col_idx = starts[i + 1] if i + 1 < len(starts) else len(header_row_values)

        # Look upwards from the 'Usecase' column to find the version title
        version_raw_title = ""
        for r_search in range(header_idx - 1, -1, -1):
            cell_val = clean_str(df.iloc[r_search, s_col_idx])
            if cell_val:  # Found a non-empty cell as version title
                version_raw_title = cell_val
                break

        # Use the raw title as the version label
        version_label = version_raw_title if version_raw_title else f"Unknown Version {i+1}"
        if version_label not in processed_versions_order:
            processed_versions_order.append(version_label)

        # Extract data for the current block
        current_block_data = df.iloc[header_idx + 1:].iloc[:, s_col_idx:e_col_idx].copy()
        current_block_data.columns = [clean_str(col) for col in header_row_values[s_col_idx:e_col_idx]]

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
                logger.debug(
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
        
        # Process each metric for the current version
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
        return pd.DataFrame(columns=["Version"] + metrics)

    # Convert collected data to DataFrame
    tidy_agg = pd.DataFrame.from_dict(collected_data_per_version, orient='index')
    tidy_agg.index.name = "Version"
    tidy_agg = tidy_agg.reset_index()

    # Reorder the DataFrame according to the detection order from Excel
    filtered_order = [v for v in processed_versions_order if v in tidy_agg["Version"].values]
    tidy_agg["Version"] = pd.Categorical(tidy_agg["Version"], categories=filtered_order, ordered=True)
    tidy_agg = tidy_agg.sort_values("Version").reset_index(drop=True)

    return tidy_agg

def parse_wide(df: pd.DataFrame, usecase: str, metrics: List[str]) -> pd.DataFrame:
    """
    Parses data from a 'wide' layout where versions are separate columns.

    Args:
        df: The raw DataFrame loaded from Excel
        usecase: The target use case to filter by
        metrics: The target metric column names

    Returns:
        DataFrame with 'Version' and specified metric columns
    """
    usecase_header_row_idx = None
    # Search for the header row containing 'Usecase'
    for r in range(min(df.shape[0], 6)):  # Search within the first 6 rows
        if any(clean_str(v).lower() == USECASE_KEYWORD for v in df.iloc[r].tolist()):
            usecase_header_row_idx = r
            break

    if usecase_header_row_idx is None:
        raise RuntimeError(
            f"Wide layout parsing failed: '{USECASE_KEYWORD.capitalize()}' header row not found within initial rows."
        )

    # Build compound header names by merging up to 3 rows (current + 2 above)
    header_rows_indices = list(range(max(0, usecase_header_row_idx - 2), usecase_header_row_idx + 1))
    cols_compound_names = build_compound_header(df, header_rows_indices)

    # Create a DataFrame from the data part with the new compound headers
    data_start_row_idx = usecase_header_row_idx + 1
    data_df = df.iloc[data_start_row_idx:].copy()
    data_df.columns = cols_compound_names

    # Identify the actual 'Usecase' column (case-insensitive)
    usecase_col_candidates = [
        c for c in data_df.columns
        if clean_str(c).lower().endswith(USECASE_KEYWORD) or clean_str(c).lower() == USECASE_KEYWORD
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

    processed_versions_order = []
    collected_data_per_version: Dict[str, Dict[str, float]] = {}
    target_metrics_lower = [m.lower() for m in metrics]

    # Iterate through columns to find version-metric data
    for col_name in data_df.columns:
        if clean_str(col_name).lower() == USECASE_KEYWORD:
            continue

        extracted_version = extract_version_from_wide_col_name(col_name)
        if not extracted_version:
            continue  # Not a version column

        if extracted_version not in processed_versions_order:
            processed_versions_order.append(extracted_version)

        # Check for all target metrics within the column name
        found_metric_for_col = None
        for metric_original_case in metrics:
            metric_lower = metric_original_case.lower()
            # Special handling for "Adjusted" and "Adj" if 'Adjusted' is a target metric
            is_adjusted_alias = (metric_original_case == "Adjusted" and 
                                "adj" in col_name.lower() and 
                                metric_lower not in col_name.lower())
            
            if metric_lower in col_name.lower() or is_adjusted_alias:
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

    # Convert collected data to DataFrame
    tidy_agg = pd.DataFrame.from_dict(collected_data_per_version, orient='index')
    tidy_agg.index.name = "Version"
    tidy_agg = tidy_agg.reset_index()

    # Reorder the DataFrame according to the detection order from Excel
    filtered_order = [v for v in processed_versions_order if v in tidy_agg["Version"].values]
    tidy_agg["Version"] = pd.Categorical(tidy_agg["Version"], categories=filtered_order, ordered=True)
    tidy_agg = tidy_agg.sort_values("Version").reset_index(drop=True)

    return tidy_agg