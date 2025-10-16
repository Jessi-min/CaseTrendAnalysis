#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for the case trend analysis tool.
"""

import re
import logging
from typing import Union, List, Optional
import pandas as pd
from config import VERSION_EXTRACT_PATTERN, USECASE_KEYWORD

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def clean_str(value: Union[str, float]) -> str:
    """
    Helper to clean and convert to string, handling NaNs.
    
    Args:
        value: The value to clean and convert
        
    Returns:
        Cleaned string value
    """
    return str(value).strip() if pd.notna(value) else ""

def extract_version_from_wide_col_name(column_name: str) -> Optional[str]:
    """
    Extracts a version label from a compound column name used in 'wide' layout.
    Examples: "M800 | Adjusted" -> "M800", "M850E (Final) Adjusted" -> "M850E"
    
    Args:
        column_name: The column name to extract version from
        
    Returns:
        Extracted version string or None if not found
    """
    column_name_upper = column_name.upper()
    match = VERSION_EXTRACT_PATTERN.search(column_name_upper)
    if match:
        return match.group(0)
    return None

def natural_sort_key(s: str) -> List[Union[int, str]]:
    """
    Returns a key for natural sorting (e.g., M800, M801, M810 instead of M800, M810, M801).
    
    Args:
        s: String to create sort key for
        
    Returns:
        List of components for natural sorting
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def detect_layout(df: pd.DataFrame) -> str:
    """
    Detects if the Excel file layout is 'grouped' (repeated column headers) or 'wide' (wide format).
    - 'grouped': A row contains two or more 'Usecase' keywords, indicating horizontal grouping.
    - 'wide': Only one 'Usecase' column, other columns contain version-specific data.

    Args:
        df: The raw DataFrame loaded from Excel (without header)

    Returns:
        'grouped' or 'wide'
    """
    usecase_hits = []

    # Search for 'Usecase' keyword in the first few rows
    for r in range(min(df.shape[0], 10)):
        row_values_str_lower = [clean_str(val).lower() for val in df.iloc[r].tolist()]
        count = sum(1 for val in row_values_str_lower if val == USECASE_KEYWORD)
        if count > 0:
            usecase_hits.append((r, count))

    if not usecase_hits:
        logger.warning(
            f"'{USECASE_KEYWORD.capitalize()}' keyword not found in header rows. "
            "Assuming 'wide' layout as a fallback."
        )
        return "wide"

    # If any row has multiple 'Usecase' occurrences, it's a grouped layout
    max_count_in_a_row = max(hit[1] for hit in usecase_hits)
    return "grouped" if max_count_in_a_row >= 2 else "wide"

def build_compound_header(df: pd.DataFrame, header_rows: List[int]) -> List[str]:
    """
    Combines multiple header rows into a single list of compound column names.
    e.g., ["M800", "Adjusted"] -> "M800|Adjusted"
    
    Args:
        df: DataFrame containing the header rows
        header_rows: List of row indices to combine
        
    Returns:
        List of compound header names
    """
    compound_headers = []
    for c in range(df.shape[1]):
        parts = []
        for r in header_rows:
            cell_val = clean_str(df.iloc[r, c])
            if cell_val:
                parts.append(cell_val)
        compound_headers.append("|".join(parts) if parts else "")
    return compound_headers