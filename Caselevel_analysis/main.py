#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: mizho
@copyright: Qualcomm (C) 2023-2024
@version: 2.0 (Refactored for better maintainability)

Main entry point for the case trend analysis tool.
Parses command line arguments and orchestrates the data processing workflow.

Usage:
  # Plot a single metric (backward compatible)
  python main.py "Caselevel_Trendency.xlsx" --sheet 0 --usecase AU45A --metric Adjusted

  # Plot multiple metrics
  python main.py "Caselevel_Trendency.xlsx" --sheet 0 --usecase AU45A --metric Adjusted CX Power

  # Plot multiple metrics & HTML for a single use case
  python main.py "Caselevel_Trendency.xlsx" --sheet 0 --usecase AU45A --metric Adjusted CX Power --html-output

  # Plot for a predefined list of China-related use cases
  # Will also generate an aggregated plot/data for all China use cases
  # AND perform fluctuation analysis on 'Adjusted' metric if requested.
  # Interactive HTML plot for aggregate data including fluctuation summary.
  python main.py "Caselevel_Trendency.xlsx" --sheet 0 --usecase China --metric Adjusted CX GFX "Total CPU" "Total Memory" --html-output --analyze-fluctuation
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

from config import DEFAULT_METRIC, DEFAULT_USECASE, CHINA_USECASES
from data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    ap = argparse.ArgumentParser(
        description="Analyze and plot trend for specific usecase(s)/metric(s) from Excel."
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
             f"Special value 'China' will process a predefined list of China-related use cases and also generate an aggregate plot. "
             f"Default: {DEFAULT_USECASE}."
    )
    ap.add_argument(
        "--metric",
        type=str,
        nargs="+",
        default=[DEFAULT_METRIC],
        help=f"Target metric column name(s) (e.g., 'Adjusted', 'CX', 'Power'). Case-insensitive matching. "
             f"Provide multiple names separated by spaces. Default: {DEFAULT_METRIC}."
    )
    ap.add_argument(
        "--analyze-fluctuation",
        action="store_true",
        help=f"If set, performs fluctuation analysis (mean, standard deviation, variance) for the '{DEFAULT_METRIC}' metric across versions for each use case when 'China' usecase is selected. In HTML, summary for all metrics will be shown."
    )
    ap.add_argument(
        "--html-output",
        action="store_true",
        help="If set, generates an interactive HTML plot for the aggregate data (requires Plotly). "
             "Otherwise, a static PNG plot is generated using Seaborn."
    )
    ap.add_argument(
        "--no-static-fallback",
        action="store_true",
        help="If --html-output is enabled and Plotly is unavailable, do NOT fallback to static PNG plot."
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for more detailed output."
    )
    
    return ap.parse_args()

def main():
    """Main function to process data and generate outputs"""
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Validate input file
    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        logger.error(f"Error: Excel file not found at '{xlsx_path}'")
        sys.exit(1)
        
    # Process sheet identifier
    sheet_identifier = int(args.sheet) if args.sheet.isdigit() else args.sheet
    
    # Determine use cases to process
    is_china_special_case = args.usecase.lower() == "china"
    usecases_to_process = CHINA_USECASES if is_china_special_case else [args.usecase]
    
    # Log processing information
    logger.info(f"Processing Excel file: '{xlsx_path}'")
    logger.info(f"Target sheet: '{sheet_identifier}'")
    logger.info(f"Target Metrics: '{', '.join(args.metric)}'")
    
    if is_china_special_case:
        logger.info(f"Special 'China' usecase selected. Processing predefined use cases: {', '.join(CHINA_USECASES)}")
    else:
        logger.info(f"Target Usecase: '{args.usecase}'")
    
    # Create output directory
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_prefix = "aggregate_output" if is_china_special_case else "output"
    output_dir_name = f"{output_dir_prefix}_{current_time_str}"
    output_dir = xlsx_path.parent / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each use case
    all_processed_dfs = []
    for usecase in usecases_to_process:
        df_long = DataProcessor.process_usecase(
            xlsx_path=xlsx_path,
            sheet_identifier=sheet_identifier,
            usecase=usecase,
            metrics=args.metric,
            output_dir=output_dir,
            html_output=args.html_output,
            no_static_fallback=args.no_static_fallback
        )
        if df_long is not None:
            all_processed_dfs.append(df_long)
    
    # Process aggregate data for China use cases
    if is_china_special_case and all_processed_dfs:
        combined_df = pd.concat(all_processed_dfs, ignore_index=True)
        DataProcessor.process_aggregate_data(
            all_data=combined_df,
            output_dir=output_dir,
            metrics=args.metric,
            usecases=usecases_to_process,
            html_output=args.html_output,
            analyze_fluctuation=args.analyze_fluctuation,
            no_static_fallback=args.no_static_fallback
        )
    elif is_china_special_case:
        logger.warning("No data collected for any China use case. Skipping aggregate output and fluctuation analysis.")

if __name__ == "__main__":
    main()