#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data processing module for the case trend analysis tool.
Contains functions to process data from Excel files and generate outputs.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
from utils import detect_layout, natural_sort_key
from data_parser import parse_grouped, parse_wide
from visualizer import Visualizer, PLOTLY_AVAILABLE
from fluctuation_analyzer import FluctuationAnalyzer
from config import DEFAULT_METRIC

# Configure logging
logger = logging.getLogger(__name__)

class DataProcessor:
    """Class to handle data processing functionality"""
    
    @staticmethod
    def process_usecase(
        xlsx_path: Path,
        sheet_identifier: Union[int, str],
        usecase: str,
        metrics: List[str],
        output_dir: Path,
        html_output: bool,
        no_static_fallback: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Process a single use case from the Excel file.
        
        Args:
            xlsx_path: Path to the Excel file
            sheet_identifier: Sheet index or name
            usecase: The use case to process
            metrics: List of metrics to extract
            output_dir: Directory to save output files
            html_output: Whether to generate HTML output
            no_static_fallback: If True, don't fall back to static plot when HTML requested but Plotly unavailable
            
        Returns:
            DataFrame in long format with the processed data, or None if processing failed
        """
        logger.info(f"\n--- Processing data for use case: '{usecase}' ---")
        
        try:
            # Read Excel file
            df_raw = pd.read_excel(xlsx_path, sheet_name=sheet_identifier, header=None)
            
            # Detect layout
            layout = detect_layout(df_raw)
            logger.debug(f"Detected layout: '{layout}' for use case '{usecase}'")
            
            # Parse data based on layout
            if layout == "grouped":
                au_data_frame = parse_grouped(df_raw, usecase=usecase, metrics=metrics)
            else:  # wide
                au_data_frame = parse_wide(df_raw, usecase=usecase, metrics=metrics)
                
            if au_data_frame.empty:
                logger.info(f"No data processed for the specified Usecase ('{usecase}') and Metrics ('{', '.join(metrics)}').")
                return None
                
            logger.info("\n--- Processed Versions and Metric values ---")
            display_cols = ["Version"] + [m for m in metrics if m in au_data_frame.columns]
            if display_cols:
                logger.info(au_data_frame[display_cols].to_string(index=False))
            else:
                logger.info("No relevant metric columns found for display.")
            logger.info("---------------------------------------------")
            
            # Convert to long format for aggregation
            df_long = au_data_frame.melt(
                id_vars=["Version"],
                value_vars=[m for m in metrics if m in au_data_frame.columns],
                var_name="Metric",
                value_name="Value"
            )
            df_long["Usecase"] = usecase
            
            # Generate output filenames - SIMPLIFIED
            metrics_str_for_filename = "_".join([m.lower().replace(" ", "") for m in metrics])
            output_base_name = f"{usecase}_{metrics_str_for_filename}"
            out_csv_path = output_dir / f"{output_base_name}.csv"
            out_png_path = output_dir / f"{output_base_name}.png"
            out_html_path = output_dir / f"{output_base_name}.html"
            
            # Save data to CSV
            try:
                # Convert back to wide format for individual CSV
                au_data_frame_for_save = df_long.pivot(
                    index='Version', columns='Metric', values='Value'
                ).reset_index()
                au_data_frame_for_save.columns.name = None
                au_data_frame_for_save.to_csv(out_csv_path, index=False)
                logger.info(f"Individual data saved to: {out_csv_path}")
            except Exception as e:
                logger.error(f"Error saving individual data to CSV for '{usecase}': {e}")
            
            # Generate plots
            metrics_to_plot = [m for m in metrics if m in au_data_frame.columns]
            if metrics_to_plot:
                if html_output and PLOTLY_AVAILABLE:
                    Visualizer.plot_trend_interactive(
                        au_data_frame, 
                        out_html_path,
                        title=f"{usecase} Trend by Version ({', '.join(metrics_to_plot)})",
                        metric_names=metrics_to_plot
                    )
                elif not html_output or (html_output and not PLOTLY_AVAILABLE and not no_static_fallback):
                    # Fallback to static PNG if HTML not requested or Plotly not available (and fallback not disabled)
                    if html_output and not PLOTLY_AVAILABLE:
                        logger.warning(f"Plotly not available for individual plot '{usecase}', falling back to static PNG.")
                    Visualizer.plot_trend_static(
                        au_data_frame, 
                        out_png_path,
                        title=f"{usecase} Trend by Version ({', '.join(metrics_to_plot)})",
                        metric_names=metrics_to_plot
                    )
                else:  # html_output and not PLOTLY_AVAILABLE and no_static_fallback
                    logger.warning(f"Plotly not available and static fallback disabled for individual plot '{usecase}'. No plot generated.")
            else:
                logger.warning(f"No specified metrics found in the processed data for '{usecase}' to plot individually.")
                
            return df_long
            
        except Exception as e:
            logger.error(f"Error processing use case '{usecase}': {e}", exc_info=True)
            return None

    @staticmethod
    def process_aggregate_data(
        all_data: pd.DataFrame,
        output_dir: Path,
        metrics: List[str],
        usecases: List[str],
        html_output: bool,
        analyze_fluctuation: bool,
        no_static_fallback: bool = False
    ) -> None:
        """
        Process aggregate data for multiple use cases.
        
        Args:
            all_data: Combined DataFrame with all use case data
            output_dir: Directory to save output files
            metrics: List of metrics to include
            usecases: List of use cases included
            html_output: Whether to generate HTML output
            analyze_fluctuation: Whether to analyze fluctuation
            no_static_fallback: If True, don't fall back to static plot when HTML requested but Plotly unavailable
        """
        logger.info("\n--- Generating aggregate plot and data for all China use cases ---")
        
        if all_data.empty:
            logger.warning("No data to aggregate. Skipping aggregate processing.")
            return
            
        # Ensure proper sorting of versions
        all_unique_versions = sorted(all_data['Version'].unique().tolist(), key=natural_sort_key)
        all_data["Version"] = pd.Categorical(all_data["Version"], categories=all_unique_versions, ordered=True)
        all_data = all_data.sort_values(by=["Usecase", "Version", "Metric"]).reset_index(drop=True)
        
        # Save aggregate data
        metrics_str_for_filename = "_".join([m.lower().replace(" ", "") for m in metrics])
        agg_csv_path = output_dir / f"China_{metrics_str_for_filename}.csv"
        try:
            all_data.to_csv(agg_csv_path, index=False)
            logger.info(f"Aggregate data (long format) saved to: {agg_csv_path}")
        except Exception as e:
            logger.error(f"Error saving aggregate data to CSV: {e}")
        
        # Generate plots
        metrics_to_plot = [m for m in metrics if m in all_data["Metric"].unique()]
        if not metrics_to_plot:
            logger.warning("No specified metrics found in the combined data to plot for the aggregate view.")
            return
            
        # Generate fluctuation report HTML if requested
        fluctuation_report_html = ""
        if analyze_fluctuation:
            fluctuation_report_html = FluctuationAnalyzer.generate_fluctuation_report_html(all_data, metrics_to_plot)
            
        if html_output and PLOTLY_AVAILABLE:
            agg_html_path = output_dir / f"China_{metrics_str_for_filename}.html"
            Visualizer.plot_aggregate_trend_interactive(
                all_data,
                agg_html_path,
                title=f"Trend for All China Use Cases ({', '.join(metrics_to_plot)})",
                metric_names=metrics_to_plot,
                usecase_names=usecases,
                fluctuation_report_html=fluctuation_report_html
            )
        elif not html_output or (html_output and not PLOTLY_AVAILABLE and not no_static_fallback):
            # Fallback to static PNG if HTML not requested or Plotly not available (and fallback not disabled)
            if html_output and not PLOTLY_AVAILABLE:
                logger.warning("Plotly not available, falling back to static PNG plot.")
            agg_png_path = output_dir / f"China_{metrics_str_for_filename}.png"
            Visualizer.plot_aggregate_trend_static(
                all_data,
                agg_png_path,
                title=f"Trend for All China Use Cases ({', '.join(metrics_to_plot)})",
                metric_names=metrics_to_plot,
                usecase_names=usecases
            )
        else:  # html_output and not PLOTLY_AVAILABLE and no_static_fallback
            logger.warning("Plotly not available and static fallback disabled. No aggregate plot generated.")
        
        # Perform fluctuation analysis if requested (for CSV output)
        if analyze_fluctuation and DEFAULT_METRIC in all_data["Metric"].unique():
            fluctuation_output_path = output_dir / f"aggregate_China_{DEFAULT_METRIC}_variance_fluctuation.csv"
            FluctuationAnalyzer.analyze_fluctuations(all_data, fluctuation_output_path, DEFAULT_METRIC)
        elif analyze_fluctuation:
            logger.warning(f"Fluctuation analysis requested for '{DEFAULT_METRIC}', but this metric was not collected.")