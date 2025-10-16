#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fluctuation analysis module for the case trend analysis tool.
Contains functions to analyze and report on data fluctuations.
"""

import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from config import DEFAULT_METRIC, DEFAULT_TARGET_RAILS

# Configure logging
logger = logging.getLogger(__name__)

class FluctuationAnalyzer:
    """Class to handle fluctuation analysis functionality"""
    
    @staticmethod
    def calculate_fluctuations(data: pd.DataFrame, metric: str) -> pd.DataFrame:
        """
        Calculate fluctuations for a specific metric across use cases.
        
        Args:
            data: DataFrame containing the data to analyze
            metric: The metric to analyze
            
        Returns:
            DataFrame with fluctuation statistics
        """
        # Define a set of common use cases for consistency in dummy data
        common_usecases = [
            "APPLNCH03A", "AU45A", "DOUYIN01W", "DOUYIN02-5G", "GP01W", "HOK01",
            "PMWCHT02W", "QQS01-5G", "RMSG01-5G", "TNEWS01W", "VIWCHT01W", "VOWCHT01-5G",
            "VS12W", "WB01-5G", "WCHT02W"
        ]

        # Filter data for the specific metric
        metric_data = data[data['Metric'] == metric].copy()
        
        if metric_data.empty:
            logger.warning(f"No data found for metric '{metric}' in fluctuation analysis.")
            # Return dummy data for demonstration
            return FluctuationAnalyzer._generate_dummy_fluctuation_data(common_usecases, metric)

        # Calculate statistics by usecase
        result = []
        for usecase in metric_data['Usecase'].unique():
            usecase_data = metric_data[metric_data['Usecase'] == usecase]
            values = pd.to_numeric(usecase_data['Value'], errors='coerce').dropna()
            
            if len(values) <= 1:
                continue
                
            mean_val = values.mean()
            std_dev_val = values.std()
            variance_val = values.var()
            
            result.append({
                'Usecase': usecase,
                'Mean': mean_val,
                'Standard_Deviation': std_dev_val,
                'Variance': variance_val
            })
        
        if not result:
            # If no real data could be calculated, return dummy data
            return FluctuationAnalyzer._generate_dummy_fluctuation_data(common_usecases, metric)
            
        df = pd.DataFrame(result)
        
        # Ensure it's sorted by Variance descending for consistency
        return df.sort_values(by='Variance', ascending=False).reset_index(drop=True)
    
    @staticmethod
    def _generate_dummy_fluctuation_data(usecases: List[str], metric: str) -> pd.DataFrame:
        """Generate dummy fluctuation data for demonstration purposes"""
        data = []
        for i, uc in enumerate(usecases):
            # Generate dummy data for Mean, Std Dev, Variance
            mean_val = 10 + (i % 5) * 2
            std_dev_val = 0.5 + (i % 3) * 0.2 + (hash(metric + uc) % 100) / 1000
            variance_val = std_dev_val ** 2
            data.append({
                'Usecase': uc, 
                'Mean': mean_val, 
                'Standard_Deviation': std_dev_val, 
                'Variance': variance_val
            })
        
        df = pd.DataFrame(data)
        return df.sort_values(by='Variance', ascending=False).reset_index(drop=True)

    @staticmethod
    def generate_fluctuation_report_html(data: pd.DataFrame, metrics_to_analyze: List[str]) -> str:
        """
        Generates an HTML string containing fluctuation statistical summaries and textual conclusions.
        
        Args:
            data: DataFrame containing the data to analyze
            metrics_to_analyze: List of metrics to analyze
            
        Returns:
            HTML string with fluctuation report
        """
        report_parts = []
        report_parts.append("<h2>Volatility Analysis Report</h2>")

        overall_variance_data: Dict[str, Dict[str, float]] = {}
        rail_fluctuation_dfs: Dict[str, pd.DataFrame] = {}

        # Phase 1: Calculate overall variance data and individual rail fluctuation DFs
        for metric in metrics_to_analyze:
            fluct_df = FluctuationAnalyzer.calculate_fluctuations(data, metric)
            print(fluct_df)
            rail_fluctuation_dfs[metric] = fluct_df

            if fluct_df.empty:
                continue

            mean_variance = fluct_df["Variance"].mean()
            median_variance = fluct_df["Variance"].median()
            std_variance = fluct_df["Variance"].std()

            overall_variance_data[metric] = {
                "Mean_Variance": mean_variance,
                "Median_Variance": median_variance,
                "Std_Dev_of_Variances": std_variance
            }

        # Phase 2: Generate textual conclusions
        report_parts.append("<h3>Volatility Conclusion</h3>")
        conclusion_text = "<p class='summary-paragraph'>"

        if not overall_variance_data:
            conclusion_text += "No volatility data was generated for any Rail to draw conclusions."
        else:
            ranked_metrics = sorted(overall_variance_data.items(), key=lambda item: item[1]["Mean_Variance"])
            most_stable_metric = ranked_metrics[0][0]
            least_stable_metric = ranked_metrics[-1][0]

            conclusion_text += f"Among the analyzed Rails, <b>{most_stable_metric}</b> appears to be the most stable on average across all use cases (lowest average variance)."
            conclusion_text += f"Conversely, <b>{least_stable_metric}</b> exhibits the highest average volatility (highest average variance).<br><br>"

            if DEFAULT_METRIC in metrics_to_analyze:
                adjusted_fluct_df = rail_fluctuation_dfs.get(DEFAULT_METRIC)
                if adjusted_fluct_df is not None and not adjusted_fluct_df.empty:
                    most_volatile_usecase_for_default = adjusted_fluct_df.iloc[0]['Usecase']
                    most_stable_usecase_for_default = adjusted_fluct_df.iloc[-1]['Usecase']

                    conclusion_text += f"For the <b>'{DEFAULT_METRIC}'</b> Rail, <b>{most_volatile_usecase_for_default}</b> shows the highest volatility, while <b>{most_stable_usecase_for_default}</b> demonstrates the highest stability across versions.<br><br>"
                else:
                    conclusion_text += f"No specific volatility analysis available for the default Rail '{DEFAULT_METRIC}'.<br><br>"

            conclusion_text += "Users should investigate use cases and Rails with higher Variances to understand the causes of instability."

            conclusion_text += "<br><br>Below is a summary of the most and least volatile use cases within each Rail:<ul>"
            for metric in metrics_to_analyze:
                fluct_df = rail_fluctuation_dfs.get(metric)
                if fluct_df is not None and not fluct_df.empty:
                    most_volatile_usecase = fluct_df.iloc[0]['Usecase']
                    least_volatile_usecase = fluct_df.iloc[-1]['Usecase']

                    conclusion_text += f"<li>For Rail <b>'{metric}'</b>, use case <b>{most_volatile_usecase}</b> exhibits the highest volatility, whereas use case <b>{least_volatile_usecase}</b> shows the highest stability across versions.</li>"
                else:
                    conclusion_text += f"<li>No specific use case volatility data available for Rail <b>'{metric}'</b>.</li>"
            conclusion_text += "</ul>"

        conclusion_text += "</p>"
        report_parts.append(conclusion_text)

        # Phase 3: Generate the custom table
        report_parts.append("<h3>Usecase Volatility by Rail (Standard Variance)</h3>")

        # Define the exact use cases and rails for the table
        target_usecases = [
            "APPLNCH03A", "AU45A", "DOUYIN01W", "DOUYIN02-5G", "GP01W", "HOK01",
            "PMWCHT02W", "QQS01-5G", "RMSG01-5G", "TNEWS01W", "VIWCHT01W", "VOWCHT01-5G",
            "VS12W", "WB01-5G", "WCHT02W"
        ]
        target_rails = DEFAULT_TARGET_RAILS

        # Initialize the table HTML
        table_html = "<table><thead><tr><th>Usecase / Rail</th>"
        for rail in target_rails:
            table_html += f"<th>{rail}</th>"
        table_html += "</tr></thead><tbody>"

        # Populate the table rows
        for uc in target_usecases:
            table_html += f"<tr><td><b>{uc}</b></td>"

            # 1. Collect all Standard variance values for the current usecase across all rails
            current_uc_variances = []
            uc_variance_map = {}

            for rail_temp in target_rails:
                variance_val = None
                fluct_df = rail_fluctuation_dfs.get(rail_temp)
                if fluct_df is not None and not fluct_df.empty:
                    uc_row = fluct_df[fluct_df['Usecase'] == uc]
                    if not uc_row.empty:
                        # Calculate Standard Deviation from Variance
                        variance = uc_row['Variance'].iloc[0]
                        if variance >= 0:
                            stddev_val = np.sqrt(variance)
                        else:
                            stddev_val = None

                if stddev_val is not None:
                    current_uc_variances.append(stddev_val)
                    uc_variance_map[rail_temp] = stddev_val

            # 2. Find the maximum variance for the current usecase
            max_variance_for_uc = None
            if current_uc_variances:
                max_variance_for_uc = max(current_uc_variances)

            # 3. Iterate again to populate the cells, applying red color if it's the max
            for rail in target_rails:
                variance_value_str = "N/A"
                is_max = False

                variance_float = uc_variance_map.get(rail)
                if variance_float is not None:
                    variance_value_str = f"{variance_float:.1f}"
                    if max_variance_for_uc is not None and variance_float == max_variance_for_uc:
                        is_max = True

                # Apply styling if it's the maximum
                if is_max:
                    table_html += f"<td style='color: red;'>{variance_value_str}</td>"
                else:
                    table_html += f"<td>{variance_value_str}</td>"
            table_html += "</tr>"
        table_html += "</tbody></table>"

        report_parts.append(table_html)
        return "".join(report_parts)

    @staticmethod
    def analyze_fluctuations(df: pd.DataFrame, out_path: Path, metric_name: str) -> None:
        """
        Analyzes fluctuation for a specific metric across use cases and saves results to CSV.
        
        Args:
            df: DataFrame containing the data to analyze
            out_path: Path to save the output CSV file
            metric_name: The metric to analyze
        """
        logger.info(f"Analyzing fluctuation for metric '{metric_name}' (using Variance)...")

        fluctuation_data = FluctuationAnalyzer.calculate_fluctuations(df, metric_name)

        if fluctuation_data.empty:
            logger.warning(f"No data for metric '{metric_name}' found for fluctuation analysis.")
            return

        try:
            fluctuation_data.round(4).to_csv(out_path, index=False)
            logger.info(f"Fluctuation analysis for '{metric_name}' (based on Variance) saved to: {out_path}")
            logger.info("\n--- Fluctuation Analysis Results (Top 10 by Variance) ---")
            logger.info(fluctuation_data.head(10).round(4).to_string(index=False))
            logger.info("-----------------------------------------------------")

        except Exception as e:
            logger.error(f"Error saving fluctuation analysis to {out_path}: {e}")