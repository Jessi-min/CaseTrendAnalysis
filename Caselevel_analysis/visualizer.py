#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization module for the case trend analysis tool.
Contains functions to create static and interactive plots.
"""

import logging
from pathlib import Path
from typing import List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import natural_sort_key
from config import PLOT_WIDTH_PER_VERSION, PLOT_HEIGHT, PLOT_DPI, COLOR_CYCLE

# Configure logging
logger = logging.getLogger(__name__)

# Check for plotly availability
try:
    import plotly.express as px
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
    pio.templates.default = "plotly_white"
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not found. Interactive HTML output will not be available. Please install it: pip install plotly pandas")

class Visualizer:
    """Class to handle all visualization functionality"""
    
    @staticmethod
    def plot_trend_static(data: pd.DataFrame, out_png: Path, title: str, metric_names: List[str]) -> None:
        """
        Generates and saves a static trend plot from the processed data.

        Args:
            data: DataFrame containing 'Version' and the specified metric columns
            out_png: Path to save the output PNG image
            title: Title for the plot
            metric_names: The actual names of the metric columns to plot
        """
        if data.empty:
            logger.warning("No data found to plot. Skipping plot generation.")
            return

        data["Version"] = data["Version"].astype(str)

        # Dynamic plot width to accommodate more data points/longer labels
        fig_width = max(PLOT_HEIGHT, len(data) * PLOT_WIDTH_PER_VERSION)
        plt.figure(figsize=(fig_width, PLOT_HEIGHT))

        # Iterate over metric_names to plot multiple lines
        for i, metric_name in enumerate(metric_names):
            if metric_name in data.columns:
                color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
                plt.plot(data["Version"], data[metric_name], marker="o", 
                        color=color, linewidth=2, label=metric_name)

                # Display values on each data point for the current metric
                for x, y in zip(data["Version"], data[metric_name]):
                    if pd.notna(y):
                        plt.text(x, y, f"{y:.2f}", ha="center", va="bottom", 
                                fontsize=8, color=color, alpha=0.8)
            else:
                logger.warning(f"Metric '{metric_name}' not found in the processed data. Skipping plot for this metric.")

        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=9)

        plt.title(title, fontsize=14)
        plt.xlabel("Version", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Metric")
        plt.tight_layout()

        try:
            plt.savefig(out_png, dpi=PLOT_DPI, bbox_inches='tight')
            logger.info(f"Plot saved to: {out_png}")
        except Exception as e:
            logger.error(f"Error saving plot to {out_png}: {e}")
        finally:
            plt.close()  # Always close the plot to free memory

    @staticmethod
    def plot_aggregate_trend_static(
        all_data: pd.DataFrame, 
        out_png: Path, 
        title: str, 
        metric_names: List[str], 
        usecase_names: List[str]
    ) -> None:
        """
        Generates and saves a comprehensive STATIC trend plot for multiple use cases and metrics.
        Uses seaborn for better visualization of grouped data.

        Args:
            all_data: Combined DataFrame containing 'Version', 'Usecase', 'Metric', and 'Value' columns
            out_png: Path to save the output PNG image
            title: Title for the plot
            metric_names: The actual names of the metric columns
            usecase_names: The list of use case names included
        """
        if all_data.empty:
            logger.warning("No aggregate data found to plot. Skipping aggregate plot generation.")
            return

        # Ensure 'Version' is string for plotting
        all_data["Version"] = all_data["Version"].astype(str)

        # Handle NaNs before plotting
        df_plot = all_data.dropna(subset=["Value"]).copy()

        if df_plot.empty:
            logger.warning("No valid data points after dropping NaNs for aggregate plot. Skipping.")
            return
        
        # Set a fixed height for each subplot and an aspect ratio
        subplot_height = 3.5
        subplot_aspect = 1.8
        num_cols_wrap = 4

        # Create the relational plot
        g = sns.relplot(
            data=df_plot,
            x="Version",
            y="Value",
            hue="Metric",
            col="Usecase",
            col_wrap=num_cols_wrap,
            kind="line",
            marker="o",
            height=subplot_height,
            aspect=subplot_aspect,
            facet_kws={'sharey': False, 'sharex': True},
            palette=COLOR_CYCLE
        )

        # Add titles and labels for each subplot
        g.set_axis_labels("Version", "Value")
        g.set_titles(col_template="{col_name}")

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

        g.fig.suptitle(title, y=1.02, fontsize=16)
        g.fig.tight_layout(rect=[0, 0.03, 1, 0.98])

        try:
            plt.savefig(out_png, dpi=PLOT_DPI, bbox_inches='tight')
            logger.info(f"Static aggregate plot saved to: {out_png}")
        except Exception as e:
            logger.error(f"Error saving static aggregate plot to {out_png}: {e}")
        finally:
            plt.close()

    @staticmethod
    def format_plotly_xaxis_tick(axis: Any) -> None:
        """Format plotly x-axis ticks for better display"""
        try:
            if all(pd.to_numeric([str(c) for c in axis.tickvals], errors='coerce').notna().all()):
                axis.tickformat = ".0f"
        except:
            pass

    @staticmethod
    def plot_trend_interactive(data: pd.DataFrame, out_html: Path, title: str, metric_names: List[str]) -> None:
        """
        Generates and saves an interactive HTML trend plot from the processed data for a single use case.
        
        Args:
            data: DataFrame containing 'Version' and the specified metric columns
            out_html: Path to save the output HTML file
            title: Title for the plot
            metric_names: The actual names of the metric columns to plot
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly is not installed. Cannot generate interactive HTML plot for single case.")
            return

        if data.empty:
            logger.warning("No data found to plot for single case. Skipping interactive HTML plot generation.")
            return

        # Melt the DataFrame to long format for Plotly Express
        df_plot_long = data.melt(
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
        fig.for_each_xaxis(Visualizer.format_plotly_xaxis_tick)

        try:
            pio.write_html(fig, file=str(out_html), auto_open=False, full_html=True)
            logger.info(f"Interactive HTML plot saved to: {out_html}")
        except Exception as e:
            logger.error(f"Error saving interactive HTML plot to {out_html}: {e}")

    @staticmethod
    def plot_aggregate_trend_interactive(
        all_data: pd.DataFrame, 
        out_html: Path, 
        title: str, 
        metric_names: List[str], 
        usecase_names: List[str],
        fluctuation_report_html: str = ""
    ) -> None:
        """
        Generates and saves a comprehensive INTERACTIVE HTML trend plot for multiple use cases and metrics.
        
        Args:
            all_data: Combined DataFrame containing 'Version', 'Usecase', 'Metric', and 'Value' columns
            out_html: Path to save the output HTML file
            title: Title for the plot
            metric_names: The actual names of the metric columns
            usecase_names: The list of use case names included
            fluctuation_report_html: Optional HTML string with fluctuation analysis to include
        """
        if not PLOTLY_AVAILABLE:
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
            num_cols_facet = 4
     
        # Create interactive plot using Plotly Express
        fig = px.line(
            df_plot,
            x="Version",
            y="Value",
            color="Metric",
            line_group="Metric",
            facet_col="Usecase",
            facet_col_wrap=num_cols_facet,
            title=title,
            labels={"Value": "Value", "Version": "Version"},
            hover_data={"Usecase": False, "Metric": True, "Value": ":.2f", "Version": True}
        )

        fig.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>", font_size=12))

        fig.update_layout(
            autosize=True,
            hovermode="x unified",
            font=dict(size=10),
            title_x=0.5,
            margin=dict(t=50, b=50, l=40, r=40),
            legend_title_text="Metric",
        )

        fig.update_xaxes(
            tickangle=45,
            categoryorder="array",
            categoryarray=all_unique_versions)
            
        fig.update_yaxes(
            nticks=4,
            tickformat=".2f",
            tickfont=dict(size=9)
        )
        fig.for_each_xaxis(Visualizer.format_plotly_xaxis_tick)

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
                    .plot-container {{ width: 100%; height: auto; min-height: 600px; }}
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