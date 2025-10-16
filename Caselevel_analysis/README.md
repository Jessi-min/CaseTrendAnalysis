# Case Trend Analysis Tool

A tool for analyzing and visualizing metric trends across different versions from Excel data files.

## Overview

This tool reads an Excel file containing performance metrics for various use cases across different versions, detects the layout format (grouped or wide), extracts specified metric data, and generates trend plots and analysis reports.

## Features

- Support for both "grouped" and "wide" Excel layouts
- Single use case or batch processing of predefined use cases
- Multiple metric analysis
- Static PNG and interactive HTML visualization
- Fluctuation analysis for metrics across versions
- CSV data export

## Installation

### Requirements

- Python 3.6+
- Required packages:
  - pandas
  - matplotlib
  - seaborn
  - numpy
  - plotly (optional, for interactive HTML plots)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/case-trend-analysis.git
cd case-trend-analysis

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Plot a single metric for a single use case
python main.py "Caselevel_Trendency.xlsx" --sheet 0 --usecase AU45A --metric Adjusted

# Plot multiple metrics for a single use case
python main.py "Caselevel_Trendency.xlsx" --sheet 0 --usecase AU45A --metric Adjusted CX Power

# Plot multiple metrics with interactive HTML output
python main.py "Caselevel_Trendency.xlsx" --sheet 0 --usecase AU45A --metric Adjusted CX Power --html-output
```

### Batch Processing for China Use Cases

```bash
# Process all China use cases with multiple metrics and fluctuation analysis
python main.py "Caselevel_Trendency.xlsx" --sheet 0 --usecase China --metric Adjusted CX GFX "Total CPU" "Total Memory" --html-output --analyze-fluctuation
```

### Command Line Arguments

- `xlsx`: Path to the Excel file
- `--sheet`: Sheet index (0-based) or sheet name (default: 0)
- `--usecase`: Target use case (default: AU45A)
  - Special value 'China' processes a predefined list of China-related use cases
- `--metric`: Target metric column name(s) (default: Adjusted)
- `--analyze-fluctuation`: Perform fluctuation analysis
- `--html-output`: Generate interactive HTML plots (requires Plotly)
- `--no-static-fallback`: Don't fall back to static PNG if Plotly is unavailable
- `--debug`: Enable debug logging

## Output Files

- For individual use cases (e.g., AU45A):
  - `AU45A_Adjusted_trend.png` (static plot)
  - `AU45A_Adjusted_trend.html` (interactive HTML plot if --html-output is used)
  - `AU45A_Adjusted_trend_data.csv`
- For 'China' usecase (aggregate functionality):
  - Individual plots/CSVs for each usecase (e.g., RMSG01-5G_trend.png)
  - `aggregate_China_Adjusted_trend.png` (Static Plot)
  - `aggregate_China_Adjusted_trend.html` (Interactive Plot with fluctuation summary)
  - `aggregate_China_Adjusted_data_long_format.csv` (Combined data for all China use cases)
  - `aggregate_China_Adjusted_fluctuation.csv` (Analysis of Adjusted metric's fluctuation)

All files are saved in a timestamped folder like `output_YYYYMMDD_HHMMSS` or `aggregate_output_YYYYMMDD_HHMMSS`.

## Project Structure

- `main.py`: Main entry point and argument parsing
- `config.py`: Configuration constants and settings
- `utils.py`: Utility functions for data handling
- `data_parser.py`: Functions to parse Excel data in different layouts
- `data_processor.py`: Core data processing functionality
- `visualizer.py`: Visualization functions for static and interactive plots
- `fluctuation_analyzer.py`: Fluctuation analysis functionality

## License

Copyright © Qualcomm (C) 2023-2024