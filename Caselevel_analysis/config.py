#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration module for the case trend analysis tool.
Contains constants, default values, and configuration settings.
"""

import re
import matplotlib.colors as mcolors

# Regular expression patterns
VERSION_EXTRACT_PATTERN = re.compile(
    r"M\d{2,3}(?:[_-]\d+)?(?:[A-Z]+)?",  # MXXX, MXXX-YY, MXXX_YY, MXXXE
    re.IGNORECASE
)

# Common header keywords
USECASE_KEYWORD = "usecase"
DEFAULT_METRIC = "Adjusted"
DEFAULT_USECASE = "AU45A"

# Predefined list of use cases for "China" parameter
CHINA_USECASES = [
    "RMSG01-5G", "DOUYIN02-5G", "AU45A", "VS12W", "DOUYIN01W", 
    "GP01W", "HOK01", "QQS01-5G", "TNEWS01W", "WCHT02W", 
    "PMWCHT02W", "WB01-5G", "VOWCHT01-5G", "VIWCHT01W", "APPLNCH03A",
]

# Plotting constants
PLOT_WIDTH_PER_VERSION = 1.2
PLOT_HEIGHT = 9
PLOT_DPI = 160
COLOR_CYCLE = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())

# Default target rails for fluctuation analysis
DEFAULT_TARGET_RAILS = ["Adjusted", "CX", "GFX", "Total CPU", "Total Memory"]