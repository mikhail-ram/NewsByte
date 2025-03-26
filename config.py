"""Configuration module for NewsByte application.

This module sets up logging configuration and debug mode settings.
It configures a logger with both console output and appropriate formatting.
"""

import logging

# Initialize logger with name "NewsByte"
logger = logging.getLogger("NewsByte")
logger.setLevel(logging.INFO)

# Configure console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Global debug mode flag
DEBUG_MODE = True

# Set debug level if debug mode is enabled
if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)
    ch.setLevel(logging.DEBUG)
