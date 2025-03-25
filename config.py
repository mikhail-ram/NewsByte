"""
Module for setting up logging configuration for NewsByte.

This module initializes a logger with the name "NewsByte" and configures it to output log messages
to the console. It sets up a stream handler with a specific log message format and adjusts the log level
based on the DEBUG_MODE flag.
"""
import logging

logger = logging.getLogger("NewsByte")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

DEBUG_MODE = True

if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)
    ch.setLevel(logging.DEBUG)
