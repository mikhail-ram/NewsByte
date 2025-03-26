"""Utility functions for NewsByte application.

This module provides helper functions for text processing and file operations.
"""

import json
from typing import Dict, Any
from config import logger


def to_snake_case(text: str) -> str:
    """Convert text to snake_case format.

    Args:
        text (str): Text to convert

    Returns:
        str: Text in snake_case format
    """
    return text.lower().replace(" ", "_")


def to_title_case(text: str) -> str:
    """Convert text to Title Case format.

    Args:
        text (str): Text to convert

    Returns:
        str: Text in Title Case format
    """
    return " ".join(w.capitalize() if w[0].islower() else w for w in text.split())


def save_news_to_json(company: str, final_output: Dict[str, Any]) -> None:
    """Save news analysis results to a JSON file.

    Args:
        company (str): Company name for the output file
        final_output (Dict[str, Any]): Analysis results to save

    Raises:
        IOError: If file operations fail
        TypeError: If JSON serialization fails
        Exception: For any other unexpected errors
    """
    filename = f"{to_snake_case(company)}_newsbyte.json"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        logger.debug(f"Saved final output to {filename}")
    except (IOError, OSError) as file_error:
        logger.error(f"File error while saving JSON: {file_error}")
    except TypeError as json_error:
        logger.error(f"JSON serialization error: {json_error}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
