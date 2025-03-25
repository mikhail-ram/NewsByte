"""
Utility functions for text transformation and saving JSON output for NewsByte.
"""
import json
from typing import Dict, Any
from config import logger


def to_snake_case(text: str) -> str:
    """
    Convert the input text to snake_case.

    Parameters:
        text (str): The text to convert.

    Returns:
        str: The text converted to snake_case.
    """
    return text.lower().replace(" ", "_")


def to_title_case(text: str) -> str:
    """
    Convert the input text to title case.

    Parameters:
        text (str): The text to convert.

    Returns:
        str: The text converted to title case.
    """
    return " ".join(w.capitalize() if w[0].islower() else w for w in text.split())


def save_news_to_json(company: str, final_output: Dict[str, Any]) -> None:
    """
    Save the final output of the news analysis to a JSON file.

    The filename is generated based on the company name in snake_case, appended with '_newsbyte.json'.

    Parameters:
        company (str): The company name used to generate the filename.
        final_output (Dict[str, Any]): The dictionary containing the final output to save.

    Returns:
        None
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
