import json
from typing import Dict, Any
from config import logger


def to_snake_case(text: str) -> str:
    return text.lower().replace(" ", "_")


def save_news_to_json(company: str, final_output: Dict[str, Any]) -> None:
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
