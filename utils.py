import json
from typing import Dict, Any

from config import logger


def to_snake_case(text: str) -> str:
    return text.lower().replace(" ", "_")


def save_news_to_json(company: str, final_output: Dict[str, Any]) -> None:
    filename = f"{to_snake_case(company)}_news.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    logger.debug(f"Saved final output to {filename}")
