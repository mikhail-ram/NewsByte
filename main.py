import time
from typing import Dict, Any, List
import asyncio

from config import logger
from scraping import fetch_news_articles
from analysis import attach_sentiment_to_articles, merge_articles, get_sentiment_distribution
from llm import create_model, extract_articles_summary, extract_comparative_sentiment_score, extract_final_sentiment_analysis
from utils import save_news_to_json
from tts import translate_text, hindi_tts


def run_newsbyte(company: str, num_articles: int = 10) -> Dict[str, Any]:
    model = create_model("deepseek/deepseek-r1:free")

    logger.debug(f"Starting news fetch for {company}")
    raw_articles = fetch_news_articles(company, num_articles=num_articles)
    logger.debug(f"Fetched {len(raw_articles)} articles.")

    # Extract topics and summaries using LLM
    articles_summaries = extract_articles_summary(model, raw_articles)
    merged_articles = merge_articles(raw_articles, articles_summaries)

    # Attach sentiment analysis based on summary
    logger.debug("Attaching sentiment to articles.")
    articles_with_sentiment = attach_sentiment_to_articles(merged_articles)

    logger.debug("Extracting comparative sentiment score.")
    comp_score = extract_comparative_sentiment_score(
        model, articles_with_sentiment)
    comp_score_dict = comp_score.model_dump()

    logger.debug("Getting sentiment distribution.")
    comp_score_dict["Sentiment Distribution"] = get_sentiment_distribution(
        articles_with_sentiment)

    logger.debug("Extracting final sentiment analysis.")
    final_analysis = extract_final_sentiment_analysis(
        model, company, comp_score_dict)

    logger.debug("Translating final sentiment analysis.")
    translated_final_analysis = asyncio.run(translate_text(final_analysis))

    logger.debug("Running TTS for final sentiment analysis.")
    output_tts_path = "hindi_tts.wav"
    hindi_tts(translated_final_analysis, output_tts_path)

    logger.debug("Done!")

    output = {
        "Company": company,
        "Articles": articles_with_sentiment,
        "Comparative Sentiment Score": comp_score_dict,
        "Final Sentiment Analysis": final_analysis,
        "Translated Final Sentiment Analysis": translated_final_analysis,
        "Audio": output_tts_path
    }
    save_news_to_json(company, output)
    return output


if __name__ == "__main__":
    company = "Tesla"
    run_newsbyte(company)
