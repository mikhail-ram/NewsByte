import argparse
from typing import Dict, Any
import asyncio
from transformers import pipeline

from config import logger
from scraping import fetch_news_articles
from analysis import attach_sentiment_to_articles, merge_articles, get_sentiment_distribution
from llm import create_model, extract_articles_summary, extract_comparative_sentiment_score, extract_final_sentiment_analysis
from utils import save_news_to_json
from tts import translate_text, hindi_tts


def run_newsbyte(company: str, num_articles: int = 10) -> Dict[str, Any]:
    """
    Run the complete NewsByte analysis workflow for a given company.

    This function performs the following steps:
      - Creates the language model and sentiment analyzer.
      - Fetches news articles for the specified company.
      - Extracts topics and summaries using the language model.
      - Merges the original articles with their summaries.
      - Attaches sentiment analysis to each article.
      - Computes a comparative sentiment score and its distribution.
      - Generates a final sentiment analysis and translates it.
      - Converts the translated analysis to audio using TTS.
      - Saves the output to a JSON file.

    Parameters:
        company (str): The company name to analyze.
        num_articles (int): The number of articles to fetch (default is 10).

    Returns:
        Dict[str, Any]: A dictionary containing the complete analysis output.
    """
    model = create_model("deepseek/deepseek-r1:free")
    sentiment_analyzer = pipeline(
        "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    logger.debug(f"Starting news fetch for {company}")
    articles = fetch_news_articles(company, num_articles=num_articles)
    logger.debug(f"Fetched {len(articles)} articles.")

    # Extract topics and summaries using LLM
    articles_summaries = extract_articles_summary(model, articles)
    merged_articles = merge_articles(articles, articles_summaries)

    # Attach sentiment analysis based on summary
    logger.debug("Attaching sentiment to articles.")
    articles_with_sentiment = attach_sentiment_to_articles(
        sentiment_analyzer, merged_articles)

    logger.debug("Extracting comparative sentiment score.")
    comp_score = extract_comparative_sentiment_score(
        model, articles_with_sentiment)
    comp_score_dict = comp_score.model_dump()

    logger.debug("Getting sentiment distribution.")
    comp_score_dict["Sentiment_Distribution"] = get_sentiment_distribution(
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
        "Comparative_Sentiment_Score": comp_score_dict,
        "Final_Sentiment_Analysis": final_analysis,
        "Translated_Final_Sentiment_Analysis": translated_final_analysis,
        "Audio": output_tts_path
    }
    save_news_to_json(company, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run NewsByte analysis on a company.')
    parser.add_argument('company', type=str,
                        help='Company name to analyze')
    parser.add_argument('--num_articles', type=int, default=10,
                        help='Number of articles to fetch (default: 10)')

    args = parser.parse_args()
    run_newsbyte(args.company, args.num_articles)
