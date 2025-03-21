from collections import Counter
from typing import List, Dict, Any

from config import logger


def analyze_sentiment(sentiment_analyzer, text: str) -> str:
    try:
        result = sentiment_analyzer(text)
        label = result[0]['label']
        rating = int(label.split()[0])
        if rating <= 2:
            sentiment = "Negative"
        elif rating == 3:
            sentiment = "Neutral"
        else:
            sentiment = "Positive"
        logger.debug(
            f"Sentiment analysis: '{text[:30]}...' => {label} mapped to {sentiment}")
        return sentiment
    except Exception as e:
        logger.debug(f"Error in sentiment analysis: {e}")
        return "Unknown"


def attach_sentiment_to_articles(sentiment_analyzer, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed_articles = []
    for article in articles:
        # Check that the article is a dictionary
        if not isinstance(article, dict):
            logger.debug(
                "Invalid article format; expected a dictionary. Skipping article.")
            continue

        # Safely retrieve 'summary'; if missing or not a string, use a default sentiment
        summary = article.get("summary")
        if not isinstance(summary, str):
            logger.debug(
                "Missing or invalid 'summary' in article; assigning 'Unknown' sentiment.")
            sentiment = "Unknown"
        else:
            try:
                sentiment = analyze_sentiment(sentiment_analyzer, summary)
            except Exception as e:
                logger.debug(f"Error analyzing sentiment for article: {e}")
                sentiment = "Unknown"

        # Attach the sentiment to the article
        new_article = article.copy()
        new_article["sentiment"] = sentiment
        processed_articles.append(new_article)
    return processed_articles


def merge_articles(articles: List[Dict[str, Any]], summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Warn if there is a length mismatch between the two lists.
    if len(articles) != len(summaries):
        logger.debug(f"Warning: Length mismatch between articles ({len(articles)}) and summaries ({len(summaries)}). "
                     "Only merging common pairs.")

    merged = []
    for article, summary in zip(articles, summaries):
        # Validate that both article and summary are dictionaries.
        if not isinstance(article, dict) or not isinstance(summary, dict):
            logger.debug(
                "Invalid format: both article and summary must be dictionaries. Skipping this pair.")
            continue
        # Merge the two dictionaries. Note that keys in summary will overwrite those in article if duplicated.
        merged_article = {**article, **summary}
        merged.append(merged_article)
    return merged


def get_sentiment_distribution(articles: List[Dict[str, Any]]) -> Dict[str, int]:
    sentiments = []
    for article in articles:
        # Check that the article is a dictionary.
        if not isinstance(article, dict):
            logger.debug(
                "Invalid article format; expected a dictionary. Skipping article.")
            continue

        # Safely retrieve the 'sentiment' value; default to 'Unknown' if missing or not a string.
        sentiment = article.get("sentiment", "Unknown")
        if not isinstance(sentiment, str):
            logger.debug(
                "Invalid sentiment type; expected a string. Using 'Unknown'.")
            sentiment = "Unknown"
        sentiments.append(sentiment)
    return dict(Counter(sentiments))
