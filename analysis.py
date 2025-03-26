from collections import Counter
from typing import List, Dict, Any

from config import logger


def analyze_sentiment(sentiment_analyzer, text: str) -> str:
    """Analyze the sentiment of a given text using a sentiment analyzer.

    Args:
        sentiment_analyzer: The sentiment analysis model/function
        text (str): The text to analyze

    Returns:
        str: Sentiment label ("Positive", "Neutral", "Negative", or "Unknown")
    """
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
    """Attach sentiment analysis results to a list of articles.

    Args:
        sentiment_analyzer: The sentiment analysis model/function
        articles (List[Dict[str, Any]]): List of articles to analyze

    Returns:
        List[Dict[str, Any]]: List of articles with added sentiment field
    """
    processed_articles = []
    for article in articles:
        if not isinstance(article, dict):
            logger.debug(
                "Invalid article format; expected a dictionary. Skipping article.")
            continue

        # if missing or not a string, use a default sentiment
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

        new_article = article.copy()
        new_article["sentiment"] = sentiment
        processed_articles.append(new_article)
    return processed_articles


def merge_articles(articles: List[Dict[str, Any]], summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge article data with their corresponding summaries.

    Args:
        articles (List[Dict[str, Any]]): List of original articles
        summaries (List[Dict[str, Any]]): List of article summaries

    Returns:
        List[Dict[str, Any]]: List of merged articles containing both original data and summaries
    """
    if len(articles) != len(summaries):
        logger.debug(f"Warning: Length mismatch between articles ({len(articles)}) and summaries ({len(summaries)}). "
                     "Only merging common pairs.")

    merged = []
    for article, summary in zip(articles, summaries):
        if not isinstance(article, dict) or not isinstance(summary, dict):
            logger.debug(
                "Invalid format: both article and summary must be dictionaries. Skipping this pair.")
            continue
        merged_article = {**article, **summary}
        merged.append(merged_article)
    return merged


def get_sentiment_distribution(articles: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate the distribution of sentiments across a list of articles.

    Args:
        articles (List[Dict[str, Any]]): List of articles with sentiment fields

    Returns:
        Dict[str, int]: Dictionary mapping sentiment labels to their counts
    """
    sentiments = []
    for article in articles:
        if not isinstance(article, dict):
            logger.debug(
                "Invalid article format; expected a dictionary. Skipping article.")
            continue

        # default to 'Unknown' if missing or not a string.
        sentiment = article.get("sentiment", "Unknown")
        if not isinstance(sentiment, str):
            logger.debug(
                "Invalid sentiment type; expected a string. Using 'Unknown'.")
            sentiment = "Unknown"
        sentiments.append(sentiment)
    return dict(Counter(sentiments))
