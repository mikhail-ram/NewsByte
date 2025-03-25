from collections import Counter
from typing import List, Dict, Any

from config import logger


def analyze_sentiment(sentiment_analyzer, text: str) -> str:
    """
    Analyze the sentiment of a given text using the provided sentiment analyzer.

    Parameters:
        sentiment_analyzer: A callable that takes a string and returns a list of dictionaries with a 'label' key.
        text (str): The text to analyze.

    Returns:
        str: A string representing the sentiment ("Negative", "Neutral", "Positive", or "Unknown").
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
    """
    Attach a sentiment value to each article based on its summary.

    For each article in the list, analyze the sentiment of the 'summary' field using
    the provided sentiment analyzer, and add a new key 'sentiment' with the result.
    If the summary is missing or not a string, assign 'Unknown' as the sentiment.

    Parameters:
        sentiment_analyzer: A callable that takes a string and returns a sentiment analysis result.
        articles (List[Dict[str, Any]]): A list of article dictionaries.

    Returns:
        List[Dict[str, Any]]: A new list of articles with an added 'sentiment' key.
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
    """
    Merge two lists of dictionaries (articles and summaries) by combining corresponding pairs.

    If the lengths of the two lists differ, only merge pairs up to the length of the shorter list.
    Each merged dictionary contains keys and values from both the article and the summary.

    Parameters:
        articles (List[Dict[str, Any]]): A list of article dictionaries.
        summaries (List[Dict[str, Any]]): A list of summary dictionaries.

    Returns:
        List[Dict[str, Any]]: A list of merged dictionaries.
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
    """
    Compute the distribution of sentiment values across a list of articles.

    Iterate through the articles and count the occurrences of each sentiment value.
    If an article does not have a valid sentiment string, default to 'Unknown'.

    Parameters:
        articles (List[Dict[str, Any]]): A list of article dictionaries with a 'sentiment' key.

    Returns:
        Dict[str, int]: A dictionary mapping each sentiment to its count.
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
