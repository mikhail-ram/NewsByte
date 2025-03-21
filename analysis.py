from collections import Counter
from typing import List, Dict, Any

from config import sentiment_analyzer, logger


def analyze_sentiment(text: str) -> str:
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


def attach_sentiment_to_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{**article, "sentiment": analyze_sentiment(article["summary"])} for article in articles]


def merge_articles(articles: List[Dict[str, Any]], summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{**article, **summary} for article, summary in zip(articles, summaries)]


def get_sentiment_distribution(articles: List[Dict[str, Any]]) -> Dict[str, int]:
    sentiments = [article['sentiment'] for article in articles]
    return dict(Counter(sentiments))
