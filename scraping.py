"""Web scraping module for NewsByte application.

This module handles fetching and processing news articles from various sources,
including Yahoo Finance API and article content extraction.
"""

import urllib.parse
import requests
from newspaper import Article
import trafilatura
from langdetect import detect
from typing import List, Dict, Any, Optional
import time

from config import logger


def build_search_url(company_name: str, num_articles: int = 10) -> str:
    """Build a Yahoo Finance Search API URL for a given company.

    Args:
        company_name (str): Name of the company to search for
        num_articles (int): Number of news articles to request

    Returns:
        str: Complete Yahoo Finance Search API URL
    """
    encoded_company = urllib.parse.quote_plus(company_name)
    # Request a few extra articles to account for non-English or un-scrapeable links
    news_count = min(num_articles + 10, 30)
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={encoded_company}&newsCount={news_count}"
    logger.debug(f"Built search URL: {url}")
    return url


def is_english(text: str) -> bool:
    """Check if text is in English.

    Args:
        text (str): Text to check

    Returns:
        bool: True if text is in English, False otherwise
    """
    try:
        return detect(text) == "en"
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return False


def extract_article_text(url: str) -> Optional[Dict[str, Any]]:
    """Extract article text and metadata from a URL.

    Args:
        url (str): URL of the article

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing article text, authors, and publish date, or None if extraction fails
    """
    try:
        article_text = ""
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(downloaded)
            if extracted:
                article_text = extracted

        if not article_text or len(article_text.split()) < 100:
            article = Article(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
            article.download()
            article.parse()
            if article.text and len(article.text.split()) > len(article_text.split()):
                article_text = article.text

        logger.debug(f"Extracted text from article at {url}")
        
        # We no longer rely on newspaper3k for authors/date because it fails on Yahoo pages.
        # These will be populated from the API JSON instead.
        return {
            "text": article_text,
        }
    except Exception as e:
        logger.error(f"Error extracting article details for URL {url}: {e}")
        return None


def process_candidate(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a Yahoo Finance news entry into a complete article.

    Args:
        entry (Dict[str, Any]): News entry from Yahoo Finance API

    Returns:
        Optional[Dict[str, Any]]: Complete article data or None if processing fails
    """
    title = entry.get("title", "").strip()
    link = entry.get("link", "").strip()
    
    if not title or not link or not is_english(title):
        logger.debug("Skipping candidate due to missing title/link or non-English title.")
        return None
        
    logger.debug(f"Processing candidate - Title: {title} | Link: {link}")
    
    candidate = {"title": title, "link": link}
    
    # Grab publisher and format the unix timestamp
    publisher = entry.get("publisher", "Unknown")
    candidate["authors"] = [publisher]
    
    pub_time = entry.get("providerPublishTime")
    if pub_time:
        import datetime
        candidate["publish_date"] = datetime.datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M:%S')
    else:
        candidate["publish_date"] = "Unknown"
        
    details = extract_article_text(link)
    
    if details is None or not details.get("text") or not is_english(details["text"]):
        logger.debug("Skipping candidate due to non-English text or empty content.")
        return None
        
    candidate["text"] = details["text"]
    return candidate


def fetch_news_articles(company: str, num_articles: int) -> List[Dict[str, Any]]:
    """Fetch a specified number of news articles for a company.

    Args:
        company (str): Company name to search for
        num_articles (int): Number of articles to fetch

    Returns:
        List[Dict[str, Any]]: List of complete article data
    """
    search_url = build_search_url(company, num_articles)
    logger.debug(f"Fetching news from Yahoo Finance API: {search_url}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        news_items = data.get("news", [])
    except Exception as e:
        logger.error(f"Failed to fetch from Yahoo Finance API: {e}")
        return []

    articles = []
    
    for entry in news_items:
        if len(articles) >= num_articles:
            break
            
        candidate = process_candidate(entry)
        if candidate:
            articles.append(candidate)
            
    return articles
