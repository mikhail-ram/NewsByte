"""Web scraping module for NewsByte application.

This module handles fetching and processing news articles from various sources,
including Bing News RSS and article content extraction.
"""

import urllib.parse
import feedparser
from newspaper import Article
import trafilatura
from langdetect import detect
from typing import List, Dict, Any, Optional
import time

from config import logger

def build_search_url(company_name: str) -> str:
    """Build a Bing News RSS search URL for a given company.

    Args:
        company_name (str): Name of the company to search for

    Returns:
        str: Complete Bing News RSS search URL
    """
    encoded_company = urllib.parse.quote_plus(f"{company_name} news")
    url = f"https://www.bing.com/news/search?q={encoded_company}&format=rss&setlang=en-us&setmkt=en-us"
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
        article = Article(url, headers={'User-Agent': 'Mozilla/5.0'})
        article.download()
        article.parse()
        article_text = article.text
        if not article_text or len(article_text.split()) < 100:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                trafilatura_text = trafilatura.extract(downloaded)
                if trafilatura_text and len(trafilatura_text.split()) > len(article_text.split()):
                    article_text = trafilatura_text
        logger.debug(f"Extracted text from article at {url}")
        return {
            "text": article_text,
            "authors": article.authors,
            "publish_date": str(article.publish_date) if article.publish_date else None
        }
    except Exception as e:
        logger.error(f"Error extracting article details for URL {url}: {e}")
        return None

def process_candidate(entry: Any) -> Optional[Dict[str, Any]]:
    """Process an RSS entry into a complete article.

    Args:
        entry (Any): RSS feed entry to process

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
    details = extract_article_text(link)
    
    if details is None or not details["text"] or not is_english(details["text"]):
        logger.debug("Skipping candidate due to non-English text or empty content.")
        return None
        
    candidate.update(details)
    return candidate

def fetch_news_articles(company: str, num_articles: int) -> List[Dict[str, Any]]:
    """Fetch a specified number of news articles for a company.

    Args:
        company (str): Company name to search for
        num_articles (int): Number of articles to fetch

    Returns:
        List[Dict[str, Any]]: List of complete article data
    """
    search_url = build_search_url(company)
    logger.debug(f"Fetching RSS feed from: {search_url}")
    
    feed = feedparser.parse(search_url)
    articles = []
    
    for entry in feed.entries:
        if len(articles) >= num_articles:
            break
            
        candidate = process_candidate(entry)
        if candidate:
            articles.append(candidate)
            
    return articles
