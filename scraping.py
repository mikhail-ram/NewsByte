import urllib.parse
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import trafilatura
from langdetect import detect
from typing import List, Dict, Any, Optional
import time

from config import logger


def build_search_url(company_name: str, start: int = 0) -> str:
    """
    Build and return a Google News search URL for the specified company name.

    Constructs the URL using query parameters for news search in English starting from the given index.

    Parameters:
        company_name (str): The name of the company to search news for.
        start (int, optional): The starting index for the search results. Defaults to 0.

    Returns:
        str: The complete URL for the news search.
    """
    params = {
        "q": f"{company_name} news",
        "tbm": "nws",
        "hl": "en",
        "lr": "lang_en",
        "tbs": "lr:lang_1en",
        "start": start
    }
    base_url = "https://www.google.com/search"
    param_str = "&".join(
        [f"{k}={urllib.parse.quote_plus(str(v))}" for k, v in params.items()])
    url = f"{base_url}?{param_str}"
    logger.debug(f"Built search URL: {url}")
    return url


def fetch_url_content(url: str, headers: Optional[Dict[str, str]] = None, retries: int = 3, delay: int = 1) -> str:
    """
    Fetch the content of a URL with retries and return its text.

    Makes an HTTP GET request to the given URL with optional headers and retries on failure.
    Delays between retries are controlled by the delay parameter.

    Parameters:
        url (str): The URL to fetch.
        headers (Optional[Dict[str, str]]): HTTP headers to include in the request. Defaults to a standard User-Agent.
        retries (int, optional): The number of retry attempts in case of failure. Defaults to 3.
        delay (int, optional): Delay in seconds between retries. Defaults to 1.

    Returns:
        str: The text content of the response.

    Raises:
        Exception: If all retry attempts fail.
    """
    if headers is None:
        headers = {"User-Agent": "Mozilla/5.0"}
    logger.debug(f"Fetching URL: {url}")
    last_exception = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            logger.debug(
                f"Received response with status code: {response.status_code}")
            return response.text
        except requests.exceptions.RequestException as e:
            last_exception = e
            logger.error(f"Attempt {attempt + 1} failed with error: {e}")
            time.sleep(delay)
    # After all retries fail, raise the last encountered exception.
    raise Exception(
        f"Failed to fetch URL content after {retries} attempts. Last error: {last_exception}")


def parse_html(html: str) -> BeautifulSoup:
    """
    Parse HTML content and return a BeautifulSoup object.

    Parameters:
        html (str): The HTML content to parse.

    Returns:
        BeautifulSoup: A BeautifulSoup object representing the parsed HTML.
    """
    return BeautifulSoup(html, "html.parser")


def extract_candidate_elements(soup: BeautifulSoup) -> List[Any]:
    """
    Extract candidate elements containing news article links from the parsed HTML.

    Searches for specific div elements with designated classes that likely contain article titles and links.

    Parameters:
        soup (BeautifulSoup): The parsed HTML content.

    Returns:
        List[Any]: A list of candidate elements found in the HTML.
    """
    elements = soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")
    logger.debug(f"Found {len(elements)} candidate elements.")
    return elements


def parse_candidate_element(element: Any) -> Optional[Dict[str, str]]:
    """
    Parse a candidate HTML element to extract the article title and link.

    Checks for the presence of a parent anchor tag and extracts the URL query parameter 'q'.

    Parameters:
        element (Any): The HTML element to parse.

    Returns:
        Optional[Dict[str, str]]: A dictionary with 'title' and 'link' keys if valid, otherwise None.
    """
    title = element.get_text().strip()
    parent_a = element.find_parent("a")
    if not parent_a or "href" not in parent_a.attrs:
        logger.debug("Skipping element without valid link.")
        return None
    raw_link = parent_a["href"]
    parsed_link = urllib.parse.parse_qs(
        urllib.parse.urlparse(raw_link).query).get("q", [None])[0]
    if not parsed_link:
        logger.debug("Parsed link is None.")
        return None
    logger.debug(f"Parsed candidate - Title: {title} | Link: {parsed_link}")
    return {"title": title, "link": parsed_link}


def is_english(text: str) -> bool:
    """
    Determine if the given text is in English.

    Uses the langdetect library to detect the language of the text.

    Parameters:
        text (str): The text to analyze.

    Returns:
        bool: True if the text is detected as English, False otherwise.
    """
    try:
        return detect(text) == "en"
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return False


def extract_article_text(url: str) -> Optional[Dict[str, Any]]:
    """
    Extract article text, authors, and publish date from a given URL.

    Attempts to extract the article using the newspaper library. If the text is too short,
    it uses trafilatura to fetch and extract the text again. Logs and returns None if extraction fails.

    Parameters:
        url (str): The URL of the article.

    Returns:
        Optional[Dict[str, Any]]: A dictionary with keys 'text', 'authors', and 'publish_date'
                                  if extraction is successful; otherwise, None.
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


def process_candidate(element: Any) -> Optional[Dict[str, Any]]:
    """
    Process a candidate HTML element to extract article details.

    Parses the candidate element for a title and link, checks if the title is in English,
    extracts article details, and verifies that the article text is in English and non-empty.

    Parameters:
        element (Any): The HTML element representing a candidate article.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing article details if successful; otherwise, None.
    """
    candidate = parse_candidate_element(element)
    if candidate is None or not is_english(candidate["title"]):
        return None
    details = extract_article_text(candidate["link"])
    if details is None or not details["text"] or not is_english(details["text"]):
        logger.debug(
            "Skipping candidate due to non-English text or empty content.")
        return None
    candidate.update(details)
    return candidate


def fetch_candidate_articles(company: str, start: int, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Fetch candidate articles for a given company starting from a specific result index.

    Builds the search URL, fetches its HTML content, parses it, extracts candidate elements,
    and processes each candidate element to obtain article details.

    Parameters:
        company (str): The company name to search articles for.
        start (int): The starting index for search results.
        headers (Dict[str, str]): HTTP headers to use when fetching the URL.

    Returns:
        List[Dict[str, Any]]: A list of candidate articles with extracted details.
    """
    search_url = build_search_url(company, start)
    html = fetch_url_content(search_url, headers, retries=3, delay=1)
    soup = parse_html(html)
    elements = extract_candidate_elements(soup)
    return [candidate for candidate in (process_candidate(el) for el in elements) if candidate]


def fetch_news_articles(company: str, num_articles: int) -> List[Dict[str, Any]]:
    """
    Fetch a specified number of news articles for a given company.

    Iteratively fetches candidate articles from search results until the desired number
    of articles is collected or no more candidates are found. Introduces delays between requests.

    Parameters:
        company (str): The company name to search articles for.
        num_articles (int): The number of articles to fetch.

    Returns:
        List[Dict[str, Any]]: A list of fetched news articles with their details.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    articles = []
    start = 0
    while len(articles) < num_articles:
        candidates = fetch_candidate_articles(company, start, headers)
        for candidate in candidates:
            if len(articles) >= num_articles:
                break
            articles.append(candidate)
        if not candidates:
            break
        start += 10
        time.sleep(1)
    return articles
