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
    return BeautifulSoup(html, "html.parser")


def extract_candidate_elements(soup: BeautifulSoup) -> List[Any]:
    elements = soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")
    logger.debug(f"Found {len(elements)} candidate elements.")
    return elements


def parse_candidate_element(element: Any) -> Optional[Dict[str, str]]:
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
    try:
        return detect(text) == "en"
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return False


def extract_article_text(url: str) -> Optional[Dict[str, Any]]:
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
    search_url = build_search_url(company, start)
    html = fetch_url_content(search_url, headers, retries=3, delay=1)
    soup = parse_html(html)
    elements = extract_candidate_elements(soup)
    return [candidate for candidate in (process_candidate(el) for el in elements) if candidate]


def fetch_news_articles(company: str, num_articles: int) -> List[Dict[str, Any]]:
    import time
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
