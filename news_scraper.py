from openai import AsyncOpenAI
import os
from outlines import models, generate
from outlines.models.openai import OpenAIConfig
from pydantic import BaseModel, conlist, RootModel, Field, model_validator
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
import json
import urllib.parse
from newspaper import Article
from langdetect import detect
import logging
import time
from transformers import pipeline
import nltk
from collections import Counter
import spacy
import trafilatura

# -----------------------------------------------------------------------------
# NLTK & spaCy Setup
# -----------------------------------------------------------------------------
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)
nlp_spacy = spacy.load("en_core_web_sm")

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
logger = logging.getLogger("NewsByte")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
DEBUG_MODE = True
if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)
    ch.setLevel(logging.DEBUG)

# -----------------------------------------------------------------------------
# Pipelines Setup
# -----------------------------------------------------------------------------
sentiment_analyzer = pipeline(
    "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def build_search_url(company_name, start=0):
    query = f"{company_name} news"
    params = {
        "q": query,
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


def get_soup(url, headers=None):
    if headers is None:
        headers = {"User-Agent": "Mozilla/5.0"}
    logger.debug(f"Fetching URL: {url}")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    logger.debug(f"Received response with status code: {response.status_code}")
    return BeautifulSoup(response.text, "html.parser")


def extract_candidate_elements(soup):
    elements = soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")
    logger.debug(f"Found {len(elements)} candidate elements.")
    return elements


def parse_candidate_element(element):
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


def is_english(text):
    try:
        return detect(text) == "en"
    except Exception as e:
        logger.debug(f"Language detection failed: {e}")
        return False


def get_article_details(url):
    try:
        logger.debug(f"Fetching article details for URL: {url}")
        # Primary extraction using newspaper3k
        article = Article(url, headers={'User-Agent': 'Mozilla/5.0'})
        article.download()
        article.parse()
        article_text = article.text

        # Fallback extraction using trafilatura if the text is missing or too short
        if not article_text or len(article_text.split()) < 100:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                trafilatura_text = trafilatura.extract(downloaded)
                if trafilatura_text and len(trafilatura_text.split()) > len(article_text.split()):
                    article_text = trafilatura_text

        details = {
            "text": article_text,
            "authors": article.authors,
            "publish_date": str(article.publish_date) if article.publish_date else None
        }
        logger.debug(f"Extracted details for article: {article.title}")
        return details
    except Exception as e:
        logger.debug(f"Error extracting article details for URL {url}: {e}")
        return None


def analyze_sentiment(text):
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

# -----------------------------------------------------------------------------
# Fetching and Processing Articles (Steps 1-4)
# -----------------------------------------------------------------------------


def fetch_news_with_text(company_name, num_articles=10):
    news_data = []
    start = 0
    headers = {"User-Agent": "Mozilla/5.0"}
    while len(news_data) < num_articles:
        search_url = build_search_url(company_name, start)
        try:
            soup = get_soup(search_url, headers)
        except Exception as e:
            logger.debug(f"Error fetching search results: {e}")
            break

        candidate_elements = extract_candidate_elements(soup)
        if not candidate_elements:
            logger.debug("No candidate elements found, exiting loop.")
            break

        for element in candidate_elements:
            if len(news_data) >= num_articles:
                break

            candidate = parse_candidate_element(element)
            if candidate is None:
                continue

            if not is_english(candidate["title"]):
                logger.debug(
                    f"Skipping candidate with non-English title: {candidate['title']}")
                continue

            details = get_article_details(candidate["link"])
            if details is None:
                continue

            if not details["text"] or not is_english(details["text"]):
                logger.debug(
                    "Skipping article due to non-English text or empty content.")
                continue

            # sentiment = analyze_sentiment(details["text"])

            candidate.update(details)
            # candidate["sentiment"] = sentiment
            news_data.append(candidate)
            logger.debug(
                f"Added article: {candidate['title']} (Total: {len(news_data)}/{num_articles})")

        start += 10
        logger.debug(f"Moving to next page, start index now: {start}")
        time.sleep(1)
    return news_data

# -----------------------------------------------------------------------------
# Comparative Analysis (Step 5)
# -----------------------------------------------------------------------------


def get_sentiment_distribution(articles):
    sentiments = [article['sentiment'] for article in articles]
    sentiment_counts = dict(Counter(sentiments))
    return sentiment_counts


def to_snake_case(text):
    return text.lower().replace(" ", "_")


def save_news_to_json(company, final_output):
    filename = f"{to_snake_case(company)}_news.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    logger.debug(f"Saved final output to {filename}")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------


class ArticleSummary(BaseModel):
    topics: conlist(str, min_length=3, max_length=3)
    summary: str


def extract_topics_and_summary(model, articles):
    num_articles = len(articles)

    class ArticlesList(RootModel[conlist(ArticleSummary, min_length=num_articles, max_length=num_articles)]):
        pass

    # Construct the prompt for the LLM
    prompt = (
        "Below is a JSON array of articles, each containing a 'title' and 'text':\n"
        f"{json.dumps(articles, indent=2)}\n\n"
        "Generate an array where each element has two keys: "
        "'topics' and 'summary'. "
        "'topics' must be an array of exactly 3 relevant keywords, no more, no less, extracted from the article's 'text'. "
        "'summary' must be a concise summary of the article's 'text' (maximum 3 sentences). "
        "Ensure the output array has exactly the same number of elements as the input array. "
        "Output must be plain text, no JSON. "
        "Output format example:\n"
        '[ { "topics": ["keyword1", "keyword2", "keyword3"], "summary": "Short summary here." }, '
        '{ "topics": ["keywordA", "keywordB", "keywordC"], "summary": "Another summary here." } ]'
    )

    generator = generate.json(model, ArticlesList)

    logger.debug("Extracting articles' topics and summary.")

    result = generator(prompt)
    articles_summary = result.root

    return articles_summary


def create_model(model_name):
    client = AsyncOpenAI(
        api_key=os.environ.get("NEWSBYTE_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    config = OpenAIConfig(model_name)
    model = models.openai(client, config)

    return model


def merge(articles, articles_update):
    merged_articles = []
    for article, update in zip(articles, articles_update):
        merged_article = {**article, **update.model_dump()}
        merged_articles.append(merged_article)

    return merged_articles


class CoverageDifference(BaseModel):
    Comparison: str
    Impact: str


class TopicOverlap(BaseModel):
    Common_Topics: List[str] = Field(..., alias="Common Topics")

    class Config:
        extra = "allow"
        populate_by_name = True

    @model_validator(mode="before")
    def validate_exact_keys_and_types(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if "Common Topics" not in data:
            raise ValueError("Missing required key 'Common Topics'.")
        common_val = data["Common Topics"]
        if not isinstance(common_val, list) or not all(isinstance(x, str) for x in common_val):
            raise ValueError("'Common Topics' must be a list of strings.")

        # Extract unique keys (all keys other than "Common Topics")
        unique_keys = [k for k in data if k != "Common Topics"]
        if not unique_keys:
            raise ValueError(
                "There must be at least one unique topics key besides 'Common Topics'.")

        # Expect keys to be exactly: "Unique Topics in Article 1", "Unique Topics in Article 2", ..., "Unique Topics in Article n"
        expected_keys = [f"Unique Topics in Article {i}" for i in range(
            1, len(unique_keys) + 1)]
        if sorted(unique_keys) != sorted(expected_keys):
            raise ValueError(
                f"Unique topics keys must be exactly {expected_keys}. Got: {sorted(unique_keys)}"
            )
        if len(data) != len(unique_keys) + 1:
            raise ValueError(
                "There must be exactly n + 1 keys (1 'Common Topics' key and n unique keys).")

        for key in unique_keys:
            val = data[key]
            if not isinstance(val, list) or not all(isinstance(item, str) for item in val):
                raise ValueError(
                    f"Value for '{key}' must be a list of strings.")
        return data


class ComparativeSentimentScore(BaseModel):
    Coverage_Differences: List[CoverageDifference] = Field(
        ..., alias="Coverage Differences")
    Topic_Overlap: TopicOverlap = Field(..., alias="Topic Overlap")


def extract_comparative_sentiment_score(model, articles):
    filtered_articles = [
        {k: article[k]
            for k in ('title', 'sentiment', 'topics', 'summary') if k in article}
        for article in articles
    ]

    # --- Generate Comparative Sentiment Score ---
    prompt_comparative = (
        "Below is a JSON array of articles, each containing a 'title', 'sentiment', 'topics', and 'summary':\n"
        f"{json.dumps(filtered_articles, indent=2)}\n\n"
        "Generate an object with two keys: 'Coverage Differences' and 'Topic Overlap'.\n\n"
        "The 'Coverage Differences' key should be an array where each element is an object with two keys:\n"
        "  'Comparison': A comparison of two articles based on their contents, referring to the articles by their number (e.g. 'Article 1').\n"
        "  'Impact': An analysis of the impact of the corresponding comparison on an investor.\n\n"
        "The 'Topic Overlap' key should be an object with exactly n + 1 keys:\n"
        "  'Common Topics': A list of common topics (strings) shared across articles.\n"
        "  'Unique Topics in Article 1', 'Unique Topics in Article 2', ..., 'Unique Topics in Article n':\n"
        "     Lists of topics (strings) that are unique to each respective article. \n\n"
        "     If the topic is not mentioned in the common topics list, it must be mentioned in the corresponding article here. \n\n"
        "Output must be plain text, no JSON.\n"
        "Output format example:\n"
        '{\n'
        '  "Coverage Differences": [\n'
        "    {\n"
        '      "Comparison": "Article 1 highlights Tesla\'s strong sales, while Article 2 discusses regulatory issues.",\n'
        '      "Impact": "The first article boosts confidence in Tesla\'s market growth, while the second raises concerns about future regulatory hurdles."\n'
        "    },\n"
        "    {\n"
        '      "Comparison": "Article 1 is focused on financial success and innovation, whereas Article 2 is about legal challenges and risks.",\n'
        '      "Impact": "Investors may react positively to growth news but stay cautious due to regulatory scrutiny."\n'
        "    }\n"
        "  ],\n"
        '  "Topic Overlap": {\n'
        '    "Common Topics": ["Electric Vehicles"],\n'
        '    "Unique Topics in Article 1": ["Stock Market", "Innovation"],\n'
        '    "Unique Topics in Article 2": ["Regulations", "Autonomous Vehicles"]\n'
        "  }\n"
        "}"
    )

    generator_comparative = generate.json(model, ComparativeSentimentScore)
    result = generator_comparative(prompt_comparative)
    return result


def extract_final_sentiment_analysis(model, company, comparative_sentiment_score):
    prompt_final_analysis = (
        f"Below is a JSON object of comparative sentiment scores for the {company} company.\n"
        "The object contains a 'Sentiment Distribution' which is a summary statistic of the sentiment ('Positive', 'Neutral', 'Negative' or 'Unknown') about the top news about the company.\n"
        "The 'Coverage Differences' is a list of JSON objects which analyzes the differences in news coverage. Each object here has a 'Comparison' key and an 'Impact' key.\n"
        "Finally, the object contains a 'Topic Overlap' key that includes a list of 'Common Topics' across articles and lists of topics unique to each article ('Unique Topics in Article n').\n"
        "Based on this information, can you provide a concise final sentiment analysis for an investor for the company that is justified. The analysis should not be longer than 4 sentences.\n"
        "Use plain text.\n"
        "Here are the comparative sentiment scores:\n\n"
        f"{json.dumps(comparative_sentiment_score, indent=2)}\n\n"
    )

    generator_final_analysis = generate.text(model)
    final_analysis = generator_final_analysis(prompt_final_analysis)
    return final_analysis


if __name__ == "__main__":
    model = create_model("deepseek/deepseek-r1:free")

    company = "Tesla"
    logger.debug(f"Starting news fetch for {company}")
    articles = fetch_news_with_text(company, num_articles=10)
    logger.debug(f"Fetched {len(articles)} articles.")
    articles_summary = extract_topics_and_summary(model, articles)
    articles = merge(articles, articles_summary)
    logger.debug("Extracting sentiment for each article.")
    articles = [
        {**article, "sentiment": analyze_sentiment(article["summary"])} for article in articles]

    logger.debug("Extracting comparative sentiment score.")
    comparative_sentiment_score = extract_comparative_sentiment_score(
        model, articles)
    comparative_sentiment_score = comparative_sentiment_score.model_dump()
    comparative_sentiment_score["Sentiment Distribution"] = get_sentiment_distribution(
        articles)

    logger.debug("Generating final sentiment analysis.")
    final_analysis = extract_final_sentiment_analysis(
        model, company, comparative_sentiment_score)

    logger.debug("Building final output.")
    output = {}
    output["Company"] = company
    output["Articles"] = articles
    output["Comparative Sentiment Score"] = comparative_sentiment_score
    output["Final Sentiment Analysis"] = final_analysis
    save_news_to_json(output["Company"], output)
