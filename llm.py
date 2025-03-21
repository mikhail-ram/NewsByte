import os
import json
from typing import List, Dict, Any

from outlines import models, generate
from outlines.models.openai import OpenAIConfig
from openai import AsyncOpenAI

from data_models import ArticlesList, ComparativeSentimentScore


def create_model(model_name: str):
    client = AsyncOpenAI(
        api_key=os.environ.get("NEWSBYTE_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    config = OpenAIConfig(model_name)
    return models.openai(client, config)


def extract_articles_summary(model, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt = (
        "Below is a JSON array of articles, each containing a 'title' and 'text':\n"
        f"{json.dumps(articles, separators=(',', ':'))}\n\n"
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
    result = generator(prompt)
    return [article_summary.model_dump() for article_summary in result.root]


def extract_comparative_sentiment_score(model, articles: List[Dict[str, Any]]) -> ComparativeSentimentScore:
    filtered_articles = [
        {k: article[k] for k in ('title', 'sentiment',
                                 'topics', 'summary') if k in article}
        for article in articles
    ]
    prompt = (
        "Below is a JSON array of articles, each containing a 'title', 'sentiment', 'topics', and 'summary':\n"
        f"{json.dumps(filtered_articles, separators=(',', ':'))}\n\n"
        "Generate an object with two keys: 'Coverage Differences' and 'Topic Overlap'.\n\n"
        "The 'Coverage Differences' key should be an array where each element is an object with two keys:\n"
        "  'Comparison': A comparison of two articles based on their contents, referring to the articles by their number (e.g. 'Article 1').\n"
        "  'Impact': An analysis of the impact of the corresponding comparison on an investor.\n\n"
        "The 'Topic Overlap' key should be an object with exactly n + 1 keys:\n"
        "  'Common Topics': A list of common topics (strings) shared across articles.\n"
        "  'Unique Topics in Article 1', 'Unique Topics in Article 2', ..., 'Unique Topics in Article n':\n"
        "     Lists of topics (strings) that are unique to each respective article. \n\n"
        "     If the topic is not mentioned in the common topics list, it must be mentioned in the corresponding article here. \n\n"
        "Output must be plain text in the format given below.\n"
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
    generator = generate.json(model, ComparativeSentimentScore)
    result = generator(prompt)
    return result


def extract_final_sentiment_analysis(model, company: str, comparative_score: Dict[str, Any]) -> str:
    prompt = (
        f"Below is a JSON object of comparative sentiment scores for the {company} company.\n"
        "The object contains a 'Sentiment Distribution' which is a summary statistic of the sentiment ('Positive', 'Neutral', 'Negative' or 'Unknown') about the top news about the company.\n"
        "The 'Coverage Differences' is a list of JSON objects which analyzes the differences in news coverage. Each object here has a 'Comparison' key and an 'Impact' key.\n"
        "Finally, the object contains a 'Topic Overlap' key that includes a list of 'Common Topics' across articles and lists of topics unique to each article ('Unique Topics in Article n').\n"
        "Based on this information, can you provide a concise final sentiment analysis for an investor for the company that is justified. The analysis should not be longer than 4 sentences.\n"
        "Sentences should be short, with each sentence on a new line.\n"
        "Use plain text.\n"
        "Here are the comparative sentiment scores:\n\n"
        f"{json.dumps(comparative_score, separators=(',', ':'))}\n\n"
    )
    generator = generate.text(model)
    return generator(prompt)
