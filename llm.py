from pydantic import ValidationError
import os
import json
import json_repair
from typing import List, Dict, Any

from outlines import models, generate
from outlines.models.openai import OpenAIConfig
from openai import AsyncOpenAI

from data_models import ArticlesList, ComparativeSentimentScore
from config import logger


def retry_prompt(generator_func, prompt: str, schema_type=None, retries: int = 3):
    last_exception = None

    for attempt in range(retries):
        try:
            return generator_func(prompt)
        except ValidationError as e:
            last_exception = e
            logger.error(f"Attempt {attempt + 1} failed with error: {e}")

            # Check if the error is related to JSONDecodeError
            errors = e.errors()
            is_json_decode_error = any(
                error.get('type') == 'value_error.jsondecode' for error in errors
            )

            if is_json_decode_error:
                # Get the input value from the first error with this type
                for error in errors:
                    if error.get('type') == 'value_error.jsondecode':
                        raw_text = error.get('input')

                        # If raw_text is found and contains backticks
                        if raw_text and "```" in raw_text:
                            # Clean the JSON text by removing markdown code formatting
                            cleaned_text = raw_text.replace(
                                "```json", "").replace("```", "").strip()
                            try:
                                # Parse manually to check if it's valid JSON
                                parsed_json = json_repair.loads(cleaned_text)
                                if schema_type:
                                    return schema_type.model_validate(parsed_json)
                                else:
                                    return parsed_json
                            except Exception as e:
                                logger.error(
                                    f"Failed to parse cleaned JSON:\n{cleaned_text}\nGetting error:\n{e}")

                        break  # Only process the first error
        except Exception as e:
            last_exception = e
            logger.error(f"Attempt {attempt + 1} failed with error: {e}")

    raise Exception(
        f"LLM prompt failed after {retries} attempts. Last error: {last_exception}")


def create_model(model_name: str):
    client = AsyncOpenAI(
        api_key=os.environ.get("NEWSBYTE_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    config = OpenAIConfig(model_name)
    return models.openai(client, config)


def extract_articles_summary(model, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt = (
        "You are an article summarizer tasked with extracting topics and summaries out of articles given to you.\n"
        "You can only communicate in pure, valid JSON. Do not use markdown code blocks, output just the JSON.\n"
        "Below is a JSON array of articles, each containing a 'title' and 'text':\n"
        f"{json.dumps(articles, separators=(',', ':'))}\n\n"
        "Generate an array where each element has two keys: "
        "'topics' and 'summary'. "
        "'topics' must be an array of exactly 3 relevant keywords, no more, no less, extracted from the article's 'text'. "
        "'summary' must be a concise summary of the article's 'text' (maximum 3 sentences). "
        "Ensure the output array has exactly the same number of elements as the input array. "
        "Provide only the valid JSON output in the format given below. "
        "Output format example:\n"
        '[ { "topics": ["keyword1", "keyword2", "keyword3"], "summary": "Short summary here." }, '
        '{ "topics": ["keywordA", "keywordB", "keywordC"], "summary": "Another summary here." } ]'
    )
    generator = generate.json(model, ArticlesList)
    result = retry_prompt(generator, prompt, ArticlesList, 3)
    return [article_summary.model_dump() for article_summary in result.root]


def extract_comparative_sentiment_score(model, articles: List[Dict[str, Any]]) -> ComparativeSentimentScore:
    filtered_articles = [
        {k: article[k] for k in ('title', 'sentiment',
                                 'topics', 'summary') if k in article}
        for article in articles
    ]
    prompt = (
        "You are an article analyzer tasked with creating a comparative analysis out of articles given to you.\n"
        "You can only communicate in pure, valid JSON. Do not use markdown code blocks, output just the JSON.\n"
        "Below is a JSON array of articles, each containing a 'title', 'sentiment', 'topics', and 'summary':\n"
        f"{json.dumps(filtered_articles, separators=(',', ':'))}\n\n"
        "Generate an JSON object with two keys: 'Coverage_Differences' and 'Topic_Overlap'.\n\n"
        "The 'Coverage_Differences' key should be an array where each element is an object with two keys:\n"
        "  'Comparison': A comparison of two articles based on their contents, referring to the articles by their number (e.g. 'Article 1').\n"
        "  'Impact': An analysis of the impact of the corresponding comparison on an investor.\n\n"
        f"The 'Topic_Overlap' key should be an object with {len(filtered_articles) + 1} keys:\n"
        "  'Common_Topics': A list of common topics (strings) shared across articles.\n"
        f"  'Unique_Topics_in_Article_1', ..., 'Unique_Topics_in_Article_{len(filtered_articles)}':\n"
        "     Lists of topics (strings) that are unique to each respective article from the given 'topics' key for that article but not in the 'Common_Topics' key. \n\n"
        "Provide only the valid JSON output in the format given below. "
        "Output format example:\n"
        '{\n'
        '  "Coverage_Differences": [\n'
        "    {\n"
        '      "Comparison": "Article 1 highlights Tesla\'s strong sales, while Article 2 discusses regulatory issues.",\n'
        '      "Impact": "The first article boosts confidence in Tesla\'s market growth, while the second raises concerns about future regulatory hurdles."\n'
        "    },\n"
        "    {\n"
        '      "Comparison": "Article 1 is focused on financial success and innovation, whereas Article 2 is about legal challenges and risks.",\n'
        '      "Impact": "Investors may react positively to growth news but stay cautious due to regulatory scrutiny."\n'
        "    }\n"
        "  ],\n"
        '  "Topic_Overlap": {\n'
        '    "Common_Topics": ["Electric Vehicles"],\n'
        '    "Unique_Topics_in_Article_1": ["Stock Market", "Innovation"],\n'
        '    "Unique_Topics_in_Article_2": ["Regulations", "Autonomous Vehicles"]\n'
        "  }\n"
        "}"
    )
    generator = generate.json(model, ComparativeSentimentScore)
    result = retry_prompt(generator, prompt, ComparativeSentimentScore, 3)
    return result


def extract_final_sentiment_analysis(model, company: str, comparative_score: Dict[str, Any]) -> str:
    prompt = (
        f"You are a market analyst tasked with generating a summarized company analysis for investors out of a comparative analysis for the {company} company given to you by your team. You can only write short sentences with each sentence appearing on a new line.\n"
        f"Below is a JSON object of comparative sentiment scores for the {company} company.\n"
        "The object contains a 'Sentiment_Distribution' which is a summary statistic of the sentiment ('Positive', 'Neutral', 'Negative' or 'Unknown') about the top news about the company.\n"
        "The 'Coverage_Differences' is a list of JSON objects which analyzes the differences in news coverage. Each object here has a 'Comparison' key and an 'Impact' key.\n"
        "Finally, the object contains a 'Topic_Overlap' key that includes a list of 'Common_Topics' across articles and lists of topics unique to each article ('Unique_Topics_in_Article_n').\n"
        "Based on this information, can you provide a concise final sentiment analysis for an investor for the company that is justified. The analysis should not be longer than 4 sentences.\n"
        "Here are the comparative sentiment scores:\n\n"
        f"{json.dumps(comparative_score, separators=(',', ':'))}\n\n"
    )
    generator = generate.text(model)
    result = retry_prompt(generator, prompt, 3)
    return result
