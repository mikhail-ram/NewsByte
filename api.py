import nltk
import spacy
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from transformers import pipeline
from contextlib import asynccontextmanager
import asyncio
from transformers import pipeline

from scraping import fetch_news_articles
from analysis import attach_sentiment_to_articles, merge_articles, get_sentiment_distribution
from llm import create_model, extract_articles_summary, extract_comparative_sentiment_score, extract_final_sentiment_analysis
from tts import translate_text, hindi_tts

# Initialize the model at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and manage application resources during startup and shutdown.

    Args:
        app (FastAPI): The FastAPI application instance

    Yields:
        None: After initialization is complete
    """
    nltk_data_dir = "/app/data/nltk_data"
    # Ensure the directory exists
    os.makedirs(nltk_data_dir, exist_ok=True)
    # Add this directory to nltk's data path
    nltk.data.path.append(nltk_data_dir)
    # Explicitly specify download location for each resource
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('averaged_perceptron_tagger',
                  download_dir=nltk_data_dir, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    nlp_spacy = spacy.load("en_core_web_sm")  # might be optional

    app.state.model = create_model(
        "deepseek/deepseek-r1-distill-qwen-32b:free")
    # qwen/qwq-32b:free, deepseek/deepseek-r1:free, meta-llama/llama-3.2-3b-instruct: free
    app.state.sentiment_analyzer = pipeline(
        "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    yield

app = FastAPI(lifespan=lifespan)


class CompanyRequest(BaseModel):
    """Request model for fetching company articles.

    Attributes:
        company (str): Name of the company to fetch articles for
        num_articles (int, optional): Number of articles to fetch. Defaults to 10.
    """
    company: str
    num_articles: int = 10


class ArticlesRequest(BaseModel):
    """Request model for article processing endpoints.

    Attributes:
        articles (list): List of articles to process
    """
    articles: list


class FinalAnalysisRequest(BaseModel):
    """Request model for final analysis endpoint.

    Attributes:
        comp_score_dict (dict): Comparative sentiment scores
        company (str): Name of the company being analyzed
    """
    comp_score_dict: dict
    company: str


@app.post("/fetch_articles")
def fetch_articles_endpoint(req: CompanyRequest):
    """Fetch news articles for a specified company.

    Args:
        req (CompanyRequest): Request containing company name and number of articles

    Returns:
        list: List of fetched articles

    Raises:
        HTTPException: If article fetching fails
    """
    try:
        articles = fetch_news_articles(req.company, req.num_articles)
        return articles
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize_articles")
def summarize_articles_endpoint(req: ArticlesRequest):
    """Generate summaries for a list of articles.

    Args:
        req (ArticlesRequest): Request containing articles to summarize

    Returns:
        list: List of articles with added summaries

    Raises:
        HTTPException: If summarization fails
    """
    try:
        model = app.state.model
        articles_summary = extract_articles_summary(model, req.articles)
        merged_articles = merge_articles(req.articles, articles_summary)
        return merged_articles
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_sentiment")
def analyze_sentiment_endpoint(req: ArticlesRequest):
    """Analyze sentiment for a list of articles.

    Args:
        req (ArticlesRequest): Request containing articles to analyze

    Returns:
        list: List of articles with added sentiment analysis

    Raises:
        HTTPException: If sentiment analysis fails
    """
    try:
        articles_with_sentiment = attach_sentiment_to_articles(
            app.state.sentiment_analyzer, req.articles)
        return articles_with_sentiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_comparative_sentiment")
def comparative_sentiment_endpoint(req: ArticlesRequest):
    """Generate comparative sentiment analysis for articles.

    Args:
        req (ArticlesRequest): Request containing articles to analyze

    Returns:
        dict: Comparative sentiment scores and distribution

    Raises:
        HTTPException: If analysis fails
    """
    try:
        model = app.state.model
        comp_score = extract_comparative_sentiment_score(
            model, req.articles)
        comp_score_dict = comp_score.model_dump()
        comp_score_dict["Sentiment_Distribution"] = get_sentiment_distribution(
            req.articles)
        return comp_score_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/final_analysis")
def final_analysis_endpoint(req: FinalAnalysisRequest):
    """Generate final sentiment analysis with translation and audio.

    Args:
        req (FinalAnalysisRequest): Request containing company and comparative scores

    Returns:
        dict: Contains final analysis, translated analysis, and audio file path

    Raises:
        HTTPException: If analysis or translation fails
    """
    try:
        model = app.state.model
        final_analysis = extract_final_sentiment_analysis(
            model, req.company, req.comp_score_dict)
        translated_final_analysis = asyncio.run(translate_text(final_analysis))
        output_tts_path = "hindi_tts.wav"
        hindi_tts(translated_final_analysis, output_tts_path)
        return {
            "Final_Sentiment_Analysis": final_analysis,
            "Translated_Final_Sentiment_Analysis": translated_final_analysis,
            "Audio": output_tts_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
