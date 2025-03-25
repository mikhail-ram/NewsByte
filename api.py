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
    """
    Async context manager for FastAPI lifespan events.

    Sets up necessary resources at startup, including:
      - Ensuring NLTK data is downloaded to a specified directory.
      - Loading the spaCy English model.
      - Initializing the language model and sentiment analyzer, which are stored in the app's state.

    Parameters:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: The context manager does not yield a value, only initializes resources.
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
        "deepseek/deepseek-r1-distill-qwen-32b:free")  # deepseek/deepseek-r1:free, meta-llama/llama-3.2-3b-instruct: free
    app.state.sentiment_analyzer = pipeline(
        "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    yield

app = FastAPI(lifespan=lifespan)


class CompanyRequest(BaseModel):
    """
    Request model for fetching news articles.

    Attributes:
        company (str): The company name to fetch articles for.
        num_articles (int): The number of articles to fetch (default is 10).
    """
    company: str
    num_articles: int = 10


class ArticlesRequest(BaseModel):
    """
    Request model for endpoints requiring a list of articles.

    Attributes:
        articles (list): A list of articles.
    """
    articles: list


class FinalAnalysisRequest(BaseModel):
    """
    Request model for final sentiment analysis.

    Attributes:
        comp_score_dict (dict): Dictionary containing comparative sentiment scores.
        company (str): The company name associated with the analysis.
    """
    comp_score_dict: dict
    company: str


@app.post("/fetch_articles")
def fetch_articles_endpoint(req: CompanyRequest):
    """
    Fetch news articles based on the provided company name and number of articles.

    Parameters:
        req (CompanyRequest): Request object containing the company name and number of articles.

    Returns:
        list: A list of fetched articles.

    Raises:
        HTTPException: If an error occurs during article fetching.
    """
    try:
        articles = fetch_news_articles(req.company, req.num_articles)
        return articles
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize_articles")
def summarize_articles_endpoint(req: ArticlesRequest):
    """
    Summarize provided articles and merge the summaries with the original articles.

    Parameters:
        req (ArticlesRequest): Request object containing a list of articles.

    Returns:
        list: A list of merged article dictionaries that include the summaries.

    Raises:
        HTTPException: If an error occurs during summarization or merging.
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
    """
    Analyze sentiment for each article in the provided list.

    Utilizes a sentiment analyzer to attach a sentiment label to each article's summary.

    Parameters:
        req (ArticlesRequest): Request object containing a list of articles.

    Returns:
        list: A list of articles with an additional 'sentiment' key.

    Raises:
        HTTPException: If an error occurs during sentiment analysis.
    """
    try:
        articles_with_sentiment = attach_sentiment_to_articles(
            app.state.sentiment_analyzer, req.articles)
        return articles_with_sentiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_comparative_sentiment")
def comparative_sentiment_endpoint(req: ArticlesRequest):
    """
    Compute a comparative sentiment score for the provided articles.

    Uses the language model to extract a comparative sentiment score and augments
    the result with a distribution of sentiments from the articles.

    Parameters:
        req (ArticlesRequest): Request object containing a list of articles.

    Returns:
        dict: A dictionary containing the comparative sentiment score and sentiment distribution.

    Raises:
        HTTPException: If an error occurs during comparative sentiment analysis.
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
    """
    Perform a final sentiment analysis and produce a translated audio output.

    Utilizes the language model to generate a final sentiment analysis for the given company and sentiment score data.
    The analysis is then translated, converted to Hindi TTS, and saved as an audio file.

    Parameters:
        req (FinalAnalysisRequest): Request object containing a comparative score dictionary and company name.

    Returns:
        dict: A dictionary containing the final sentiment analysis, the translated version, and the path to the audio file.

    Raises:
        HTTPException: If an error occurs during final analysis, translation, or TTS conversion.
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
