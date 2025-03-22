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

    nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    # Add this directory to nltk's data path
    nltk.data.path.append(nltk_data_dir)
    # Download the required model to the new directory
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)
    nlp_spacy = spacy.load("en_core_web_sm")  # might be optional

    app.state.model = create_model("deepseek/deepseek-r1:free")
    app.state.sentiment_analyzer = pipeline(
        "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    yield

app = FastAPI(lifespan=lifespan)


class CompanyRequest(BaseModel):
    company: str
    num_articles: int = 10


class ArticlesRequest(BaseModel):
    articles: list


class FinalAnalysisRequest(BaseModel):
    comp_score_dict: dict
    company: str


@app.post("/fetch_articles")
def fetch_articles_endpoint(req: CompanyRequest):
    try:
        articles = fetch_news_articles(req.company, req.num_articles)
        return articles
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize_articles")
def summarize_articles_endpoint(req: ArticlesRequest):
    try:
        model = app.state.model
        articles_summary = extract_articles_summary(model, req.articles)
        merged_articles = merge_articles(req.articles, articles_summary)
        return merged_articles
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_sentiment")
def analyze_sentiment_endpoint(req: ArticlesRequest):
    try:
        articles_with_sentiment = attach_sentiment_to_articles(
            app.state.sentiment_analyzer, req.articles)
        return articles_with_sentiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_comparative_sentiment")
def comparative_sentiment_endpoint(req: ArticlesRequest):
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


'''
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
