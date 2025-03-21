from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from transformers import pipeline

from main import (
    create_model,
    fetch_news_articles,
    extract_articles_summary,
    merge_articles,
    attach_sentiment_to_articles,
    extract_comparative_sentiment_score,
    get_sentiment_distribution,
    extract_final_sentiment_analysis,
    translate_text,
    hindi_tts,
)

app = FastAPI()

# Initialize the model at startup


@app.on_event("startup")
async def startup_event():
    app.state.model = create_model("deepseek/deepseek-r1:free")
    app.state.sentiment_analyzer = pipeline(
        "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Request models for various endpoints


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
        comp_score_dict["Sentiment Distribution"] = get_sentiment_distribution(
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
            "Final Sentiment Analysis": final_analysis,
            "Translated Final Sentiment Analysis": translated_final_analysis,
            "Audio": output_tts_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
