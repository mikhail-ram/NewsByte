---
title: NewsByte
emoji: 📈
colorFrom: indigo
colorTo: yellow
sdk: docker
pinned: false
license: mit
short_description: Summarizes and analyzes company news with Hindi TTS.
---

# NewsByte

## Overview

NewsByte is a comprehensive sentiment analysis application designed to analyze news articles related to a specific company. It leverages web scraping, Large Language Models (LLMs), sentiment analysis, and text-to-speech (TTS) to provide a detailed and insightful sentiment summary. NewsByte offers both a command-line interface (CLI) and a user-friendly Streamlit web application.

## Project Structure

The project is organized into several modules, each with a specific responsibility:

- **`config.py`:** Centralized configuration for logging and debug mode.
- **`analysis.py`:** Performs sentiment analysis and data manipulation of news articles.
- **`tts.py`:** Handles text-to-speech (TTS) conversion, specifically for Hindi.
- **`llm.py`:** Manages interactions with the Large Language Model (LLM) for summarization and comparative analysis.
- **`api.py`:** Implements a FastAPI server exposing the NewsByte pipeline as a RESTful API.
- **`cli.py`:** Provides a command-line interface to run the NewsByte pipeline.
- **`utils.py`:** Contains utility functions for file handling and string manipulation.
- **`app.py`:** A Streamlit application providing a user interface to the API.
- **`data_models.py`:** Defines Pydantic data models for structured data handling.
- **`scraping.py`:** Handles web scraping of news articles using Google Search.

## Dependencies

The project relies on several Python libraries. Install them using pip:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include at least the following:

```
fastapi
uvicorn
pydantic
requests
beautifulsoup4
newspaper3k
trafilatura
langdetect
nltk
transformers
streamlit
streamlit-extras
kokoro
googletrans==4.0.0-rc1  # Important: Specify googletrans version due to potential conflicts
soundfile
numpy
argparse
outlines
openai
```

You will also need to set the `NEWSBYTE_API_KEY` environment variable with your OpenAI API key.

## Usage

### Command-Line Interface (CLI)

Run the analysis from the command line:

```bash
python cli.py <company_name> [number_of_articles]
```

Replace `<company_name>` with the name of the company you want to analyze. The `number_of_articles` is optional; it defaults to 10. The output is saved to a JSON file named after the company (in snake_case).

### Streamlit Web Application

Run the Streamlit application:

```bash
streamlit run app.py
```

And in another terminal:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

This launches a web application where you can interactively input a company name and step through the analysis process.

## Model Details

### Summarization Model

- **Model:** `deepseek/deepseek-r1-distill-qwen-32b` model used for summarization and topic extraction using OpenRouter API.
- **Implementation:** The `llm.py` module handles article summarization using prompts and the `outlines` module for structured JSON generation to extract key information from news articles.
- **Framework:** Uses the Outlines library for JSON schema validation and structured generation
- **Error Handling:** Implements a retry mechanism (up to 3 attempts) with JSON repair for malformed LLM outputs
- **Validation:** Uses Pydantic models defined in `data_models.py` to validate and structure LLM responses
- **Configuration:** Model used can be adjusted as a variable in `api.py`.

### Sentiment Analysis Model

- **Model:** `nlptown/bert-base-multilingual-uncased-sentiment` model from Hugging Face
- **Implementation:** The `analysis.py` module processes article text to determine sentiment polarity
- **Categories:** Sentiment is classified into four categories: "Positive", "Negative", "Neutral", and "Unknown"
- **Features:** Provides sentiment scores for positive, negative, neutral and unknown sentiments.
- **Pipeline:** Uses the Hugging Face `transformers` pipeline for inference

### Text-to-Speech (TTS) Model

- **Model:** `hexgrad/Kokoro-82M` used for open-source, efficient TTS in Hindi.
- **Implementation:** The `tts.py` module handles conversion of text to speech specifically optimized for Hindi.
- **Translation:** Uses the `googletrans` library (version 4.0.0-rc1) for English to Hindi translation
- **Output:** Generates audio files in `.wav` format that can be played in the web application or saved locally.

## API Development

### FastAPI Server (api.py)

The application exposes a RESTful API through a FastAPI server that follows the NewsByte pipeline steps.

#### Model Loading and Initialization

The API uses an `asynccontextmanager` to initialize:

- NLTK data with a specified download directory
- spaCy English language model
- LLM model via OpenRouter
- Sentiment analysis pipeline

#### API Endpoints

1. **Fetch Articles Endpoint**

   - **URL:** `/fetch_articles`
   - **Method:** POST
   - **Description:** Retrieves news articles about a company using web scraping
   - **Parameters:**
     - Company name
     - Number of articles (default: 10)

2. **Summarize Articles Endpoint**

   - **URL:** `/summarize_articles`
   - **Method:** POST
   - **Description:** Generates summaries and extracts topics using LLM
   - **Input:** List of articles
   - **Output:** Articles enhanced with summaries and topics

3. **Analyze Sentiment Endpoint**

   - **URL:** `/analyze_sentiment`
   - **Method:** POST
   - **Description:** Performs sentiment analysis on articles
   - **Input:** Articles with summaries
   - **Output:** Articles with sentiment labels ("Positive", "Negative", "Neutral", or "Unknown")

4. **Comparative Sentiment Endpoint**

   - **URL:** `/get_comparative_sentiment`
   - **Method:** POST
   - **Description:** Compares sentiment across articles using LLM
   - **Input:** Articles with sentiment scores
   - **Output:** Comparative analysis with coverage differences and topic overlap

5. **Final Analysis Endpoint**
   - **URL:** `/final_analysis`
   - **Method:** POST
   - **Description:** Generates final analysis, translates to Hindi, and creates TTS
   - **Input:** Comparative sentiment data and company name
   - **Output:** Final analysis in English and Hindi, plus audio file path

### Accessing the API

#### Using Postman

1. Launch Postman and create a new request
2. Set the request method to POST
3. Enter the appropriate URL (e.g., `http://localhost:8000/fetch_articles`)
4. Add a JSON request body as specified for each endpoint
5. Send the request and view the JSON response

#### Using cURL

```bash
curl -X POST "http://localhost:8000/fetch_articles" \
  -H "Content-Type: application/json" \
  -d '{"company":"Tesla","num_articles":5}'
```

#### Using Python Requests

```python
import requests
response = requests.post(
    "http://localhost:8000/fetch_articles",
    json={"company": "Apple", "num_articles": 5}
)
data = response.json()
```

## API Usage (Third-Party)

### OpenAI-Compatible API via OpenRouter

- **Purpose:** Powers the LLM-based summarization and sentiment analysis
- **Integration:** Accessed through the `llm.py` module using an AsyncOpenAI client
- **Configuration:**
  - Requires an API key set as the `NEWSBYTE_API_KEY` environment variable
  - Uses the OpenRouter API (base URL: `https://openrouter.ai/api/v1`)
- **Model Used:** `deepseek/deepseek-r1-distill-qwen-32b:free`
- **Framework:** Uses the Outlines library for structured generation with JSON validation

### Hugging Face Transformers

- **Purpose:** Sentiment analysis of article content
- **Model:** `nlptown/bert-base-multilingual-uncased-sentiment`
- **Integration:** Used via the transformers pipeline API

### Google Search (Implicit)

- **Purpose:** Used for scraping news articles
- **Integration:** Implemented in the `scraping.py` module using web scraping techniques
- **Usage Constraints:** Subject to Google's terms of service and rate limiting

### Googletrans API

- **Purpose:** Translates final analysis to Hindi before TTS conversion
- **Integration:** Used in the `tts.py` module

## Data Flow

The NewsByte pipeline follows these steps:

1. **Web Scraping (`scraping.py`):** Fetches news articles about the specified company from Google Search.
2. **Summarization and Topic Extraction (`llm.py`):** Uses an LLM to generate summaries and extract key topics from each article.
3. **Sentiment Analysis (`analysis.py`):** Performs sentiment analysis on the article summaries.
4. **Comparative Sentiment Analysis (`llm.py`):** Uses an LLM to compare sentiment across articles, identifying trends and differences.
5. **Final Sentiment Analysis (`llm.py`):** Generates a concise, investor-oriented summary of the overall sentiment.
6. **Translation and TTS (`tts.py`):** Translates the final analysis into Hindi and generates an audio file.
7. **Output (`cli.py`, `app.py`):** Presents the results via CLI or Streamlit web app, including the audio file. Results are also saved to a JSON file.

### Assumptions

- **Internet Connectivity:** The application assumes a stable internet connection for web scraping and API access
- **API Keys:** Users have valid OpenAI API keys with sufficient credits for OpenRouter access
- **News Availability:** Relevant and recent news articles are available for the specified company
- **Language Detection:** News articles are primarily in English; other languages may not be processed correctly
- **LLM Output Format:** The LLM is assumed to generate properly structured JSON responses, with a retry mechanism to handle exceptions
- **Processing Time:** Analysis may take several minutes depending on the number of articles and API response times

### Limitations

- **Scraping Reliability:** Web scraping may break if search engine layouts change
- **API Costs:** OpenRouter API usage may incur costs based on token usage and model selection
- **Rate Limiting:** Google Search, LLM APIs, and other services have rate limits that may affect the application
- **Sentiment Accuracy:** Sentiment analysis is limited to four categories ("Positive", "Negative", "Neutral", and "Unknown")
- **LLM Response Quality:** The quality of summarization and analysis depends on the specific LLM model used
- **Error Handling:** While there is a retry mechanism for LLM responses, persistent API failures will stop the pipeline
- **Hindi TTS Quality:** The Hindi TTS functionality may have pronunciation issues with certain words or phrases
- **Legal Considerations:** Web scraping must comply with terms of service of target websites
- **Real-time Updates:** The application does not support real-time monitoring of news sentiment

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

The MIT License is in use.

## Disclaimer

This project uses web scraping techniques, which are subject to the terms of service of the websites being scraped (primarily Google Search). Always respect the `robots.txt` file and avoid overloading servers. The accuracy of sentiment analysis and LLM outputs depends on the quality of the input data and the models used. Use the results responsibly and at your own discretion. The use of the OpenAI API incurs costs, and you are responsible for managing your API usage and costs. The Hindi TTS functionality relies on external libraries, and their accuracy and availability are not guaranteed.
