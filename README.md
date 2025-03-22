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

## Data Flow

The NewsByte pipeline follows these steps:

1. **Web Scraping (`scraping.py`):** Fetches news articles about the specified company from Google Search.
2. **Summarization and Topic Extraction (`llm.py`):** Uses an LLM to generate summaries and extract key topics from each article.
3. **Sentiment Analysis (`analysis.py`):** Performs sentiment analysis on the article summaries.
4. **Comparative Sentiment Analysis (`llm.py`):** Uses an LLM to compare sentiment across articles, identifying trends and differences.
5. **Final Sentiment Analysis (`llm.py`):** Generates a concise, investor-oriented summary of the overall sentiment.
6. **Translation and TTS (`tts.py`):** Translates the final analysis into Hindi and generates an audio file.
7. **Output (`cli.py`, `app.py`):** Presents the results via CLI or Streamlit web app, including the audio file. Results are also saved to a JSON file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

The MIT License is in use.

## Disclaimer

This project uses web scraping techniques, which are subject to the terms of service of the websites being scraped (primarily Google Search). Always respect the `robots.txt` file and avoid overloading servers. The accuracy of sentiment analysis and LLM outputs depends on the quality of the input data and the models used. Use the results responsibly and at your own discretion. The use of the OpenAI API incurs costs, and you are responsible for managing your API usage and costs. The Hindi TTS functionality relies on external libraries, and their accuracy and availability are not guaranteed.
