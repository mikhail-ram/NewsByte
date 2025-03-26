"""Streamlit web interface for NewsByte application.

This module provides a user-friendly web interface for the NewsByte application,
allowing users to:
1. Fetch news articles for a company
2. Generate article summaries and topics
3. Analyze sentiment
4. View comparative sentiment scores
5. Get final analysis with Hindi translation and audio
"""

import streamlit as st
from streamlit_extras.tags import tagger_component
import requests
import json

from utils import to_snake_case, to_title_case

# Constants
BASE_URL = "http://localhost:8000"

# UI Configuration
"""
Configure the Streamlit UI with custom CSS styles for consistent formatting
and improved readability of the application.
"""
st.markdown(
    """
    <style>
    a {
        text-decoration: none !important;
        color: inherit !important;
    }
    a:hover {
        text-decoration: underline !important;
        color: inherit !important;
    }
    .justified-text {
        text-align: justify;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Application Header
"""
Display the application title and description.
"""
st.title("NewsByte")
st.markdown(
    f'<div class="justified-text">NewsByte is a web app that fetches news about a company, summarizes articles, analyzes sentiment, compares trends, and generates a Hindi audio summary. Built with Python, FastAPI, and Streamlit.</div>',
    unsafe_allow_html=True
)
st.write("")

# Step 1: Article Fetching
"""
Handle the first step of the analysis pipeline:
1. Allow users to input a company name
2. Fetch raw articles from the API
3. Display articles with their metadata and content
"""
st.header("Step 1: Fetch Raw Articles")
company = st.text_input("Enter Company Name")

if st.button("Fetch Raw Articles"):
    payload = {"company": company, "num_articles": 10}
    response = requests.post(f"{BASE_URL}/fetch_articles", json=payload)
    if response.ok:
        articles = response.json()
        st.session_state.articles = articles
    else:
        st.error("Failed to fetch raw articles.")

if "articles" in st.session_state:
    for idx, article in enumerate(st.session_state.articles, start=1):
        with st.container(border=True):
            title = article.get("title", "No Title")
            link = article.get("link")
            if link:
                st.header(f"[{title}]({link})")
            else:
                st.header(title)

            st.write(f'Date: {article.get("publish_date", "Unknown")}')

            authors = article.get("authors", "Unknown")
            if isinstance(authors, list):
                authors = ", ".join(authors)
            st.caption(authors)

            st.markdown(
                f'<div class="justified-text">{article.get("text", "No text provided.")}</div>',
                unsafe_allow_html=True
            )
            st.write("")

# Step 2: Article Summarization
"""
Handle the second step of the analysis pipeline:
1. Generate summaries and topics for fetched articles
2. Display articles with their summaries and identified topics
"""
st.header("Step 2: Generate Summaries and Topics")
if st.button("Summarize Articles"):
    payload = {"articles": st.session_state.articles}
    response = requests.post(
        f"{BASE_URL}/summarize_articles", json=payload)
    if response.ok:
        summarized_articles = response.json()
        st.session_state.summarized_articles = summarized_articles
    else:
        st.error("Failed to generate summaries.")

if "summarized_articles" in st.session_state:
    for idx, article in enumerate(st.session_state.summarized_articles, start=1):
        with st.container(border=True):
            title = article.get("title", "No Title")
            link = article.get("link")
            if link:
                st.header(f"[{title}]({link})")
            else:
                st.header(title)

            topics = article.get("topics", "Unknown")
            if isinstance(topics, list):
                topics = ", ".join(to_title_case(topic)
                                   for topic in topics)
            st.caption(f"{topics}")
            st.markdown(
                f'<div class="justified-text">{article.get("summary", "No summary provided.")}</div>',
                unsafe_allow_html=True
            )
            st.write("")

# Step 3: Sentiment Analysis
"""
Handle the third step of the analysis pipeline:
1. Analyze sentiment for each article
2. Display articles with their sentiment scores and visual indicators
"""
st.header("Step 3: Attach Sentiment Analysis")
if st.button("Analyze Sentiment"):
    payload = {"articles": st.session_state.summarized_articles}
    response = requests.post(f"{BASE_URL}/analyze_sentiment", json=payload)
    if response.ok:
        articles_with_sentiment = response.json()
        st.session_state.articles_with_sentiment = articles_with_sentiment
    else:
        st.error("Failed to analyze sentiment.")

if "articles_with_sentiment" in st.session_state:
    for idx, article in enumerate(st.session_state.articles_with_sentiment, start=1):
        with st.container(border=True):
            title = article.get("title", "No Title")
            link = article.get("link")
            if link:
                st.header(f"[{title}]({link})")
            else:
                st.header(title)

            sentiment_to_color = {
                "Unknown": "blue", "Negative": "red", "Neutral": "yellow", "Positive": "green"}
            sentiment = article.get("sentiment", "Unknown")
            tagger_component(
                "",
                [sentiment],
                color_name=[sentiment_to_color[sentiment]],
            )

            topics = article.get("topics", "Unknown")
            if isinstance(topics, list):
                topics = ", ".join(to_title_case(topic)
                                   for topic in topics)
            st.caption(f"{topics}")

            st.markdown(
                f'<div class="justified-text">{article.get("summary", "No summary provided.")}</div>',
                unsafe_allow_html=True
            )
            st.write("")

# Step 4: Comparative Analysis
"""
Handle the fourth step of the analysis pipeline:
1. Generate comparative sentiment scores
2. Display coverage differences, topic overlap, and sentiment distribution
"""
st.header("Step 4: Comparative Sentiment Score")
if st.button("Get Comparative Sentiment Score"):
    payload = {"articles": st.session_state.articles_with_sentiment}
    response = requests.post(
        f"{BASE_URL}/get_comparative_sentiment", json=payload)
    if response.ok:
        comp_sentiment = response.json()
        st.session_state.comp_sentiment = comp_sentiment
    else:
        st.error("Failed to get comparative sentiment score.")


if "comp_sentiment" in st.session_state:
    with st.container(border=True):
        st.header("Coverage Differences")
        coverage_differences = st.session_state.comp_sentiment.get(
            "Coverage_Differences", [])
        if coverage_differences:
            for idx, diff in enumerate(coverage_differences, start=1):
                with st.container():
                    st.subheader(f"Difference {idx}")
                    st.markdown(
                        f'<div class="justified-text"><b>Comparison</b>: {diff.get("Comparison", "No comparison provided.")}</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div class="justified-text"><b>Impact</b>: {diff.get("Impact", "No impact provided.")}</div>',
                        unsafe_allow_html=True
                    )

                    if idx < len(coverage_differences):
                        st.write("---")
                    else:
                        st.write("")
        else:
            st.write("No Coverage Differences found.")

    with st.container(border=True):
        st.header("Topic Overlap")
        topic_overlap = st.session_state.comp_sentiment.get(
            "Topic_Overlap", {})
        common_topics = topic_overlap.get("Common_Topics", [])
        st.markdown(
            f'<div class="justified-text"><b>Common Topics</b>: {", ".join(common_topics) if common_topics else "None"}</div>',
            unsafe_allow_html=True
        )

        for key in topic_overlap:
            if key != "Common_Topics":
                topics = topic_overlap[key]
                topics_formatted = ", ".join(
                    to_title_case(topic) for topic in topics)
                st.markdown(
                    f'<div class="justified-text"><b>{key.replace("_", " ")}</b>: {topics_formatted}</div>',
                    unsafe_allow_html=True
                )
        st.write("")

    with st.container(border=True):
        st.header("Sentiment Distribution")
        sentiment_distribution = st.session_state.comp_sentiment.get(
            "Sentiment_Distribution", {})

        positive_count = sentiment_distribution.get("Positive", 0)
        negative_count = sentiment_distribution.get("Negative", 0)
        neutral_count = sentiment_distribution.get("Neutral", 0)
        unknown_count = sentiment_distribution.get("Unknown", 0)

        total = positive_count + negative_count
        ratio = positive_count / total if total > 0 else 0.5

        if ratio > 0.5:
            overall_color = "green"
        elif ratio < 0.5:
            overall_color = "red"
        else:
            overall_color = "yellow"
        display_text = f"Positive: {positive_count} | Neutral: {neutral_count} | Negative: {negative_count} | Unknown: {unknown_count}"
        tagger_component("", [display_text],
                         color_name=[overall_color])


# Step 5: Final Analysis and Audio Generation
"""
Handle the final step of the analysis pipeline:
1. Generate final sentiment analysis
2. Translate the analysis to Hindi
3. Generate audio from the translated text
4. Provide download options for the complete analysis
"""
st.header("Step 5: Final Analysis & TTS")
if st.button("Get Final Analysis"):
    payload = {
        "comp_score_dict": st.session_state.comp_sentiment, "company": company}
    response = requests.post(f"{BASE_URL}/final_analysis", json=payload)
    if response.ok:
        analysis = response.json()
        st.session_state.analysis = analysis
    else:
        st.error("Failed to get final analysis.")

if "analysis" in st.session_state:
    with st.container(border=True):
        st.header("Final Sentiment Analysis")
        st.markdown(
            f'<div class="justified-text">{st.session_state.analysis["Final_Sentiment_Analysis"]}</div>',
            unsafe_allow_html=True
        )
        st.write("")

    with st.container(border=True):
        st.header("Translated Final Sentiment Analysis")
        st.markdown(
            f'<div class="justified-text">{st.session_state.analysis["Translated_Final_Sentiment_Analysis"]}</div>',
            unsafe_allow_html=True
        )
        st.write("")
        audio_path = st.session_state.analysis.get("Audio")
        if audio_path:
            st.audio(audio_path)

    # Provide a download button for the final output JSON
    st.session_state.output = {"Company": company, "Articles": st.session_state.articles_with_sentiment,
                               "Comparative_Sentiment_Score": st.session_state.comp_sentiment, **st.session_state.analysis}

if "output" in st.session_state:
    with st.container(border=True):
        st.header("Complete Output")
        st.json(st.session_state.output)
        json_data = json.dumps(st.session_state.output,
                               indent=2, ensure_ascii=False)
        st.download_button("Download Final Output", data=json_data,
                           file_name=f"{to_snake_case(company)}_newsbyte.json", mime="application/json")
