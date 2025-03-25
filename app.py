import streamlit as st
from streamlit_extras.tags import tagger_component
import requests
import json

from utils import to_snake_case, to_title_case

BASE_URL = "http://localhost:8000"


def set_page_style() -> None:
    """
    Set the page style, title, and description using Streamlit markdown.
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
    st.title("NewsByte")
    st.markdown(
        f'<div class="justified-text">NewsByte is a web app that fetches news about a company, summarizes articles, analyzes sentiment, compares trends, and generates a Hindi audio summary. Built with Python, FastAPI, and Streamlit.</div>',
        unsafe_allow_html=True
    )
    st.write("")


def fetch_raw_articles() -> None:
    """
    Fetch raw articles for the given company when the 'Fetch Raw Articles' button is clicked.
    Stores the fetched articles in the Streamlit session state.
    """
    st.header("Step 1: Fetch Raw Articles")
    company = st.text_input("Enter Company Name")
    if st.button("Fetch Raw Articles"):
        """
        Fetch raw articles for a given company.

        Sends a POST request to the /fetch_articles endpoint with the company name and
        a fixed number of articles (10). On success, stores the fetched articles in the
        session state.
        """
        payload = {"company": company, "num_articles": 10}
        response = requests.post(f"{BASE_URL}/fetch_articles", json=payload)
        if response.ok:
            articles = response.json()
            st.session_state.articles = articles
        else:
            st.error("Failed to fetch raw articles.")
    # Display raw articles if available
    if "articles" in st.session_state:
        display_raw_articles()


def display_raw_articles() -> None:
    """
    Display raw articles stored in the session state.
    """
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


def generate_summaries() -> None:
    """
    Generate summaries and topics for the fetched articles when the 'Summarize Articles' button is clicked.
    Stores the summarized articles in the Streamlit session state.
    """
    st.header("Step 2: Generate Summaries and Topics")
    if st.button("Summarize Articles"):
        """
        Generate summaries and topics for the fetched articles.

        Sends a POST request to the /summarize_articles endpoint with the articles from
        session state. On success, stores the summarized articles in the session state.
        """
        payload = {"articles": st.session_state.articles}
        response = requests.post(
            f"{BASE_URL}/summarize_articles", json=payload)
        if response.ok:
            summarized_articles = response.json()
            st.session_state.summarized_articles = summarized_articles
        else:
            st.error("Failed to generate summaries.")
    # Display summarized articles if available
    if "summarized_articles" in st.session_state:
        display_summarized_articles()


def display_summarized_articles() -> None:
    """
    Display summarized articles stored in the session state.
    """
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


def attach_sentiment() -> None:
    """
    Attach sentiment analysis to each summarized article when the 'Analyze Sentiment' button is clicked.
    Stores the articles with sentiment in the Streamlit session state.
    """
    st.header("Step 3: Attach Sentiment Analysis")
    if st.button("Analyze Sentiment"):
        """
        Attach sentiment analysis results to each summarized article.

        Sends a POST request to the /analyze_sentiment endpoint with the summarized articles.
        On success, updates the session state with articles that now include sentiment labels.
        """
        payload = {"articles": st.session_state.summarized_articles}
        response = requests.post(f"{BASE_URL}/analyze_sentiment", json=payload)
        if response.ok:
            articles_with_sentiment = response.json()
            st.session_state.articles_with_sentiment = articles_with_sentiment
        else:
            st.error("Failed to analyze sentiment.")
    # Display articles with sentiment if available
    if "articles_with_sentiment" in st.session_state:
        display_articles_with_sentiment()


def display_articles_with_sentiment() -> None:
    """
    Display articles with attached sentiment analysis from the session state.
    """
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


def get_comparative_sentiment() -> None:
    """
    Retrieve the comparative sentiment score when the 'Get Comparative Sentiment Score' button is clicked.
    Stores the result in the Streamlit session state.
    """
    st.header("Step 4: Comparative Sentiment Score")
    if st.button("Get Comparative Sentiment Score"):
        """
        Compute a comparative sentiment score for the analyzed articles.

        Sends a POST request to the /get_comparative_sentiment endpoint with articles that
        include sentiment labels. On success, stores the comparative sentiment score in the session state.
        """
        payload = {"articles": st.session_state.articles_with_sentiment}
        response = requests.post(
            f"{BASE_URL}/get_comparative_sentiment", json=payload)
        if response.ok:
            comp_sentiment = response.json()
            st.session_state.comp_sentiment = comp_sentiment
        else:
            st.error("Failed to get comparative sentiment score.")
    if "comp_sentiment" in st.session_state:
        display_comparative_sentiment()


def display_comparative_sentiment() -> None:
    """
    Display the comparative sentiment details including coverage differences, topic overlap,
    and sentiment distribution from the session state.
    """
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
            tagger_component("", [display_text], color_name=[overall_color])


def perform_final_analysis() -> None:
    """
    Perform the final sentiment analysis and TTS conversion when the 'Get Final Analysis' button is clicked.
    Stores the final analysis result in the session state.
    """
    st.header("Step 5: Final Analysis & TTS")
    if st.button("Get Final Analysis"):
        """
        Perform the final sentiment analysis and generate a Hindi audio summary.

        Sends a POST request to the /final_analysis endpoint with the comparative sentiment score
        and company name. On success, stores the analysis results, translated text, and audio file path
        in the session state.
        """
        company = st.session_state.articles[0].get(
            "company") if st.session_state.articles else ""
        payload = {
            "comp_score_dict": st.session_state.comp_sentiment,
            "company": company
        }
        response = requests.post(f"{BASE_URL}/final_analysis", json=payload)
        if response.ok:
            analysis = response.json()
            st.session_state.analysis = analysis
        else:
            st.error("Failed to get final analysis.")
    if "analysis" in st.session_state:
        display_final_analysis()


def display_final_analysis() -> None:
    """
    Display the final sentiment analysis results including the original analysis, translated analysis,
    audio playback, and a downloadable JSON output.
    """
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
        st.session_state.output = {
            "Company": st.session_state.articles[0].get("company") if st.session_state.articles else "",
            "Articles": st.session_state.articles_with_sentiment,
            "Comparative_Sentiment_Score": st.session_state.comp_sentiment,
            **st.session_state.analysis
        }
        if "output" in st.session_state:
            with st.container(border=True):
                st.header("Complete Output")
                st.json(st.session_state.output)
                json_data = json.dumps(
                    st.session_state.output, indent=2, ensure_ascii=False)
                st.download_button("Download Final Output", data=json_data,
                                   file_name=f"{to_snake_case(st.session_state.articles[0].get('company', 'newsbyte'))}_newsbyte.json",
                                   mime="application/json")


def main() -> None:
    """
    Main function to run the Streamlit NewsByte app by calling each step function in sequence.
    """
    set_page_style()
    fetch_raw_articles()
    generate_summaries()
    attach_sentiment()
    get_comparative_sentiment()
    perform_final_analysis()


if __name__ == "__main__":
    main()
