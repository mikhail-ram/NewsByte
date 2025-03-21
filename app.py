import streamlit as st
from streamlit_extras.tags import tagger_component
import requests
import json

# Update this URL to your actual backend base URL
BASE_URL = "http://localhost:8000"

st.title("NewsByte Sentiment Analysis Pipeline")

# ---------------------------
# Step 1: Fetch Raw Articles
# ---------------------------
# Add CSS for justification
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

st.header("Step 1: Fetch Raw Articles")
company = st.text_input("Enter Company Name")

if st.button("Fetch Raw Articles"):
    payload = {"company": company, "num_articles": 10}
    response = requests.post(f"{BASE_URL}/fetch_articles", json=payload)
    if response.ok:
        articles = response.json()
        # Save for later steps
        st.session_state.articles = articles

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
    else:
        st.error("Failed to fetch raw articles.")

# -----------------------------------------
# Step 2: Generate Summaries and Topics
# -----------------------------------------
if "articles" in st.session_state:
    st.header("Step 2: Generate Summaries and Topics")
    if st.button("Summarize Articles"):
        payload = {"articles": st.session_state.articles}
        response = requests.post(
            f"{BASE_URL}/summarize_articles", json=payload)
        if response.ok:
            summarized_articles = response.json()
            st.write("Articles with Summaries and Topics:")
            st.session_state.summarized_articles = summarized_articles

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
                        topics = ", ".join(topic.title() for topic in topics)
                    st.caption(f"{topics}")
                    st.markdown(
                        f'<div class="justified-text">{article.get("summary", "No summary provided.")}</div>',
                        unsafe_allow_html=True
                    )
                    st.write("")
        else:
            st.error("Failed to generate summaries.")

# -----------------------------------
# Step 3: Attach Sentiment Analysis
# -----------------------------------
if "summarized_articles" in st.session_state:
    st.header("Step 3: Attach Sentiment Analysis")
    if st.button("Analyze Sentiment"):
        payload = {"articles": st.session_state.summarized_articles}
        response = requests.post(f"{BASE_URL}/analyze_sentiment", json=payload)
        if response.ok:
            articles_with_sentiment = response.json()
            st.write("Articles with Sentiment:")
            st.session_state.articles_with_sentiment = articles_with_sentiment

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
                        topics = ", ".join(topic.title() for topic in topics)
                    st.caption(f"{topics}")

                    st.markdown(
                        f'<div class="justified-text">{article.get("summary", "No summary provided.")}</div>',
                        unsafe_allow_html=True
                    )
                    st.write("")
        else:
            st.error("Failed to analyze sentiment.")

# ---------------------------------------------------------
# Step 4: Get Comparative Sentiment Score & Distribution
# ---------------------------------------------------------
if "articles_with_sentiment" in st.session_state:
    st.header("Step 4: Comparative Sentiment Score")
    if st.button("Get Comparative Sentiment Score"):
        payload = {"articles": st.session_state.articles_with_sentiment}
        response = requests.post(
            f"{BASE_URL}/get_comparative_sentiment", json=payload)
        if response.ok:
            comp_sentiment = response.json()
            st.session_state.comp_sentiment = comp_sentiment

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
                common_topics = topic_overlap.get("Common Topics", [])
                st.markdown(
                    f'<div class="justified-text"><b>Common Topics</b>: {", ".join(common_topics) if common_topics else "None"}</div>',
                    unsafe_allow_html=True
                )

                # Loop through the unique topic keys:
                for key in topic_overlap:
                    if key != "Common Topics":
                        topics = topic_overlap[key]
                        # Each list of topics is converted to title case and joined by commas.
                        topics_formatted = ", ".join(
                            topic.title() for topic in topics)
                        st.markdown(
                            f'<div class="justified-text"><b>{key}</b>: {topics_formatted}</div>',
                            unsafe_allow_html=True
                        )
                st.write("")

            # --- Sentiment Distribution ---
            with st.container(border=True):
                st.header("Sentiment Distribution")
                sentiment_distribution = st.session_state.comp_sentiment.get(
                    "Sentiment Distribution", {})

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
        else:
            st.error("Failed to get comparative sentiment score.")

# -----------------------------------------------------
# Step 5: Final Analysis, Translation, and TTS Output
# -----------------------------------------------------
if "comp_sentiment" in st.session_state:
    st.header("Step 5: Final Analysis & TTS")
    if st.button("Get Final Analysis"):
        payload = {
            "comp_score_dict": st.session_state.comp_sentiment, "company": company}
        response = requests.post(f"{BASE_URL}/final_analysis", json=payload)
        if response.ok:
            final_output = response.json()
            # Display Sentiment Analysis
            with st.container(border=True):
                st.header("Final Sentiment Analysis")
                st.markdown(
                    f'<div class="justified-text">{final_output["Final Sentiment Analysis"]}</div>',
                    unsafe_allow_html=True
                )
                st.write("")

            with st.container(border=True):
                st.header("Translated Final Sentiment Analysis")
                st.markdown(
                    f'<div class="justified-text">{final_output["Translated Final Sentiment Analysis"]}</div>',
                    unsafe_allow_html=True
                )
                st.write("")
                # Play TTS audio if available
                audio_path = final_output.get("Audio")
                if audio_path:
                    st.audio(audio_path)

            # Provide a download button for the final output JSON
            output = {"Company": company, "Articles": st.session_state.articles_with_sentiment,
                      "Comparative Sentiment Score": st.session_state.comp_sentiment, **final_output}
            with st.container(border=True):
                st.header("Complete Output")
                st.json(output)
                json_data = json.dumps(output, indent=2, ensure_ascii=False)
                st.download_button("Download Final Output", data=json_data,
                                   file_name=f"{company}_newsbyte.json", mime="application/json")
        else:
            st.error("Failed to get final analysis.")
