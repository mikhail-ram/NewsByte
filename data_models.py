"""Data models for NewsByte application.

This module defines Pydantic models for structured data handling,
including article summaries, sentiment analysis, and comparative analysis.
"""

from typing import List
from pydantic import BaseModel, conlist, RootModel


class ArticleSummary(BaseModel):
    """Model for article summary with topics and text.

    Attributes:
        topics (conlist): List of exactly 3 topic keywords
        summary (str): Concise summary of the article
    """
    topics: conlist(str, min_length=3, max_length=3)
    summary: str


class ArticlesList(RootModel[conlist(ArticleSummary, min_length=1)]):
    """Root model for a list of article summaries.

    Ensures the list contains at least one article summary.
    """
    pass


class CoverageDifference(BaseModel):
    """Model for comparing coverage between articles.

    Attributes:
        Comparison (str): Description of differences between articles
        Impact (str): Analysis of impact on investors
    """
    Comparison: str
    Impact: str


class TopicOverlap(BaseModel):
    """Model for analyzing topic overlap between articles.

    Attributes:
        Common_Topics (List[str]): Topics shared across articles
    """
    Common_Topics: List[str]

    class Config:
        extra = "allow"


class ComparativeSentimentScore(BaseModel):
    """Model for comparative sentiment analysis results.

    Attributes:
        Coverage_Differences (List[CoverageDifference]): List of article comparisons
        Topic_Overlap (TopicOverlap): Analysis of shared and unique topics
    """
    Coverage_Differences: List[CoverageDifference]
    Topic_Overlap: TopicOverlap
