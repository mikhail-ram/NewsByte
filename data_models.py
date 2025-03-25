from typing import List, Dict, Any
from pydantic import BaseModel, conlist, RootModel, Field, model_validator


class ArticleSummary(BaseModel):
    """
    Model representing a summary of an article.

    Attributes:
        topics (conlist[str, min_length=3, max_length=3]): A list of exactly three topics related to the article.
        summary (str): A summary text for the article.
    """
    topics: conlist(str, min_length=3, max_length=3)
    summary: str


class ArticlesList(RootModel[conlist(ArticleSummary, min_length=1)]):
    """
    Root model representing a list of article summaries.

    Ensures that the list contains at least one ArticleSummary.
    """
    pass


class CoverageDifference(BaseModel):
    """
    Model representing a difference in coverage between articles.

    Attributes:
        Comparison (str): The comparison description between different articles.
        Impact (str): The impact description of the coverage difference.
    """
    Comparison: str
    Impact: str


class TopicOverlap(BaseModel):
    """
    Model representing the overlap in topics across articles.

    Attributes:
        Common_Topics (List[str]): A list of topics that are common across the articles.

    Note:
        Additional fields are allowed as extra information.
    """
    Common_Topics: List[str]

    class Config:
        extra = "allow"


class ComparativeSentimentScore(BaseModel):
    """
    Model representing the comparative sentiment score of a set of articles.

    Attributes:
        Coverage_Differences (List[CoverageDifference]): A list of coverage differences with their respective comparisons and impacts.
        Topic_Overlap (TopicOverlap): An object detailing the overlapping topics among the articles.
    """
    Coverage_Differences: List[CoverageDifference]
    Topic_Overlap: TopicOverlap
