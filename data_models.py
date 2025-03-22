from typing import List, Dict, Any
from pydantic import BaseModel, conlist, RootModel, Field, model_validator


class ArticleSummary(BaseModel):
    topics: conlist(str, min_length=3, max_length=3)
    summary: str


class ArticlesList(RootModel[conlist(ArticleSummary, min_length=1)]):
    pass


class CoverageDifference(BaseModel):
    Comparison: str
    Impact: str


class TopicOverlap(BaseModel):
    Common_Topics: List[str]

    class Config:
        extra = "allow"
        populate_by_name = True

    @model_validator(mode="before")
    def validate_exact_keys_and_types(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if "Common_Topics" not in data:
            raise ValueError("Missing required key 'Common_Topics'.")
        common_val = data["Common_Topics"]
        if not isinstance(common_val, list) or not all(isinstance(x, str) for x in common_val):
            raise ValueError("'Common_Topics' must be a list of strings.")
        unique_keys = [k for k in data if k != "Common_Topics"]
        if not unique_keys:
            raise ValueError(
                "There must be at least one unique topics key besides 'Common_Topics'.")
        expected_keys = [f"Unique_Topics_in_Article_{i}" for i in range(
            1, len(unique_keys) + 1)]
        if sorted(unique_keys) != sorted(expected_keys):
            raise ValueError(
                f"Unique topics keys must be exactly {expected_keys}. Got: {sorted(unique_keys)}")
        if len(data) != len(unique_keys) + 1:
            raise ValueError(
                "There must be exactly n + 1 keys (1 'Common_Topics' key and n unique keys).")
        for key in unique_keys:
            val = data[key]
            if not isinstance(val, list) or not all(isinstance(item, str) for item in val):
                raise ValueError(
                    f"Value for '{key}' must be a list of strings.")
        return data


class ComparativeSentimentScore(BaseModel):
    Coverage_Differences: List[CoverageDifference]
    Topic_Overlap: TopicOverlap
