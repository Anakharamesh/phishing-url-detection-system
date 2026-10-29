"""Pydantic schemas for request validation and response formatting."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """URL prediction request body."""

    url: str = Field(..., description="The URL string to analyze for phishing detection.")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("URL cannot be empty.")
        if len(stripped) > 4096:
            raise ValueError("URL is too long (maximum 4096 characters allowed).")
        return stripped


class ModelPredictionDetail(BaseModel):
    id: str
    name: str
    type: str
    preprocessing: str
    accuracy: str
    prediction: str
    is_phishing: bool
    phishing_score: float
    decision_margin: Optional[float] = None
    metric_label: str


class PredictionResponse(BaseModel):
    url: str
    overall_verdict: str
    is_phishing: bool
    phishing_votes: int
    legitimate_votes: int
    total_models: int
    model_consensus_percentage: float
    consensus_text: str
    models: List[ModelPredictionDetail]
    features: Dict[str, float]


class FeatureMetadata(BaseModel):
    index: int
    name: str
    category: str
    description: str
