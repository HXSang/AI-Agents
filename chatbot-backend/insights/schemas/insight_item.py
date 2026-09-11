"""Schema for a single InsightItem generated from one Q&A question."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class InsightItem(BaseModel):
    insight_id: str = Field(..., description="UUID unique identifier for this insight")
    domain: str = Field(..., description='"health" | "productivity" | "overall"')
    question_id: str = Field(..., description="Question identifier, e.g. 'h1_sleep_recovery'")
    question_text: str = Field(..., description="Original question text")
    metrics: List[str] = Field(
        default_factory=list,
        description="Signal keys used to answer this question, e.g. ['sleep', 'activity']",
    )
    current_state: str = Field(
        ..., description="Description of the current state from data"
    )
    cause: str = Field(..., description="Root cause analysis")
    action: str = Field(
        ..., description="ONE specific recommended action, context-aware"
    )
    expected_outcome: str = Field(
        ..., description="What should happen if the action is taken"
    )
    evidence: str = Field(
        ...,
        description="Specific data points supporting current_state (conversational, no math)",
    )
    action_family: str = Field(
        ..., description="Action category: recovery | movement | focus | mental_break | planning | cross_domain_synthesis | ..."
    )
    tone: str = Field(
        ..., description="one of: encouraging | urgent | gentle | celebratory | informative"
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence score for this insight quality",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
