"""Sylo Q&A question registry + light post-filters for the insight pool.

Production domains each have one overview question. ``get_active_questions``
keeps the pool path aligned with ``qa_prompts.*_GROUPS``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from insights.schemas.insight_item import InsightItem


@dataclass
class Question:
    id: str
    text: str
    metrics: List[str]
    action_family: str
    tone: str
    min_time_phase: List[str] = field(default_factory=list)
    max_time_phase: List[str] = field(default_factory=list)
    requires_metrics: List[str] = field(default_factory=list)


HEALTH_QUESTIONS: List[Question] = [
    Question(
        id="h_overview_infer",
        text="What is the single highest-value health story today?",
        metrics=["sleep", "heart_rate", "activity", "energy", "mood"],
        action_family="recovery",
        tone="gentle",
    ),
]

PRODUCTIVITY_QUESTIONS: List[Question] = [
    Question(
        id="p_overview_infer",
        text="What is the single highest-value productivity story today?",
        metrics=["productivity", "calendar", "tasks", "reminder"],
        action_family="planning",
        tone="informative",
    ),
]

OVERALL_QUESTIONS: List[Question] = [
    Question(
        id="o_overview_infer",
        text="What is the single highest-value cross-domain story today?",
        metrics=["health", "productivity", "finance", "goals", "balance"],
        action_family="cross_domain_synthesis",
        tone="informative",
    ),
]

ALL_QUESTION_SETS: Dict[str, List[Question]] = {
    "health": HEALTH_QUESTIONS,
    "productivity": PRODUCTIVITY_QUESTIONS,
    "overall": OVERALL_QUESTIONS,
}


def get_active_questions(
    domain: str,
    time_phase: str,
    meta: Optional[Dict[str, Any]] = None,
) -> List[Question]:
    """Return overview questions for the domain (Sylo: one per domain)."""
    _ = time_phase, meta
    return list(ALL_QUESTION_SETS.get(domain, []))


def post_filter_insights(
    insights: List[InsightItem],
    time_phase: str,
) -> List[InsightItem]:
    """Drop very-low-confidence / duplicate pool items."""
    _ = time_phase
    if not insights:
        return []

    filtered: List[InsightItem] = []
    seen_signatures: Dict[str, InsightItem] = {}

    for insight in insights:
        if insight.confidence < 0.3:
            continue

        sig = f"{insight.domain}|{insight.question_id}|{insight.action_family}"
        existing = seen_signatures.get(sig)
        if existing is not None:
            if insight.confidence <= existing.confidence:
                continue
            filtered = [i for i in filtered if i is not existing]

        filtered.append(insight)
        seen_signatures[sig] = insight

    return filtered


def score_insight_relevance(insight: InsightItem, time_phase: str) -> float:
    """Rank pool items; time_phase is unused for Sylo overview (kept for API)."""
    _ = time_phase
    return float(insight.confidence or 0.0)
