"""Post-writer validation — deprecated regex path; coherence handled by LLM auditor."""

from __future__ import annotations

from typing import Any, Dict, Optional


def validate_insight_output(
    parsed: Dict[str, Any],
    domain: str,
    domain_plan: Optional[Dict[str, Any]] = None,
    day_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pass-through; semantic ownership checks run in insight_coherence_auditor."""
    return parsed
