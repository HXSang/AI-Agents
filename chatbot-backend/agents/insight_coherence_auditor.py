"""LLM Coherence Auditor — semantic ownership and cross-domain fix."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, Optional

from agents.llm_helper import llm_response_text, parse_json_object_from_llm_text
from agents.prompt_coherence_auditor import get_coherence_auditor_prompt
from langchain_core.messages import HumanMessage, SystemMessage
from utils.logger import logger

_DOMAIN_KEYS = {
    "productivity": ("productivity", "productivity_insight"),
    "health": ("health", "health_insight"),
    "overall": ("overall", "overall_insight"),
}


def _normalize_domain(domain: str) -> str:
    if domain in ("health", "health_insight"):
        return "health"
    if domain in ("overall", "overall_insight"):
        return "overall"
    return "productivity"


def _apply_rewrites(
    insights_by_domain: Dict[str, Dict[str, Any]],
    rewrites: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Merge LLM rewrites into insight dicts."""
    fixed = copy.deepcopy(insights_by_domain)
    for raw_domain, fields in (rewrites or {}).items():
        if not isinstance(fields, dict) or not fields:
            continue
        domain = _normalize_domain(raw_domain)
        target = fixed.get(domain) or fixed.get(f"{domain}_insight")
        if not target:
            continue
        key = domain if domain in fixed else f"{domain}_insight"
        for field, value in fields.items():
            if value and isinstance(value, str):
                fixed[key][field] = value
    return fixed


async def audit_and_fix_cycle(
    llm: Any,
    day_profile: Dict[str, Any],
    reasoning_plan: Dict[str, Any],
    insights_by_domain: Dict[str, Dict[str, Any]],
    language: str = "en-US",
) -> Dict[str, Dict[str, Any]]:
    """Run LLM coherence audit; return fixed insights (or originals on failure)."""
    if not insights_by_domain or len(insights_by_domain) < 3:
        return insights_by_domain

    # Normalize keys to productivity / health / overall
    normalized: Dict[str, Dict[str, Any]] = {}
    for domain in ("productivity", "health", "overall"):
        for key in (domain, f"{domain}_insight", "health_insight", "overall_insight"):
            if key in insights_by_domain and domain not in normalized:
                if key == "health_insight":
                    normalized["health"] = insights_by_domain[key]
                elif key == "overall_insight":
                    normalized["overall"] = insights_by_domain[key]
                else:
                    normalized[domain] = insights_by_domain[key]
                break

    if len(normalized) < 3:
        return insights_by_domain

    try:
        system_prompt, user_prompt = get_coherence_auditor_prompt(
            day_profile=day_profile,
            reasoning_plan=reasoning_plan,
            insights_by_domain=normalized,
            language=language,
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await llm.ainvoke(messages)
        text = llm_response_text(response)
        parsed = parse_json_object_from_llm_text(text)
        if not parsed:
            logger.warning("Coherence auditor: could not parse LLM response")
            return insights_by_domain

        rewrites = parsed.get("rewrites") or {}
        if parsed.get("cycle_ok") and not any(rewrites.values()):
            logger.info(
                f"Coherence auditor: cycle OK — {parsed.get('coherence_summary', '')[:80]}"
            )
            return insights_by_domain

        violations = parsed.get("violations") or []
        if violations:
            logger.info(
                f"Coherence auditor: {len(violations)} violation(s) — "
                f"{parsed.get('coherence_summary', '')[:80]}"
            )

        return _apply_rewrites(normalized, rewrites)
    except Exception as e:
        logger.warning(f"Coherence auditor failed, keeping originals: {e}")
        return insights_by_domain


def parse_cached_insight(cached: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Parse insight dict from Redis cache entry."""
    if not cached:
        return None
    raw = cached.get("insight")
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None
