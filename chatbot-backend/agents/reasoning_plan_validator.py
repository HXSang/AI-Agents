"""Post-validation and auto-correction for V2 ownership ReasoningPlan."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set

from agents.action_registry import (
    ACTION_REGISTRY,
    build_default_domain_plan,
    get_action_category,
    get_domain_defaults,
)
from agents.prompt_reasoning_plan import ensure_reasoning_plan

# Wind-down: productivity/overall cannot use these categories
WIND_DOWN_FORBIDDEN_CATEGORIES = frozenset(
    {"movement", "sleep", "recovery", "focus", "planning"}
)


def _get_recommended_action(domain_plan: Dict[str, Any]) -> str:
    return (
        domain_plan.get("recommended_action")
        or domain_plan.get("opportunity_theme")
        or domain_plan.get("suggestion_theme")
        or ""
    )


def _set_recommended_action(domain_plan: Dict[str, Any], action: str) -> None:
    domain_plan["recommended_action"] = action
    domain_plan.pop("suggestion_theme", None)
    domain_plan.pop("opportunity_theme", None)


def _enforce_ownership(
    domain: str,
    domain_plan: Dict[str, Any],
    in_wind_down: bool,
) -> None:
    """Ensure recommended_action category matches domain ownership."""
    defaults = get_domain_defaults(domain)
    allowed: Set[str] = set(defaults["allowed"])  # type: ignore[arg-type]
    forbidden: Set[str] = set(defaults["forbidden"])  # type: ignore[arg-type]
    action = _get_recommended_action(domain_plan)
    if action and action not in ACTION_REGISTRY:
        fallback = str(defaults["default_action"])
        _set_recommended_action(domain_plan, fallback)
        action = fallback
    category = get_action_category(action)

    if in_wind_down and domain in ("productivity", "overall"):
        fallback = (
            "energy_management" if domain == "productivity" else "priority_framing"
        )
        if category in WIND_DOWN_FORBIDDEN_CATEGORIES or not category:
            _set_recommended_action(domain_plan, fallback)
            domain_plan["allowed_action_categories"] = sorted(
                allowed - WIND_DOWN_FORBIDDEN_CATEGORIES or allowed
            )
        return

    if category and (category in forbidden or category not in allowed):
        fallback = str(defaults["default_action"])
        _set_recommended_action(domain_plan, fallback)
        category = get_action_category(fallback)

    domain_plan.setdefault("ownership", defaults["ownership"])
    domain_plan.setdefault("allowed_action_categories", sorted(allowed))
    domain_plan.setdefault("forbidden_action_categories", sorted(forbidden))


def _dedupe_categories(domain_plans: Dict[str, Any]) -> None:
    """Ensure each action category appears in at most one domain."""
    category_owner: Dict[str, str] = {}
    priority = {"health": 0, "productivity": 1, "overall": 2}

    for domain in sorted(domain_plans.keys(), key=lambda d: priority.get(d, 99)):
        dp = domain_plans.get(domain) or {}
        action = _get_recommended_action(dp)
        category = get_action_category(action)
        if not category or category in ("reflection", "tradeoff", "priority", "safety"):
            continue
        if category in category_owner:
            defaults = get_domain_defaults(domain)
            _set_recommended_action(dp, str(defaults["default_action"]))
            forbidden = dp.setdefault("forbidden_action_categories", [])
            if isinstance(forbidden, list) and category not in forbidden:
                forbidden.append(category)
        else:
            category_owner[category] = domain


def validate_and_fix_reasoning_plan(
    plan: Optional[Dict[str, Any]],
    day_profile: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Validate ownership plan and apply deterministic fixes."""
    primary_driver = (day_profile or {}).get("primary_driver", "balanced_day")
    fixed = ensure_reasoning_plan(plan, day_profile)
    fixed = copy.deepcopy(fixed)

    time_phase = (day_profile or {}).get("time_phase", "")
    hard_constraints = (day_profile or {}).get("hard_constraints") or []
    in_wind_down = time_phase in ("wind_down", "bedtime") or any(
        "wind_down" in c or "bedtime" in c for c in hard_constraints
    )

    domain_plans = fixed.get("domain_plans") or {}
    for domain in ("productivity", "health", "overall"):
        dp = domain_plans.setdefault(
            domain, build_default_domain_plan(domain, primary_driver)
        )
        _enforce_ownership(domain, dp, in_wind_down)

    _dedupe_categories(domain_plans)

    # Wind-down: only health may own sleep category
    if in_wind_down:
        for domain in ("productivity", "overall"):
            dp = domain_plans[domain]
            cat = get_action_category(_get_recommended_action(dp))
            if cat == "sleep":
                defaults = get_domain_defaults(domain)
                _set_recommended_action(dp, str(defaults["default_action"]))

    fixed["coherence_check"] = (
        fixed.get("coherence_check", "")
        + " [validator: ownership + category dedup applied]"
    ).strip()
    fixed["primary_driver"] = primary_driver
    return fixed
