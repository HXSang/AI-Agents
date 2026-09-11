"""Action Registry and domain ownership constants for Insight V2."""

from __future__ import annotations

from typing import Dict, List, Optional

# Action key -> category mapping (schema vocabulary for ReasoningPlan)
ACTION_REGISTRY: Dict[str, Dict[str, str]] = {
    "outdoor_walk": {"category": "movement"},
    "brief_movement_snack": {"category": "movement"},
    "movement_snack": {"category": "movement"},
    "stretch": {"category": "movement"},
    "steps_catchup": {"category": "movement"},
    "hydration_anchor": {"category": "recovery"},
    "nap": {"category": "recovery"},
    "recovery_break": {"category": "recovery"},
    "breathing_reset": {"category": "recovery"},
    "focus_block": {"category": "focus"},
    "deep_work": {"category": "focus"},
    "admin_batch": {"category": "planning"},
    "calendar_buffer": {"category": "planning"},
    "work_planning": {"category": "planning"},
    "execution_sprint": {"category": "execution"},
    "energy_management": {"category": "energy_management"},
    "sleep_hygiene": {"category": "sleep"},
    "dim_lights": {"category": "sleep"},
    "screen_off": {"category": "sleep"},
    "wind_down": {"category": "sleep"},
    "priority_framing": {"category": "priority"},
    "pattern_synthesis": {"category": "reflection"},
    "tradeoff_framing": {"category": "tradeoff"},
    "celebration": {"category": "reflection"},
    "safety_only": {"category": "safety"},
}

# Human-readable category descriptions for LLM auditor prompts (not regex)
ACTION_CATEGORIES_DOC: Dict[str, str] = {
    "movement": "Physical activity: walk, stroll, steps, stretch, outdoor movement",
    "sleep": "Sleep prep: wind-down, dim lights, screen off, bedtime routine",
    "recovery": "Rest/recovery: nap, micro-break, breathing reset, hydration as recovery",
    "focus": "Deep work: focus block, protect concentration, uninterrupted work",
    "planning": "Admin/planning: inbox triage, calendar buffer, day planning",
    "execution": "Task execution: sprint on deliverables, ship work",
    "energy_management": "Pace energy: batch tasks, short breaks between blocks (not nap/walk)",
    "priority": "Life priority: name anchor priority, tradeoff framing",
    "reflection": "Pattern synthesis: connect signals, celebrate consistency",
    "tradeoff": "Tradeoff framing: what to defer vs protect",
    "safety": "Safety only: rest and monitor when health risk",
}

# Domain -> allowed/forbidden categories and default ownership
DOMAIN_OWNERSHIP: Dict[str, Dict[str, object]] = {
    "productivity": {
        "ownership": "work_effectiveness",
        "allowed": frozenset({"focus", "planning", "execution", "energy_management"}),
        "forbidden": frozenset({"movement", "sleep", "recovery"}),
        "default_action": "focus_block",
    },
    "health": {
        "ownership": "physical_health",
        "allowed": frozenset({"movement", "recovery", "sleep"}),
        "forbidden": frozenset({"focus", "planning"}),
        "default_action": "brief_movement_snack",
    },
    "overall": {
        "ownership": "life_balance",
        "allowed": frozenset({"priority", "reflection", "tradeoff"}),
        "forbidden": frozenset({"movement", "sleep", "focus"}),
        "default_action": "priority_framing",
    },
}

WRITER_CONTRACT = """
**WRITER CONTRACT (Layer 3 — mandatory):**
- You may: explain, contextualize, personalize the assigned recommended_action.
- You may NOT: invent new actions, switch action categories, or override ownership.
- point1: Observation → Interpretation — explain WHY primary_driver matters from YOUR domain perspective.
  Metrics may support the insight; metrics must NOT be the insight.
- point2 / opportunity: MUST express the assigned recommended_action in natural language.
  Do NOT suggest nap, walk, sleep hygiene, or exercise unless your plan allows that category.
- All three domains share one primary_driver and shared_storyline — add your angle, do not repeat other domains' actions.
"""


def get_action_category(action_key: str) -> Optional[str]:
    """Return category for an action key."""
    entry = ACTION_REGISTRY.get(action_key)
    return entry.get("category") if entry else None


def get_domain_defaults(domain: str) -> Dict[str, object]:
    """Return ownership defaults for a domain."""
    return DOMAIN_OWNERSHIP.get(domain, DOMAIN_OWNERSHIP["productivity"])


def build_default_domain_plan(
    domain: str, primary_driver: str = "balanced_day"
) -> Dict[str, object]:
    """Build fallback domain plan when LLM plan is missing."""
    defaults = get_domain_defaults(domain)
    perspectives = {
        "productivity": f"How {primary_driver} impacts work effectiveness and energy allocation",
        "health": f"How {primary_driver} impacts physical recovery and body signals",
        "overall": f"Why {primary_driver} is today's key priority for life balance",
    }
    return {
        "ownership": defaults["ownership"],
        "perspective": perspectives.get(domain, ""),
        "recommended_action": defaults["default_action"],
        "allowed_action_categories": sorted(defaults["allowed"]),
        "forbidden_action_categories": sorted(defaults["forbidden"]),
        "point1_angle": perspectives.get(domain, ""),
        "reasoning_note": f"Default ownership plan for {domain}",
    }


def format_categories_doc() -> str:
    """Format ACTION_CATEGORIES_DOC for LLM prompts."""
    return "\n".join(f"- {cat}: {desc}" for cat, desc in ACTION_CATEGORIES_DOC.items())
