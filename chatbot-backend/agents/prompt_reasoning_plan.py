"""Layer 2 — unified cross-domain ReasoningPlan with ownership model (V2)."""

import json
from typing import Any, Dict, Optional

from agents.action_registry import (
    ACTION_REGISTRY,
    DOMAIN_OWNERSHIP,
    WRITER_CONTRACT,
    build_default_domain_plan,
)
from agents.prompt import get_language_name
from agents.prompt_insight_reasoning import (
    CROSS_DOMAIN_COHERENCE_RULES,
    NOT_DO_CONSTRAINT_GUARDS,
    _get_signal_weight,
)

_OWNERSHIP_SCHEMA = """
Output valid JSON only:
{
  "shared_storyline": "1 sentence: one shared understanding of today (references primary_driver)",
  "shared_lens": "trade_off | reinforcing | hidden_imbalance | constraint | momentum | carry_over | recovery_readiness | goal_alignment",
  "primary_driver": "from DayProfile — do not change",
  "domain_plans": {
    "productivity": {
      "ownership": "work_effectiveness",
      "perspective": "How primary_driver impacts work effectiveness (1 sentence)",
      "recommended_action": "one action key from registry",
      "allowed_action_categories": ["focus", "planning", "execution", "energy_management"],
      "forbidden_action_categories": ["movement", "sleep", "recovery"],
      "point1_angle": "calendar/work angle (1 sentence)",
      "reasoning_note": "why this perspective for productivity"
    },
    "health": {
      "ownership": "physical_health",
      "perspective": "How primary_driver impacts physical recovery (1 sentence)",
      "recommended_action": "one action key",
      "allowed_action_categories": ["movement", "recovery", "sleep"],
      "forbidden_action_categories": ["focus", "planning"],
      "point1_angle": "body/energy angle (1 sentence)",
      "reasoning_note": "why this perspective for health"
    },
    "overall": {
      "ownership": "life_balance",
      "perspective": "Why primary_driver is today's key priority (1 sentence)",
      "recommended_action": "one action key",
      "allowed_action_categories": ["priority", "reflection", "tradeoff"],
      "forbidden_action_categories": ["movement", "sleep", "focus"],
      "great_job_angle": "what is going well (1 phrase)",
      "need_attention_angle": "what needs attention (1 phrase)",
      "reasoning_note": "why this synthesis for overall"
    }
  },
  "coherence_check": "1 sentence: confirm each domain has different perspective AND different action category"
}
"""

_ACTION_KEYS_DOC = "\n".join(
    f"- {key} (category: {meta['category']})"
    for key, meta in sorted(ACTION_REGISTRY.items())
)


def get_reasoning_plan_prompt(
    day_profile: Dict[str, Any],
    group: str,
    language: str = "en",
) -> tuple[str, str]:
    """Build prompts for unified 3-domain ownership ReasoningPlan."""
    language_name = get_language_name(language)
    signal_weight = _get_signal_weight(group)
    profile_json = json.dumps(day_profile, ensure_ascii=False, indent=2)
    primary_driver = day_profile.get("primary_driver", "balanced_day")

    ownership_doc = "\n".join(
        f"- {domain}: ownership={d['ownership']}, allowed={sorted(d['allowed'])}, forbidden={sorted(d['forbidden'])}"
        for domain, d in DOMAIN_OWNERSHIP.items()
    )

    system_prompt = f"""You are a cross-domain insight reasoning planner (Layer 2 — V2 Ownership).
Given a DayProfile with primary_driver, create ONE shared storyline and THREE distinct domain perspectives.

**Language:** Plan narrative fields in {language_name}

**Design principle: One Reality → One Storyline → Three Perspectives**
- primary_driver from DayProfile is the single root cause for this cycle: {primary_driver}
- shared_storyline: one sentence all three domains share
- Each domain gets a DIFFERENT cognitive perspective on the same driver
- Each domain gets a DIFFERENT action CATEGORY (movement/sleep/focus/etc.)
- Productivity NEVER owns movement, sleep, or recovery actions
- Health owns physical recovery (movement, sleep, recovery)
- Overall owns prioritization/tradeoff framing (priority, reflection, tradeoff)

{NOT_DO_CONSTRAINT_GUARDS}

{CROSS_DOMAIN_COHERENCE_RULES}

**Domain ownership (mandatory):**
{ownership_doc}

**Valid recommended_action keys:**
{_ACTION_KEYS_DOC}

{signal_weight}

{_OWNERSHIP_SCHEMA}

Rules:
- recommended_action must be from the action registry above.
- Each domain's recommended_action category must be in its allowed_action_categories.
- movement category can appear in AT MOST ONE domain (health preferred).
- sleep category: only health during wind_down/bedtime; productivity/overall must NOT suggest sleep actions.
- coherence_check must confirm different perspectives, not just different words.
"""

    user_prompt = f"""Create an ownership ReasoningPlan for insight_group="{group}".

**DayProfile (primary_driver is canonical):**
{profile_json}

Return the ReasoningPlan JSON."""

    return system_prompt, user_prompt


def parse_reasoning_plan(plan_str: str) -> Optional[Dict[str, Any]]:
    """Parse ReasoningPlan JSON string."""
    if not plan_str or not plan_str.strip():
        return None
    try:
        data = json.loads(plan_str)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def ensure_reasoning_plan(
    plan: Optional[Dict[str, Any]],
    day_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fill missing domain plans with ownership defaults."""
    primary_driver = (day_profile or {}).get("primary_driver", "balanced_day")
    if not plan or not isinstance(plan, dict):
        plan = {}
    domain_plans = plan.get("domain_plans") or {}
    for domain in ("productivity", "health", "overall"):
        if domain not in domain_plans or not domain_plans[domain]:
            domain_plans[domain] = build_default_domain_plan(domain, primary_driver)
    plan["domain_plans"] = domain_plans
    plan.setdefault("primary_driver", primary_driver)
    plan.setdefault(
        "shared_storyline",
        f"Today's pattern is shaped by {primary_driver.replace('_', ' ')}.",
    )
    plan.setdefault("shared_lens", "constraint")
    plan.setdefault("coherence_check", "default ownership plans applied")
    return plan


def format_domain_plan_block(
    day_profile: Optional[Dict[str, Any]],
    domain_plan: Optional[Dict[str, Any]],
    domain: str,
    reasoning_plan: Optional[Dict[str, Any]] = None,
) -> str:
    """Format DayProfile + domain ownership plan for writer injection."""
    parts = ["**REASONING PLAN (Layer 2 — ownership assignment, mandatory):**"]
    rp = reasoning_plan or {}
    if rp.get("shared_storyline"):
        parts.append(f"- shared_storyline: {rp['shared_storyline']}")
    if day_profile:
        parts.append(f"- primary_driver: {day_profile.get('primary_driver', '')}")
        parts.append(f"- time_phase: {day_profile.get('time_phase', '')}")
        if day_profile.get("dominant_pattern"):
            parts.append(f"- dominant_pattern: {day_profile['dominant_pattern']}")
        constraints = day_profile.get("hard_constraints") or []
        if constraints:
            parts.append(f"- hard_constraints: {'; '.join(constraints)}")
    if domain_plan:
        parts.append(f"- domain: {domain}")
        parts.append(f"- ownership: {domain_plan.get('ownership', '')}")
        if domain_plan.get("perspective"):
            parts.append(f"- perspective: {domain_plan['perspective']}")
        if domain_plan.get("recommended_action"):
            action = domain_plan["recommended_action"]
            parts.append(
                f"- recommended_action (MANDATORY for point2/opportunity): {action}"
            )
            parts.append(
                f"- MANDATORY_ACTION: point2/opportunity MUST express `{action}` "
                f"in natural language. Paraphrase allowed; changing action category is FORBIDDEN."
            )
            _examples = {
                "productivity": (
                    "GOOD (focus_block): Protect 20 minutes of uninterrupted work on your top priority.\n"
                    "BAD: Take a short nap to recover — that is health/recovery, not work_effectiveness."
                ),
                "health": (
                    "GOOD (brief_movement_snack): A 5–10 minute light walk helps recovery without a full workout.\n"
                    "BAD: Triage your inbox now — that is planning, not physical_health."
                ),
                "overall": (
                    "GOOD (priority_framing): Name today's one anchor priority given how sleep and load interact.\n"
                    "BAD: Go for a walk now — that is movement owned by health, not life_balance."
                ),
            }
            if domain in _examples:
                parts.append(f"- EXAMPLE:\n{_examples[domain]}")
        allowed = domain_plan.get("allowed_action_categories") or []
        forbidden = domain_plan.get("forbidden_action_categories") or []
        if allowed:
            parts.append(f"- allowed_action_categories: {', '.join(allowed)}")
        if forbidden:
            parts.append(f"- forbidden_action_categories: {', '.join(forbidden)}")
        if domain_plan.get("point1_angle"):
            parts.append(f"- point1_angle: {domain_plan['point1_angle']}")
        if domain == "overall":
            if domain_plan.get("great_job_angle"):
                parts.append(f"- great_job_angle: {domain_plan['great_job_angle']}")
            if domain_plan.get("need_attention_angle"):
                parts.append(
                    f"- need_attention_angle: {domain_plan['need_attention_angle']}"
                )
    parts.append(WRITER_CONTRACT.strip())
    return "\n".join(parts)
