"""Prompts for LLM Coherence Auditor — semantic ownership and cross-domain checks."""

from __future__ import annotations

import json
from typing import Any, Dict

from agents.action_registry import DOMAIN_OWNERSHIP, format_categories_doc
from agents.prompt import get_language_name


def get_coherence_auditor_prompt(
    day_profile: Dict[str, Any],
    reasoning_plan: Dict[str, Any],
    insights_by_domain: Dict[str, Dict[str, Any]],
    language: str = "en-US",
) -> tuple[str, str]:
    """Build system + user prompts for cycle-level coherence audit."""
    language_name = get_language_name(language)
    ownership_doc = "\n".join(
        f"- {domain}: ownership={d['ownership']}, allowed={sorted(d['allowed'])}, "
        f"forbidden={sorted(d['forbidden'])}"
        for domain, d in DOMAIN_OWNERSHIP.items()
    )

    system_prompt = f"""You are a strict semantic QA auditor for a wellness/productivity insight app.
Analyze all three domain insights at the SAME moment and fix ownership violations and semantic duplicates.

**Language:** Write all rewritten text in {language_name}.

**Action categories (semantic — not keyword matching):**
{format_categories_doc()}

**Domain ownership (mandatory):**
{ownership_doc}

**Audit rules:**
1. OWNERSHIP: Each domain's point2/opportunity must stay within its allowed categories semantically.
   - productivity must NEVER suggest movement, sleep prep, nap, walk, exercise, or recovery actions
   - health owns physical recovery (movement, sleep, recovery)
   - overall owns priority/reflection/tradeoff — not walk, sleep, or deep work
2. PLAN COMPLIANCE: point2/opportunity must express the assigned recommended_action (paraphrase OK, category change NOT OK)
3. SEMANTIC DEDUP: walk/stroll/steps catchup = same movement family; dim lights/screen off/wind down = same sleep family
   — at most ONE domain may prescribe each action family
4. PERSPECTIVE: All three share primary_driver and shared_storyline but offer DIFFERENT cognitive angles
5. POINT1: Must be Observation → Interpretation (causal reasoning). Raw metrics alone = violation
6. TIME: Respect hard_constraints and time_phase (no deep work during wind_down/bedtime)

Return valid JSON only:
{{
  "cycle_ok": true,
  "violations": [
    {{"domain": "productivity", "field": "point2", "code": "ownership_violation",
      "message": "brief explanation"}}
  ],
  "rewrites": {{
    "productivity": {{"point2": "rewritten text or omit if OK"}},
    "health": {{}},
    "overall": {{"opportunity": "rewritten text or omit if OK"}}
  }},
  "coherence_summary": "one sentence on cross-domain alignment"
}}

Rules for rewrites:
- Only include fields that need fixing in rewrites; empty object {{}} if domain is OK
- Rewrites must express the domain's recommended_action from the ReasoningPlan
- Keep the same JSON field names (point1, point2, great_job, need_attention, opportunity)
- Do NOT add new fields"""

    payload = {
        "day_profile": {
            "primary_driver": day_profile.get("primary_driver"),
            "time_phase": day_profile.get("time_phase"),
            "hard_constraints": day_profile.get("hard_constraints"),
            "dominant_pattern": day_profile.get("dominant_pattern"),
        },
        "reasoning_plan": reasoning_plan,
        "insights": insights_by_domain,
    }

    user_prompt = f"""Audit this insight cycle and fix any violations.

{json.dumps(payload, ensure_ascii=False, indent=2)}

Return the audit JSON with rewrites for violating fields only."""

    return system_prompt, user_prompt
