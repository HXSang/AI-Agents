"""Rule-based helpers for DayProfile pre-computation (time_phase, constraints, signals)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from services.executor.constant import (
    HealthDataConstants,
    InsightGroupConstants,
    TimeDataConstants,
)

_GROUP_TO_TIME_PHASE: Dict[str, str] = {
    InsightGroupConstants.WIND_DOWN_WINDOW: "wind_down",
    InsightGroupConstants.BEDTIME_WINDOW: "bedtime",
    InsightGroupConstants.MORNING_WINDOW: "morning",
    InsightGroupConstants.EVENING_FLEXIBLE: "evening",
    InsightGroupConstants.EVENING_AFTER_WORK: "after_work",
    InsightGroupConstants.ACTIVE_WINDOW: "active",
    InsightGroupConstants.HEALTH_GOAL: "active",
    InsightGroupConstants.CALENDAR_PATTERN: "active",
    InsightGroupConstants.RECOVERY_BREAK: "active",
    InsightGroupConstants.TASK_TYPE: "active",
    InsightGroupConstants.WEEKEND_LIFESTYLE: "active",
    InsightGroupConstants.SAFETY_RISK: "active",
    InsightGroupConstants.MOOD_BASED: "active",
    InsightGroupConstants.CONTEXTUAL_SUGGESTION: "active",
}


def compute_time_phase(group: str, time_data: Optional[Dict[str, Any]] = None) -> str:
    """Map detected insight group to a canonical time phase."""
    if group in _GROUP_TO_TIME_PHASE:
        return _GROUP_TO_TIME_PHASE[group]

    time_data = time_data or {}
    hour = time_data.get(TimeDataConstants.KEY_CURRENT_HOUR)
    if hour is None:
        return "active"
    hour = int(hour)
    if hour <= 5:
        return "bedtime"
    if hour <= 8:
        return "morning"
    if hour <= 16:
        return "active"
    if hour <= 18:
        return "after_work"
    if hour <= 20:
        return "evening"
    if hour <= 22:
        return "wind_down"
    return "bedtime"


def compute_hard_constraints(
    group: str,
    time_data: Optional[Dict[str, Any]] = None,
    health_params: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Derive NOT_DO-style hard constraints from group and time context."""
    time_data = time_data or {}
    health_params = health_params or {}
    constraints: List[str] = []

    time_phase = compute_time_phase(group, time_data)
    time_to_bedtime = time_data.get(TimeDataConstants.KEY_TIME_TO_BEDTIME)
    wind_down_buf = health_params.get(
        HealthDataConstants.KEY_WIND_DOWN_BUFFER_MINS,
        HealthDataConstants.DEFAULT_WIND_DOWN_BUFFER_MINS,
    )
    current_hour = time_data.get(TimeDataConstants.KEY_CURRENT_HOUR)
    in_wind_down = (
        group == InsightGroupConstants.WIND_DOWN_WINDOW
        or group == InsightGroupConstants.BEDTIME_WINDOW
        or (time_to_bedtime is not None and time_to_bedtime <= wind_down_buf)
    )

    if in_wind_down or time_phase in ("wind_down", "bedtime"):
        constraints.extend(
            [
                "wind_down/bedtime: no steps catch-up",
                "wind_down/bedtime: no outdoor walk",
                "wind_down/bedtime: no work planning or inbox",
                "wind_down/bedtime: calm activities only",
            ]
        )

    if current_hour is not None and int(current_hour) >= 18:
        sleep_last3 = health_params.get(HealthDataConstants.KEY_SLEEP_LAST3NIGHTS) or []
        sleep_goal = health_params.get(HealthDataConstants.KEY_SLEEP_GOAL)
        sleep_ref = health_params.get(HealthDataConstants.KEY_SLEEP_LASTNIGHT)
        if sleep_ref is None and sleep_last3:
            for v in sleep_last3:
                if v is not None:
                    sleep_ref = v
                    break
        if sleep_goal and sleep_ref is not None and sleep_ref < sleep_goal:
            constraints.append(
                "evening + sleep debt: no late intense workouts; prefer sleep hygiene over outdoor walk"
            )

    if group == InsightGroupConstants.SAFETY_RISK:
        constraints.append("safety_risk: non-safety suggestions blocked")

    return constraints


def build_key_signals(
    health_params: Optional[Dict[str, Any]] = None,
    calendar_metrics: Optional[Dict[str, Any]] = None,
    time_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build structured key_signals block for DayProfile."""
    health_params = health_params or {}
    calendar_metrics = calendar_metrics or {}
    time_data = time_data or {}

    sleep_lastnight = health_params.get(HealthDataConstants.KEY_SLEEP_LASTNIGHT)
    sleep_last3 = health_params.get(HealthDataConstants.KEY_SLEEP_LAST3NIGHTS) or []
    sleep_ref = sleep_lastnight
    if sleep_ref is None and sleep_last3:
        for v in sleep_last3:
            if v is not None:
                sleep_ref = v
                break

    steps_today = health_params.get(HealthDataConstants.KEY_STEPS_TODAY)
    steps_goal = health_params.get(HealthDataConstants.KEY_STEPS_GOAL)
    steps_behind = (
        steps_today is not None
        and steps_goal is not None
        and steps_goal > 0
        and steps_today < steps_goal
    )

    return {
        "sleep": {
            "last_night": sleep_lastnight,
            "trend_3nights": sleep_last3,
            "quality": health_params.get(HealthDataConstants.KEY_SLEEP_QUALITY),
            "goal": health_params.get(HealthDataConstants.KEY_SLEEP_GOAL),
            "reference_hours": sleep_ref,
        },
        "steps": {
            "today": steps_today,
            "goal": steps_goal,
            "behind": steps_behind,
        },
        "calendar": {
            "density": calendar_metrics.get("calendar_density"),
            "back_to_back": calendar_metrics.get("back_to_back_count", 0),
            "free_slot_mins": time_data.get(TimeDataConstants.KEY_FREE_SLOT_LENGTH),
            "time_to_next_event": time_data.get(
                TimeDataConstants.KEY_TIME_TO_NEXT_EVENT
            ),
        },
        "body": {
            "resting_hr": health_params.get(HealthDataConstants.KEY_RESTING_HEART_RATE),
            "baseline_hr": health_params.get(
                HealthDataConstants.KEY_BASELINE_RESTING_HR
            ),
            "stress_high": health_params.get("stress_signal_high", False),
            "mood": health_params.get("mood"),
        },
    }


def infer_primary_driver(
    key_signals: Dict[str, Any],
    group: str,
    health_params: Optional[Dict[str, Any]] = None,
    calendar_metrics: Optional[Dict[str, Any]] = None,
) -> tuple[str, str]:
    """Rule-based primary driver inference. Returns (driver, evidence)."""
    health_params = health_params or {}
    calendar_metrics = calendar_metrics or {}
    evidence_parts: List[str] = []

    if group == InsightGroupConstants.SAFETY_RISK:
        return "health_risk", "safety_risk group active"

    body = key_signals.get("body") or {}
    resting_hr = body.get("resting_hr")
    baseline_hr = body.get("baseline_hr")
    if resting_hr and resting_hr > 100:
        return "health_risk", f"resting HR {resting_hr} bpm elevated"
    if body.get("stress_high"):
        return "health_risk", "stress signal high"
    if resting_hr and baseline_hr and resting_hr > baseline_hr + 15:
        return "health_risk", f"resting HR {resting_hr} above baseline {baseline_hr}"

    sleep = key_signals.get("sleep") or {}
    sleep_ref = sleep.get("reference_hours")
    sleep_goal = sleep.get("goal")
    sleep_quality = sleep.get("quality")
    if sleep_ref is not None and sleep_goal and sleep_ref < sleep_goal:
        evidence_parts.append(f"sleep {sleep_ref}h vs goal {sleep_goal}h")
        driver = "sleep_debt"
    elif sleep_quality in ("low", "very_low"):
        evidence_parts.append(f"sleep quality {sleep_quality}")
        driver = "sleep_debt"
    else:
        driver = None

    calendar = key_signals.get("calendar") or {}
    back_to_back = calendar.get("back_to_back") or 0
    density = calendar.get("density")
    if back_to_back >= 2 or (density and str(density).lower() in ("high", "dense")):
        if driver:
            evidence_parts.append(f"back_to_back={back_to_back}")
        else:
            driver = "calendar_overload"
            evidence_parts.append(f"back_to_back={back_to_back}, density={density}")

    steps = key_signals.get("steps") or {}
    if steps.get("behind"):
        if driver:
            evidence_parts.append(f"steps {steps.get('today')}/{steps.get('goal')}")
        else:
            driver = "low_activity"
            evidence_parts.append(
                f"steps {steps.get('today')} behind goal {steps.get('goal')}"
            )

    continuous = calendar_metrics.get("continuous_events_minutes", 0) or 0
    if continuous >= 120 and not driver:
        driver = "recovery_deficit"
        evidence_parts.append(f"continuous events {continuous} min")

    if not driver:
        driver = "balanced_day"
        evidence_parts.append("no dominant deficit signal")

    return driver, "; ".join(evidence_parts)


def build_data_gaps(health_params: Optional[Dict[str, Any]] = None) -> List[str]:
    """List honest data gaps for the profile."""
    health_params = health_params or {}
    gaps: List[str] = []

    if health_params.get(HealthDataConstants.KEY_SLEEP_LASTNIGHT) is None:
        sleep_last3 = health_params.get(HealthDataConstants.KEY_SLEEP_LAST3NIGHTS) or []
        has_recent = any(v is not None for v in sleep_last3)
        if has_recent:
            gaps.append(
                "sleep_lastnight unavailable — using most recent value from 3-night trend"
            )
        else:
            gaps.append("sleep_lastnight unavailable — no recent sleep data")

    if health_params.get(HealthDataConstants.KEY_STEPS_TODAY) is None:
        gaps.append("steps_today unavailable")

    return gaps


def merge_day_profile(
    llm_json: Dict[str, Any],
    *,
    time_phase: str,
    hard_constraints: List[str],
    key_signals: Dict[str, Any],
    data_gaps: List[str],
    group: str,
    primary_driver: str = "balanced_day",
    primary_driver_evidence: str = "",
) -> Dict[str, Any]:
    """Merge LLM narrative fields with rule-based pre-computed fields."""
    cross_domain = llm_json.get("cross_domain") or llm_json.get(
        "cross_domain_tension", ""
    )
    return {
        "time_phase": time_phase,
        "insight_group": group,
        "primary_driver": primary_driver,
        "primary_driver_evidence": primary_driver_evidence,
        "dominant_pattern": llm_json.get("dominant_pattern", ""),
        "cross_domain_tension": cross_domain,
        "forward_context": llm_json.get("forward_context", ""),
        "key_signals": key_signals,
        "data_gaps": data_gaps,
        "hard_constraints": hard_constraints,
    }


def day_profile_to_json(profile: Dict[str, Any]) -> str:
    """Serialize DayProfile dict to JSON string."""
    return json.dumps(profile, ensure_ascii=False)


def parse_day_profile(context_str: str) -> Optional[Dict[str, Any]]:
    """Parse cached DayProfile JSON string."""
    if not context_str or not context_str.strip():
        return None
    try:
        data = json.loads(context_str)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None
