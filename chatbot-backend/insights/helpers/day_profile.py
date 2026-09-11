"""Rule-based helpers for DayProfile pre-computation (time_phase, constraints, signals)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from insights.health.canonical_field_mapping import CanonicalField
from insights.health.health_processor.common.utils import health_param_value
from services.executor.constant import HealthDataConstants
from services.executor.constant import InsightGroupConstants
from services.executor.constant import TimeDataConstants

_GROUP_TO_TIME_PHASE: Dict[str, str] = {
    InsightGroupConstants.WIND_DOWN_WINDOW: "wind_down",
    InsightGroupConstants.BEDTIME_WINDOW: "bedtime",
    InsightGroupConstants.MORNING_WINDOW: "morning",
    InsightGroupConstants.EVENING_FLEXIBLE: "evening",
    InsightGroupConstants.EVENING_AFTER_WORK: "after_work",
    InsightGroupConstants.ACTIVE_WINDOW: "afternoon",
}

def compute_time_phase(
    group: str,
    time_data: Optional[Dict[str, Any]] = None,
    health_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Map detected insight group to a canonical time phase, falling back to real time."""
    if group in _GROUP_TO_TIME_PHASE:
        return _GROUP_TO_TIME_PHASE[group]

    time_data = time_data or {}
    hour = time_data.get(TimeDataConstants.KEY_CURRENT_HOUR)
    if hour is None:
        return "Unknown"

    # Parse active window boundary from user's schedule preference
    active_end_str = time_data.get("active_hours_end_time", "18:00")
    try:
        aeh, aem = active_end_str.split(":")
        active_end_hour = int(aeh) + int(aem) / 60.0
    except Exception:
        active_end_hour = 18.0

    # Dynamic bedtime check
    time_to_bedtime = time_data.get(TimeDataConstants.KEY_TIME_TO_BEDTIME)

    # Fallback if time_to_bedtime is missing but we have current_hour and bedtime_start
    if time_to_bedtime is None and hour is not None:
        bedtime_str = time_data.get("bedtime_start")
        if bedtime_str:
            try:
                b_parts = bedtime_str.split(":")
                b_hour = int(b_parts[0])
                b_min = int(b_parts[1]) if len(b_parts) > 1 else 0
                b_time = b_hour + b_min / 60.0
                curr_hour_float = float(hour)

                # Handling next day (e.g. current=23, bedtime=01 => diff=2)
                diff = b_time - curr_hour_float
                if diff < -12:
                    diff += 24
                elif diff > 12:
                    diff -= 24

                time_to_bedtime = int(diff * 60)
            except Exception:
                pass

    if time_to_bedtime is not None:
        # Use health_params for buffer, default to 60 if not available
        wind_down_buf = 60
        if health_params:
            wind_down_buf = health_param_value(
                health_params,
                "health_params_wind_down_buffer_mins",
                HealthDataConstants.KEY_WIND_DOWN_BUFFER_MINS,
            ) or 60

        if time_to_bedtime <= 0:
            return "bedtime"
        elif time_to_bedtime <= wind_down_buf:
            return "wind_down"

    hour = int(hour)
    if hour < 5:
        return "late_night"
    if hour < 12:
        return "morning"
    if hour < active_end_hour:
        return "afternoon"
    if hour < 22:
        return "evening"
    return "late_night"


def compute_hard_constraints(
    group: str,
    time_data: Optional[Dict[str, Any]] = None,
    health_params: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Determine hard constraints based on group or current time phase."""
    constraints = []

    # 1. Base constraints from time_phase
    time_phase = compute_time_phase(group, time_data, health_params)
    time_to_bedtime = time_data.get(TimeDataConstants.KEY_TIME_TO_BEDTIME)
    wind_down_buf = health_param_value(
        health_params,
        "health_params_wind_down_buffer_mins",
        HealthDataConstants.KEY_WIND_DOWN_BUFFER_MINS,
    ) or HealthDataConstants.DEFAULT_WIND_DOWN_BUFFER_MINS
    current_hour = time_data.get(TimeDataConstants.KEY_CURRENT_HOUR)

    # Only allow wind_down/bedtime constraints when the time phase actually warrants it.
    # Guard against time_to_bedtime being None or unexpectedly small during morning/active hours.
    in_wind_down = (
        group
        in (
            InsightGroupConstants.WIND_DOWN_WINDOW,
            InsightGroupConstants.BEDTIME_WINDOW,
        )
        or time_phase in ("wind_down", "bedtime")
        or (
            time_phase not in ("morning", "active")
            and time_to_bedtime is not None
            and time_to_bedtime <= wind_down_buf
        )
    )

    if in_wind_down:
        constraints.extend(
            [
                "wind_down/bedtime: no steps catch-up",
                "wind_down/bedtime: no outdoor walk",
                "wind_down/bedtime: no work planning or inbox",
                "wind_down/bedtime: calm activities only",
            ]
        )
    elif time_phase in ("evening", "after_work"):
        constraints.extend(
            [
                "evening/after_work: no outdoor walk or step catch-up",
                "evening/after_work: no intense workout",
                "evening/after_work: prefer calm indoor activities",
            ]
        )

    if current_hour is not None:
        active_end_str = time_data.get("active_hours_end_time", "18:00")
        try:
            aeh, aem = active_end_str.split(":")
            active_end_hour = int(aeh) + int(aem) / 60.0
        except Exception:
            active_end_hour = 18.0
        if float(current_hour) > active_end_hour:
            sleep_last3 = health_param_value(
                health_params,
                "health_params_sleep_last3nights",
                HealthDataConstants.KEY_SLEEP_LAST3NIGHTS,
            ) or []
            sleep_goal = health_param_value(
                health_params,
                "health_params_sleep_goal",
                HealthDataConstants.KEY_SLEEP_GOAL,
            )
            sleep_ref = health_param_value(
                health_params,
                "health_params_sleep_lastnight",
                HealthDataConstants.KEY_SLEEP_LASTNIGHT,
            )
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
    latest_mood: Optional[Dict[str, Any]] = None,
    calendar_events: Optional[List[Dict[str, Any]]] = None,
    balance_snapshot: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build structured key_signals block for DayProfile."""
    health_params = health_params or {}
    calendar_metrics = calendar_metrics or {}
    time_data = time_data or {}
    latest_mood = latest_mood or {}

    sleep_lastnight = health_param_value(
        health_params,
        "health_params_sleep_lastnight",
        HealthDataConstants.KEY_SLEEP_LASTNIGHT,
    )
    sleep_last3 = health_param_value(
        health_params,
        "health_params_sleep_last3nights",
        HealthDataConstants.KEY_SLEEP_LAST3NIGHTS,
    ) or []
    sleep_ref = sleep_lastnight
    if sleep_ref is None and sleep_last3:
        for v in sleep_last3:
            if v is not None:
                sleep_ref = v
                break

    steps_today = health_param_value(
        health_params,
        "health_params_steps_today",
        HealthDataConstants.KEY_STEPS_TODAY,
    )
    steps_goal = health_param_value(
        health_params,
        "health_params_steps_goal",
        HealthDataConstants.KEY_STEPS_GOAL,
    )
    steps_behind = (
        steps_today is not None
        and steps_goal is not None
        and steps_goal > 0
        and steps_today < steps_goal
    )

        # Enrichment: normalize the (min, max) tuple so consumers can use both
    # endpoints independently without re-validating the shape.
    sleep_hr_range = health_param_value(
        health_params,
        "health_params_sleep_hr_range",
        HealthDataConstants.KEY_SLEEP_HR_RANGE,
    )
    sleep_hr_min = sleep_hr_max = None
    if isinstance(sleep_hr_range, (tuple, list)) and len(sleep_hr_range) == 2:
        try:
            v0 = sleep_hr_range[0]
            v1 = sleep_hr_range[1]
            sleep_hr_min = int(v0) if v0 is not None else None
            sleep_hr_max = int(v1) if v1 is not None else None
        except (TypeError, ValueError):
            sleep_hr_min = sleep_hr_max = None

    return {
        "sleep": {
            "last_night": sleep_lastnight,
            "trend_3nights": sleep_last3,
            "quality": health_param_value(
                health_params,
                "health_params_sleep_quality",
                HealthDataConstants.KEY_SLEEP_QUALITY,
            ),
            "goal": health_param_value(
                health_params,
                "health_params_sleep_goal",
                HealthDataConstants.KEY_SLEEP_GOAL,
            ),
            "reference_hours": sleep_ref,
            "bedtime": health_param_value(
                health_params,
                "health_params_bedtime",
                "bedtime",
            )
            or health_param_value(
                health_params,
                "health_params_first_sleep_time",
                HealthDataConstants.KEY_FIRST_SLEEP_TIME,
            ),
            "wake_time": health_param_value(
                health_params,
                "health_params_wake_time",
                HealthDataConstants.KEY_WAKE_TIME,
            )
            or health_param_value(
                health_params,
                "health_params_last_wake_time",
                HealthDataConstants.KEY_LAST_WAKE_TIME,
            ),
            "sleep_score": health_param_value(
                health_params,
                "health_params_sleep_quality_score",
                HealthDataConstants.KEY_SLEEP_QUALITY_SCORE,
            ),
            # Enrichment — BE aggregates over the rolling week.
            "average_awake_hours": health_param_value(
                health_params,
                "health_params_average_awake_hours",
                HealthDataConstants.KEY_AVERAGE_AWAKE_HOURS,
            ),
            "average_total_sleep_hours": health_param_value(
                health_params,
                "health_params_average_total_sleep_hours",
                HealthDataConstants.KEY_AVERAGE_TOTAL_SLEEP_HOURS,
            ),
        },
        "steps": {
            "today": steps_today,
            "goal": steps_goal,
            "behind": steps_behind,
            # Enrichment — per-day healthScore. Either a flat list (HR
            # preferred, see prepare_health_data) or None. Lets the narrative
            # mention weekly patterns: "your health score held 78+ all week".
            "health_score_trend": health_param_value(
                health_params,
                "health_params_daily_health_score_trend",
                HealthDataConstants.KEY_DAILY_HEALTH_SCORE_TREND,
            ),
        },
        "calendar": {
            "density": calendar_metrics.get("calendar_density"),
            "back_to_back": calendar_metrics.get("back_to_back_count", 0),
            "free_slot_mins": time_data.get(TimeDataConstants.KEY_FREE_SLOT_LENGTH),
            "time_to_next_event": time_data.get(
                TimeDataConstants.KEY_TIME_TO_NEXT_EVENT
            ),
            # Enrichment — work-load, density, event aggregates.
            # All Optional/0 to stay graceful when BE omits a field.
            "work_events_hours_week": calendar_metrics.get(
                "work_events_hours_week"
            ),
            "work_events_hours_weekend": calendar_metrics.get(
                "work_events_hours_weekend"
            ),
            "work_load_high": calendar_metrics.get("work_load_high"),
            "event_count_today": (
                len(calendar_events)
                if isinstance(calendar_events, list)
                else calendar_metrics.get("event_count_today")
            ),
            "total_participant_count": calendar_metrics.get(
                "total_participant_count"
            )
            or _sum_participant_count(calendar_events),
            "events_with_location_count": calendar_metrics.get(
                "events_with_location_count"
            )
            or _count_events_with_location(calendar_events),
            "unique_locations": calendar_metrics.get("unique_locations")
            or _unique_locations(calendar_events),
        },
        "body": {
            "resting_hr": health_param_value(
                health_params,
                "health_params_resting_heart_rate",
                HealthDataConstants.KEY_RESTING_HEART_RATE,
            ),
            "baseline_hr": health_param_value(
                health_params,
                "health_params_baseline_resting_hr",
                HealthDataConstants.KEY_BASELINE_RESTING_HR,
            ),
            "stress_high": health_param_value(
                health_params,
                "health_params_stress_signal_high",
                "stress_signal_high",
            )
            or health_param_value(
                health_params,
                "health_params_stress_high",
                "stress_high",
            ),
            "mood": latest_mood.get("current_mood") or latest_mood.get("mood"),
            # Enrichment — cardio load + energy trend. Walking HR reflects
            # sub-max exertion; energy pct compares active burn vs monthly
            # baseline so the narrative can claim "your burn is up X%".
            "walking_hr_avg": health_param_value(
                health_params,
                "health_params_walking_heart_rate_avg",
                HealthDataConstants.KEY_WALKING_HEART_RATE_AVG,
            ),
            "energy_week_vs_prior_month_pct": health_param_value(
                health_params,
                "health_params_energy_week_vs_prior_month_pct",
                HealthDataConstants.KEY_ENERGY_WEEK_VS_PRIOR_MONTH_PCT,
            ),
            "mood_score_avg": latest_mood.get("mood_score_avg")
            or latest_mood.get("m_mood_score_avg"),
            "mood_score_min": latest_mood.get("mood_score_min")
            or latest_mood.get("m_mood_score_min"),
            "mood_score_max": latest_mood.get("mood_score_max")
            or latest_mood.get("m_mood_score_max"),
            "mood_variance": latest_mood.get("mood_variance")
            or latest_mood.get("m_mood_variance"),
            "mood_first": latest_mood.get("mood_first")
            or latest_mood.get("m_first_mood"),
            "mood_sequence": latest_mood.get("mood_sequence")
            or latest_mood.get("m_mood_sequence"),
            "mood_log_count": latest_mood.get("mood_log_count")
            or latest_mood.get("m_log_count"),
            "mood_has_notes": bool(latest_mood.get("notes")),
            "mood_date": (
                latest_mood.get("current_mood_date")
                or latest_mood.get("moodDate")
                or latest_mood.get("mood_date")
            ),
        },
        "balance": _extract_balance_block(balance_snapshot),
    }


def _extract_balance_block(balance_snapshot: Any) -> Dict[str, Any]:
    # Pydantic v2 path — use model_dump for clean nested types.
    if hasattr(balance_snapshot, "model_dump"):
        try:
            snap = balance_snapshot.model_dump()
        except Exception:
            snap = {}
    elif isinstance(balance_snapshot, dict):
        snap = balance_snapshot
    else:
        snap = {}

    # Defensive per-field extraction so a partial payload still surfaces
    # whatever the BE actually returned.
    def _g(key: str) -> Any:
        return snap.get(key)

    return {
        "score": _g("balance_score"),
        "productivity_score": _g("productivity_score"),
        "health_score": _g("health_score"),
        "finance_score": _g("finance_score"),
        "date": _g("balance_date"),
        "timezone": _g("balance_timezone"),
        # Weekly aggregates — narrate "you averaged X this week"
        "score_7d_avg": _g("balance_score_7d_avg"),
        "productivity_7d_avg": _g("productivity_score_7d_avg"),
        "health_7d_avg": _g("health_score_7d_avg"),
        "finance_7d_avg": _g("finance_score_7d_avg"),
        "score_7d_std": _g("balance_score_7d_std"),
        # Trend delta (positive=improving, negative=declining)
        "trend_delta": _g("balance_score_trend_delta"),
        # Good-days count lets narrative say "you had 5 strong days"
        "good_days_7d": _g("balance_score_7d_good_days"),
        "total_days_7d": _g("balance_score_7d_total_days"),
    }

def _sum_participant_count(
    events: Optional[List[Dict[str, Any]]]
) -> Optional[int]:
    if not isinstance(events, list) or not events:
        return None
    total = 0
    saw_any = False
    for e in events:
        if not isinstance(e, dict):
            continue
        p = e.get("participants")
        if isinstance(p, list) and p:
            total += len(p)
            saw_any = True
    return total if saw_any else 0


def _count_events_with_location(
    events: Optional[List[Dict[str, Any]]]
) -> Optional[int]:
    if not isinstance(events, list) or not events:
        return None
    return sum(
        1
        for e in events
        if isinstance(e, dict) and str(e.get("location") or "").strip()
    )


def _unique_locations(
    events: Optional[List[Dict[str, Any]]]
) -> List[str]:
    if not isinstance(events, list) or not events:
        return []
    seen: List[str] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        loc = str(e.get("location") or "").strip()
        if loc and loc not in seen:
            seen.append(loc)
    return seen


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


def build_data_gaps(
    health_params: Optional[Dict[str, Any]] = None,
    historical_snapshots: Optional[List[Any]] = None,
) -> List[str]:
    """Build list of data gaps for narrative warning.

    Checks both health_params (from API) and historical_snapshots (local cache)
    to properly detect when data is truly missing vs when it exists but wasn't
    fetched due to API errors.
    """
    health_params = health_params or {}
    historical_snapshots = historical_snapshots or []
    gaps: List[str] = []

    # Helper to get latest value from historical_snapshots
    def _get_from_snapshots(field: str) -> Any:
        """Get the most recent non-None value for a field from snapshots."""
        if not historical_snapshots:
            return None
        # Sort by date descending to get most recent
        sorted_snaps = sorted(
            [s for s in historical_snapshots if s.get("snapshot_date")],
            key=lambda x: x.get("snapshot_date", ""),
            reverse=True,
        )
        for snap in sorted_snaps:
            val = snap.get(field)
            if val is not None:
                return val
        return None

    # Check sleep_lastnight
    sleep_lastnight_in_hp = health_param_value(
        health_params,
        "health_params_sleep_lastnight",
        HealthDataConstants.KEY_SLEEP_LASTNIGHT,
    )
    if sleep_lastnight_in_hp is None:
        sleep_from_snap = _get_from_snapshots(CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS)
        if sleep_from_snap is not None:
            gaps.append(
                f"sleep_lastnight unavailable — using fallback from snapshots ({sleep_from_snap}h)"
            )
        else:
            sleep_last3 = health_param_value(
                health_params,
                "health_params_sleep_last3nights",
                HealthDataConstants.KEY_SLEEP_LAST3NIGHTS,
            ) or []
            if any(v is not None for v in sleep_last3):
                gaps.append(
                    "sleep_lastnight unavailable — using most recent value from 3-night trend"
                )
            else:
                gaps.append("sleep_lastnight unavailable — no recent sleep data")

    # Check steps_today
    if health_param_value(
        health_params,
        "health_params_steps_today",
        HealthDataConstants.KEY_STEPS_TODAY,
    ) is None:
        steps_from_snap = _get_from_snapshots(CanonicalField.STEP_COUNT_ONE_DAY)
        if steps_from_snap is not None:
            gaps.append(
                f"steps_today unavailable — using fallback from snapshots ({steps_from_snap})"
            )
        else:
            gaps.append("steps_today unavailable")

    # Check sleep_3night_trend
    sleep_last3 = health_param_value(
        health_params,
        "health_params_sleep_last3nights",
        HealthDataConstants.KEY_SLEEP_LAST3NIGHTS,
    ) or []
    if not any(v is not None for v in sleep_last3):
        # Try to get from snapshots
        sleep_vals_from_snap = []
        for snap in sorted(
            [s for s in historical_snapshots if s.get("snapshot_date")],
            key=lambda x: x.get("snapshot_date", ""),
            reverse=True,
        )[:3]:
            val = snap.get(CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS) or snap.get("h_sleep_hours")
            if val is not None:
                sleep_vals_from_snap.append(val)
        if sleep_vals_from_snap:
            gaps.append(
                f"sleep_3night_trend unavailable — using fallback from snapshots ({sleep_vals_from_snap})"
            )
        else:
            gaps.append("sleep_3night_trend unavailable — sleep_trend signal will be STABLE")

    # Check steps_streak
    if health_param_value(
        health_params,
        "health_params_steps_streak",
        HealthDataConstants.KEY_STEPS_STREAK,
    ) is None:
        gaps.append("steps_streak unavailable")

    # Check bedtime_streak
    if health_param_value(
        health_params,
        "health_params_bedtime_streak",
        HealthDataConstants.KEY_BEDTIME_STREAK,
    ) is None:
        gaps.append("bedtime_streak unavailable")

    # Check HRV score
    if health_param_value(
        health_params,
        "health_params_hrv_score",
        HealthDataConstants.KEY_HRV_SCORE,
    ) is None:
        gaps.append("hrv_score unavailable — cardio mechanism will fall back to resting HR")

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
