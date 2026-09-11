"""Schedule-boundary goals derived from profile + time_data.

Used to give the productivity Q&A prompt the user's bedtime and active/work-hours
window so a suggested action never lands outside those boundaries.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from insights.health.health_processor.common.utils import health_param_value
from insights.schemas.processed_context import ExtractedSignals
from services.executor.constant import HealthDataConstants


def _dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items() if v is not None and v != ""}
    if isinstance(obj, list):
        return [_dump(x) for x in obj if x is not None]
    return obj

def _coalesce_goals_lists(profile: Mapping[str, Any], raw: Mapping[str, Any]) -> list[list[Any]]:
    out: list[list[Any]] = []
    seen: set[int] = set()
    for src, keys in (
        (profile, ("user_goals", "goals")),
        (raw, ("goals",)),
    ):
        if not isinstance(src, Mapping):
            continue
        for key in keys:
            val = src.get(key)
            if isinstance(val, list) and id(val) not in seen:
                seen.add(id(val))
                out.append(val)
    return out

def _profile_goals(profile: Mapping[str, Any]) -> list[Any]:
    val = profile.get("user_goals")
    if isinstance(val, list):
        return val
    val = profile.get("goals")
    if isinstance(val, list):
        return val
    return []

def _raw_goals(raw: Mapping[str, Any]) -> list[Any]:
    val = raw.get("goals")
    return val if isinstance(val, list) else []


def _merged_profile(signals: Optional[ExtractedSignals]) -> Dict[str, Any]:
    if not signals:
        return {}
    typed = _dump(signals.user_profile) or {}
    raw: Dict[str, Any] = {}
    if isinstance(signals.raw_data, dict):
        cand = signals.raw_data.get("user_profile") or {}
        if isinstance(cand, dict):
            raw = cand
    merged = dict(raw)
    if isinstance(typed, dict):
        for k, v in typed.items():
            if v is not None and v != "" and v != []:
                merged[k] = v
    return merged


def build_bedtime_goals(
    signals: Optional[ExtractedSignals],
) -> Dict[str, Any]:
    """Bedtime / sleep-duration goals built from profile + time_data.

    No extra health-device fetch required.
    """
    out: Dict[str, Any] = {}
    if not signals:
        return out

    raw = signals.raw_data if isinstance(signals.raw_data, dict) else {}
    profile = _merged_profile(signals)
    time_data = raw.get("time_data") if isinstance(raw.get("time_data"), dict) else {}
    health_params = (
        raw.get("health_params") if isinstance(raw.get("health_params"), dict) else {}
    )

    bedtime_start = (
        time_data.get("time_data_bedtime_start_str")
        or profile.get("bedtime_start")
        or profile.get("bedtimeStart")
    )
    bedtime_end = (
        time_data.get("time_data_bedtime_end_str")
        or profile.get("bedtime_end")
        or profile.get("bedtimeEnd")
    )
    time_to_bedtime = time_data.get("time_data_time_to_bedtime")

    sleep_goal_h = health_param_value(
        health_params,
        "health_params_sleep_goal",
        HealthDataConstants.KEY_SLEEP_GOAL,
    )
    sleep_target = time_data.get("time_data_sleep_target")
    if sleep_goal_h is None and isinstance(sleep_target, dict):
        sleep_goal_h = sleep_target.get("value") or sleep_target.get("target")
    sleep_duration_sig = (
        signals.health_signals.sleep.signals.sleep_duration
        if signals.health_signals and signals.health_signals.sleep
        else None
    )
    sleep_goal_h = (
        sleep_duration_sig.metrics.get("target_hours")
        if sleep_duration_sig
        else None
    )

    goal_notes: list[str] = []
    goal_lists: list = []
    for goal_list in _coalesce_goals_lists(profile, raw):
        goal_lists.append(goal_list)
    for goal_list in goal_lists:
        for goal in goal_list:
            if not isinstance(goal, dict):
                continue
            cat = (goal.get("category_code") or goal.get("category") or "").upper()
            action = (goal.get("action_code") or goal.get("action") or "").upper()
            name = goal.get("goal_name") or goal.get("name") or ""
            target = goal.get("target_value") or goal.get("targetValue")
            if cat == "SLEEP" and sleep_goal_h is None and target is not None:
                sleep_goal_h = target
            if "BEDTIME" in action or "BEDTIME" in name.upper() or "SLEEP" in cat:
                note = name or action or cat
                if target is not None:
                    note = f"{note} (target={target})"
                if str(note) not in goal_notes:
                    goal_notes.append(str(note))

    if bedtime_start:
        out["bedtime_start"] = bedtime_start
    if bedtime_end:
        out["bedtime_end"] = bedtime_end
    if time_to_bedtime is not None:
        out["time_to_bedtime_mins"] = time_to_bedtime
    if sleep_goal_h is not None:
        out["sleep_goal_hours"] = sleep_goal_h
    if goal_notes:
        out["related_goals"] = goal_notes[:5]

    if bedtime_start or sleep_goal_h is not None:
        bits = []
        if bedtime_start:
            bits.append(f"bedtime goal ~{bedtime_start}")
        if bedtime_end:
            bits.append(f"wake window ~{bedtime_end}")
        if sleep_goal_h is not None:
            bits.append(f"sleep duration goal ~{sleep_goal_h}h")
        if time_to_bedtime is not None:
            bits.append(f"~{time_to_bedtime} min until bedtime start")
        out["summary"] = "; ".join(bits)

    return out


def build_work_hours_goals(
    signals: Optional[ExtractedSignals],
) -> Dict[str, Any]:
    """Active / work-hours window goals from profile + time_data.

    Includes the productivity work-hours target when available.
    """
    out: Dict[str, Any] = {}
    if not signals:
        return out

    raw = signals.raw_data if isinstance(signals.raw_data, dict) else {}
    profile = _merged_profile(signals)
    time_data = raw.get("time_data") if isinstance(raw.get("time_data"), dict) else {}

    active_start = (
        time_data.get("time_data_active_start_str")
        or time_data.get("active_hours_start_time")
        or profile.get("active_hours_start_time")
        or profile.get("activeHoursStartTime")
    )
    active_end = (
        time_data.get("active_hours_end_time")
        or time_data.get("time_data_active_end_str")
        or profile.get("active_hours_end_time")
        or profile.get("activeHoursEndTime")
        or (signals.meta.active_hours_end_time if signals.meta else None)
    )
    in_active = time_data.get("time_data_in_active_window")

    if active_start:
        out["active_hours_start"] = active_start
    if active_end:
        out["active_hours_end"] = active_end
    if in_active is not None:
        out["time_data_in_active_window"] = bool(in_active)

    # Weekly work-hours target / load when productivity signals exist
    if signals.productivity_signals is not None:
        ps = signals.productivity_signals
        if ps.work_hours_7d_target is not None:
            out["work_hours_7d_target"] = ps.work_hours_7d_target
        if ps.work_hours_7d_scheduled_avg is not None:
            out["work_hours_7d_scheduled_avg"] = ps.work_hours_7d_scheduled_avg
        if ps.work_hours_overload_days_7d:
            out["work_hours_overload_days_7d"] = ps.work_hours_overload_days_7d
        if ps.work_hours_underload_days_7d:
            out["work_hours_underload_days_7d"] = ps.work_hours_underload_days_7d

    # Related WORK / PRODUCTIVITY goals from profile
    work_notes: list[str] = []
    for goal in _profile_goals(profile) + _raw_goals(raw):
        if not isinstance(goal, dict):
            continue
        cat = (goal.get("category_code") or goal.get("category") or "").upper()
        action = (goal.get("action_code") or goal.get("action") or "").upper()
        name = goal.get("goal_name") or goal.get("name") or ""
        target = goal.get("target_value") or goal.get("targetValue")
        blob = f"{cat} {action} {name}".upper()
        if any(
            k in blob
            for k in ("WORK", "PRODUCTIVITY", "FOCUS", "DEEP_WORK", "HOURS")
        ):
            note = name or action or cat
            if target is not None:
                note = f"{note} (target={target})"
            if str(note) not in work_notes:
                work_notes.append(str(note))
    if work_notes:
        out["related_goals"] = work_notes[:5]

    if active_start or active_end or out.get("work_hours_7d_target") is not None:
        bits = []
        if active_start and active_end:
            bits.append(f"active hours {active_start}–{active_end}")
        elif active_end:
            bits.append(f"active hours end ~{active_end}")
        elif active_start:
            bits.append(f"active hours start ~{active_start}")
        if out.get("work_hours_7d_target") is not None:
            bits.append(f"work-hours target ~{out['work_hours_7d_target']}h/day")
        if out.get("work_hours_7d_scheduled_avg") is not None:
            bits.append(f"scheduled avg ~{out['work_hours_7d_scheduled_avg']}h/day")
        if in_active is not None:
            bits.append("in active window" if in_active else "outside active window")
        out["summary"] = "; ".join(bits)

    return out
