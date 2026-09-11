from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TARGET_HOURS = 8.0
OVERLOAD_MULTIPLIER = 1.20
UNDERLOAD_MULTIPLIER = 0.50


def _parse_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _filter_to_recent_7d(
    entries: List[Dict[str, Any]], today: date
) -> List[Dict[str, Any]]:
    """Keep only entries that fall within today..today+6d window."""
    if not entries:
        return []
    # Rolling 7-day window: today and the previous 6 days.
    cutoff_low = today.toordinal() - 6
    cutoff_high = today.toordinal()
    filtered: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        d_raw = (
            entry.get("date")
            or entry.get("day")
            or entry.get("dateKey")
        )
        if not d_raw or not isinstance(d_raw, str):
            # If no date, assume it belongs to the window — caller controls.
            filtered.append(entry)
            continue
        try:
            d = date.fromisoformat(d_raw[:10])
        except ValueError:
            filtered.append(entry)
            continue
        if cutoff_low <= d.toordinal() <= cutoff_high:
            filtered.append(entry)
    return filtered


def compute_work_hours_signals(
    work_hours_7d: List[Dict[str, Any]],
    today: Optional[date] = None,
) -> Dict[str, Any]:
    """
    Returns primitives safe to splat onto ProductivitySignalsBlock.

    Schema:
        {
          "work_hours_7d_scheduled_avg": Optional[float],
          "work_hours_7d_target": Optional[float],
          "work_hours_overload_days_7d": int,
          "work_hours_underload_days_7d": int,
        }
    """
    if not work_hours_7d:
        return {
            "work_hours_7d_scheduled_avg": None,
            "work_hours_7d_target": None,
            "work_hours_overload_days_7d": 0,
            "work_hours_underload_days_7d": 0,
        }

    if today is not None:
        entries = _filter_to_recent_7d(work_hours_7d, today)
    else:
        entries = [e for e in work_hours_7d if isinstance(e, dict)]

    scheduled_vals: List[float] = []
    target_vals: List[float] = []
    overload_days = 0
    underload_days = 0

    for entry in entries:
        scheduled = _parse_float(
            entry.get("scheduledHours")
            or entry.get("totalHours")           # BE renamed: scheduledHours → totalHours
            or entry.get("scheduled_hours")
            or entry.get("hours")
        )
        target = _parse_float(
            entry.get("targetHours")
            or entry.get("target_hours")
            or entry.get("dailyTarget")
        )
        if scheduled is not None:
            scheduled_vals.append(scheduled)
        if target is not None:
            target_vals.append(target)

        # Threshold check requires both, otherwise default to safe fallback.
        if scheduled is None:
            continue
        effective_target = (
            target if target is not None and target > 0 else DEFAULT_TARGET_HOURS
        )
        if scheduled > effective_target * OVERLOAD_MULTIPLIER:
            overload_days += 1
        elif scheduled < effective_target * UNDERLOAD_MULTIPLIER:
            underload_days += 1

    def _avg(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    return {
        "work_hours_7d_scheduled_avg": _avg(scheduled_vals),
        "work_hours_7d_target": _avg(target_vals),
        "work_hours_overload_days_7d": overload_days,
        "work_hours_underload_days_7d": underload_days,
    }