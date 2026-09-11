"""Deterministic Context Signal Builder (Processor layer).

Detects situational context that should influence Narrative AI behavior.
Does NOT generate user-facing text or decide the final insight.

Phase 1 signals (docs/techplans/insights/health/context-signal.md):
  sleep_window, weekend, work_hours, recent_workout
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from insights.health.health_processor.common.utils import health_param_value
from insights.health.health_processor.context.constants import (
    CONTEXT_PRIORITY_RECENT_WORKOUT,
    CONTEXT_PRIORITY_SLEEP_WINDOW,
    CONTEXT_PRIORITY_WEEKEND,
    CONTEXT_PRIORITY_WORK_HOURS,
    CONTEXT_RECENT_WORKOUT_MAX_HOURS,
    CONTEXT_RECENT_WORKOUT_MIN_DURATION,
)
from insights.health.health_processor.context.goal_context import (
    build_bedtime_goals,
    build_work_hours_goals,
)
from insights.schemas.processed_context import (
    ContextSignalItem,
    ExtractedSignals,
)
from services.executor.constant import HealthDataConstants


def _parse_hhmm(raw: Any) -> Optional[time]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    # Accept HH:MM or HH:MM:SS
    parts = text.replace(".", ":").split(":")
    try:
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return time(hour=hour, minute=minute)
    except (TypeError, ValueError, IndexError):
        return None
    return None


def _in_clock_window(now: time, start: time, end: time) -> bool:
    """True if ``now`` is inside [start, end), supporting midnight wrap."""
    if start == end:
        return False
    if start < end:
        return start <= now < end
    # Crosses midnight (e.g. 22:30 → 07:00)
    return now >= start or now < end


def _resolve_local_now(signals: ExtractedSignals) -> Tuple[datetime, str]:
    raw = signals.raw_data if isinstance(signals.raw_data, dict) else {}
    time_data = raw.get("time_data") if isinstance(raw.get("time_data"), dict) else {}
    tz_name = (
        raw.get("time_data_timezone")
        or raw.get("timezone")
        or time_data.get("timezone")
    )
    if not tz_name:
        raise KeyError(
            "timezone missing in raw_data/time_data; expected resolver output "
            "(payload → profile → DEFAULT_TIMEZONE) at extract() time."
        )
    try:
        tz = ZoneInfo(str(tz_name))
    except Exception:
        tz = ZoneInfo("UTC")
        tz_name = "UTC"

    # Prefer wall-clock now in user TZ so time-sensitive signals stay fresh
    # even when health metrics are served from a 30-minute cache.
    now = datetime.now(tz)

    # If extract provided a current_time and it is within a few minutes, prefer it
    # (keeps debug/demo pipelines deterministic).
    iso = (
        time_data.get("time_data_current_time_iso")
        or time_data.get("currentTimeIso")
        or (signals.meta.current_time if signals.meta else None)
        or raw.get("current_time")
    )
    if isinstance(iso, datetime):
        candidate = iso if iso.tzinfo else iso.replace(tzinfo=tz)
        candidate = candidate.astimezone(tz)
        if abs((now - candidate).total_seconds()) <= 5 * 60:
            now = candidate
    elif isinstance(iso, str) and iso.strip():
        try:
            text = iso.strip().replace("Z", "+00:00")
            candidate = datetime.fromisoformat(text)
            if candidate.tzinfo is None:
                candidate = candidate.replace(tzinfo=tz)
            candidate = candidate.astimezone(tz)
            if abs((now - candidate).total_seconds()) <= 5 * 60:
                now = candidate
        except ValueError:
            pass

    return now, str(tz_name)


def _item(
    *,
    status: str,
    priority: str,
    evidence: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ContextSignalItem:
    return ContextSignalItem(
        status=status,
        priority=priority,
        evidence=evidence or {},
        metadata=metadata or {},
    )


class ContextSignalBuilder:
    """Build / refresh deterministic context signals onto health_signals."""

    @classmethod
    def build(cls, signals: ExtractedSignals) -> Dict[str, ContextSignalItem]:
        now, tz_name = _resolve_local_now(signals)
        out: Dict[str, ContextSignalItem] = {}
        out["sleep_window"] = cls._sleep_window(signals, now, tz_name)
        out["weekend"] = cls._weekend(signals, now, tz_name)
        out["work_hours"] = cls._work_hours(signals, now, tz_name)
        out["recent_workout"] = cls._recent_workout(signals, now, tz_name)
        return out

    @classmethod
    def attach(cls, signals: ExtractedSignals) -> ExtractedSignals:
        """Compute and attach context_signals onto health_signals (in place)."""
        if signals.health_signals is None:
            return signals
        signals.health_signals.context_signals = cls.build(signals)
        return signals

    @classmethod
    def refresh(cls, signals: ExtractedSignals) -> ExtractedSignals:
        now, _tz = _resolve_local_now(signals)
        if signals.meta is not None:
            signals.meta.current_time = now.isoformat()
        return cls.attach(signals)

    # ── detectors ──────────────────────────────────────────────────────────

    @staticmethod
    def _sleep_window(
        signals: ExtractedSignals, now: datetime, tz_name: str
    ) -> ContextSignalItem:
        goals = build_bedtime_goals(signals)
        start = _parse_hhmm(goals.get("bedtime_start"))
        end = _parse_hhmm(goals.get("bedtime_end"))

        # Fallbacks from sleep flats / time_data
        if start is None or end is None:
            raw = signals.raw_data if isinstance(signals.raw_data, dict) else {}
            time_data = (
                raw.get("time_data") if isinstance(raw.get("time_data"), dict) else {}
            )
            sleep = signals.health_signals.sleep if signals.health_signals else None
            bedtime_timing = sleep.signals.bedtime_timing if sleep else None
            bedtime_metrics = bedtime_timing.metrics if bedtime_timing else None
            bedtime_goal_start = (
                bedtime_metrics.get("goal_start") if isinstance(bedtime_metrics, dict) else
                getattr(bedtime_metrics, "goal_start", None) if bedtime_metrics else None
            )
            bedtime_goal_end = (
                bedtime_metrics.get("goal_end") if isinstance(bedtime_metrics, dict) else
                getattr(bedtime_metrics, "goal_end", None) if bedtime_metrics else None
            )
            start = start or _parse_hhmm(
                time_data.get("time_data_bedtime_start_str")
                or time_data.get("bedtime_start")
                or bedtime_goal_start
            )
            end = end or _parse_hhmm(
                time_data.get("time_data_bedtime_end_str")
                or time_data.get("bedtime_end")
                or bedtime_goal_end
            )

        current_hhmm = now.strftime("%H:%M")
        evidence: Dict[str, Any] = {
            "current_time": current_hhmm,
            "time_data_timezone": tz_name,
        }
        if start:
            evidence["sleep_window_start"] = start.strftime("%H:%M")
        if end:
            evidence["sleep_window_end"] = end.strftime("%H:%M")

        if start is None or end is None:
            return _item(
                status="inactive",
                priority=CONTEXT_PRIORITY_SLEEP_WINDOW,
                evidence=evidence,
                metadata={"reason": "sleep_window_not_configured"},
            )

        active = _in_clock_window(now.time(), start, end)
        # Also treat explicit bedtime time_phase as active when window is configured
        if not active and signals.meta and signals.meta.time_phase == "bedtime":
            active = True
            evidence["time_phase"] = "bedtime"

        return _item(
            status="active" if active else "inactive",
            priority=CONTEXT_PRIORITY_SLEEP_WINDOW,
            evidence=evidence,
        )

    @staticmethod
    def _weekend(
        signals: ExtractedSignals, now: datetime, tz_name: str
    ) -> ContextSignalItem:
        day_name = now.strftime("%A")
        is_weekend = now.weekday() >= 5  # Sat=5, Sun=6
        # Prefer meta when present and date aligns
        if signals.meta and signals.meta.day_of_week:
            meta_day = str(signals.meta.day_of_week)
            if meta_day.lower()[:3] == day_name.lower()[:3]:
                is_weekend = bool(signals.meta.is_weekend)
                day_name = meta_day
        return _item(
            status="active" if is_weekend else "inactive",
            priority=CONTEXT_PRIORITY_WEEKEND,
            evidence={"day_of_week": day_name, "time_data_timezone": tz_name},
        )

    @staticmethod
    def _work_hours(
        signals: ExtractedSignals, now: datetime, tz_name: str
    ) -> ContextSignalItem:
        goals = build_work_hours_goals(signals)
        start = _parse_hhmm(goals.get("active_hours_start"))
        end = _parse_hhmm(goals.get("active_hours_end"))

        current_hhmm = now.strftime("%H:%M")
        evidence: Dict[str, Any] = {
            "current_time": current_hhmm,
            "time_data_timezone": tz_name,
        }
        if start:
            evidence["work_start"] = start.strftime("%H:%M")
        if end:
            evidence["work_end"] = end.strftime("%H:%M")

        if start is None or end is None:
            # Fall back to explicit in_active_window flag when only one bound exists
            in_active = goals.get("time_data_in_active_window")
            if in_active is not None and (start is not None or end is not None):
                return _item(
                    status="active" if in_active else "inactive",
                    priority=CONTEXT_PRIORITY_WORK_HOURS,
                    evidence=evidence,
                    metadata={"reason": "partial_work_hours_config"},
                )
            return _item(
                status="inactive",
                priority=CONTEXT_PRIORITY_WORK_HOURS,
                evidence=evidence,
                metadata={"reason": "work_hours_not_configured"},
            )

        active = _in_clock_window(now.time(), start, end)
        # Weekend typically outside work hours unless explicitly configured otherwise
        if now.weekday() >= 5:
            active = False
            evidence["suppressed_by"] = "weekend"

        return _item(
            status="active" if active else "inactive",
            priority=CONTEXT_PRIORITY_WORK_HOURS,
            evidence=evidence,
        )

    @staticmethod
    def _recent_workout(
        signals: ExtractedSignals, now: datetime, tz_name: str
    ) -> ContextSignalItem:
        hs = signals.health_signals
        raw = signals.raw_data if isinstance(signals.raw_data, dict) else {}
        health_params = (
            raw.get("health_params") if isinstance(raw.get("health_params"), dict) else {}
        )

        duration_min: Optional[float] = None
        activity_type: Optional[str] = None
        if hs and hs.steps:
            if hs.steps.workout_duration_min:
                duration_min = float(hs.steps.workout_duration_min)
            activity_type = hs.steps.workout_activity_type
        if (duration_min is None or duration_min <= 0) and hs and hs.energy:
            if hs.energy.workout_duration_min:
                duration_min = float(hs.energy.workout_duration_min)
            activity_type = activity_type or hs.energy.workout_activity_type
        if duration_min is None or duration_min <= 0:
            workout_sec = health_param_value(
                health_params,
                "health_params_workout_duration",
                HealthDataConstants.KEY_WORKOUT_DURATION,
            )
            if isinstance(workout_sec, (int, float)) and workout_sec > 0:
                duration_min = float(workout_sec) / 60.0
            activity_type = activity_type or health_param_value(
                health_params,
                "health_params_workout_activity_type",
                HealthDataConstants.KEY_WORKOUT_ACTIVITY_TYPE,
            )

        high_load = bool(
            hs
            and hs.period_aggregates
            and hs.period_aggregates.workout_high_recent_load
        )

        hours_since: Optional[float] = None
        end_raw = (
            health_params.get("workout_end_time")
            or health_params.get("workoutEndTime")
            or health_params.get("last_workout_end")
        )
        if isinstance(end_raw, str) and end_raw.strip():
            try:
                end_dt = datetime.fromisoformat(end_raw.strip().replace("Z", "+00:00"))
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=now.tzinfo)
                hours_since = round(
                    max((now - end_dt.astimezone(now.tzinfo)).total_seconds() / 3600.0, 0.0),
                    1,
                )
            except ValueError:
                hours_since = None

        evidence: Dict[str, Any] = {"time_data_timezone": tz_name}
        if duration_min is not None and duration_min > 0:
            evidence["duration_minutes"] = round(duration_min, 1)
        if activity_type:
            evidence["activity_type"] = activity_type
        if hours_since is not None:
            evidence["hours_since_workout"] = hours_since
        if high_load:
            evidence["workout_high_recent_load"] = True

        significant = bool(
            (
                duration_min is not None
                and duration_min >= CONTEXT_RECENT_WORKOUT_MIN_DURATION
            )
            or high_load
        )
        recent_enough = (
            hours_since is None or hours_since <= CONTEXT_RECENT_WORKOUT_MAX_HOURS
        )
        active = significant and recent_enough and (
            hours_since is not None
            or (duration_min is not None and duration_min > 0)
            or high_load
        )

        meta: Dict[str, Any] = {}
        if active and hours_since is None:
            meta["hours_since_estimated"] = True
            meta["note"] = "workout end time unavailable; treated as today session"

        return _item(
            status="active" if active else "inactive",
            priority=CONTEXT_PRIORITY_RECENT_WORKOUT,
            evidence=evidence,
            metadata=meta,
        )
