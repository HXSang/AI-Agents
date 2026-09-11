import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from insights.health.canonical_field_mapping import CanonicalField
from insights.health.health_processor.common.utils import health_param_value

from insights.helpers.day_profile import build_data_gaps
from insights.helpers.day_profile import build_key_signals
from insights.helpers.day_profile import compute_hard_constraints
from insights.helpers.day_profile import compute_time_phase
from insights.helpers.empty_value_cleanser import (
    DEFAULT_KEEP_TOP_LEVEL_LIST_KEYS,
    DISPLAY_DROP_EMPTY_LIST_KEYS,
    clean_payload,
)

_PROCESSOR_DROP_KEYS = DISPLAY_DROP_EMPTY_LIST_KEYS

from insights.processors.behavioral_pattern_processor import BehavioralPatternProcessor
from insights.processors.calendar_intelligence_processor import CalendarIntelligenceProcessor
from insights.processors.finance_signal_processor import FinanceSignalProcessor
from insights.processors.goal_progress_processor import GoalProgressProcessor
from insights.health.health_processor.signal_processor import HealthSignalProcessor
from insights.processors.historical_trends_processor import HistoricalTrendsProcessor
from insights.processors.productivity_signal_processor import ProductivitySignalProcessor

from insights.schemas.processed_context import DynamicSignalsBlock
from insights.schemas.processed_context import ExtractedSignals
from insights.schemas.processed_context import MetaBlock
from insights.schemas.processed_context import StaticSignalsBlock

from zoneinfo import ZoneInfo
from services.executor.constant import HealthDataConstants

logger = logging.getLogger(__name__)

_MOOD_RANK: Dict[str, int] = {
    "terrible": 0,
    "sad": 1,
    "low": 1,
    "neutral": 2,
    "okay": 2,
    "good": 3,
    "great": 4,
    "happy": 4,
}


def _parse_today(time_data: Dict[str, Any]):
    raw = time_data.get("time_data_current_time_iso")
    if not raw:
        raise ValueError("DataProcessor requires 'current_time_iso' in time_data")
    return datetime.fromisoformat(raw).date()


def _snapshot_field(item: Any, field: str) -> Any:
    """Read a field off a snapshot row regardless of dict or object shape."""
    if isinstance(item, dict):
        return item.get(field)
    return getattr(item, field, None)


def _snapshot_date(item: Any) -> Optional[date]:
    raw = _snapshot_field(item, "daily_snapshot_date")
    if not raw:
        return None
    try:
        if isinstance(raw, date) and not isinstance(raw, datetime):
            return raw
        return datetime.fromisoformat(str(raw)).date()
    except Exception:
        return None


def _safe_get(obj: Any, *keys, default=None):
    cur = obj
    for k in keys:
        if cur is None:
            return default
        try:
            if isinstance(cur, dict):
                cur = cur.get(k, default)
            else:
                cur = getattr(cur, k, default)
        except Exception:
            return default
        if cur is None:
            return default
    return cur


def _format_hhmm(value: Any) -> Optional[str]:
    """Best-effort 'HH:MM' formatter for ISO datetimes or already-HH:MM strings."""
    if not value:
        return None
    s = str(value)
    try:
        if "T" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.strftime("%H:%M")
        if len(s) >= 5 and s[2] == ":":
            return s[:5]
    except Exception:
        return s
    return s

def _date_tag(field_name: str, meta: Any) -> str:
    if meta is None:
        return ""
    staleness_days = _safe_get(meta, f"{field_name}_data_staleness_days")
    last_date = _safe_get(meta, f"{field_name}_last_data_date")
    if staleness_days == 999 or staleness_days is None:
        return "[NO DATA]"
    if staleness_days == 0:
        return "[TODAY fresh]"
    date_part = f" (last {last_date})" if last_date else ""
    return f"[STALE {staleness_days}d{date_part}]"


def _aggregate_tag() -> str:
    """Tag for weekly / rolling aggregates that should never be quoted as
    a single day's reading.
    """
    return "[PAST 3d / weekly — NOT today's reading]"


def _is_metric_date_fresh(
    latest_hm: Optional[Dict[str, Any]],
    current_date: Optional[str],
) -> bool:
    if not isinstance(latest_hm, dict) or not current_date:
        return False
    md = str(latest_hm.get("metric_date") or "").strip()
    if not md:
        return False
    return md == str(current_date).strip()


def _backfill_health_params(
    health_params: Dict[str, Any],
    raw_data: Dict[str, Any],
) -> None:
    if not isinstance(health_params, dict):
        return

    current_date = None
    time_data = _safe_get(raw_data, "time_data")
    if isinstance(time_data, dict):
        current_date = time_data.get("time_data_current_date")

    today_stats = _safe_get(raw_data, "today_health_stats") or {}
    profile = _safe_get(raw_data, "user_profile") or {}
    latest_hm = _safe_get(profile, "latest_health_metrics") or {}
    snapshots = _safe_get(raw_data, "historical_snapshots") or []

    today_local_date: Optional[date] = None
    if isinstance(current_date, date) and not isinstance(current_date, datetime):
        today_local_date = current_date
    elif isinstance(current_date, str):
        try:
            today_local_date = datetime.fromisoformat(current_date).date()
        except Exception:
            today_local_date = None
    if today_local_date is None:
        cur_dt_raw = (
            (isinstance(time_data, dict) and time_data.get("current_dt"))
            or (isinstance(time_data, dict) and time_data.get("current_time_iso"))
        )
        tz_name = (
            (isinstance(time_data, dict) and time_data.get("timezone"))
            or raw_data.get("timezone")
            or "UTC"
        )
        if cur_dt_raw:
            try:
                cur_dt = cur_dt_raw if isinstance(cur_dt_raw, datetime) else datetime.fromisoformat(str(cur_dt_raw))
                if cur_dt.tzinfo is None:
                    cur_dt = cur_dt.replace(tzinfo=ZoneInfo(tz_name))
                today_local_date = cur_dt.astimezone(ZoneInfo(tz_name)).date()
            except Exception:
                today_local_date = None
    if today_local_date is None:
        today_local_date = datetime.now(ZoneInfo("UTC")).date()

    # Fallback chain for downstream renders — same as tz_name above.
    user_tz_name = (
        (isinstance(time_data, dict) and time_data.get("timezone"))
        or raw_data.get("timezone")
        or "UTC"
    )
    user_tz = ZoneInfo(user_tz_name)

    def _today_entry_from_stats(key: str) -> Optional[Dict[str, Any]]:
        entry = _safe_get(today_stats, key) or {}
        data_list = entry.get("data") or []
        if not data_list:
            return None
        # First, prefer exact today match
        for d in data_list:
            if not isinstance(d, dict):
                continue
            date_str = d.get("date")
            if not date_str:
                continue
            try:
                dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=user_tz)
                if dt.astimezone(user_tz).date() == today_local_date:
                    return d
            except Exception:
                continue
        for d in reversed(data_list):
            if isinstance(d, dict):
                return d
        return None

    def _first_today_stat(key: str) -> Any:
        return _today_entry_from_stats(key) or {}

    def _newest_snapshot() -> Dict[str, Any]:
        """Return the historical snapshot closest to today (last in list)."""
        if not snapshots:
            return {}
        if isinstance(snapshots[-1], dict):
            return snapshots[-1]
        return {}

    snap = _newest_snapshot()

    # ── steps_today ──────────────────────────────────────────────────────
    if not health_params.get("health_params_steps_today"):
        steps_entry = _first_today_stat("STEPS") or {}
        v = steps_entry.get("steps")
        if v:
            health_params["health_params_steps_today"] = v
        elif (
            latest_hm.get("steps_per_day")
            and _is_metric_date_fresh(latest_hm, current_date)
        ):
            health_params["health_params_steps_today"] = latest_hm["steps_per_day"]
            if not health_params.get("steps_last_data_date"):
                profile_md = str(latest_hm.get("metric_date") or "").strip()
                if profile_md:
                    health_params["steps_last_data_date"] = profile_md
        elif latest_hm.get("steps_per_day"):
            logger.info(
                "⚠️ Skipping stale profile steps fallback "
                "(user_profile.latest_health_metrics.metric_date != today)"
            )
        elif snap.get(CanonicalField.STEP_COUNT_ONE_DAY):
            health_params["health_params_steps_today"] = snap.get(CanonicalField.STEP_COUNT_ONE_DAY)

    # ── remaining_steps ──────────────────────────────────────────────────
    if health_params.get("health_params_remaining_steps") is None:
        today_v = health_params.get("health_params_steps_today") or 0
        goal_v = health_params.get("health_params_steps_goal") or 0
        if today_v and goal_v:
            health_params["health_params_remaining_steps"] = max(0, goal_v - today_v)

    # ── resting_heart_rate ───────────────────────────────────────────────
    if health_params.get("health_params_resting_heart_rate") in (None, 0):
        hr_entry = _first_today_stat("RESTING_HEART_RATE") or {}
        v = hr_entry.get("value")
        if v:
            health_params["health_params_resting_heart_rate"] = v
        elif (
            latest_hm.get("heart_rate_avg")
            and _is_metric_date_fresh(latest_hm, current_date)
        ):
            health_params["health_params_resting_heart_rate"] = latest_hm["heart_rate_avg"]
        elif latest_hm.get("heart_rate_avg"):
            logger.info(
                "⚠️ Skipping stale profile resting_heart_rate fallback "
                "(user_profile.latest_health_metrics.metric_date != today)"
            )
        elif snap.get(CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM) or snap.get("h_resting_heart_rate"):
            health_params["health_params_resting_heart_rate"] = snap.get(CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM) or snap.get("h_resting_heart_rate")

    # ── baseline_resting_hr (lifetime avg) ───────────────────────────────
    if health_params.get("health_params_baseline_resting_hr") in (None, 0):
        hr_entry = _first_today_stat("RESTING_HEART_RATE") or {}
        b = hr_entry.get("baseline")
        if b:
            health_params["health_params_baseline_resting_hr"] = b

    # ── latest_heart_rate ────────────────────────────────────────────────
    if health_params.get("health_params_latest_heart_rate") in (None, 0):
        hr_entry = _first_today_stat("RESTING_HEART_RATE") or {}
        v = hr_entry.get("latest") or hr_entry.get("value")
        if v:
            health_params["health_params_latest_heart_rate"] = v
        elif snap.get(CanonicalField.AVERAGE_HEART_RATE_ONE_DAY_BPM):
            health_params["health_params_latest_heart_rate"] = snap.get(CanonicalField.AVERAGE_HEART_RATE_ONE_DAY_BPM) or snap.get("h_avg_heart_rate")

    # ── sleep dates (for staleness surfacing) ────────────────────────────
    sleep_entry = _first_today_stat("SLEEPS") or {}
    if not health_params.get("sleep_last_data_date") and sleep_entry.get("date"):
        health_params["sleep_last_data_date"] = sleep_entry["date"]
    if not health_params.get("steps_last_data_date"):
        steps_entry = _first_today_stat("STEPS") or {}
        if steps_entry.get("date"):
            health_params["steps_last_data_date"] = steps_entry["date"]
    if not health_params.get("hr_last_data_date"):
        hr_entry = _first_today_stat("RESTING_HEART_RATE") or {}
        if hr_entry.get("date"):
            health_params["hr_last_data_date"] = hr_entry["date"]

    # ── mood_last_data_date (from latest_mood if available) ──────────────
    if not health_params.get("mood_last_data_date"):
        latest_mood = _safe_get(raw_data, "latest_mood") or {}
        md = (
            latest_mood.get("current_mood_date")
            or latest_mood.get("moodDate")
            or latest_mood.get("mood_date")
        )
        if md:
            health_params["mood_last_data_date"] = str(md).split("T")[0]


def _build_minimal_health_signals_from_raw(
    raw_data: Dict[str, Any],
    health_params: Dict[str, Any],
    time_data: Dict[str, Any],
) -> "HealthSignalsBlock":  # type: ignore[name-defined]
    
    from insights.schemas.processed_context import (
        ActivitySignal,
        CardioAndStressSignal,
        HealthSignalsBlock,
    )

    # Walk back through snapshots to pick up last known hrv_score, etc.
    snapshots = _safe_get(raw_data, "historical_snapshots") or []
    last_snap = snapshots[-1] if snapshots and isinstance(snapshots[-1], dict) else {}

    from insights.health.health_processor.sleep.processor import SleepSignalProcessor

    sleep = SleepSignalProcessor.process(
        health_params=health_params,
        historical_snapshots=snapshots,
        time_data=time_data,
        raw_data=raw_data,
        staleness_info={"sleep_data_staleness_days": 0},
    )

    steps_today = (
        health_param_value(
            health_params,
            "health_params_steps_today",
            HealthDataConstants.KEY_STEPS_TODAY,
        )
        or 0
    )
    steps_goal = (
        health_param_value(
            health_params,
            "health_params_steps_goal",
            HealthDataConstants.KEY_STEPS_GOAL,
        )
        or 0
    )
    pace_ratio = (steps_today / steps_goal) if steps_goal > 0 else 0.0
    activity = ActivitySignal(
        level=(
            "ok" if pace_ratio >= 0.7
            else "mild" if pace_ratio >= 0.4
            else "moderate" if pace_ratio >= 0.15
            else "severe"
        ),
        steps_today=steps_today,
        steps_goal=steps_goal,
        remaining_steps=health_param_value(
            health_params,
            "health_params_remaining_steps",
            HealthDataConstants.KEY_REMAINING_STEPS,
        ),
        pace_ratio=round(pace_ratio, 2),
        active_hours_left=None,
        steps_streak=(
            health_param_value(
                health_params,
                "health_params_steps_streak",
                HealthDataConstants.KEY_STEPS_STREAK,
            )
            or 0
        ),
        workout_activity_type=health_param_value(
            health_params,
            "health_params_workout_activity_type",
            HealthDataConstants.KEY_WORKOUT_ACTIVITY_TYPE,
        ),
        workout_duration_min=(
            health_params.get("workout_duration_min", 0)
            or health_param_value(
                health_params,
                "health_params_workout_duration",
                HealthDataConstants.KEY_WORKOUT_DURATION,
            )
            / 60.0
            if health_param_value(
                health_params,
                "health_params_workout_duration",
                HealthDataConstants.KEY_WORKOUT_DURATION,
            )
            else 0
        ),
        calories_burned_today=health_param_value(
            health_params,
            "health_params_calories_burned_today",
            HealthDataConstants.KEY_CALORIES_BURNED_TODAY,
        ),
    )

    latest_mood = _safe_get(raw_data, "latest_mood") or {}
    cardio = CardioAndStressSignal(
        resting_heart_rate=health_param_value(
            health_params,
            "health_params_resting_heart_rate",
            HealthDataConstants.KEY_RESTING_HEART_RATE,
        ),
        baseline_resting_hr=health_param_value(
            health_params,
            "health_params_baseline_resting_hr",
            HealthDataConstants.KEY_BASELINE_RESTING_HR,
        ),
        latest_heart_rate=health_param_value(
            health_params,
            "health_params_latest_heart_rate",
            HealthDataConstants.KEY_LATEST_HEART_RATE,
        ),
        hrv_score=health_param_value(
            health_params,
            "health_params_hrv_score",
            HealthDataConstants.KEY_HRV_SCORE,
        ),
        primary_mood=(
            latest_mood.get("current_mood")
            or latest_mood.get("mood")
        ),
        mood_date=(
            latest_mood.get("current_mood_date")
            or latest_mood.get("moodDate")
            or latest_mood.get("mood_date")
        ),
        stress_high=bool(
            health_param_value(
                health_params,
                "health_params_stress_high",
                "stress_high",
            )
        ),
        energy_week_vs_prior_month_pct=health_param_value(
            health_params,
            "health_params_energy_week_vs_prior_month_pct",
            HealthDataConstants.KEY_,
        ),
    )

    return HealthSignalsBlock.from_legacy_parts(
        sleep=sleep, activity=activity, cardio=cardio
    )


def _narrative_user_intro(user_profile: Dict[str, Any]) -> Optional[str]:
    """Concise identity hook: name, role, primary goal, conditions."""
    name = _safe_get(user_profile, "name") or ""
    age = _safe_get(user_profile, "age")
    gender = _safe_get(user_profile, "gender")
    job = _safe_get(user_profile, "job_title")
    region = _safe_get(user_profile, "region")
    primary_goal = _safe_get(user_profile, "primary_goal")
    conditions = _safe_get(user_profile, "current_health_conditions") or []
    if isinstance(conditions, str):
        conditions = [conditions]

    if not any([name, age, job, primary_goal]):
        return None

    bits = []
    if name:
        bits.append(name)
    if age:
        age_part = f"{age}"
        if gender:
            age_part += f" {gender.lower()}"
        bits.append(age_part)
    if job:
        loc_part = job
        if region:
            loc_part += f" in {region}"
        bits.append(loc_part)
    base = ", ".join(bits) + "."

    extras = []
    if primary_goal:
        extras.append(f"Primary goal: {primary_goal}")
    if conditions:
        extras.append("Conditions: " + ", ".join(str(c) for c in conditions))
    if extras:
        base += " " + ". ".join(extras) + "."
    return base


def _narrative_health_snapshot(health_signals: Any, meta: Any = None) -> Optional[str]:
    sl = _safe_get(health_signals, "sleep")
    if sl is None:
        return None

    # Read from summary + signals (single source of truth).
    summary = _safe_get(sl, "summary")
    signals = _safe_get(sl, "signals")

    duration_sig = _safe_get(signals, "sleep_duration")
    debt_sig = _safe_get(signals, "sleep_debt")
    quality_sig = _safe_get(signals, "sleep_quality")
    trend_sig = _safe_get(signals, "sleep_trend")
    timing_sig = _safe_get(signals, "bedtime_timing")

    duration_metrics = _safe_get(duration_sig, "metrics") or {}
    debt_metrics = _safe_get(debt_sig, "metrics") or {}
    quality_metrics = _safe_get(quality_sig, "metrics") or {}
    trend_metrics = _safe_get(trend_sig, "metrics") or {}
    timing_metrics = _safe_get(timing_sig, "metrics") or {}

    last = duration_metrics.get("last_night_hours")
    goal = duration_metrics.get("target_hours")
    score = quality_metrics.get("sleep_score")
    bed = _format_hhmm(timing_metrics.get("bedtime"))
    wake = _format_hhmm(timing_metrics.get("wake_time"))
    bed_source = timing_metrics.get("bedtime_source") or "raw"
    show_bedtime = bool(bed and bed_source == "raw")
    quality = _safe_get(quality_sig, "status")
    trend = _safe_get(trend_sig, "status")
    debt = debt_metrics.get("last_night_debt_h")
    consec_debt = debt_metrics.get("consecutive_debt_days")

    staleness_days = _safe_get(meta, "sleep_data_staleness_days") if meta is not None else None
    last_data_date = _safe_get(meta, "sleep_last_data_date") if meta is not None else None
    is_no_data_ever = staleness_days == 999
    is_stale = (
        staleness_days is not None
        and not is_no_data_ever
        and staleness_days >= 1
    )

    if last is None and score is None and not bed and not (is_stale or is_no_data_ever):
        return None

    parts = []
    stale_warning: Optional[str] = None

    if is_no_data_ever:
        parts.append("[NO DATA] no data available yet (device not synced)")
        # Nothing else meaningful to print when there's no record at all.
        txt = "Sleep: " + "; ".join(parts) + "."
        return txt

    if is_stale:
        if last_data_date:
            date_part = f"from {last_data_date}"
        else:
            date_part = f"from {staleness_days}d ago"
        parts.append(
            f"[NO DATA TODAY] last sleep record is {date_part}; "
            f"do NOT assume current sleep status"
        )
        stale_warning = (
            "User has no sleep data for today. Invite them to sync/update "
            "their device rather than quoting any historical number."
        )
    elif last is not None:
        # Sleep metrics are ALWAYS about the completed night that just ended
        # (last night → this morning), even when the tag is "fresh/synced today".
        # Do NOT treat these as tonight's plan or as caused by today's calendar.
        if goal:
            delta = round(goal - last, 1)
            sign = "−" if delta > 0 else "+"
            parts.append(
                f"[LAST NIGHT session — synced today] slept {last}h "
                f"vs duration goal {goal}h ({sign}{abs(delta)}h)"
            )
        else:
            parts.append(f"[LAST NIGHT session — synced today] slept {last}h")
    elif goal:
        parts.append(
            f"[NO LAST-NIGHT SESSION] sleep duration goal {goal}h "
            f"(no completed night logged yet)"
        )

    if not is_stale:
        # Only surface score / bedtime / debt when the data is fresh;
        # otherwise those fields are also stale and would mislead.
        if score is not None:
            parts.append(f"quality {score}/100")
        if show_bedtime and bed and wake:
            parts.append(
                f"LAST_NIGHT_BEDTIME {bed} → wake {wake} "
                f"(closed history — NOT tonight; NOT caused by today's calendar events)"
            )
        elif show_bedtime and bed:
            parts.append(
                f"LAST_NIGHT_BEDTIME {bed} "
                f"(closed history — NOT tonight; NOT caused by today's calendar events)"
            )
        if trend and trend != "stable":
            parts.append(f"trend {trend}")
        if consec_debt and consec_debt >= 2:
            parts.append(f"sleep debt {consec_debt} nights in a row")
        elif debt and debt > 0:
            parts.append(f"sleep debt {round(debt, 1)}h")

    if not parts:
        return None
    txt = "Sleep: " + "; ".join(parts) + "."
    if stale_warning:
        txt += f" {stale_warning}"
    if quality and quality not in (None, "unknown"):
        txt += f" (logged: {quality})"
    txt += (
        " TEMPORAL RULE: last-night sleep fields must not be blamed on today's "
        "calendar; today's events can only affect tonight's upcoming wind-down."
    )
    return txt


def _narrative_activity_snapshot(health_signals: Any, time_data: Dict[str, Any], meta: Any = None) -> Optional[str]:
    act = _safe_get(health_signals, "activity")
    if act is None:
        return None
    today = _safe_get(act, "health_params_steps_today")
    goal = _safe_get(act, "health_params_steps_goal")
    remaining = _safe_get(act, "remaining_steps")
    workout = _safe_get(act, "workout_activity_type")
    workout_dur = _safe_get(act, "workout_duration_min")
    streak = _safe_get(act, "steps_streak")

    steps_staleness = _safe_get(meta, "steps_data_staleness_days") if meta is not None else None
    steps_last_date = _safe_get(meta, "steps_last_data_date") if meta is not None else None
    is_steps_no_data_ever = steps_staleness == 999
    is_steps_stale = (
        steps_staleness is not None
        and not is_steps_no_data_ever
        and steps_staleness >= 1
    )

    # When steps are stale, drop any path that quotes a step number — even
    # the workout-only path can be misleading if today's plan depended on
    # step progress. Workout data is itself a fresh entry, so keep it but
    # avoid "X / Y steps (Z%)" claims that would be fabricated.
    if is_steps_no_data_ever:
        parts = ["[NO DATA] no data available yet (device not synced)"]
        if workout and workout_dur:
            parts.append(f"worked out {workout} for {workout_dur} min")
        if streak:
            parts.append(f"steps streak {streak} days")
        return "Activity today: " + "; ".join(parts) + "."

    if is_steps_stale:
        if steps_last_date:
            date_part = f"from {steps_last_date}"
        else:
            date_part = f"from {steps_staleness}d ago"
        parts = [
            f"[NO DATA TODAY] last reading is {date_part}; "
            f"do NOT assume current step count"
        ]
        if workout and workout_dur:
            parts.append(f"worked out {workout} for {workout_dur} min")
        if streak:
            parts.append(f"steps streak {streak} days")
        return "Activity today: " + "; ".join(parts) + "."

    if today is None and not workout:
        return None

    parts = []
    if today is not None and goal:
        pct = int(round((today / goal) * 100)) if goal else 0
        parts.append(f"[TODAY fresh] {today:,} / {goal:,} steps ({pct}% of goal)")
    elif today is not None:
        parts.append(f"[TODAY fresh] {today:,} steps")

    if remaining and remaining > 0:
        hours_left = _safe_get(time_data, "hours_remaining_today") or _safe_get(time_data, "remaining_hours_today")
        if hours_left:
            parts.append(f"{remaining:,} steps remaining ({hours_left}h left in the day)")
        else:
            parts.append(f"{remaining:,} steps remaining")
    if workout and workout_dur:
        parts.append(f"worked out {workout} for {workout_dur} min")
    if streak:
        parts.append(f"steps streak {streak} days")
    if not parts:
        return None
    return "Activity today: " + "; ".join(parts) + "."


def _narrative_cardio_snapshot(health_signals: Any, meta: Any = None) -> Optional[str]:
    
    cardio = _safe_get(health_signals, "cardio_stress")
    if cardio is None:
        return None
    rhr = _safe_get(cardio, "resting_heart_rate")
    base = _safe_get(cardio, "baseline_resting_hr")
    latest = _safe_get(cardio, "latest_heart_rate")
    hrv = _safe_get(cardio, "hrv_score")
    primary_mood = _safe_get(cardio, "primary_mood")
    mood_avg = _safe_get(cardio, "mood_score_avg")
    stress_high = _safe_get(cardio, "stress_high")
    walking_hr = _safe_get(cardio, "walking_heart_rate_avg")
    energy_pct = _safe_get(cardio, "health_params_energy_week_vs_prior_month_pct")

    hr_staleness = _safe_get(meta, "hr_data_staleness_days") if meta is not None else None
    hr_last_date = _safe_get(meta, "hr_last_data_date") if meta is not None else None

    if (
        all(
            v is None
            for v in (rhr, base, latest, hrv, primary_mood, mood_avg, walking_hr, energy_pct)
        )
        and not stress_high
    ):
        return None

    is_no_data_ever = hr_staleness == 999
    is_stale = (
        hr_staleness is not None
        and not is_no_data_ever
        and hr_staleness >= 1
    )

    parts = []
    stale_suffix = ""

    if is_no_data_ever:
        # Don't quote any HR number when there's never been a reading.
        parts.append("[NO DATA] no data available yet (device not synced)")
    elif is_stale:
        if hr_last_date:
            date_part = f"from {hr_last_date}"
        else:
            date_part = f"from {hr_staleness}d ago"
        parts.append(
            f"[NO HR DATA TODAY] last reading is {date_part}; "
            f"do NOT assume current resting HR, HRV, or walking HR"
        )
        stale_suffix = (
            f"User has no HR data for today. Invite them to sync/update their "
            f"device rather than quoting any historical number."
        )
    else:
        if rhr is not None and base:
            delta = rhr - base
            sign = "+" if delta > 0 else ""
            parts.append(
                f"[TODAY fresh] resting HR {rhr} bpm (baseline {base}, {sign}{delta})"
            )
        elif rhr is not None:
            parts.append(f"[TODAY fresh] resting HR {rhr} bpm")
        elif latest is not None:
            parts.append(f"[TODAY fresh] latest HR {latest} bpm")
        if hrv is not None:
            parts.append(f"HRV {hrv}")
        if walking_hr is not None and not is_stale:
            parts.append(f"{_aggregate_tag()} walking HR {walking_hr} bpm")
    if energy_pct is not None and not is_stale and not is_no_data_ever:
        sign = "+" if energy_pct > 0 else ""
        direction = "up" if energy_pct > 0 else ("down" if energy_pct < 0 else "flat")
        parts.append(
            f"{_aggregate_tag()} energy vs prior month: "
            f"{sign}{round(energy_pct, 1)}% ({direction})"
        )

    mood_staleness = _safe_get(meta, "mood_data_staleness_days") if meta is not None else None
    mood_last_date = _safe_get(meta, "mood_last_data_date") if meta is not None else None
    is_mood_no_data_ever = mood_staleness == 999
    is_mood_stale = (
        mood_staleness is not None
        and not is_mood_no_data_ever
        and mood_staleness >= 1
    )
    if is_mood_stale:
        if mood_last_date:
            date_part = f"from {mood_last_date}"
        else:
            date_part = f"from {mood_staleness}d ago"
        parts.append(
            f"[NO MOOD DATA TODAY] last entry is {date_part}; "
            f"do NOT assume current mood"
        )
    elif primary_mood:
        if mood_avg is not None:
            parts.append(f"[TODAY fresh] mood: {primary_mood} ({round(mood_avg, 1)}/10)")
        else:
            parts.append(f"[TODAY fresh] mood: {primary_mood}")
    if stress_high:
        parts.append("stress high")

    if not parts:
        return None
    txt = "Cardio & mood: " + "; ".join(parts) + "."
    if stale_suffix:
        txt += f" {stale_suffix}"
    return txt


def _narrative_week_pattern(
    historical_trends: Any,
    behavioral_patterns: Any,
) -> Optional[str]:
    """3-day aggregates + any notable behavior pattern."""
    if historical_trends is None:
        return None
    avg = _safe_get(historical_trends, "averages_3d")
    recent = _safe_get(historical_trends, "recent_days") or []
    parts = []

    def _avg(field):
        return _safe_get(avg, field)

    steps_avg = _avg(CanonicalField.STEP_COUNT_ONE_DAY)
    sleep_avg = _avg(CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS)
    mood_avg = _avg("mood_score_avg")
    wellness_avg = _avg(CanonicalField.WELLNESS_SCORE_ONE_DAY)
    fin_avg = _avg("net_cashflow")
    fin_score_avg = _avg("financial_health_score")
    prod_avg = _avg("productivity_score")
    hr_avg = _avg(CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM)

    if steps_avg is not None:
        parts.append(f"steps avg {int(round(steps_avg)):,}/day")
    if sleep_avg is not None:
        parts.append(f"sleep avg {round(sleep_avg, 1)}h/night")
    if mood_avg is not None:
        parts.append(f"mood avg {round(mood_avg, 1)}/10")
    if wellness_avg is not None:
        parts.append(f"wellness {int(round(wellness_avg))}")
    if hr_avg is not None:
        parts.append(f"resting HR avg {int(round(hr_avg))} bpm")
    if prod_avg is not None:
        parts.append(f"productivity {int(round(prod_avg))}")
    if fin_score_avg is not None:
        parts.append(f"finance {int(round(fin_score_avg))}")
    elif fin_avg is not None:
        sign = "+" if fin_avg > 0 else ""
        parts.append(f"cashflow avg {sign}{int(round(fin_avg)):,}")

    if not parts:
        # Fall back to recent_days mean
        if recent:
            n = len(recent)
            sums = {CanonicalField.STEP_COUNT_ONE_DAY: 0.0, CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS: 0.0, CanonicalField.WELLNESS_SCORE_ONE_DAY: 0.0}
            counts = {k: 0 for k in sums}
            for d in recent:
                for k in sums:
                    v = _safe_get(d, k)
                    if v is not None:
                        sums[k] += float(v)
                        counts[k] += 1
            if counts[CanonicalField.STEP_COUNT_ONE_DAY]:
                parts.append(f"steps avg {int(round(sums[CanonicalField.STEP_COUNT_ONE_DAY] / counts[CanonicalField.STEP_COUNT_ONE_DAY])):,}/day ({n}d)")
            if counts[CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS]:
                parts.append(f"sleep avg {round(sums[CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS] / counts[CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS], 1)}h/night")
            if counts[CanonicalField.WELLNESS_SCORE_ONE_DAY]:
                parts.append(f"wellness avg {int(round(sums[CanonicalField.WELLNESS_SCORE_ONE_DAY] / counts[CanonicalField.WELLNESS_SCORE_ONE_DAY]))}")
        if not parts:
            return None

    trend_bits = []
    ws_trend = _safe_get(historical_trends, "trends", "wellness_score_trend")
    if ws_trend and ws_trend != "stable":
        trend_bits.append(f"wellness {ws_trend}")

    txt = f"Past 3 days [{_aggregate_tag()}]: " + "; ".join(parts) + "."
    if trend_bits:
        txt += " " + " | ".join(trend_bits) + "."
    return txt


def _narrative_mood(
    health_signals: Any,
    historical_trends: Any,
    raw_data: Dict[str, Any],
    meta: Any = None,
) -> Optional[str]:
    cardio = _safe_get(health_signals, "cardio_stress")
    mood_today = (
        _safe_get(cardio, "primary_mood")
        or _safe_get(cardio, "mood_first")
        or _safe_get(raw_data, "latest_mood", "mood")
    )
    mood_score = (
        _safe_get(cardio, "mood_score_avg")
        or _safe_get(raw_data, "latest_mood", "moodScore")
    )
    sequence = (
        _safe_get(cardio, "mood_sequence")
        or _safe_get(raw_data, "latest_mood", "moodSequence")
        or []
    )

    mood_staleness = _safe_get(meta, "mood_data_staleness_days") if meta is not None else None
    mood_last_date = _safe_get(meta, "mood_last_data_date") if meta is not None else None
    is_mood_no_data_ever = mood_staleness == 999
    is_mood_stale = (
        mood_staleness is not None
        and not is_mood_no_data_ever
        and mood_staleness >= 1
    )

    recent = _safe_get(historical_trends, "recent_days") or []
    seq_from_snaps = []
    for d in recent:
        # Mood: try mood_primary_mood first (canonical), fall back to m_primary_mood
        m = _safe_get(d, "mood_first_mood") or _safe_get(d, "m_first_mood") or _safe_get(d, "mood_primary_mood") or _safe_get(d, "m_primary_mood")
        if m:
            seq_from_snaps.append(m)

    seq_full = list(sequence) + seq_from_snaps
    if not seq_full and not mood_today:
        return None

    parts = []
    if seq_full:
        last = seq_full[-3:] if len(seq_full) >= 3 else seq_full
        last_reversed = list(reversed(last))
        if len(last_reversed) >= 2:
            arrow = " → ".join(last_reversed)
            parts.append(f"mood over the last 3 days: {arrow}")
        elif last_reversed:
            parts.append(f"most recent mood: {last_reversed[0]}")

    if is_mood_no_data_ever:
        parts.append("logged today: no data available yet (device not synced)")
    elif is_mood_stale:
        if mood_last_date:
            date_part = f"from {mood_last_date}"
        else:
            date_part = f"from {mood_staleness}d ago"
        parts.append(
            f"logged today: NO MOOD DATA TODAY — last entry is {date_part}; "
            f"do NOT assume current mood"
        )
    elif mood_today:
        if mood_score is not None:
            parts.append(f"logged today: {mood_today} ({round(mood_score, 1)}/10)")
        else:
            parts.append(f"logged today: {mood_today}")

    if not parts:
        return None
    return "Mood: " + "; ".join(parts) + "."


def _narrative_productivity_today(
    productivity_signals: Any,
    calendar_intelligence: Any,
) -> Optional[str]:
    """Today's productivity + calendar load."""
    parts = []
    if productivity_signals is not None:
        completion = _safe_get(productivity_signals, "events_completion_rate")
        meeting_completion = _safe_get(productivity_signals, "meeting_completion_rate")
        reminders_completion = _safe_get(productivity_signals, "reminders_completion_rate")
        focus = _safe_get(productivity_signals, "focus_score")
        fatigue = _safe_get(productivity_signals, "meeting_fatigue")
        reminders = _safe_get(productivity_signals, "reminders_due_today")
        meetings_today = _safe_get(productivity_signals, "meetings_due_today")
        overdue = _safe_get(productivity_signals, "overdue_reminders_count")
        nearest = _safe_get(productivity_signals, "nearest_reminder_title")
        nearest_in = _safe_get(productivity_signals, "nearest_reminder_due_in_mins")
        if meetings_today and meeting_completion is not None:
            m_done = int(round(meeting_completion * int(meetings_today)))
            parts.append(f"{m_done}/{int(meetings_today)} meetings attended ({int(round(meeting_completion * 100))}%)")
        elif meetings_today:
            parts.append(f"{meetings_today} meetings today")
        if reminders and reminders_completion is not None:
            r_done = int(round(reminders_completion * int(reminders)))
            parts.append(f"reminders {r_done}/{int(reminders)} done (separate from productivity)")
        elif reminders:
            parts.append(f"{int(reminders)} reminders today")
        if completion is not None and not meetings_today:
            parts.append(f"task completion {int(round(completion * 100))}%")
        if focus is not None:
            if focus == "low":
                parts.append("no focus block ≥ 90 min on the schedule today")
            elif focus == "moderate":
                parts.append("1 focus block ≥ 90 min on the schedule today")
            else:
                parts.append("2+ focus blocks ≥ 90 min on the schedule today")
        if fatigue:
            parts.append(f"meeting fatigue {fatigue}")
        if overdue:
            parts.append(f"{overdue} reminders overdue")
        if email:
            parts.append(f"email load: {email}")
        if nearest and nearest_in is not None:
            parts.append(f"next up: {nearest} ({nearest_in}m)")

    if calendar_intelligence is not None:
        total = _safe_get(calendar_intelligence, "total_events")
        if total is not None:
            parts.append(f"{total} events")
        free_window = _safe_get(calendar_intelligence, "free_window_minutes")
        next_event = _safe_get(calendar_intelligence, "next_event_title")
        next_in = _safe_get(calendar_intelligence, "next_event_in_minutes")
        if free_window and free_window > 0:
            parts.append(f"free window {int(free_window)}m")
        if next_event and next_in is not None:
            parts.append(f"next event: {next_event} in {next_in}m")

    if not parts:
        return None
    return "Today: " + "; ".join(parts) + "."


def _narrative_finance(
    finance_signals: Any,
    user_profile: Dict[str, Any],
    raw_data: Dict[str, Any],
) -> Optional[str]:
    """This month's spending vs budget, goal progress."""
    parts = []
    if finance_signals is not None:
        month_spend = _safe_get(finance_signals, "month_total_expense")
        month_income = _safe_get(finance_signals, "month_total_income")
        budget = _safe_get(finance_signals, "month_budget")
        util = _safe_get(finance_signals, "budget_utilization")
        net = _safe_get(finance_signals, "month_net_cashflow")
        top_cat = _safe_get(finance_signals, "top_spending_category")
        goal_progress = _safe_get(finance_signals, "goal_progress_summary")
        if month_spend is not None:
            spend_txt = f"spent {int(round(month_spend)):,}"
            if budget:
                spend_txt += f" / {int(round(budget)):,}"
            if util is not None:
                spend_txt += f" ({int(round(util * 100))}%)"
            parts.append(spend_txt)
        if month_income is not None:
            parts.append(f"income {int(round(month_income)):,}")
        if net is not None:
            sign = "+" if net > 0 else ""
            parts.append(f"net cashflow {sign}{int(round(net)):,}")
        if top_cat:
            parts.append(f"top spending category: {top_cat}")
        if goal_progress:
            parts.append(goal_progress)

    if not parts:
        fs = _safe_get(raw_data, "finance_summary") or {}
        if fs:
            # finance_summary has flat fields like totalSpending/totalIncome
            if not any("spent" in p for p in parts):
                spend = fs.get("totalSpending")
                if spend is not None:
                    spend_txt = f"spent {int(spend):,}"
                    util = fs.get("spendingBudgetPercentage")
                    if util is not None:
                        spend_txt += f" ({int(util)}%)"
                    parts.append(spend_txt)
            if not any("income" in p for p in parts):
                income = fs.get("totalIncome")
                if income is not None:
                    parts.append(f"income {int(income):,}")
            cash = fs.get("availableCash") or fs.get("availableCashThisMonth")
            if cash is not None and not any("cashflow" in p for p in parts):
                sign = "+" if cash > 0 else ""
                parts.append(f"net cashflow {sign}{int(cash):,}")
            score = fs.get("financeScore")
            if score is not None:
                parts.append(f"finance score {int(round(score))}")

        finance_goals = _safe_get(raw_data, "finance_goals") or []
        if finance_goals and not any("fund" in p for p in parts):
            ranked = sorted(
                finance_goals,
                key=lambda g: float(g.get("progressPercentage") or g.get("progress") or 0),
                reverse=True,
            )
            for g in ranked[:2]:
                name = g.get("name") or g.get("goalName")
                pct = g.get("progressPercentage") or g.get("progress")
                if name and pct is not None:
                    parts.append(f"{name}: {int(round(float(pct)))}%")

    if not parts:
        return None
    return "Finance: " + "; ".join(parts) + "."


def _narrative_goal_progress(
    goal_signals: Any,
    user_profile: Dict[str, Any],
    raw_data: Dict[str, Any],
) -> Optional[str]:
    parts = []

    weight = (
        _safe_get(user_profile, "weight_kg")
        or _safe_get(user_profile, "user_weight_kilograms")
        or _safe_get(raw_data, "user_profile", "weight_kg")
        or _safe_get(raw_data, "user_profile", "user_weight_kilograms")
    )
    height = (
        _safe_get(user_profile, "height_cm")
        or _safe_get(user_profile, "user_height_centimeters")
        or _safe_get(raw_data, "user_profile", "height_cm")
        or _safe_get(raw_data, "user_profile", "user_height_centimeters")
    )
    primary_goal = (
        _safe_get(user_profile, "primary_goal")
        or _safe_get(user_profile, "user_primary_goal")
        or ""
    )
    if weight and height:
        try:
            height_m = float(height) / 100.0
            bmi = round(float(weight) / (height_m * height_m), 1) if height_m else None
        except (TypeError, ValueError):
            bmi = None
        if bmi and (
            "lose" in str(primary_goal).lower()
            or "weight" in str(primary_goal).lower()
            or "kg" in str(primary_goal).lower()
            or bmi >= 25
        ):
            if bmi >= 30:
                cls = "obese"
            elif bmi >= 25:
                cls = "overweight"
            else:
                cls = "normal"
            parts.append(f"weight {weight} kg (BMI {bmi} – {cls})")

    health_params = _safe_get(raw_data, "health_params") or {}
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
    if steps_today and steps_goal:
        pct = int(round(steps_today / steps_goal * 100))
        parts.append(f"steps goal at {pct}% ({int(steps_today):,}/{int(steps_goal):,})")

    if goal_signals is not None:
        goal_list = (
            _safe_get(goal_signals, "goals")
            or _safe_get(goal_signals, "finance_goal_progress")
            or _safe_get(goal_signals, "custom_goals")
            or []
        )
        for g in goal_list[:3]:
            name = (
                _safe_get(g, "goal_label")
                or _safe_get(g, "goal_name")
                or _safe_get(g, "name")
            )
            raw_pct = _safe_get(g, "pace_ratio")
            if raw_pct is None:
                raw_pct = _safe_get(g, "progress_pct") or _safe_get(g, "progressPct")
            if name and raw_pct is not None:
                pct = int(round(raw_pct * 100 if raw_pct <= 1.0 else raw_pct))
                parts.append(f"goal {name}: {pct}%")

    if not parts:
        user_goals = (
            _safe_get(user_profile, "user_goals")
            or _safe_get(user_profile, "goals")
            or _safe_get(raw_data, "user_profile", "user_goals")
            or _safe_get(raw_data, "user_profile", "goals")
            or []
        )
        for g in user_goals[:3]:
            name = (
                _safe_get(g, "goal_name")
                or _safe_get(g, "name")
                or _safe_get(g, "goal_label")
            )
            target = _safe_get(g, "target_value")
            unit = _safe_get(g, "unit") or ""
            if name and target is not None:
                parts.append(f"goal {name}: target {target} {unit}".strip())
            elif name:
                parts.append(f"goal {name} configured")

    if not parts:
        return None
    return "Goals: " + "; ".join(parts) + "."


def _narrative_calendar_outlook(
    calendar_intelligence: Any,
    calendar_events: List[Any],
) -> Optional[str]:
    """Next notable event today + health-related events count."""
    parts = []
    next_event_title = _safe_get(calendar_intelligence, "next_event_title")
    next_event_in = _safe_get(calendar_intelligence, "next_event_in_minutes")
    next_event_start = _safe_get(calendar_intelligence, "next_event_start")
    next_event_loc = _safe_get(calendar_intelligence, "next_event_location")
    if next_event_title:
        bits = [next_event_title]
        if next_event_in is not None and next_event_in >= 0:
            bits.append(f"in {next_event_in}m")
        elif next_event_start:
            bits.append(_format_hhmm(next_event_start) or "")
        if next_event_loc:
            bits.append(f"at {next_event_loc}")
        parts.append("next: " + " ".join(filter(None, bits)))

    if calendar_events:
        health_kw = (
            "yoga", "gym", "workout", "doctor", "telehealth",
            "health", "run", "running", "pilates", "exercise",
        )
        health_events = [
            e for e in calendar_events
            if any(
                kw in str(_safe_get(e, "title") or _safe_get(e, "summary") or "").lower()
                for kw in health_kw
            )
        ]
        if health_events:
            parts.append(f"{len(health_events)} health-related events today")

    if not parts:
        return None
    return "Schedule: " + "; ".join(parts) + "."


def _narrative_risk_flags(
    health_signals: Any,
    user_profile: Dict[str, Any],
    historical_trends: Any,
) -> Optional[str]:
    """List risks: sleep debt, BMI, RHR delta vs baseline, conditions."""
    flags = []

    debt_sig = _safe_get(health_signals, "sleep", "signals", "sleep_debt")
    debt_metrics = _safe_get(debt_sig, "metrics") or {}
    debt = debt_metrics.get("last_night_debt_h")
    consec = debt_metrics.get("consecutive_debt_days")
    try:
        debt_num = float(debt) if debt is not None else None
    except (TypeError, ValueError):
        debt_num = None
    if debt_num is not None and debt_num >= 1.5:
        flags.append(f"sleep debt {round(debt_num, 1)}h")
    elif consec and consec >= 3:
        flags.append(f"sleep debt {consec} nights")

    weight = _safe_get(user_profile, "weight_kg")
    height = _safe_get(user_profile, "height_cm")
    if weight and height:
        height_m = height / 100.0
        bmi = weight / (height_m * height_m) if height_m else None
        if bmi:
            if bmi >= 30:
                flags.append(f"BMI {round(bmi, 1)} (obese)")
            elif bmi >= 25:
                flags.append(f"BMI {round(bmi, 1)} (overweight)")

    rhr = _safe_get(health_signals, "cardio_stress", "resting_heart_rate")
    base = _safe_get(health_signals, "cardio_stress", "health_params_baseline_resting_hr")
    if rhr and base and rhr - base >= 5:
        flags.append(f"resting HR {rhr} above baseline {base} (+{rhr - base})")

    conds = _safe_get(user_profile, "current_health_conditions") or []
    if isinstance(conds, str):
        conds = [conds]
    if conds:
        flags.append(", ".join(str(c) for c in conds))

    if not flags:
        return None
    return "Watch out: " + "; ".join(flags) + "."


def _narrative_positive_signals(
    health_signals: Any,
    historical_trends: Any,
    raw_data: Dict[str, Any],
) -> Optional[str]:
    """List positives: sleep quality, wellness score, streaks, goals on track."""
    positives = []

    sleep_score = _safe_get(
        health_signals, "sleep", "signals", "sleep_quality", "metrics", "sleep_score"
    )
    if sleep_score and sleep_score >= 80:
        positives.append(f"high sleep quality ({sleep_score}/100)")

    # bedtime_streak is computed by prepare_health_data and not exposed via SleepSignal.
    bed_streak = _safe_get(health_signals, "sleep", "bedtime_streak")
    if bed_streak and bed_streak >= 3:
        positives.append(f"consistent bedtime streak of {bed_streak} days")

    # wellness_today uses the canonical name
    wellness_today = _safe_get(historical_trends, "recent_days", 0, CanonicalField.WELLNESS_SCORE_ONE_DAY)
    if wellness_today and wellness_today >= 70:
        positives.append(f"today's wellness score is {int(wellness_today)}")

    wellness_avg = _safe_get(historical_trends, "averages_3d", CanonicalField.WELLNESS_SCORE_ONE_DAY)
    if wellness_avg and wellness_avg >= 70 and not wellness_today:
        positives.append(f"3-day wellness average is {int(round(wellness_avg))}")

    steps_streak = _safe_get(health_signals, "activity", "steps_streak")
    if steps_streak and steps_streak >= 3:
        positives.append(f"steps streak of {steps_streak} days")

    fin_score = _safe_get(historical_trends, "averages_3d", "financial_health_score")
    if fin_score and fin_score >= 70:
        positives.append("finances are stable")

    trend = _safe_get(historical_trends, "trends", "wellness_score_trend")
    if trend == "improving":
        positives.append("wellness is trending up")

    if not positives:
        return None
    return "Positives: " + "; ".join(positives) + "."

_NARRATIVE_DUAL_PATH_LOGGED = False


def _build_narrative_context(
    raw_data: Dict[str, Any],
    meta: Any,
    health_signals: Any,
    historical_trends: Any,
    calendar_intelligence: Any,
    productivity_signals: Any,
    finance_signals: Any,
    user_profile: Dict[str, Any],
    behavioral_patterns: Any,
    goal_signals: Any,
) -> Dict[str, str]:
    global _NARRATIVE_DUAL_PATH_LOGGED
    if not _NARRATIVE_DUAL_PATH_LOGGED:
        logger.warning(
            "⚠️ _build_narrative_context is dual-path with the canonical "
            "context_narrative_templates.build_3part_narrative_parts — see "
            "TODO to collapse. (This message is logged once per process.)"
        )
        _NARRATIVE_DUAL_PATH_LOGGED = True

    ctx: Dict[str, str] = {}

    candidates = [
        ("user_intro", _narrative_user_intro(user_profile)),
        ("health_snapshot", _narrative_health_snapshot(health_signals, meta)),
        ("activity_snapshot", _narrative_activity_snapshot(health_signals, raw_data.get("time_data") or {}, meta)),
        ("cardio_snapshot", _narrative_cardio_snapshot(health_signals, meta)),
        ("week_pattern", _narrative_week_pattern(historical_trends, behavioral_patterns)),
        ("mood_narrative", _narrative_mood(health_signals, historical_trends, raw_data, meta)),
        ("productivity_today", _narrative_productivity_today(productivity_signals, calendar_intelligence)),
        ("finance_snapshot", _narrative_finance(finance_signals, user_profile, raw_data)),
        ("goal_progress", _narrative_goal_progress(goal_signals, user_profile, raw_data)),
        ("calendar_outlook", _narrative_calendar_outlook(calendar_intelligence, raw_data.get("calendar_events") or [])),
        ("risk_flags", _narrative_risk_flags(health_signals, user_profile, historical_trends)),
        ("positive_signals", _narrative_positive_signals(health_signals, historical_trends, raw_data)),
    ]
    for key, val in candidates:
        if val:
            ctx[key] = val
    return ctx


def _compute_mood_trend_7d(
    historical_snapshots: List[Any],
    today: date,
    window_days: int = 7,
) -> Optional[str]:
    del historical_snapshots, today, window_days
    return None


def _latest_snapshot_value(
    historical_snapshots: List[Any], today: date, field: str
) -> Any:
    candidates = sorted(
        (row for row in historical_snapshots if _snapshot_date(row) is not None),
        key=lambda r: _snapshot_date(r),
        reverse=True,
    )
    for row in candidates:
        val = _snapshot_field(row, field)
        if val is not None:
            return val
    return None

MOOD_SNAPSHOT_TO_CANONICAL = {
    # canonical → canonical (already canonical)
    "mood_primary_mood": "mood_primary_mood",
    "mood_first_mood": "mood_first_mood",
    "mood_score_avg": "mood_score_avg",
    "mood_score_min": "mood_score_min",
    "mood_score_max": "mood_score_max",
    "mood_variance": "mood_variance",
    "mood_sequence": "mood_sequence",
    "mood_log_count": "mood_log_count",
    "mood_has_notes": "mood_has_notes",
    # old → canonical
    "m_primary_mood": "mood_primary_mood",
    "m_first_mood": "mood_first_mood",
    "m_mood_score_avg": "mood_score_avg",
    "m_mood_score_min": "mood_score_min",
    "m_mood_score_max": "mood_score_max",
    "m_mood_variance": "mood_variance",
    "m_mood_sequence": "mood_sequence",
    "m_log_count": "mood_log_count",
    "m_has_notes": "mood_has_notes",
}
MOOD_SNAPSHOT_FIELDS = tuple(MOOD_SNAPSHOT_TO_CANONICAL.keys())


def _latest_mood_snapshot_row(
    historical_snapshots: List[Any],
) -> Optional[Any]:
    candidates = sorted(
        (row for row in historical_snapshots if _snapshot_date(row) is not None),
        key=lambda r: _snapshot_date(r),
        reverse=True,
    )
    for row in candidates:
        if any(_snapshot_field(row, f) is not None for f in MOOD_SNAPSHOT_FIELDS):
            return row
    return None


def _build_mood_data_from_snapshot(
    historical_snapshots: List[Any],
    latest_mood: Optional[Dict[str, Any]],
    today: Optional[date] = None,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    row = _latest_mood_snapshot_row(historical_snapshots)
    if row is not None:
        snap_date = _snapshot_date(row)
        for src_field, canonical_key in MOOD_SNAPSHOT_TO_CANONICAL.items():
            val = _snapshot_field(row, src_field)
            if val is not None:
                # Mirror under both spellings so consumers can read
                # either ``m_primary_mood`` (BE) or ``primary_mood``
                # (canonical attribute on CardioAndStressSignal).
                merged[src_field] = val
                merged[canonical_key] = val
        if snap_date is not None:
            merged["mood_date"] = snap_date.isoformat()

    if isinstance(latest_mood, dict) and latest_mood:
        # latest_mood has been canonicalized: mood→current_mood, moodDate→current_mood_date.
        # Fall back to legacy keys for safety against raw/uncanonicalized callers.
        mood_val = (
            latest_mood.get("current_mood")
            or latest_mood.get("mood")
        )
        if mood_val:
            merged["primary_mood"] = mood_val
            merged["mood"] = mood_val
        if latest_mood.get("notes"):
            merged["notes"] = latest_mood["notes"]
        md = (
            latest_mood.get("current_mood_date")
            or latest_mood.get("moodDate")
            or latest_mood.get("mood_date")
        )
        if md:
            merged["mood_date"] = md

    return merged


def _compute_data_availability_context(
    historical_snapshots: List[Any],
    today: Optional[date],
    health_params: Dict[str, Any],
    raw_data: Dict[str, Any],
    actual_today: Optional[date] = None,
) -> Dict[str, Any]:
    
    result: Dict[str, Any] = {
        "sleep_data_range": None,
        "sleep_days_missing": None,
        "sleep_data_staleness_days": None,
        "sleep_last_data_date": None,
        "hr_data_staleness_days": None,
        "hr_last_data_date": None,
        "steps_data_staleness_days": None,
        "steps_last_data_date": None,
        "mood_data_range": None,
        "mood_days_logged": None,
        "mood_data_staleness_days": None,
        "mood_last_data_date": None,
        "energy_data_staleness_days": None,
        "energy_last_data_date": None,
        "steps_data_range": None,
    }

    if today is None:
        return result

    if actual_today is None:
        actual_today = today

    now = datetime.now(ZoneInfo("UTC"))
    days_since_monday = now.weekday()
    start_of_week = now - timedelta(days=days_since_monday)
    start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)

    # Always fetch from start of week to today
    data_start = start_of_week
    data_end = now
    sleep_range = "start of week to today"

    result["sleep_data_range"] = sleep_range
    result["mood_data_range"] = "last 7 days"
    result["steps_data_range"] = sleep_range  # Same logic

    sleep_last_data_date_str = (
        health_param_value(
            health_params,
            "health_params_sleep_last_data_date",
            "sleep_last_data_date",
        )
        or health_params.get("health_params_sleep_lastnight_date")
        or health_params.get("sleep_last_data_date")
        or health_params.get("sleep_lastnight_date")
    )
    sleep_lastnight_h = health_param_value(
        health_params,
        "health_params_sleep_lastnight",
        HealthDataConstants.KEY_SLEEP_LASTNIGHT,
    )

    # Try to parse the sleep date
    last_sleep_date_from_api: Optional[date] = None
    if sleep_last_data_date_str:
        try:
            if "T" in sleep_last_data_date_str:
                last_sleep_date_from_api = date.fromisoformat(sleep_last_data_date_str.split("T")[0])
            else:
                last_sleep_date_from_api = date.fromisoformat(sleep_last_data_date_str)
        except Exception:
            pass

    # Fallback: try to get date from historical_snapshots
    by_date: Dict[date, float] = {}
    last_sleep_date: Optional[date] = last_sleep_date_from_api  # Start with API data

    for row in historical_snapshots:
        d = _snapshot_date(row)
        hours = _snapshot_field(row, CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS)
        if d is not None and hours is not None:
            by_date[d] = hours
            if last_sleep_date is None or d > last_sleep_date:
                last_sleep_date = d

    # Also check if today's sleep is available (from health_params)
    if sleep_lastnight_h is not None:
        by_date[today] = sleep_lastnight_h
        # If we have sleep hours but no date, check if it's actually from today
        if last_sleep_date_from_api is None:
            # Try to determine if sleep is from today or yesterday based on time
            if last_sleep_date is None or today > last_sleep_date:
                last_sleep_date = today

    # Count missing days in range
    total_days = (data_end.date() - data_start.date()).days + 1
    days_with_data = 0
    cursor = data_start.date()
    while cursor <= data_end.date():
        if cursor in by_date:
            days_with_data += 1
        cursor += timedelta(days=1)

    result["sleep_days_missing"] = max(0, total_days - days_with_data)

    cutoff_7d = today - timedelta(days=7)
    mood_days_count = 0
    moods_7d = raw_data.get("moods_7d") or []
    if isinstance(moods_7d, list):
        for entry in moods_7d:
            if not isinstance(entry, dict):
                continue
            mood_date_str = (
                entry.get("moodDate")
                or entry.get("mood_date")
                or entry.get("current_mood_date")
            )
            if not mood_date_str:
                continue
            try:
                if "T" in str(mood_date_str):
                    mood_d = date.fromisoformat(str(mood_date_str).split("T")[0])
                else:
                    mood_d = date.fromisoformat(str(mood_date_str))
            except Exception:
                continue
            if mood_d >= cutoff_7d and mood_d < today:
                mood_days_count += 1

    # Also check latest_mood
    latest_mood = raw_data.get("latest_mood")
    if latest_mood and isinstance(latest_mood, dict):
        mood_days_count = max(mood_days_count, 1)

    result["mood_days_logged"] = mood_days_count

    # Calculate mood staleness
    mood_last_data_date_str = health_param_value(
        health_params,
        "health_params_mood_last_data_date",
        "mood_last_data_date",
    )
    if mood_last_data_date_str:
        try:
            if "T" in mood_last_data_date_str:
                mood_date = date.fromisoformat(mood_last_data_date_str.split("T")[0])
            else:
                mood_date = date.fromisoformat(mood_last_data_date_str)
            result["mood_data_staleness_days"] = (actual_today - mood_date).days
            result["mood_last_data_date"] = mood_date.isoformat()
        except Exception:
            pass
    else:
        result["mood_data_staleness_days"] = 999
        result["mood_last_data_date"] = None

    # Calculate HR staleness
    hr_last_data_date_str = health_param_value(
        health_params,
        "health_params_hr_last_data_date",
        "hr_last_data_date",
    )
    if hr_last_data_date_str:
        try:
            if "T" in hr_last_data_date_str:
                hr_date = date.fromisoformat(hr_last_data_date_str.split("T")[0])
            else:
                hr_date = date.fromisoformat(hr_last_data_date_str)
            result["hr_data_staleness_days"] = (actual_today - hr_date).days
            result["hr_last_data_date"] = hr_date.isoformat()
        except Exception:
            pass
    else:
        result["hr_data_staleness_days"] = 999
        result["hr_last_data_date"] = None

    # Calculate steps staleness
    steps_last_data_date_str = health_param_value(
        health_params,
        "health_params_steps_last_data_date",
        "steps_last_data_date",
    )
    if steps_last_data_date_str:
        try:
            if "T" in steps_last_data_date_str:
                steps_date = date.fromisoformat(steps_last_data_date_str.split("T")[0])
            else:
                steps_date = date.fromisoformat(steps_last_data_date_str)
            result["steps_data_staleness_days"] = (actual_today - steps_date).days
            result["steps_last_data_date"] = steps_date.isoformat()
        except Exception:
            pass
    else:
        result["steps_data_staleness_days"] = 999
        result["steps_last_data_date"] = None

    # Calculate energy staleness (same style as steps: use explicit last-data date)
    energy_last_data_date_str = health_param_value(
        health_params,
        "health_params_energy_last_data_date",
        HealthDataConstants.KEY_ENERGY_LAST_DATA_DATE,
    )

    if energy_last_data_date_str:
        try:
            if "T" in energy_last_data_date_str:
                energy_date = date.fromisoformat(
                    energy_last_data_date_str.split("T")[0]
                )
            else:
                energy_date = date.fromisoformat(energy_last_data_date_str)
            result["energy_data_staleness_days"] = (actual_today - energy_date).days
            result["energy_last_data_date"] = energy_date.isoformat()
        except Exception:
            pass
    else:
        result["energy_data_staleness_days"] = 999
        result["energy_last_data_date"] = None

    # Sleep staleness: also handle missing case for symmetry.
    sleep_last_data_date_str = health_param_value(
        health_params,
        "health_params_sleep_last_data_date",
        "sleep_last_data_date",
    )
    if sleep_last_data_date_str:
        try:
            if "T" in sleep_last_data_date_str:
                sleep_date_d = date.fromisoformat(sleep_last_data_date_str.split("T")[0])
            else:
                sleep_date_d = date.fromisoformat(sleep_last_data_date_str)
            # Only override if the value is still None (preserves earlier
            # calculations that may already have set it from snapshots).
            if result.get("sleep_data_staleness_days") is None:
                result["sleep_data_staleness_days"] = (today - sleep_date_d).days
            if result.get("sleep_last_data_date") is None:
                result["sleep_last_data_date"] = sleep_date_d.isoformat()
        except Exception:
            pass
    else:
        if result.get("sleep_data_staleness_days") is None:
            result["sleep_data_staleness_days"] = 999
        if result.get("sleep_last_data_date") is None:
            result["sleep_last_data_date"] = None

    # Log all staleness info
    logger.info(
        f"📊 DATA STALENESS SUMMARY | "
        f"sleep={result.get('sleep_data_staleness_days')}d ({result.get('sleep_last_data_date')}) | "
        f"hr={result.get('hr_data_staleness_days')}d ({result.get('hr_last_data_date')}) | "
        f"steps={result.get('steps_data_staleness_days')}d ({result.get('steps_last_data_date')}) | "
        f"mood={result.get('mood_data_staleness_days')}d ({result.get('mood_last_data_date')}) | "
        f"energy={result.get('energy_data_staleness_days')}d ({result.get('energy_last_data_date')})"
    )

    return result


def _coerce_user_profile_health_conditions(value: Any) -> List[str]:
    if not value:
        return []
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str):
            if item:
                out.append(item)
            continue
        if not isinstance(item, dict):
            continue
        label = (
            item.get("condition_name")
            or item.get("condition_code")
            or item.get("name")
        )
        if isinstance(label, str) and label:
            out.append(label)
    return out


class DataProcessor:

    @staticmethod
    def _map_user_profile(raw_profile: Dict[str, Any]) -> Any:
        # Import here to avoid circular import at module level
        from insights.schemas.processed_context import UserProfileBlock

        return UserProfileBlock(
            user_id=raw_profile.get("userId") or raw_profile.get("user_id"),
            name=raw_profile.get("name"),
            age=(
                raw_profile.get("age")
                or raw_profile.get("user_age_years")
            ),
            gender=raw_profile.get("gender") or raw_profile.get("user_gender"),
            height_cm=raw_profile.get("height_cm") or raw_profile.get("heightCm") or raw_profile.get("user_height_centimeters"),
            weight_kg=raw_profile.get("weight_kg") or raw_profile.get("weightKg") or raw_profile.get("user_weight_kilograms"),
            bmi=raw_profile.get("bmi") or raw_profile.get("user_body_mass_index"),
            region=raw_profile.get("region"),
            timezone=raw_profile.get("time_data_timezone") or raw_profile.get("timezone") or raw_profile.get("user_timezone"),
            climate_type=raw_profile.get("climate_type"),
            job_title=raw_profile.get("jobTitle") or raw_profile.get("job_title") or raw_profile.get("user_job_title"),
            work_status=raw_profile.get("workStatus") or raw_profile.get("work_status") or raw_profile.get("user_work_status"),
            primary_goal=raw_profile.get("primaryGoal") or raw_profile.get("primary_goal"),
            diet_type=raw_profile.get("dietType") or raw_profile.get("diet_type"),
            smoking_status=raw_profile.get("smokingStatus") or raw_profile.get("smoking_status"),
            alcohol_consumption=raw_profile.get("alcoholConsumption") or raw_profile.get("alcohol_consumption"),
            current_health_conditions=_coerce_user_profile_health_conditions(
                raw_profile.get("currentHealthConditions")
                or raw_profile.get("current_health_conditions")
                or raw_profile.get("user_current_health_conditions")
            ),
        )

    @staticmethod
    def split_into_static_dynamic(
        signals: ExtractedSignals, today: Optional[date] = None
    ) -> Tuple[StaticSignalsBlock, DynamicSignalsBlock]:
        cutoff = today or date.today()
        cutoff_str = cutoff.isoformat()

        ps = signals.productivity_signals
        bs = signals.balance_snapshot
        ht = signals.historical_trends
        bp = signals.behavioral_patterns

        # ── Productivity 7d bucket (pure aggregates, no today fields) ────
        productivity_7d: Optional[Dict[str, Any]] = None
        productivity_reminders_7d: Optional[Dict[str, Any]] = None
        productivity_work_hours_7d: Optional[Dict[str, Any]] = None
        productivity_today: Optional[Dict[str, Any]] = None
        if ps is not None:
            productivity_7d = {
                "meeting_minutes_7d_avg": ps.meeting_minutes_7d_avg,
                "meeting_minutes_vs_avg_pct": ps.meeting_minutes_vs_avg_pct,
                "focus_minutes_7d_avg": ps.focus_minutes_7d_avg,
                "focus_minutes_vs_avg_pct": ps.focus_minutes_vs_avg_pct,
                "events_completion_7d_avg": ps.events_completion_7d_avg,
                "events_completion_trend": ps.events_completion_trend,
                "chronic_overload_days": ps.chronic_overload_days,
            }
            productivity_reminders_7d = {
                "high_priority_reminders_7d": ps.high_priority_reminders_7d,
                "reminders_by_type": dict(ps.reminders_by_type or {}),
                "reminders_dismissed_7d": ps.reminders_dismissed_7d,
                "reminders_pending_7d": ps.reminders_pending_7d,
            }
            productivity_work_hours_7d = {
                "work_hours_7d_scheduled_avg": ps.work_hours_7d_scheduled_avg,
                "work_hours_7d_target": ps.work_hours_7d_target,
                "work_hours_overload_days_7d": ps.work_hours_overload_days_7d,
                "work_hours_underload_days_7d": ps.work_hours_underload_days_7d,
            }
            productivity_today = {
                "level": ps.level,
                "meeting_fatigue": ps.meeting_fatigue,
                "schedule_focus_blocks": ps.focus_score,
                "events_completion_rate": ps.events_completion_rate,
                "meeting_completion_rate": ps.meeting_completion_rate,
                "reminders_completion_rate": ps.reminders_completion_rate,
                "reminders_due_today": ps.reminders_due_today,
                "meetings_due_today": ps.meetings_due_today,
                "upcoming_deadlines": ps.upcoming_deadlines,
                "nearest_reminder_title": ps.nearest_reminder_title,
                "nearest_reminder_due_in_mins": ps.nearest_reminder_due_in_mins,
                "nearest_reminder_priority": ps.nearest_reminder_priority,
                "overdue_reminders_count": ps.overdue_reminders_count,
                "reminders_collision": ps.reminders_collision,
            }

        # ── Balance 7d bucket (avg/std/trend delta from range endpoint) ───
        balance_7d: Optional[Dict[str, Any]] = None
        balance_today: Optional[Dict[str, Any]] = None
        if bs is not None:
            balance_7d = {
                "balance_score_7d_avg": bs.balance_score_7d_avg,
                "balance_score_7d_std": bs.balance_score_7d_std,
                "balance_score_trend_delta": bs.balance_score_trend_delta,
                "balance_score_7d_good_days": bs.balance_score_7d_good_days,
                "balance_score_7d_total_days": bs.balance_score_7d_total_days,
                "productivity_score_7d_avg": bs.productivity_score_7d_avg,
                "health_score_7d_avg": bs.health_score_7d_avg,
                "finance_score_7d_avg": bs.finance_score_7d_avg,
            }
            balance_today = {
                "productivity_score": bs.productivity_score,
                "health_score": bs.health_score,
                "finance_score": bs.finance_score,
                "balance_score": bs.balance_score,
                "balance_date": bs.balance_date,
                "balance_timezone": bs.balance_timezone,
            }
        health_7d_aggregates: Optional[Dict[str, Any]] = None
        if ht is not None and getattr(ht, "averages_3d", None):
            health_7d_aggregates = {
                "averages_3d": dict(ht.averages_3d or {}),
                "recent_days": [
                    d.model_dump() if hasattr(d, "model_dump") else d
                    for d in (getattr(ht, "recent_days", None) or [])
                ],
            }

        static_block = StaticSignalsBlock(
            historical_trends=ht,
            behavioral_patterns=bp,
            productivity_7d=productivity_7d,
            productivity_reminders_7d=productivity_reminders_7d,
            productivity_work_hours_7d=productivity_work_hours_7d,
            balance_7d=balance_7d,
            health_7d_aggregates=health_7d_aggregates,
            user_profile=signals.user_profile,
            cutoff_date=cutoff_str,
        )

        dynamic_block = DynamicSignalsBlock(
            meta=signals.meta,
            calendar_intelligence=signals.calendar_intelligence,
            health_signals=signals.health_signals,
            productivity_today=productivity_today,
            balance_today=balance_today,
            goal_signals=signals.goal_signals,
            finance_signals=signals.finance_signals,
            raw_data=signals.raw_data,
        )

        return static_block, dynamic_block

    @staticmethod
    def extract_signals(raw_data: Dict[str, Any]) -> ExtractedSignals:
        raw_data = clean_payload(
            raw_data,
            preserve_empty_items=True,
            keep_top_level_list_keys=(
                DEFAULT_KEEP_TOP_LEVEL_LIST_KEYS - _PROCESSOR_DROP_KEYS
            ),
            log_tag="extract_signals",
        )

        health_params = raw_data.get("health_params") or {}
        time_data = raw_data.get("time_data") or {}
        calendar_events = raw_data.get("calendar_events") or []
        calendar_metrics = raw_data.get("calendar_metrics") or {}
        historical_snapshots = raw_data.get("historical_snapshots") or []
        if not historical_snapshots:
            logger.debug(
                "historical_snapshots empty — snapshot API returned 0 rows; "
                "trend/behavioral processors will gracefully no-op; LLM payload "
                "strips the key (see qa_prompts.py: historical_snapshots is "
                "removed from glossary when empty to avoid hallucination)"
            )
        goal_contributions = raw_data.get("goal_contributions") or []
        # Transform goalContributions to match _process_custom_goal expected format
        goals = raw_data.get("goals") or []
        if not goals and not goal_contributions:
            user_profile = raw_data.get("user_profile") or {}
            up_goals = user_profile.get("user_goals") or user_profile.get("goals") or []
            if up_goals:
                goals = [
                    {
                        "goal_name": g.get("goal_name") or g.get("name", ""),
                        "category_code": g.get("category_code") or g.get("category", ""),
                        "action_code": g.get("action_code") or g.get("action", ""),
                        "target_value": g.get("target_value") or g.get("target", 0),
                        "unit": g.get("unit", ""),
                    }
                    for g in up_goals
                    if isinstance(g, dict)
                ]
        if goal_contributions and not goals:
            goals = [
                {
                    "goal_name": g.get("goalName", ""),
                    "category_code": g.get("category", "").upper(),
                    "target_value": g.get("targetAmount", 0),
                    "contributed": g.get("contributed", 0),
                    "goal_type": "monthly",
                }
                for g in goal_contributions
            ]
        user_profile = raw_data.get("user_profile") or {}
        group = user_profile.get("group", "")
        today_for_backfill = None
        try:
            today_for_backfill = _parse_today(time_data)
        except ValueError:
            pass

        try:
            _backfill_health_params(health_params, raw_data)
        except Exception as _e:
            logger.warning(f"health_params backfill skipped: {_e}")

        from insights.processors.balance_snapshot_processor import (
            BalanceSnapshotProcessor,
        )

        balance_snapshot = None
        try:
            balance_snapshot = BalanceSnapshotProcessor.process(raw_data=raw_data)
        except Exception as e:
            logger.error(f"DataProcessor Block 9 (BalanceSnapshot) failed: {e}")

        if today_for_backfill is not None and historical_snapshots:
            for hp_key, snapshot_field in (
                ("health_params_sleep_quality_score", CanonicalField.SLEEP_QUALITY_SCORE_ONE_NIGHT),
                ("bedtime", "h_bedtime"),
                ("wake_time", "h_wake_time"),
            ):
                if health_params.get(hp_key) is None:
                    backfilled = _latest_snapshot_value(
                        historical_snapshots, today_for_backfill, snapshot_field
                    )
                    if backfilled is not None:
                        health_params[hp_key] = backfilled
                        
        key_signals = build_key_signals(
            health_params=health_params,
            calendar_metrics=calendar_metrics,
            time_data=time_data,
            latest_mood=raw_data.get("latest_mood"),
            calendar_events=calendar_events,
            balance_snapshot=balance_snapshot,
        )

        def get_recent_completed_tasks(
            raw_data: Dict[str, Any], time_data: Dict[str, Any]
        ) -> List[str]:
            current_time_str = time_data.get("time_data_current_time_iso", "")
            if not current_time_str:
                return []
            if "+" in current_time_str:
                current_time_str = current_time_str.split("+")[0]
            else:
                idx = current_time_str.rfind("-")
                if idx > 10:
                    current_time_str = current_time_str[:idx]

            completed = []
            today_date = current_time_str[:10]

            events = raw_data.get("calendar_events", [])
            for event in events:
                end_time = event.get("endTime")
                if not end_time:
                    continue
                if end_time.startswith(today_date) and end_time <= current_time_str:
                    summary = event.get("summary", "Unnamed Task")
                    completed.append(summary)
            return completed

        def get_next_event_info(raw_data: Dict[str, Any], time_data: Dict[str, Any]):
            """Return (summary, minutes_away, start_time, category, event_type)."""
            from zoneinfo import ZoneInfo

            current_time_str = time_data.get("time_data_current_time_iso", "")
            if not current_time_str:
                return None, None, None, None, None

            tz_name = (
                raw_data.get("timezone")
                or time_data.get("timezone")
                or time_data.get("user_timezone")
                or "UTC"
            )
            try:
                tz = ZoneInfo(str(tz_name))
            except Exception:
                tz = ZoneInfo("UTC")

            try:
                current_dt = datetime.fromisoformat(
                    current_time_str.replace("Z", "+00:00")
                )
                if current_dt.tzinfo is None:
                    current_dt = current_dt.replace(tzinfo=tz)
                else:
                    current_dt = current_dt.astimezone(tz)
            except Exception:
                fallback_mins = time_data.get("time_data_time_to_next_event")
                return None, fallback_mins, None, None, None

            events = raw_data.get("calendar_events", [])
            best_mins: Optional[int] = None
            best_summary: Optional[str] = None
            best_start: Optional[str] = None
            best_category: Optional[str] = None
            best_event_type: Optional[str] = None

            for event in events:
                start_time = event.get("startTime") or event.get("start_time")
                if not start_time:
                    continue
                try:
                    event_start = datetime.fromisoformat(
                        start_time.replace("Z", "+00:00")
                    )
                    if event_start.tzinfo is None:
                        event_start = event_start.replace(tzinfo=tz)
                    else:
                        event_start = event_start.astimezone(tz)
                except Exception:
                    continue

                if event_start <= current_dt:
                    continue

                mins = int((event_start - current_dt).total_seconds() / 60)
                if best_mins is None or mins < best_mins:
                    best_mins = mins
                    best_summary = (
                        event.get("summary")
                        or event.get("title")
                        or "Unnamed Event"
                    )
                    best_start = start_time
                    raw_cat = event.get("category")
                    if raw_cat is not None and str(raw_cat).strip():
                        from services.executor.constant import (
                            CalendarEventCategoryConstants,
                        )

                        best_category = CalendarEventCategoryConstants.resolve(raw_cat)
                    else:
                        best_category = None
                    raw_type = event.get("eventType") or event.get("event_type")
                    best_event_type = (
                        str(raw_type).strip().lower() if raw_type else None
                    ) or None

            if best_mins is not None:
                return (
                    best_summary,
                    best_mins,
                    best_start,
                    best_category,
                    best_event_type,
                )

            return None, time_data.get("time_data_time_to_next_event"), None, None, None

        def get_lifecycle_stage(raw_data: Dict[str, Any]) -> str:
            snapshots = raw_data.get("historical_snapshots", [])
            return "day_1_new_user" if len(snapshots) == 0 else "returning_user"

        def get_data_confidence(
            health_params: Dict[str, Any],
            time_data: Dict[str, Any],
            calendar_events: list,
        ) -> Dict[str, str]:
            confidence = {}
            # Health
            if (
                health_param_value(
                    health_params,
                    "health_params_hrv_score",
                    HealthDataConstants.KEY_HRV_SCORE,
                )
                is None
                and health_param_value(
                    health_params,
                    "health_params_resting_heart_rate",
                    HealthDataConstants.KEY_RESTING_HEART_RATE,
                )
                is None
            ):
                confidence["health"] = "simulated_no_device"
            else:
                confidence["health"] = "real_connected"
            # Calendar
            if not calendar_events:
                confidence["calendar"] = "simulated_no_sync"
            else:
                confidence["calendar"] = "real_connected"
            return confidence

        # Compute data availability context for narrative enrichment
        data_avail = _compute_data_availability_context(
            historical_snapshots=historical_snapshots,
            today=today_for_backfill,
            health_params=health_params,
            raw_data=raw_data,
            actual_today=datetime.now().date(),
        )

        meta = None
        try:
            (
                next_event_summary,
                minutes_until_next_event,
                next_event_start_time,
                next_event_category,
                next_event_type,
            ) = get_next_event_info(raw_data, time_data)
            meta = MetaBlock(
                current_time=time_data.get("time_data_current_time_iso", ""),
                time_phase=compute_time_phase(
                    group=group, time_data=time_data, health_params=health_params
                ),
                day_of_week=time_data.get("day_of_week", ""),
                is_weekend=bool(time_data.get("time_data_is_weekend", False)),
                lifecycle_stage=get_lifecycle_stage(raw_data),
                minutes_until_next_event=minutes_until_next_event,
                next_event_summary=next_event_summary,
                next_event_start_time=next_event_start_time,
                next_event_category=next_event_category,
                next_event_type=next_event_type,
                recent_completed_tasks=get_recent_completed_tasks(raw_data, time_data),
                data_confidence=get_data_confidence(
                    health_params, time_data, calendar_events
                ),
                hard_constraints=compute_hard_constraints(
                    group=group, time_data=time_data, health_params=health_params
                ),
                data_gaps=build_data_gaps(health_params=health_params),
                active_hours_end_time=time_data.get("active_hours_end_time"),
                sleep_data_range=data_avail.get("sleep_data_range"),
                sleep_days_missing=data_avail.get("sleep_days_missing"),
                sleep_data_staleness_days=data_avail.get("sleep_data_staleness_days"),
                sleep_last_data_date=data_avail.get("sleep_last_data_date"),
                hr_data_staleness_days=data_avail.get("hr_data_staleness_days"),
                hr_last_data_date=data_avail.get("hr_last_data_date"),
                steps_data_staleness_days=data_avail.get("steps_data_staleness_days"),
                steps_last_data_date=data_avail.get("steps_last_data_date"),
                mood_data_range=data_avail.get("mood_data_range"),
                mood_days_logged=data_avail.get("mood_days_logged"),
                mood_data_staleness_days=data_avail.get("mood_data_staleness_days"),
                mood_last_data_date=data_avail.get("mood_last_data_date"),
                energy_data_staleness_days=data_avail.get("energy_data_staleness_days"),
                energy_last_data_date=data_avail.get("energy_last_data_date"),
                steps_data_range=data_avail.get("steps_data_range"),
            )
        except Exception as e:
            logger.error(f"DataProcessor Block 1 (Meta) failed: {e}")

        health_signals = None
        try:
            # Pass staleness info to HealthSignalProcessor for stale data filtering
            staleness_info = data_avail
            health_signals = HealthSignalProcessor.process(
                raw_data=raw_data,
                time_data=time_data,
                historical_snapshots=historical_snapshots,
                staleness_info=staleness_info,
            )
        except Exception as e:
            logger.error(
                f"DataProcessor Block 2 (HealthSignals) failed: {e}; "
                "falling back to minimal signals built from raw_data"
            )
            try:
                health_signals = _build_minimal_health_signals_from_raw(
                    raw_data=raw_data,
                    health_params=health_params,
                    time_data=time_data,
                )
            except Exception as _fallback_err:
                logger.error(
                    f"Minimal HealthSignals fallback also failed: {_fallback_err}"
                )
                health_signals = None
        if health_signals is not None:
            try:
                pass  # mood fields now owned exclusively by MoodSignal; no cardio mirror needed
            except Exception as e:
                logger.error(
                    f"DataProcessor Block 2b (HealthSignals enrichment) failed: {e}"
                )

        calendar_intelligence = None
        try:
            calendar_intelligence = CalendarIntelligenceProcessor.process(
                calendar_events=calendar_events,
                calendar_metrics=calendar_metrics,
                time_data=time_data,
                reminders=raw_data.get("today_reminders") or [],
                user_timezone=(
                    raw_data.get("time_data_timezone")
                    or (time_data or {}).get("timezone")
                    or (raw_data.get("user_profile") or {}).get(
                        "time_data_timezone"
                    )
                    or (raw_data.get("user_profile") or {}).get("timezone")
                ),
            )
        except Exception as e:
            logger.error(f"DataProcessor Block 3 (CalendarIntelligence) failed: {e}")

        goal_signals = None
        if calendar_intelligence is not None:
            try:
                goal_signals = GoalProgressProcessor.process(
                    health_signals=health_signals,
                    calendar_intelligence=calendar_intelligence,
                    user_goals=goal_contributions or goals,
                    user_profile=user_profile,
                )
            except Exception as e:
                logger.error(f"DataProcessor Block 4 (GoalProgress) failed: {e}")
        else:
            logger.warning(
                "DataProcessor Block 4 skipped: calendar_intelligence unavailable"
            )

        behavioral_patterns = None
        try:
            today = (
                today_for_backfill
                if today_for_backfill is not None
                else _parse_today(time_data)
            )
            behavioral_patterns = BehavioralPatternProcessor.process(
                historical_snapshots=historical_snapshots,
                today=today,
                steps_goal=health_param_value(
                    health_params,
                    "health_params_steps_goal",
                    HealthDataConstants.KEY_STEPS_GOAL,
                ),
                raw_data=raw_data,
            )
        except Exception as e:
            logger.error(f"DataProcessor Block 5 (BehavioralPatterns) failed: {e}")

        historical_trends = None
        try:
            if today_for_backfill is not None:
                historical_trends = HistoricalTrendsProcessor.process(
                    historical_snapshots,
                    today_for_backfill,
                    today_health_stats=raw_data.get("today_health_stats"),
                    productivity_summary=raw_data.get("productivity_summary"),
                    latest_mood=raw_data.get("latest_mood"),
                    productivity_summaries_7d=raw_data.get(
                        "productivity_summaries_7d"
                    ),
                )
        except Exception as e:
            logger.error(f"DataProcessor Block 7 (HistoricalTrends) failed: {e}")

        productivity_signals = None
        try:
            productivity_signals = ProductivitySignalProcessor.process(
                raw_data=raw_data
            )
        except Exception as e:
            logger.error(f"DataProcessor Block 6 (ProductivitySignals) failed: {e}")

        finance_signals = None
        collect_domain = (raw_data.get("collect_domain") or "").strip().lower()
        collect_topics = raw_data.get("collect_topics") or []
        allow_finance = (
            collect_domain == "all"
            or (
                isinstance(collect_topics, list)
                and "finance" in collect_topics
            )
            or any(
                raw_data.get(k)
                for k in (
                    "finance_summary",
                    "finance_goals",
                    "finance_bills",
                    "finance_logs",
                    "finance_budgets",
                )
            )
        )
        # Daily insight domains (health / productivity / overall) never build finance signals
        if collect_domain in ("health", "productivity", "overall"):
            allow_finance = False
        if allow_finance:
            try:
                finance_signals = FinanceSignalProcessor.process(raw_data=raw_data)
            except Exception as e:
                logger.error(f"DataProcessor Block 8 (FinanceSignals) failed: {e}")
        else:
            # Ensure raw_data cannot reintroduce finance downstream
            for _fk in (
                "finance_summary",
                "finance_goals",
                "finance_bills",
                "finance_logs",
                "finance_budgets",
                "goal_contributions",
            ):
                raw_data.pop(_fk, None)

        balance_score = raw_data.get("balance_score")
        if balance_score is not None:
            if isinstance(balance_score, dict):
                pass  # already a dict
            elif hasattr(balance_score, "model_dump"):
                balance_score = balance_score.model_dump()
            else:
                balance_score = None

        user_profile_block = DataProcessor._map_user_profile(user_profile)

        try:
            narrative_ctx = _build_narrative_context(
                raw_data=raw_data,
                meta=meta,
                health_signals=health_signals,
                historical_trends=historical_trends,
                calendar_intelligence=calendar_intelligence,
                productivity_signals=productivity_signals,
                finance_signals=finance_signals,
                user_profile=user_profile,
                behavioral_patterns=behavioral_patterns,
                goal_signals=goal_signals,
            )
        except Exception as _narr_err:
            logger.warning(f"_build_narrative_context failed: {_narr_err}")
            narrative_ctx = {}

        extracted = ExtractedSignals(
            meta=meta,
            health_signals=health_signals,
            calendar_intelligence=calendar_intelligence,
            goal_signals=goal_signals,
            behavioral_patterns=behavioral_patterns,
            historical_trends=historical_trends,
            user_profile=user_profile_block,
            productivity_signals=productivity_signals,
            raw_data=raw_data,
            balance_score=balance_score,
            balance_snapshot=balance_snapshot,
            finance_signals=finance_signals,
            narrative_context=narrative_ctx,
        )
        return extracted
