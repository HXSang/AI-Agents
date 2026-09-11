from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from insights.health.canonical_field_mapping import CanonicalField
from insights.health.health_processor.heart_rate.constants import (
    HEART_RATE_RESTING_ELEVATED_DELTA_BPM,
    HEART_RATE_RESTING_HIGH_BPM,
)
from insights.schemas.processed_context import (
    HealthPeriodAggregates,
    HealthPeriodMetric,
)
from services.executor.constant import APIResponseKeys

_STEPS_DAILY_FIELDS: Tuple[str, ...] = (
    CanonicalField.STEP_COUNT_ONE_DAY,
    "steps",
)

_SLEEP_DAILY_FIELDS: Tuple[str, ...] = (
    CanonicalField.SLEEP_DURATION_FROM_SUMMARY_ONE_NIGHT_HOURS,
    "total",
)

_HR_LATEST_DAILY_FIELDS: Tuple[str, ...] = (
    CanonicalField.MOST_RECENT_HEART_RATE_MEASUREMENT_ONE_DAY_BPM,
    "latestHR",
)

_HR_RESTING_DAILY_FIELDS: Tuple[str, ...] = (
    CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM,
    "restingHeartRate",
)

_ENERGY_BURN_DAILY_FIELDS: Tuple[str, ...] = (
    CanonicalField.ENERGY_BURN_ONE_DAY_KCAL,
    "energyBurn",
)


def _type_block(stats: Optional[Dict[str, Any]], type_key: str) -> Dict[str, Any]:
    """Return ``stats[type_key]`` if it is a dict, else ``{}``."""
    if not isinstance(stats, dict):
        return {}
    entry = stats.get(type_key)
    return entry if isinstance(entry, dict) else {}


def _data_rows(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = block.get("data")
    return data if isinstance(data, list) else []


def _summary_data(block: Dict[str, Any]) -> Dict[str, Any]:
    sd = block.get("summaryData")
    if sd is None:
        sd = block.get("summary_data")
    return sd if isinstance(sd, dict) else {}


def _row_date_local(row: Dict[str, Any], tz: Optional[str]) -> Optional[date]:
    raw = row.get("date")
    if not raw:
        return None
    try:
        if isinstance(raw, date) and not isinstance(raw, datetime):
            return raw
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None
    if tz:
        try:
            from zoneinfo import ZoneInfo

            dt = dt.astimezone(ZoneInfo(tz))
        except Exception:
            pass
    return dt.date()


def _row_value(row: Dict[str, Any], fields: Tuple[str, ...]) -> Optional[float]:
    """Return the first numeric value in ``row`` for any key in ``fields``."""
    for f in fields:
        v = row.get(f)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def _calendar_week_range(
    today: date, tz: Optional[str]
) -> Tuple[date, date]:
    if tz:
        try:
            from zoneinfo import ZoneInfo

            now_local = datetime.combine(
                today, datetime.min.time(), tzinfo=ZoneInfo(tz)
            )
            week_start = today - timedelta(days=now_local.weekday())
            return week_start, today
        except Exception:
            pass
    week_start = today - timedelta(days=today.weekday())
    return week_start, today


def _rows_in_calendar_week(
    rows: List[Dict[str, Any]],
    today: date,
    tz: Optional[str],
) -> List[Dict[str, Any]]:
    """Return only ``rows`` whose local date falls in ``[week_start, today]``."""
    week_start, _ = _calendar_week_range(today, tz)
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        d = _row_date_local(r, tz)
        if d is None:
            continue
        if week_start <= d <= today:
            out.append(r)
    return out


def _get_current_calendar_week_boundaries(
    today: date, tz: Optional[str]
) -> Tuple[date, date]:
    week_start, _ = _calendar_week_range(today, tz)
    week_end = week_start + timedelta(days=6)
    return week_start, week_end


def _compute_current_week_avg(
    rows: List[Dict[str, Any]],
    value_getter,
    today: date,
    tz: Optional[str],
) -> Optional[float]:
    week_rows = _rows_in_calendar_week(rows, today, tz)
    vals: List[float] = []
    for r in week_rows:
        v = value_getter(r)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(float(v))
    return _mean(vals)


def _calendar_week_avg_for_metric(
    stats: Dict[str, Any],
    metric: str,
    today: date,
    tz: Optional[str],
) -> Optional[float]:
    src = _CALENDAR_WEEK_METRIC_SOURCES.get(metric)
    if src is not None:
        type_key, daily_fields, summary_key = src
        block = _type_block(stats, type_key)
        sd = _summary_data(block)
        if summary_key and isinstance(sd.get(summary_key), (int, float)):
            return float(sd[summary_key])
        # Fall through to data[] mean
        rows = _data_rows(block)
        vals: List[float] = []
        for r in _rows_in_calendar_week(rows, today, tz):
            v = _row_value(r, daily_fields)
            if v is not None:
                vals.append(v)
        if vals:
            return _mean(vals)

    if metric == "steps":
        fields = _STEPS_DAILY_FIELDS
        type_key = "STEPS"
    elif metric == "sleep_hours":
        fields = _SLEEP_DAILY_FIELDS
        type_key = "SLEEP"
    elif metric == "resting_heart_rate":
        fields = _HR_LATEST_DAILY_FIELDS
        type_key = "HR"
    elif metric == "calories_burned":
        fields = _ENERGY_BURN_DAILY_FIELDS
        type_key = "ENERGY"
    else:
        return None
    block = _type_block(stats, type_key)
    rows = _data_rows(block)
    vals = []
    for r in _rows_in_calendar_week(rows, today, tz):
        v = _row_value(r, fields)
        if v is not None:
            vals.append(v)
    return _mean(vals)

_CALENDAR_WEEK_METRIC_SOURCES: Dict[str, Tuple[str, Tuple[str, ...], Optional[str]]] = {
    "steps": ("STEPS", _STEPS_DAILY_FIELDS, None),
    "sleep_hours": ("SLEEP", _SLEEP_DAILY_FIELDS, None),
    "resting_heart_rate": ("HR", _HR_LATEST_DAILY_FIELDS, None),
    "calories_burned": (
        "ENERGY",
        _ENERGY_BURN_DAILY_FIELDS,
        APIResponseKeys.AVG_CURRENT_WEEK_ENERGY_BURN,
    ),
}


def _today_row(
    rows: List[Dict[str, Any]],
    today: date,
    tz: Optional[str],
) -> Optional[Dict[str, Any]]:
    best: Optional[Tuple[datetime, Dict[str, Any]]] = None
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        raw = r.get("date")
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except Exception:
            continue
        if tz:
            try:
                from zoneinfo import ZoneInfo

                dt = dt.astimezone(ZoneInfo(tz))
            except Exception:
                pass
        local_date = dt.date()
        if local_date > today:
            continue
        if best is None or dt > best[0]:
            best = (dt, r)
    return best[1] if best else None


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _round_opt(v: Optional[float], ndigits: int = 2) -> Optional[float]:
    if v is None:
        return None
    return round(float(v), ndigits)


def _steps_today_and_week(
    block: Dict[str, Any],
    today: date,
    tz: Optional[str],
) -> Tuple[Optional[float], Optional[float]]:
    rows = _data_rows(block)
    today_row = _today_row(rows, today, tz)
    today_v = (
        _row_value(today_row, _STEPS_DAILY_FIELDS) if today_row else None
    )
    week_rows = _rows_in_calendar_week(rows, today, tz)
    week_vals = [
        v for v in (_row_value(r, _STEPS_DAILY_FIELDS) for r in week_rows) if v is not None
    ]
    return today_v, _mean(week_vals)


def _sleep_today_and_week(
    block: Dict[str, Any],
    today: date,
    tz: Optional[str],
) -> Tuple[Optional[float], Optional[float]]:
    rows = _data_rows(block)
    today_row = _today_row(rows, today, tz)
    today_v = (
        _row_value(today_row, _SLEEP_DAILY_FIELDS) if today_row else None
    )
    week_rows = _rows_in_calendar_week(rows, today, tz)
    week_vals = [
        v for v in (_row_value(r, _SLEEP_DAILY_FIELDS) for r in week_rows) if v is not None
    ]
    return today_v, _mean(week_vals)


def _hr_today_and_week(
    block: Dict[str, Any],
    today: date,
    tz: Optional[str],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    rows = _data_rows(block)
    today_row = _today_row(rows, today, tz)
    week_rows = _rows_in_calendar_week(rows, today, tz)
    today_v = (
        _row_value(today_row, _HR_LATEST_DAILY_FIELDS) if today_row else None
    )
    true_rhr = (
        _row_value(today_row, _HR_RESTING_DAILY_FIELDS) if today_row else None
    )
    week_vals = [
        v
        for v in (
            _row_value(r, _HR_LATEST_DAILY_FIELDS) for r in week_rows
        )
        if v is not None
    ]
    return today_v, _mean(week_vals), true_rhr


def _energy_today_and_week(
    block: Dict[str, Any],
    today: date,
    tz: Optional[str],
    energy_today_summary: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    today_v = energy_today_summary
    if today_v is None:
        rows = _data_rows(block)
        today_row = _today_row(rows, today, tz)
        if today_row is not None:
            today_v = _row_value(today_row, _ENERGY_BURN_DAILY_FIELDS)
    sd = _summary_data(block)
    week_val: Any = None
    for key in (
        APIResponseKeys.AVG_CURRENT_WEEK_ENERGY_BURN,
        CanonicalField.AVERAGE_DAILY_ACTIVE_ENERGY_BURN_CURRENT_WEEK_KCAL,
    ):
        week_val = sd.get(key)
        if week_val is not None:
            break
    week_avg: Optional[float] = None
    if isinstance(week_val, (int, float)) and not isinstance(week_val, bool):
        week_avg = float(week_val)
    return today_v, week_avg


def _build_period_metric(
    *,
    today_v: Optional[float],
    week_avg: Optional[float],
    month_avg: Optional[float],
    prev_month_avg: Optional[float],
    baseline_avg: Optional[float],
) -> Optional[HealthPeriodMetric]:
    if all(
        x is None
        for x in (today_v, week_avg, month_avg, prev_month_avg, baseline_avg)
    ):
        return None
    delta_baseline = (
        _round_opt(today_v - baseline_avg)
        if today_v is not None and baseline_avg is not None
        else None
    )
    delta_week = (
        _round_opt(today_v - week_avg)
        if today_v is not None and week_avg is not None
        else None
    )
    return HealthPeriodMetric(
        today=_round_opt(today_v),
        week_avg=_round_opt(week_avg),
        month_avg=_round_opt(month_avg),
        prev_month_avg=_round_opt(prev_month_avg),
        baseline_avg=_round_opt(baseline_avg),
        delta_vs_baseline=delta_baseline,
        delta_vs_week=delta_week,
    )


def _steps_summary(
    block: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float]]:
    sd = _summary_data(block)

    def _nested_steps(value: Any) -> Optional[float]:
        if isinstance(value, dict):
            v = value.get("steps")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    cur = _nested_steps(sd.get(APIResponseKeys.CURRENT_MONTH_AVG))
    if cur is None:
        cur = _nested_steps(sd.get(CanonicalField.AVERAGE_DAILY_STEPS_CURRENT_MONTH))
    prev = _nested_steps(sd.get(APIResponseKeys.PREVIOUS_MONTH_AVG))
    if prev is None:
        prev = _nested_steps(sd.get(CanonicalField.AVERAGE_DAILY_STEPS_PREVIOUS_MONTH))
    return cur, prev


def _energy_summary(
    block: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    sd = _summary_data(block)

    def _f(*keys: str) -> Optional[float]:
        for k in keys:
            v = sd.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        return None

    return {
        "month_resting_total": _f(
            APIResponseKeys.TOTAL_RESTING_ENERGY,
            CanonicalField.RESTING_ENERGY_ONE_DAY_KCAL,
        ),
        "total_active_energy": _f(
            APIResponseKeys.TOTAL_ACTIVE_ENERGY,
            CanonicalField.ACTIVE_ENERGY_ONE_DAY_KCAL,
        ),
        "current_month_avg": _f(
            APIResponseKeys.CURRENT_MONTH_AVG,
            CanonicalField.AVERAGE_DAILY_ACTIVE_ENERGY_BURN_CURRENT_MONTH_KCAL,
        ),
        "week_avg": _f(
            APIResponseKeys.AVG_CURRENT_WEEK_ENERGY_BURN,
            CanonicalField.AVERAGE_DAILY_ACTIVE_ENERGY_BURN_CURRENT_WEEK_KCAL,
        ),
        "previous_month_avg": _f(
            APIResponseKeys.PREVIOUS_MONTH_AVG,
            CanonicalField.AVERAGE_DAILY_ACTIVE_ENERGY_BURN_PREVIOUS_MONTH_KCAL,
        ),
    }


def compute_health_period_aggregates(
    *,
    today_health_stats: Optional[Dict[str, Any]] = None,
    today: Optional[date] = None,
    tz: Optional[str] = None,
    timezone: Optional[str] = None,
    health_params: Optional[Dict[str, Any]] = None,
    historical_snapshots: Optional[List[Any]] = None,  # noqa: ARG001 — ignored
) -> HealthPeriodAggregates:
    if today is None:
        today = datetime.utcnow().date()
    tz_effective = tz or timezone
    _ = historical_snapshots
    _ = health_params

    stats = today_health_stats if isinstance(today_health_stats, dict) else {}

    steps_block = _type_block(stats, "STEPS")
    sleep_block = _type_block(stats, "SLEEP")
    hr_block = _type_block(stats, "HR")
    energy_block = _type_block(stats, "ENERGY")

    steps_today, steps_week = _steps_today_and_week(steps_block, today, tz_effective)
    steps_month_api, steps_prev_month_api = _steps_summary(steps_block)
    sleep_today, sleep_week = _sleep_today_and_week(sleep_block, today, tz_effective)
    hr_today, hr_week, true_rhr_today = _hr_today_and_week(
        hr_block, today, tz_effective
    )
    rhr_baseline: Optional[float] = None

    # ---- ENERGY / CALORIES --------------------------------------------------
    energy_summary = _energy_summary(energy_block)
    energy_today_summary = energy_summary.get("total_active_energy")
    calories_today, calories_week = _energy_today_and_week(
        energy_block, today, tz_effective, energy_today_summary
    )

    # ---- Build per-metric HealthPeriodMetric --------------------------------
    steps_metric = _build_period_metric(
        today_v=steps_today,
        week_avg=steps_week,
        month_avg=steps_month_api,
        prev_month_avg=steps_prev_month_api,
        baseline_avg=None,  # API does not expose steps baseline
    )

    sleep_metric = _build_period_metric(
        today_v=sleep_today,
        week_avg=sleep_week,
        month_avg=None,
        prev_month_avg=None,
        baseline_avg=None,
    )

    rhr_metric = _build_period_metric(
        today_v=hr_today,
        week_avg=hr_week,
        month_avg=None,
        prev_month_avg=None,
        baseline_avg=rhr_baseline,
    )

    active_minutes_metric = None

    total_workout_min_metric = None

    calories_metric = _build_period_metric(
        today_v=calories_today,
        week_avg=calories_week,
        month_avg=energy_summary.get("current_month_avg"),
        prev_month_avg=energy_summary.get("previous_month_avg"),
        baseline_avg=None,
    )

    rhr_elevated: Optional[bool] = None
    if true_rhr_today is not None and rhr_baseline is not None:
        rhr_elevated = (
            (true_rhr_today - rhr_baseline) >= HEART_RATE_RESTING_ELEVATED_DELTA_BPM
        ) or (true_rhr_today >= HEART_RATE_RESTING_HIGH_BPM)

    # workout_high_recent_load — not exposed by /api/health/summaries/latest.
    workout_high: Optional[bool] = None

    # energy_week_vs_prior_month_pct: derived from ENERGY summary fields.
    energy_week_val = energy_summary.get("week_avg")
    energy_prev_val = energy_summary.get("previous_month_avg")
    energy_pct: Optional[float] = None
    if energy_week_val is not None and energy_prev_val not in (None, 0):
        energy_pct = round(
            (energy_week_val - energy_prev_val) / energy_prev_val * 100, 1
        )

    return HealthPeriodAggregates(
        steps=steps_metric,
        sleep_hours=sleep_metric,
        resting_heart_rate=rhr_metric,
        active_minutes=active_minutes_metric,
        total_workout_min=total_workout_min_metric,
        calories_burned=calories_metric,
        energy_month_resting_total=_round_opt(
            energy_summary.get("month_resting_total")
        ),
        energy_total_active_today=_round_opt(
            energy_summary.get("total_active_energy")
        ),
        energy_current_month_avg=_round_opt(
            energy_summary.get("current_month_avg")
        ),
        energy_week_avg=_round_opt(energy_summary.get("week_avg")),
        energy_prev_month_avg=_round_opt(energy_summary.get("previous_month_avg")),
        energy_week_vs_prior_month_pct=_round_opt(energy_pct, 1),
        rhr_elevated=rhr_elevated,
        workout_high_recent_load=workout_high,
    )