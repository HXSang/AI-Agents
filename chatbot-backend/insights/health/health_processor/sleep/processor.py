"""Sleep signal processor — single source of truth for sleep calculations.

Pipeline: raw aggregation → derived metrics → trend → baseline → anomalies
→ evidence-rich signals (summary / signals / overall). The LLM only interprets.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from insights.health.canonical_field_mapping import CanonicalField
from insights.health.health_processor.common.utils import health_param_value
from insights.health.health_processor.sleep.constants import (
    SLEEP_ANOMALY_DEEP_SHARE_LOW,
    SLEEP_ANOMALY_EARLY_WAKE_THRESHOLD_MIN,
    SLEEP_ANOMALY_LATE_BEDTIME_THRESHOLD_MIN,
    SLEEP_ANOMALY_LONG_H,
    SLEEP_ANOMALY_LOW_EFFICIENCY,
    SLEEP_ANOMALY_REM_SHARE_LOW,
    SLEEP_ANOMALY_SHORT_H,
    SLEEP_BASELINE_WINDOW_DAYS,
    SLEEP_BASELINE_WINDOW_DAYS_EFFICIENCY,
    SLEEP_BEDTIME_CONSISTENT_STD_MIN,
    SLEEP_BEDTIME_LATE_CREEP_MIN,
    SLEEP_DEBT_MILD_H,
    SLEEP_DEBT_MODERATE_H,
    SLEEP_DEBT_OK_H,
    SLEEP_DEEP_SHARE_LOW,
    SLEEP_DEEP_SHARE_OK,
    SLEEP_DEFAULT_BEDTIME_EARLY_HOUR,
    SLEEP_DEFAULT_BEDTIME_LATE_HOUR,
    SLEEP_EFFICIENCY_FAIR,
    SLEEP_EFFICIENCY_GOOD,
    SLEEP_ESTIMATE_DEEP_RATIO,
    SLEEP_ESTIMATE_LIGHT_RATIO,
    SLEEP_ESTIMATE_REM_RATIO,
    SLEEP_GOAL_MAX_H,
    SLEEP_GOAL_MIN_H,
    SLEEP_GOAL_RECOMMENDED_H,
    SLEEP_MIN_DAYS_FOR_BASELINE,
    SLEEP_MIN_DAYS_FOR_CONSISTENCY,
    SLEEP_MIN_DAYS_FOR_TREND,
    SLEEP_RECOMMENDED_FLOOR_H,
    SLEEP_OVERALL_WEIGHT_DEBT,
    SLEEP_OVERALL_WEIGHT_DURATION,
    SLEEP_OVERALL_WEIGHT_EFFICIENCY,
    SLEEP_OVERALL_WEIGHT_SCORE,
    SLEEP_REM_SHARE_LOW,
    SLEEP_REM_SHARE_OK,
    SLEEP_SCORE_HIGH,
    SLEEP_SCORE_LOW,
    SLEEP_SCORE_OK,
    SLEEP_SCORE_VERY_HIGH,
    SLEEP_SUMMARY_WINDOW_DAYS,
    SLEEP_TREND_SIGNIFICANT_DIFF_EFF,
    SLEEP_TREND_SIGNIFICANT_DIFF_H,
    SLEEP_VS_BASELINE_HIGHER_EFF,
    SLEEP_VS_BASELINE_HIGHER_H,
    SLEEP_VS_BASELINE_HIGHER_SCORE,
    SLEEP_VS_BASELINE_LOWER_EFF,
    SLEEP_VS_BASELINE_LOWER_H,
    SLEEP_VS_BASELINE_LOWER_SCORE,
)
from insights.schemas.processed_context import (
    SleepAnomaly,
    SleepDimensionSignal,
    SleepEvidence,
    SleepOverall,
    SleepSignal,
    SleepSignalsMap,
    SleepSummaryMetrics,
)
from services.executor.constant import HealthDataConstants


# ── helpers ──────────────────────────────────────────────────────────────


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _snapshot_field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _snapshot_date(item: Any) -> Optional[date]:
    raw = _snapshot_field(item, "daily_snapshot_date")
    if not raw:
        return None
    try:
        if isinstance(raw, date) and not isinstance(raw, datetime):
            return raw
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
    except Exception:
        return None


def _parse_clock(raw: Optional[str]) -> Optional[Tuple[int, int]]:
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    try:
        if "T" in text:
            wall = text.split("T", 1)[1]
            for tz_suffix in ("Z", "+00:00", "-00:00"):
                if wall.endswith(tz_suffix):
                    wall = wall[: -len(tz_suffix)]
                    break
            # If still has a non-Z offset (e.g. +07:00), drop it too.
            for sep in ("+", "-"):
                idx = wall.find(sep)
                if idx > 0:
                    wall = wall[:idx]
                    break
            # Trim sub-second precision if any.
            if "." in wall:
                wall = wall.split(".", 1)[0]
            hh, mm = wall[:5].split(":")
            return int(hh), int(mm)
        if len(text) >= 5 and text[2] == ":":
            hh, mm = text[:5].split(":")
            return int(hh), int(mm)
    except Exception:
        return None
    return None


def _format_hhmm(raw: Optional[str]) -> Optional[str]:
    clock = _parse_clock(raw)
    if clock is None:
        return None
    return f"{clock[0]:02d}:{clock[1]:02d}"


def _clock_minutes(hour: int, minute: int) -> int:
    return hour * 60 + minute


def _bedtime_minutes_circular(raw: Optional[str]) -> Optional[int]:
    """Minutes from midnight, shifted so late-night bedtimes stay continuous."""
    clock = _parse_clock(raw)
    if clock is None:
        return None
    mins = _clock_minutes(*clock)
    # Treat early-morning bedtimes (00–06) as previous evening + 24h
    if clock[0] < 6:
        mins += 24 * 60
    return mins


def _minutes_to_hhmm(mins: float) -> str:
    total = int(round(mins)) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def _normalize_to_hours(value: Optional[float], *, min_hours: float = 0.5, max_hours: float = 18.0) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    if min_hours <= v <= max_hours:
        return v
    return v / 60.0


def _hours_between_times(bedtime_raw: Optional[str], wake_raw: Optional[str]) -> Optional[float]:
    if not bedtime_raw or not wake_raw:
        return None
    bed_clock = _parse_clock(bedtime_raw)
    wake_clock = _parse_clock(wake_raw)
    if bed_clock is None or wake_clock is None:
        return None
    bed_mins = bed_clock[0] * 60 + bed_clock[1]
    wake_mins = wake_clock[0] * 60 + wake_clock[1]
    delta = (wake_mins - bed_mins) / 60.0
    if delta < 0:
        delta += 24.0
    if delta <= 0 or delta > 24:
        return None
    return round(delta, 2)


def _per_night_efficiencies(nights: List[SleepNight]) -> List[float]:
    """Per-night efficiency = duration_h / time_in_bed_h, capped at 1.0."""
    out: List[float] = []
    for n in nights:
        if n.duration_h is None or n.duration_h <= 0:
            continue
        tib_h = _hours_between_times(n.actual_bedtime, n.actual_wake_time)
        if tib_h is None or tib_h <= 0:
            continue
        eff = n.duration_h / tib_h
        if eff > 1.0:
            eff = 1.0
        out.append(eff)
    return out


def _mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _stdev(vals: List[float]) -> Optional[float]:
    if len(vals) < 2:
        return None
    return statistics.stdev(vals)


def _is_stale(staleness_days: Optional[int], threshold: int = 0) -> bool:
    if staleness_days is None:
        return True
    return staleness_days > threshold


@dataclass
class SleepNight:
    night_date: date
    duration_h: Optional[float] = None
    awake_h: Optional[float] = None
    rem_min: Optional[int] = None
    core_min: Optional[int] = None
    deep_min: Optional[int] = None
    sleep_score: Optional[int] = None
    health_score: Optional[float] = None
    bedtime: Optional[str] = None
    wake_time: Optional[str] = None
    bedtime_source: Optional[str] = None
    wake_time_source: Optional[str] = None
    actual_bedtime: Optional[str] = None
    actual_wake_time: Optional[str] = None
    estimated: List[str] = field(default_factory=list)


def _dim(
    status: str,
    metrics: Optional[Dict[str, Any]] = None,
    comparison: Optional[Dict[str, Any]] = None,
    trend: Optional[str] = None,
    evidence_metrics: Optional[Dict[str, Any]] = None,
    evidence_comparison: Optional[Dict[str, Any]] = None,
    evidence_context: Optional[Dict[str, Any]] = None,
) -> SleepDimensionSignal:
    metrics = metrics or {}
    comparison = comparison or {}
    return SleepDimensionSignal(
        status=status,
        metrics=metrics,
        comparison=comparison,
        trend=trend,
        evidence=SleepEvidence(
            metrics=evidence_metrics or {},
            comparison=evidence_comparison or {},
            context=evidence_context or {},
        ),
    )


def _vs_baseline_status(
    delta: Optional[float],
    *,
    higher: float,
    lower: float,
) -> Optional[str]:
    if delta is None:
        return None
    if delta >= higher:
        return "higher_than_usual"
    if delta <= lower:
        return "lower_than_usual"
    return "typical"


def _trend_label(latest: float, prior_avg: float) -> Tuple[str, float]:
    delta = round(latest - prior_avg, 2)
    if latest >= prior_avg + SLEEP_TREND_SIGNIFICANT_DIFF_H:
        return "improving", delta
    if latest <= prior_avg - SLEEP_TREND_SIGNIFICANT_DIFF_H:
        return "declining", delta
    return "stable", delta


class SleepSignalProcessor:
    """Transform raw sleep inputs into evidence-rich SleepSignal."""

    @classmethod
    def process(
        cls,
        health_params: Dict[str, Any],
        historical_snapshots: Optional[List[Any]] = None,
        staleness_info: Optional[Dict[str, Any]] = None,
        time_data: Optional[Dict[str, Any]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> SleepSignal:
        try:
            return cls._process_unsafe(
                health_params=health_params or {},
                historical_snapshots=historical_snapshots or [],
                staleness_info=staleness_info or {},
                time_data=time_data or {},
                raw_data=raw_data or {},
            )
        except Exception:
            try:
                from utils.logger import logger as _logger

                _logger.warning("SleepSignalProcessor failed", exc_info=True)
            except Exception:
                pass
            return SleepSignal(
                overall=SleepOverall(status="insufficient_data", confidence="low"),
            )

    @classmethod
    def _process_unsafe(
        cls,
        health_params: Dict[str, Any],
        historical_snapshots: List[Any],
        staleness_info: Dict[str, Any],
        time_data: Dict[str, Any],
        raw_data: Dict[str, Any],
    ) -> SleepSignal:
        today = cls._resolve_today(time_data)
        sleep_staleness = staleness_info.get("sleep_data_staleness_days")
        stale = _is_stale(sleep_staleness)
        last_night_val = health_param_value(
            health_params,
            "health_params_sleep_lastnight",
            HealthDataConstants.KEY_SLEEP_LASTNIGHT,
        )
        last_night_val = _safe_float(last_night_val)
        if stale and last_night_val is not None and last_night_val > 0:
            stale = False

        nights = cls._aggregate_nights(
            health_params=health_params,
            historical_snapshots=historical_snapshots,
            raw_data=raw_data,
            today=today,
            stale=stale,
        )

        goal_h = _safe_float(
            health_param_value(
                health_params,
                "health_params_sleep_goal",
                HealthDataConstants.KEY_SLEEP_GOAL,
            )
        )
        goal_status = cls._goal_status(goal_h)
        bedtime_goal_start = time_data.get("time_data_bedtime_start_str")
        bedtime_goal_end = time_data.get("time_data_bedtime_end_str")

        summary_nights = cls._window_nights(
            nights, today, SLEEP_SUMMARY_WINDOW_DAYS, include_today=True
        )
        baseline_nights = cls._baseline_nights(nights, today)
        baseline_nights_eff = cls._baseline_nights_efficiency(nights, today)

        summary = cls._build_summary(summary_nights)
        if summary.avg_duration_h is None:
            asleep_hp = _safe_float(
                health_param_value(
                    health_params,
                    "health_params_average_total_sleep_hours",
                    HealthDataConstants.KEY_AVERAGE_TOTAL_SLEEP_HOURS,
                )
            )
            if asleep_hp is not None:
                summary.avg_duration_h = _normalize_to_hours(asleep_hp)
        derived = cls._derived_metrics(summary_nights, goal_h, summary)

        latest = summary_nights[-1] if summary_nights else None
        last_night_h = None if stale else (latest.duration_h if latest else None)
        bedtime = None if stale else (latest.bedtime if latest else None)
        wake_time = None if stale else (latest.wake_time if latest else None)
        sleep_score = None if stale else (latest.sleep_score if latest else None)
        bedtime_source = None if stale else (getattr(latest, "bedtime_source", None) if latest else None)
        wake_time_source = None if stale else (getattr(latest, "wake_time_source", None) if latest else None)

        # Prefer prepared quality bucket when fresh
        quality = cls._quality_from_score(sleep_score, stale=stale)
        prepared_q = health_param_value(
            health_params,
            "health_params_sleep_quality",
            HealthDataConstants.KEY_SLEEP_QUALITY,
        )
        if (
            not stale
            and isinstance(prepared_q, str)
            and prepared_q
            and prepared_q != "unknown"
            and sleep_score is not None
        ):
            quality = prepared_q

        bedtime_timing = (
            "unknown"
            if stale
            else cls._bedtime_timing(
                bedtime, goal_start=bedtime_goal_start, goal_end=bedtime_goal_end
            )
        )

        debt_hours = 0.0
        if not stale and goal_h is not None and last_night_h is not None:
            debt_hours = max(0.0, float(goal_h) - float(last_night_h))

        level = cls._level_from_debt(
            debt_hours, has_goal=goal_h is not None, stale=stale
        )

        trend, trend_window, trend_delta = cls._compute_duration_trend(
            nights, today
        )
        eff_trend, eff_trend_window, eff_trend_delta = cls._compute_efficiency_trend(
            nights, today
        )

        consistency = cls._bedtime_consistency(summary_nights)
        consecutive_debt = cls._consecutive_debt_days(
            nights, today, goal_h, last_night_h
        )

        anomalies = cls._detect_anomalies(
            summary_nights,
            goal_h,
            bedtime_goal_start=bedtime_goal_start,
            bedtime_goal_end=bedtime_goal_end,
        )

        # Stage estimates for latest night when missing
        estimated_fields: List[str] = list(latest.estimated) if latest else []
        deep_min = latest.deep_min if latest else None
        rem_min = latest.rem_min if latest else None
        core_min = latest.core_min if latest else None
        ref_h = last_night_h or summary.avg_duration_h
        if deep_min is None:
            deep_min = cls._estimate_stage_min(ref_h, "deep")
            if deep_min is not None:
                estimated_fields.append("deep_sleep_min")
        if rem_min is None:
            rem_min = cls._estimate_stage_min(ref_h, "rem")
            if rem_min is not None:
                estimated_fields.append("rem_sleep_min")
        if core_min is None:
            core_min = cls._estimate_stage_min(ref_h, "light")
            if core_min is not None:
                estimated_fields.append("light_sleep_min")

        avg_asleep = summary.avg_duration_h
        if avg_asleep is None:
            asleep_hp = _safe_float(
                health_param_value(
                    health_params,
                    "health_params_average_total_sleep_hours",
                    HealthDataConstants.KEY_AVERAGE_TOTAL_SLEEP_HOURS,
                )
            )
            avg_asleep = _normalize_to_hours(asleep_hp) if asleep_hp is not None else None

        sleep_hr_min, sleep_hr_max = cls._sleep_hr_range(
            health_params, historical_snapshots, today
        )
        range_from_params = isinstance(
            health_param_value(
                health_params,
                "health_params_sleep_hr_range",
                HealthDataConstants.KEY_SLEEP_HR_RANGE,
            ),
            (tuple, list),
        )
        if sleep_hr_min is not None and not range_from_params:
            estimated_fields.append("sleep_hr_min")
        if sleep_hr_max is not None and not range_from_params:
            estimated_fields.append("sleep_hr_max")

        baseline_cmp = cls._baseline_comparison(
            summary, baseline_nights, baseline_nights_eff=baseline_nights_eff
        )

        signals = cls._build_signals(
            summary=summary,
            derived=derived,
            goal_h=goal_h,
            goal_status=goal_status,
            last_night_h=last_night_h,
            debt_hours=debt_hours,
            quality=quality,
            sleep_score=sleep_score,
            deep_min=deep_min,
            rem_min=rem_min,
            consistency=consistency,
            bedtime_timing=bedtime_timing,
            bedtime=bedtime,
            wake_time=wake_time,
            trend=trend,
            trend_window=trend_window,
            trend_delta=trend_delta,
            eff_trend=eff_trend,
            eff_trend_window=eff_trend_window,
            eff_trend_delta=eff_trend_delta,
            baseline_cmp=baseline_cmp,
            consecutive_debt=consecutive_debt,
            measured_days=summary.measured_days,
            missing_days=summary.missing_days,
            latest=latest,
            summary_nights=summary_nights,
            stale=stale,
            bedtime_goal_start=bedtime_goal_start,
            bedtime_goal_end=bedtime_goal_end,
        )

        overall = cls._build_overall(
            summary=summary,
            derived=derived,
            debt_hours=debt_hours,
            sleep_score=sleep_score,
            goal_h=goal_h,
            measured_days=summary.measured_days,
            stale=stale,
        )

        return SleepSignal(
            summary=summary,
            signals=signals,
            anomalies=anomalies,
            overall=overall,
        )

    # ── aggregation ─────────────────────────────────────────────────────

    @classmethod
    def _resolve_today(cls, time_data: Dict[str, Any]) -> Optional[date]:
        raw = time_data.get("time_data_current_time_iso")
        if not raw:
            return None
        try:
            return date.fromisoformat(str(raw).split("T")[0])
        except Exception:
            return None

    @classmethod
    def _aggregate_nights(
        cls,
        *,
        health_params: Dict[str, Any],
        historical_snapshots: List[Any],
        raw_data: Dict[str, Any],
        today: Optional[date],
        stale: bool,
    ) -> List[SleepNight]:
        by_date: Dict[date, SleepNight] = {}

        cls._merge_raw_sleep_entries(by_date, raw_data)

        if today is not None and not stale:
            last_h = _safe_float(
                health_param_value(
                    health_params,
                    "health_params_sleep_lastnight",
                    HealthDataConstants.KEY_SLEEP_LASTNIGHT,
                )
            )
            if last_h is None:
                # Fallback from most recent snapshot
                if by_date:
                    last_snap = max(by_date.keys())
                    last_h = by_date[last_snap].duration_h
                    if last_snap != today and last_h is not None:
                        # keep snapshot night; don't invent today
                        pass
            sleep_quality_score_val = health_param_value(
                health_params,
                "health_params_sleep_quality_score",
                HealthDataConstants.KEY_SLEEP_QUALITY_SCORE,
            )
            if last_h is not None or sleep_quality_score_val is not None:
                existing = by_date.get(today) or SleepNight(night_date=today)
                # Only fill if raw_data didn't provide a value (single source-of-truth).
                if last_h is not None and existing.duration_h is None:
                    existing.duration_h = last_h
                if sleep_quality_score_val is not None and existing.sleep_score is None:
                    existing.sleep_score = _safe_int(sleep_quality_score_val)
                bedtime_val = health_param_value(
                    health_params,
                    "health_params_first_sleep_time",
                    HealthDataConstants.KEY_FIRST_SLEEP_TIME,
                )
                if bedtime_val is None:
                    bedtime_val = health_param_value(
                        health_params,
                        "health_params_bedtime",
                        "bedtime",
                    )
                if existing.bedtime is None:
                    existing.bedtime = _format_hhmm(bedtime_val)
                    if existing.bedtime:
                        existing.bedtime_source = "health_param"
                # Prefer actual last_wake_time; fall back to health_params_wake_time
                # (which may be profile target if no actual wake).
                wake_val = health_param_value(
                    health_params,
                    "health_params_last_wake_time",
                    HealthDataConstants.KEY_LAST_WAKE_TIME,
                )
                if wake_val is None:
                    wake_val = health_param_value(
                        health_params,
                        "health_params_wake_time",
                        HealthDataConstants.KEY_WAKE_TIME,
                    )
                if existing.wake_time is None:
                    existing.wake_time = _format_hhmm(wake_val)
                    if existing.wake_time:
                        existing.wake_time_source = "health_param"
                if existing.deep_min is None:
                    deep_val = _safe_int(
                        health_param_value(
                            health_params,
                            "health_params_deep_sleep_min",
                            HealthDataConstants.KEY_DEEP_SLEEP,
                        )
                    )
                    if deep_val is not None:
                        existing.deep_min = deep_val
                if existing.rem_min is None:
                    rem_val = _safe_int(
                        health_param_value(
                            health_params,
                            "health_params_rem_sleep_min",
                            HealthDataConstants.KEY_REM_SLEEP,
                        )
                    )
                    if rem_val is not None:
                        existing.rem_min = rem_val
                if existing.core_min is None:
                    core_val = _safe_int(
                        health_param_value(
                            health_params,
                            "health_params_light_sleep_min",
                            HealthDataConstants.KEY_LIGHT_SLEEP,
                        )
                    )
                    if core_val is not None:
                        existing.core_min = core_val
                by_date[today] = existing

        return sorted(by_date.values(), key=lambda n: n.night_date)

    @classmethod
    def _merge_raw_sleep_entries(
        cls, by_date: Dict[date, SleepNight], raw_data: Dict[str, Any]
    ) -> None:
        sleep_block: Any = None
        # 1) Legacy nested layout
        health_data = raw_data.get("health_data") or {}
        if isinstance(health_data, dict):
            sleep_block = health_data.get(HealthDataConstants.SLEEP_TYPE) or health_data.get(
                "SLEEPS"
            )
        if sleep_block is None:
            today_stats = raw_data.get("today_health_stats") or {}
            if isinstance(today_stats, dict):
                sleep_block = today_stats.get("SLEEP") or today_stats.get(
                    HealthDataConstants.SLEEP_TYPE
                )
        if sleep_block is None:
            return
        data = getattr(sleep_block, "data", None)
        if data is None and isinstance(sleep_block, dict):
            data = sleep_block.get("data")
        if not data:
            return

        flat_entries: List[Any] = []
        for week in data:
            days = getattr(week, "data", None)
            if days is None and isinstance(week, dict):
                days = week.get("data")
            if days:
                flat_entries.extend(days)
            else:
                flat_entries.append(week)

        profile = raw_data.get("user_profile") or {}

        for entry in flat_entries:
            date_raw = entry.get("date")
            total = entry.get(CanonicalField.SLEEP_DURATION_FROM_SUMMARY_ONE_NIGHT_HOURS) or entry.get("total")
            deep = entry.get(CanonicalField.DEEP_SLEEP_DURATION_ONE_NIGHT_HOURS) or entry.get("deep")
            rem = entry.get(CanonicalField.REM_SLEEP_DURATION_ONE_NIGHT_HOURS) or entry.get("rem")
            core = entry.get(CanonicalField.CORE_SLEEP_DURATION_ONE_NIGHT_HOURS) or entry.get("core")
            score = entry.get(CanonicalField.SLEEP_SCORE_ONE_NIGHT) or entry.get("sleepScore")
            bedtime = entry.get(CanonicalField.FIRST_SLEEP_TIME_ONE_NIGHT_LOCAL_DATETIME) or entry.get("firstSleepTime")
            wake = entry.get(CanonicalField.LAST_WAKE_TIME_ONE_NIGHT_LOCAL_DATETIME) or entry.get("lastWakeTime")

            if not date_raw:
                continue
            try:
                d = date.fromisoformat(str(date_raw).split("T")[0])
            except Exception:
                continue
            night = by_date.get(d) or SleepNight(night_date=d)
            if total is not None:
                total_f = _safe_float(total)
                if total_f is not None and total_f > 0:
                    night.duration_h = total_f
            if deep is not None:
                deep_f = _safe_float(deep)
                # API deep/rem/core are hours
                if deep_f is not None and deep_f >= 0:
                    night.deep_min = int(deep_f * 60)
            if rem is not None:
                rem_f = _safe_float(rem)
                if rem_f is not None and rem_f >= 0:
                    night.rem_min = int(rem_f * 60)
            if core is not None:
                core_f = _safe_float(core)
                if core_f is not None and core_f >= 0:
                    night.core_min = int(core_f * 60)
            if score is not None:
                score_i = _safe_int(score)
                if score_i is not None and 0 <= score_i <= 100:
                    night.sleep_score = score_i
            # IMPORTANT: prefer actual bedtime from raw entry.
            # Profile bedtime_start is TONIGHT's target and must NOT be used
            # to fill night.bedtime — doing so causes narratives like
            # "LAST_NIGHT_BEDTIME 22:00" while wake_time is 01:34 (3.5h gap)
            # while last_night_h = 8h, producing self-contradicting context.
            if bedtime:
                actual = _format_hhmm(str(bedtime))
                if actual:
                    night.actual_bedtime = actual
                    night.bedtime = actual
                    night.bedtime_source = "raw"
            if wake:
                actual_wake = _format_hhmm(str(wake))
                if actual_wake:
                    night.actual_wake_time = actual_wake
                    night.wake_time = actual_wake
                    night.wake_time_source = "raw"
            by_date[d] = night

    @classmethod
    def _window_nights(
        cls,
        nights: List[SleepNight],
        today: Optional[date],
        window_days: int,
        *,
        include_today: bool,
    ) -> List[SleepNight]:
        if today is None:
            return list(nights[-window_days:])
        cutoff = today - timedelta(days=window_days)
        out: List[SleepNight] = []
        for n in nights:
            if include_today:
                if cutoff < n.night_date <= today:
                    out.append(n)
            elif cutoff <= n.night_date < today:
                out.append(n)
        return out

    @classmethod
    def _baseline_nights(
        cls, nights: List[SleepNight], today: Optional[date]
    ) -> List[SleepNight]:
        """Prior window before the summary week for baseline comparison."""
        if today is None:
            return []
        end = today - timedelta(days=SLEEP_SUMMARY_WINDOW_DAYS)
        start = end - timedelta(days=SLEEP_BASELINE_WINDOW_DAYS)
        return [n for n in nights if start < n.night_date <= end]

    @classmethod
    def _baseline_nights_efficiency(
        cls, nights: List[SleepNight], today: Optional[date]
    ) -> List[SleepNight]:
        if today is None:
            return []
        end = today - timedelta(days=SLEEP_SUMMARY_WINDOW_DAYS)
        start = end - timedelta(days=SLEEP_BASELINE_WINDOW_DAYS_EFFICIENCY)
        return [n for n in nights if start < n.night_date <= end]

    # ── summary & derived ───────────────────────────────────────────────

    @classmethod
    def _build_summary(cls, nights: List[SleepNight]) -> SleepSummaryMetrics:
        durations = [n.duration_h for n in nights if n.duration_h is not None and n.duration_h > 0]
        rems = [float(n.rem_min) for n in nights if n.rem_min is not None and n.rem_min > 0]
        cores = [float(n.core_min) for n in nights if n.core_min is not None and n.core_min > 0]
        deeps = [float(n.deep_min) for n in nights if n.deep_min is not None and n.deep_min > 0]
        scores = [float(n.sleep_score) for n in nights if n.sleep_score is not None and n.sleep_score > 0]
        bed_mins = []
        wake_mins = []
        for n in nights:
            if n.actual_bedtime:
                m = _bedtime_minutes_circular(n.actual_bedtime)
                if m is not None:
                    bed_mins.append(m)
            if n.actual_wake_time:
                c = _parse_clock(n.actual_wake_time)
                if c is not None:
                    wake_mins.append(_clock_minutes(*c))

        measured = len(durations)
        window = SLEEP_SUMMARY_WINDOW_DAYS
        missing = max(0, window - measured)
        coverage = round(100.0 * measured / window, 1) if window else None

        return SleepSummaryMetrics(
            avg_duration_h=round(_mean(durations), 2) if durations else None,
            avg_rem_min=round(_mean(rems), 1) if rems else None,
            avg_core_min=round(_mean(cores), 1) if cores else None,
            avg_deep_min=round(_mean(deeps), 1) if deeps else None,
            avg_sleep_score=round(_mean(scores), 1) if scores else None,
            avg_efficiency=(
                round(_mean([e for e in _per_night_efficiencies(nights)]), 3)
                if any(n.duration_h for n in nights)
                else None
            ),
            avg_bedtime=_minutes_to_hhmm(_mean(bed_mins)) if bed_mins else None,
            avg_wake_time=_minutes_to_hhmm(_mean(wake_mins)) if wake_mins else None,
            measured_days=measured,
            missing_days=missing,
            window_days=window,
            coverage_pct=coverage,
        )

    @classmethod
    def _derived_metrics(
        cls,
        nights: List[SleepNight],
        goal_h: Optional[float],
        summary: SleepSummaryMetrics,
    ) -> Dict[str, Any]:
        per_night_efficiencies = _per_night_efficiencies(nights)
        efficiency = (
            round(sum(per_night_efficiencies) / len(per_night_efficiencies), 3)
            if per_night_efficiencies
            else None
        )

        deep_share = None
        rem_share = None
        avg_asleep = summary.avg_duration_h
        if avg_asleep and avg_asleep > 0:
            if summary.avg_deep_min is not None:
                deep_share = round((summary.avg_deep_min / 60.0) / avg_asleep, 3)
            if summary.avg_rem_min is not None:
                rem_share = round((summary.avg_rem_min / 60.0) / avg_asleep, 3)

        durations = [n.duration_h for n in nights if n.duration_h is not None and n.duration_h > 0]
        duration_var = round(_stdev(durations), 2) if durations and len(durations) >= 2 else None

        bed_mins = []
        wake_mins_list = []
        for n in nights:
            if n.actual_bedtime:
                m = _bedtime_minutes_circular(n.actual_bedtime)
                if m is not None:
                    bed_mins.append(m)
            if n.actual_wake_time:
                c = _parse_clock(n.actual_wake_time)
                if c is not None:
                    wake_mins_list.append(_clock_minutes(*c))
        bedtime_var = (
            round(_stdev(bed_mins), 1) if bed_mins and len(bed_mins) >= 2 else None
        )
        wake_var = (
            round(_stdev(wake_mins_list), 1)
            if wake_mins_list and len(wake_mins_list) >= 2
            else None
        )

        # Cumulative sleep debt over window vs goal
        sleep_debt = None
        if goal_h is not None and durations:
            sleep_debt = round(
                sum(max(0.0, goal_h - h) for h in durations), 2
            )

        consistency_score = None
        if bedtime_var is not None:
            # Map std-dev minutes → 0–100 (30min → ~100, 90min → ~0)
            consistency_score = round(
                max(0.0, min(100.0, 100.0 * (1 - bedtime_var / 90.0))), 1
            )

        return {
            "efficiency": efficiency,
            "deep_share": deep_share,
            "rem_share": rem_share,
            "duration_variance_h": duration_var,
            "bedtime_variance_min": bedtime_var,
            "wake_variance_min": wake_var,
            "sleep_debt_h": sleep_debt,
            "consistency_score": consistency_score,
            "coverage_pct": summary.coverage_pct,
        }

    # ── trend / baseline / consistency / debt ───────────────────────────

    @classmethod
    def _compute_duration_trend(
        cls,
        nights: List[SleepNight],
        today: Optional[date],
    ) -> Tuple[str, str, Optional[float]]:
        window_label = f"{SLEEP_SUMMARY_WINDOW_DAYS}d"
        wn = cls._window_nights(
            nights, today, SLEEP_SUMMARY_WINDOW_DAYS, include_today=True
        )
        series = [n.duration_h for n in wn if n.duration_h is not None]
        if len(series) < SLEEP_MIN_DAYS_FOR_TREND:
            return "insufficient_data", window_label, None
        latest = series[-1]
        prior = series[:-1]
        if not prior:
            return "insufficient_data", window_label, None
        prior_avg = sum(prior) / len(prior)
        label, delta = _trend_label(latest, prior_avg)
        return label, window_label, delta

    @classmethod
    def _compute_efficiency_trend(
        cls,
        nights: List[SleepNight],
        today: Optional[date],
    ) -> Tuple[str, str, Optional[float]]:
        """Compare latest night efficiency to the prior nights in the summary window."""
        window_label = f"{SLEEP_SUMMARY_WINDOW_DAYS}d"
        wn = cls._window_nights(
            nights, today, SLEEP_SUMMARY_WINDOW_DAYS, include_today=True
        )
        series = _per_night_efficiencies(wn)
        if len(series) < SLEEP_MIN_DAYS_FOR_TREND:
            return "insufficient_data", window_label, None
        latest = series[-1]
        prior = series[:-1]
        if not prior:
            return "insufficient_data", window_label, None
        prior_avg = sum(prior) / len(prior)
        delta = round(latest - prior_avg, 3)
        if latest >= prior_avg + SLEEP_TREND_SIGNIFICANT_DIFF_EFF:
            return "improving", window_label, delta
        if latest <= prior_avg - SLEEP_TREND_SIGNIFICANT_DIFF_EFF:
            return "declining", window_label, delta
        return "stable", window_label, delta

    @classmethod
    def _baseline_comparison(
        cls,
        summary: SleepSummaryMetrics,
        baseline_nights: List[SleepNight],
        baseline_nights_eff: Optional[List[SleepNight]] = None,
    ) -> Dict[str, Any]:
        base_dur = [
            n.duration_h for n in baseline_nights if n.duration_h is not None
        ]
        base_deep = [
            float(n.deep_min) for n in baseline_nights if n.deep_min is not None
        ]
        base_rem = [
            float(n.rem_min) for n in baseline_nights if n.rem_min is not None
        ]
        base_score = [
            float(n.sleep_score) for n in baseline_nights if n.sleep_score is not None
        ]
        if len(base_dur) < SLEEP_MIN_DAYS_FOR_BASELINE and not base_score:
            return {}

        out: Dict[str, Any] = {"baseline_days": len(base_dur) or len(base_score)}
        if base_dur and summary.avg_duration_h is not None:
            b = _mean(base_dur)
            delta = round(summary.avg_duration_h - b, 2)
            out["duration"] = {
                "baseline_h": round(b, 2),
                "delta_h": delta,
                "status": _vs_baseline_status(
                    delta,
                    higher=SLEEP_VS_BASELINE_HIGHER_H,
                    lower=SLEEP_VS_BASELINE_LOWER_H,
                ),
            }
        if base_deep and summary.avg_deep_min is not None:
            b = _mean(base_deep)
            delta = round(summary.avg_deep_min - b, 1)
            out["deep"] = {
                "baseline_min": round(b, 1),
                "delta_min": delta,
                "status": _vs_baseline_status(
                    delta / 60.0 if delta is not None else None,
                    higher=SLEEP_VS_BASELINE_HIGHER_H,
                    lower=SLEEP_VS_BASELINE_LOWER_H,
                ),
            }
        if base_rem and summary.avg_rem_min is not None:
            b = _mean(base_rem)
            delta = round(summary.avg_rem_min - b, 1)
            out["rem"] = {
                "baseline_min": round(b, 1),
                "delta_min": delta,
                "status": _vs_baseline_status(
                    delta / 60.0 if delta is not None else None,
                    higher=SLEEP_VS_BASELINE_HIGHER_H,
                    lower=SLEEP_VS_BASELINE_LOWER_H,
                ),
            }
        if base_score and summary.avg_sleep_score is not None:
            b = _mean(base_score)
            delta = round(summary.avg_sleep_score - b, 1)
            out["score"] = {
                "baseline": round(b, 1),
                "delta": delta,
                "status": _vs_baseline_status(
                    delta,
                    higher=SLEEP_VS_BASELINE_HIGHER_SCORE,
                    lower=SLEEP_VS_BASELINE_LOWER_SCORE,
                ),
            }
        if (
            summary.avg_efficiency is not None
            and baseline_nights_eff
        ):
            base_eff = _per_night_efficiencies(baseline_nights_eff)
            if len(base_eff) >= SLEEP_MIN_DAYS_FOR_BASELINE:
                b = _mean(base_eff)
                delta = round(summary.avg_efficiency - b, 3)
                out["efficiency"] = {
                    "baseline": round(b, 3),
                    "delta": delta,
                    "status": _vs_baseline_status(
                        delta,
                        higher=SLEEP_VS_BASELINE_HIGHER_EFF,
                        lower=SLEEP_VS_BASELINE_LOWER_EFF,
                    ),
                }
        return out

    @classmethod
    def _bedtime_consistency(cls, nights: List[SleepNight]) -> Optional[str]:
        mins = []
        for n in nights:
            if n.actual_bedtime:
                m = _bedtime_minutes_circular(n.actual_bedtime)
                if m is not None:
                    mins.append(m)
        if len(mins) < SLEEP_MIN_DAYS_FOR_CONSISTENCY:
            return None
        sd = _stdev(mins)
        if sd is None:
            return None
        if sd <= SLEEP_BEDTIME_CONSISTENT_STD_MIN:
            return "consistent"
        increasing = sum(1 for i in range(1, len(mins)) if mins[i] > mins[i - 1])
        if (
            increasing >= len(mins) - 2
            and mins[-1] - mins[0] >= SLEEP_BEDTIME_LATE_CREEP_MIN
        ):
            return "late_creep"
        return "shifting"

    @classmethod
    def _consecutive_debt_days(
        cls,
        nights: List[SleepNight],
        today: Optional[date],
        goal_h: Optional[float],
        last_night_h: Optional[float],
        max_lookback: int = 14,
    ) -> int:
        if goal_h is None or goal_h <= 0 or today is None:
            return 0
        if last_night_h is None or last_night_h >= goal_h:
            return 0
        by_date = {
            n.night_date: n.duration_h
            for n in nights
            if n.duration_h is not None
        }
        streak = 1
        cursor = today - timedelta(days=1)
        for _ in range(max_lookback):
            hours = by_date.get(cursor)
            if hours is None:
                break
            if hours < goal_h:
                streak += 1
                cursor -= timedelta(days=1)
            else:
                break
        return streak

    # ── anomalies ───────────────────────────────────────────────────────

    @classmethod
    def _detect_anomalies(
        cls,
        nights: List[SleepNight],
        goal_h: Optional[float],
        bedtime_goal_start: Optional[str] = None,
        bedtime_goal_end: Optional[str] = None,
    ) -> List[SleepAnomaly]:
        anomalies: List[SleepAnomaly] = []
        late_threshold = SLEEP_ANOMALY_LATE_BEDTIME_THRESHOLD_MIN
        early_threshold = SLEEP_ANOMALY_EARLY_WAKE_THRESHOLD_MIN

        def _fwd_minutes(a_mins: int, b_mins: int) -> int:
            return (b_mins - a_mins) % (24 * 60)

        end_clock = _parse_clock(bedtime_goal_end)
        end_mins = _clock_minutes(*end_clock) if end_clock else None

        for n in nights:
            d = n.night_date.isoformat()
            if n.duration_h is not None:
                if n.duration_h < SLEEP_ANOMALY_SHORT_H:
                    anomalies.append(
                        SleepAnomaly(
                            kind="short_sleep",
                            night_date=d,
                            value=n.duration_h,
                            threshold=SLEEP_ANOMALY_SHORT_H,
                            evidence={"duration_h": n.duration_h, "goal_h": goal_h},
                        )
                    )
                elif n.duration_h > SLEEP_ANOMALY_LONG_H:
                    anomalies.append(
                        SleepAnomaly(
                            kind="long_sleep",
                            night_date=d,
                            value=n.duration_h,
                            threshold=SLEEP_ANOMALY_LONG_H,
                            evidence={"duration_h": n.duration_h},
                        )
                    )
            timing = cls._bedtime_timing(
                n.bedtime,
                goal_start=bedtime_goal_start,
                goal_end=bedtime_goal_end,
            )
            if timing == "late":
                late_value: Optional[float] = None
                evidence = {"bedtime": n.bedtime}
                bt_clock = _parse_clock(n.bedtime)
                if bt_clock and end_mins is not None:
                    bt_mins = _clock_minutes(*bt_clock)
                    late_value = float(_fwd_minutes(end_mins, bt_mins))
                anomalies.append(
                    SleepAnomaly(
                        kind="late_bedtime",
                        night_date=d,
                        value=late_value,
                        threshold=float(late_threshold),
                        evidence=evidence,
                    )
                )
            if n.wake_time:
                wake = _parse_clock(n.wake_time)
                if wake and wake[0] < 5:
                    early_value: Optional[float] = None
                    early_clock = (5, 0)
                    if wake:
                        wake_mins = _clock_minutes(*wake)
                        early_mins = _clock_minutes(*early_clock)
                        early_value = float(_fwd_minutes(wake_mins, early_mins))
                    anomalies.append(
                        SleepAnomaly(
                            kind="early_wake",
                            night_date=d,
                            value=early_value,
                            threshold=float(early_threshold),
                            evidence={
                                "wake_time": n.wake_time,
                                "early_baseline": "05:00",
                            },
                        )
                    )
            if n.duration_h and n.duration_h > 0:
                if n.deep_min is not None:
                    share = (n.deep_min / 60.0) / n.duration_h
                    if share < SLEEP_ANOMALY_DEEP_SHARE_LOW:
                        anomalies.append(
                            SleepAnomaly(
                                kind="low_deep_sleep",
                                night_date=d,
                                value=round(share, 3),
                                threshold=SLEEP_ANOMALY_DEEP_SHARE_LOW,
                                evidence={
                                    "deep_min": n.deep_min,
                                    "duration_h": n.duration_h,
                                },
                            )
                        )
                if n.rem_min is not None:
                    share = (n.rem_min / 60.0) / n.duration_h
                    if share < SLEEP_ANOMALY_REM_SHARE_LOW:
                        anomalies.append(
                            SleepAnomaly(
                                kind="low_rem",
                                night_date=d,
                                value=round(share, 3),
                                threshold=SLEEP_ANOMALY_REM_SHARE_LOW,
                                evidence={
                                    "rem_min": n.rem_min,
                                    "duration_h": n.duration_h,
                                },
                            )
                        )
                if n.awake_h is not None and n.duration_h is not None:
                    total = n.duration_h + n.awake_h
                    if total > 0:
                        eff = n.duration_h / total
                        if eff < SLEEP_ANOMALY_LOW_EFFICIENCY:
                            anomalies.append(
                                SleepAnomaly(
                                    kind="poor_sleep_efficiency",
                                    night_date=d,
                                    value=round(eff, 3),
                                    threshold=SLEEP_ANOMALY_LOW_EFFICIENCY,
                                    evidence={
                                        "duration_h": n.duration_h,
                                        "awake_h": n.awake_h,
                                    },
                                )
                            )
        return anomalies

    # ── signals ─────────────────────────────────────────────────────────

    @classmethod
    def _build_signals(cls, **kw: Any) -> SleepSignalsMap:
        summary: SleepSummaryMetrics = kw["summary"]
        derived: Dict[str, Any] = kw["derived"]
        goal_h = kw["goal_h"]
        last_night_h = kw["last_night_h"]
        debt_hours = kw["debt_hours"]
        quality = kw["quality"]
        sleep_score = kw["sleep_score"]
        deep_min = kw["deep_min"]
        rem_min = kw["rem_min"]
        consistency = kw["consistency"]
        bedtime_timing = kw["bedtime_timing"]
        bedtime = kw["bedtime"]
        wake_time = kw["wake_time"]
        trend = kw["trend"]
        trend_window = kw["trend_window"]
        trend_delta = kw["trend_delta"]
        eff_trend = kw.get("eff_trend")
        eff_trend_window = kw.get("eff_trend_window")
        eff_trend_delta = kw.get("eff_trend_delta")
        baseline_cmp: Dict[str, Any] = kw["baseline_cmp"]
        consecutive_debt = kw["consecutive_debt"]
        measured_days = kw["measured_days"]
        missing_days = kw["missing_days"]
        latest: Optional[SleepNight] = kw["latest"]
        summary_nights: List[SleepNight] = kw["summary_nights"]
        stale = kw["stale"]
        bedtime_goal_start = kw.get("bedtime_goal_start")
        bedtime_goal_end = kw.get("bedtime_goal_end")

        ctx_common = {
            "latest_date": latest.night_date.isoformat() if latest else None,
        }

        # Duration
        avg_h = summary.avg_duration_h
        target = goal_h or SLEEP_GOAL_RECOMMENDED_H
        if stale or avg_h is None:
            dur_status = "insufficient_data"
        elif avg_h >= target:
            dur_status = "adequate"
        elif avg_h >= target - SLEEP_DEBT_MILD_H:
            dur_status = "slightly_low"
        elif avg_h >= target - SLEEP_DEBT_MODERATE_H:
            dur_status = "low"
        else:
            dur_status = "very_low"
            
        vs_goal = (
            round(avg_h - goal_h, 2) if avg_h is not None and goal_h else None
        )
        vs_rec_floor = (
            round(avg_h - SLEEP_RECOMMENDED_FLOOR_H, 2)
            if avg_h is not None
            else None
        )
        rec_met = (
            bool(avg_h is not None and avg_h >= SLEEP_RECOMMENDED_FLOOR_H)
            if avg_h is not None
            else None
        )
        dur_cmp: Dict[str, Any] = {
            "vs_goal": vs_goal,
            "vs_recommended_floor": vs_rec_floor,
            "recommended_floor_h": SLEEP_RECOMMENDED_FLOOR_H,
            "recommended_met": rec_met,
        }
        if baseline_cmp.get("duration"):
            dur_cmp["vs_baseline"] = baseline_cmp["duration"]

        sleep_duration = SleepDimensionSignal(
            status=dur_status,
            metrics={
                "average_hours": avg_h,
                "target_hours": target,
                "last_night_hours": last_night_h,
            },
            comparison=dur_cmp,
            trend=trend,
            evidence=SleepEvidence(
                context={
                    "latest_date": latest.night_date.isoformat() if latest else None,
                },
            ),
        )

        # Efficiency
        eff = derived.get("efficiency")
        if eff is None:
            eff_status = "insufficient_data"
        elif eff >= SLEEP_EFFICIENCY_GOOD:
            eff_status = "good"
        elif eff >= SLEEP_EFFICIENCY_FAIR:
            eff_status = "fair"
        else:
            eff_status = "poor"
        eff_cmp: Dict[str, Any] = {}
        if baseline_cmp.get("efficiency"):
            eff_cmp["vs_baseline"] = baseline_cmp["efficiency"]
        eff_metrics: Dict[str, Any] = {"efficiency": eff}
        if eff_trend_window is not None:
            eff_metrics["trend_window"] = eff_trend_window
        if eff_trend_delta is not None:
            eff_metrics["trend_delta"] = eff_trend_delta
        sleep_efficiency = _dim(
            eff_status,
            metrics=eff_metrics,
            comparison=eff_cmp,
            trend=eff_trend,
            evidence_context=ctx_common,
        )

        # Consistency
        cons_score = derived.get("consistency_score")
        if consistency is None and cons_score is None:
            cons_status = "insufficient_data"
        elif consistency == "consistent" or (
            cons_score is not None and cons_score >= 70
        ):
            cons_status = "consistent"
        elif consistency == "late_creep":
            cons_status = "late_creep"
        else:
            cons_status = consistency or "shifting"
        sleep_consistency = _dim(
            cons_status,
            metrics={
                "consistency_score": cons_score,
                "bedtime_variance_min": derived.get("bedtime_variance_min"),
                "wake_variance_min": derived.get("wake_variance_min"),
                "duration_variance_h": derived.get("duration_variance_h"),
            },
        )

        # Deep
        deep_share = derived.get("deep_share")
        if deep_share is None and deep_min is None:
            deep_status = "insufficient_data"
        elif deep_share is not None:
            if deep_share >= SLEEP_DEEP_SHARE_OK:
                deep_status = "good"
            elif deep_share >= SLEEP_DEEP_SHARE_LOW:
                deep_status = "fair"
            else:
                deep_status = "low"
        else:
            deep_status = "insufficient_data"
        deep_cmp = {}
        if baseline_cmp.get("deep"):
            deep_cmp["vs_baseline"] = baseline_cmp["deep"]
        deep_sleep = _dim(
            deep_status,
            metrics={
                "avg_deep_min": summary.avg_deep_min,
                "latest_deep_min": deep_min,
                "deep_share": deep_share,
            },
            comparison=deep_cmp,
            evidence_context=ctx_common,
        )

        # REM
        rem_share = derived.get("rem_share")
        if rem_share is None and rem_min is None:
            rem_status = "insufficient_data"
        elif rem_share is not None:
            if rem_share >= SLEEP_REM_SHARE_OK:
                rem_status = "good"
            elif rem_share >= SLEEP_REM_SHARE_LOW:
                rem_status = "fair"
            else:
                rem_status = "low"
        else:
            rem_status = "insufficient_data"
        rem_cmp = {}
        if baseline_cmp.get("rem"):
            rem_cmp["vs_baseline"] = baseline_cmp["rem"]
        rem_sleep = _dim(
            rem_status,
            metrics={
                "avg_rem_min": summary.avg_rem_min,
                "latest_rem_min": rem_min,
                "rem_share": rem_share,
            },
            comparison=rem_cmp,
            evidence_context=ctx_common,
        )

        # Debt
        if stale or goal_h is None:
            debt_status = "insufficient_data"
        elif debt_hours <= SLEEP_DEBT_OK_H:
            debt_status = "ok"
        elif debt_hours <= SLEEP_DEBT_MILD_H:
            debt_status = "mild"
        elif debt_hours <= SLEEP_DEBT_MODERATE_H:
            debt_status = "moderate"
        else:
            debt_status = "severe"
        sleep_debt = _dim(
            debt_status,
            metrics={
                "last_night_debt_h": round(debt_hours, 1),
                "window_debt_h": derived.get("sleep_debt_h"),
                "consecutive_debt_days": consecutive_debt,
                "goal_h": goal_h,
            },
            evidence_context=ctx_common,
        )

        # Quality / recovery-ish score signal
        score_cmp = {}
        if baseline_cmp.get("score"):
            score_cmp["vs_baseline"] = baseline_cmp["score"]
        sleep_quality = _dim(
            quality if quality else "no_data",
            metrics={
                "sleep_score": sleep_score,
                "avg_sleep_score": summary.avg_sleep_score,
            },
            comparison=score_cmp,
            trend=trend,
            evidence_context=ctx_common,
        )

        # Bedtime timing
        bedtime_sig = _dim(
            bedtime_timing or "unknown",
            metrics={
                "bedtime": bedtime,
                "wake_time": wake_time,
                "avg_bedtime": summary.avg_bedtime,
                "avg_wake_time": summary.avg_wake_time,
                "goal_start": bedtime_goal_start,
                "goal_end": bedtime_goal_end,
            },
            evidence_context=ctx_common,
        )

        # Trend signal
        sleep_trend = _dim(
            trend,
            metrics={
                "window": trend_window,
                "delta_h": trend_delta,
            },
            evidence_context=ctx_common,
        )

        # Recovery: combine score + debt + efficiency
        if stale or (sleep_score is None and avg_h is None):
            recovery_status = "insufficient_data"
        elif debt_status in ("ok", "insufficient_data") and (
            quality in ("very_high", "high", "ok") or (eff or 0) >= SLEEP_EFFICIENCY_GOOD
        ):
            recovery_status = "recovered"
        elif debt_status in ("mild",) or quality in ("ok", "low"):
            recovery_status = "partial"
        else:
            recovery_status = "under_recovered"
        recovery = _dim(
            recovery_status,
            metrics={
                "debt_h": round(debt_hours, 1),
                "quality": quality,
                "components": {
                    "sleep_score": sleep_score,
                    "efficiency": eff,
                    "avg_h": avg_h,
                },
            },
            evidence_context=ctx_common,
        )

        return SleepSignalsMap(
            sleep_duration=sleep_duration,
            sleep_efficiency=sleep_efficiency,
            sleep_consistency=sleep_consistency,
            deep_sleep=deep_sleep,
            rem_sleep=rem_sleep,
            sleep_debt=sleep_debt,
            sleep_quality=sleep_quality,
            bedtime_timing=bedtime_sig,
            sleep_trend=sleep_trend,
            recovery=recovery,
        )

    @classmethod
    def _build_overall(
        cls,
        *,
        summary: SleepSummaryMetrics,
        derived: Dict[str, Any],
        debt_hours: float,
        sleep_score: Optional[int],
        goal_h: Optional[float],
        measured_days: int,
        stale: bool,
    ) -> SleepOverall:
        if stale or measured_days == 0:
            return SleepOverall(status="insufficient_data", confidence="low")

        target = goal_h or SLEEP_GOAL_RECOMMENDED_H
        avg = summary.avg_duration_h
        duration_pts = 50.0
        if avg is not None and target > 0:
            duration_pts = max(0.0, min(100.0, 100.0 * (avg / target)))

        eff = derived.get("efficiency")
        eff_pts = 70.0 if eff is None else max(0.0, min(100.0, eff * 100.0))

        score_pts = float(sleep_score) if sleep_score is not None else (
            float(summary.avg_sleep_score) if summary.avg_sleep_score is not None else 70.0
        )

        if debt_hours <= SLEEP_DEBT_OK_H:
            debt_pts = 100.0
        elif debt_hours <= SLEEP_DEBT_MILD_H:
            debt_pts = 75.0
        elif debt_hours <= SLEEP_DEBT_MODERATE_H:
            debt_pts = 50.0
        else:
            debt_pts = 25.0

        score = round(
            duration_pts * SLEEP_OVERALL_WEIGHT_DURATION
            + eff_pts * SLEEP_OVERALL_WEIGHT_EFFICIENCY
            + score_pts * SLEEP_OVERALL_WEIGHT_SCORE
            + debt_pts * SLEEP_OVERALL_WEIGHT_DEBT,
            1,
        )
        if score >= 75:
            status = "good"
        elif score >= 55:
            status = "fair"
        else:
            status = "poor"

        if measured_days >= 5:
            confidence = "high"
        elif measured_days >= 3:
            confidence = "medium"
        else:
            confidence = "low"

        return SleepOverall(status=status, score=score, confidence=confidence)

    # ── small classifiers ───────────────────────────────────────────────

    @classmethod
    def _quality_from_score(cls, score: Optional[int], *, stale: bool) -> str:
        if stale:
            return "no_fresh_data"
        if score is None:
            return "no_data"
        s = int(score)
        if s >= SLEEP_SCORE_VERY_HIGH:
            return "very_high"
        if s >= SLEEP_SCORE_HIGH:
            return "high"
        if s >= SLEEP_SCORE_OK:
            return "ok"
        if s >= SLEEP_SCORE_LOW:
            return "low"
        return "very_low"

    @classmethod
    def _bedtime_timing(
        cls,
        bedtime: Optional[str],
        *,
        goal_start: Optional[str],
        goal_end: Optional[str],
    ) -> Optional[str]:
        bt = _parse_clock(bedtime)
        if bt is None:
            return "unknown"
        bt_mins = _clock_minutes(*bt)
        start = _parse_clock(goal_start)
        end = _parse_clock(goal_end)
        if start is not None and end is not None:
            start_m = _clock_minutes(*start)
            end_m = _clock_minutes(*end)

            def _in_window(val: int, lo: int, hi: int) -> bool:
                if lo <= hi:
                    return lo <= val <= hi
                return val >= lo or val <= hi

            if _in_window(bt_mins, start_m, end_m):
                return "on_time"

            def _fwd(a: int, b: int) -> int:
                return (b - a) % (24 * 60)

            if _fwd(bt_mins, start_m) <= _fwd(end_m, bt_mins):
                return "early"
            return "late"

        hour = bt[0]
        if hour >= SLEEP_DEFAULT_BEDTIME_LATE_HOUR:
            return "late"
        if hour < SLEEP_DEFAULT_BEDTIME_EARLY_HOUR:
            return "early"
        return "on_time"

    @classmethod
    def _goal_status(cls, goal_h: Optional[float]) -> str:
        if goal_h is None:
            return "missing"
        if goal_h < SLEEP_GOAL_MIN_H:
            return "too_low"
        if goal_h > SLEEP_GOAL_MAX_H:
            return "too_high"
        return "ok"

    @classmethod
    def _level_from_debt(
        cls, debt_hours: float, *, has_goal: bool, stale: bool
    ) -> str:
        if stale or not has_goal:
            return "ok"
        if debt_hours <= SLEEP_DEBT_OK_H:
            return "ok"
        if debt_hours <= SLEEP_DEBT_MILD_H:
            return "mild"
        if debt_hours <= SLEEP_DEBT_MODERATE_H:
            return "moderate"
        return "severe"

    @classmethod
    def _estimate_stage_min(
        cls, total_hours: Optional[float], stage: str
    ) -> Optional[int]:
        if total_hours is None or total_hours <= 0:
            return None
        ratios = {
            "deep": SLEEP_ESTIMATE_DEEP_RATIO,
            "rem": SLEEP_ESTIMATE_REM_RATIO,
            "light": SLEEP_ESTIMATE_LIGHT_RATIO,
        }
        pct = ratios.get(stage)
        if pct is None:
            return None
        return int(round(total_hours * pct * 60))

    @classmethod
    def _sleep_hr_range(
        cls,
        health_params: Dict[str, Any],
        snapshots: List[Any],
        today: Optional[date],
    ) -> Tuple[Optional[int], Optional[int]]:
        range_value = health_param_value(
            health_params,
            "health_params_sleep_hr_range",
            HealthDataConstants.KEY_SLEEP_HR_RANGE,
        )
        lo = hi = None
        if isinstance(range_value, (tuple, list)) and len(range_value) == 2:
            lo = _safe_int(range_value[0])
            hi = _safe_int(range_value[1])
        if lo is not None and hi is not None:
            return lo, hi
        if today is None:
            return lo, hi
        # Fallback from resting HR snapshots
        cutoff = today - timedelta(days=SLEEP_SUMMARY_WINDOW_DAYS)
        vals = []
        for row in snapshots or []:
            d = _snapshot_date(row)
            if d is None or not (cutoff <= d < today):
                continue
            hr = _snapshot_field(row, CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM)
            if hr is not None:
                try:
                    vals.append(int(hr))
                except (TypeError, ValueError):
                    pass
        if not vals:
            return lo, hi
        return min(vals), max(vals) + 10
