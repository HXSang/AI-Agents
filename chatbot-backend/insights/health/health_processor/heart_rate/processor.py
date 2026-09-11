"""Heart-rate signal processor — single source of truth for HR calculations.

Pipeline: aggregation → derived → recovery → trend → baseline → stability
→ anomalies → evidence-rich signals. The LLM only interprets.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from insights.health.canonical_field_mapping import CanonicalField
from insights.health.health_processor.common.utils import health_param_value
from insights.health.health_processor.heart_rate.constants import (
    HEART_RATE_RESTING_ELEVATED_DELTA_BPM,
    HEART_RATE_RESTING_HIGH_BPM,
    HR_ANOMALY_HIGH_LATEST_BPM,
    HR_ANOMALY_HIGH_RHR_BPM,
    HR_ANOMALY_LOW_HRV,
    HR_ANOMALY_LOW_LATEST_BPM,
    HR_ANOMALY_LOW_RHR_BPM,
    HR_ANOMALY_SUDDEN_RHR_JUMP_BPM,
    HR_BASELINE_WINDOW_DAYS,
    HR_HRV_EXCELLENT,
    HR_HRV_FAIR,
    HR_HRV_GOOD,
    HR_HRV_LOW,
    HR_MIN_DAYS_FOR_BASELINE,
    HR_MIN_DAYS_FOR_STABILITY,
    HR_MIN_DAYS_FOR_TREND,
    HR_OVERALL_WEIGHT_BASELINE,
    HR_OVERALL_WEIGHT_RECOVERY,
    HR_OVERALL_WEIGHT_STABILITY,
    HR_OVERALL_WEIGHT_TREND,
    HR_RECOVERY_WEIGHT_HRV,
    HR_RECOVERY_WEIGHT_RHR,
    HR_RHR_ELEVATED_DELTA_BPM,
    HR_RHR_LOW_DELTA_BPM,
    HR_RHR_NORMAL_DELTA_BPM,
    HR_STABILITY_HIGH_STD,
    HR_STABILITY_MED_STD,
    HR_SUMMARY_WINDOW_DAYS,
    HR_TREND_SIGNIFICANT_HRV,
    HR_TREND_SIGNIFICANT_RHR_BPM,
    HR_TREND_WINDOWS_DAYS,
)
from insights.schemas.processed_context import (
    HeartRateAnomaly,
    HeartRateDimensionSignal,
    HeartRateEvidence,
    HeartRateOverall,
    HeartRateSignal,
    HeartRateSignalsMap,
    HeartRateSummaryMetrics,
)
from services.executor.constant import HealthDataConstants


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
        return int(round(float(v)))
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


def _mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _stdev(vals: List[float]) -> Optional[float]:
    if len(vals) < 2:
        return None
    return statistics.pstdev(vals)


def _is_stale(staleness_days: Optional[int], threshold: int = 0) -> bool:
    if staleness_days is None:
        return True
    return staleness_days > threshold


def _dim(
    status: str,
    metrics: Optional[Dict[str, Any]] = None,
    comparison: Optional[Dict[str, Any]] = None,
    trend: Optional[str] = None,
    evidence_context: Optional[Dict[str, Any]] = None,
) -> HeartRateDimensionSignal:
    metrics = metrics or {}
    comparison = comparison or {}
    return HeartRateDimensionSignal(
        status=status,
        metrics=metrics,
        comparison=comparison,
        trend=trend,
        evidence=HeartRateEvidence(
            metrics=dict(metrics),
            comparison=dict(comparison),
            context=evidence_context or {},
        ),
    )


@dataclass
class HeartRateDay:
    day_date: date
    resting: Optional[float] = None
    latest: Optional[float] = None
    hrv: Optional[float] = None
    health_score: Optional[float] = None
    walking: Optional[float] = None
    workout_avg: Optional[float] = None
    sleep_hr: Optional[float] = None


class HeartRateSignalProcessor:
    """Transform raw HR inputs into evidence-rich HeartRateSignal."""

    @classmethod
    def process(
        cls,
        health_params: Dict[str, Any],
        historical_snapshots: Optional[List[Any]] = None,
        staleness_info: Optional[Dict[str, Any]] = None,
        time_data: Optional[Dict[str, Any]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> HeartRateSignal:
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

                _logger.warning("HeartRateSignalProcessor failed", exc_info=True)
            except Exception:
                pass
            return HeartRateSignal(
                overall=HeartRateOverall(
                    status="insufficient_data", confidence="low"
                ),
                stress_high=False,
            )

    @classmethod
    def _process_unsafe(
        cls,
        health_params: Dict[str, Any],
        historical_snapshots: List[Any],
        staleness_info: Dict[str, Any],
        time_data: Dict[str, Any],
        raw_data: Dict[str, Any],
    ) -> HeartRateSignal:
        today = cls._resolve_today(time_data)
        hr_staleness = staleness_info.get("hr_data_staleness_days")
        stale = _is_stale(hr_staleness)
        estimated: List[str] = []

        api = cls._extract_api_fields(health_params, raw_data)

        baseline = _safe_float(
            health_param_value(
                health_params,
                "health_params_baseline_resting_hr",
                HealthDataConstants.KEY_BASELINE_RESTING_HR,
            )
        ) or api.get("baseline_resting")

        today_in_hp = (
            api.get("resting") is not None
            or api.get("latest") is not None
            or api.get("hrv") is not None
        )
        today_in_stats = cls._latest_hr_entry(raw_data) is not None
        if stale and (today_in_hp or today_in_stats):
            stale = False

        if stale:
            today_rhr = today_latest = today_hrv = None
            workout_avg = workout_max = walking = None
        else:
            today_rhr = api.get("resting")
            today_latest = api.get("latest")
            today_hrv = api.get("hrv")
            workout_avg = api.get("workout_avg")
            workout_max = api.get("workout_max")
            walking = api.get("walking")

        if today_hrv is None and not stale and today_rhr is not None:
            today_hrv = cls._estimate_hrv_from_rhr(today_rhr)
            if today_hrv is not None:
                estimated.append("hrv_score")

        if walking is None and not stale:
            walking = cls._fallback_walking(historical_snapshots)
            if walking is not None:
                estimated.append("walking_heart_rate_avg")

        days = cls._aggregate_days(
            historical_snapshots=historical_snapshots,
            raw_data=raw_data,
            today=today,
            today_rhr=today_rhr,
            today_latest=today_latest,
            today_hrv=today_hrv,
            walking=walking,
            workout_avg=workout_avg,
        )
        window = cls._window_days(days, today, HR_SUMMARY_WINDOW_DAYS)
        baseline_days = cls._baseline_days(days, today)

        summary = cls._build_summary(
            window,
            today_rhr=today_rhr,
            today_latest=today_latest,
            today_hrv=today_hrv,
            baseline=baseline,
            range_min=api.get("range_min"),
            range_max=api.get("range_max"),
        )

        rhr_vals = [d.resting for d in window if d.resting is not None]
        hrv_vals = [d.hrv for d in window if d.hrv is not None]
        latest_vals = [d.latest for d in window if d.latest is not None]

        derived = cls._derived_metrics(rhr_vals, hrv_vals, summary)
        recovery = cls._recovery_assessment(today_rhr or summary.avg_resting_heart_rate, today_hrv, baseline)
        rhr_trend, rhr_trend_meta = cls._series_trend(rhr_vals, days, today, "resting")
        hrv_trend, hrv_trend_meta = cls._series_trend(hrv_vals, days, today, "hrv")
        latest_trend, _ = cls._series_trend(latest_vals, days, today, "latest")

        anomalies = cls._detect_anomalies(window, baseline, summary)

        ctx = {
            "days_measured": summary.days_measured,
            "missing_days": summary.missing_days,
            "coverage_pct": summary.coverage_pct,
            "highest_rhr": max(rhr_vals) if rhr_vals else None,
            "lowest_rhr": min(rhr_vals) if rhr_vals else None,
            "range_min": summary.range_min or summary.min_heart_rate,
            "range_max": summary.range_max or summary.max_heart_rate,
        }

        base_cmp = cls._baseline_comparison(
            today_rhr or summary.avg_resting_heart_rate,
            today_hrv,
            baseline,
            baseline_days,
            recovery.get("score"),
        )

        signals = cls._build_signals(
            summary=summary,
            derived=derived,
            recovery=recovery,
            today_rhr=today_rhr,
            today_latest=today_latest,
            today_hrv=today_hrv,
            baseline=baseline,
            rhr_trend=rhr_trend,
            rhr_trend_meta=rhr_trend_meta,
            hrv_trend=hrv_trend,
            hrv_trend_meta=hrv_trend_meta,
            latest_trend=latest_trend,
            base_cmp=base_cmp,
            walking=walking,
            workout_avg=workout_avg,
            workout_max=workout_max,
            ctx=ctx,
            stale=stale,
        )

        overall = cls._build_overall(
            recovery=recovery,
            derived=derived,
            base_cmp=base_cmp,
            rhr_trend=rhr_trend,
            measured=summary.days_measured,
            stale=stale,
            has_data=today_rhr is not None
            or summary.avg_resting_heart_rate is not None
            or today_hrv is not None,
        )

        stress_high = cls._stress_high(
            today_rhr, baseline, today_hrv, recovery.get("status")
        )

        return HeartRateSignal(
            summary=summary,
            signals=signals,
            anomalies=anomalies,
            overall=overall,
            resting_heart_rate=today_rhr,
            baseline_resting_hr=baseline,
            latest_heart_rate=_safe_int(today_latest),
            workout_avg_hr=_safe_int(workout_avg),
            workout_max_hr=_safe_int(workout_max),
            stress_high=stress_high,
            hrv_score=_safe_int(today_hrv),
            walking_heart_rate_avg=walking,
            estimated_fields=estimated,
        )

    # ── helpers / aggregation ───────────────────────────────────────────

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
    def _estimate_hrv_from_rhr(cls, rhr: float) -> Optional[float]:
        if rhr <= 0:
            return None
        try:
            return float(round(6000 / float(rhr)))
        except Exception:
            return None

    @classmethod
    def _fallback_walking(cls, snapshots: List[Any]) -> Optional[float]:
        vals = []
        for row in snapshots or []:
            v = _snapshot_field(row, CanonicalField.AVERAGE_HEART_RATE_ONE_DAY_BPM)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            return None
        return round(sum(vals) / len(vals), 1)

    @classmethod
    def _extract_api_fields(
        cls, health_params: Dict[str, Any], raw_data: Dict[str, Any]
    ) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {
            "resting": _safe_float(
                health_param_value(
                    health_params,
                    "health_params_resting_heart_rate",
                    HealthDataConstants.KEY_RESTING_HEART_RATE,
                )
            ),
            "latest": _safe_float(
                health_param_value(
                    health_params,
                    "health_params_latest_heart_rate",
                    HealthDataConstants.KEY_LATEST_HEART_RATE,
                )
            ),
            "hrv": _safe_float(
                health_param_value(
                    health_params,
                    "health_params_hrv_score",
                    HealthDataConstants.KEY_HRV_SCORE,
                )
            ),
            "walking": _safe_float(
                health_param_value(
                    health_params,
                    "health_params_walking_heart_rate_avg",
                    HealthDataConstants.KEY_WALKING_HEART_RATE_AVG,
                )
            ),
            "workout_avg": _safe_float(
                health_param_value(
                    health_params,
                    "health_params_workout_avg_hr",
                    HealthDataConstants.KEY_WORKOUT_AVG_HR,
                )
            ),
            "workout_max": _safe_float(
                health_param_value(
                    health_params,
                    "health_params_workout_max_hr",
                    HealthDataConstants.KEY_WORKOUT_MAX_HR,
                )
            ),
            "baseline_resting": _safe_float(
                health_param_value(
                    health_params,
                    "health_params_baseline_resting_hr",
                    HealthDataConstants.KEY_BASELINE_RESTING_HR,
                )
            ),
            "range_min": None,
            "range_max": None,
        }

        sd = cls._raw_summary_data(raw_data)
        if sd:
            if out["hrv"] is None:
                out["hrv"] = _safe_float(
                    sd.get(CanonicalField.AVERAGE_HEART_RATE_VARIABILITY_CURRENT)
                )
            hr_range_min = sd.get(CanonicalField.MINIMUM_HEART_RATE_CURRENT_SUMMARY_PERIOD_BPM)
            hr_range_max = sd.get(CanonicalField.MAXIMUM_HEART_RATE_CURRENT_SUMMARY_PERIOD_BPM)
            if hr_range_min is not None or hr_range_max is not None:
                out["range_min"] = _safe_float(hr_range_min)
                out["range_max"] = _safe_float(hr_range_max)

        # Latest day entry from raw HR data
        entry = cls._latest_hr_entry(raw_data)
        if entry:
            if out["resting"] is None:
                out["resting"] = _safe_float(
                    entry.get(CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM)
                )
            if out["latest"] is None:
                out["latest"] = _safe_float(
                    entry.get(CanonicalField.MOST_RECENT_HEART_RATE_MEASUREMENT_ONE_DAY_BPM)
                )
            if out["walking"] is None:
                out["walking"] = _safe_float(
                    entry.get(CanonicalField.WALKING_HEART_RATE_ONE_DAY_BPM)
                )
            whr = entry.get(CanonicalField.WORKOUT_HEART_RATE_ONE_DAY)
            if isinstance(whr, dict):
                if out["workout_avg"] is None:
                    out["workout_avg"] = _safe_float(
                        whr.get("avg") or whr.get("average")
                    )
                if out["workout_max"] is None:
                    out["workout_max"] = _safe_float(whr.get("max"))
        return out

    @classmethod
    def _raw_summary_data(cls, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        stats = raw_data.get("today_health_stats") or {}
        if isinstance(stats, dict):
            entry = stats.get("HR") or stats.get("RESTING_HEART_RATE") or {}
            if isinstance(entry, dict):
                sd = entry.get("summaryData") or entry.get("summary_data") or {}
                if isinstance(sd, dict):
                    return sd
        health_data = raw_data.get("health_data") or {}
        block = health_data.get(HealthDataConstants.HR_TYPE) or health_data.get("HR")
        if block is None:
            return {}
        sd = getattr(block, "summaryData", None)
        if sd is None and isinstance(block, dict):
            sd = block.get("summaryData") or block.get("summary_data")
        if sd is None:
            return {}
        if hasattr(sd, "model_dump"):
            return sd.model_dump(by_alias=True)
        if isinstance(sd, dict):
            return sd
        return {}

    @classmethod
    def _latest_hr_entry(cls, raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for source in (
            (raw_data.get("today_health_stats") or {}).get("HR"),
            (raw_data.get("today_health_stats") or {}).get("RESTING_HEART_RATE"),
        ):
            if not isinstance(source, dict):
                continue
            data = source.get("data") or []
            if isinstance(data, list) and data:
                last = data[-1]
                if isinstance(last, dict):
                    # nested week.data
                    inner = last.get("data")
                    if isinstance(inner, list) and inner:
                        e = inner[-1]
                        return e if isinstance(e, dict) else None
                    return last
        health_data = raw_data.get("health_data") or {}
        block = health_data.get(HealthDataConstants.HR_TYPE) or health_data.get("HR")
        data = getattr(block, "data", None) if block is not None else None
        if data is None and isinstance(block, dict):
            data = block.get("data")
        if not data:
            return None
        last_week = data[-1]
        days = getattr(last_week, "data", None)
        if days is None and isinstance(last_week, dict):
            days = last_week.get("data")
            if days is None and last_week.get("date"):
                return last_week
        if not days:
            return None
        last = days[-1]
        if hasattr(last, "model_dump"):
            return last.model_dump(by_alias=True)
        return last if isinstance(last, dict) else None

    @classmethod
    def _aggregate_days(
        cls,
        *,
        historical_snapshots: List[Any],
        raw_data: Dict[str, Any],
        today: Optional[date],
        today_rhr: Optional[float],
        today_latest: Optional[float],
        today_hrv: Optional[float],
        walking: Optional[float],
        workout_avg: Optional[float],
    ) -> List[HeartRateDay]:
        by_date: Dict[date, HeartRateDay] = {}

        # Collect health scores from fallback sources (similar to steps processor)
        # to ensure avg_health_score is non-null whenever daily data exists.
        daily_health_score_by_date: Dict[date, float] = {}
        daily_health_score_by_date = cls._collect_health_scores_from_sources(raw_data)

        for row in historical_snapshots or []:
            d = _snapshot_date(row)
            if d is None:
                continue
            # Try to get health_score from snapshot row, then fallback to collected scores
            score = _safe_float(
                _snapshot_field(row, CanonicalField.HEALTH_SCORE_ONE_DAY)
                or _snapshot_field(row, "healthScore")
                or _snapshot_field(row, "health_score")
                or _snapshot_field(row, "h_health_score")
            )
            if score is None:
                score = daily_health_score_by_date.get(d)
            day = HeartRateDay(
                day_date=d,
                resting=_safe_float(
                    _snapshot_field(row, CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM)
                ),
                latest=_safe_float(
                    _snapshot_field(row, CanonicalField.AVERAGE_HEART_RATE_ONE_DAY_BPM)
                ),
                # HRV does not live on snapshot rows; authoritative source is
                # today_health_stats["HR"]["summaryData"] via _extract_api_fields.
                hrv=None,
                health_score=score,
                walking=_safe_float(
                    _snapshot_field(row, CanonicalField.WALKING_HEART_RATE_ONE_DAY_BPM)
                ),
            )
            if day.resting is not None or day.latest is not None:
                by_date[d] = day

        cls._merge_raw_hr_entries(by_date, raw_data, daily_health_score_by_date)

        if today is not None and (
            today_rhr is not None or today_latest is not None or today_hrv is not None
        ):
            existing = by_date.get(today) or HeartRateDay(day_date=today)
            if today_rhr is not None:
                existing.resting = today_rhr
            if today_latest is not None:
                existing.latest = today_latest
            if today_hrv is not None:
                existing.hrv = today_hrv
            if walking is not None:
                existing.walking = walking
            if workout_avg is not None:
                existing.workout_avg = workout_avg
            by_date[today] = existing

        return sorted(by_date.values(), key=lambda x: x.day_date)

    @classmethod
    def _collect_health_scores_from_sources(
        cls, raw_data: Dict[str, Any]
    ) -> Dict[date, float]:
        """Collect health scores from balance_scores_30d and balance_scores_7d
        to ensure avg_health_score is non-null whenever any daily data exists in the window."""
        daily_health_score_by_date: Dict[date, float] = {}

        # Source: balance_scores_30d and balance_scores_7d
        for src_key in ("balance_scores_30d", "balance_scores_7d"):
            for item in raw_data.get(src_key) or []:
                if hasattr(item, "model_dump") and not isinstance(item, dict):
                    try:
                        item = item.model_dump(by_alias=True, mode="json")
                    except Exception:
                        continue
                if not isinstance(item, dict):
                    continue
                raw_d = (
                    item.get("date")
                    or item.get("day")
                    or item.get("snapshot_date")
                )
                if not raw_d:
                    continue
                try:
                    d = date.fromisoformat(str(raw_d).split("T")[0])
                except Exception:
                    continue
                hs = _safe_float(
                    item.get("healthScore")
                    or item.get("health_score")
                    or item.get("h_health_score")
                )
                if hs is not None and d not in daily_health_score_by_date:
                    daily_health_score_by_date[d] = hs

        if daily_health_score_by_date:
            try:
                from utils.logger import logger as _logger
                _logger.info(
                    f"[heart_rate_health_score_fallback] source=fallback_sources "
                    f"days={len(daily_health_score_by_date)}"
                )
            except Exception:
                pass

        return daily_health_score_by_date

    @classmethod
    def _merge_raw_hr_entries(
        cls,
        by_date: Dict[date, HeartRateDay],
        raw_data: Dict[str, Any],
        fallback_health_scores: Optional[Dict[date, float]] = None,
    ) -> None:
        health_data = raw_data.get("health_data") or {}
        block = health_data.get(HealthDataConstants.HR_TYPE) or health_data.get("HR")
        data = getattr(block, "data", None) if block is not None else None
        if data is None and isinstance(block, dict):
            data = block.get("data")
        stats = raw_data.get("today_health_stats") or {}
        if data is None and isinstance(stats, dict):
            entry = stats.get("HR") or {}
            if isinstance(entry, dict):
                data = entry.get("data")
        if not data:
            return
        for week in data:
            days = getattr(week, "data", None)
            if days is None and isinstance(week, dict):
                if week.get("date") or week.get("restingHeartRate"):
                    days = [week]
                else:
                    days = week.get("data")
            if not days:
                continue
            for entry in days:
                if isinstance(entry, dict):
                    date_raw = entry.get("date")
                    rhr = (
                        entry.get(CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM)
                        or entry.get("restingHeartRate")
                        or entry.get("resting_heart_rate")
                    )
                    latest = (
                        entry.get(CanonicalField.MOST_RECENT_HEART_RATE_MEASUREMENT_ONE_DAY_BPM)
                        or entry.get("latestHR")
                        or entry.get("latest_hr")
                        or entry.get(CanonicalField.MOST_RECENT_HEART_RATE_MEASUREMENT_ONE_DAY_BPM)
                    )
                    score = (
                        entry.get("healthScore")
                        or entry.get(CanonicalField.HEALTH_SCORE_ONE_DAY)
                    )
                    walking = entry.get("walkingHeartRateAverage")
                    sleep_hr = entry.get("sleepHeartRate")
                else:
                    date_raw = getattr(entry, "date", None)
                    rhr = getattr(entry, "restingHeartRate", None)
                    latest = getattr(entry, "latestHR", None)
                    score = getattr(entry, "healthScore", None)
                    walking = getattr(entry, "walkingHeartRateAverage", None)
                    sleep_hr = getattr(entry, "sleepHeartRate", None)
                if not date_raw:
                    continue
                try:
                    d = date.fromisoformat(str(date_raw).split("T")[0])
                except Exception:
                    continue
                day = by_date.get(d) or HeartRateDay(day_date=d)
                if rhr is not None:
                    day.resting = _safe_float(rhr)
                if latest is not None:
                    day.latest = _safe_float(latest)
                if score is not None:
                    day.health_score = _safe_float(score)
                elif fallback_health_scores is not None and day.health_score is None:
                    # Fallback: use health_score from balance_scores
                    day.health_score = fallback_health_scores.get(d)
                if walking is not None:
                    day.walking = _safe_float(walking)
                if sleep_hr is not None:
                    day.sleep_hr = _safe_float(sleep_hr)
                by_date[d] = day

    @classmethod
    def _window_days(
        cls, days: List[HeartRateDay], today: Optional[date], window_days: int
    ) -> List[HeartRateDay]:
        if today is None:
            return list(days[-window_days:])
        cutoff = today - timedelta(days=window_days)
        return [d for d in days if cutoff < d.day_date <= today]

    @classmethod
    def _baseline_days(
        cls, days: List[HeartRateDay], today: Optional[date]
    ) -> List[HeartRateDay]:
        if today is None:
            return []
        end = today - timedelta(days=HR_SUMMARY_WINDOW_DAYS)
        start = end - timedelta(days=HR_BASELINE_WINDOW_DAYS)
        return [d for d in days if start < d.day_date <= end]

    # ── summary / derived / recovery / trend ────────────────────────────

    @classmethod
    def _build_summary(
        cls,
        days: List[HeartRateDay],
        *,
        today_rhr: Optional[float],
        today_latest: Optional[float],
        today_hrv: Optional[float],
        baseline: Optional[float],
        range_min: Optional[float],
        range_max: Optional[float],
    ) -> HeartRateSummaryMetrics:
        rhrs = [d.resting for d in days if d.resting is not None]
        latests = [d.latest for d in days if d.latest is not None]
        hrvs = [d.hrv for d in days if d.hrv is not None]
        scores = [d.health_score for d in days if d.health_score is not None]
        all_hrs = rhrs + latests
        measured = len({d.day_date for d in days if d.resting is not None or d.latest is not None or d.hrv is not None})
        window = HR_SUMMARY_WINDOW_DAYS
        missing = max(0, window - measured)
        coverage = round(100.0 * measured / window, 1) if window else None

        min_hr = min(all_hrs) if all_hrs else range_min
        max_hr = max(all_hrs) if all_hrs else range_max
        if range_min is not None:
            min_hr = min(min_hr, range_min) if min_hr is not None else range_min
        if range_max is not None:
            max_hr = max(max_hr, range_max) if max_hr is not None else range_max

        return HeartRateSummaryMetrics(
            avg_latest_heart_rate=round(_mean(latests), 1) if latests else today_latest,
            avg_resting_heart_rate=round(_mean(rhrs), 1) if rhrs else today_rhr,
            min_heart_rate=round(min_hr, 1) if min_hr is not None else None,
            max_heart_rate=round(max_hr, 1) if max_hr is not None else None,
            avg_health_score=round(_mean(scores), 1) if scores else None,
            days_measured=measured,
            missing_days=missing,
            window_days=window,
            coverage_pct=coverage,
            today_resting=today_rhr,
            today_latest=today_latest,
            today_hrv=today_hrv,
            baseline_resting=baseline,
            range_min=range_min,
            range_max=range_max,
        )

    @classmethod
    def _derived_metrics(
        cls,
        rhr_vals: List[float],
        hrv_vals: List[float],
        summary: HeartRateSummaryMetrics,
    ) -> Dict[str, Any]:
        rhr_stdev = _stdev(rhr_vals)
        hrv_stdev = _stdev(hrv_vals)
        rhr_std = round(rhr_stdev, 2) if rhr_stdev is not None else None
        hrv_std = round(hrv_stdev, 2) if hrv_stdev is not None else None
        rhr_mean = _mean(rhr_vals)
        rhr_cv = (
            round(rhr_std / rhr_mean, 3)
            if rhr_std is not None and rhr_mean and rhr_mean > 0
            else None
        )
        stability_score = None
        if rhr_std is not None:
            # 0 bpm std → 100; 8 bpm → 0
            stability_score = round(max(0.0, min(100.0, 100.0 * (1 - rhr_std / 8.0))), 1)

        hr_range = None
        if summary.min_heart_rate is not None and summary.max_heart_rate is not None:
            hr_range = round(summary.max_heart_rate - summary.min_heart_rate, 1)

        return {
            "rhr_std": rhr_std,
            "hrv_std": hrv_std,
            "rhr_cv": rhr_cv,
            "stability_score": stability_score,
            "heart_rate_range_span": hr_range,
        }

    @classmethod
    def _recovery_assessment(
        cls,
        rhr: Optional[float],
        hrv: Optional[float],
        baseline: Optional[float],
    ) -> Dict[str, Any]:
        if rhr is None and hrv is None:
            return {"status": "insufficient_data", "score": None}

        # RHR points: lower vs baseline / absolute is better
        if rhr is None:
            rhr_pts = 70.0
        elif baseline is not None:
            delta = rhr - baseline
            if delta <= 0:
                rhr_pts = min(100.0, 85.0 - delta * 2)
            elif delta <= HR_RHR_NORMAL_DELTA_BPM:
                rhr_pts = 75.0
            elif delta <= HR_RHR_ELEVATED_DELTA_BPM:
                rhr_pts = 55.0
            else:
                rhr_pts = 35.0
        else:
            if rhr <= 60:
                rhr_pts = 90.0
            elif rhr <= 70:
                rhr_pts = 75.0
            elif rhr <= 80:
                rhr_pts = 55.0
            else:
                rhr_pts = 35.0

        if hrv is None:
            hrv_pts = 70.0
        elif hrv >= HR_HRV_EXCELLENT:
            hrv_pts = 95.0
        elif hrv >= HR_HRV_GOOD:
            hrv_pts = 80.0
        elif hrv >= HR_HRV_FAIR:
            hrv_pts = 65.0
        elif hrv >= HR_HRV_LOW:
            hrv_pts = 45.0
        else:
            hrv_pts = 30.0

        score = round(
            rhr_pts * HR_RECOVERY_WEIGHT_RHR + hrv_pts * HR_RECOVERY_WEIGHT_HRV, 1
        )
        if score >= 85:
            status = "excellent"
        elif score >= 70:
            status = "good"
        elif score >= 55:
            status = "fair"
        else:
            status = "poor"
        return {
            "status": status,
            "score": score,
            "resting_heart_rate": rhr,
            "hrv": hrv,
        }

    @classmethod
    def _series_trend(
        cls,
        series: List[float],
        all_days: List[HeartRateDay],
        today: Optional[date],
        kind: str,
    ) -> Tuple[str, Dict[str, Any]]:
        significant = (
            HR_TREND_SIGNIFICANT_HRV
            if kind == "hrv"
            else HR_TREND_SIGNIFICANT_RHR_BPM
        )
        for days_n in HR_TREND_WINDOWS_DAYS:
            vals = series
            if today is not None:
                wd = cls._window_days(all_days, today, days_n)
                if kind == "resting":
                    vals = [d.resting for d in wd if d.resting is not None]
                elif kind == "hrv":
                    vals = [d.hrv for d in wd if d.hrv is not None]
                else:
                    vals = [d.latest for d in wd if d.latest is not None]
            if len(vals) < HR_MIN_DAYS_FOR_TREND:
                continue
            first, last = vals[0], vals[-1]
            prior = vals[:-1]
            prior_avg = sum(prior) / len(prior) if prior else first
            change = round(last - prior_avg, 2)
            # For RHR, lower is improving; for HRV, higher is improving
            if kind == "hrv":
                if change >= significant:
                    status = "improving"
                elif change <= -significant:
                    status = "declining"
                else:
                    status = "stable"
            else:
                if change <= -significant:
                    status = "improving"
                elif change >= significant:
                    status = "declining"
                else:
                    status = "stable"
            return status, {
                "window": f"{days_n}d",
                "change": change,
                "first": first,
                "last": last,
            }
        return "insufficient_data", {}

    @classmethod
    def _baseline_comparison(
        cls,
        rhr: Optional[float],
        hrv: Optional[float],
        baseline: Optional[float],
        baseline_days: List[HeartRateDay],
        recovery_score: Optional[float],
    ) -> Dict[str, Any]:
        base_rhrs = [d.resting for d in baseline_days if d.resting is not None]
        base_hrvs = [d.hrv for d in baseline_days if d.hrv is not None]
        ref_rhr = baseline if baseline is not None else _mean(base_rhrs)
        ref_hrv = _mean(base_hrvs)

        rhr_diff = (
            round(rhr - ref_rhr, 1) if rhr is not None and ref_rhr is not None else None
        )
        hrv_diff = (
            round(hrv - ref_hrv, 1) if hrv is not None and ref_hrv is not None else None
        )

        if rhr_diff is None and hrv_diff is None:
            status = "insufficient_data"
        elif rhr_diff is not None:
            if abs(rhr_diff) <= HR_RHR_NORMAL_DELTA_BPM:
                status = "normal"
            elif rhr_diff >= HR_RHR_ELEVATED_DELTA_BPM:
                status = "elevated"
            elif rhr_diff <= HR_RHR_LOW_DELTA_BPM:
                status = "lower_than_usual"
            else:
                status = "slightly_off"
        else:
            status = "normal"

        enough = (
            (baseline is not None)
            or len(base_rhrs) >= HR_MIN_DAYS_FOR_BASELINE
            or len(base_hrvs) >= HR_MIN_DAYS_FOR_BASELINE
        )
        if not enough and status != "insufficient_data":
            # keep status but note thin baseline
            pass

        return {
            "status": status if enough or baseline is not None else "insufficient_data",
            "resting_heart_rate_difference": rhr_diff,
            "hrv_difference": hrv_diff,
            "baseline_resting": ref_rhr,
            "baseline_hrv": round(ref_hrv, 1) if ref_hrv is not None else None,
            "recovery_score": recovery_score,
            "baseline_days": len(base_rhrs) or len(base_hrvs),
        }

    # ── anomalies / signals / overall ───────────────────────────────────

    @classmethod
    def _detect_anomalies(
        cls,
        days: List[HeartRateDay],
        baseline: Optional[float],
        summary: HeartRateSummaryMetrics,
    ) -> List[HeartRateAnomaly]:
        anomalies: List[HeartRateAnomaly] = []
        prev_rhr: Optional[float] = None
        for d in days:
            ds = d.day_date.isoformat()
            if d.resting is not None:
                if d.resting >= HR_ANOMALY_HIGH_RHR_BPM or (
                    baseline is not None
                    and d.resting >= baseline + HEART_RATE_RESTING_ELEVATED_DELTA_BPM
                ):
                    anomalies.append(
                        HeartRateAnomaly(
                            kind="elevated_resting_hr",
                            day_date=ds,
                            value=d.resting,
                            threshold=float(
                                baseline + HEART_RATE_RESTING_ELEVATED_DELTA_BPM
                                if baseline is not None
                                else HR_ANOMALY_HIGH_RHR_BPM
                            ),
                            evidence={"resting": d.resting, "baseline": baseline},
                        )
                    )
                if d.resting <= HR_ANOMALY_LOW_RHR_BPM:
                    anomalies.append(
                        HeartRateAnomaly(
                            kind="unusually_low_heart_rate",
                            day_date=ds,
                            value=d.resting,
                            threshold=float(HR_ANOMALY_LOW_RHR_BPM),
                            evidence={"resting": d.resting},
                        )
                    )
                if (
                    prev_rhr is not None
                    and abs(d.resting - prev_rhr) >= HR_ANOMALY_SUDDEN_RHR_JUMP_BPM
                ):
                    anomalies.append(
                        HeartRateAnomaly(
                            kind="sudden_recovery_drop"
                            if d.resting > prev_rhr
                            else "sudden_rhr_drop",
                            day_date=ds,
                            value=d.resting,
                            threshold=float(HR_ANOMALY_SUDDEN_RHR_JUMP_BPM),
                            evidence={"resting": d.resting, "previous": prev_rhr},
                        )
                    )
                prev_rhr = d.resting
            if d.hrv is not None and d.hrv <= HR_ANOMALY_LOW_HRV:
                anomalies.append(
                    HeartRateAnomaly(
                        kind="low_hrv",
                        day_date=ds,
                        value=d.hrv,
                        threshold=float(HR_ANOMALY_LOW_HRV),
                        evidence={"hrv": d.hrv},
                    )
                )
            if d.latest is not None and d.latest >= HR_ANOMALY_HIGH_LATEST_BPM:
                anomalies.append(
                    HeartRateAnomaly(
                        kind="unusually_high_heart_rate",
                        day_date=ds,
                        value=d.latest,
                        threshold=float(HR_ANOMALY_HIGH_LATEST_BPM),
                        evidence={"latest": d.latest},
                    )
                )
            if d.latest is not None and d.latest <= HR_ANOMALY_LOW_LATEST_BPM:
                anomalies.append(
                    HeartRateAnomaly(
                        kind="unusually_low_heart_rate",
                        day_date=ds,
                        value=d.latest,
                        threshold=float(HR_ANOMALY_LOW_LATEST_BPM),
                        evidence={"latest": d.latest},
                    )
                )
        return anomalies

    @classmethod
    def _build_signals(cls, **kw: Any) -> HeartRateSignalsMap:
        summary: HeartRateSummaryMetrics = kw["summary"]
        derived: Dict[str, Any] = kw["derived"]
        recovery: Dict[str, Any] = kw["recovery"]
        today_rhr = kw["today_rhr"]
        today_latest = kw["today_latest"]
        today_hrv = kw["today_hrv"]
        baseline = kw["baseline"]
        rhr_trend = kw["rhr_trend"]
        rhr_trend_meta = kw["rhr_trend_meta"]
        hrv_trend = kw["hrv_trend"]
        hrv_trend_meta = kw["hrv_trend_meta"]
        latest_trend = kw["latest_trend"]
        base_cmp: Dict[str, Any] = kw["base_cmp"]
        walking = kw["walking"]
        workout_avg = kw["workout_avg"]
        workout_max = kw["workout_max"]
        ctx = kw["ctx"]
        stale = kw["stale"]

        # Resting HR
        avg_rhr = summary.avg_resting_heart_rate
        latest_rhr = today_rhr
        if stale and avg_rhr is None:
            rhr_status = "insufficient_data"
        elif latest_rhr is None and avg_rhr is None:
            rhr_status = "insufficient_data"
        else:
            rhr_status = base_cmp.get("status") or "normal"
            if latest_rhr is not None and latest_rhr >= HEART_RATE_RESTING_HIGH_BPM:
                rhr_status = "elevated"
        rhr_diff = base_cmp.get("resting_heart_rate_difference")
        resting_sig = _dim(
            rhr_status,
            metrics={
                "average": avg_rhr,
                "latest": latest_rhr,
                "baseline": baseline,
            },
            comparison={"vs_baseline": rhr_diff},
            trend=rhr_trend if rhr_trend != "insufficient_data" else None,
            evidence_context={
                **ctx,
                "range": {"min": ctx.get("lowest_rhr"), "max": ctx.get("highest_rhr")},
            },
        )

        # HRV
        if today_hrv is None:
            hrv_status = "insufficient_data"
        else:
            v = today_hrv
            if v >= HR_HRV_EXCELLENT:
                hrv_status = "excellent"
            elif v >= HR_HRV_GOOD:
                hrv_status = "good"
            elif v >= HR_HRV_FAIR:
                hrv_status = "fair"
            elif v >= HR_HRV_LOW:
                hrv_status = "low"
            else:
                hrv_status = "poor"
        hrv_sig = _dim(
            hrv_status,
            metrics={
                "latest": today_hrv,
                "std_dev": derived.get("hrv_std"),
            },
            comparison={"vs_baseline": base_cmp.get("hrv_difference")},
            trend=hrv_trend if hrv_trend != "insufficient_data" else None,
            evidence_context=ctx,
        )

        # Range
        if summary.min_heart_rate is None and summary.max_heart_rate is None:
            range_status = "insufficient_data"
        else:
            span = derived.get("heart_rate_range_span")
            range_status = "wide" if span is not None and span >= 60 else "normal"
        range_sig = _dim(
            range_status,
            metrics={
                "min": summary.min_heart_rate,
                "max": summary.max_heart_rate,
                "span": derived.get("heart_rate_range_span"),
                "api_range_min": summary.range_min,
                "api_range_max": summary.range_max,
            },
            evidence_context=ctx,
        )

        # Stability
        stab = derived.get("stability_score")
        std = derived.get("rhr_std")
        if stab is None or summary.days_measured < HR_MIN_DAYS_FOR_STABILITY:
            stab_status = "insufficient_data"
        elif std is not None and std <= HR_STABILITY_HIGH_STD:
            stab_status = "high"
        elif std is not None and std <= HR_STABILITY_MED_STD:
            stab_status = "medium"
        else:
            stab_status = "low"
        stability_sig = _dim(
            stab_status,
            metrics={
                "score": stab,
                "standard_deviation": std,
                "cv": derived.get("rhr_cv"),
            },
            evidence_context=ctx,
        )

        # Recovery state
        recovery_sig = _dim(
            recovery.get("status") or "insufficient_data",
            metrics={
                "score": recovery.get("score"),
                "resting_heart_rate": recovery.get("resting_heart_rate"),
                "hrv": recovery.get("hrv"),
            },
            evidence_context=ctx,
        )

        # Recovery trend (combine RHR + HRV trends)
        if rhr_trend == "insufficient_data" and hrv_trend == "insufficient_data":
            rec_trend_status = "insufficient_data"
        elif rhr_trend == "declining" or hrv_trend == "declining":
            rec_trend_status = "declining"
        elif rhr_trend == "improving" or hrv_trend == "improving":
            rec_trend_status = "improving"
        else:
            rec_trend_status = "stable"
        recovery_trend_sig = _dim(
            rec_trend_status,
            metrics={
                "resting_heart_rate_change": rhr_trend_meta.get("change"),
                "hrv_change": hrv_trend_meta.get("change"),
                "latest_hr_trend": latest_trend,
                "rhr_trend": rhr_trend,
                "hrv_trend": hrv_trend,
            },
            evidence_context=ctx,
        )

        # Baseline comparison signal
        baseline_sig = _dim(
            base_cmp.get("status") or "insufficient_data",
            metrics={
                "baseline_resting": base_cmp.get("baseline_resting"),
                "baseline_hrv": base_cmp.get("baseline_hrv"),
                "recovery_score": base_cmp.get("recovery_score"),
            },
            comparison={
                "resting_heart_rate_difference": base_cmp.get(
                    "resting_heart_rate_difference"
                ),
                "hrv_difference": base_cmp.get("hrv_difference"),
            },
            evidence_context={**ctx, "baseline_days": base_cmp.get("baseline_days")},
        )

        walking_sig = None
        if walking is not None:
            walking_sig = _dim(
                "available",
                metrics={"average": walking},
                evidence_context=ctx,
            )

        workout_sig = None
        if workout_avg is not None or workout_max is not None:
            workout_sig = _dim(
                "available",
                metrics={"avg": workout_avg, "max": workout_max},
                evidence_context=ctx,
            )

        stress_status = "high" if kw.get("stale") is False and (
            (today_rhr is not None and baseline is not None
             and today_rhr >= baseline + HEART_RATE_RESTING_ELEVATED_DELTA_BPM)
            or (today_hrv is not None and today_hrv <= HR_ANOMALY_LOW_HRV)
            or recovery.get("status") == "poor"
        ) else ("insufficient_data" if stale else "normal")
        stress_sig = _dim(
            stress_status,
            metrics={
                "resting_heart_rate": today_rhr,
                "hrv": today_hrv,
                "baseline": baseline,
                "recovery_status": recovery.get("status"),
            },
            evidence_context=ctx,
        )

        return HeartRateSignalsMap(
            resting_heart_rate=resting_sig,
            heart_rate_variability=hrv_sig,
            heart_rate_range=range_sig,
            heart_rate_stability=stability_sig,
            recovery_state=recovery_sig,
            recovery_trend=recovery_trend_sig,
            baseline_comparison=baseline_sig,
            walking_heart_rate=walking_sig,
            workout_heart_rate=workout_sig,
            stress_indicator=stress_sig,
        )

    @classmethod
    def _stress_high(
        cls,
        rhr: Optional[float],
        baseline: Optional[float],
        hrv: Optional[float],
        recovery_status: Optional[str],
    ) -> bool:
        if rhr is not None and baseline is not None:
            if rhr >= baseline + HEART_RATE_RESTING_ELEVATED_DELTA_BPM:
                return True
        if rhr is not None and rhr >= HEART_RATE_RESTING_HIGH_BPM:
            return True
        if hrv is not None and hrv <= HR_ANOMALY_LOW_HRV:
            return True
        if recovery_status == "poor":
            return True
        return False

    @classmethod
    def _build_overall(
        cls,
        *,
        recovery: Dict[str, Any],
        derived: Dict[str, Any],
        base_cmp: Dict[str, Any],
        rhr_trend: str,
        measured: int,
        stale: bool,
        has_data: bool,
    ) -> HeartRateOverall:
        if not has_data:
            return HeartRateOverall(status="insufficient_data", confidence="low")

        rec_score = recovery.get("score")
        rec_pts = float(rec_score) if rec_score is not None else 65.0
        stab = derived.get("stability_score")
        stab_pts = float(stab) if stab is not None else 70.0

        base_status = base_cmp.get("status")
        if base_status == "normal":
            base_pts = 85.0
        elif base_status == "slightly_off":
            base_pts = 70.0
        elif base_status == "elevated":
            base_pts = 40.0
        elif base_status == "lower_than_usual":
            base_pts = 75.0
        else:
            base_pts = 65.0

        if rhr_trend == "improving":
            trend_pts = 90.0
        elif rhr_trend == "stable":
            trend_pts = 75.0
        elif rhr_trend == "declining":
            trend_pts = 45.0
        else:
            trend_pts = 65.0

        score = round(
            rec_pts * HR_OVERALL_WEIGHT_RECOVERY
            + stab_pts * HR_OVERALL_WEIGHT_STABILITY
            + base_pts * HR_OVERALL_WEIGHT_BASELINE
            + trend_pts * HR_OVERALL_WEIGHT_TREND,
            1,
        )
        if score >= 75:
            status = "good"
        elif score >= 55:
            status = "fair"
        else:
            status = "poor"

        if measured >= 5 or base_cmp.get("baseline_resting") is not None:
            confidence = "high"
        elif measured >= 3:
            confidence = "medium"
        else:
            confidence = "low"
        if stale:
            confidence = "low"

        return HeartRateOverall(status=status, score=score, confidence=confidence)
