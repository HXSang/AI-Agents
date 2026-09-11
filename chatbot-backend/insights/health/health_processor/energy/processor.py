"""Energy signal processor — single source of truth for energy calculations.

Pipeline: raw aggregation → derived → trend → baseline → pattern → anomalies
→ evidence-rich signals (summary / signals / overall). The LLM only interprets.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from insights.health.canonical_field_mapping import CanonicalField
from insights.health.health_processor.common.utils import health_param_value
from insights.health.health_processor.energy.constants import (
    ENERGY_ANOMALY_DROP_FRAC,
    ENERGY_ANOMALY_HIGH_FRAC,
    ENERGY_ANOMALY_LOW_FRAC,
    ENERGY_ANOMALY_SPIKE_FRAC,
    ENERGY_CONSISTENCY_HIGH_CV,
    ENERGY_CONSISTENCY_MED_CV,
    ENERGY_LEVEL_MILD_PCT,
    ENERGY_LEVEL_MODERATE_PCT,
    ENERGY_LEVEL_SEVERE_PCT,
    ENERGY_MIN_DAYS_FOR_CONSISTENCY,
    ENERGY_MIN_DAYS_FOR_PATTERN,
    ENERGY_MIN_DAYS_FOR_TREND,
    ENERGY_OVERALL_WEIGHT_BASELINE,
    ENERGY_OVERALL_WEIGHT_CONSISTENCY,
    ENERGY_OVERALL_WEIGHT_TREND,
    ENERGY_OVERALL_WEIGHT_VOLUME,
    ENERGY_SUMMARY_WINDOW_DAYS,
    ENERGY_TREND_SIGNIFICANT_FRAC,
    ENERGY_TREND_WINDOWS_DAYS,
    ENERGY_VS_BASELINE_HIGHER_PCT,
    ENERGY_VS_BASELINE_LOWER_PCT,
    ENERGY_VS_BASELINE_NORMAL_PCT,
)
from insights.schemas.processed_context import (
    EnergyAnomaly,
    EnergyDimensionSignal,
    EnergyEvidence,
    EnergyOverall,
    EnergySignal,
    EnergySignalsMap,
    EnergySummaryMetrics,
)
from services.executor.constant import APIResponseKeys, HealthDataConstants


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
) -> EnergyDimensionSignal:
    metrics = metrics or {}
    comparison = comparison or {}
    return EnergyDimensionSignal(
        status=status,
        metrics=metrics,
        comparison=comparison,
        trend=trend,
        evidence=EnergyEvidence(
            metrics=dict(metrics),
            comparison=dict(comparison),
            context=evidence_context or {},
        ),
    )


def _pct_delta(current: Optional[float], ref: Optional[float]) -> Optional[float]:
    if current is None or ref is None or ref == 0:
        return None
    return round((current - ref) / ref * 100.0, 1)


def _baseline_status(pct: Optional[float]) -> str:
    if pct is None:
        return "insufficient_data"
    if abs(pct) <= ENERGY_VS_BASELINE_NORMAL_PCT:
        return "normal"
    if pct >= ENERGY_VS_BASELINE_HIGHER_PCT:
        return "higher_than_usual"
    if pct <= ENERGY_VS_BASELINE_LOWER_PCT:
        return "lower_than_usual"
    return "slightly_off"


@dataclass
class EnergyDay:
    day_date: date
    energy_burn: Optional[float] = None
    health_score: Optional[float] = None


class EnergySignalProcessor:
    """Transform raw energy inputs into evidence-rich EnergySignal."""

    @classmethod
    def process(
        cls,
        health_params: Dict[str, Any],
        historical_snapshots: Optional[List[Any]] = None,
        staleness_info: Optional[Dict[str, Any]] = None,
        time_data: Optional[Dict[str, Any]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
        *,
        activity_calories: Optional[float] = None,
        workout_duration_min: Optional[int] = None,
        workout_activity_type: Optional[str] = None,
        cardio_week_pct: Optional[float] = None,
    ) -> EnergySignal:
        try:
            return cls._process_unsafe(
                health_params=health_params or {},
                historical_snapshots=historical_snapshots or [],
                staleness_info=staleness_info or {},
                time_data=time_data or {},
                raw_data=raw_data or {},
                activity_calories=activity_calories,
                workout_duration_min=workout_duration_min,
                workout_activity_type=workout_activity_type,
                cardio_week_pct=cardio_week_pct,
            )
        except Exception:
            try:
                from utils.logger import logger as _logger

                _logger.warning("EnergySignalProcessor failed", exc_info=True)
            except Exception:
                pass
            return EnergySignal(
                level="unavailable",
                overall=EnergyOverall(status="insufficient_data", confidence="low"),
            )

    @classmethod
    def compute_level(
        cls,
        calories: Optional[float],
        week_vs_prior_pct: Optional[float],
        *,
        stale: bool,
    ) -> str:
        """Legacy severity ladder used by flat `level` + period sync."""
        if stale or (calories is None and week_vs_prior_pct is None):
            return "unavailable"
        if calories is not None and calories <= 0:
            return "not_started"
        if week_vs_prior_pct is None:
            return "ok"
        if week_vs_prior_pct <= ENERGY_LEVEL_SEVERE_PCT:
            return "severe"
        if week_vs_prior_pct <= ENERGY_LEVEL_MODERATE_PCT:
            return "moderate"
        if week_vs_prior_pct <= ENERGY_LEVEL_MILD_PCT:
            return "mild"
        return "ok"

    @classmethod
    def _process_unsafe(
        cls,
        health_params: Dict[str, Any],
        historical_snapshots: List[Any],
        staleness_info: Dict[str, Any],
        time_data: Dict[str, Any],
        raw_data: Dict[str, Any],
        activity_calories: Optional[float],
        workout_duration_min: Optional[int],
        workout_activity_type: Optional[str],
        cardio_week_pct: Optional[float],
    ) -> EnergySignal:
        today = cls._resolve_today(time_data)
        energy_staleness = staleness_info.get("energy_data_staleness_days")
        stale = _is_stale(energy_staleness)

        api = cls._extract_summary_data(health_params, raw_data)
        if stale and (
            api.get("total_active_energy") is not None
            or api.get("current_month_avg") is not None
        ):
            stale = False
        estimated: List[str] = []

        today_active = api.get("total_active_energy")
        if today_active is None:
            today_active = _safe_float(
                health_param_value(
                    health_params,
                    "health_params_calories_burned_today",
                    HealthDataConstants.KEY_CALORIES_BURNED_TODAY,
                )
            )
        if today_active is None:
            today_active = _safe_float(activity_calories)
        if today_active is None and not stale:
            today_active = cls._extract_calories_today(raw_data)
            if today_active is not None:
                estimated.append("total_active_energy")

        month_resting_total = api.get("month_resting_total")
        month_avg = api.get("current_month_avg")
        week_avg = api.get("week_avg")
        prev_month = api.get("previous_month_avg")

        week_pct = _safe_float(cardio_week_pct)
        if week_pct is None:
            week_pct = _safe_float(
                health_param_value(
                    health_params,
                    "health_params_energy_week_vs_prior_month_pct",
                    HealthDataConstants.KEY_ENERGY_WEEK_VS_PRIOR_MONTH_PCT,
                )
            )
        if (
            week_pct is None
            and week_avg is not None
            and prev_month is not None
            and prev_month
        ):
            week_pct = round((week_avg - prev_month) / prev_month * 100.0, 1)
            estimated.append("week_vs_prior_month_pct")

        active_minutes = None
        if not stale:
            active_minutes = _safe_float(
                health_param_value(
                    health_params,
                    "health_params_active_minutes_today",
                    HealthDataConstants.KEY_ACTIVE_MINUTES_TODAY,
                )
            )

        if stale:
            # Keep period averages; clear today-only readings
            today_active_out = None
            month_resting_total_out = None
            active_minutes = None
            workout_duration_min = None
            workout_activity_type = None
        else:
            today_active_out = today_active
            month_resting_total_out = month_resting_total

        days = cls._aggregate_days(
            historical_snapshots=historical_snapshots,
            raw_data=raw_data,
            today=today,
            today_burn=today_active_out,
        )
        window_days = cls._window_days(days, today, ENERGY_SUMMARY_WINDOW_DAYS)

        summary = cls._build_summary(
            window_days,
            today_active=today_active_out,
            month_resting_total=month_resting_total_out,
            month_avg=month_avg,
            week_avg=week_avg,
            prev_month=prev_month,
        )
        # Prefer API week avg when daily series is thin
        if summary.avg_energy_burn is None and week_avg is not None:
            summary.avg_energy_burn = week_avg
        if summary.total_energy_burn is None and week_avg is not None:
            summary.total_energy_burn = round(week_avg * max(summary.days_measured, 1), 1)

        burns = [d.energy_burn for d in window_days if d.energy_burn is not None]
        derived = cls._derived_metrics(
            burns=burns,
            today_active=today_active_out,
            week_avg=week_avg or summary.avg_energy_burn,
        )

        trend_status, trend_meta = cls._compute_trend(burns, days, today)
        health_trend, _ = cls._compute_health_score_trend(window_days)

        anomalies = cls._detect_anomalies(window_days, burns)

        ctx_common = {
            "days_measured": summary.days_measured,
            "missing_days": summary.missing_days,
            "coverage_pct": summary.coverage_pct,
            "highest_day": max(burns) if burns else None,
            "lowest_day": min(burns) if burns else None,
            "today_active": today_active_out,
            "month_resting_total": month_resting_total_out,
        }

        signals = cls._build_signals(
            summary=summary,
            derived=derived,
            week_pct=week_pct,
            month_avg=month_avg,
            prev_month=prev_month,
            week_avg=week_avg or summary.avg_energy_burn,
            today_active=today_active_out,
            month_resting_total=month_resting_total_out,
            trend_status=trend_status,
            trend_meta=trend_meta,
            health_trend=health_trend,
            burns=burns,
            ctx_common=ctx_common,
            stale=stale,
        )

        overall = cls._build_overall(
            summary=summary,
            derived=derived,
            week_pct=week_pct,
            trend_status=trend_status,
            stale=stale,
        )

        level = cls.compute_level(
            today_active_out,
            week_pct,
            stale=stale and today_active_out is None and week_avg is None,
        )

        return EnergySignal(
            summary=summary,
            signals=signals,
            anomalies=anomalies,
            overall=overall,
            level=level,
            month_resting_total=month_resting_total_out,
            total_active_energy=today_active_out,
            current_month_avg=month_avg,
            week_avg=week_avg,
            previous_month_avg=prev_month,
            week_vs_prior_month_pct=week_pct,
            calories_burned_today=today_active_out,
            prev_month_avg=prev_month,
            active_minutes_today=active_minutes,
            workout_duration_min=(
                None if stale else workout_duration_min
            ),
            workout_activity_type=(
                None if stale else workout_activity_type
            ),
            estimated_fields=estimated,
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
    def _extract_summary_data(
        cls, health_params: Dict[str, Any], raw_data: Dict[str, Any]
    ) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {
            "month_resting_total": _safe_float(
                health_param_value(
                    health_params,
                    "health_params_energy_month_resting_total",
                    HealthDataConstants.KEY_ENERGY_MONTH_RESTING_TOTAL,
                )
            ),
            "total_active_energy": _safe_float(
                health_param_value(
                    health_params,
                    "health_params_total_active_energy",
                    HealthDataConstants.KEY_TOTAL_ACTIVE_ENERGY,
                )
            )
            or _safe_float(
                health_param_value(
                    health_params,
                    "health_params_calories_burned_today",
                    HealthDataConstants.KEY_CALORIES_BURNED_TODAY,
                )
            ),
            "current_month_avg": _safe_float(
                health_param_value(
                    health_params,
                    "health_params_energy_current_month_avg",
                    HealthDataConstants.KEY_ENERGY_CURRENT_MONTH_AVG,
                )
            ),
            "week_avg": _safe_float(
                health_param_value(
                    health_params,
                    "health_params_energy_week_avg",
                    HealthDataConstants.KEY_ENERGY_WEEK_AVG,
                )
            ),
            "previous_month_avg": _safe_float(
                health_param_value(
                    health_params,
                    "health_params_energy_previous_month_avg",
                    HealthDataConstants.KEY_ENERGY_PREVIOUS_MONTH_AVG,
                )
            ),
        }

        sd = cls._raw_summary_data(raw_data)
        if not sd:
            return out

        if out["total_active_energy"] is None:
            out["total_active_energy"] = _safe_float(
                sd.get(CanonicalField.ACTIVE_ENERGY_ONE_DAY_KCAL)
                or sd.get(APIResponseKeys.TOTAL_ACTIVE_ENERGY)
                or sd.get("totalActiveEnergy")
            )
        if out["current_month_avg"] is None:
            cur = (
                sd.get(CanonicalField.AVERAGE_DAILY_ACTIVE_ENERGY_BURN_CURRENT_MONTH_KCAL)
                or sd.get(APIResponseKeys.CURRENT_MONTH_AVG)
                or sd.get("currentMonthAvg")
            )
            if not isinstance(cur, dict):
                out["current_month_avg"] = _safe_float(cur)
        if out["week_avg"] is None:
            out["week_avg"] = _safe_float(
                sd.get(CanonicalField.AVERAGE_DAILY_ACTIVE_ENERGY_BURN_CURRENT_WEEK_KCAL)
                or sd.get(APIResponseKeys.AVG_CURRENT_WEEK_ENERGY_BURN)
                or sd.get("avgCurrentWeekEnergyBurn")
            )
        if out["previous_month_avg"] is None:
            prev = (
                sd.get(CanonicalField.AVERAGE_DAILY_ACTIVE_ENERGY_BURN_PREVIOUS_MONTH_KCAL)
                or sd.get(APIResponseKeys.PREVIOUS_MONTH_AVG)
                or sd.get("previousMonthAvg")
            )
            if not isinstance(prev, dict):
                out["previous_month_avg"] = _safe_float(prev)
        return out

    @classmethod
    def _raw_summary_data(cls, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        stats = raw_data.get("today_health_stats") or {}
        if isinstance(stats, dict):
            entry = stats.get("ENERGY") or stats.get("ACTIVE_ENERGY") or {}
            if isinstance(entry, dict):
                sd = entry.get("summaryData") or entry.get("summary_data") or {}
                if isinstance(sd, dict):
                    return sd

        health_data = raw_data.get("health_data") or {}
        block = health_data.get(HealthDataConstants.ENERGY_TYPE) or health_data.get(
            "ENERGY"
        )
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
    def _extract_calories_today(cls, raw_data: Optional[Dict[str, Any]]) -> Optional[float]:
        if not raw_data:
            return None
        stats = raw_data.get("today_health_stats") or {}
        if not isinstance(stats, dict):
            return None
        for key in ("ENERGY", "ACTIVE_ENERGY", "CALORIES"):
            entry = stats.get(key) or {}
            if not isinstance(entry, dict):
                continue
            sd = entry.get("summaryData") or {}
            if isinstance(sd, dict):
                v = sd.get("totalActiveEnergy") or sd.get("total")
                if isinstance(v, (int, float)):
                    return float(v)
            data = entry.get("data") or []
            if data and isinstance(data, list):
                last = data[-1]
                if isinstance(last, dict):
                    for k in ("energyBurn", "total", "value", "calories"):
                        v = last.get(k)
                        if isinstance(v, (int, float)):
                            return float(v)
        return None

    @classmethod
    def _aggregate_days(
        cls,
        *,
        historical_snapshots: List[Any],
        raw_data: Dict[str, Any],
        today: Optional[date],
        today_burn: Optional[float],
    ) -> List[EnergyDay]:
        by_date: Dict[date, EnergyDay] = {}

        for row in historical_snapshots or []:
            d = _snapshot_date(row)
            if d is None:
                continue
            burn = _safe_float(
                _snapshot_field(row, CanonicalField.TOTAL_CALORIES_BURNED_ONE_DAY_KCAL)
            )
            score = _safe_float(_snapshot_field(row, CanonicalField.HEALTH_SCORE_ONE_DAY))
            if burn is not None or score is not None:
                by_date[d] = EnergyDay(day_date=d, energy_burn=burn, health_score=score)

        cls._merge_raw_energy_entries(by_date, raw_data)

        if today is not None and today_burn is not None:
            existing = by_date.get(today) or EnergyDay(day_date=today)
            existing.energy_burn = today_burn
            by_date[today] = existing

        return sorted(by_date.values(), key=lambda x: x.day_date)

    @classmethod
    def _merge_raw_energy_entries(
        cls, by_date: Dict[date, EnergyDay], raw_data: Dict[str, Any]
    ) -> None:
        health_data = raw_data.get("health_data") or {}
        block = health_data.get(HealthDataConstants.ENERGY_TYPE) or health_data.get(
            "ENERGY"
        )
        data = None
        if block is not None:
            data = getattr(block, "data", None)
            if data is None and isinstance(block, dict):
                data = block.get("data")
        stats = raw_data.get("today_health_stats") or {}
        if data is None and isinstance(stats, dict):
            entry = stats.get("ENERGY") or stats.get("ACTIVE_ENERGY") or {}
            if isinstance(entry, dict):
                data = entry.get("data")

        if not data:
            return
        for week in data:
            days = getattr(week, "data", None)
            if days is None and isinstance(week, dict):
                # week may itself be a day entry
                if week.get("date") or week.get("energyBurn"):
                    days = [week]
                else:
                    days = week.get("data")
            if not days:
                continue
            for entry in days:
                if isinstance(entry, dict):
                    date_raw = entry.get("date")
                    burn = entry.get(CanonicalField.ENERGY_BURN_ONE_DAY_KCAL)
                else:
                    date_raw = getattr(entry, "date", None)
                    burn = getattr(entry, CanonicalField.ENERGY_BURN_ONE_DAY_KCAL, None)
                score = None
                if not date_raw:
                    continue
                try:
                    d = date.fromisoformat(str(date_raw).split("T")[0])
                except Exception:
                    continue
                day = by_date.get(d) or EnergyDay(day_date=d)
                if burn is not None:
                    day.energy_burn = _safe_float(burn)
                if score is not None:
                    day.health_score = _safe_float(score)
                by_date[d] = day

    @classmethod
    def _window_days(
        cls,
        days: List[EnergyDay],
        today: Optional[date],
        window_days: int,
    ) -> List[EnergyDay]:
        if today is None:
            return list(days[-window_days:])
        cutoff = today - timedelta(days=window_days)
        return [d for d in days if cutoff < d.day_date <= today]

    # ── summary / derived / trend ───────────────────────────────────────

    @classmethod
    def _build_summary(
        cls,
        days: List[EnergyDay],
        *,
        today_active: Optional[float],
        month_resting_total: Optional[float],
        month_avg: Optional[float],
        week_avg: Optional[float],
        prev_month: Optional[float],
    ) -> EnergySummaryMetrics:
        burns = [d.energy_burn for d in days if d.energy_burn is not None]
        scores = [d.health_score for d in days if d.health_score is not None]
        measured = len(burns)
        window = ENERGY_SUMMARY_WINDOW_DAYS
        missing = max(0, window - measured)
        coverage = round(100.0 * measured / window, 1) if window else None

        return EnergySummaryMetrics(
            avg_energy_burn=round(_mean(burns), 1) if burns else None,
            total_energy_burn=round(sum(burns), 1) if burns else None,
            avg_resting_energy=None,
            avg_health_score=round(_mean(scores), 1) if scores else None,
            days_measured=measured,
            missing_days=missing,
            window_days=window,
            coverage_pct=coverage,
            today_active=today_active,
            month_resting_total=month_resting_total,
            current_month_avg=month_avg,
            week_avg=week_avg,
            previous_month_avg=prev_month,
        )

    @classmethod
    def _derived_metrics(
        cls,
        *,
        burns: List[float],
        today_active: Optional[float],
        week_avg: Optional[float],
    ) -> Dict[str, Any]:
        std = round(_stdev(burns), 1) if burns else None
        avg = _mean(burns)
        cv = None
        if avg and avg > 0 and std is not None:
            cv = round(std / avg, 3)

        consistency_score = None
        if cv is not None:
            # Map CV → 0–100 (0 CV → 100, 0.5 CV → 0)
            consistency_score = round(max(0.0, min(100.0, 100.0 * (1 - cv / 0.5))), 1)

        weekly_total = None
        if burns:
            weekly_total = round(sum(burns), 1)
        elif week_avg is not None:
            weekly_total = round(week_avg * ENERGY_SUMMARY_WINDOW_DAYS, 1)

        return {
            "std_dev": std,
            "cv": cv,
            "consistency_score": consistency_score,
            "weekly_total": weekly_total,
            "daily_average": round(avg, 1) if avg is not None else week_avg,
        }

    @classmethod
    def _compute_trend(
        cls,
        burns: List[float],
        all_days: List[EnergyDay],
        today: Optional[date],
    ) -> Tuple[str, Dict[str, Any]]:
        for days_n in ENERGY_TREND_WINDOWS_DAYS:
            series = burns
            if today is not None:
                wd = cls._window_days(all_days, today, days_n)
                series = [d.energy_burn for d in wd if d.energy_burn is not None]
            if len(series) < ENERGY_MIN_DAYS_FOR_TREND:
                continue
            first, last = series[0], series[-1]
            prior = series[:-1]
            prior_avg = sum(prior) / len(prior) if prior else first
            change = round(last - first, 1)
            if prior_avg <= 0:
                return "stable", {
                    "window": f"{days_n}d",
                    "change": change,
                    "first_day": first,
                    "last_day": last,
                }
            frac = (last - prior_avg) / prior_avg
            if frac >= ENERGY_TREND_SIGNIFICANT_FRAC:
                status = "improving"
            elif frac <= -ENERGY_TREND_SIGNIFICANT_FRAC:
                status = "declining"
            else:
                status = "stable"
            return status, {
                "window": f"{days_n}d",
                "change": change,
                "first_day": first,
                "last_day": last,
                "delta_vs_prior_avg": round(last - prior_avg, 1),
            }
        return "insufficient_data", {}

    @classmethod
    def _compute_health_score_trend(
        cls, days: List[EnergyDay]
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        scores = [d.health_score for d in days if d.health_score is not None]
        if len(scores) < ENERGY_MIN_DAYS_FOR_TREND:
            return None, {}
        first, last = scores[0], scores[-1]
        delta = last - first
        if delta >= 5:
            return "improving", {"change": round(delta, 1)}
        if delta <= -5:
            return "declining", {"change": round(delta, 1)}
        return "stable", {"change": round(delta, 1)}

    # ── anomalies / signals / overall ───────────────────────────────────

    @classmethod
    def _detect_anomalies(
        cls, days: List[EnergyDay], burns: List[float]
    ) -> List[EnergyAnomaly]:
        if not burns or len(burns) < 2:
            return []
        avg = _mean(burns)
        if not avg or avg <= 0:
            return []
        anomalies: List[EnergyAnomaly] = []
        prev: Optional[float] = None
        for d in days:
            if d.energy_burn is None:
                continue
            v = d.energy_burn
            ds = d.day_date.isoformat()
            if v < avg * ENERGY_ANOMALY_LOW_FRAC:
                anomalies.append(
                    EnergyAnomaly(
                        kind="low_activity",
                        day_date=ds,
                        value=v,
                        threshold=round(avg * ENERGY_ANOMALY_LOW_FRAC, 1),
                        evidence={"energy_burn": v, "window_avg": round(avg, 1)},
                    )
                )
            elif v > avg * ENERGY_ANOMALY_HIGH_FRAC:
                anomalies.append(
                    EnergyAnomaly(
                        kind="high_activity",
                        day_date=ds,
                        value=v,
                        threshold=round(avg * ENERGY_ANOMALY_HIGH_FRAC, 1),
                        evidence={"energy_burn": v, "window_avg": round(avg, 1)},
                    )
                )
            if prev is not None and prev > 0:
                ratio = v / prev
                if ratio >= ENERGY_ANOMALY_SPIKE_FRAC:
                    anomalies.append(
                        EnergyAnomaly(
                            kind="sudden_spike",
                            day_date=ds,
                            value=v,
                            threshold=round(prev * ENERGY_ANOMALY_SPIKE_FRAC, 1),
                            evidence={"energy_burn": v, "previous_day": prev},
                        )
                    )
                elif ratio <= ENERGY_ANOMALY_DROP_FRAC:
                    anomalies.append(
                        EnergyAnomaly(
                            kind="sudden_drop",
                            day_date=ds,
                            value=v,
                            threshold=round(prev * ENERGY_ANOMALY_DROP_FRAC, 1),
                            evidence={"energy_burn": v, "previous_day": prev},
                        )
                    )
            prev = v
        return anomalies

    @classmethod
    def _build_signals(cls, **kw: Any) -> EnergySignalsMap:
        summary: EnergySummaryMetrics = kw["summary"]
        derived: Dict[str, Any] = kw["derived"]
        week_pct = kw["week_pct"]
        month_avg = kw["month_avg"]
        prev_month = kw["prev_month"]
        week_avg = kw["week_avg"]
        today_active = kw["today_active"]
        month_resting_total = kw["month_resting_total"]
        trend_status = kw["trend_status"]
        trend_meta: Dict[str, Any] = kw["trend_meta"]
        health_trend = kw["health_trend"]
        burns: List[float] = kw["burns"]
        ctx = kw["ctx_common"]
        stale = kw["stale"]

        daily_avg = derived.get("daily_average") or summary.avg_energy_burn
        weekly_total = derived.get("weekly_total") or summary.total_energy_burn
        vs_month = _pct_delta(daily_avg or today_active, month_avg)
        vs_prev = _pct_delta(daily_avg, prev_month)
        comparison_pct = vs_prev if vs_prev is not None else week_pct

        # activity volume
        if stale and daily_avg is None and today_active is None:
            vol_status = "insufficient_data"
        elif today_active is not None and today_active <= 0 and daily_avg is None:
            vol_status = "not_started"
        elif comparison_pct is not None and comparison_pct <= ENERGY_LEVEL_SEVERE_PCT:
            vol_status = "low"
        elif comparison_pct is not None and comparison_pct <= ENERGY_LEVEL_MILD_PCT:
            vol_status = "fair"
        elif daily_avg is not None or today_active is not None:
            vol_status = "good"
        else:
            vol_status = "insufficient_data"

        activity_volume = _dim(
            vol_status,
            metrics={
                "daily_average": daily_avg,
                "weekly_total": weekly_total,
                "today_active": today_active,
            },
            comparison={
                "vs_current_month_pct": vs_month,
                "vs_previous_month_pct": vs_prev,
                "vs_current_month_abs": (
                    round((daily_avg or today_active or 0) - month_avg, 1)
                    if (daily_avg or today_active) is not None and month_avg is not None
                    else None
                ),
                "vs_previous_month_abs": (
                    round((daily_avg or 0) - prev_month, 1)
                    if daily_avg is not None and prev_month is not None
                    else None
                ),
            },
            trend=trend_status if trend_status != "insufficient_data" else None,
            evidence_context=ctx,
        )

        # consistency
        cons_score = derived.get("consistency_score")
        cv = derived.get("cv")
        if cons_score is None or len(burns) < ENERGY_MIN_DAYS_FOR_CONSISTENCY:
            cons_status = "insufficient_data"
        elif cv is not None and cv <= ENERGY_CONSISTENCY_HIGH_CV:
            cons_status = "high"
        elif cv is not None and cv <= ENERGY_CONSISTENCY_MED_CV:
            cons_status = "medium"
        else:
            cons_status = "low"
        energy_consistency = _dim(
            cons_status,
            metrics={
                "score": cons_score,
                "std_dev": derived.get("std_dev"),
                "cv": cv,
            },
            evidence_context=ctx,
        )

        # trend
        energy_trend = _dim(
            trend_status,
            metrics={
                **trend_meta,
                "health_score_trend": health_trend,
            },
            evidence_context=ctx,
        )

        # baseline
        base_status = _baseline_status(vs_prev if vs_prev is not None else vs_month)
        if week_pct is not None and base_status == "insufficient_data":
            base_status = _baseline_status(week_pct)
        baseline_comparison = _dim(
            base_status,
            metrics={
                "week_avg": week_avg,
                "current_month_avg": month_avg,
                "previous_month_avg": prev_month,
            },
            comparison={
                "vs_current_month_pct": vs_month,
                "vs_previous_month_pct": vs_prev if vs_prev is not None else week_pct,
                "week_vs_prior_month_pct": week_pct,
            },
            evidence_context=ctx,
        )

        # activity pattern
        if len(burns) < ENERGY_MIN_DAYS_FOR_PATTERN:
            pattern_status = "insufficient_data"
        elif trend_status == "improving":
            pattern_status = "increasing_activity"
        elif trend_status == "declining":
            pattern_status = "declining_activity"
        elif cons_status == "high":
            pattern_status = "consistent_activity"
        elif cons_status == "low":
            pattern_status = "fluctuating_activity"
        else:
            pattern_status = "stable_activity"
        activity_pattern = _dim(
            pattern_status,
            metrics={
                "consistency_score": cons_score,
                "std_dev": derived.get("std_dev"),
                "highest_day": ctx.get("highest_day"),
                "lowest_day": ctx.get("lowest_day"),
            },
            trend=trend_status if trend_status != "insufficient_data" else None,
            evidence_context=ctx,
        )

        return EnergySignalsMap(
            activity_volume=activity_volume,
            energy_consistency=energy_consistency,
            energy_trend=energy_trend,
            baseline_comparison=baseline_comparison,
            activity_pattern=activity_pattern,
        )

    @classmethod
    def _build_overall(
        cls,
        *,
        summary: EnergySummaryMetrics,
        derived: Dict[str, Any],
        week_pct: Optional[float],
        trend_status: str,
        stale: bool,
    ) -> EnergyOverall:
        has_data = (
            summary.avg_energy_burn is not None
            or summary.today_active is not None
            or summary.week_avg is not None
            or summary.previous_month_avg is not None
        )
        if stale and not has_data:
            return EnergyOverall(status="insufficient_data", confidence="low")
        if not has_data:
            return EnergyOverall(status="insufficient_data", confidence="low")

        # Volume points from week vs prior
        if week_pct is None:
            vol_pts = 70.0
        elif week_pct >= 0:
            vol_pts = min(100.0, 75.0 + week_pct * 0.5)
        elif week_pct <= ENERGY_LEVEL_SEVERE_PCT:
            vol_pts = 25.0
        elif week_pct <= ENERGY_LEVEL_MODERATE_PCT:
            vol_pts = 45.0
        elif week_pct <= ENERGY_LEVEL_MILD_PCT:
            vol_pts = 60.0
        else:
            vol_pts = 70.0

        cons = derived.get("consistency_score")
        cons_pts = float(cons) if cons is not None else 70.0

        if trend_status == "improving":
            trend_pts = 90.0
        elif trend_status == "stable":
            trend_pts = 75.0
        elif trend_status == "declining":
            trend_pts = 45.0
        else:
            trend_pts = 65.0

        base_pct = week_pct
        if base_pct is None:
            base_pts = 70.0
        elif abs(base_pct) <= ENERGY_VS_BASELINE_NORMAL_PCT:
            base_pts = 85.0
        elif base_pct > 0:
            base_pts = 80.0
        else:
            base_pts = max(30.0, 70.0 + base_pct)

        score = round(
            vol_pts * ENERGY_OVERALL_WEIGHT_VOLUME
            + cons_pts * ENERGY_OVERALL_WEIGHT_CONSISTENCY
            + trend_pts * ENERGY_OVERALL_WEIGHT_TREND
            + base_pts * ENERGY_OVERALL_WEIGHT_BASELINE,
            1,
        )
        if score >= 75:
            status = "good"
        elif score >= 55:
            status = "fair"
        else:
            status = "poor"

        measured = summary.days_measured
        if measured >= 5 or (
            summary.week_avg is not None and summary.previous_month_avg is not None
        ):
            confidence = "high"
        elif measured >= 3 or summary.week_avg is not None:
            confidence = "medium"
        else:
            confidence = "low"

        return EnergyOverall(status=status, score=score, confidence=confidence)
