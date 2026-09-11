"""Steps signal processor — evidence model per step-processor.md.

Pipeline: aggregation → goal → volume → trend → baseline → consistency →
pattern → anomalies → evidence signals (summary / signals / overall).
Legacy flat ActivitySignal fields are filled for older consumers.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from insights.health.canonical_field_mapping import CanonicalField
from insights.health.health_processor.common.utils import (
    health_param_value,
    is_data_stale,
    snapshot_date,
    snapshot_field,
)
from insights.health.health_processor.steps.constants import (
    STEPS_ACTIVE_MAX,
    STEPS_ANOMALY_DROP_FRAC,
    STEPS_ANOMALY_EXTREME_HIGH_ABS,
    STEPS_ANOMALY_EXTREME_HIGH_FRAC,
    STEPS_ANOMALY_SPIKE_FRAC,
    STEPS_ANOMALY_VERY_LOW_ABS,
    STEPS_ANOMALY_VERY_LOW_FRAC,
    STEPS_CONSISTENCY_HIGH_CV,
    STEPS_CONSISTENCY_MED_CV,
    STEPS_GOAL_HIGH_PCT,
    STEPS_GOAL_LOW_PCT,
    STEPS_GOAL_MODERATE_PCT,
    STEPS_LIGHTLY_ACTIVE_MAX,
    STEPS_MIN_DAYS_FOR_CONSISTENCY,
    STEPS_MIN_DAYS_FOR_PATTERN,
    STEPS_MIN_DAYS_FOR_TREND,
    STEPS_OVERALL_WEIGHT_CONSISTENCY,
    STEPS_OVERALL_WEIGHT_GOAL,
    STEPS_OVERALL_WEIGHT_TREND,
    STEPS_OVERALL_WEIGHT_VOLUME,
    STEPS_SEDENTARY_MAX,
    STEPS_SUMMARY_WINDOW_DAYS,
    STEPS_TREND_SIGNIFICANT_FRAC,
    STEPS_VOLUME_GOOD,
    STEPS_VOLUME_HIGH,
    STEPS_VOLUME_MODERATE,
    STEPS_VS_BASELINE_HIGHER_PCT,
    STEPS_VS_BASELINE_LOWER_PCT,
    STEPS_VS_BASELINE_NORMAL_PCT,
)
from insights.insight_config import HealthConfig
from insights.insight_config import HealthLevel
from insights.schemas.processed_context import (
    StepsAnomaly,
    StepsDimensionSignal,
    StepsOverall,
    StepsSignal,
    StepsSignalsMap,
    StepsSummaryMetrics,
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
    f = _safe_float(v)
    if f is None:
        return None
    return int(round(f))


def _mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _stdev(vals: List[float]) -> Optional[float]:
    if len(vals) < 2:
        return None
    return statistics.pstdev(vals)


def _pct_delta(current: Optional[float], ref: Optional[float]) -> Optional[float]:
    if current is None or ref is None or ref == 0:
        return None
    return round((current - ref) / ref * 100.0, 1)


def _dim(
    status: str,
    metrics: Optional[Dict[str, Any]] = None,
    comparison: Optional[Dict[str, Any]] = None,
    trend: Optional[str] = None,
) -> StepsDimensionSignal:
    metrics = metrics or {}
    comparison = comparison or {}
    return StepsDimensionSignal(
        status=status,
        metrics=metrics,
        comparison=comparison,
        trend=trend,
    )


def _classify_steps(steps: float) -> str:
    if steps <= STEPS_SEDENTARY_MAX:
        return "sedentary"
    if steps <= STEPS_LIGHTLY_ACTIVE_MAX:
        return "lightly_active"
    if steps <= STEPS_ACTIVE_MAX:
        return "active"
    return "highly_active"


def _volume_status(avg_steps: Optional[float]) -> str:
    if avg_steps is None:
        return "insufficient_data"
    if avg_steps >= STEPS_VOLUME_HIGH:
        return "high"
    if avg_steps >= STEPS_VOLUME_GOOD:
        return "good"
    if avg_steps >= STEPS_VOLUME_MODERATE:
        return "moderate"
    return "low"


def _goal_status(completion_pct: Optional[float]) -> str:
    if completion_pct is None:
        return "insufficient_data"
    if completion_pct >= STEPS_GOAL_HIGH_PCT:
        return "high"
    if completion_pct >= STEPS_GOAL_MODERATE_PCT:
        return "moderate"
    if completion_pct >= STEPS_GOAL_LOW_PCT:
        return "low"
    return "very_low"


def _baseline_status(pct: Optional[float]) -> str:
    if pct is None:
        return "insufficient_data"
    if abs(pct) <= STEPS_VS_BASELINE_NORMAL_PCT:
        return "normal"
    if pct >= STEPS_VS_BASELINE_HIGHER_PCT:
        return "higher"
    if pct <= STEPS_VS_BASELINE_LOWER_PCT:
        return "lower"
    return "slightly_off"


def _consistency_status(cv: Optional[float]) -> Tuple[str, Optional[float]]:
    if cv is None:
        return "insufficient_data", None
    if cv <= STEPS_CONSISTENCY_HIGH_CV:
        score = round(max(0.0, 100.0 - cv * 100.0), 1)
        return "high", score
    if cv <= STEPS_CONSISTENCY_MED_CV:
        score = round(max(0.0, 85.0 - (cv - STEPS_CONSISTENCY_HIGH_CV) * 200.0), 1)
        return "moderate", score
    score = round(max(0.0, 60.0 - (cv - STEPS_CONSISTENCY_MED_CV) * 100.0), 1)
    return "low", score


@dataclass
class StepsDay:
    day_date: date
    steps: Optional[float] = None
    distance: Optional[float] = None
    health_score: Optional[float] = None
    is_partial: bool = False


class StepsSignalProcessor:
    """Transform raw step inputs into evidence-rich StepsSignal."""

    @classmethod
    def process(
        cls,
        health_params: Dict[str, Any],
        time_data: Dict[str, Any],
        staleness_info: Optional[Dict[str, Any]] = None,
        historical_snapshots: Optional[List[Any]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> StepsSignal:
        try:
            return cls._process_unsafe(
                health_params=health_params or {},
                time_data=time_data or {},
                staleness_info=staleness_info or {},
                historical_snapshots=historical_snapshots or [],
                raw_data=raw_data or {},
            )
        except Exception:
            try:
                from utils.logger import logger as _logger

                _logger.warning("StepsSignalProcessor failed", exc_info=True)
            except Exception:
                pass
            return StepsSignal(
                level=HealthLevel.NONE,
                overall=StepsOverall(status="insufficient_data", confidence="low"),
            )

    @classmethod
    def _process_unsafe(
        cls,
        health_params: Dict[str, Any],
        time_data: Dict[str, Any],
        staleness_info: Dict[str, Any],
        historical_snapshots: List[Any],
        raw_data: Dict[str, Any],
    ) -> StepsSignal:
        today = cls._resolve_today(time_data)
        steps_staleness = staleness_info.get("steps_data_staleness_days")
        stale = is_data_stale(steps_staleness)
        steps_today_val = health_param_value(
            health_params,
            "health_params_steps_today",
            HealthDataConstants.KEY_STEPS_TODAY,
        )
        if stale and steps_today_val is not None:
            stale = False
        estimated: List[str] = []

        api = cls._extract_summary_data(health_params, raw_data)
        daily_target = api.get("daily_target")
        if daily_target is None:
            daily_target = _safe_float(
                health_param_value(
                    health_params,
                    "health_params_steps_goal",
                    HealthDataConstants.KEY_STEPS_GOAL,
                )
            )
        month_avg = api.get("current_month_avg")
        prev_month = api.get("previous_month_avg")

        today_steps = _safe_float(
            health_param_value(
                health_params,
                "health_params_steps_today",
                HealthDataConstants.KEY_STEPS_TODAY,
            )
        )
        if today_steps is None:
            today_steps = api.get("today_steps")
        today_distance = api.get("today_distance")

        if stale:
            today_steps_out = None
            today_distance_out = None
        else:
            today_steps_out = today_steps
            today_distance_out = today_distance

        days = cls._aggregate_days(
            historical_snapshots=historical_snapshots,
            raw_data=raw_data,
            today=today,
            today_steps=today_steps_out,
            today_distance=today_distance_out,
        )
        # Exclude partial today from trend/consistency windows
        complete_days = [d for d in days if not d.is_partial]
        window_days = cls._window_days(complete_days, today, STEPS_SUMMARY_WINDOW_DAYS)
        # Summary includes complete window; today kept separately
        summary = cls._build_summary(
            window_days,
            today_steps=today_steps_out,
            today_distance=today_distance_out,
            daily_target=daily_target,
            month_avg=month_avg,
            prev_month=prev_month,
        )

        step_vals = [d.steps for d in window_days if d.steps is not None]
        trend_status, trend_meta = cls._compute_trend(complete_days, today)
        anomalies = cls._detect_anomalies(
            window_days=window_days,
            all_days=days,
            step_vals=step_vals,
            today=today,
            today_steps=today_steps_out,
            stale=stale,
        )

        # Single shared context — hoisted to StepsSignal.common_context
        common_context = {
            "days_measured": summary.days_measured,
            "missing_days": summary.missing_days,
            "coverage_pct": summary.coverage_pct,
            "highest_day": max(step_vals) if step_vals else None,
            "lowest_day": min(step_vals) if step_vals else None,
            "today_steps": today_steps_out,
            "daily_target": daily_target,
        }

        signals = cls._build_signals(
            summary=summary,
            step_vals=step_vals,
            window_days=window_days,
            trend_status=trend_status,
            trend_meta=trend_meta,
            month_avg=month_avg,
            prev_month=prev_month,
            daily_target=daily_target,
            stale=stale,
        )

        overall = cls._build_overall(
            summary=summary,
            signals=signals,
            trend_status=trend_status,
            stale=stale,
        )

        # ── Legacy flat fields ──────────────────────────────────────────
        goal_i = _safe_int(daily_target)
        today_i = _safe_int(today_steps_out)
        streak = health_param_value(
            health_params,
            "health_params_steps_streak",
            HealthDataConstants.KEY_STEPS_STREAK,
        )
        if streak is None:
            streak = 0
        try:
            streak_i = int(streak) if streak is not None else 0
        except (TypeError, ValueError):
            streak_i = 0

        if stale or today_i is None:
            remaining = None
            pace_ratio = None
            level = HealthLevel.NONE
        else:
            remaining = None if goal_i is None else max(0, goal_i - today_i)
            pace_ratio = (today_i / goal_i) if goal_i and goal_i > 0 else 0.0
            if pace_ratio >= HealthConfig.STEPS_PACE_OK:
                level = HealthLevel.OK
            elif pace_ratio >= HealthConfig.STEPS_PACE_MILD:
                level = HealthLevel.MILD
            elif pace_ratio >= HealthConfig.STEPS_PACE_MODERATE:
                level = HealthLevel.MODERATE
            else:
                level = HealthLevel.SEVERE

        active_hours_left = cls._active_hours_left(time_data)

        workout_duration_min = 0
        _workout_duration_sec = health_param_value(
            health_params,
            "health_params_workout_duration",
            HealthDataConstants.KEY_WORKOUT_DURATION,
        )
        if _workout_duration_sec is None:
            _workout_duration_sec = 0
        try:
            if _workout_duration_sec:
                workout_duration_min = round(float(_workout_duration_sec) / 60, 1)
        except (TypeError, ValueError):
            workout_duration_min = 0

        daily_health_score_trend = health_param_value(
            health_params,
            "health_params_daily_health_score_trend",
            HealthDataConstants.KEY_DAILY_HEALTH_SCORE_TREND,
        )
        if not daily_health_score_trend:
            daily_health_score_trend = cls._fallback_daily_health_score_trend(
                historical_snapshots
            )
            if daily_health_score_trend is not None:
                estimated.append("daily_health_score_trend")

        energy_staleness = staleness_info.get("energy_data_staleness_days")
        calories_burned_today: Optional[float] = None
        if not is_data_stale(energy_staleness):
            hp_cal = health_param_value(
                health_params,
                "health_params_calories_burned_today",
                HealthDataConstants.KEY_CALORIES_BURNED_TODAY,
            )
            if isinstance(hp_cal, (int, float)):
                calories_burned_today = float(hp_cal)
            else:
                calories_burned_today = cls._extract_calories_today(raw_data)
                if calories_burned_today is not None:
                    estimated.append("calories_burned_today")

        return StepsSignal(
            summary=summary,
            signals=signals,
            anomalies=anomalies,
            overall=overall,
            common_context=common_context,
            level=level,
            steps_today=today_i,
            steps_goal=goal_i,
            remaining_steps=remaining,
            pace_ratio=round(pace_ratio, 2) if pace_ratio is not None else None,
            active_hours_left=active_hours_left,
            steps_streak=streak_i,
            workout_activity_type=health_param_value(
                health_params,
                "health_params_workout_activity_type",
                HealthDataConstants.KEY_WORKOUT_ACTIVITY_TYPE,
            ),
            workout_duration_min=int(workout_duration_min)
            if isinstance(workout_duration_min, float)
            else workout_duration_min,
            calories_burned_today=calories_burned_today,
            daily_health_score_trend=daily_health_score_trend,
            estimated_fields=estimated,
        )

    # ── time / API helpers ──────────────────────────────────────────────

    @classmethod
    def _resolve_today(cls, time_data: Dict[str, Any]) -> Optional[date]:
        raw = time_data.get("time_data_current_time_iso") or time_data.get("time_data_current_date")
        if not raw:
            return None
        try:
            return date.fromisoformat(str(raw).split("T")[0])
        except Exception:
            return None

    @classmethod
    def _active_hours_left(cls, time_data: Dict[str, Any]) -> Optional[float]:
        current_hour_str = time_data.get("time_data_current_hour")
        try:
            current_hour = (
                float(current_hour_str) if current_hour_str is not None else None
            )
        except (ValueError, TypeError):
            current_hour = None
        active_end_str = time_data.get("time_data_active_end_str") or ""
        try:
            eh, em = active_end_str.split(":")
            active_end = int(eh) + int(em) / 60
        except Exception:
            active_end = None
        if current_hour is None or active_end is None:
            return None
        return max(0.0, round(active_end - current_hour, 1))

    @classmethod
    def _extract_summary_data(
        cls, health_params: Dict[str, Any], raw_data: Dict[str, Any]
    ) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {
            "daily_target": _safe_float(
                health_param_value(
                    health_params,
                    "health_params_steps_goal",
                    HealthDataConstants.KEY_STEPS_GOAL,
                )
            ),
            "current_month_avg": None,
            "previous_month_avg": None,
            "today_steps": _safe_float(
                health_param_value(
                    health_params,
                    "health_params_steps_today",
                    HealthDataConstants.KEY_STEPS_TODAY,
                )
            ),
            "today_distance": None,
        }
        sd = cls._raw_summary_data(raw_data)
        if not sd:
            return out

        if out["daily_target"] is None:
            # Backend (camelCase) → primary; legacy snake_case as fallback
            out["daily_target"] = _safe_float(
                sd.get("dailyTarget") or sd.get("daily_steps_target")
            )

        cur = sd.get("currentMonthAvg")
        if cur is None:
            cur = sd.get(CanonicalField.AVERAGE_DAILY_STEPS_CURRENT_MONTH)
        if isinstance(cur, dict):
            out["current_month_avg"] = _safe_float(cur.get("steps"))
        else:
            out["current_month_avg"] = _safe_float(cur)

        prev = sd.get("previousMonthAvg")
        if prev is None:
            prev = sd.get(CanonicalField.AVERAGE_DAILY_STEPS_PREVIOUS_MONTH)
        if isinstance(prev, dict):
            out["previous_month_avg"] = _safe_float(prev.get("steps"))
        else:
            out["previous_month_avg"] = _safe_float(prev)

        stats = raw_data.get("today_health_stats") or {}
        steps_entry = stats.get("STEPS") or stats.get("steps") or {}
        if isinstance(steps_entry, dict):
            data_rows = steps_entry.get("data") or []
            today_iso = None
            try:
                today_iso = date.today().isoformat()
            except Exception:
                today_iso = None
            for row in data_rows:
                if not isinstance(row, dict):
                    continue
                raw_d = row.get("date") or row.get("day") or row.get("daily_snapshot_date")
                if not raw_d:
                    continue
                row_iso = str(raw_d).split("T", 1)[0]
                if today_iso and row_iso != today_iso:
                    continue
                # Backend (camelCase) → primary; legacy snake_case as fallback
                d_val = _safe_float(row.get("distance") or row.get(CanonicalField.DISTANCE_ONE_DAY))
                if d_val is not None:
                    out["today_distance"] = d_val
                    break
            if out["today_distance"] is None:
                # Fallback: same key priority
                out["today_distance"] = _safe_float(
                    sd.get("distanceForOneDay") or sd.get(CanonicalField.DISTANCE_ONE_DAY)
                )
        return out

    @classmethod
    def _raw_summary_data(cls, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        stats = raw_data.get("today_health_stats") or {}
        if isinstance(stats, dict):
            entry = stats.get("STEPS") or stats.get("steps") or {}
            if isinstance(entry, dict):
                sd = entry.get("summaryData") or entry.get("summary_data") or {}
                if isinstance(sd, dict):
                    return sd
        health_data = raw_data.get("health_data") or {}
        block = health_data.get("STEPS") or health_data.get("steps")
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
    def _extract_calories_today(cls, raw_data: Dict[str, Any]) -> Optional[float]:
        stats = raw_data.get("today_health_stats") or {}
        if not isinstance(stats, dict):
            return None
        entry = stats.get("ENERGY") or stats.get("ACTIVE_ENERGY") or {}
        if not isinstance(entry, dict):
            return None
        sd = entry.get("summaryData") or entry.get("summary_data") or {}
        if isinstance(sd, dict):
            v = sd.get(CanonicalField.ACTIVE_ENERGY_ONE_DAY_KCAL)
            if isinstance(v, (int, float)):
                return float(v)
        return None

    @classmethod
    def _fallback_daily_health_score_trend(
        cls, snapshots, n: int = 7
    ) -> Optional[List[int]]:
        pairs = []
        for row in snapshots or []:
            d = snapshot_date(row)
            if d is None:
                continue
            score = snapshot_field(row, CanonicalField.HEALTH_SCORE_ONE_DAY)
            if score is None:
                continue
            try:
                pairs.append((d, int(score)))
            except (TypeError, ValueError):
                continue
        if not pairs:
            return None
        pairs.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in pairs[:n]]

    # ── aggregation ─────────────────────────────────────────────────────

    @classmethod
    def _aggregate_days(
        cls,
        *,
        historical_snapshots: List[Any],
        raw_data: Dict[str, Any],
        today: Optional[date],
        today_steps: Optional[float],
        today_distance: Optional[float],
    ) -> List[StepsDay]:
        by_date: Dict[date, StepsDay] = {}
        daily_health_score_by_date: Dict[date, float] = {}

        # healthScore is collected from balance_scores_30d/7d below
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

        for row in historical_snapshots or []:
            d = snapshot_date(row)
            if d is None:
                continue
            steps = _safe_float(
                snapshot_field(row, CanonicalField.STEP_COUNT_ONE_DAY)
            )
            # Distance is not stored on snapshot rows; it lives in
            # today_health_stats["STEPS"]["data"][].distance (canonical
            # distance_one_day) and is merged below in the STEPS.data block.
            distance = None
            _score = (
                snapshot_field(row, CanonicalField.HEALTH_SCORE_ONE_DAY)
                or snapshot_field(row, "healthScore")
                or snapshot_field(row, "health_score")
                or snapshot_field(row, "h_health_score")
            )
            score = _safe_float(_score)
            if score is None:
                score = daily_health_score_by_date.get(d)
            if steps is not None or score is not None:
                by_date[d] = StepsDay(
                    day_date=d,
                    steps=steps,
                    distance=distance,
                    health_score=score,
                )

        stats = raw_data.get("today_health_stats") or {}
        entry = stats.get("STEPS") or stats.get("steps") or {}
        if hasattr(entry, "model_dump") and not isinstance(entry, dict):
            try:
                entry = entry.model_dump(by_alias=True, mode="json")
            except Exception:
                entry = {}
        if isinstance(entry, dict):
            rows = entry.get("data") or []
            for row in rows:
                # Normalize Pydantic StepsDataEntry → dict so .get() works
                if hasattr(row, "model_dump") and not isinstance(row, dict):
                    try:
                        row = row.model_dump(by_alias=True, mode="json")
                    except Exception:
                        continue
                if not isinstance(row, dict):
                    continue
                raw_d = row.get("date") or row.get("day") or row.get("daily_snapshot_date")
                if not raw_d:
                    continue
                try:
                    d = date.fromisoformat(str(raw_d).split("T")[0])
                except Exception:
                    continue
                # Backend (camelCase) → primary; legacy snake_case as fallback
                steps = _safe_float(
                    row.get("steps") or row.get(CanonicalField.STEP_COUNT_ONE_DAY)
                )
                distance = _safe_float(
                    row.get("distance") or row.get(CanonicalField.DISTANCE_ONE_DAY)
                )
                _score = (
                    row.get("healthScore")
                    or row.get(CanonicalField.HEALTH_SCORE_ONE_DAY)
                    or row.get("health_score")
                    or row.get("h_health_score")
                )
                score = _safe_float(_score)
                # STEPS API rows lack per-day healthScore; fall back to the
                # balance_score history keyed by date so avg_health_score is
                # non-null whenever any daily balance data exists.
                if score is None:
                    score = daily_health_score_by_date.get(d)
                existing = by_date.get(d)
                if existing is None:
                    by_date[d] = StepsDay(
                        day_date=d,
                        steps=steps,
                        distance=distance,
                        health_score=score,
                    )
                else:
                    # Prefer STEPS API rows over snapshot when both present
                    if steps is not None:
                        existing.steps = steps
                    if distance is not None:
                        existing.distance = distance
                    if score is not None:
                        existing.health_score = score

        if today is not None and today_steps is not None:
            existing = by_date.get(today)
            if existing is None:
                by_date[today] = StepsDay(
                    day_date=today,
                    steps=today_steps,
                    distance=today_distance,
                    is_partial=True,
                )
            else:
                existing.steps = today_steps
                if today_distance is not None:
                    existing.distance = today_distance
                existing.is_partial = True

        return sorted(by_date.values(), key=lambda x: x.day_date)

    @classmethod
    def _window_days(
        cls, days: List[StepsDay], today: Optional[date], window: int
    ) -> List[StepsDay]:
        if not days:
            return []
        end = today or days[-1].day_date
        start = end - timedelta(days=window - 1)
        return [d for d in days if start <= d.day_date <= end and not d.is_partial]

    @classmethod
    def _build_summary(
        cls,
        window_days: List[StepsDay],
        *,
        today_steps: Optional[float],
        today_distance: Optional[float],
        daily_target: Optional[float],
        month_avg: Optional[float],
        prev_month: Optional[float],
    ) -> StepsSummaryMetrics:
        steps = [d.steps for d in window_days if d.steps is not None]
        distances = [d.distance for d in window_days if d.distance is not None]
        scores = [d.health_score for d in window_days if d.health_score is not None]
        days_measured = len(steps)
        days_with_data = days_measured + (1 if today_steps is not None else 0)
        missing = max(0, STEPS_SUMMARY_WINDOW_DAYS - days_with_data)
        coverage = (
            round(100.0 * days_with_data / STEPS_SUMMARY_WINDOW_DAYS, 1)
            if STEPS_SUMMARY_WINDOW_DAYS
            else None
        )
        avg_s = _mean(steps)
        avg_d = _mean(distances)
        return StepsSummaryMetrics(
            avg_steps=round(avg_s, 1) if avg_s is not None else None,
            total_steps=round(sum(steps), 1) if steps else None,
            avg_distance=round(avg_d, 2) if avg_d is not None else None,
            total_distance=round(sum(distances), 2) if distances else None,
            avg_health_score=round(_mean(scores), 1) if scores else None,
            days_measured=days_measured,
            days_with_data=days_with_data,
            missing_days=missing,
            window_days=STEPS_SUMMARY_WINDOW_DAYS,
            coverage_pct=coverage,
            today_steps=today_steps,
            today_distance=today_distance,
            daily_target=daily_target,
            current_month_avg=month_avg,
            previous_month_avg=prev_month,
        )

    # ── trend / anomalies / signals ─────────────────────────────────────

    @classmethod
    def _compute_trend(
        cls, complete_days: List[StepsDay], today: Optional[date]
    ) -> Tuple[str, Dict[str, Any]]:
        """Exclude partial current day; compare first→last complete day in window."""
        end = today or (complete_days[-1].day_date if complete_days else None)
        if end is None:
            return "insufficient_data", {}
        start = end - timedelta(days=STEPS_SUMMARY_WINDOW_DAYS - 1)
        series = [
            d
            for d in complete_days
            if start <= d.day_date <= end and d.steps is not None and not d.is_partial
        ]
        if len(series) < STEPS_MIN_DAYS_FOR_TREND:
            return "insufficient_data", {"days": len(series)}
        first = series[0].steps
        last = series[-1].steps
        assert first is not None and last is not None
        change = last - first
        frac = change / first if first else 0.0
        if abs(frac) < STEPS_TREND_SIGNIFICANT_FRAC:
            direction = "stable"
        elif frac > 0:
            direction = "improving"
        else:
            direction = "declining"
        return direction, {
            "first_day": round(first, 1),
            "last_complete_day": round(last, 1),
            "change": round(change, 1),
            "days": len(series),
        }

    @classmethod
    def _detect_anomalies(
        cls,
        *,
        window_days: List[StepsDay],
        all_days: List[StepsDay],
        step_vals: List[float],
        today: Optional[date],
        today_steps: Optional[float],
        stale: bool,
    ) -> List[StepsAnomaly]:
        out: List[StepsAnomaly] = []
        mean_s = _mean(step_vals)

        for d in window_days:
            if d.steps is None:
                continue
            # Absolute extremes
            if d.steps <= STEPS_ANOMALY_VERY_LOW_ABS:
                out.append(
                    StepsAnomaly(
                        kind="very_low_activity",
                        day_date=d.day_date.isoformat(),
                        value=d.steps,
                        threshold=float(STEPS_ANOMALY_VERY_LOW_ABS),
                        evidence={"reason": "below absolute low threshold"},
                    )
                )
            if d.steps >= STEPS_ANOMALY_EXTREME_HIGH_ABS:
                out.append(
                    StepsAnomaly(
                        kind="extremely_high_activity",
                        day_date=d.day_date.isoformat(),
                        value=d.steps,
                        threshold=float(STEPS_ANOMALY_EXTREME_HIGH_ABS),
                        evidence={"reason": "above absolute high threshold"},
                    )
                )
            if mean_s and mean_s > 0:
                ratio = d.steps / mean_s
                if ratio <= STEPS_ANOMALY_DROP_FRAC:
                    out.append(
                        StepsAnomaly(
                            kind="sudden_decrease",
                            day_date=d.day_date.isoformat(),
                            value=d.steps,
                            threshold=round(mean_s * STEPS_ANOMALY_DROP_FRAC, 1),
                            evidence={"vs_mean_ratio": round(ratio, 2), "mean": mean_s},
                        )
                    )
                elif ratio >= STEPS_ANOMALY_SPIKE_FRAC:
                    kind = (
                        "extremely_high_activity"
                        if ratio >= STEPS_ANOMALY_EXTREME_HIGH_FRAC
                        else "sudden_increase"
                    )
                    out.append(
                        StepsAnomaly(
                            kind=kind,
                            day_date=d.day_date.isoformat(),
                            value=d.steps,
                            threshold=round(mean_s * STEPS_ANOMALY_SPIKE_FRAC, 1),
                            evidence={"vs_mean_ratio": round(ratio, 2), "mean": mean_s},
                        )
                    )
                elif ratio <= STEPS_ANOMALY_VERY_LOW_FRAC:
                    out.append(
                        StepsAnomaly(
                            kind="very_low_activity",
                            day_date=d.day_date.isoformat(),
                            value=d.steps,
                            threshold=round(mean_s * STEPS_ANOMALY_VERY_LOW_FRAC, 1),
                            evidence={"vs_mean_ratio": round(ratio, 2), "mean": mean_s},
                        )
                    )

        # Partial-day (current day incomplete)
        if not stale and today is not None and today_steps is not None:
            partial = next(
                (d for d in all_days if d.day_date == today and d.is_partial), None
            )
            if partial is not None:
                out.append(
                    StepsAnomaly(
                        kind="partial_day",
                        day_date=today.isoformat(),
                        value=today_steps,
                        evidence={"reason": "Current day incomplete"},
                    )
                )

        # Dedupe by (kind, day)
        seen = set()
        deduped: List[StepsAnomaly] = []
        for a in out:
            key = (a.kind, a.day_date)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(a)
        return deduped

    @classmethod
    def _build_signals(
        cls,
        *,
        summary: StepsSummaryMetrics,
        step_vals: List[float],
        window_days: List[StepsDay],
        trend_status: str,
        trend_meta: Dict[str, Any],
        month_avg: Optional[float],
        prev_month: Optional[float],
        daily_target: Optional[float],
        stale: bool,
    ) -> StepsSignalsMap:
        avg = summary.avg_steps
        vol_status = _volume_status(avg)
        # activity_volume: no metrics here (avg_steps, total_steps, avg_distance,
        # total_distance all live in summary). Status alone carries the verdict.
        activity_volume = _dim(status=vol_status)

        # Goal achievement
        completion = None
        goal_days = 0
        if daily_target and daily_target > 0:
            if avg is not None:
                completion = round(100.0 * avg / daily_target, 1)
            for d in window_days:
                if d.steps is not None and d.steps >= daily_target:
                    goal_days += 1
        goal_diff = None
        if avg is not None and daily_target is not None:
            goal_diff = round(avg - daily_target, 1)
        # completion_rate + goal_days are unique to this signal
        goal_achievement = _dim(
            status=_goal_status(completion),
            metrics={
                "completion_rate": completion,
                "goal_days": goal_days,
            },
            comparison={"goal_difference": goal_diff},
        )

        # Consistency — std/cv/score are all unique
        std = _stdev(step_vals)
        cv = None
        if std is not None and avg and avg > 0 and len(step_vals) >= STEPS_MIN_DAYS_FOR_CONSISTENCY:
            cv = std / avg
        c_status, c_score = _consistency_status(cv)
        if len(step_vals) < STEPS_MIN_DAYS_FOR_CONSISTENCY:
            c_status, c_score = "insufficient_data", None
        activity_consistency = _dim(
            status=c_status,
            metrics={
                "std_deviation": round(std, 1) if std is not None else None,
                "coefficient_of_variation": round(cv, 3) if cv is not None else None,
                "consistency_score": c_score,
            },
        )

        # Trend — trend_meta already contains only unique trend fields
        step_trend = _dim(
            status=trend_status,
            metrics=trend_meta,
            trend=trend_status if trend_status != "insufficient_data" else None,
        )

        # Baseline — month_avg/prev_month are unique; average_steps lives in summary
        vs_cur = None
        vs_prev = None
        if avg is not None:
            if month_avg is not None:
                vs_cur = round(avg - month_avg, 1)
            if prev_month is not None:
                vs_prev = round(avg - prev_month, 1)
        pct_vs_month = _pct_delta(avg, month_avg)
        pct_vs_prev = _pct_delta(avg, prev_month)
        # Prefer previous month for qualitative when present, else current month
        base_pct = pct_vs_prev if pct_vs_prev is not None else pct_vs_month
        baseline_comparison = _dim(
            status=_baseline_status(base_pct),
            metrics={
                "current_month_avg": month_avg,
                "previous_month_avg": prev_month,
            },
            comparison={
                "vs_current_month": vs_cur,
                "vs_previous_month": vs_prev,
                "vs_current_month_pct": pct_vs_month,
                "vs_previous_month_pct": pct_vs_prev,
            },
        )

        # Pattern classification — band counts are unique
        sedentary_days = lightly = active = highly = 0
        if len(window_days) >= STEPS_MIN_DAYS_FOR_PATTERN:
            for d in window_days:
                if d.steps is None:
                    continue
                band = _classify_steps(d.steps)
                if band == "sedentary":
                    sedentary_days += 1
                elif band == "lightly_active":
                    lightly += 1
                elif band == "active":
                    active += 1
                else:
                    highly += 1
            # Dominant band
            counts = {
                "sedentary": sedentary_days,
                "lightly_active": lightly,
                "active": active,
                "highly_active": highly,
            }
            pattern_status = max(counts, key=counts.get) if any(counts.values()) else "insufficient_data"
        else:
            pattern_status = "insufficient_data"
        # Prefer avg-based label when pattern days thin but avg known
        if pattern_status == "insufficient_data" and avg is not None:
            pattern_status = _classify_steps(avg)

        # Pattern band counts are unique to this signal
        activity_pattern = _dim(
            status=pattern_status,
            metrics={
                "active_days": active + highly,
                "light_days": lightly,
                "sedentary_days": sedentary_days,
                "highly_active_days": highly,
            },
        )

        if stale and summary.days_measured == 0:
            # Soft-mark volumes when completely stale with no history
            for sig in (
                activity_volume,
                goal_achievement,
                activity_consistency,
                step_trend,
                baseline_comparison,
                activity_pattern,
            ):
                if sig.status not in ("insufficient_data",):
                    pass

        return StepsSignalsMap(
            activity_volume=activity_volume,
            goal_achievement=goal_achievement,
            activity_consistency=activity_consistency,
            step_trend=step_trend,
            baseline_comparison=baseline_comparison,
            activity_pattern=activity_pattern,
        )

    @classmethod
    def _build_overall(
        cls,
        *,
        summary: StepsSummaryMetrics,
        signals: StepsSignalsMap,
        trend_status: str,
        stale: bool,
    ) -> StepsOverall:
        if summary.days_measured == 0 and summary.today_steps is None:
            return StepsOverall(status="insufficient_data", confidence="low")

        avg = summary.avg_steps or summary.today_steps
        status = _classify_steps(avg) if avg is not None else "insufficient_data"
        # Map "moderate" language from doc overall example when mid-band
        if status == "lightly_active":
            display = "moderate"
        else:
            display = status

        # Score 0–100
        volume_score = 0.0
        if avg is not None:
            volume_score = min(100.0, 100.0 * avg / max(STEPS_VOLUME_HIGH, 1))

        goal_score = 50.0
        ga = signals.goal_achievement
        if ga and ga.metrics.get("completion_rate") is not None:
            goal_score = min(100.0, float(ga.metrics["completion_rate"]))

        consistency_score = 50.0
        ac = signals.activity_consistency
        if ac and ac.metrics.get("consistency_score") is not None:
            consistency_score = float(ac.metrics["consistency_score"])

        trend_score = 50.0
        if trend_status == "improving":
            trend_score = 80.0
        elif trend_status == "declining":
            trend_score = 30.0
        elif trend_status == "stable":
            trend_score = 65.0

        score = round(
            volume_score * STEPS_OVERALL_WEIGHT_VOLUME
            + goal_score * STEPS_OVERALL_WEIGHT_GOAL
            + consistency_score * STEPS_OVERALL_WEIGHT_CONSISTENCY
            + trend_score * STEPS_OVERALL_WEIGHT_TREND,
            1,
        )

        if summary.days_measured >= 5:
            confidence = "high"
        elif summary.days_measured >= 3:
            confidence = "medium"
        else:
            confidence = "low"
        if stale and summary.today_steps is None:
            confidence = "low"

        return StepsOverall(status=display, score=score, confidence=confidence)
