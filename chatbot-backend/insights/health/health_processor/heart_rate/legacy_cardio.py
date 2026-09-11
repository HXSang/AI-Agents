"""Legacy cardio_stress mirror (HR flat fields + mood mirrors for older consumers)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from insights.health.canonical_field_mapping import CanonicalField
from insights.health.health_processor.common.utils import (
    health_param_value,
    is_data_stale,
    safe_get,
    snapshot_field,
)
from insights.schemas.processed_context import CardioAndStressSignal
from services.executor.constant import HealthDataConstants


def _fallback_walking_heart_rate_avg(snapshots) -> Optional[float]:
    vals = []
    for row in snapshots or []:
        v = snapshot_field(row, CanonicalField.AVERAGE_HEART_RATE_ONE_DAY_BPM)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return round(sum(vals) / len(vals), 1)


def _fallback_hrv_score_from_resting_hr(
    resting_heart_rate: Optional[float],
) -> Optional[int]:
    if resting_heart_rate is None or resting_heart_rate <= 0:
        return None
    try:
        score = round(6000 / float(resting_heart_rate))
    except (TypeError, ValueError):
        return None
    return max(1, min(150, score))


def _fallback_mood_trend_7d(snapshots, n: int = 7) -> Optional[str]:

    del snapshots, n
    return None


class LegacyCardioStressProcessor:
    """Build CardioAndStressSignal for legacy mirrors (pre five-focus consumers)."""

    @classmethod
    def process(
        cls,
        health_params: Dict[str, Any],
        staleness_info: Optional[Dict[str, Any]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
        historical_snapshots: Optional[List[Any]] = None,
    ) -> CardioAndStressSignal:
        try:
            return cls._process_unsafe(
                health_params,
                staleness_info,
                raw_data,
                historical_snapshots,
            )
        except Exception:
            try:
                from utils.logger import logger as _logger

                _logger.warning(
                    "LegacyCardioStressProcessor failed; returning empty signal",
                    exc_info=True,
                )
            except Exception:
                pass
            return CardioAndStressSignal(stress_high=False)

    @classmethod
    def _process_unsafe(
        cls,
        health_params: Dict[str, Any],
        staleness_info: Optional[Dict[str, Any]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
        historical_snapshots: Optional[List[Any]] = None,
    ) -> CardioAndStressSignal:
        hr_staleness = (
            safe_get(staleness_info, "hr_data_staleness_days") if staleness_info else None
        )
        hr_is_stale = is_data_stale(hr_staleness)
        # health_params may carry today's HR from a live call even when
        # snapshots are stale. If so, treat the signal as fresh.
        if hr_is_stale and (
            health_param_value(
                health_params,
                "health_params_resting_heart_rate",
                HealthDataConstants.KEY_RESTING_HEART_RATE,
            )
            is not None
            or health_param_value(
                health_params,
                "health_params_latest_heart_rate",
                HealthDataConstants.KEY_LATEST_HEART_RATE,
            )
            is not None
            or health_param_value(
                health_params,
                "health_params_hrv_score",
                HealthDataConstants.KEY_HRV_SCORE,
            )
            is not None
        ):
            hr_is_stale = False

        if hr_is_stale:
            resting_heart_rate = None
            baseline_resting_hr = health_param_value(
                health_params,
                "health_params_baseline_resting_hr",
                HealthDataConstants.KEY_BASELINE_RESTING_HR,
            )
            latest_heart_rate = None
            workout_avg_hr = None
            workout_max_hr = None
            hrv_score = None
            walking_heart_rate_avg = None
        else:
            resting_heart_rate = health_param_value(
                health_params,
                "health_params_resting_heart_rate",
                HealthDataConstants.KEY_RESTING_HEART_RATE,
            )
            baseline_resting_hr = health_param_value(
                health_params,
                "health_params_baseline_resting_hr",
                HealthDataConstants.KEY_BASELINE_RESTING_HR,
            )
            latest_heart_rate = health_param_value(
                health_params,
                "health_params_latest_heart_rate",
                HealthDataConstants.KEY_LATEST_HEART_RATE,
            )
            workout_avg_hr = health_param_value(
                health_params,
                "health_params_workout_avg_hr",
                HealthDataConstants.KEY_WORKOUT_AVG_HR,
            )
            workout_max_hr = health_param_value(
                health_params,
                "health_params_workout_max_hr",
                HealthDataConstants.KEY_WORKOUT_MAX_HR,
            )
            hrv_score = health_param_value(
                health_params,
                "health_params_hrv_score",
                HealthDataConstants.KEY_HRV_SCORE,
            )
            walking_heart_rate_avg = health_param_value(
                health_params,
                "health_params_walking_heart_rate_avg",
                HealthDataConstants.KEY_WALKING_HEART_RATE_AVG,
            )

        estimated: List[str] = []

        if walking_heart_rate_avg is None and not hr_is_stale:
            walking_heart_rate_avg = _fallback_walking_heart_rate_avg(
                historical_snapshots
            )
            if walking_heart_rate_avg is not None:
                estimated.append("walking_heart_rate_avg")

        if hrv_score is None and not hr_is_stale:
            hrv_score = _fallback_hrv_score_from_resting_hr(resting_heart_rate)
            if hrv_score is not None:
                estimated.append("hrv_score")

        mood_trend_7d = _fallback_mood_trend_7d(historical_snapshots)
        if mood_trend_7d is not None:
            estimated.append("mood_trend_7d")
        latest_mood = (raw_data or {}).get("latest_mood") or {}
        if not isinstance(latest_mood, dict):
            latest_mood = (
                latest_mood.model_dump(by_alias=True)
                if hasattr(latest_mood, "model_dump")
                else {}
            )
        mood_date = (
            latest_mood.get("current_mood_date")
            or latest_mood.get("moodDate")
            or latest_mood.get("mood_date")
        )
        primary_mood = (
            latest_mood.get("current_mood")
            or latest_mood.get("mood")
        )
        mood_notes = latest_mood.get("notes")
        mood_has_notes = bool(mood_notes) if mood_notes is not None else None

        return CardioAndStressSignal(
            resting_heart_rate=resting_heart_rate,
            baseline_resting_hr=baseline_resting_hr,
            latest_heart_rate=latest_heart_rate,
            workout_avg_hr=workout_avg_hr,
            workout_max_hr=workout_max_hr,
            stress_high=bool(
                health_param_value(
                    health_params,
                    "health_params_stress_high",
                    "stress_high",
                )
            ),
            primary_mood=primary_mood,
            mood_trend_7d=mood_trend_7d,
            hrv_score=hrv_score,
            walking_heart_rate_avg=walking_heart_rate_avg,
            energy_week_vs_prior_month_pct=health_param_value(
                health_params,
                "health_params_energy_week_vs_prior_month_pct",
                HealthDataConstants.KEY_ENERGY_WEEK_VS_PRIOR_MONTH_PCT,
            ),
            mood_first=primary_mood,
            mood_log_count=1 if primary_mood else None,
            mood_has_notes=mood_has_notes,
            mood_date=mood_date,
            estimated_fields=estimated,
        )
