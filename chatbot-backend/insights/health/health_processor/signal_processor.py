"""Orchestrate per-group health signal processors into HealthSignalsBlock.

Group processors live under ``insights.health.health_processor.<group>``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from insights.health.health_processor.energy.processor import EnergySignalProcessor
from insights.health.health_processor.health_score.processor import HealthScoreSignalProcessor
from insights.health.health_processor.heart_rate.legacy_cardio import LegacyCardioStressProcessor
from insights.health.health_processor.heart_rate.processor import HeartRateSignalProcessor
from insights.health.health_processor.mood.processor import MoodSignalProcessor
from insights.health.health_processor.sleep.processor import SleepSignalProcessor
from insights.health.health_processor.steps.processor import StepsSignalProcessor
from insights.schemas.processed_context import HealthSignalsBlock


class HealthSignalProcessor:
    """Facade: sleep / HR / energy / steps / mood / health_score → one block."""

    @staticmethod
    def process(
        raw_data: Dict[str, Any],
        time_data: Dict[str, Any],
        historical_snapshots: Optional[List[Any]] = None,
        staleness_info: Optional[Dict[str, Any]] = None,
    ) -> HealthSignalsBlock:
        health_params = raw_data.get("health_params", {})

        sleep = SleepSignalProcessor.process(
            health_params=health_params,
            historical_snapshots=historical_snapshots,
            staleness_info=staleness_info,
            time_data=time_data,
            raw_data=raw_data,
        )
        activity = StepsSignalProcessor.process(
            health_params=health_params,
            time_data=time_data,
            staleness_info=staleness_info,
            historical_snapshots=historical_snapshots,
            raw_data=raw_data,
        )
        cardio = LegacyCardioStressProcessor.process(
            health_params=health_params,
            staleness_info=staleness_info,
            raw_data=raw_data,
            historical_snapshots=historical_snapshots,
        )
        heart_rate = HeartRateSignalProcessor.process(
            health_params=health_params,
            historical_snapshots=historical_snapshots,
            staleness_info=staleness_info,
            time_data=time_data,
            raw_data=raw_data,
        )
        if heart_rate is not None:
            cardio.resting_heart_rate = heart_rate.resting_heart_rate
            cardio.baseline_resting_hr = heart_rate.baseline_resting_hr
            cardio.latest_heart_rate = heart_rate.latest_heart_rate
            cardio.workout_avg_hr = heart_rate.workout_avg_hr
            cardio.workout_max_hr = heart_rate.workout_max_hr
            cardio.hrv_score = heart_rate.hrv_score
            cardio.walking_heart_rate_avg = heart_rate.walking_heart_rate_avg
            cardio.stress_high = heart_rate.stress_high
            if heart_rate.estimated_fields:
                cardio.estimated_fields = list(
                    dict.fromkeys(
                        list(cardio.estimated_fields or [])
                        + list(heart_rate.estimated_fields)
                    )
                )

        energy = EnergySignalProcessor.process(
            health_params=health_params,
            historical_snapshots=historical_snapshots,
            staleness_info=staleness_info,
            time_data=time_data,
            raw_data=raw_data,
            activity_calories=activity.calories_burned_today,
            workout_duration_min=activity.workout_duration_min,
            workout_activity_type=activity.workout_activity_type,
            cardio_week_pct=cardio.energy_week_vs_prior_month_pct,
        )

        if energy.week_vs_prior_month_pct is not None:
            cardio.energy_week_vs_prior_month_pct = energy.week_vs_prior_month_pct

        mood = MoodSignalProcessor.process(
            health_params=health_params,
            historical_snapshots=historical_snapshots,
            staleness_info=staleness_info,
            time_data=time_data,
            raw_data=raw_data,
            sleep_context={
                "quality": getattr(sleep, "quality", None),
                "level": getattr(sleep, "level", None),
                "last_night_h": getattr(sleep, "last_night_h", None),
                "debt_hours": getattr(sleep, "debt_hours", None),
            },
            energy_context={
                "level": getattr(energy, "level", None),
                "total_active_energy": getattr(energy, "total_active_energy", None),
            },
            hr_context={"stress_high": getattr(heart_rate, "stress_high", False)},
        )
        health_score = HealthScoreSignalProcessor.process(
            health_params=health_params,
            historical_snapshots=historical_snapshots,
            time_data=time_data,
            raw_data=raw_data,
        )

        return HealthSignalsBlock.from_legacy_parts(
            sleep=sleep,
            activity=activity.to_activity_signal(),
            cardio=cardio,
            heart_rate=heart_rate,
            energy=energy,
            mood=mood,
            steps=activity,
            health_score=health_score,
        )
