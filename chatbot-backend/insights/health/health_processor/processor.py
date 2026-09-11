"""Health insight processor — signals + week/month period aggregates."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from insights.data_processor import DataProcessor
from insights.health.health_processor.common.utils import health_param_value
from insights.health.health_processor.common.constants import (
    HEART_RATE_RESTING_SANITY_HIGH_BPM,
    HEART_RATE_RESTING_SANITY_LOW_BPM,
)
from insights.health.health_processor.sleep.constants import SLEEP_SEVERE_LOW_HOURS
from insights.health.health_processor.steps.constants import STEPS_VERY_LOW_DAY_PROGRESS
from insights.health.health_processor.context.processor import ContextSignalBuilder
from insights.health.health_processor.period_metrics import compute_health_period_aggregates
from insights.schemas.processed_context import ExtractedSignals
from services.executor.constant import HealthDataConstants, TimeDataConstants
from utils.logger import logger


class HealthInsightProcessor:
    """Process extracted raw_data into ExtractedSignals for health insight.

    1. Run existing DataProcessor.extract_signals (typed health/calendar/…)
    2. Attach week/month/baseline period aggregates (chat-parity calcs)
    3. Align cardio baseline_resting_hr with period RHR baseline when useful
    4. Attach deterministic Context Signals (sleep window, weekend, …)
    """

    @staticmethod
    def process(raw_data: Dict[str, Any]) -> ExtractedSignals:
        extracted = DataProcessor.extract_signals(raw_data=raw_data)

        today = HealthInsightProcessor._resolve_today(raw_data)
        health_params = raw_data.get("health_params") or {}
        timezone_str = (
            raw_data.get("timezone")
            or (raw_data.get("time_data") or {}).get("timezone")
            or "UTC"
        )
        period = compute_health_period_aggregates(
            today_health_stats=raw_data.get("today_health_stats"),
            today=today,
            tz=timezone_str,
            health_params=health_params,
        )

        if extracted.health_signals is not None:
            extracted.health_signals.period_aggregates = period
            HealthInsightProcessor._sync_rhr_baseline(
                extracted, period, health_params
            )
            HealthInsightProcessor._sync_energy_from_period(extracted, period)
            HealthInsightProcessor._mirror_five_focuses(extracted)
            HealthInsightProcessor._apply_health_sanity_guard(
                extracted, health_params
            )
            ContextSignalBuilder.attach(extracted)
        else:
            logger.warning(
                "[health_processor] health_signals missing; period_aggregates skipped"
            )

        return extracted

    @staticmethod
    def _resolve_today(raw_data: Dict[str, Any]) -> date:
        time_data = raw_data.get("time_data") or {}
        current = (
            time_data.get(TimeDataConstants.KEY_CURRENT_TIME_ISO)
            or raw_data.get("current_time")
        )
        timezone = raw_data.get("timezone") or time_data.get("timezone")
        if not timezone:
            logger.warning(
                "_resolve_today: timezone missing in raw_data/time_data; "
                "falling back to UTC. Caller should resolve via "
                "DataCollector.resolve_timezone (FE → profile → DEFAULT)."
            )
            timezone = "UTC"
        if isinstance(current, datetime):
            return current.date()
        if isinstance(current, str) and current:
            try:
                return datetime.fromisoformat(
                    current.replace("Z", "+00:00")
                ).date()
            except Exception:
                pass
        try:
            return datetime.now(ZoneInfo(timezone)).date()
        except Exception:
            return datetime.utcnow().date()

    @staticmethod
    def _sync_rhr_baseline(
        extracted: ExtractedSignals,
        period,
        health_params: Dict[str, Any],
    ) -> None:
        """Prefer computed period baseline when prepare baseline is missing."""
        hs = extracted.health_signals
        if hs is None or hs.cardio_stress is None:
            return
        cardio = hs.cardio_stress
        period_rhr = period.resting_heart_rate if period else None
        period_baseline = (
            period_rhr.baseline_avg if period_rhr is not None else None
        )
        if cardio.baseline_resting_hr is None and period_baseline is not None:
            cardio.baseline_resting_hr = period_baseline
            health_params[HealthDataConstants.KEY_BASELINE_RESTING_HR] = (
                period_baseline
            )
        if (
            cardio.resting_heart_rate is None
            and period_rhr is not None
            and period_rhr.today is not None
        ):
            cardio.resting_heart_rate = period_rhr.today

    @staticmethod
    def _sync_energy_from_period(extracted: ExtractedSignals, period) -> None:
        """Fill energy focus from full ENERGY.summaryData + period aggregates."""
        hs = extracted.health_signals
        if hs is None or period is None:
            return
        energy = hs.energy
        if energy is None:
            return

        if energy.month_resting_total is None:
            energy.month_resting_total = period.energy_month_resting_total
        if energy.total_active_energy is None:
            energy.total_active_energy = period.energy_total_active_today
        if energy.calories_burned_today is None:
            energy.calories_burned_today = (
                energy.total_active_energy
                or (period.calories_burned.today if period.calories_burned else None)
            )
        if energy.total_active_energy is None and energy.calories_burned_today is not None:
            energy.total_active_energy = energy.calories_burned_today

        if energy.current_month_avg is None:
            energy.current_month_avg = period.energy_current_month_avg
            if energy.current_month_avg is None and period.calories_burned:
                energy.current_month_avg = period.calories_burned.month_avg

        if energy.week_avg is None:
            energy.week_avg = period.energy_week_avg
        if energy.previous_month_avg is None:
            energy.previous_month_avg = period.energy_prev_month_avg
        if energy.prev_month_avg is None:
            energy.prev_month_avg = energy.previous_month_avg
        if energy.week_vs_prior_month_pct is None:
            energy.week_vs_prior_month_pct = period.energy_week_vs_prior_month_pct

        if energy.active_minutes_today is None and period.active_minutes:
            energy.active_minutes_today = period.active_minutes.today
        if energy.workout_duration_min is None and period.total_workout_min:
            today_w = period.total_workout_min.today
            if today_w is not None:
                energy.workout_duration_min = int(round(today_w))

        from insights.health.health_processor.energy.processor import EnergySignalProcessor

        has_data = (
            energy.total_active_energy is not None
            or energy.calories_burned_today is not None
            or energy.week_avg is not None
            or energy.previous_month_avg is not None
            or energy.current_month_avg is not None
            or energy.month_resting_total is not None
            or (
                energy.summary is not None
                and (
                    energy.summary.avg_energy_burn is not None
                    or energy.summary.week_avg is not None
                )
            )
        )
        # Mirror into summary if still empty
        if energy.summary is not None:
            if energy.summary.today_active is None:
                energy.summary.today_active = energy.total_active_energy
            if energy.summary.month_resting_total is None:
                energy.summary.month_resting_total = energy.month_resting_total
            if energy.summary.current_month_avg is None:
                energy.summary.current_month_avg = energy.current_month_avg
            if energy.summary.week_avg is None:
                energy.summary.week_avg = energy.week_avg
            if energy.summary.previous_month_avg is None:
                energy.summary.previous_month_avg = energy.previous_month_avg

        energy.level = EnergySignalProcessor.compute_level(
            energy.total_active_energy or energy.calories_burned_today,
            energy.week_vs_prior_month_pct,
            stale=not has_data,
        )
        if hs.cardio_stress is not None and energy.week_vs_prior_month_pct is not None:
            hs.cardio_stress.energy_week_vs_prior_month_pct = (
                energy.week_vs_prior_month_pct
            )

    @staticmethod
    def _mirror_five_focuses(extracted: ExtractedSignals) -> None:
        """Keep steps/activity and heart_rate ↔ cardio_stress aligned."""
        hs = extracted.health_signals
        if hs is None:
            return
        if hs.steps is not None:
            hs.activity = hs.steps.to_activity_signal()
        elif hs.activity is not None:
            # Promote flat legacy activity into StepsSignal envelope when steps missing
            from insights.schemas.processed_context import StepsOverall, StepsSignal

            act = hs.activity
            hs.steps = StepsSignal(
                level=act.level,
                steps_today=act.steps_today,
                steps_goal=act.steps_goal,
                remaining_steps=act.remaining_steps,
                pace_ratio=act.pace_ratio,
                active_hours_left=act.active_hours_left,
                steps_streak=act.steps_streak,
                workout_activity_type=act.workout_activity_type,
                workout_duration_min=act.workout_duration_min,
                calories_burned_today=act.calories_burned_today,
                daily_health_score_trend=act.daily_health_score_trend,
                estimated_fields=list(act.estimated_fields or []),
                overall=StepsOverall(status="insufficient_data", confidence="low"),
            )

        cardio = hs.cardio_stress
        hr = hs.heart_rate
        if cardio is not None and hr is not None:
            # Heart-rate processor is source of truth — mirror into cardio legacy
            cardio.resting_heart_rate = hr.resting_heart_rate
            cardio.baseline_resting_hr = hr.baseline_resting_hr
            cardio.latest_heart_rate = hr.latest_heart_rate
            cardio.workout_avg_hr = hr.workout_avg_hr
            cardio.workout_max_hr = hr.workout_max_hr
            cardio.stress_high = hr.stress_high
            cardio.hrv_score = hr.hrv_score
            cardio.walking_heart_rate_avg = hr.walking_heart_rate_avg
            # Keep summary mirrors in sync when period filled gaps on flat fields
            if hr.summary is not None:
                if hr.summary.today_resting is None and hr.resting_heart_rate is not None:
                    hr.summary.today_resting = hr.resting_heart_rate
                if hr.summary.today_latest is None and hr.latest_heart_rate is not None:
                    hr.summary.today_latest = float(hr.latest_heart_rate)
                if hr.summary.today_hrv is None and hr.hrv_score is not None:
                    hr.summary.today_hrv = float(hr.hrv_score)
                if hr.summary.baseline_resting is None:
                    hr.summary.baseline_resting = hr.baseline_resting_hr

        mood = hs.mood

    @staticmethod
    def _apply_health_sanity_guard(
        extracted: ExtractedSignals,
        health_params: Dict[str, Any],
    ) -> None:
        """Sanity-check extreme health values before narrative/QA layers.

        These checks are for data hygiene and phrasing safety, not diagnosis.
        """
        hs = extracted.health_signals
        if hs is None:
            return

        alerts = list((extracted.raw_data or {}).get("health_abnormal_flags") or [])

        # 1) RHR out-of-range: treat as invalid reading for this run.
        cardio = hs.cardio_stress
        if cardio is not None and cardio.resting_heart_rate is not None:
            rhr = float(cardio.resting_heart_rate)
            if (
                rhr < HEART_RATE_RESTING_SANITY_LOW_BPM
                or rhr > HEART_RATE_RESTING_SANITY_HIGH_BPM
            ):
                alerts.append(
                    {
                        "metric": "resting_heart_rate",
                        "kind": "out_of_sanity_range",
                        "value": rhr,
                        "range": [
                            HEART_RATE_RESTING_SANITY_LOW_BPM,
                            HEART_RATE_RESTING_SANITY_HIGH_BPM,
                        ],
                    }
                )
                cardio.resting_heart_rate = None
                health_params[HealthDataConstants.KEY_RESTING_HEART_RATE] = None
                if hs.heart_rate is not None:
                    hs.heart_rate.resting_heart_rate = None
                    if hs.heart_rate.summary is not None:
                        hs.heart_rate.summary.today_resting = None
                        hs.heart_rate.summary.avg_resting_heart_rate = None

                pa = hs.period_aggregates
                if pa and pa.resting_heart_rate:
                    pa.resting_heart_rate.today = None
                    pa.rhr_elevated = None

        # 2) Very short sleep: mark severe (keep value for transparent reporting).
        sleep = hs.sleep
        if sleep is not None and sleep.signals.sleep_duration is not None:
            duration_sig = sleep.signals.sleep_duration
            last_night_h = (
                duration_sig.metrics.get("last_night_hours")
                if duration_sig.metrics
                else None
            )
            if last_night_h is not None:
                sl = float(last_night_h)
                if sl < SLEEP_SEVERE_LOW_HOURS:
                    alerts.append(
                        {
                            "metric": "sleep_lastnight",
                            "kind": "very_low_sleep_duration",
                            "value": sl,
                            "threshold": SLEEP_SEVERE_LOW_HOURS,
                        }
                    )
                    if sleep.signals.sleep_debt is not None:
                        sleep.signals.sleep_debt.status = "severe"

        # 3) Steps very low progress: attach explicit low-progress alert.
        activity = hs.activity
        if activity is not None and activity.steps_today is not None:
            st = int(activity.steps_today)
            if st < STEPS_VERY_LOW_DAY_PROGRESS:
                alerts.append(
                    {
                        "metric": "steps_today",
                        "kind": "very_low_progress",
                        "value": st,
                        "threshold": STEPS_VERY_LOW_DAY_PROGRESS,
                    }
                )

        # 4) Ensure workout minutes is available when workout duration exists
        # (duration is typically stored in seconds in health_params).
        if (
            activity is not None
            and (activity.workout_duration_min is None or activity.workout_duration_min <= 0)
        ):
            workout_seconds = health_param_value(
                health_params,
                "health_params_workout_duration",
                HealthDataConstants.KEY_WORKOUT_DURATION,
            )
            if isinstance(workout_seconds, (int, float)) and workout_seconds > 0:
                activity.workout_duration_min = int(round(float(workout_seconds) / 60.0))
                if hs.steps is not None:
                    hs.steps.workout_duration_min = activity.workout_duration_min

        if extracted.raw_data is not None and alerts:
            extracted.raw_data["health_abnormal_flags"] = alerts
