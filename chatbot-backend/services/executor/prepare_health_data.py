"""Prepare Health Data for Insight Analysis"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from insights.health.canonical_field_mapping import CanonicalField
from models.models import (
    EnergyHealthSummary,
    HealthSummary,
    HRHealthSummary,
    SleepHealthSummary,
    StepsHealthSummary,
)
from services.executor.constant import (
    APIResponseKeys,
    HealthDataConstants,
    TargetKeys,
    TimeDataConstants,
)
from services.external_api_service import IExternalAPIService
from utils.logger import logger


class PrepareHealthData:
    def __init__(
        self,
        external_api_service: Optional[IExternalAPIService],
        user_id: str,
        user_profile: Optional[Dict[str, Any]],
        timezone: Optional[str] = None,
        time_data: Optional[Dict[str, Any]] = None,
    ):
        self.external_api_service = external_api_service
        self.user_id = user_id
        self.user_profile = user_profile
        if timezone:
            self.timezone = timezone
        else:
            logger.warning(
                "PrepareHealthData instantiated without timezone for user "
                f"{user_id}; defaulting to UTC."
            )
            self.timezone = "UTC"
        self.time_data = time_data
        self.health_data = None
        self._health_history_2w = None

    async def prepare_all_health_params(self) -> Dict[str, Any]:
        params = {}

        # Fetch all health data ONCE (2 weeks) - uses single API call
        await self._fetch_2_week_health_data()

        # Extract goals from user profile
        goals = self._extract_goals()
        params.update(goals)

        advanced_metrics = await self._calculate_advanced_metrics()
        params.update(advanced_metrics)
        enrichment_metrics = self._extract_enrichment_metrics()
        params.update(enrichment_metrics)

        basic_params = self._extract_basic_health_data()
        params.update(basic_params)

        sleep_quality = self._calculate_sleep_quality(
            params.get(HealthDataConstants.KEY_SLEEP_QUALITY_SCORE)
        )
        if sleep_quality:
            params[HealthDataConstants.KEY_SLEEP_QUALITY] = sleep_quality

        params["stress_high"] = params.get("stress_high", False)
        params["stress_signal_high"] = params.get("stress_signal_high", False)

        # Add thresholds
        thresholds = self._get_thresholds()
        params.update(thresholds)

        return params

    def get_health_data(self) -> Optional[Dict[str, HealthSummary]]:
        return self.health_data

    async def _fetch_2_week_health_data(self) -> None:
        if self._health_history_2w is not None:
            return  # Already cached

        if not self.external_api_service:
            return

        try:
            tz = ZoneInfo(self.timezone)
            now = datetime.now(tz)

            # Calculate date range: Previous Monday to today (2 weeks)
            days_since_monday = now.weekday()
            start_of_current_week = now - timedelta(days=days_since_monday)
            start_of_previous_week = start_of_current_week - timedelta(weeks=1)

            start_date = start_of_previous_week.strftime("%Y-%m-%d")
            end_date = now.strftime("%Y-%m-%d")

            logger.info(
                f"📊 Fetching 2-week health data: {start_date} to {end_date} "
                f"(single API call for both basic and advanced metrics)"
            )

            # Single API call for all data
            all_data = await self.external_api_service.get_health_summaries_by_range(
                user_id=self.user_id,
                start_date=start_date,
                end_date=end_date,
                timezone=self.timezone,
            )

            if not all_data:
                self._health_history_2w = []
                self.health_data = {}
                return

            # Cache raw data for advanced metrics
            self._health_history_2w = all_data

            # Group current week data by type for basic + enrichment
            summaries_by_type = {}
            for item in all_data:
                item_type = item.type
                if item_type not in summaries_by_type:
                    summaries_by_type[item_type] = item
                else:
                    # Keep the most recent (by startDate)
                    if (
                        item.dateRange.startDate
                        > summaries_by_type[item_type].dateRange.startDate
                    ):
                        summaries_by_type[item_type] = item

            self.health_data = summaries_by_type

        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch 2-week health data: {str(e)}")
            self._health_history_2w = []
            self.health_data = {}

    async def _fetch_current_week_data(self) -> None:
        """Legacy method - now redirects to _fetch_2_week_health_data."""
        await self._fetch_2_week_health_data()

    def _extract_basic_health_data(self) -> Dict[str, Any]:
        params = {
            HealthDataConstants.KEY_STEPS_TODAY: None,
            HealthDataConstants.KEY_SLEEP_LASTNIGHT: None,
            HealthDataConstants.KEY_SLEEP_QUALITY_SCORE: None,
            HealthDataConstants.KEY_LATEST_HEART_RATE: None,
            HealthDataConstants.KEY_RESTING_HEART_RATE: None,
            HealthDataConstants.KEY_LAST_WAKE_TIME: None,
            HealthDataConstants.KEY_FIRST_SLEEP_TIME: None,
            HealthDataConstants.KEY_WORKOUT_HEART_RATE: None,
            HealthDataConstants.KEY_WORKOUT_AVG_HR: None,
            HealthDataConstants.KEY_WORKOUT_MAX_HR: None,
            HealthDataConstants.KEY_WORKOUT_MIN_HR: None,
            HealthDataConstants.KEY_WORKOUT_DURATION: None,
            HealthDataConstants.KEY_WORKOUT_ACTIVITY_TYPE: None,
            HealthDataConstants.KEY_CALORIES_BURNED_TODAY: None,
            HealthDataConstants.KEY_ACTIVE_MINUTES_TODAY: None,
            # Date tracking for staleness detection
            "steps_last_data_date": None,
            "sleep_last_data_date": None,
            "hr_last_data_date": None,
            "mood_last_data_date": None,
            HealthDataConstants.KEY_ENERGY_LAST_DATA_DATE: None,
        }

        if not self.health_data:
            return params

        try:
            # Get STEPS data
            steps_summary = self.health_data.get(HealthDataConstants.STEPS_TYPE)
            if steps_summary:
                latest_steps_entry = self._get_latest_data_entry(steps_summary)
                if latest_steps_entry:
                    params[HealthDataConstants.KEY_STEPS_TODAY] = (
                        latest_steps_entry.get(APIResponseKeys.STEPS)
                    )
                    # Track steps date for staleness
                    params["steps_last_data_date"] = latest_steps_entry.get(APIResponseKeys.DATE)

            if (
                params.get(HealthDataConstants.KEY_STEPS_TODAY) is None
                and self.user_profile
            ):
                latest_metrics = self.user_profile.get("latest_health_metrics") or {}
                profile_steps = latest_metrics.get("steps_per_day")
                profile_metric_date = str(
                    latest_metrics.get("metric_date") or ""
                ).strip()
                current_date = (
                    self.time_data.get(TimeDataConstants.KEY_CURRENT_DATE)
                    if self.time_data
                    else None
                )
                if (
                    profile_steps is not None
                    and current_date
                    and profile_metric_date == str(current_date)
                ):
                    params[HealthDataConstants.KEY_STEPS_TODAY] = profile_steps
                    params["steps_last_data_date"] = profile_metric_date
                elif profile_steps is not None:
                    logger.info(
                        f"⚠️ Skipping stale profile_steps fallback: "
                        f"profile metric_date={profile_metric_date!r} != "
                        f"current_date={current_date!r}"
                    )

            # Get SLEEP data
            sleep_summary = self.health_data.get(HealthDataConstants.SLEEP_TYPE)
            if sleep_summary:
                latest_sleep_entry = self._get_latest_data_entry(sleep_summary)
                if latest_sleep_entry:
                    # Log sleep data source for debugging
                    sleep_date = latest_sleep_entry.get(APIResponseKeys.DATE, "unknown")
                    # Store sleep date for staleness detection
                    params["sleep_last_data_date"] = sleep_date

                    params[HealthDataConstants.KEY_SLEEP_LASTNIGHT] = (
                        latest_sleep_entry.get(APIResponseKeys.TOTAL)
                    )
                    # Try both sleepScore and qualityScore for compatibility
                    params[HealthDataConstants.KEY_SLEEP_QUALITY_SCORE] = (
                        latest_sleep_entry.get(APIResponseKeys.SLEEP_SCORE)
                        or latest_sleep_entry.get(APIResponseKeys.QUALITY_SCORE)
                    )
                    last_wake_time_raw = latest_sleep_entry.get(
                        APIResponseKeys.LAST_WAKE_TIME
                    )
                    if last_wake_time_raw:
                        params[HealthDataConstants.KEY_LAST_WAKE_TIME] = (
                            self._convert_datetime_to_timezone(
                                last_wake_time_raw, self.timezone
                            )
                        )

                    first_sleep_time_raw = latest_sleep_entry.get(
                        APIResponseKeys.FIRST_SLEEP_TIME
                    )
                    if first_sleep_time_raw:
                        first_sleep_time_local = (
                            self._convert_datetime_to_timezone(
                                first_sleep_time_raw, self.timezone
                            )
                        )
                        params[HealthDataConstants.KEY_FIRST_SLEEP_TIME] = (
                            first_sleep_time_local
                        )
                        params["bedtime"] = first_sleep_time_local

                    # Sleep stage breakdown (deep, rem, core — in hours from API)
                    deep_h = latest_sleep_entry.get("deep", 0) or 0
                    rem_h = latest_sleep_entry.get("rem", 0) or 0
                    core_h = latest_sleep_entry.get("core", 0) or 0
                    params[HealthDataConstants.KEY_DEEP_SLEEP] = int(deep_h * 60)
                    params[HealthDataConstants.KEY_REM_SLEEP] = int(rem_h * 60)
                    params[HealthDataConstants.KEY_LIGHT_SLEEP] = int(core_h * 60)

            # Get HR data
            hr_summary = self.health_data.get(HealthDataConstants.HR_TYPE)
            if hr_summary:
                latest_hr_entry = self._get_latest_data_entry(hr_summary)
                if latest_hr_entry:
                    # Track HR date for staleness
                    params["hr_last_data_date"] = latest_hr_entry.get(APIResponseKeys.DATE)

                    params[HealthDataConstants.KEY_LATEST_HEART_RATE] = (
                        latest_hr_entry.get(APIResponseKeys.LATEST_HR)
                    )
                    params[HealthDataConstants.KEY_RESTING_HEART_RATE] = (
                        latest_hr_entry.get(APIResponseKeys.RESTING_HEART_RATE)
                    )
                    # HRV from summary data
                    hr_summary = self.health_data.get(HealthDataConstants.HR_TYPE)
                    if hr_summary and hasattr(hr_summary, "summaryData"):
                        avg_hrv = getattr(
                            hr_summary.summaryData,
                            APIResponseKeys.AVERAGE_HEART_RATE_VARIABILITY_CURRENT,
                            None,
                        )
                        if avg_hrv is not None:
                            params[HealthDataConstants.KEY_HRV_SCORE] = avg_hrv

                    # Extract workout heart rate data
                    workout_hr = latest_hr_entry.get(APIResponseKeys.WORKOUT_HEART_RATE)
                    if workout_hr:
                        # Handle both Pydantic model and dict
                        if hasattr(workout_hr, "model_dump"):
                            workout_hr_dict = workout_hr.model_dump()
                        elif hasattr(workout_hr, "__dict__"):
                            workout_hr_dict = dict(workout_hr)
                        elif isinstance(workout_hr, dict):
                            workout_hr_dict = workout_hr
                        else:
                            workout_hr_dict = None

                        if workout_hr_dict:
                            params[HealthDataConstants.KEY_WORKOUT_HEART_RATE] = (
                                workout_hr_dict
                            )
                            params[HealthDataConstants.KEY_WORKOUT_AVG_HR] = (
                                workout_hr_dict.get(APIResponseKeys.WORKOUT_AVG_HR)
                            )
                            params[HealthDataConstants.KEY_WORKOUT_MAX_HR] = (
                                workout_hr_dict.get(APIResponseKeys.WORKOUT_MAX_HR)
                            )
                            params[HealthDataConstants.KEY_WORKOUT_MIN_HR] = (
                                workout_hr_dict.get(APIResponseKeys.WORKOUT_MIN_HR)
                            )
                            params[HealthDataConstants.KEY_WORKOUT_DURATION] = (
                                workout_hr_dict.get(APIResponseKeys.WORKOUT_DURATION)
                            )
                            params[HealthDataConstants.KEY_WORKOUT_ACTIVITY_TYPE] = (
                                workout_hr_dict.get(
                                    APIResponseKeys.WORKOUT_ACTIVITY_TYPE
                                )
                            )

            # Get ENERGY data — full ENERGY.summaryData set + latest daily entry
            energy_summary = self.health_data.get(HealthDataConstants.ENERGY_TYPE)
            if energy_summary:
                latest_energy_entry = self._get_latest_data_entry(energy_summary)
                if latest_energy_entry:
                    params[HealthDataConstants.KEY_ENERGY_LAST_DATA_DATE] = (
                        latest_energy_entry.get(APIResponseKeys.DATE)
                    )
                    for key in (
                        "energyBurn",
                        CanonicalField.ENERGY_BURN_ONE_DAY_KCAL,
                        APIResponseKeys.TOTAL_ACTIVE_ENERGY,
                        "activeEnergyBurn",
                        "caloriesBurned",
                        "calories",
                        "energy",
                    ):
                        v = latest_energy_entry.get(key)
                        if isinstance(v, (int, float)):
                            params[HealthDataConstants.KEY_CALORIES_BURNED_TODAY] = float(v)
                            params[HealthDataConstants.KEY_TOTAL_ACTIVE_ENERGY] = float(v)
                            break
                    for key in ("activeMinutes", "active_minutes", "minutes"):
                        v = latest_energy_entry.get(key)
                        if isinstance(v, (int, float)):
                            params[HealthDataConstants.KEY_ACTIVE_MINUTES_TODAY] = float(v)
                            break

                # Full ENERGY.summaryData:
                # totalRestingEnergy, totalActiveEnergy, currentMonthAvg,
                # avgCurrentWeekEnergyBurn, previousMonthAvg
                sd = self._safe_get_summary_data(energy_summary) or {}
                resting = sd.get(APIResponseKeys.TOTAL_RESTING_ENERGY)
                if isinstance(resting, (int, float)):
                    params[HealthDataConstants.KEY_ENERGY_MONTH_RESTING_TOTAL] = float(
                        resting
                    )

                # Only use summaryData.totalActiveEnergy if daily entry didn't provide it
                if params.get(HealthDataConstants.KEY_TOTAL_ACTIVE_ENERGY) is None:
                    active = sd.get(APIResponseKeys.TOTAL_ACTIVE_ENERGY)
                    if isinstance(active, (int, float)):
                        params[HealthDataConstants.KEY_TOTAL_ACTIVE_ENERGY] = float(active)

                month_avg = sd.get(APIResponseKeys.CURRENT_MONTH_AVG)
                if isinstance(month_avg, (int, float)):
                    params[HealthDataConstants.KEY_ENERGY_CURRENT_MONTH_AVG] = float(
                        month_avg
                    )

                week_avg = sd.get(APIResponseKeys.AVG_CURRENT_WEEK_ENERGY_BURN)
                if isinstance(week_avg, (int, float)):
                    params[HealthDataConstants.KEY_ENERGY_WEEK_AVG] = float(week_avg)

                prev_month = sd.get(APIResponseKeys.PREVIOUS_MONTH_AVG)
                if isinstance(prev_month, (int, float)):
                    params[HealthDataConstants.KEY_ENERGY_PREVIOUS_MONTH_AVG] = float(
                        prev_month
                    )

        except Exception as e:
            logger.warning(f"⚠️ Failed to extract basic health data: {str(e)}")
        return params

    def _extract_enrichment_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            HealthDataConstants.KEY_DAILY_HEALTH_SCORE_TREND: None,
            HealthDataConstants.KEY_SLEEP_HR_RANGE: None,
            HealthDataConstants.KEY_WALKING_HEART_RATE_AVG: None,
            HealthDataConstants.KEY_AVERAGE_AWAKE_HOURS: None,
            HealthDataConstants.KEY_AVERAGE_TIME_IN_BED_HOURS: None,
            HealthDataConstants.KEY_AVERAGE_TOTAL_SLEEP_HOURS: None,
            HealthDataConstants.KEY_ENERGY_WEEK_VS_PRIOR_MONTH_PCT: None,
        }
        if not self.health_data:
            return metrics

        try:
            self._populate_health_score_trend(metrics)
            self._populate_sleep_hr_range(metrics)
            self._populate_walking_hr_avg(metrics)
            self._populate_sleep_time_aggs(metrics)
            self._populate_energy_pct_change(metrics)
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract enrichment metrics: {str(e)}")

        return metrics

    def _safe_get_summary_data(
        self, summary
    ) -> Optional[Dict[str, Any]]:
        if not summary:
            return None
        sd = getattr(summary, "summaryData", None)
        if sd is None:
            return None
        if isinstance(sd, dict):
            return sd
        if hasattr(sd, "model_dump"):
            return sd.model_dump()
        try:
            return dict(sd)
        except Exception:
            return None

    def _safe_get_data_entries(self, summary) -> List[Dict[str, Any]]:
        """Convert ``summary.data`` (list of Pydantic/dict) to list of dict."""
        if not summary or not getattr(summary, "data", None):
            return []
        out: List[Dict[str, Any]] = []
        for entry in summary.data:
            if isinstance(entry, dict):
                out.append(entry)
            elif hasattr(entry, "model_dump"):
                out.append(entry.model_dump())
            else:
                try:
                    out.append(dict(entry))
                except Exception:
                    continue
        return out

    def _populate_health_score_trend(self, metrics: Dict[str, Any]) -> None:
        trend: List[Optional[int]] = []
        for type_key in (
            HealthDataConstants.HR_TYPE,
            HealthDataConstants.SLEEP_TYPE,
            HealthDataConstants.STEPS_TYPE,
        ):
            entries = self._safe_get_data_entries(
                self.health_data.get(type_key)
            )
            for e in entries:
                score = e.get(APIResponseKeys.HEALTH_SCORE)
                if score is not None and len(trend) < 14:
                    trend.append(int(score))
            if trend:
                break  # prefer HR → SLEEP → STEPS in that order
        if trend:
            metrics[
                HealthDataConstants.KEY_DAILY_HEALTH_SCORE_TREND
            ] = trend

    def _populate_sleep_hr_range(self, metrics: Dict[str, Any]) -> None:
        """Aggregate (min, max) of HR-during-sleep across the week."""
        hr_entries = self._safe_get_data_entries(
            self.health_data.get(HealthDataConstants.HR_TYPE)
        )
        max_vals: List[int] = []
        min_vals: List[int] = []
        for entry in hr_entries:
            sh = entry.get(APIResponseKeys.SLEEP_HEART_RATE)
            if isinstance(sh, dict):
                if not sh.get("lastTimeRecord"):
                    continue
                max_v = sh.get(APIResponseKeys.MAX)
                min_v = sh.get(APIResponseKeys.MIN)
                if isinstance(max_v, (int, float)) and max_v > 0:
                    max_vals.append(int(max_v))
                if isinstance(min_v, (int, float)) and min_v > 0:
                    min_vals.append(int(min_v))
        if not max_vals and not min_vals:
            return
        metrics[HealthDataConstants.KEY_SLEEP_HR_RANGE] = (
            min(min_vals) if min_vals else None,
            max(max_vals) if max_vals else None,
        )

    def _populate_walking_hr_avg(self, metrics: Dict[str, Any]) -> None:
        """Mean of walkingHeartRateAverage across the week."""
        hr_entries = self._safe_get_data_entries(
            self.health_data.get(HealthDataConstants.HR_TYPE)
        )
        vals: List[int] = []
        for entry in hr_entries:
            v = entry.get(APIResponseKeys.WALKING_HEART_RATE_AVERAGE)
            if isinstance(v, (int, float)):
                vals.append(int(v))
        if vals:
            metrics[
                HealthDataConstants.KEY_WALKING_HEART_RATE_AVG
            ] = round(sum(vals) / len(vals), 1)

    def _populate_sleep_time_aggs(self, metrics: Dict[str, Any]) -> None:
        sleep_summary = self.health_data.get(
            HealthDataConstants.SLEEP_TYPE
        )
        sd = self._safe_get_summary_data(sleep_summary)
        if not sd:
            return
        awake = sd.get(CanonicalField.AVERAGE_AWAKE_HOURS)
        tib = sd.get(CanonicalField.AVERAGE_TIME_IN_BED_HOURS)
        asleep = sd.get(CanonicalField.AVERAGE_TOTAL_SLEEP_HOURS)
        if isinstance(awake, (int, float)):
            metrics[HealthDataConstants.KEY_AVERAGE_AWAKE_HOURS] = round(
                float(awake), 2
            )
        if isinstance(tib, (int, float)):
            metrics[HealthDataConstants.KEY_AVERAGE_TIME_IN_BED_HOURS] = round(
                float(tib), 2
            )
        if isinstance(asleep, (int, float)):
            metrics[HealthDataConstants.KEY_AVERAGE_TOTAL_SLEEP_HOURS] = round(
                float(asleep), 2
            )

    def _populate_energy_pct_change(self, metrics: Dict[str, Any]) -> None:
        """Compare this-week avg burn to previous-month avg; return %."""
        energy_summary = self.health_data.get(
            HealthDataConstants.ENERGY_TYPE
        )
        sd = self._safe_get_summary_data(energy_summary)
        if not sd:
            return
        week_avg = sd.get(APIResponseKeys.AVG_CURRENT_WEEK_ENERGY_BURN)
        prior = sd.get(APIResponseKeys.PREVIOUS_MONTH_AVG)
        if not (
            isinstance(week_avg, (int, float))
            and isinstance(prior, (int, float))
            and prior
        ):
            return
        pct = (week_avg - prior) / prior * 100
        metrics[
            HealthDataConstants.KEY_ENERGY_WEEK_VS_PRIOR_MONTH_PCT
        ] = round(pct, 1)

    def _convert_datetime_to_timezone(
        self, datetime_str: str, target_timezone: str
    ) -> Optional[str]:
        if not datetime_str:
            return None

        try:
            # Parse ISO datetime string (handle both with and without timezone)
            if datetime_str.endswith("Z"):
                # UTC timezone
                dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
            elif "+" in datetime_str or datetime_str.count("-") > 2:
                # Has timezone info
                dt = datetime.fromisoformat(datetime_str)
            else:
                # No timezone, assume UTC
                dt = datetime.fromisoformat(datetime_str + "+00:00")

            # Convert to target timezone
            tz = ZoneInfo(target_timezone)
            dt_local = dt.astimezone(tz)

            # Format as YYYY-MM-DDTHH:MM:SS (no timezone suffix)
            return dt_local.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to convert datetime {datetime_str} to {target_timezone}: {str(e)}"
            )
            return None

    def _extract_goals(self) -> Dict[str, Any]:
        """Extract goals từ user_profile."""
        params = {
            HealthDataConstants.KEY_STEPS_GOAL: None,
            HealthDataConstants.KEY_SLEEP_GOAL: None,
        }

        if not self.user_profile:
            return params

        # Try new goals-list path first, then fall back to legacy targets dict
        goals_list = self.user_profile.get("goals") or []
        for goal in goals_list:
            target_val = goal.get("target_value")
            category = goal.get("category_code") or ""
            action = goal.get("action_code") or ""
            if category == "FITNESS" and action == "EXERCISE_STEPS":
                params[HealthDataConstants.KEY_STEPS_GOAL] = target_val
            elif category == "SLEEP":
                params[HealthDataConstants.KEY_SLEEP_GOAL] = target_val

        # Fallback to legacy targets dict for backwards compatibility
        if (
            params[HealthDataConstants.KEY_STEPS_GOAL] is None
            or params[HealthDataConstants.KEY_SLEEP_GOAL] is None
        ):
            targets = self.user_profile.get(TargetKeys.TARGETS) or {}
            if params[HealthDataConstants.KEY_STEPS_GOAL] is None:
                steps_goal_obj = targets.get(TargetKeys.TARGET_STEPS_PER_DAY) or {}
                if isinstance(steps_goal_obj, dict):
                    params[HealthDataConstants.KEY_STEPS_GOAL] = steps_goal_obj.get(
                        TargetKeys.VALUE
                    )
            if params[HealthDataConstants.KEY_SLEEP_GOAL] is None:
                sleep_goal_obj = targets.get(TargetKeys.TARGET_SLEEP_HOURS) or {}
                if isinstance(sleep_goal_obj, dict):
                    params[HealthDataConstants.KEY_SLEEP_GOAL] = sleep_goal_obj.get(
                        TargetKeys.VALUE
                    )

        return params

    async def _calculate_advanced_metrics(self) -> Dict[str, Any]:
        metrics = {}

        if not self._health_history_2w:
            return metrics

        try:
            # Group data by type using cached data
            history_by_type = self._group_history_by_type(self._health_history_2w)

            # Calculate baseline_resting_hr from HR history
            baseline_hr = self._calculate_baseline_resting_hr_from_history(
                history_by_type.get(HealthDataConstants.HR_TYPE, []), weeks=2
            )
            if baseline_hr:
                metrics[HealthDataConstants.KEY_BASELINE_RESTING_HR] = baseline_hr

            # Calculate sleep metrics from SLEEP history
            sleep_metrics = self._calculate_sleep_metrics_from_history(
                history_by_type.get(HealthDataConstants.SLEEP_TYPE, [])
            )
            metrics.update(sleep_metrics)

            # Calculate steps metrics from STEPS history
            steps_metrics = self._calculate_steps_metrics_from_history(
                history_by_type.get(HealthDataConstants.STEPS_TYPE, [])
            )
            metrics.update(steps_metrics)

            # Extract wake_time
            wake_time = self._extract_wake_time()
            if wake_time:
                metrics[HealthDataConstants.KEY_WAKE_TIME] = wake_time

        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate advanced metrics: {str(e)}")

        return metrics

    def _group_history_by_type(
        self, all_history: List[HealthSummary]
    ) -> Dict[str, List[HealthSummary]]:
        """Group history data theo type."""
        history_by_type = {}
        for item in all_history:
            item_type = item.type
            if item_type:
                if item_type not in history_by_type:
                    history_by_type[item_type] = []
                history_by_type[item_type].append(item)
        return history_by_type

    def _calculate_baseline_resting_hr_from_history(
        self, hr_history: List[HRHealthSummary], weeks: int = 2
    ) -> Optional[float]:
        """Tính baseline resting HR từ HR history data (đã được group)."""
        if not hr_history:
            return None

        try:
            # Sort theo dateRange.startDate DESC
            hr_history.sort(
                key=lambda x: x.dateRange.startDate,
                reverse=True,
            )

            # Collect tất cả restingHeartRate values từ N tuần gần nhất
            resting_hrs = []
            for week in hr_history[:weeks]:  # Lấy N tuần gần nhất
                for day_data in week.data:
                    resting_hr = day_data.restingHeartRate
                    if resting_hr and resting_hr > 0:
                        resting_hrs.append(resting_hr)

            if not resting_hrs:
                return None

            # Tính average
            baseline = sum(resting_hrs) / len(resting_hrs)
            return round(baseline, 1)

        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate baseline_resting_hr: {str(e)}")
            return None

    def _calculate_sleep_metrics_from_history(
        self, sleep_history: List[SleepHealthSummary]
    ) -> Dict[str, Any]:
        """Tính toán các sleep metrics từ SLEEP history data (đã được group)."""
        metrics = {}

        if not sleep_history:
            return metrics

        try:
            # Sort theo dateRange.startDate DESC
            sleep_history.sort(
                key=lambda x: x.dateRange.startDate,
                reverse=True,
            )

            # Calculate sleep_last3nights
            sleep_last3_result = self._calculate_sleep_last3nights(sleep_history)
            if sleep_last3_result:
                metrics[HealthDataConstants.KEY_SLEEP_LAST3NIGHTS] = (
                    sleep_last3_result.get(HealthDataConstants.KEY_SLEEP_HOURS)
                )
                # Add flag for all 3 nights have data
                metrics[HealthDataConstants.KEY_ALL_3_NIGHTS_HAVE_DATA] = (
                    sleep_last3_result.get(
                        HealthDataConstants.KEY_ALL_3_NIGHTS_HAVE_DATA, False
                    )
                )

            # Calculate bedtime_streak
            if self.user_profile:
                sleep_goal = self.user_profile.get(TargetKeys.SLEEP_GOAL, {})
                bedtime_start = sleep_goal.get(TargetKeys.BEDTIME_START)
                bedtime_end = sleep_goal.get(TargetKeys.BEDTIME_END)
                if bedtime_start and bedtime_end:
                    streak = self._calculate_bedtime_streak(
                        sleep_history, bedtime_start, bedtime_end
                    )
                    metrics[HealthDataConstants.KEY_BEDTIME_STREAK] = streak

        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate sleep metrics: {str(e)}")

        return metrics

    def _calculate_sleep_last3nights(
        self, sleep_weekly_data: List[SleepHealthSummary]
    ) -> Optional[Dict[str, Any]]:
        if not sleep_weekly_data or not self.time_data:
            return None
        logger.info(f"current_date: {sleep_weekly_data}, tz: {self.time_data}")

        # Get current date và timezone từ time_data
        current_date = self.time_data.get(TimeDataConstants.KEY_CURRENT_DATE)
        tz = self.time_data.get(TimeDataConstants.KEY_TZ)

        if not current_date or not tz:
            return None

        logger.info(f"current_date: {current_date}, tz: {tz}")
        target_dates = [
            current_date - timedelta(days=0),  # Last night (yesterday)
            current_date - timedelta(days=1),  # Night before last
            current_date - timedelta(days=2),  # 2 nights ago
        ]

        # Collect tất cả sleep entries từ tất cả weeks và tạo dict theo date
        sleep_by_date = {}
        for week in sleep_weekly_data:
            for day_data in week.data:
                if day_data.date:
                    try:
                        # Parse date và convert về user timezone
                        entry_date = datetime.fromisoformat(
                            day_data.date.replace("Z", "+00:00")
                        )
                        entry_date = entry_date.astimezone(tz)
                        entry_date_only = entry_date.date()

                        # Lưu vào dict (nếu có total > 0)
                        if day_data.total and day_data.total > 0:
                            sleep_by_date[entry_date_only] = day_data.total
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Failed to parse sleep date {day_data.date}: {str(e)}"
                        )
                        continue

        # Lấy sleep hours cho 3 ngày gần nhất
        sleep_hours = []
        for target_date in target_dates:
            if target_date in sleep_by_date:
                sleep_hours.append(sleep_by_date[target_date])
            else:
                sleep_hours.append(None)

        # Check flag: cả 3 ngày đều có data
        all_3_nights_have_data = all(hour is not None for hour in sleep_hours)

        return {
            HealthDataConstants.KEY_SLEEP_HOURS: sleep_hours,
            HealthDataConstants.KEY_ALL_3_NIGHTS_HAVE_DATA: all_3_nights_have_data,
        }

    def _calculate_bedtime_streak(
        self,
        sleep_weekly_data: List[SleepHealthSummary],
        bedtime_start: str,
        bedtime_end: str,
    ) -> int:
        """Tính bedtime streak (số ngày liên tiếp đi ngủ trong bedtime window)."""
        if not sleep_weekly_data:
            return 0

        # Parse bedtime window
        try:
            bedtime_start_hour, bedtime_start_min = map(int, bedtime_start.split(":"))
            bedtime_end_hour, bedtime_end_min = map(int, bedtime_end.split(":"))
        except:
            logger.warning(f"⚠️ Invalid bedtime format: {bedtime_start} - {bedtime_end}")
            return 0

        # Collect sleep dates
        sleep_dates = []
        for week in sleep_weekly_data:
            for day_data in week.data:
                date_str = day_data.date
                if date_str:
                    sleep_dates.append(date_str)

        if not sleep_dates:
            return 0

        # Sort dates DESC (newest first)
        sleep_dates.sort(reverse=True)

        # Check streak từ ngày gần nhất
        streak = 0
        tz = ZoneInfo(self.timezone)

        for i, date_str in enumerate(sleep_dates):
            try:
                sleep_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                if self.timezone:
                    sleep_date = sleep_date.astimezone(tz)

                if i == 0:  # First (most recent) night
                    streak = 1
                else:
                    # Check if consecutive days
                    prev_date = datetime.fromisoformat(
                        sleep_dates[i - 1].replace("Z", "+00:00")
                    )
                    if self.timezone:
                        prev_date = prev_date.astimezone(tz)

                    # Check if dates are consecutive
                    date_diff = (prev_date.date() - sleep_date.date()).days
                    if date_diff == 1:
                        streak += 1
                    else:
                        break  # Streak broken
            except Exception as e:
                logger.warning(f"⚠️ Error parsing sleep date {date_str}: {str(e)}")
                continue

        return streak

    def _extract_wake_time(self) -> Optional[str]:
        """Extract wake time từ cached sleep data (latest sleep entry)."""
        if not self.health_data:
            return None

        try:
            # Get SLEEP summary from cached data
            sleep_summary = self.health_data.get(HealthDataConstants.SLEEP_TYPE)
            if not sleep_summary or not sleep_summary.data:
                return None

            # Get latest sleep entry
            sorted_entries = sorted(
                sleep_summary.data,
                key=lambda x: x.date,
                reverse=True,
            )

            latest_entry = sorted_entries[0]
            date_str = latest_entry.date

            if not date_str:
                return None

            # Calculate wake time = sleep date + sleep duration
            sleep_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if self.timezone:
                tz = ZoneInfo(self.timezone)
                sleep_date = sleep_date.astimezone(tz)

            total_sleep = latest_entry.total  # hours
            wake_time = sleep_date + timedelta(hours=total_sleep)

            return wake_time.isoformat()
        except Exception as e:
            logger.warning(f"⚠️ Error calculating wake time: {str(e)}")
            return None

    def _calculate_steps_metrics_from_history(
        self, steps_history: List[StepsHealthSummary]
    ) -> Dict[str, Any]:
        """Tính toán các steps metrics từ STEPS history data (đã được group)."""
        metrics = {}

        if not steps_history or not self.user_profile:
            return metrics

        steps_goal = self.user_profile.get(TargetKeys.TARGETS, {}).get(
            TargetKeys.TARGET_STEPS_PER_DAY, {}
        )
        if isinstance(steps_goal, dict):
            steps_goal = steps_goal.get(TargetKeys.VALUE)

        if not steps_goal:
            return metrics

        try:
            # Sort theo dateRange.startDate DESC
            steps_history.sort(
                key=lambda x: x.dateRange.startDate,
                reverse=True,
            )

            # Calculate steps_streak
            streak = self._calculate_steps_streak(steps_history, steps_goal)
            metrics[HealthDataConstants.KEY_STEPS_STREAK] = streak

            # Calculate weekly_health_progress
            weekly_progress = self._calculate_weekly_health_progress(
                steps_history, steps_goal
            )
            if weekly_progress:
                metrics[HealthDataConstants.KEY_WEEKLY_HEALTH_PROGRESS] = (
                    weekly_progress
                )

            # Calculate daily_health_progress
            daily_progress = self._calculate_daily_health_progress(steps_goal)
            if daily_progress:
                metrics[HealthDataConstants.KEY_DAILY_HEALTH_PROGRESS] = daily_progress

        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate steps metrics: {str(e)}")

        return metrics

    def _calculate_steps_streak(
        self, steps_weekly_data: List[StepsHealthSummary], steps_goal: float
    ) -> int:
        """Tính steps streak (số ngày liên tiếp đạt steps goal)."""
        if not steps_weekly_data or not steps_goal:
            return 0

        # Collect steps từ các ngày gần nhất
        steps_days = []

        for week in steps_weekly_data:
            # Sort day data theo date DESC (newest first)
            day_data_list = sorted(
                week.data,
                key=lambda x: x.date,
                reverse=True,
            )

            for day_data in day_data_list:
                steps = day_data.steps
                date_str = day_data.date
                if steps is not None and date_str:
                    steps_days.append(
                        {
                            APIResponseKeys.DATE: date_str,
                            APIResponseKeys.STEPS: steps,
                            HealthDataConstants.KEY_MET_GOAL: steps >= steps_goal,
                        }
                    )

        if not steps_days:
            return 0

        # Check streak từ ngày gần nhất
        streak = 0
        for day_info in steps_days:
            if day_info[HealthDataConstants.KEY_MET_GOAL]:
                streak += 1
            else:
                break  # Streak broken

        return streak

    def _calculate_weekly_health_progress(
        self, steps_weekly_data: List[StepsHealthSummary], steps_goal: float
    ) -> Optional[Dict[str, Any]]:
        """Tính weekly health progress (steps, workouts, etc.)."""
        if not steps_weekly_data:
            return None

        # Lấy tuần gần nhất
        current_week = steps_weekly_data[0] if steps_weekly_data else None
        if not current_week:
            return None

        # Calculate từ data của tuần
        total_steps = 0
        days_met_goal = 0
        total_days = len(current_week.data)

        for day_data in current_week.data:
            steps = day_data.steps
            total_steps += steps
            if steps >= steps_goal:
                days_met_goal += 1

        avg_daily_steps = total_steps / total_days if total_days > 0 else 0
        progress_percentage = (
            (days_met_goal / total_days * 100) if total_days > 0 else 0
        )

        return {
            HealthDataConstants.KEY_TOTAL_STEPS: total_steps,
            HealthDataConstants.KEY_AVG_DAILY_STEPS: round(avg_daily_steps, 1),
            HealthDataConstants.KEY_DAYS_MET_GOAL: days_met_goal,
            HealthDataConstants.KEY_TOTAL_DAYS: total_days,
            HealthDataConstants.KEY_PROGRESS_PERCENTAGE: round(progress_percentage, 1),
        }

    def _calculate_daily_health_progress(
        self, steps_goal: float
    ) -> Optional[Dict[str, Any]]:
        if not steps_goal or not self.health_data:
            return None

        try:
            # Get STEPS summary from cached data
            steps_summary = self.health_data.get(HealthDataConstants.STEPS_TYPE)
            if not steps_summary or not steps_summary.data:
                return None

            today_entry = self._get_latest_data_entry(steps_summary)
            if not today_entry:
                return None

            steps_today = today_entry.get(APIResponseKeys.STEPS)
            if steps_today is None:
                return None

            progress_percentage = (
                (steps_today / steps_goal * 100) if steps_goal > 0 else 0
            )
            remaining_steps = max(0, steps_goal - steps_today)

            return {
                HealthDataConstants.KEY_STEPS_TODAY: steps_today,
                HealthDataConstants.KEY_STEPS_GOAL: steps_goal,
                HealthDataConstants.KEY_PROGRESS_PERCENTAGE: round(
                    progress_percentage, 1
                ),
                HealthDataConstants.KEY_REMAINING_STEPS: round(remaining_steps),
                HealthDataConstants.KEY_MET_GOAL: steps_today >= steps_goal,
            }
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate daily health progress: {str(e)}")
            return None

    def _calculate_sleep_quality(
        self, sleep_quality_score: Optional[int]
    ) -> Optional[str]:
        if sleep_quality_score is None:
            return None

        try:
            # So sánh với các ngưỡng (theo thứ tự từ cao xuống thấp)
            if sleep_quality_score >= 96:
                return "very_high"
            elif sleep_quality_score >= 81:
                return "high"
            elif sleep_quality_score >= 61:
                return "ok"
            elif sleep_quality_score >= 41:
                return "low"
            else:
                return "very_low"
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate sleep quality: {str(e)}")
            return None

    def _get_thresholds(self) -> Dict[str, Any]:
        """Trả về các thresholds và constants."""
        return {
            HealthDataConstants.KEY_QUALITY_THRESHOLD: HealthDataConstants.SLEEP_QUALITY_THRESHOLD,
            HealthDataConstants.KEY_QUALITY_GOOD: HealthDataConstants.SLEEP_QUALITY_GOOD,
            HealthDataConstants.KEY_QUALITY_HIGH: HealthDataConstants.SLEEP_QUALITY_HIGH,
            HealthDataConstants.KEY_QUALITY_VERY_HIGH: HealthDataConstants.SLEEP_QUALITY_VERY_HIGH,
            HealthDataConstants.KEY_QUALITY_LOW: HealthDataConstants.SLEEP_QUALITY_LOW,
            HealthDataConstants.KEY_WIND_DOWN_BUFFER_MINS: HealthDataConstants.DEFAULT_WIND_DOWN_BUFFER_MINS,
        }

    def _get_latest_data_entry(
        self,
        summary: Optional[HealthSummary],
    ) -> Optional[Dict[str, Any]]:
        if not summary or not summary.data:
            return None

        # Convert Pydantic model entries to dicts
        data_array = []
        for entry in summary.data:
            if hasattr(entry, "model_dump"):
                data_array.append(entry.model_dump())
            elif isinstance(entry, dict):
                data_array.append(entry)
            else:
                # Fallback: convert to dict using __dict__ or attributes
                try:
                    data_array.append(dict(entry))
                except Exception:
                    continue

        if not data_array:
            return None

        # Get current date from time_data if available
        current_date = None
        timezone = None
        if self.time_data:
            current_date = self.time_data.get(TimeDataConstants.KEY_CURRENT_DATE)
            timezone = self.time_data.get(TimeDataConstants.KEY_TIMEZONE)

        # Filter and sort data
        try:
            filtered_data = []
            for entry in data_array:
                entry_date_str = entry.get(APIResponseKeys.DATE, "")
                if not entry_date_str:
                    continue

                # If we have current_date, filter by matching date
                if current_date:
                    try:
                        # Parse entry date
                        entry_dt = datetime.fromisoformat(
                            entry_date_str.replace("Z", "+00:00")
                        )
                        # Convert to target timezone
                        entry_dt = entry_dt.astimezone(ZoneInfo(timezone))
                        entry_date = entry_dt.date()
                        # Only include entries matching current date
                        if entry_date == current_date:
                            filtered_data.append(entry)
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Failed to parse entry date {entry_date_str}: {str(e)}"
                        )
                        # If parsing fails, include it anyway (fallback)
                        filtered_data.append(entry)
                else:
                    tz = ZoneInfo(self.timezone)
                    actual_today = datetime.now(tz).date()
                    try:
                        entry_dt = datetime.fromisoformat(
                            entry_date_str.replace("Z", "+00:00")
                        )
                        entry_dt = entry_dt.astimezone(tz)
                        entry_date = entry_dt.date()
                        if entry_date == actual_today:
                            filtered_data.append(entry)
                    except Exception:
                        pass

            # Sort by date (descending) to get latest entry first
            if filtered_data:
                sorted_data = sorted(
                    filtered_data,
                    key=lambda x: x.get(APIResponseKeys.DATE, ""),
                    reverse=True,
                )
                return sorted_data[0] if sorted_data else None
            else:
                return None
            return None
        except Exception as e:
            logger.warning(f"⚠️ Failed to filter/sort data entries: {str(e)}")
            return None
