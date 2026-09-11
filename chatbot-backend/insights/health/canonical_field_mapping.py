from __future__ import annotations

from copy import deepcopy
from typing import Any, Final, Mapping


class CanonicalField:
    # User profile
    USER_LANGUAGE_AND_LOCALE = "user_language_and_locale"
    USER_HEIGHT_CENTIMETERS = "user_height_centimeters"
    USER_WEIGHT_KILOGRAMS = "user_weight_kilograms"
    USER_DATE_OF_BIRTH = "user_date_of_birth"
    USER_JOB_TITLE = "user_job_title"
    USER_EMPLOYMENT_TYPE = "user_employment_type"
    USER_WORK_STATUS = "user_work_status"
    USER_ACTIVE_HOURS_START_TIME = "user_active_hours_start_time"
    USER_ACTIVE_HOURS_END_TIME = "user_active_hours_end_time"
    USER_BEDTIME_WINDOW_START = "user_bedtime_window_start"
    USER_BEDTIME_WINDOW_END = "user_bedtime_window_end"
    USER_PRIMARY_HEALTH_GOAL = "user_primary_health_goal"
    USER_CURRENT_HEALTH_CONDITIONS = "user_current_health_conditions"
    USER_AGE_YEARS = "user_age_years"
    USER_GENDER = "user_gender"
    USER_BODY_MASS_INDEX = "user_body_mass_index"
    USER_INDUSTRY = "user_industry"
    USER_HEALTH_GOALS = "user_goals"
    USER_TIMEZONE = "user_timezone"

    # Fields inside user_goals[]
    GOAL_CATEGORY = "goal_category"
    GOAL_ACTION = "goal_action"
    GOAL_NAME = "goal_name"
    GOAL_TARGET_VALUE = "goal_target_value"
    GOAL_UNIT = "goal_unit"

    # Daily snapshot
    DAILY_SNAPSHOT_DATE = "daily_snapshot_date"
    DAILY_BEDTIME_LOCAL_DATETIME = "daily_bedtime_local_datetime"
    DAILY_WAKE_TIME_LOCAL_DATETIME = "daily_wake_time_local_datetime"
    DAILY_STEP_COUNT = "daily_step_count"
    DAILY_ACTIVE_MINUTES = "daily_active_minutes"
    DAILY_CALORIES_BURNED_KCAL = "daily_calories_burned_kcal"
    DAILY_EXERCISE_SESSION_COUNT = "daily_exercise_session_count"
    DAILY_WORKOUT_DURATION_MINUTES = "daily_workout_duration_minutes"
    DAILY_AVERAGE_HEART_RATE_BPM = "daily_average_heart_rate_bpm"
    DAILY_RESTING_HEART_RATE_BPM = "daily_resting_heart_rate_bpm"
    DAILY_SLEEP_DURATION_HOURS = "daily_sleep_duration_hours"
    DAILY_SLEEP_QUALITY_SCORE = "daily_sleep_quality_score"
    DAILY_SLEEP_SCORE = "daily_sleep_score"
    DAILY_WATER_INTAKE_LITERS = "daily_water_intake_liters"
    DAILY_CALORIES_CONSUMED_KCAL = "daily_calories_consumed_kcal"
    DAILY_BODY_WEIGHT_KG = "daily_body_weight_kg"
    DAILY_HEALTH_SCORE = "daily_health_score"
    DAILY_STEPS_GOAL_COMPLETION_PERCENTAGE = ("daily_steps_goal_completion_percentage")
    DAILY_ACTIVITY_LEVEL = "daily_activity_level"
    DAILY_ACTIVE_HEALTH_CONDITIONS = "daily_active_health_conditions"
    DAILY_MEETING_DURATION_MINUTES = "daily_meeting_duration_minutes"
    DAILY_LONGEST_MEETING_DURATION_MINUTES = ("daily_longest_meeting_duration_minutes")
    DAILY_WORK_SPAN_MINUTES = "daily_work_span_minutes"
    DAILY_EVENTS_AFTER_9_PM_COUNT = "daily_events_after_9_pm_count"
    DAILY_WELLNESS_SCORE = "daily_wellness_score"

    # Balance
    BALANCE_SCORE_DATE = "balance_score_date"
    BALANCE_HEALTH_SCORE = "balance_health_score"

    # Calendar / reminders / events / work-hours
    CALENDAR_EVENT_TITLE = "calendar_event_title"
    CALENDAR_EVENT_COMPLETION_STATUS = "calendar_event_completion_status"
    CALENDAR_EVENT_START_TIME = "calendar_event_start_time"
    CALENDAR_EVENT_END_TIME = "calendar_event_end_time"
    CALENDAR_EVENT_TYPE = "calendar_event_type"
    REMINDER_TITLE = "reminder_title"
    REMINDER_DUE_TIME = "reminder_due_time"
    REMINDER_COMPLETION_STATUS = "reminder_completion_status"
    REMINDER_RECURRENCE_FREQUENCY = "reminder_recurrence_frequency"
    REMINDER_RECURRENCE_INTERVAL = "reminder_recurrence_interval"
    WORK_HOURS_DATE = "work_hours_date"
    SCHEDULED_WORK_HOURS = "scheduled_work_hours"
    SCHEDULED_WORK_EVENT_COUNT = "scheduled_work_event_count"

    # Mood
    CURRENT_MOOD = "current_mood"
    CURRENT_MOOD_DATE = "current_mood_date"

    # Productivity
    MEETING_TIME_PERCENTAGE_TODAY = "meeting_time_percentage_today"
    MEETING_COUNT_TODAY = "meeting_count_today"
    COMPLETED_MEETING_COUNT_TODAY = "completed_meeting_count_today"
    REMINDER_COUNT_TODAY = "reminder_count_today"
    COMPLETED_REMINDER_COUNT_TODAY = "completed_reminder_count_today"

    # Health summaries — flat fields (used by Phase 2 selectors)
    STEP_COUNT_ONE_DAY = "step_count_one_day"
    ACTIVE_MINUTES_ONE_DAY = "active_minutes_one_day"
    TOTAL_CALORIES_BURNED_ONE_DAY_KCAL = "total_calories_burned_one_day_kcal"
    EXERCISE_SESSION_COUNT_ONE_DAY = "exercise_session_count_one_day"
    TOTAL_WORKOUT_DURATION_ONE_DAY_MINUTES = "total_workout_duration_one_day_minutes"
    AVERAGE_HEART_RATE_ONE_DAY_BPM = "average_heart_rate_one_day_bpm"
    RESTING_HEART_RATE_ONE_DAY_BPM = "resting_heart_rate_one_day_bpm"
    SLEEP_DURATION_ONE_NIGHT_HOURS = "sleep_duration_one_night_hours"
    SLEEP_QUALITY_SCORE_ONE_NIGHT = "sleep_quality_score_one_night"
    WATER_INTAKE_ONE_DAY_LITERS = "water_intake_one_day_liters"
    CALORIES_CONSUMED_ONE_DAY_KCAL = "calories_consumed_one_day_kcal"
    BODY_WEIGHT_ONE_DAY_KG = "body_weight_one_day_kg"
    HEALTH_SCORE_ONE_DAY = "health_score_one_day"
    STEPS_GOAL_COMPLETION_PERCENTAGE_ONE_DAY = "steps_goal_completion_percentage_one_day"
    ACTIVITY_LEVEL_ONE_DAY = "activity_level_one_day"
    ACTIVE_HEALTH_CONDITIONS_ONE_DAY = "active_health_conditions_one_day"
    MEETING_DURATION_ONE_DAY_MINUTES = "meeting_duration_one_day_minutes"
    LONGEST_MEETING_DURATION_ONE_DAY_MINUTES = "longest_meeting_duration_one_day_minutes"
    WORK_SPAN_ONE_DAY_MINUTES = "work_span_one_day_minutes"
    EVENTS_AFTER_9_PM_ONE_DAY_COUNT = "events_after_9_pm_one_day_count"
    WELLNESS_SCORE_ONE_DAY = "wellness_score_one_day"
    TOTAL_EVENTS_ONE_DAY = "total_events_one_day"

    # Health summaries — ENERGY
    ENERGY_BURN_ONE_DAY_KCAL = "energy_burn_one_day_kcal"
    ACTIVE_ENERGY_ONE_DAY_KCAL = "active_energy_one_day_kcal"
    RESTING_ENERGY_ONE_DAY_KCAL = "resting_energy_one_day_kcal"
    AVERAGE_DAILY_ACTIVE_ENERGY_BURN_CURRENT_MONTH_KCAL = (
        "average_daily_active_energy_burn_current_month_kcal"
    )
    AVERAGE_DAILY_ACTIVE_ENERGY_BURN_CURRENT_WEEK_KCAL = (
        "average_daily_active_energy_burn_current_week_kcal"
    )
    AVERAGE_DAILY_ACTIVE_ENERGY_BURN_PREVIOUS_MONTH_KCAL = (
        "average_daily_active_energy_burn_previous_month_kcal"
    )

    # Health summaries — HR
    MOST_RECENT_HEART_RATE_MEASUREMENT_ONE_DAY_BPM = (
        "most_recent_heart_rate_measurement_one_day_bpm"
    )
    MINIMUM_HEART_RATE_CURRENT_SUMMARY_PERIOD_BPM = (
        "minimum_heart_rate_current_summary_period_bpm"
    )
    MAXIMUM_HEART_RATE_CURRENT_SUMMARY_PERIOD_BPM = (
        "maximum_heart_rate_current_summary_period_bpm"
    )
    WALKING_HEART_RATE_ONE_DAY_BPM = "walking_heart_rate_one_day_bpm"
    WORKOUT_HEART_RATE_ONE_DAY = "workout_heart_rate_one_day"
    SLEEP_HEART_RATE_ONE_NIGHT_BPM = "sleep_heart_rate_one_night_bpm"
    AVERAGE_HEART_RATE_VARIABILITY_CURRENT = (
        "average_heart_rate_variability_current"
    )

    # Health summaries — SLEEP
    REM_SLEEP_DURATION_ONE_NIGHT_HOURS = "rem_sleep_duration_one_night_hours"
    CORE_SLEEP_DURATION_ONE_NIGHT_HOURS = "core_sleep_duration_one_night_hours"
    DEEP_SLEEP_DURATION_ONE_NIGHT_HOURS = "deep_sleep_duration_one_night_hours"
    SLEEP_DURATION_FROM_SUMMARY_ONE_NIGHT_HOURS = (
        "sleep_duration_from_summary_one_night_hours"
    )
    SLEEP_SCORE_ONE_NIGHT = "sleep_score_one_night"
    LAST_WAKE_TIME_ONE_NIGHT_LOCAL_DATETIME = (
        "last_wake_time_one_night_local_datetime"
    )
    FIRST_SLEEP_TIME_ONE_NIGHT_LOCAL_DATETIME = (
        "first_sleep_time_one_night_local_datetime"
    )

    # Sleep summary averages 
    AVERAGE_LIGHT_SLEEP_HOURS = "average_light_sleep_hours"
    AVERAGE_AWAKE_HOURS = "average_awake_hours"
    AVERAGE_TIME_IN_BED_HOURS = "average_time_in_bed_hours"
    AVERAGE_TOTAL_SLEEP_HOURS = "average_total_sleep_hours"
    AVERAGE_DEEP_SLEEP_HOURS = "average_deep_sleep_hours"
    AVERAGE_REM_SLEEP_HOURS = "average_rem_sleep_hours"

    # Health summaries — STEPS
    DISTANCE_ONE_DAY = "distance_one_day"
    DISTANCE_ONE_DAY_SUMMARY = "distance_one_day_summary"
    STEP_COUNT_ONE_DAY_SUMMARY = "step_count_one_day_summary"
    AVERAGE_DAILY_STEPS_CURRENT_MONTH = "average_daily_steps_current_month"
    AVERAGE_DAILY_STEPS_PREVIOUS_MONTH = "average_daily_steps_previous_month"
    DAILY_STEPS_TARGET = "daily_steps_target"

    # Onboarding / profile
    USER_LANGUAGE = "user_language"
    USER_PRIMARY_GOAL = "user_primary_goal"
    USER_GOAL_CATEGORY = "user_goal_category"
    USER_GOAL_ACTION = "user_goal_action"
    USER_GOAL_NAME = "user_goal_name"
    USER_GOAL_TARGET_VALUE = "user_goal_target_value"
    USER_GOAL_UNIT = "user_goal_unit"

    # Derived: health_params (built by PrepareHealthData)
    HEALTH_PARAMS_STEPS_TODAY = "health_params_steps_today"
    HEALTH_PARAMS_SLEEP_LASTNIGHT = "health_params_sleep_lastnight"
    HEALTH_PARAMS_SLEEP_QUALITY_SCORE = "health_params_sleep_quality_score"
    HEALTH_PARAMS_SLEEP_QUALITY = "health_params_sleep_quality"
    HEALTH_PARAMS_LATEST_HEART_RATE = "health_params_latest_heart_rate"
    HEALTH_PARAMS_RESTING_HEART_RATE = "health_params_resting_heart_rate"
    HEALTH_PARAMS_LAST_WAKE_TIME = "health_params_last_wake_time"
    HEALTH_PARAMS_FIRST_SLEEP_TIME = "health_params_first_sleep_time"
    HEALTH_PARAMS_BEDTIME = "health_params_bedtime"
    HEALTH_PARAMS_WORKOUT_HEART_RATE = "health_params_workout_heart_rate"
    HEALTH_PARAMS_WORKOUT_AVG_HR = "health_params_workout_avg_hr"
    HEALTH_PARAMS_WORKOUT_MAX_HR = "health_params_workout_max_hr"
    HEALTH_PARAMS_WORKOUT_MIN_HR = "health_params_workout_min_hr"
    HEALTH_PARAMS_WORKOUT_DURATION = "health_params_workout_duration"
    HEALTH_PARAMS_WORKOUT_ACTIVITY_TYPE = "health_params_workout_activity_type"
    HEALTH_PARAMS_STEPS_GOAL = "health_params_steps_goal"
    HEALTH_PARAMS_SLEEP_GOAL = "health_params_sleep_goal"
    HEALTH_PARAMS_BASELINE_RESTING_HR = "health_params_baseline_resting_hr"
    HEALTH_PARAMS_HRV_SCORE = "health_params_hrv_score"
    HEALTH_PARAMS_BEDTIME_STREAK = "health_params_bedtime_streak"
    HEALTH_PARAMS_WAKE_TIME = "health_params_wake_time"
    HEALTH_PARAMS_SLEEP_LAST3NIGHTS = "health_params_sleep_last3nights"
    HEALTH_PARAMS_SLEEP_HOURS = "health_params_sleep_hours"
    HEALTH_PARAMS_ALL_3_NIGHTS_HAVE_DATA = "health_params_all_3_nights_have_data"
    HEALTH_PARAMS_DEEP_SLEEP_MIN = "health_params_deep_sleep_min"
    HEALTH_PARAMS_REM_SLEEP_MIN = "health_params_rem_sleep_min"
    HEALTH_PARAMS_LIGHT_SLEEP_MIN = "health_params_light_sleep_min"
    HEALTH_PARAMS_STEPS_STREAK = "health_params_steps_streak"
    HEALTH_PARAMS_WEEKLY_HEALTH_PROGRESS = "health_params_weekly_health_progress"
    HEALTH_PARAMS_DAILY_HEALTH_PROGRESS = "health_params_daily_health_progress"
    HEALTH_PARAMS_QUALITY_THRESHOLD = "health_params_quality_threshold"
    HEALTH_PARAMS_QUALITY_GOOD = "health_params_quality_good"
    HEALTH_PARAMS_QUALITY_HIGH = "health_params_quality_high"
    HEALTH_PARAMS_QUALITY_VERY_HIGH = "health_params_quality_very_high"
    HEALTH_PARAMS_QUALITY_LOW = "health_params_quality_low"
    HEALTH_PARAMS_WIND_DOWN_BUFFER_MINS = "health_params_wind_down_buffer_mins"
    HEALTH_PARAMS_PROGRESS_PERCENTAGE = "health_params_progress_percentage"
    HEALTH_PARAMS_REMAINING_STEPS = "health_params_remaining_steps"
    HEALTH_PARAMS_MET_GOAL = "health_params_met_goal"
    HEALTH_PARAMS_TOTAL_STEPS = "health_params_total_steps"
    HEALTH_PARAMS_AVG_DAILY_STEPS = "health_params_avg_daily_steps"
    HEALTH_PARAMS_DAYS_MET_GOAL = "health_params_days_met_goal"
    HEALTH_PARAMS_TOTAL_DAYS = "health_params_total_days"
    HEALTH_PARAMS_DAILY_HEALTH_SCORE_TREND = "health_params_daily_health_score_trend"
    HEALTH_PARAMS_SLEEP_HR_RANGE = "health_params_sleep_hr_range"
    HEALTH_PARAMS_WALKING_HEART_RATE_AVG = "health_params_walking_heart_rate_avg"
    HEALTH_PARAMS_AVERAGE_AWAKE_HOURS = "health_params_average_awake_hours"
    HEALTH_PARAMS_AVERAGE_TIME_IN_BED_HOURS = "health_params_average_time_in_bed_hours"
    HEALTH_PARAMS_AVERAGE_TOTAL_SLEEP_HOURS = "health_params_average_total_sleep_hours"
    HEALTH_PARAMS_CALORIES_BURNED_TODAY = "health_params_calories_burned_today"
    HEALTH_PARAMS_ACTIVE_MINUTES_TODAY = "health_params_active_minutes_today"
    HEALTH_PARAMS_ENERGY_LAST_DATA_DATE = "health_params_energy_last_data_date"
    HEALTH_PARAMS_ENERGY_WEEK_VS_PRIOR_MONTH_PCT = (
        "health_params_energy_week_vs_prior_month_pct"
    )
    HEALTH_PARAMS_ENERGY_MONTH_RESTING_TOTAL = (
        "health_params_energy_month_resting_total"
    )
    HEALTH_PARAMS_TOTAL_ACTIVE_ENERGY = "health_params_total_active_energy"
    HEALTH_PARAMS_ENERGY_CURRENT_MONTH_AVG = "health_params_energy_current_month_avg"
    HEALTH_PARAMS_ENERGY_WEEK_AVG = "health_params_energy_week_avg"
    HEALTH_PARAMS_ENERGY_PREVIOUS_MONTH_AVG = "health_params_energy_previous_month_avg"
    HEALTH_PARAMS_STEPS_LAST_DATA_DATE = "health_params_steps_last_data_date"
    HEALTH_PARAMS_SLEEP_LAST_DATA_DATE = "health_params_sleep_last_data_date"
    HEALTH_PARAMS_HR_LAST_DATA_DATE = "health_params_hr_last_data_date"
    HEALTH_PARAMS_MOOD_LAST_DATA_DATE = "health_params_mood_last_data_date"
    HEALTH_PARAMS_STRESS_HIGH = "health_params_stress_high"
    HEALTH_PARAMS_STRESS_SIGNAL_HIGH = "health_params_stress_signal_high"

    # Derived: time_data (built by PrepareTimeData)
    TIME_DATA_CURRENT_DT = "time_data_current_dt"
    TIME_DATA_CURRENT_HOUR = "time_data_current_hour"
    TIME_DATA_CURRENT_MINUTE = "time_data_current_minute"
    TIME_DATA_IS_WEEKEND = "time_data_is_weekend"
    TIME_DATA_TZ = "time_data_tz"
    TIME_DATA_TIMEZONE = "time_data_timezone"
    TIME_DATA_CURRENT_DATE = "time_data_current_date"
    TIME_DATA_CURRENT_TIME_ISO = "time_data_current_time_iso"
    TIME_DATA_BEDTIME_START_STR = "time_data_bedtime_start_str"
    TIME_DATA_BEDTIME_END_STR = "time_data_bedtime_end_str"
    TIME_DATA_ACTIVE_START_STR = "time_data_active_start_str"
    TIME_DATA_ACTIVE_END_STR = "time_data_active_end_str"
    TIME_DATA_SLEEP_TARGET = "time_data_sleep_target"
    TIME_DATA_TIME_TO_BEDTIME = "time_data_time_to_bedtime"
    TIME_DATA_TIME_TO_NEXT_EVENT = "time_data_time_to_next_event"
    TIME_DATA_IN_ACTIVE_WINDOW = "time_data_in_active_window"
    TIME_DATA_FREE_SLOT_LENGTH = "time_data_free_slot_length"

    # Derived: calendar_metrics (built by PrepareCalendarData)
    CALENDAR_METRICS_BACK_TO_BACK_COUNT = "calendar_metrics_back_to_back_count"
    CALENDAR_METRICS_MEETING_MINUTES = "calendar_metrics_meeting_minutes"
    CALENDAR_METRICS_CALENDAR_DENSITY = "calendar_metrics_calendar_density"
    CALENDAR_METRICS_CONTINUOUS_EVENTS_MINUTES = (
        "calendar_metrics_continuous_events_minutes"
    )
    CALENDAR_METRICS_WORK_EVENTS_HOURS_WEEKEND = (
        "calendar_metrics_work_events_hours_weekend"
    )
    CALENDAR_METRICS_WORK_EVENTS_HOURS_WEEK = (
        "calendar_metrics_work_events_hours_week"
    )
    CALENDAR_METRICS_WORK_LOAD_HIGH = "calendar_metrics_work_load_high"


PROFILE_CANONICAL_FIELD_MAP = {
    "language": CanonicalField.USER_LANGUAGE,
    "height_cm": CanonicalField.USER_HEIGHT_CENTIMETERS,
    "weight_kg": CanonicalField.USER_WEIGHT_KILOGRAMS,
    "date_of_birth": CanonicalField.USER_DATE_OF_BIRTH,
    "job_title": CanonicalField.USER_JOB_TITLE,
    "employment_type": CanonicalField.USER_EMPLOYMENT_TYPE,
    "work_status": CanonicalField.USER_WORK_STATUS,
    "active_hours_start_time": CanonicalField.USER_ACTIVE_HOURS_START_TIME,
    "active_hours_end_time": CanonicalField.USER_ACTIVE_HOURS_END_TIME,
    "bedtime_start": CanonicalField.USER_BEDTIME_WINDOW_START,
    "bedtime_end": CanonicalField.USER_BEDTIME_WINDOW_END,
    "primary_goal": CanonicalField.USER_PRIMARY_HEALTH_GOAL,
    "current_health_conditions": CanonicalField.USER_CURRENT_HEALTH_CONDITIONS,
    "age": CanonicalField.USER_AGE_YEARS,
    "gender": CanonicalField.USER_GENDER,
    "bmi": CanonicalField.USER_BODY_MASS_INDEX,
    "industry": CanonicalField.USER_INDUSTRY,
    "goals": CanonicalField.USER_HEALTH_GOALS,
    "timezone": CanonicalField.USER_TIMEZONE,
}

PROFILE_GOAL_CANONICAL_FIELD_MAP = {
    "category_code": CanonicalField.GOAL_CATEGORY,
    "action_code": CanonicalField.GOAL_ACTION,
    "goal_name": CanonicalField.GOAL_NAME,
    "target_value": CanonicalField.GOAL_TARGET_VALUE,
    "unit": CanonicalField.GOAL_UNIT,
}

DAILY_SNAPSHOT_CANONICAL_FIELD_MAP = {
    "snapshot_date": CanonicalField.DAILY_SNAPSHOT_DATE,
    "h_steps": CanonicalField.STEP_COUNT_ONE_DAY,
    "h_active_minutes": CanonicalField.ACTIVE_MINUTES_ONE_DAY,
    "h_calories_burned": CanonicalField.TOTAL_CALORIES_BURNED_ONE_DAY_KCAL,
    "h_exercise_sessions": CanonicalField.EXERCISE_SESSION_COUNT_ONE_DAY,
    "h_total_workout_min": CanonicalField.TOTAL_WORKOUT_DURATION_ONE_DAY_MINUTES,
    "h_avg_heart_rate": CanonicalField.AVERAGE_HEART_RATE_ONE_DAY_BPM,
    "h_resting_heart_rate": CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM,
    "h_sleep_hours": CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS,
    "h_sleep_quality": CanonicalField.SLEEP_QUALITY_SCORE_ONE_NIGHT,
    "h_sleep_score": CanonicalField.SLEEP_SCORE_ONE_NIGHT,
    "h_water_liters": CanonicalField.WATER_INTAKE_ONE_DAY_LITERS,
    "h_calories_consumed": CanonicalField.CALORIES_CONSUMED_ONE_DAY_KCAL,
    "h_weight_kg": CanonicalField.BODY_WEIGHT_ONE_DAY_KG,
    "h_health_score": CanonicalField.HEALTH_SCORE_ONE_DAY,
    "h_steps_goal_pct": CanonicalField.STEPS_GOAL_COMPLETION_PERCENTAGE_ONE_DAY,
    "h_activity_level": CanonicalField.ACTIVITY_LEVEL_ONE_DAY,
    "h_active_conditions": CanonicalField.ACTIVE_HEALTH_CONDITIONS_ONE_DAY,
    "p_meeting_minutes": CanonicalField.MEETING_DURATION_ONE_DAY_MINUTES,
    "p_longest_meeting_min": CanonicalField.LONGEST_MEETING_DURATION_ONE_DAY_MINUTES,
    "p_work_span_minutes": CanonicalField.WORK_SPAN_ONE_DAY_MINUTES,
    "p_events_after_9pm": CanonicalField.EVENTS_AFTER_9_PM_ONE_DAY_COUNT,
    "wellness_score": CanonicalField.WELLNESS_SCORE_ONE_DAY,
}

BALANCE_CANONICAL_FIELD_MAP = {
    # balance — today
    ("balance_score", "date"): CanonicalField.BALANCE_SCORE_DATE,
    ("balance_score", "healthScore"): CanonicalField.BALANCE_HEALTH_SCORE,

    # balance — 30-day history (range fetch for baseline_30d)
    ("balance_scores_30d", "date"): CanonicalField.BALANCE_SCORE_DATE,
    ("balance_scores_30d", "healthScore"): CanonicalField.BALANCE_HEALTH_SCORE,
    # balance — 7-day history (legacy alias kept for back-compat)
    ("balance_scores_7d", "date"): CanonicalField.BALANCE_SCORE_DATE,
    ("balance_scores_7d", "healthScore"): CanonicalField.BALANCE_HEALTH_SCORE,
}

MOOD_DATA_CANONICAL_FIELD_MAP = {
    "mood": CanonicalField.CURRENT_MOOD,
    "mood_date": CanonicalField.CURRENT_MOOD_DATE,
}

TIME_DATA_CANONICAL_FIELD_MAP = {
    "current_dt": CanonicalField.TIME_DATA_CURRENT_DT,
    "current_hour": CanonicalField.TIME_DATA_CURRENT_HOUR,
    "current_minute": CanonicalField.TIME_DATA_CURRENT_MINUTE,
    "is_weekend": CanonicalField.TIME_DATA_IS_WEEKEND,
    "tz": CanonicalField.TIME_DATA_TZ,
    "timezone": CanonicalField.TIME_DATA_TIMEZONE,
    "current_date": CanonicalField.TIME_DATA_CURRENT_DATE,
    "current_time_iso": CanonicalField.TIME_DATA_CURRENT_TIME_ISO,
    "bedtime_start_str": CanonicalField.TIME_DATA_BEDTIME_START_STR,
    "bedtime_end_str": CanonicalField.TIME_DATA_BEDTIME_END_STR,
    "active_start_str": CanonicalField.TIME_DATA_ACTIVE_START_STR,
    "active_end_str": CanonicalField.TIME_DATA_ACTIVE_END_STR,
    "sleep_target": CanonicalField.TIME_DATA_SLEEP_TARGET,
    "time_to_bedtime": CanonicalField.TIME_DATA_TIME_TO_BEDTIME,
    "time_to_next_event": CanonicalField.TIME_DATA_TIME_TO_NEXT_EVENT,
    "in_active_window": CanonicalField.TIME_DATA_IN_ACTIVE_WINDOW,
    "free_slot_length": CanonicalField.TIME_DATA_FREE_SLOT_LENGTH,
}

CALENDAR_METRICS_CANONICAL_FIELD_MAP = {
    "back_to_back_count": CanonicalField.CALENDAR_METRICS_BACK_TO_BACK_COUNT,
    "meeting_minutes": CanonicalField.CALENDAR_METRICS_MEETING_MINUTES,
    "calendar_density": CanonicalField.CALENDAR_METRICS_CALENDAR_DENSITY,
    "continuous_events_minutes": (
        CanonicalField.CALENDAR_METRICS_CONTINUOUS_EVENTS_MINUTES
    ),
    "work_events_hours_weekend": (
        CanonicalField.CALENDAR_METRICS_WORK_EVENTS_HOURS_WEEKEND
    ),
    "work_events_hours_week": CanonicalField.CALENDAR_METRICS_WORK_EVENTS_HOURS_WEEK,
    "work_load_high": CanonicalField.CALENDAR_METRICS_WORK_LOAD_HIGH,
}

HEALTH_PARAMS_CANONICAL_FIELD_MAP = {
    # Basic (PrepareHealthData._extract_basic_health_data)
    "steps_today": CanonicalField.HEALTH_PARAMS_STEPS_TODAY,
    "sleep_lastnight": CanonicalField.HEALTH_PARAMS_SLEEP_LASTNIGHT,
    "sleep_quality_score": CanonicalField.HEALTH_PARAMS_SLEEP_QUALITY_SCORE,
    "sleep_quality": CanonicalField.HEALTH_PARAMS_SLEEP_QUALITY,
    "latest_heart_rate": CanonicalField.HEALTH_PARAMS_LATEST_HEART_RATE,
    "resting_heart_rate": CanonicalField.HEALTH_PARAMS_RESTING_HEART_RATE,
    "last_wake_time": CanonicalField.HEALTH_PARAMS_LAST_WAKE_TIME,
    "first_sleep_time": CanonicalField.HEALTH_PARAMS_FIRST_SLEEP_TIME,
    "workout_heart_rate": CanonicalField.HEALTH_PARAMS_WORKOUT_HEART_RATE,
    "workout_avg_hr": CanonicalField.HEALTH_PARAMS_WORKOUT_AVG_HR,
    "workout_max_hr": CanonicalField.HEALTH_PARAMS_WORKOUT_MAX_HR,
    "workout_min_hr": CanonicalField.HEALTH_PARAMS_WORKOUT_MIN_HR,
    "workout_duration": CanonicalField.HEALTH_PARAMS_WORKOUT_DURATION,
    "workout_activity_type": CanonicalField.HEALTH_PARAMS_WORKOUT_ACTIVITY_TYPE,
    "calories_burned_today": CanonicalField.HEALTH_PARAMS_CALORIES_BURNED_TODAY,
    "active_minutes_today": CanonicalField.HEALTH_PARAMS_ACTIVE_MINUTES_TODAY,
    "steps_last_data_date": CanonicalField.HEALTH_PARAMS_STEPS_LAST_DATA_DATE,
    "sleep_last_data_date": CanonicalField.HEALTH_PARAMS_SLEEP_LAST_DATA_DATE,
    "hr_last_data_date": CanonicalField.HEALTH_PARAMS_HR_LAST_DATA_DATE,
    "mood_last_data_date": CanonicalField.HEALTH_PARAMS_MOOD_LAST_DATA_DATE,
    "energy_last_data_date": CanonicalField.HEALTH_PARAMS_ENERGY_LAST_DATA_DATE,
    "bedtime": CanonicalField.HEALTH_PARAMS_BEDTIME,
    # Goals
    "steps_goal": CanonicalField.HEALTH_PARAMS_STEPS_GOAL,
    "sleep_goal": CanonicalField.HEALTH_PARAMS_SLEEP_GOAL,
    # Advanced metrics
    "baseline_resting_hr": CanonicalField.HEALTH_PARAMS_BASELINE_RESTING_HR,
    "hrv_score": CanonicalField.HEALTH_PARAMS_HRV_SCORE,
    "bedtime_streak": CanonicalField.HEALTH_PARAMS_BEDTIME_STREAK,
    "sleep_streak": CanonicalField.HEALTH_PARAMS_SLEEP_LAST3NIGHTS,
    "wake_time": CanonicalField.HEALTH_PARAMS_WAKE_TIME,
    "sleep_last3nights": CanonicalField.HEALTH_PARAMS_SLEEP_LAST3NIGHTS,
    "sleep_hours": CanonicalField.HEALTH_PARAMS_SLEEP_HOURS,
    "all_3_nights_have_data": CanonicalField.HEALTH_PARAMS_ALL_3_NIGHTS_HAVE_DATA,
    "deep_sleep_min": CanonicalField.HEALTH_PARAMS_DEEP_SLEEP_MIN,
    "rem_sleep_min": CanonicalField.HEALTH_PARAMS_REM_SLEEP_MIN,
    "light_sleep_min": CanonicalField.HEALTH_PARAMS_LIGHT_SLEEP_MIN,
    "steps_streak": CanonicalField.HEALTH_PARAMS_STEPS_STREAK,
    # Progress / nested
    "weekly_health_progress": CanonicalField.HEALTH_PARAMS_WEEKLY_HEALTH_PROGRESS,
    "daily_health_progress": CanonicalField.HEALTH_PARAMS_DAILY_HEALTH_PROGRESS,
    "progress_percentage": CanonicalField.HEALTH_PARAMS_PROGRESS_PERCENTAGE,
    "remaining_steps": CanonicalField.HEALTH_PARAMS_REMAINING_STEPS,
    "met_goal": CanonicalField.HEALTH_PARAMS_MET_GOAL,
    "total_steps": CanonicalField.HEALTH_PARAMS_TOTAL_STEPS,
    "avg_daily_steps": CanonicalField.HEALTH_PARAMS_AVG_DAILY_STEPS,
    "days_met_goal": CanonicalField.HEALTH_PARAMS_DAYS_MET_GOAL,
    "total_days": CanonicalField.HEALTH_PARAMS_TOTAL_DAYS,
    # Enrichment metrics
    "daily_health_score_trend": CanonicalField.HEALTH_PARAMS_DAILY_HEALTH_SCORE_TREND,
    "sleep_hr_range": CanonicalField.HEALTH_PARAMS_SLEEP_HR_RANGE,
    "walking_heart_rate_avg": CanonicalField.HEALTH_PARAMS_WALKING_HEART_RATE_AVG,
    "average_awake_hours": CanonicalField.HEALTH_PARAMS_AVERAGE_AWAKE_HOURS,
    "average_time_in_bed_hours": CanonicalField.HEALTH_PARAMS_AVERAGE_TIME_IN_BED_HOURS,
    "average_total_sleep_hours": CanonicalField.HEALTH_PARAMS_AVERAGE_TOTAL_SLEEP_HOURS,
    "energy_week_vs_prior_month_pct": (
        CanonicalField.HEALTH_PARAMS_ENERGY_WEEK_VS_PRIOR_MONTH_PCT
    ),
    # ENERGY.summaryData full set
    "energy_month_resting_total": (
        CanonicalField.HEALTH_PARAMS_ENERGY_MONTH_RESTING_TOTAL
    ),
    "total_active_energy": CanonicalField.HEALTH_PARAMS_TOTAL_ACTIVE_ENERGY,
    "energy_current_month_avg": CanonicalField.HEALTH_PARAMS_ENERGY_CURRENT_MONTH_AVG,
    "energy_week_avg": CanonicalField.HEALTH_PARAMS_ENERGY_WEEK_AVG,
    "energy_previous_month_avg": (
        CanonicalField.HEALTH_PARAMS_ENERGY_PREVIOUS_MONTH_AVG
    ),
    # Thresholds
    "quality_threshold": CanonicalField.HEALTH_PARAMS_QUALITY_THRESHOLD,
    "quality_good": CanonicalField.HEALTH_PARAMS_QUALITY_GOOD,
    "quality_high": CanonicalField.HEALTH_PARAMS_QUALITY_HIGH,
    "quality_very_high": CanonicalField.HEALTH_PARAMS_QUALITY_VERY_HIGH,
    "quality_low": CanonicalField.HEALTH_PARAMS_QUALITY_LOW,
    "wind_down_buffer_mins": CanonicalField.HEALTH_PARAMS_WIND_DOWN_BUFFER_MINS,
    # Stress flags (forced on top of computed params)
    "stress_high": CanonicalField.HEALTH_PARAMS_STRESS_HIGH,
    "stress_signal_high": CanonicalField.HEALTH_PARAMS_STRESS_SIGNAL_HIGH,
}


CANONICAL_FIELD_MAP: Final[
    dict[tuple[str, str], str]
] = {

    # ==============================================================
    # daily-user-snapshots (historical_snapshots)
    # ==============================================================

    ("daily-user-snapshots", "snapshot_date"):
        CanonicalField.DAILY_SNAPSHOT_DATE,
    ("daily-user-snapshots", "h_steps"):
        CanonicalField.STEP_COUNT_ONE_DAY,
    ("daily-user-snapshots", "h_active_minutes"):
        CanonicalField.ACTIVE_MINUTES_ONE_DAY,
    ("daily-user-snapshots", "h_calories_burned"):
        CanonicalField.TOTAL_CALORIES_BURNED_ONE_DAY_KCAL,
    ("daily-user-snapshots", "h_exercise_sessions"):
        CanonicalField.EXERCISE_SESSION_COUNT_ONE_DAY,
    ("daily-user-snapshots", "h_total_workout_min"):
        CanonicalField.TOTAL_WORKOUT_DURATION_ONE_DAY_MINUTES,
    ("daily-user-snapshots", "h_avg_heart_rate"):
        CanonicalField.AVERAGE_HEART_RATE_ONE_DAY_BPM,
    ("daily-user-snapshots", "h_resting_heart_rate"):
        CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM,
    ("daily-user-snapshots", "h_sleep_hours"):
        CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS,
    ("daily-user-snapshots", "h_sleep_quality"):
        CanonicalField.SLEEP_QUALITY_SCORE_ONE_NIGHT,
    ("daily-user-snapshots", "h_sleep_score"):
        CanonicalField.SLEEP_SCORE_ONE_NIGHT,
    ("daily-user-snapshots", "h_water_liters"):
        CanonicalField.WATER_INTAKE_ONE_DAY_LITERS,
    ("daily-user-snapshots", "h_calories_consumed"):
        CanonicalField.CALORIES_CONSUMED_ONE_DAY_KCAL,
    ("daily-user-snapshots", "h_weight_kg"):
        CanonicalField.BODY_WEIGHT_ONE_DAY_KG,
    ("daily-user-snapshots", "h_health_score"):
        CanonicalField.HEALTH_SCORE_ONE_DAY,
    ("daily-user-snapshots", "h_steps_goal_pct"):
        CanonicalField.STEPS_GOAL_COMPLETION_PERCENTAGE_ONE_DAY,
    ("daily-user-snapshots", "h_activity_level"):
        CanonicalField.ACTIVITY_LEVEL_ONE_DAY,
    ("daily-user-snapshots", "h_active_conditions"):
        CanonicalField.ACTIVE_HEALTH_CONDITIONS_ONE_DAY,
    ("daily-user-snapshots", "p_meeting_minutes"):
        CanonicalField.MEETING_DURATION_ONE_DAY_MINUTES,
    ("daily-user-snapshots", "p_longest_meeting_min"):
        CanonicalField.LONGEST_MEETING_DURATION_ONE_DAY_MINUTES,
    ("daily-user-snapshots", "p_work_span_minutes"):
        CanonicalField.WORK_SPAN_ONE_DAY_MINUTES,
    ("daily-user-snapshots", "p_events_after_9pm"):
        CanonicalField.EVENTS_AFTER_9_PM_ONE_DAY_COUNT,
    ("daily-user-snapshots", "wellness_score"):
        CanonicalField.WELLNESS_SCORE_ONE_DAY,

    ("daily-user-snapshots", "h_bedtime"):
        CanonicalField.DAILY_BEDTIME_LOCAL_DATETIME,

    ("daily-user-snapshots", "h_wake_time"):
        CanonicalField.DAILY_WAKE_TIME_LOCAL_DATETIME,

    # ==============================================================
    # balance
    # ==============================================================

    ("balance", "date"):
        CanonicalField.BALANCE_SCORE_DATE,

    ("balance", "healthScore"):
        CanonicalField.BALANCE_HEALTH_SCORE,

    ("balance_scores_30d", "date"):
        CanonicalField.BALANCE_SCORE_DATE,

    ("balance_scores_30d", "healthScore"):
        CanonicalField.BALANCE_HEALTH_SCORE,

    # ==============================================================
    # calendar / reminders / events / work-hours
    # ==============================================================

    ("calendar/event", "summary"):
        CanonicalField.CALENDAR_EVENT_TITLE,

    ("calendar/event", "completed"):
        CanonicalField.CALENDAR_EVENT_COMPLETION_STATUS,

    ("calendar/event", "startTime"):
        CanonicalField.CALENDAR_EVENT_START_TIME,

    ("calendar/event", "endTime"):
        CanonicalField.CALENDAR_EVENT_END_TIME,

    ("calendar/event", "eventType"):
        CanonicalField.CALENDAR_EVENT_TYPE,

    ("calendar/reminders", "title"):
        CanonicalField.REMINDER_TITLE,

    ("calendar/reminders", "dueDate"):
        CanonicalField.REMINDER_DUE_TIME,

    ("calendar/reminders", "completed"):
        CanonicalField.REMINDER_COMPLETION_STATUS,

    ("calendar/reminders", "recurrence.frequency"):
        CanonicalField.REMINDER_RECURRENCE_FREQUENCY,

    ("calendar/reminders", "recurrence.interval"):
        CanonicalField.REMINDER_RECURRENCE_INTERVAL,

    ("calendar/work-hours", "date"):
        CanonicalField.WORK_HOURS_DATE,

    ("calendar/work-hours", "totalHours"):
        CanonicalField.SCHEDULED_WORK_HOURS,

    ("calendar/work-hours", "eventCount"):
        CanonicalField.SCHEDULED_WORK_EVENT_COUNT,

    # ==============================================================
    # mood
    # ==============================================================

    ("moods/latest", "mood"):
        CanonicalField.CURRENT_MOOD,

    ("moods/latest", "moodDate"):
        CanonicalField.CURRENT_MOOD_DATE,

    ("moods", "mood"):
        CanonicalField.CURRENT_MOOD,

    ("moods", "moodDate"):
        CanonicalField.CURRENT_MOOD_DATE,

    # ==============================================================
    # productivity
    # ==============================================================

    ("productivity/summaries", "meetingPercent"):
        CanonicalField.MEETING_TIME_PERCENTAGE_TODAY,

    ("productivity/summaries", "meetingCount"):
        CanonicalField.MEETING_COUNT_TODAY,

    ("productivity/summaries", "completedMeetingSoFar"):
        CanonicalField.COMPLETED_MEETING_COUNT_TODAY,

    ("productivity/summaries", "reminderCount"):
        CanonicalField.REMINDER_COUNT_TODAY,

    ("productivity/summaries", "completedReminderSoFar"):
        CanonicalField.COMPLETED_REMINDER_COUNT_TODAY,

    # ==============================================================
    # health/summaries — ENERGY
    # ==============================================================

    ("health/summaries", "ENERGY.data.energyBurn"):
        CanonicalField.ENERGY_BURN_ONE_DAY_KCAL,

    ("health/summaries", "ENERGY.summaryData.totalActiveEnergy"):
        CanonicalField.ACTIVE_ENERGY_ONE_DAY_KCAL,

    ("health/summaries", "ENERGY.summaryData.totalRestingEnergy"):
        CanonicalField.RESTING_ENERGY_ONE_DAY_KCAL,

    ("health/summaries", "ENERGY.summaryData.currentMonthAvg"):
        CanonicalField.AVERAGE_DAILY_ACTIVE_ENERGY_BURN_CURRENT_MONTH_KCAL,

    ("health/summaries", "ENERGY.summaryData.avgCurrentWeekEnergyBurn"):
        CanonicalField.AVERAGE_DAILY_ACTIVE_ENERGY_BURN_CURRENT_WEEK_KCAL,

    ("health/summaries", "ENERGY.summaryData.previousMonthAvg"):
        CanonicalField.AVERAGE_DAILY_ACTIVE_ENERGY_BURN_PREVIOUS_MONTH_KCAL,

    # ==============================================================
    # health/summaries — HEART RATE
    # ==============================================================

    ("health/summaries", "HR.data.latestHR"):
        CanonicalField.MOST_RECENT_HEART_RATE_MEASUREMENT_ONE_DAY_BPM,

    ("health/summaries", "HR.data.restingHeartRate"):
        CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM,

    ("health/summaries", "HR.summaryData.heartRateRange.min"):
        CanonicalField.MINIMUM_HEART_RATE_CURRENT_SUMMARY_PERIOD_BPM,

    ("health/summaries", "HR.summaryData.heartRateRange.max"):
        CanonicalField.MAXIMUM_HEART_RATE_CURRENT_SUMMARY_PERIOD_BPM,

    ("health/summaries", "HR.data.walkingHeartRateAverage"):
        CanonicalField.WALKING_HEART_RATE_ONE_DAY_BPM,

    ("health/summaries", "HR.data.workoutHeartRate"):
        CanonicalField.WORKOUT_HEART_RATE_ONE_DAY,

    ("health/summaries", "HR.data.sleepHeartRate"):
        CanonicalField.SLEEP_HEART_RATE_ONE_NIGHT_BPM,

    # ==============================================================
    # health/summaries — SLEEP
    # ==============================================================

    ("health/summaries", "SLEEP.data.rem"):
        CanonicalField.REM_SLEEP_DURATION_ONE_NIGHT_HOURS,

    ("health/summaries", "SLEEP.data.core"):
        CanonicalField.CORE_SLEEP_DURATION_ONE_NIGHT_HOURS,

    ("health/summaries", "SLEEP.data.deep"):
        CanonicalField.DEEP_SLEEP_DURATION_ONE_NIGHT_HOURS,

    ("health/summaries", "SLEEP.data.total"):
        CanonicalField.SLEEP_DURATION_FROM_SUMMARY_ONE_NIGHT_HOURS,

    ("health/summaries", "SLEEP.data.sleepScore"):
        CanonicalField.SLEEP_SCORE_ONE_NIGHT,

    ("health/summaries", "SLEEP.data.lastWakeTime"):
        CanonicalField.LAST_WAKE_TIME_ONE_NIGHT_LOCAL_DATETIME,

    ("health/summaries", "SLEEP.data.firstSleepTime"):
        CanonicalField.FIRST_SLEEP_TIME_ONE_NIGHT_LOCAL_DATETIME,

    ("health/summaries", "SLEEP.summaryData.avgCore"):
        CanonicalField.AVERAGE_LIGHT_SLEEP_HOURS,
    ("health/summaries", "SLEEP.summaryData.avgAwake"):
        CanonicalField.AVERAGE_AWAKE_HOURS,
    ("health/summaries", "SLEEP.summaryData.avgTimeInBed"):
        CanonicalField.AVERAGE_TIME_IN_BED_HOURS,
    ("health/summaries", "SLEEP.summaryData.avgTimeAsleep"):
        CanonicalField.AVERAGE_TOTAL_SLEEP_HOURS,
    ("health/summaries", "SLEEP.summaryData.avgDeep"):
        CanonicalField.AVERAGE_DEEP_SLEEP_HOURS,
    ("health/summaries", "SLEEP.summaryData.avgRem"):
        CanonicalField.AVERAGE_REM_SLEEP_HOURS,

    # ==============================================================
    # health/summaries — STEPS
    # ==============================================================

    ("health/summaries", "STEPS.data.steps"):
        CanonicalField.STEP_COUNT_ONE_DAY,

    ("health/summaries", "STEPS.data.distance"):
        CanonicalField.DISTANCE_ONE_DAY,

    # Direct upstream step aggregates
    ("health/summaries", "STEPS.summaryData.currentMonthAvg"):
        CanonicalField.AVERAGE_DAILY_STEPS_CURRENT_MONTH,

    ("health/summaries", "STEPS.summaryData.previousMonthAvg"):
        CanonicalField.AVERAGE_DAILY_STEPS_PREVIOUS_MONTH,

    ("health/summaries", "STEPS.summaryData.dailyTarget"):
        CanonicalField.DAILY_STEPS_TARGET,

    # ==============================================================
    # onboarding / profile
    # ==============================================================

    ("onboarding/users/profiles", "language"):
        CanonicalField.USER_LANGUAGE,

    ("onboarding/users/profiles", "height_cm"):
        CanonicalField.USER_HEIGHT_CENTIMETERS,

    ("onboarding/users/profiles", "weight_kg"):
        CanonicalField.USER_WEIGHT_KILOGRAMS,

    ("onboarding/users/profiles", "date_of_birth"):
        CanonicalField.USER_DATE_OF_BIRTH,

    ("onboarding/users/profiles", "job_title"):
        CanonicalField.USER_JOB_TITLE,

    ("onboarding/users/profiles", "employment_type"):
        CanonicalField.USER_EMPLOYMENT_TYPE,

    ("onboarding/users/profiles", "work_status"):
        CanonicalField.USER_WORK_STATUS,

    ("onboarding/users/profiles", "active_hours_start_time"):
        CanonicalField.USER_ACTIVE_HOURS_START_TIME,

    ("onboarding/users/profiles", "active_hours_end_time"):
        CanonicalField.USER_ACTIVE_HOURS_END_TIME,

    ("onboarding/users/profiles", "bedtime_start"):
        CanonicalField.USER_BEDTIME_WINDOW_START,

    ("onboarding/users/profiles", "bedtime_end"):
        CanonicalField.USER_BEDTIME_WINDOW_END,

    ("onboarding/users/profiles", "primary_goal"):
        CanonicalField.USER_PRIMARY_GOAL,

    ("onboarding/users/profiles", "current_health_conditions"):
        CanonicalField.USER_CURRENT_HEALTH_CONDITIONS,

    ("onboarding/users/profiles", "age"):
        CanonicalField.USER_AGE_YEARS,

    ("onboarding/users/profiles", "gender"):
        CanonicalField.USER_GENDER,

    ("onboarding/users/profiles", "bmi"):
        CanonicalField.USER_BODY_MASS_INDEX,

    ("onboarding/users/profiles", "industry"):
        CanonicalField.USER_INDUSTRY,

    ("onboarding/users/profiles", "timezone"):
        CanonicalField.USER_TIMEZONE,

    ("onboarding/users/profiles", "goals"):
        CanonicalField.USER_HEALTH_GOALS,

    ("onboarding/users/profiles", "user_goals[].category_code"):
        CanonicalField.USER_GOAL_CATEGORY,

    ("onboarding/users/profiles", "user_goals[].action_code"):
        CanonicalField.USER_GOAL_ACTION,

    ("onboarding/users/profiles", "user_goals[].goal_name"):
        CanonicalField.USER_GOAL_NAME,

    ("onboarding/users/profiles", "user_goals[].target_value"):
        CanonicalField.USER_GOAL_TARGET_VALUE,

    ("onboarding/users/profiles", "user_goals[].unit"):
        CanonicalField.USER_GOAL_UNIT,
}


# ----------------------------------------------------------------------
# Semantic groups used by Phase 2
# ----------------------------------------------------------------------

CANONICAL_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "today": (
        CanonicalField.HEALTH_SCORE_ONE_DAY,
        CanonicalField.STEP_COUNT_ONE_DAY,
        CanonicalField.AVERAGE_HEART_RATE_ONE_DAY_BPM,
        CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM,
        CanonicalField.TOTAL_CALORIES_BURNED_ONE_DAY_KCAL,
        CanonicalField.ACTIVE_MINUTES_ONE_DAY,
        CanonicalField.WELLNESS_SCORE_ONE_DAY,
    ),

    "sleep": (
        CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS,
        CanonicalField.SLEEP_QUALITY_SCORE_ONE_NIGHT,
        CanonicalField.REM_SLEEP_DURATION_ONE_NIGHT_HOURS,
        CanonicalField.CORE_SLEEP_DURATION_ONE_NIGHT_HOURS,
        CanonicalField.DEEP_SLEEP_DURATION_ONE_NIGHT_HOURS,
        CanonicalField.FIRST_SLEEP_TIME_ONE_NIGHT_LOCAL_DATETIME,
        CanonicalField.LAST_WAKE_TIME_ONE_NIGHT_LOCAL_DATETIME,
        CanonicalField.SLEEP_HEART_RATE_ONE_NIGHT_BPM,
        CanonicalField.DAILY_BEDTIME_LOCAL_DATETIME,
        CanonicalField.DAILY_WAKE_TIME_LOCAL_DATETIME,
    ),

    "steps": (
        CanonicalField.STEP_COUNT_ONE_DAY,
        CanonicalField.DAILY_STEPS_TARGET,
        CanonicalField.STEPS_GOAL_COMPLETION_PERCENTAGE_ONE_DAY,
        CanonicalField.DISTANCE_ONE_DAY,
    ),

    "heart_rate": (
        CanonicalField.MOST_RECENT_HEART_RATE_MEASUREMENT_ONE_DAY_BPM,
        CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM,
        CanonicalField.AVERAGE_HEART_RATE_ONE_DAY_BPM,
        CanonicalField.WALKING_HEART_RATE_ONE_DAY_BPM,
        CanonicalField.WORKOUT_HEART_RATE_ONE_DAY,
        CanonicalField.SLEEP_HEART_RATE_ONE_NIGHT_BPM,
    ),

    "energy": (
        CanonicalField.TOTAL_CALORIES_BURNED_ONE_DAY_KCAL,
        CanonicalField.ACTIVE_ENERGY_ONE_DAY_KCAL,
        CanonicalField.RESTING_ENERGY_ONE_DAY_KCAL,
        CanonicalField.ENERGY_BURN_ONE_DAY_KCAL,
        CanonicalField.AVERAGE_DAILY_ACTIVE_ENERGY_BURN_CURRENT_WEEK_KCAL,
        CanonicalField.AVERAGE_DAILY_ACTIVE_ENERGY_BURN_CURRENT_MONTH_KCAL,
        CanonicalField.AVERAGE_DAILY_ACTIVE_ENERGY_BURN_PREVIOUS_MONTH_KCAL,
    ),

    "mood": (
        CanonicalField.CURRENT_MOOD,
        CanonicalField.CURRENT_MOOD_DATE,
    ),

    "productivity": (
        CanonicalField.MEETING_TIME_PERCENTAGE_TODAY,
        CanonicalField.MEETING_COUNT_TODAY,
        CanonicalField.COMPLETED_MEETING_COUNT_TODAY,
        CanonicalField.REMINDER_COUNT_TODAY,
        CanonicalField.COMPLETED_REMINDER_COUNT_TODAY,
        CanonicalField.MEETING_DURATION_ONE_DAY_MINUTES,
        CanonicalField.LONGEST_MEETING_DURATION_ONE_DAY_MINUTES,
        CanonicalField.WORK_SPAN_ONE_DAY_MINUTES,
    ),

    "calendar": (
        CanonicalField.CALENDAR_EVENT_TITLE,
        CanonicalField.CALENDAR_EVENT_COMPLETION_STATUS,
        CanonicalField.CALENDAR_EVENT_START_TIME,
        CanonicalField.CALENDAR_EVENT_END_TIME,
        CanonicalField.CALENDAR_EVENT_TYPE,
        CanonicalField.REMINDER_TITLE,
        CanonicalField.REMINDER_DUE_TIME,
        CanonicalField.REMINDER_COMPLETION_STATUS,
        CanonicalField.REMINDER_RECURRENCE_FREQUENCY,
        CanonicalField.REMINDER_RECURRENCE_INTERVAL,
        CanonicalField.SCHEDULED_WORK_HOURS,
        CanonicalField.SCHEDULED_WORK_EVENT_COUNT,
    ),

    "profile": (
        CanonicalField.USER_LANGUAGE,
        CanonicalField.USER_HEIGHT_CENTIMETERS,
        CanonicalField.USER_WEIGHT_KILOGRAMS,
        CanonicalField.USER_DATE_OF_BIRTH,
        CanonicalField.USER_JOB_TITLE,
        CanonicalField.USER_EMPLOYMENT_TYPE,
        CanonicalField.USER_WORK_STATUS,
        CanonicalField.USER_ACTIVE_HOURS_START_TIME,
        CanonicalField.USER_ACTIVE_HOURS_END_TIME,
        CanonicalField.USER_BEDTIME_WINDOW_START,
        CanonicalField.USER_BEDTIME_WINDOW_END,
        CanonicalField.USER_PRIMARY_GOAL,
        CanonicalField.USER_CURRENT_HEALTH_CONDITIONS,
        CanonicalField.USER_AGE_YEARS,
        CanonicalField.USER_GENDER,
        CanonicalField.USER_BODY_MASS_INDEX,
        CanonicalField.USER_INDUSTRY,
        CanonicalField.USER_TIMEZONE,
        CanonicalField.USER_GOAL_CATEGORY,
        CanonicalField.USER_GOAL_ACTION,
        CanonicalField.USER_GOAL_NAME,
        CanonicalField.USER_GOAL_TARGET_VALUE,
        CanonicalField.USER_GOAL_UNIT,
    ),
}


# ----------------------------------------------------------------------
# Top-key → source name for canonicalize_extracted_data
# ----------------------------------------------------------------------

_SOURCE_TOP_KEY: Final[dict[str, str]] = {
    # Raw API payloads
    "historical_snapshots": "daily-user-snapshots",
    "balance_score": "balance",
    "balance_scores_30d": "balance",
    "balance_scores_7d": "balance",  # legacy alias
    "today_reminders": "calendar/reminders",
    "calendar_events": "calendar/event",
    "calendar_events_week": "calendar/event",
    "latest_mood": "moods/latest",
    "moods_7d": "moods",
    "productivity_summary": "productivity/summaries",
    "productivity_summaries_7d": "productivity/summaries",
    "calendar_work_hours": "calendar/work-hours",
    "today_health_stats": "health/summaries",
    "user_profile": "onboarding/users/profiles",
    # Derived dicts (source = top-key, 1:1 with semantic boundary)
    "health_params": "health_params",
    "time_data": "time_data",
    "calendar_metrics": "calendar_metrics",
    "mood_data": "mood_data",
}


def canonical_name_for(source: str, field_path: str) -> str | None:
    return CANONICAL_FIELD_MAP.get((source, field_path))


def fields_group(group: str) -> tuple[str, ...]:
    try:
        return CANONICAL_GROUPS[group]
    except KeyError as exc:
        raise KeyError(f"Unknown canonical group: {group!r}") from exc


def canonicalize_mapping_keys(
    data: Mapping[str, Any],
    source: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        if source in _DERIVED_TOP_LEVEL_MAPS:
            canonical = _DERIVED_TOP_LEVEL_MAPS[source].get(key)
        else:
            canonical = canonical_name_for(source, key)
        result[canonical or key] = _canonicalize_nested_known_fields(source, key, value)
    return result

_DERIVED_TOP_LEVEL_MAPS: Final[dict[str, Mapping[str, str]]] = {
    # Add user_profile so top-level "timezone" resolves via PROFILE_CANONICAL_FIELD_MAP
    # (not via CANONICAL_FIELD_MAP's time_data entry).
    "user_profile": PROFILE_CANONICAL_FIELD_MAP,
    "health_params": HEALTH_PARAMS_CANONICAL_FIELD_MAP,
    "time_data": TIME_DATA_CANONICAL_FIELD_MAP,
    "calendar_metrics": CALENDAR_METRICS_CANONICAL_FIELD_MAP,
    "mood_data": MOOD_DATA_CANONICAL_FIELD_MAP,
}


def _canonicalize_nested_known_fields(
    source: str,
    parent_key: str,
    value: Any,
) -> Any:
    if source == "calendar/reminders" and parent_key == "recurrence" and isinstance(value, Mapping):
        return _canonicalize_reminder_recurrence(value, source)

    if source == "onboarding/users/profiles" and parent_key == "user_goals" and isinstance(value, list):
        return _canonicalize_profile_goals(value, source)

    if source == "health/summaries":
        return _canonicalize_health_summary_value(parent_key, value)

    return value


def _canonicalize_reminder_recurrence(value: Mapping[str, Any], source: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        canonical = CANONICAL_FIELD_MAP.get((source, f"recurrence.{key}"))
        result[canonical or key] = child
    return result


def _canonicalize_profile_goals(value: list[Any], source: str) -> list[Any]:
    out: list[Any] = []
    for goal in value:
        if not isinstance(goal, Mapping):
            out.append(goal)
            continue
        canonical_goal: dict[str, Any] = {}
        for key, child in goal.items():
            canonical = CANONICAL_FIELD_MAP.get((source, f"user_goals[].{key}"))
            canonical_goal[canonical or key] = child
        out.append(canonical_goal)
    return out


def _canonicalize_health_summary_value(parent_key: str, value: Any) -> Any:
    """Canonicalize supported health summary children without changing values."""
    if not isinstance(value, Mapping) and not isinstance(value, list):
        return value

    if parent_key == "ENERGY" and isinstance(value, Mapping):
        result = dict(value)
        _rename_if_present(result, "data", _canonicalize_energy_data)
        summary = result.get("summaryData")
        if isinstance(summary, Mapping):
            result["summaryData"] = _canonicalize_energy_summary(summary)
        return result

    if parent_key == "HR" and isinstance(value, Mapping):
        result = dict(value)
        data = result.get("data")
        if isinstance(data, list):
            result["data"] = [_canonicalize_hr_entry(item) for item in data]
        summary = result.get("summaryData")
        if isinstance(summary, Mapping):
            result["summaryData"] = _canonicalize_hr_summary(summary)
        return result

    if parent_key == "SLEEP" and isinstance(value, Mapping):
        result = dict(value)
        data = result.get("data")
        if isinstance(data, list):
            result["data"] = [_canonicalize_sleep_entry(item) for item in data]
        summary = result.get("summaryData")
        if isinstance(summary, Mapping):
            result["summaryData"] = _canonicalize_sleep_summary(summary)
        return result

    if parent_key == "STEPS" and isinstance(value, Mapping):
        result = dict(value)
        data = result.get("data")
        if isinstance(data, list):
            result["data"] = [_canonicalize_steps_entry(item) for item in data]
        summary = result.get("summaryData")
        if isinstance(summary, Mapping):
            result["summaryData"] = _canonicalize_steps_summary(summary)
        return result

    return value


def _rename_if_present(mapping: dict[str, Any], key: str, fn) -> None:
    value = mapping.get(key)
    if isinstance(value, list):
        mapping[key] = [fn(item) for item in value]


def _canonicalize_energy_data(item: Any) -> Any:
    if not isinstance(item, Mapping):
        return item
    result = dict(item)
    if "energyBurn" in result:
        result[CanonicalField.ENERGY_BURN_ONE_DAY_KCAL] = result.pop("energyBurn")
    return result


def _canonicalize_energy_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    direct = {
        "totalActiveEnergy": CanonicalField.ACTIVE_ENERGY_ONE_DAY_KCAL,
        "totalRestingEnergy": CanonicalField.RESTING_ENERGY_ONE_DAY_KCAL,
        "avgCurrentWeekEnergyBurn": (
            CanonicalField.AVERAGE_DAILY_ACTIVE_ENERGY_BURN_CURRENT_WEEK_KCAL
        ),
        "previousMonthAvg": (
            CanonicalField.AVERAGE_DAILY_ACTIVE_ENERGY_BURN_PREVIOUS_MONTH_KCAL
        ),
        "currentMonthAvg": (
            CanonicalField.AVERAGE_DAILY_ACTIVE_ENERGY_BURN_CURRENT_MONTH_KCAL
        ),
    }
    result = dict(item)
    for source_key, canonical_key in direct.items():
        if source_key in result:
            result[canonical_key] = result.pop(source_key)
    return result


def _canonicalize_sleep_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize SLEEP.summaryData avg fields — drop abbrevs, use full names."""
    direct = {
        "avgCore": CanonicalField.AVERAGE_LIGHT_SLEEP_HOURS,
        "avgAwake": CanonicalField.AVERAGE_AWAKE_HOURS,
        "avgTimeInBed": CanonicalField.AVERAGE_TIME_IN_BED_HOURS,
        "avgTimeAsleep": CanonicalField.AVERAGE_TOTAL_SLEEP_HOURS,
        "avgDeep": CanonicalField.AVERAGE_DEEP_SLEEP_HOURS,
        "avgRem": CanonicalField.AVERAGE_REM_SLEEP_HOURS,
    }
    result = dict(item)
    for source_key, canonical_key in direct.items():
        if source_key in result:
            result[canonical_key] = result.pop(source_key)
    return result


def _canonicalize_hr_entry(item: Any) -> Any:
    if not isinstance(item, Mapping):
        return item
    result = dict(item)
    mapping = {
        "latestHR": CanonicalField.MOST_RECENT_HEART_RATE_MEASUREMENT_ONE_DAY_BPM,
        "restingHeartRate": CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM,
        "walkingHeartRateAverage": CanonicalField.WALKING_HEART_RATE_ONE_DAY_BPM,
        "sleepHeartRate": CanonicalField.SLEEP_HEART_RATE_ONE_NIGHT_BPM,
        "workoutHeartRate": CanonicalField.WORKOUT_HEART_RATE_ONE_DAY,
    }
    for source_key, canonical_key in mapping.items():
        if source_key in result:
            result[canonical_key] = result.pop(source_key)
    return result


def _canonicalize_hr_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(item)
    if isinstance(result.get("heartRateRange"), Mapping):
        hr_range = dict(result.pop("heartRateRange"))
        if "min" in hr_range:
            result[CanonicalField.MINIMUM_HEART_RATE_CURRENT_SUMMARY_PERIOD_BPM] = hr_range.pop("min")
        if "max" in hr_range:
            result[CanonicalField.MAXIMUM_HEART_RATE_CURRENT_SUMMARY_PERIOD_BPM] = hr_range.pop("max")
        if hr_range:
            result["heart_rate_range_details"] = hr_range
    if "average_heart_rate_variability_current" in result:
        result[
            CanonicalField.AVERAGE_HEART_RATE_VARIABILITY_CURRENT
        ] = result.pop("average_heart_rate_variability_current")
    return result


def _canonicalize_sleep_entry(item: Any) -> Any:
    if not isinstance(item, Mapping):
        return item
    result = dict(item)
    mapping = {
        "rem": CanonicalField.REM_SLEEP_DURATION_ONE_NIGHT_HOURS,
        "core": CanonicalField.CORE_SLEEP_DURATION_ONE_NIGHT_HOURS,
        "deep": CanonicalField.DEEP_SLEEP_DURATION_ONE_NIGHT_HOURS,
        "total": CanonicalField.SLEEP_DURATION_FROM_SUMMARY_ONE_NIGHT_HOURS,
        "sleepScore": CanonicalField.SLEEP_SCORE_ONE_NIGHT,
        "lastWakeTime": CanonicalField.LAST_WAKE_TIME_ONE_NIGHT_LOCAL_DATETIME,
        "firstSleepTime": CanonicalField.FIRST_SLEEP_TIME_ONE_NIGHT_LOCAL_DATETIME,
    }
    for source_key, canonical_key in mapping.items():
        if source_key in result:
            result[canonical_key] = result.pop(source_key)
    return result


def _canonicalize_steps_entry(item: Any) -> Any:
    if not isinstance(item, Mapping):
        return item
    result = dict(item)
    if "steps" in result:
        result[CanonicalField.STEP_COUNT_ONE_DAY] = result.pop("steps")
    if "distance" in result:
        result[CanonicalField.DISTANCE_ONE_DAY] = result.pop("distance")
    return result


def _canonicalize_steps_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(item)
    if "currentMonthAvg" in result:
        result[CanonicalField.AVERAGE_DAILY_STEPS_CURRENT_MONTH] = result.pop("currentMonthAvg")
    if "previousMonthAvg" in result:
        result[CanonicalField.AVERAGE_DAILY_STEPS_PREVIOUS_MONTH] = result.pop("previousMonthAvg")
    if "dailyTarget" in result:
        result[CanonicalField.DAILY_STEPS_TARGET] = result.pop("dailyTarget")
    if "dailyStepsTarget" in result:
        result[CanonicalField.DAILY_STEPS_TARGET] = result.pop("dailyStepsTarget")
    if "totalSteps" in result:
        result[CanonicalField.STEP_COUNT_ONE_DAY_SUMMARY] = result.pop("totalSteps")
    if "totalDistance" in result:
        result[CanonicalField.DISTANCE_ONE_DAY_SUMMARY] = result.pop("totalDistance")
    return result


def canonicalize_extracted_data(raw_data: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = deepcopy(dict(raw_data))

    for top_key, source in _SOURCE_TOP_KEY.items():
        if top_key not in result:
            continue
        value = result[top_key]

        if isinstance(value, list):
            result[top_key] = [
                canonicalize_mapping_keys(item, source)
                if isinstance(item, Mapping)
                else item
                for item in value
            ]
        elif isinstance(value, Mapping):
            result[top_key] = canonicalize_mapping_keys(value, source)

    return result


def validate_mapping() -> None:
    seen: dict[tuple[str, str], str] = {}
    for (source, field_path), name in CANONICAL_FIELD_MAP.items():
        prev = seen.get((source, field_path))
        if prev is not None and prev != name:
            raise ValueError(
                f"Duplicate CANONICAL_FIELD_MAP entry for ({source!r}, "
                f"{field_path!r}): {prev!r} vs {name!r}"
            )
        seen[(source, field_path)] = name

    forbidden = {"hr", "rhr", "hrv", "avg", "pct", "min", "sec"}
    seen_names: set[str] = set()
    for name in CANONICAL_FIELD_MAP.values():
        if name in seen_names:
            continue
        seen_names.add(name)
        tokens = set(name.split("_"))
        bad = tokens & forbidden
        if bad:
            raise ValueError(
                f"Canonical name {name!r} contains forbidden abbreviation token(s): {sorted(bad)}"
            )

def summarize_canonical_diff(
    raw_data: Mapping[str, Any],
    canonical_data: Mapping[str, Any],
    *,
    max_items_per_section: int = 6,
) -> str:
    lines: list[str] = []

    for top_key in _SOURCE_TOP_KEY:
        if top_key not in raw_data or top_key not in canonical_data:
            continue
        raw_view = raw_data[top_key]
        canon_view = canonical_data[top_key]

        if isinstance(raw_view, list) and raw_view:
            raw_sample = raw_view[0] if isinstance(raw_view[0], Mapping) else None
            canon_sample = (
                canon_view[0]
                if isinstance(canon_view, list) and isinstance(canon_view[0], Mapping)
                else None
            )
        elif isinstance(raw_view, Mapping):
            raw_sample = raw_view
            canon_sample = canon_view if isinstance(canon_view, Mapping) else None
        else:
            continue

        if not isinstance(raw_sample, Mapping) or not isinstance(canon_sample, Mapping):
            continue

        raw_keys = list(raw_sample.keys())
        canon_keys = list(canon_sample.keys())

        renames: list[tuple[str, str]] = []
        for ck in canon_keys:
            if ck in raw_keys:
                continue
            source = _SOURCE_TOP_KEY.get(top_key, "")
            for rk in raw_keys:
                if canonical_name_for(source, rk) == ck:
                    renames.append((rk, ck))
                    break

        new_canon_only = [k for k in canon_keys if k not in raw_keys]
        dropped_raw_only = [k for k in raw_keys if k not in canon_keys]

        lines.append(f"  • {top_key}:")
        if isinstance(raw_view, list):
            lines.append(f"      items={len(raw_view)} (showing first row)")
        if renames:
            lines.append("      renamed:")
            for rk, ck in renames[:max_items_per_section]:
                lines.append(f"        {rk} → {ck}")
            if len(renames) > max_items_per_section:
                lines.append(f"        … and {len(renames) - max_items_per_section} more")
        if new_canon_only:
            lines.append(
                f"      canonical-only: {', '.join(new_canon_only[:max_items_per_section])}"
            )
        if dropped_raw_only:
            lines.append(
                f"      non-canonical kept: {', '.join(dropped_raw_only[:max_items_per_section])}"
            )

    if not lines:
        return "  (no source-rooted payloads to diff)"

    total_renamed = sum(line.count("→") for line in lines)
    return "\n".join(lines) + f"\n  total renamed leaves (approx): {total_renamed}"


validate_mapping()

__all__ = [
    "BALANCE_CANONICAL_FIELD_MAP",
    "CANONICAL_FIELD_MAP",
    "CANONICAL_GROUPS",
    "CALENDAR_METRICS_CANONICAL_FIELD_MAP",
    "CanonicalField",
    "DAILY_SNAPSHOT_CANONICAL_FIELD_MAP",
    "HEALTH_PARAMS_CANONICAL_FIELD_MAP",
    "MOOD_DATA_CANONICAL_FIELD_MAP",
    "PROFILE_CANONICAL_FIELD_MAP",
    "PROFILE_GOAL_CANONICAL_FIELD_MAP",
    "TIME_DATA_CANONICAL_FIELD_MAP",
    "canonical_name_for",
    "canonicalize_extracted_data",
    "canonicalize_mapping_keys",
    "fields_group",
    "summarize_canonical_diff",
]
