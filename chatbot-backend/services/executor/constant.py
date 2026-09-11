"""Constants for health data preparation and insight analysis"""


class HealthDataConstants:
    """Constants class for health data preparation"""

    # ============================================================================
    # Sleep Quality Thresholds
    # ============================================================================
    SLEEP_QUALITY_HIGH = 81  # Sleep score >= 81 is considered high quality
    SLEEP_QUALITY_VERY_HIGH = 96  # Sleep score >= 96 is considered very high quality
    SLEEP_QUALITY_THRESHOLD = 61  # Sleep score < 61 is considered low quality
    SLEEP_QUALITY_LOW = 41  # Sleep score < 41 is considered very low quality
    SLEEP_QUALITY_GOOD = (
        70  # Sleep score >= 70 is considered good (for backward compatibility)
    )

    # ============================================================================
    # Health Threshold
    # ============================================================================
    HEALTH_THRESHOLD = 75

    # ============================================================================
    # Time Constants
    # ============================================================================
    DEFAULT_WIND_DOWN_BUFFER_MINS = 60  # Default 1 hour = 60 minutes

    # ============================================================================
    # Health Data Parameter Keys
    # ============================================================================

    # Basic Health Data Keys
    KEY_STEPS_TODAY = "steps_today"
    KEY_SLEEP_LASTNIGHT = "sleep_lastnight"
    KEY_SLEEP_QUALITY_SCORE = "sleep_quality_score"
    KEY_SLEEP_QUALITY = (
        "sleep_quality"  # Quality level: "very_high", "high", "good", "low", "very_low"
    )
    KEY_LATEST_HEART_RATE = "latest_heart_rate"
    KEY_RESTING_HEART_RATE = "resting_heart_rate"
    KEY_LAST_WAKE_TIME = "last_wake_time"
    KEY_FIRST_SLEEP_TIME = "first_sleep_time"
    KEY_WORKOUT_HEART_RATE = (
        "workout_heart_rate"  # Dict with avgHR, maxHR, minHR, duration, activityType
    )
    KEY_WORKOUT_AVG_HR = "workout_avg_hr"
    KEY_WORKOUT_MAX_HR = "workout_max_hr"
    KEY_WORKOUT_MIN_HR = "workout_min_hr"
    KEY_WORKOUT_DURATION = "workout_duration"  # Duration in seconds
    KEY_WORKOUT_ACTIVITY_TYPE = "workout_activity_type"

    # Goal Keys
    KEY_STEPS_GOAL = "steps_goal"
    KEY_SLEEP_GOAL = "sleep_goal"

    # Advanced Metrics Keys
    KEY_BASELINE_RESTING_HR = "baseline_resting_hr"
    KEY_HRV_SCORE = "hrv_score"
    KEY_BEDTIME_STREAK = "bedtime_streak"
    KEY_WAKE_TIME = "wake_time"
    KEY_SLEEP_LAST3NIGHTS = "sleep_last3nights"
    KEY_SLEEP_HOURS = "sleep_hours"
    KEY_ALL_3_NIGHTS_HAVE_DATA = "all_3_nights_have_data"
    KEY_DEEP_SLEEP = "deep_sleep_min"
    KEY_REM_SLEEP = "rem_sleep_min"
    KEY_LIGHT_SLEEP = "light_sleep_min"
    KEY_STEPS_STREAK = "steps_streak"

    # Progress Metrics Keys
    KEY_WEEKLY_HEALTH_PROGRESS = "weekly_health_progress"
    KEY_DAILY_HEALTH_PROGRESS = "daily_health_progress"

    # Threshold Keys
    KEY_QUALITY_THRESHOLD = "quality_threshold"
    KEY_QUALITY_GOOD = "quality_good"
    KEY_QUALITY_HIGH = "quality_high"
    KEY_QUALITY_VERY_HIGH = "quality_very_high"
    KEY_QUALITY_LOW = "quality_low"
    KEY_WIND_DOWN_BUFFER_MINS = "wind_down_buffer_mins"
    KEY_PROGRESS_PERCENTAGE = "progress_percentage"
    KEY_REMAINING_STEPS = "remaining_steps"
    KEY_MET_GOAL = "met_goal"
    KEY_TOTAL_STEPS = "total_steps"
    KEY_AVG_DAILY_STEPS = "avg_daily_steps"
    KEY_DAYS_MET_GOAL = "days_met_goal"
    KEY_TOTAL_DAYS = "total_days"
    KEY_DAILY_HEALTH_SCORE_TREND = "daily_health_score_trend"
    KEY_SLEEP_HR_RANGE = "sleep_hr_range"
    KEY_WALKING_HEART_RATE_AVG = "walking_heart_rate_avg"
    KEY_AVERAGE_AWAKE_HOURS = "average_awake_hours"
    KEY_AVERAGE_TIME_IN_BED_HOURS = "average_time_in_bed_hours"
    KEY_AVERAGE_TOTAL_SLEEP_HOURS = "average_total_sleep_hours"
    KEY_CALORIES_BURNED_TODAY = "calories_burned_today"
    KEY_ACTIVE_MINUTES_TODAY = "active_minutes_today"
    KEY_ENERGY_LAST_DATA_DATE = "energy_last_data_date"
    KEY_ENERGY_WEEK_VS_PRIOR_MONTH_PCT = "energy_week_vs_prior_month_pct"
    KEY_ENERGY_MONTH_RESTING_TOTAL = "energy_month_resting_total"
    KEY_TOTAL_ACTIVE_ENERGY = "total_active_energy"  # today active
    KEY_ENERGY_CURRENT_MONTH_AVG = "energy_current_month_avg"
    KEY_ENERGY_WEEK_AVG = "energy_week_avg"  # avgCurrentWeekEnergyBurn
    KEY_ENERGY_PREVIOUS_MONTH_AVG = "energy_previous_month_avg"
    SLEEP_TYPE = "SLEEP"
    STEPS_TYPE = "STEPS"
    HR_TYPE = "HR"
    ENERGY_TYPE = "ENERGY"


class TimeDataConstants:
    """Constants class for time context and time metrics keys"""

    # Time context keys
    KEY_CURRENT_DT = "current_dt"
    KEY_CURRENT_HOUR = "current_hour"
    KEY_CURRENT_MINUTE = "current_minute"
    KEY_IS_WEEKEND = "is_weekend"
    KEY_TZ = "tz"
    KEY_TIMEZONE = "timezone"
    KEY_CURRENT_DATE = "current_date"
    KEY_CURRENT_TIME_ISO = "current_time_iso"
    KEY_BEDTIME_START_STR = "bedtime_start_str"
    KEY_BEDTIME_END_STR = "bedtime_end_str"
    KEY_ACTIVE_START_STR = "active_start_str"
    KEY_ACTIVE_END_STR = "active_end_str"
    KEY_SLEEP_TARGET = "sleep_target"

    # Time metrics keys
    KEY_TIME_TO_BEDTIME = "time_to_bedtime"
    KEY_TIME_TO_NEXT_EVENT = "time_to_next_event"
    KEY_IN_ACTIVE_WINDOW = "in_active_window"
    KEY_FREE_SLOT_LENGTH = "free_slot_length"


class APIResponseKeys:
    """Keys for API response from /api/health/summaries"""

    # ============================================================================
    # Common Response Keys
    # ============================================================================
    TYPE = "type"
    DATA = "data"
    DATE = "date"
    DATE_RANGE = "dateRange"
    START_DATE = "startDate"
    END_DATE = "endDate"
    START_TIME = "startTime"

    # ============================================================================
    # STEPS Data Keys
    # ============================================================================
    STEPS = "steps"
    DISTANCE = "distance"
    HEALTH_SCORE = "healthScore"

    # ============================================================================
    # SLEEP Data Keys
    # ============================================================================
    TOTAL = "total"
    REM = "rem"
    CORE = "core"
    DEEP = "deep"
    AWAKE = "awake"
    SLEEP_SCORE = "sleepScore"
    QUALITY_SCORE = "qualityScore"  # Alternative key for sleep score

    # ============================================================================
    # HR (Heart Rate) Data Keys
    # ============================================================================
    LATEST_HR = "latestHR"
    RESTING_HEART_RATE = "restingHeartRate"
    LATEST_HR_TIME = "latestHRTime"
    SLEEP_HEART_RATE = "sleepHeartRate"
    WORKOUT_HEART_RATE = "workoutHeartRate"
    WALKING_HEART_RATE_AVERAGE = "walkingHeartRateAverage"
    # Workout Heart Rate sub-keys (when workoutHeartRate is an object)
    WORKOUT_AVG_HR = "avgHR"
    WORKOUT_MAX_HR = "maxHR"
    WORKOUT_MIN_HR = "minHR"
    WORKOUT_DURATION = "duration"
    WORKOUT_ACTIVITY_TYPE = "activityType"

    # ============================================================================
    # Summary Data Keys
    # ============================================================================
    SUMMARY_DATA = "summaryData"
    LAST_WAKE_TIME = "lastWakeTime"
    FIRST_SLEEP_TIME = "firstSleepTime"
    DAILY_TARGET = "dailyTarget"
    TOTAL_STEPS = "totalSteps"
    TOTAL_DISTANCE = "totalDistance"
    CURRENT_MONTH_AVG = "currentMonthAvg"
    PREVIOUS_MONTH_AVG = "previousMonthAvg"
    TOTAL_ACTIVE_ENERGY = "totalActiveEnergy"
    TOTAL_RESTING_ENERGY = "totalRestingEnergy"
    AVG_CORE = "avgCore"
    AVG_AWAKE = "avgAwake"
    AVG_TIME_IN_BED = "avgTimeInBed"
    AVG_TIME_ASLEEP = "avgTimeAsleep"
    AVG_DEEP = "avgDeep"
    AVG_REM = "avgRem"
    HEART_RATE_RANGE = "heartRateRange"
    MAX = "max"
    MIN = "min"
    AVERAGE_HEART_RATE_VARIABILITY_CURRENT = (
        "average_heart_rate_variability_current"
    )
    AVG_CURRENT_WEEK_ENERGY_BURN = "avgCurrentWeekEnergyBurn"

    PRODUCTIVITY_SCORE = "productivityScore"
    FINANCE_SCORE = "financeScore"


class TargetKeys:
    """Keys for user profile targets and goals"""

    # ============================================================================
    # User Profile Root Keys
    # ============================================================================
    TARGETS = "targets"
    SLEEP_GOAL = "sleep_goal"

    # ============================================================================
    # Target Keys
    # ============================================================================
    TARGET_STEPS_PER_DAY = "target_steps_per_day"
    TARGET_SLEEP_HOURS = "target_sleep_hours"
    VALUE = "value"

    # ============================================================================
    # Sleep Goal Keys
    # ============================================================================
    BEDTIME_START = "bedtime_start"
    BEDTIME_END = "bedtime_end"

    # ============================================================================
    # Active Hours Keys
    # ============================================================================
    ACTIVE_HOURS_START_TIME = "active_hours_start_time"
    ACTIVE_HOURS_END_TIME = "active_hours_end_time"

    # ============================================================================
    # Break Keys
    # ============================================================================
    BREAK_INTERVAL_HOURS = "break_interval_hours"
    BREAK_DURATION_MIN = "break_duration_min"


class CalendarDataConstants:
    """Constants class for calendar data preparation"""

    # ============================================================================
    # Calendar Event Keys
    # ============================================================================
    KEY_SUMMARY = "summary"
    KEY_START_TIME = "startTime"
    KEY_END_TIME = "endTime"
    KEY_LOCATION = "location"
    KEY_DESCRIPTION = "description"
    KEY_ALL_DAY = "allDay"
    KEY_PARTICIPANTS = "participants"
    KEY_CATEGORY = "category"
    KEY_NAME = "name"
    KEY_EMAIL = "email"

    # ============================================================================
    # Calendar Metrics Keys
    # ============================================================================
    KEY_BACK_TO_BACK_COUNT = "back_to_back_count"
    KEY_MEETING_MINUTES = "meeting_minutes"
    KEY_CALENDAR_DENSITY = "calendar_density"
    KEY_CONTINUOUS_EVENTS_MINUTES = "continuous_events_minutes"
    KEY_WORK_EVENTS_HOURS_WEEKEND = "work_events_hours_weekend"
    KEY_WORK_EVENTS_HOURS_WEEK = "work_events_hours_week"
    KEY_WORK_LOAD_HIGH = "work_load_high"

    # ============================================================================
    # Return Dictionary Keys
    # ============================================================================
    KEY_EVENTS = "events"
    KEY_FORMATTED_CALENDAR_DATA = "formatted_calendar_data"
    KEY_CALENDAR_METRICS = "calendar_metrics"
    KEY_CALENDAR_EVENTS_WEEK = "calendar_events_week"
    KEY_CALENDAR_EVENTS_NEXT7D = "calendar_events_next7d"
    KEY_CALENDAR_EVENTS_NEXT48H = "calendar_events_next48h"
    KEY_CALENDAR_EVENTS_NEXT3H = "calendar_events_next3h"

    # ============================================================================
    # Calendar Calculation Constants
    # ============================================================================
    BACK_TO_BACK_GAP_THRESHOLD_MINUTES = (
        5  # Events with gap < 5 minutes are considered back-to-back
    )
    MEETING_WINDOW_HOURS = 3  # Calculate meeting minutes in last 3 hours
    MAX_PARTICIPANTS_TO_SHOW = 3  # Show first 3 participants in formatted text
    CONTINUOUS_EVENTS_THRESHOLD_MINUTES = 90  # Continuous work >=90 mins
    WORK_LOAD_HIGH_THRESHOLD_HOURS = 40  # Weekly work hours > 40 hours
    PRODUCTIVITY_THRESHOLD = 75
    WORK_LOAD_HIGH_THRESHOLD_HOURS = 35

    # ============================================================================
    # Default Messages
    # ============================================================================
    DEFAULT_NO_EVENTS_MESSAGE = "No calendar events found for today."


class CalendarEventCategoryConstants:
    """Allowed calendar event category codes (chatbot-backend standalone)."""

    MAX_BATCH_SIZE = 50

    ALLOWED = frozenset(
        {
            "FOCUS",
            "MEETINGS",
            "ADMIN",
            "EXECUTION",
            "STRATEGY",
            "GROWTH",
            "PEOPLE",
            "HEALTH",
            "SOCIAL",
            "LIFESTYLE",
            "LEISURE",
            "HOME",
            "LEARNING",
            "TRAVEL",
            "OTHER",
            "MEAL",
            "BEDTIME",
            "TIME_OFF",
            "HOLIDAY",
            "BIRTHDAY",
            "ANNIVERSARY",
            "CELEBRATION",
        }
    )

    @classmethod
    def normalize(cls, value):
        if value is None:
            return None
        text = str(value).strip().upper()
        return text or None

    @classmethod
    def resolve(cls, value):
        """Normalize category; values outside ALLOWED become OTHER."""
        norm = cls.normalize(value)
        if not norm:
            return "OTHER"
        if norm in cls.ALLOWED:
            return norm
        return "OTHER"


class InsightGroupConstants:
    """Constants class for insight group names"""

    # ============================================================================
    # Common Groups (used in both productivity and overall insights)
    # ============================================================================
    KEY_PRODUCTIVITY_TYPE = "productivity"
    KEY_OVERALL_TYPE = "overall"
    KEY_INSIGHT = "insight"
    CONTEXTUAL_SUGGESTION = "contextual_suggestion"  # Default fallback
    SAFETY_RISK = "safety_risk"
    WIND_DOWN_WINDOW = "wind_down_window"
    BEDTIME_WINDOW = "bedtime_window"
    MORNING_WINDOW = "morning_window"
    ACTIVE_WINDOW = "active_window"
    WEEKEND_LIFESTYLE = "weekend_lifestyle"
    CALENDAR_PATTERN = "calendar_pattern"
    HEALTH_GOAL = "health_goal"
    EVENING_FLEXIBLE = "evening_flexible"
    RECOVERY_BREAK = "recovery_break"
    TASK_TYPE = "task_type"

    # ============================================================================
    # Overall Insight Only Groups
    # ============================================================================
    MOOD_BASED = "mood_based"
    EVENING_AFTER_WORK = "evening_after_work"


class FinanceDataConstants:
    """Constants class for finance data preparation"""

    # ============================================================================
    # Finance Data Keys
    # ============================================================================
    KEY_FINANCE_SCORE = "finance_score"
    KEY_FINANCE_THRESHOLD = 75
    KEY_SAVINGS_RATE_THRESHOLD = 0.20
