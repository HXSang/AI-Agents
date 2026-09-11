"""Context signal thresholds (sleep_window, weekend, work_hours, recent_workout)."""

# Significant workout threshold (minutes)
CONTEXT_RECENT_WORKOUT_MIN_DURATION = 30
# Treat as "recent" when end time is unknown but workout is today
CONTEXT_RECENT_WORKOUT_MAX_HOURS = 12

# Priorities (critical > high > medium > low)
CONTEXT_PRIORITY_SLEEP_WINDOW = "critical"
CONTEXT_PRIORITY_RECENT_WORKOUT = "high"
CONTEXT_PRIORITY_WORK_HOURS = "medium"
CONTEXT_PRIORITY_WEEKEND = "low"
