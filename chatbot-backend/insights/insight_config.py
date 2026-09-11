"""
Insight Pipeline Configurations

Centralized configuration for all thresholds, magic numbers, and default fallbacks
used by the insight generation pipeline (DataProcessor & Processors).
"""


class UrgencyLevel:
    ON_TRACK = "on_track"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


class HealthLevel:
    OK = "ok"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    ELEVATED = "elevated"
    CRITICAL = "critical"
    NONE = "none"


class CalendarLoad:
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    OVERLOADED = "overloaded"


class TrendStatus:
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    AHEAD = "ahead"
    SIMILAR = "similar"
    BEHIND = "behind"
    UNDERPERFORMING = "underperforming"
    INSUFFICIENT_DATA = "insufficient_data"


class InsightUrgencyConfig:
    """Urgency ranks for aggregation across different signal types (higher = more urgent)."""

    URGENCY_RANKS = {
        # Goal/Behavior levels
        UrgencyLevel.ON_TRACK: 0,
        UrgencyLevel.LOW: 1,
        UrgencyLevel.MODERATE: 2,
        UrgencyLevel.HIGH: 3,
        UrgencyLevel.SEVERE: 4,
        # Calendar load levels
        CalendarLoad.LOW: 0,
        CalendarLoad.MODERATE: 1,
        CalendarLoad.HIGH: 2,
        CalendarLoad.OVERLOADED: 3,
    }

    HEALTH_RANKS = {
        # Health signal levels
        HealthLevel.OK: 0,
        HealthLevel.MILD: 1,
        HealthLevel.MODERATE: 2,
        HealthLevel.SEVERE: 3,
        # Body/Stress levels
        HealthLevel.NONE: 0,
        HealthLevel.ELEVATED: 2,
        HealthLevel.CRITICAL: 3,
    }


class CalendarConfig:
    """Calendar Intelligence thresholds and constants."""

    MIN_GAP_MINUTES = 10
    LUNCH_WINDOW_START_HOUR = 11
    LUNCH_WINDOW_END_HOUR = 14

    # Active hours fallbacks
    DEFAULT_ACTIVE_START_HOUR = 8
    DEFAULT_ACTIVE_START_MINUTE = 0
    DEFAULT_ACTIVE_END_HOUR = 20
    DEFAULT_ACTIVE_END_MINUTE = 0

    # Cognitive Load Thresholds
    # Format: [Threshold for OVERLOADED, Threshold for HIGH, Threshold for MODERATE]
    B2B_COUNT_THRESHOLDS = [3, 2, 1]
    CONTINUOUS_MINUTES_THRESHOLDS = [180, 120, 60]

    # Meeting Classification Heuristics
    MEETING_KEYWORDS_FOCUS = ["focus", "deep work", "block", "do not disturb"]
    MEETING_KEYWORDS_ADMIN = [
        "standup",
        "update",
        "status",
        "sync",
        "planning",
        "review",
        "brainstorm",
        "all hands",
        "meeting",
        "team",
    ]
    MEETING_KEYWORDS_SOCIAL = [
        "1:1",
        "1-on-1",
        "catch-up",
        "lunch",
        "coffee",
        "chat",
        "connect",
    ]


class HealthConfig:
    """Health Signal thresholds and constants."""

    # Active window fallbacks — ONLY used when user_profile does not have these fields
    # DEFAULT_ACTIVE_START_HOUR = 8.0
    # DEFAULT_ACTIVE_END_HOUR = 20.0
    # DEFAULT_CURRENT_HOUR = 12.0

    # Sleep thresholds
    SLEEP_DEBT_OK = 0.5
    SLEEP_DEBT_MILD = 1.0
    SLEEP_DEBT_MODERATE = 2.0
    # Everything > 2.0 is severe
    SLEEP_TREND_SIGNIFICANT_DIFF = 0.5  # Hours difference to trigger trend change

    # Sleep score → quality buckets (aligned with PrepareHealthData)
    SLEEP_SCORE_VERY_HIGH = 96
    SLEEP_SCORE_HIGH = 81
    SLEEP_SCORE_OK = 61
    SLEEP_SCORE_LOW = 41

    # Steps thresholds
    # STEPS_DEFAULT_GOAL = 10000  # REMOVED — always read from user_profile.targets
    # Pace ratio (actual / expected_by_now)
    STEPS_PACE_OK = 0.9
    STEPS_PACE_MILD = 0.7
    STEPS_PACE_MODERATE = 0.5
    # Everything < 0.5 is severe

    # Heart Rate Thresholds
    HR_ELEVATED_ABOVE_BASELINE = 10
    HR_CRITICAL_ABOVE_BASELINE = 20
    HR_ELEVATED_ABSOLUTE = 90
    HR_CRITICAL_ABSOLUTE = 100


class BehaviorConfig:
    """Behavioral Pattern thresholds and constants."""

    HISTORY_CUTOFF_DAYS = 28
    MIN_DAYS_REQUIRED = 14
    UNDERPERFORMING_RATIO = 0.8
    STABLE_TREND_LOWER_BOUND = 0.9
    STABLE_TREND_UPPER_BOUND = 1.1
    TREND_DELTA = 0.25
    SIMILAR_THRESHOLD = 0.10


class GoalConfig:
    """Goal Progress thresholds and constants."""

    DEFAULT_DAYS_REMAINING = 1
    # Multiplier to determine if off-track (e.g. pace is less than half expected)
    SEVERE_PACE_RATIO = 0.5

    # Steps Goal Thresholds
    STEPS_PACE_THRESHOLDS = [0.9, 0.6, 0.3]
    STEPS_MIN_GAP_MINUTES = 15

    # Sleep Goal Thresholds
    SLEEP_PACE_THRESHOLDS = [0.9, 0.7, 0.5]
    SLEEP_MIN_GAP_MINUTES = 20
    SLEEP_MIN_DEBT_FOR_NAP = 0.5

    # Custom Goal Thresholds
    CUSTOM_PACE_THRESHOLDS = [0.9, 0.6, 0.3]
