"""Mood signal thresholds (descriptive only — never diagnostic)."""

# Taxonomy (internal scores only — never expose to users in copy)
MOOD_SCORE_MAP = {
    "terrible": 1,
    "sad": 2,
    "okay": 3,
    "happy": 4,
    "amazing": 5,
    # aliases
    "low": 2,
    "neutral": 3,
    "good": 4,
    "great": 5,
}

MOOD_BASELINE_LOOKBACK_DAYS = 14
MOOD_TREND_LOOKBACK_DAYS = 7
MOOD_FREQUENCY_LOOKBACK_DAYS = (7, 14, 30)
MOOD_MIN_BASELINE_ENTRIES = 5
MOOD_MIN_TREND_ENTRIES = 3
MOOD_IMPROVED_DELTA = 1
MOOD_DECLINED_DELTA = -1
MOOD_STREAK_MIN_DAYS = 3
MOOD_VOLATILITY_MEDIUM_STD = 0.7
MOOD_VOLATILITY_HIGH_STD = 1.2

# Logging consistency bands (entries in 7d)
MOOD_LOGGING_HIGH_ENTRIES_7D = 5
MOOD_LOGGING_MED_ENTRIES_7D = 3

# Overall weights
MOOD_OVERALL_WEIGHT_CURRENT = 0.35
MOOD_OVERALL_WEIGHT_TREND = 0.25
MOOD_OVERALL_WEIGHT_STABILITY = 0.20
MOOD_OVERALL_WEIGHT_LOGGING = 0.20

# Positive / negative mood sets
MOOD_POSITIVE = frozenset({"happy", "amazing", "great", "good"})
MOOD_NEGATIVE = frozenset({"terrible", "sad", "low"})
