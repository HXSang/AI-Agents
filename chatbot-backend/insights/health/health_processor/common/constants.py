"""Shared lookback / sanity thresholds used across health signal groups."""

# Physiological sanity bounds (for input guardrails, not diagnosis)
HEART_RATE_RESTING_SANITY_LOW_BPM = 40
HEART_RATE_RESTING_SANITY_HIGH_BPM = 180
SLEEP_SEVERE_LOW_HOURS = 4.0
STEPS_VERY_LOW_DAY_PROGRESS = 2000

# Snapshot lookback windows
WEEK_LOOKBACK_DAYS = 7
MONTH_LOOKBACK_DAYS = 30
BASELINE_LOOKBACK_DAYS = 14
