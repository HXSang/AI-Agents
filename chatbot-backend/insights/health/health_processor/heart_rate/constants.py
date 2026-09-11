"""Heart-rate / cardio signal thresholds."""

# Resting HR elevated vs personal baseline (bpm)
HEART_RATE_RESTING_ELEVATED_DELTA_BPM = 7
HEART_RATE_RESTING_HIGH_BPM = 100

# Physiological sanity bounds (also mirrored in common for shared guards)
HEART_RATE_RESTING_SANITY_LOW_BPM = 40
HEART_RATE_RESTING_SANITY_HIGH_BPM = 180

# Recent workout minutes above baseline → high load
WORKOUT_HIGH_LOAD_DELTA_MIN = 45

HR_SUMMARY_WINDOW_DAYS = 7
HR_BASELINE_WINDOW_DAYS = 14
HR_MIN_DAYS_FOR_TREND = 2
HR_MIN_DAYS_FOR_STABILITY = 3
HR_MIN_DAYS_FOR_BASELINE = 4

# Trend significance (bpm for RHR / latest; points for HRV)
HR_TREND_SIGNIFICANT_RHR_BPM = 3.0
HR_TREND_SIGNIFICANT_HRV = 5.0
HR_TREND_WINDOWS_DAYS = (3, 7)

# RHR qualitative bands vs baseline
HR_RHR_NORMAL_DELTA_BPM = 3.0
HR_RHR_ELEVATED_DELTA_BPM = 7.0  # alias of elevated gate
HR_RHR_LOW_DELTA_BPM = -5.0

# HRV qualitative (ms / score units from API)
HR_HRV_LOW = 30
HR_HRV_FAIR = 40
HR_HRV_GOOD = 50
HR_HRV_EXCELLENT = 60

# Stability: RHR std-dev (bpm)
HR_STABILITY_HIGH_STD = 2.0
HR_STABILITY_MED_STD = 4.0

# Anomaly thresholds
HR_ANOMALY_HIGH_RHR_BPM = 80
HR_ANOMALY_LOW_RHR_BPM = 45
HR_ANOMALY_HIGH_LATEST_BPM = 120
HR_ANOMALY_LOW_LATEST_BPM = 45
HR_ANOMALY_LOW_HRV = 25
HR_ANOMALY_SUDDEN_RHR_JUMP_BPM = 8.0

# Recovery score weights
HR_RECOVERY_WEIGHT_RHR = 0.45
HR_RECOVERY_WEIGHT_HRV = 0.55

# Overall weights
HR_OVERALL_WEIGHT_RECOVERY = 0.40
HR_OVERALL_WEIGHT_STABILITY = 0.25
HR_OVERALL_WEIGHT_BASELINE = 0.20
HR_OVERALL_WEIGHT_TREND = 0.15

# Optional future-signal placeholders (enabled when data present)
HR_WALKING_AVAILABLE = True
HR_WORKOUT_AVAILABLE = True
HR_SLEEP_HR_AVAILABLE = True
