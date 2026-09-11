from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


class MetaBlock(BaseModel):
    current_time: str
    time_phase: str
    day_of_week: str
    is_weekend: bool
    lifecycle_stage: str
    minutes_until_next_event: Optional[int] = None
    next_event_summary: Optional[str] = None
    next_event_start_time: Optional[str] = None
    next_event_category: Optional[str] = None
    next_event_type: Optional[str] = None
    recent_completed_tasks: List[str] = Field(default_factory=list)
    data_confidence: Dict[str, str] = Field(default_factory=dict)
    data_gaps: List[str] = Field(default_factory=list)
    hard_constraints: List[str] = Field(default_factory=list)
    active_hours_end_time: Optional[str] = None
    # Data availability context for narrative enrichment
    sleep_data_range: Optional[str] = None  # e.g., "3 ngày gần đây" or "Mon → Today"
    sleep_days_missing: Optional[int] = None  # number of days with no sleep data in range
    sleep_data_staleness_days: Optional[int] = None  # how many days since last sleep data
    sleep_last_data_date: Optional[str] = None  # actual date of last sleep data (for warning)
    hr_data_staleness_days: Optional[int] = None  # how many days since last HR data
    hr_last_data_date: Optional[str] = None  # actual date of last HR data
    steps_data_staleness_days: Optional[int] = None  # how many days since last steps data
    steps_last_data_date: Optional[str] = None  # actual date of last steps data
    mood_data_range: Optional[str] = None  # e.g., "7 ngày gần đây"
    mood_days_logged: Optional[int] = None  # how many days mood was logged
    mood_data_staleness_days: Optional[int] = None  # how many days since last mood data
    mood_last_data_date: Optional[str] = None  # actual date of last mood data
    energy_data_staleness_days: Optional[int] = None  # days since last energy reading
    energy_last_data_date: Optional[str] = None  # actual date of last energy data
    steps_data_range: Optional[str] = None


class SleepEvidence(BaseModel):
    """Provenance for one sleep dimension signal. Empty sub-dicts are dropped on dump."""

    metrics: Dict[str, Any] = Field(default_factory=dict)
    comparison: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)

    @model_serializer
    def _ser(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.metrics:
            out["metrics"] = self.metrics
        if self.comparison:
            out["comparison"] = self.comparison
        if self.context:
            out["context"] = self.context
        return out


class SleepDimensionSignal(BaseModel):

    status: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    comparison: Dict[str, Any] = Field(default_factory=dict)
    trend: Optional[str] = None  # improving / stable / declining / insufficient_data
    evidence: Optional[SleepEvidence] = None

    @model_serializer
    def _ser(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"status": self.status}
        # Only emit metrics if non-empty
        if self.metrics:
            out["metrics"] = self.metrics
        # Drop empty comparison blocks instead of emitting `{}`
        if self.comparison:
            out["comparison"] = self.comparison
        if self.trend is not None:
            out["trend"] = self.trend
        if self.evidence is not None:
            ev = self.evidence.model_dump()
            if ev:
                out["evidence"] = ev
        return out


class SleepSummaryMetrics(BaseModel):
    """Aggregated sleep stats over the summary window (default 7d)."""

    avg_duration_h: Optional[float] = None
    avg_rem_min: Optional[float] = None
    avg_core_min: Optional[float] = None
    avg_deep_min: Optional[float] = None
    avg_sleep_score: Optional[float] = None
    avg_efficiency: Optional[float] = None  # mean(duration_h / time_in_bed_h), 0–1
    avg_bedtime: Optional[str] = None  # HH:MM
    avg_wake_time: Optional[str] = None
    measured_days: int = 0
    missing_days: int = 0
    window_days: int = 7
    coverage_pct: Optional[float] = None


class SleepSignalsMap(BaseModel):
    """Independent sleep dimension signals for the Insight Engine."""

    sleep_duration: Optional[SleepDimensionSignal] = None
    sleep_efficiency: Optional[SleepDimensionSignal] = None
    sleep_consistency: Optional[SleepDimensionSignal] = None
    deep_sleep: Optional[SleepDimensionSignal] = None
    rem_sleep: Optional[SleepDimensionSignal] = None
    sleep_debt: Optional[SleepDimensionSignal] = None
    sleep_quality: Optional[SleepDimensionSignal] = None
    bedtime_timing: Optional[SleepDimensionSignal] = None
    sleep_trend: Optional[SleepDimensionSignal] = None
    recovery: Optional[SleepDimensionSignal] = None


class SleepAnomaly(BaseModel):
    kind: str
    night_date: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)


class SleepOverall(BaseModel):
    status: str  # good / fair / poor / insufficient_data
    score: Optional[float] = None  # 0–100
    confidence: str = "low"  # high / medium / low


class SleepSignal(BaseModel):
    """Evidence-rich sleep block (processor is single source of numerical truth).

    Primary payload for Insight Engine:
      summary → signals → anomalies → overall

    Flat convenience fields were removed; consumers must read summary/signals.
    """

    summary: SleepSummaryMetrics = Field(default_factory=SleepSummaryMetrics)
    signals: SleepSignalsMap = Field(default_factory=SleepSignalsMap)
    anomalies: List[SleepAnomaly] = Field(default_factory=list)
    overall: SleepOverall = Field(
        default_factory=lambda: SleepOverall(status="insufficient_data", confidence="low")
    )


class ActivitySignal(BaseModel):
    """Legacy flat steps/activity payload (kept for older consumers)."""

    level: str  # not_started / ok / mild / moderate / severe
    steps_today: Optional[int] = None  # None when stale / no data — NOT 0.
    steps_goal: Optional[int] = None
    remaining_steps: Optional[int] = None
    pace_ratio: Optional[float] = None  # None when level is not_started or no fresh data
    active_hours_left: Optional[float] = None
    steps_streak: Optional[int] = 0
    workout_activity_type: Optional[str] = None
    workout_duration_min: Optional[int] = 0
    calories_burned_today: Optional[float] = None
    daily_health_score_trend: Optional[List[int]] = None
    estimated_fields: List[str] = []


class StepsSummaryMetrics(BaseModel):
    avg_steps: Optional[float] = None
    total_steps: Optional[float] = None
    avg_distance: Optional[float] = None
    total_distance: Optional[float] = None
    avg_health_score: Optional[float] = None
    days_measured: int = 0
    days_with_data: int = 0
    missing_days: int = 0
    window_days: int = 7
    coverage_pct: Optional[float] = None
    # Today / API mirrors
    today_steps: Optional[float] = None
    today_distance: Optional[float] = None
    daily_target: Optional[float] = None
    current_month_avg: Optional[float] = None
    previous_month_avg: Optional[float] = None


class StepsDimensionSignal(BaseModel):
    """Common signal envelope: status + metrics + comparison + trend."""

    status: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    comparison: Dict[str, Any] = Field(default_factory=dict)
    trend: Optional[str] = None  # improving / stable / declining / insufficient_data


class StepsSignalsMap(BaseModel):
    """Independent steps dimension signals for the Insight Engine."""

    activity_volume: Optional[StepsDimensionSignal] = None
    goal_achievement: Optional[StepsDimensionSignal] = None
    activity_consistency: Optional[StepsDimensionSignal] = None
    step_trend: Optional[StepsDimensionSignal] = None
    baseline_comparison: Optional[StepsDimensionSignal] = None
    activity_pattern: Optional[StepsDimensionSignal] = None


class StepsAnomaly(BaseModel):
    kind: str  # very_low_activity / extremely_high_activity / sudden_increase / sudden_decrease / partial_day
    day_date: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)


class StepsOverall(BaseModel):
    status: str  # sedentary / lightly_active / active / highly_active / moderate / insufficient_data
    score: Optional[float] = None  # 0–100
    confidence: str = "low"  # high / medium / low


class StepsSignal(BaseModel):
    """Evidence-rich steps block (processor is single source of numerical truth).

    Primary payload: summary → signals → anomalies → overall
    Flat fields kept for legacy ActivitySignal consumers.
    """

    summary: StepsSummaryMetrics = Field(default_factory=StepsSummaryMetrics)
    signals: StepsSignalsMap = Field(default_factory=StepsSignalsMap)
    anomalies: List[StepsAnomaly] = Field(default_factory=list)
    overall: StepsOverall = Field(
        default_factory=lambda: StepsOverall(
            status="insufficient_data", confidence="low"
        )
    )

    # Shared context across all signals — eliminates 6-copy redundancy of evidence.context
    common_context: Dict[str, Any] = Field(default_factory=dict)

    # ── Legacy / convenience flat fields (ActivitySignal parity) ─────────
    level: str = "unavailable"  # unavailable / not_started / ok / mild / moderate / severe
    steps_today: Optional[int] = None
    steps_goal: Optional[int] = None
    remaining_steps: Optional[int] = None
    pace_ratio: Optional[float] = None
    active_hours_left: Optional[float] = None
    steps_streak: Optional[int] = 0
    workout_activity_type: Optional[str] = None
    workout_duration_min: Optional[int] = 0
    calories_burned_today: Optional[float] = None
    daily_health_score_trend: Optional[List[int]] = None
    estimated_fields: List[str] = Field(default_factory=list)

    def to_activity_signal(self) -> "ActivitySignal":
        """Legacy flat mirror for older consumers (`health_signals.activity`)."""
        return ActivitySignal(
            level=self.level,
            steps_today=self.steps_today,
            steps_goal=self.steps_goal,
            remaining_steps=self.remaining_steps,
            pace_ratio=self.pace_ratio,
            active_hours_left=self.active_hours_left,
            steps_streak=self.steps_streak,
            workout_activity_type=self.workout_activity_type,
            workout_duration_min=self.workout_duration_min,
            calories_burned_today=self.calories_burned_today,
            daily_health_score_trend=self.daily_health_score_trend,
            estimated_fields=list(self.estimated_fields or []),
        )


class CardioAndStressSignal(BaseModel):
    resting_heart_rate: Optional[float] = None  # float to handle API values like 70.2
    baseline_resting_hr: Optional[float] = None
    latest_heart_rate: Optional[int] = None
    workout_avg_hr: Optional[int] = None
    workout_max_hr: Optional[int] = None
    stress_high: bool = False
    hrv_score: Optional[int] = None
    walking_heart_rate_avg: Optional[float] = None
    energy_week_vs_prior_month_pct: Optional[float] = None
    estimated_fields: List[str] = [] 


class HeartRateEvidence(BaseModel):
    """Machine-readable evidence for one heart-rate dimension signal."""

    metrics: Dict[str, Any] = Field(default_factory=dict)
    comparison: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)


class HeartRateDimensionSignal(BaseModel):
    """Common signal envelope: status + metrics + comparison + trend + evidence."""

    status: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    comparison: Dict[str, Any] = Field(default_factory=dict)
    trend: Optional[str] = None  # improving / stable / declining / insufficient_data
    evidence: HeartRateEvidence = Field(default_factory=HeartRateEvidence)


class HeartRateSummaryMetrics(BaseModel):
    """Aggregated HR stats over the summary window (default 7d)."""

    avg_latest_heart_rate: Optional[float] = None
    avg_resting_heart_rate: Optional[float] = None
    min_heart_rate: Optional[float] = None
    max_heart_rate: Optional[float] = None
    avg_health_score: Optional[float] = None
    days_measured: int = 0
    missing_days: int = 0
    window_days: int = 7
    coverage_pct: Optional[float] = None
    # Today / API mirrors
    today_resting: Optional[float] = None
    today_latest: Optional[float] = None
    today_hrv: Optional[float] = None
    baseline_resting: Optional[float] = None
    range_min: Optional[float] = None
    range_max: Optional[float] = None


class HeartRateSignalsMap(BaseModel):
    """Independent HR dimension signals for the Insight Engine."""

    resting_heart_rate: Optional[HeartRateDimensionSignal] = None
    heart_rate_variability: Optional[HeartRateDimensionSignal] = None
    heart_rate_range: Optional[HeartRateDimensionSignal] = None
    heart_rate_stability: Optional[HeartRateDimensionSignal] = None
    recovery_state: Optional[HeartRateDimensionSignal] = None
    recovery_trend: Optional[HeartRateDimensionSignal] = None
    baseline_comparison: Optional[HeartRateDimensionSignal] = None
    # Optional when data present
    walking_heart_rate: Optional[HeartRateDimensionSignal] = None
    workout_heart_rate: Optional[HeartRateDimensionSignal] = None
    stress_indicator: Optional[HeartRateDimensionSignal] = None


class HeartRateAnomaly(BaseModel):
    kind: str
    day_date: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)


class HeartRateOverall(BaseModel):
    status: str  # good / fair / poor / insufficient_data
    score: Optional[float] = None  # 0–100
    confidence: str = "low"  # high / medium / low


class HeartRateSignal(BaseModel):
    """Evidence-rich heart-rate block (processor is single source of numerical truth).

    Primary payload: summary → signals → anomalies → overall
    Flat fields kept for legacy consumers / cardio mirror.
    """

    summary: HeartRateSummaryMetrics = Field(default_factory=HeartRateSummaryMetrics)
    signals: HeartRateSignalsMap = Field(default_factory=HeartRateSignalsMap)
    anomalies: List[HeartRateAnomaly] = Field(default_factory=list)
    overall: HeartRateOverall = Field(
        default_factory=lambda: HeartRateOverall(
            status="insufficient_data", confidence="low"
        )
    )

    # ── Legacy / convenience flat fields ─────────────────────────────────
    resting_heart_rate: Optional[float] = None
    baseline_resting_hr: Optional[float] = None
    latest_heart_rate: Optional[int] = None
    workout_avg_hr: Optional[int] = None
    workout_max_hr: Optional[int] = None
    stress_high: bool = False
    hrv_score: Optional[int] = None
    walking_heart_rate_avg: Optional[float] = None
    estimated_fields: List[str] = Field(default_factory=list)


class EnergyEvidence(BaseModel):
    """Machine-readable evidence for one energy dimension signal."""

    metrics: Dict[str, Any] = Field(default_factory=dict)
    comparison: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)


class EnergyDimensionSignal(BaseModel):
    """Common signal envelope: status + metrics + comparison + trend + evidence."""

    status: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    comparison: Dict[str, Any] = Field(default_factory=dict)
    trend: Optional[str] = None  # improving / stable / declining / insufficient_data
    evidence: EnergyEvidence = Field(default_factory=EnergyEvidence)


class EnergySummaryMetrics(BaseModel):
    """Aggregated energy stats over the summary window (default 7d)."""

    avg_energy_burn: Optional[float] = None
    total_energy_burn: Optional[float] = None
    avg_resting_energy: Optional[float] = None
    avg_health_score: Optional[float] = None
    days_measured: int = 0
    missing_days: int = 0
    window_days: int = 7
    coverage_pct: Optional[float] = None
    # Today / API summary mirrors
    today_active: Optional[float] = None
    month_resting_total: Optional[float] = None
    current_month_avg: Optional[float] = None
    week_avg: Optional[float] = None
    previous_month_avg: Optional[float] = None


class EnergySignalsMap(BaseModel):
    """Independent energy dimension signals for the Insight Engine."""

    activity_volume: Optional[EnergyDimensionSignal] = None
    energy_consistency: Optional[EnergyDimensionSignal] = None
    energy_trend: Optional[EnergyDimensionSignal] = None
    baseline_comparison: Optional[EnergyDimensionSignal] = None
    activity_pattern: Optional[EnergyDimensionSignal] = None


class EnergyAnomaly(BaseModel):
    kind: str  # low_activity / high_activity / sudden_drop / sudden_spike
    day_date: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)


class EnergyOverall(BaseModel):
    status: str  # good / fair / poor / insufficient_data
    score: Optional[float] = None  # 0–100
    confidence: str = "low"  # high / medium / low


class EnergySignal(BaseModel):
    """Evidence-rich energy block (processor is single source of numerical truth).

    Primary payload: summary → signals → anomalies → overall
    Flat fields kept for legacy consumers.
    """

    summary: EnergySummaryMetrics = Field(default_factory=EnergySummaryMetrics)
    signals: EnergySignalsMap = Field(default_factory=EnergySignalsMap)
    anomalies: List[EnergyAnomaly] = Field(default_factory=list)
    overall: EnergyOverall = Field(
        default_factory=lambda: EnergyOverall(
            status="insufficient_data", confidence="low"
        )
    )

    # ── Legacy / convenience flat fields ─────────────────────────────────
    level: str = "unavailable"  # unavailable / not_started / ok / mild / moderate / severe
    month_resting_total: Optional[float] = None
    total_active_energy: Optional[float] = None
    current_month_avg: Optional[float] = None
    week_avg: Optional[float] = None
    previous_month_avg: Optional[float] = None
    week_vs_prior_month_pct: Optional[float] = None
    calories_burned_today: Optional[float] = None  # == total_active_energy
    prev_month_avg: Optional[float] = None  # == previous_month_avg
    active_minutes_today: Optional[float] = None
    workout_duration_min: Optional[int] = None
    workout_activity_type: Optional[str] = None
    estimated_fields: List[str] = Field(default_factory=list)


class MoodEvidence(BaseModel):
    """Machine-readable evidence for one mood dimension signal."""

    metrics: Dict[str, Any] = Field(default_factory=dict)
    comparison: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)


class MoodDimensionSignal(BaseModel):
    """Common signal envelope: status + metrics + comparison + trend."""

    status: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    comparison: Dict[str, Any] = Field(default_factory=dict)
    trend: Optional[str] = None


class MoodSummaryMetrics(BaseModel):
    """Mood snapshot + coverage summary."""

    has_mood: bool = False
    current_mood: Optional[str] = None
    current_mood_display_name: Optional[str] = None
    current_mood_score: Optional[int] = None  # internal only
    current_mood_date: Optional[str] = None
    mood_age_hours: Optional[float] = None
    entries_last_7_days: int = 0
    entries_last_14_days: int = 0
    entries_last_30_days: int = 0
    last_logged_days_ago: Optional[int] = None
    days_measured: int = 0
    coverage_pct: Optional[float] = None


class MoodSignalsMap(BaseModel):
    """Independent mood dimension signals for the Insight Engine."""

    snapshot: Optional[MoodDimensionSignal] = None
    baseline: Optional[MoodDimensionSignal] = None
    trend: Optional[MoodDimensionSignal] = None
    pattern: Optional[MoodDimensionSignal] = None
    stability: Optional[MoodDimensionSignal] = None
    frequency: Optional[MoodDimensionSignal] = None
    change: Optional[MoodDimensionSignal] = None
    correlation: Optional[MoodDimensionSignal] = None


class MoodAnomaly(BaseModel):
    kind: str
    day_date: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)


class MoodOverall(BaseModel):
    status: str  # positive / neutral / low / insufficient_data
    score: Optional[float] = None  # 0–100 internal wellness-ish composite
    confidence: str = "low"


class MoodSignal(BaseModel):
    """Evidence-rich mood block (processor is single source of numerical truth).

    Primary payload: summary → signals → anomalies → overall
    Flat fields kept for legacy consumers / cardio mirror.
    Mood is descriptive only — never diagnostic.
    """

    summary: MoodSummaryMetrics = Field(default_factory=MoodSummaryMetrics)
    signals: MoodSignalsMap = Field(default_factory=MoodSignalsMap)
    anomalies: List[MoodAnomaly] = Field(default_factory=list)
    overall: MoodOverall = Field(
        default_factory=lambda: MoodOverall(
            status="insufficient_data", confidence="low"
        )
    )

    # ── Legacy / convenience flat fields ─────────────────────────────────
    # REMOVED: All flat fields moved to MoodSummaryMetrics and MoodSignalsMap.
    # Access via mood.summary.current_mood / mood.signals.trend.status / etc.


class HealthScoreFiredSignal(BaseModel):
    """One deterministic health-score signal key from the config engine."""

    key: str  # e.g. HEALTH_SCORE_CRITICAL
    category: str  # state | change | trend | pattern
    severity: Optional[str] = None  # DROP ladder: LOW|MEDIUM|HIGH|CRITICAL
    priority: float = 0.0
    evidence: Dict[str, Any] = Field(default_factory=dict)


class HealthScoreSummaryMetrics(BaseModel):
    """Current + history snapshot for healthScore-only series."""

    current_score: Optional[float] = None
    previous_score: Optional[float] = None
    current_date: Optional[str] = None
    previous_date: Optional[str] = None
    timezone: Optional[str] = None
    delta: Optional[float] = None
    baseline_7d: Optional[float] = None
    baseline_30d: Optional[float] = None
    days_measured: int = 0
    window_days: int = 7
    coverage_pct: Optional[float] = None
    series: List[Dict[str, Any]] = Field(default_factory=list)


class HealthScoreAggregate(BaseModel):
    """Grouped insight candidate when multiple signals fire on the same day."""

    theme: Optional[str] = None  # HEALTH_SCORE_DECLINE | HEALTH_SCORE_IMPROVEMENT
    priority: Optional[str] = None  # high | medium | low
    current_score: Optional[float] = None
    previous_score: Optional[float] = None
    baseline: Optional[float] = None
    delta: Optional[float] = None
    signals: List[str] = Field(default_factory=list)


class HealthScoreOverall(BaseModel):
    status: str = "insufficient_data"
    # critical | low | declining | recovering | improving | volatile | stable | insufficient_data
    score: Optional[float] = None
    confidence: str = "low"


class HealthScoreSignal(BaseModel):
    """Evidence-rich healthScore block (techplan health-score.md).

    Uses only date + timezone + healthScore. Null scores are ignored.
    """

    summary: HealthScoreSummaryMetrics = Field(
        default_factory=HealthScoreSummaryMetrics
    )
    signals: List[HealthScoreFiredSignal] = Field(default_factory=list)
    aggregate: Optional[HealthScoreAggregate] = None
    overall: HealthScoreOverall = Field(
        default_factory=lambda: HealthScoreOverall(
            status="insufficient_data", confidence="low"
        )
    )


class HealthPeriodMetric(BaseModel):
    """Today vs week/month/baseline aggregates for one health metric."""

    today: Optional[float] = None
    week_avg: Optional[float] = None  # calendar Mon→yesterday / last 7d with data
    month_avg: Optional[float] = None  # API currentMonthAvg or ~30d snapshot avg
    prev_month_avg: Optional[float] = None
    baseline_avg: Optional[float] = None  # personal baseline (rolling lookback)
    delta_vs_baseline: Optional[float] = None
    delta_vs_week: Optional[float] = None


class HealthPeriodAggregates(BaseModel):
    """Week/month period calcs (ported from chat smart-suggestions logic)."""

    steps: Optional[HealthPeriodMetric] = None
    sleep_hours: Optional[HealthPeriodMetric] = None
    resting_heart_rate: Optional[HealthPeriodMetric] = None
    active_minutes: Optional[HealthPeriodMetric] = None
    total_workout_min: Optional[HealthPeriodMetric] = None
    calories_burned: Optional[HealthPeriodMetric] = None
    # ENERGY.summaryData mirrors
    energy_month_resting_total: Optional[float] = None  # totalRestingEnergy
    energy_total_active_today: Optional[float] = None  # totalActiveEnergy
    energy_current_month_avg: Optional[float] = None  # currentMonthAvg
    energy_week_avg: Optional[float] = None  # avgCurrentWeekEnergyBurn
    energy_prev_month_avg: Optional[float] = None  # previousMonthAvg
    energy_week_vs_prior_month_pct: Optional[float] = None
    rhr_elevated: Optional[bool] = None
    workout_high_recent_load: Optional[bool] = None


class ContextSignalItem(BaseModel):
    """Deterministic situational context for Narrative AI (not user-facing text)."""

    status: str  # active / inactive
    priority: str  # critical / high / medium / low
    evidence: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthSignalsBlock(BaseModel):
    """Five health focuses + period aggregates.

    Canonical focuses (match HEALTH_GROUPS): sleep, heart_rate, energy, steps, mood.
    ``activity`` / ``cardio_stress`` remain as legacy mirrors for older consumers.
    ``context_signals`` holds deterministic situational context (sleep window, weekend, …).
    ``health_score`` is the composite daily healthScore signal engine (optional).
    """

    sleep: SleepSignal
    heart_rate: HeartRateSignal
    energy: EnergySignal
    steps: StepsSignal
    mood: MoodSignal
    health_score: Optional[HealthScoreSignal] = None
    period_aggregates: Optional[HealthPeriodAggregates] = None
    context_signals: Dict[str, ContextSignalItem] = Field(default_factory=dict)
    # Legacy mirrors (flat steps / HR+mood+energy composite)
    activity: Optional[ActivitySignal] = None
    cardio_stress: Optional[CardioAndStressSignal] = None

    @model_validator(mode="before")
    @classmethod
    def _backfill_five_focuses(cls, data: Any) -> Any:
        """Accept legacy payloads that only had sleep/activity/cardio_stress."""
        if not isinstance(data, dict):
            return data
        activity = data.get("activity") or data.get("steps")
        cardio = data.get("cardio_stress") or {}
        if not isinstance(cardio, dict):
            cardio = (
                cardio.model_dump()
                if hasattr(cardio, "model_dump")
                else {}
            )
        if data.get("steps") is None and activity is not None:
            data["steps"] = activity
        if data.get("activity") is None and activity is not None:
            # Prefer flat ActivitySignal shape for legacy mirror
            if hasattr(activity, "to_activity_signal"):
                data["activity"] = activity.to_activity_signal()
            elif isinstance(activity, dict) and (
                "summary" in activity or "signals" in activity
            ):
                flat_keys = (
                    "level",
                    "steps_today",
                    "steps_goal",
                    "remaining_steps",
                    "pace_ratio",
                    "active_hours_left",
                    "steps_streak",
                    "workout_activity_type",
                    "workout_duration_min",
                    "calories_burned_today",
                    "daily_health_score_trend",
                    "estimated_fields",
                )
                data["activity"] = {k: activity.get(k) for k in flat_keys if k in activity}
                if "level" not in data["activity"]:
                    data["activity"]["level"] = "unavailable"
            else:
                data["activity"] = activity

        if data.get("heart_rate") is None:
            data["heart_rate"] = {
                "resting_heart_rate": cardio.get("resting_heart_rate"),
                "baseline_resting_hr": cardio.get("baseline_resting_hr"),
                "latest_heart_rate": cardio.get("latest_heart_rate"),
                "workout_avg_hr": cardio.get("workout_avg_hr"),
                "workout_max_hr": cardio.get("workout_max_hr"),
                "stress_high": cardio.get("stress_high", False),
                "hrv_score": cardio.get("hrv_score"),
                "walking_heart_rate_avg": cardio.get("walking_heart_rate_avg"),
                "estimated_fields": cardio.get("estimated_fields") or [],
            }
        if data.get("mood") is None:
            data["mood"] = {
                "primary_mood": cardio.get("primary_mood"),
                "mood_score_avg": cardio.get("mood_score_avg"),
                "mood_variance": cardio.get("mood_variance"),
                "mood_trend_7d": cardio.get("mood_trend_7d"),
                "mood_first": cardio.get("mood_first"),
                "mood_score_min": cardio.get("mood_score_min"),
                "mood_score_max": cardio.get("mood_score_max"),
                "mood_sequence": cardio.get("mood_sequence"),
                "mood_log_count": cardio.get("mood_log_count"),
                "mood_has_notes": cardio.get("mood_has_notes"),
                "mood_date": cardio.get("mood_date"),
                "estimated_fields": [],
            }
        if data.get("energy") is None:
            act = activity if isinstance(activity, dict) else (
                activity.model_dump() if hasattr(activity, "model_dump") else {}
            )
            active = act.get("calories_burned_today")
            prev = cardio.get("energy_week_vs_prior_month_pct")
            data["energy"] = {
                "level": "ok" if active is not None else "unavailable",
                "total_active_energy": active,
                "calories_burned_today": active,
                "week_vs_prior_month_pct": prev,
                "workout_duration_min": act.get("workout_duration_min"),
                "workout_activity_type": act.get("workout_activity_type"),
                "estimated_fields": [],
            }
        return data

    @classmethod
    def from_legacy_parts(
        cls,
        *,
        sleep: SleepSignal,
        activity: ActivitySignal,
        cardio: CardioAndStressSignal,
        heart_rate: Optional[HeartRateSignal] = None,
        energy: Optional[EnergySignal] = None,
        mood: Optional[MoodSignal] = None,
        steps: Optional[StepsSignal] = None,
        health_score: Optional[HealthScoreSignal] = None,
        period_aggregates: Optional[HealthPeriodAggregates] = None,
    ) -> "HealthSignalsBlock":
        """Build the 5-focus block, backfilling from legacy cardio/activity."""
        hr = heart_rate or HeartRateSignal(
            resting_heart_rate=cardio.resting_heart_rate,
            baseline_resting_hr=cardio.baseline_resting_hr,
            latest_heart_rate=cardio.latest_heart_rate,
            workout_avg_hr=cardio.workout_avg_hr,
            workout_max_hr=cardio.workout_max_hr,
            stress_high=cardio.stress_high,
            hrv_score=cardio.hrv_score,
            walking_heart_rate_avg=cardio.walking_heart_rate_avg,
            estimated_fields=list(cardio.estimated_fields or []),
        )
        active = activity.calories_burned_today
        en = energy or EnergySignal(
            level="ok" if active is not None else "unavailable",
            total_active_energy=active,
            calories_burned_today=active,
            workout_duration_min=activity.workout_duration_min,
            workout_activity_type=activity.workout_activity_type,
            week_vs_prior_month_pct=cardio.energy_week_vs_prior_month_pct,
            estimated_fields=[],
        )
        md = mood or MoodSignal(
            summary=MoodSummaryMetrics(),
            signals=MoodSignalsMap(),
            anomalies=[],
            overall=MoodOverall(status="insufficient_data", confidence="low"),
        )
        st = steps
        if st is None:
            # Promote flat ActivitySignal into StepsSignal envelope
            st = StepsSignal(
                level=activity.level,
                steps_today=activity.steps_today,
                steps_goal=activity.steps_goal,
                remaining_steps=activity.remaining_steps,
                pace_ratio=activity.pace_ratio,
                active_hours_left=activity.active_hours_left,
                steps_streak=activity.steps_streak,
                workout_activity_type=activity.workout_activity_type,
                workout_duration_min=activity.workout_duration_min,
                calories_burned_today=activity.calories_burned_today,
                daily_health_score_trend=activity.daily_health_score_trend,
                estimated_fields=list(activity.estimated_fields or []),
                overall=StepsOverall(status="insufficient_data", confidence="low"),
            )
        return cls(
            sleep=sleep,
            heart_rate=hr,
            energy=en,
            steps=st,
            mood=md,
            health_score=health_score,
            period_aggregates=period_aggregates,
            activity=st.to_activity_signal(),
            cardio_stress=cardio,
        )

class FreeWindow(BaseModel):
    start_time: str
    duration_mins: int
    position: str
    # When set: ISO end of this free slot (helps LLM not invent end time)
    ends_at: Optional[str] = None
    # What actually ends this slot — NEVER confuse with a later calendar event
    # active_hours_end | next_event | bedtime
    capped_by: Optional[str] = None


class NextEvent(BaseModel):
    title: str
    start_time: str
    minutes_away: int
    category: Optional[str] = None  # e.g. MEETINGS / SOCIAL / HEALTH / FOCUS
    event_type: Optional[str] = None  # e.g. meeting / focus / personal


class CurrentEvent(BaseModel):
    title: str
    start_time: str
    end_time: str
    minutes_left: int
    category: Optional[str] = None
    event_type: Optional[str] = None


class PreviousEvent(BaseModel):
    title: str
    end_time: str
    minutes_since: int
    category: Optional[str] = None
    event_type: Optional[str] = None


class CalendarEventBrief(BaseModel):
    """Lightweight upcoming/past event card for Narrative AI (title + category)."""

    title: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    minutes_away: Optional[int] = None
    category: Optional[str] = None
    event_type: Optional[str] = None


class MeetingTypeBreakdown(BaseModel):
    focus_blocks: int
    social: int
    admin: int
    other: int = 0
    # Category counts from API taxonomy (MEETINGS / HEALTH / SOCIAL / …)
    by_category: Dict[str, int] = Field(default_factory=dict)


class CalendarIntelligenceBlock(BaseModel):
    cognitive_load_so_far: str
    back_to_back_count: int
    continuous_minutes: int
    free_windows_today: List[FreeWindow]
    next_event: Optional[NextEvent] = None
    current_event: Optional[CurrentEvent] = None
    previous_event: Optional[PreviousEvent] = None
    upcoming_events: List[CalendarEventBrief] = Field(default_factory=list)
    meeting_type_breakdown: MeetingTypeBreakdown
    meeting_minutes_today: float = 0.0
    recent_meeting_count_2h: Optional[int] = 0
    nearest_reminder: Optional["UpcomingReminder"] = None
    overdue_reminder_titles: List[str] = Field(default_factory=list)
    work_events_hours_week: Optional[float] = None
    work_events_hours_weekend: Optional[float] = None
    work_load_high: Optional[bool] = None
    calendar_density: Optional[float] = None
    event_count_today: Optional[int] = None
    total_participant_count: Optional[int] = None
    events_with_location_count: Optional[int] = None
    unique_locations: List[str] = Field(default_factory=list)


class UpcomingReminder(BaseModel):
    title: str
    due_at: str  # ISO datetime in user TZ
    minutes_until: int  # negative when overdue
    priority: Optional[str] = None
    category: Optional[str] = None
    notes: Optional[str] = None


class GoalSignal(BaseModel):
    goal_type: str
    domain: str = "overall"
    goal_label: str
    urgency_level: str  # on_track / low / moderate / high
    pace_ratio: Optional[float] = None
    days_remaining: int
    daily_needed: float
    calendar_opportunity: Optional[FreeWindow] = None
    insight_summary: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class GoalSignalsBlock(BaseModel):
    goals: List[GoalSignal]


class WeekComparison(BaseModel):
    steps: str  # ahead / behind / similar
    sleep: str  # ahead / behind / similar
    focus_hours: str  # ahead / behind / similar


class BehavioralPatternsBlock(BaseModel):
    sleep_trend_7d: Optional[str] = None  # improving / stable / declining
    activity_trend_7d: Optional[str] = None  # improving / stable / declining
    this_week_vs_last: WeekComparison
    personal_context: str  # Pre-digested string about historical patterns
    meeting_fragmentation: Optional[str] = None
    work_span_trend: Optional[str] = None
    collaboration_load: Optional[str] = None
    task_completion_momentum: Optional[str] = None
    mood_avg_7d_label: Optional[str] = None


class ProductivitySignalsBlock(BaseModel):
    # Canonical processor-first block (doc-aligned): summary -> signals -> anomalies -> overall
    summary: Dict[str, Any] = Field(default_factory=dict)
    signals: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    overall: Dict[str, Any] = Field(default_factory=dict)
    source_processors: List[str] = Field(default_factory=list)

    # ── Legacy flat fields (kept for existing consumers) ───────────────────
    level: str  # ok / overloaded / underutilized
    meeting_fatigue: str  # none / mild / high
    focus_score: str  # low / moderate / high
    events_completion_rate: Optional[float] = None  # % events (calendar items with start/end) completed today — NOT reminders
    meeting_completion_rate: Optional[float] = None  # today's meeting completion rate
    reminders_completion_rate: Optional[float] = None  # % reminders completed (NOT a productivity signal)
    reminders_due_today: int = 0
    meetings_due_today: int = 0
    upcoming_deadlines: int = 0

    # ── 7-day trend fields (from /api/productivity/summaries) ───────────
    meeting_minutes_7d_avg: Optional[float] = None
    meeting_counts_7d_avg: Optional[float] = None
    meeting_completion_7d_avg: Optional[float] = None
    meeting_minutes_vs_avg_pct: Optional[float] = None
    focus_minutes_7d_avg: Optional[float] = None
    focus_minutes_vs_avg_pct: Optional[float] = None
    events_completion_7d_avg: Optional[float] = None
    events_completion_trend: Optional[str] = None  # improving / stable / declining
    chronic_overload_days: int = 0  # days in 7d with wholeDayMeetingCount > 0
    meeting_overload_days: int = 0  # days in 7d with meetingCount > 5
    active_days_7d: int = 0  # days with any activity

    # ── Commitments (from productivity_summaries_7d) ───────────────────
    commitments_7d_avg: Optional[float] = None  # avg completion rate
    commitments_7d_total: int = 0  # total commitments due in 7d
    commitments_7d_completed: int = 0  # total commitments completed in 7d

    # ── Reminder-derived signals (full payload, not just count) ─────────
    high_priority_reminders_7d: int = 0
    nearest_reminder_title: Optional[str] = None
    nearest_reminder_due_in_mins: Optional[int] = None
    nearest_reminder_priority: Optional[str] = None
    overdue_reminders_count: int = 0
    reminders_collision: bool = False
    reminders_by_type: Dict[str, int] = Field(default_factory=dict)
    reminders_dismissed_7d: int = 0
    reminders_pending_7d: int = 0

    # ── Work-hours contrast (from /api/calendar/work-hours) ─────────────
    work_hours_7d_scheduled_avg: Optional[float] = None
    work_hours_7d_target: Optional[float] = None
    work_hours_overload_days_7d: int = 0
    work_hours_underload_days_7d: int = 0


class FinanceSignalsBlock(BaseModel):
    budget_utilization: Optional[float] = None
    bills_due_today: int = 0
    cashflow_status: str  # positive / negative / neutral
    active_goals_on_track: int = 0
    active_goals_behind: int = 0
    critical_alerts: List[str] = Field(default_factory=list)


class BalanceSnapshotBlock(BaseModel):

    # Per-day scores (today, from /api/balance/score)
    productivity_score: Optional[float] = None
    health_score: Optional[float] = None
    finance_score: Optional[float] = None
    balance_score: Optional[float] = None
    balance_date: Optional[str] = None
    balance_timezone: Optional[str] = None

    # 7-day aggregates (from /api/balance range)
    balance_score_7d_avg: Optional[float] = None
    productivity_score_7d_avg: Optional[float] = None
    health_score_7d_avg: Optional[float] = None
    finance_score_7d_avg: Optional[float] = None
    # Variance on the headline number — high = volatile, low = steady.
    balance_score_7d_std: Optional[float] = None
    # Trend vs the previous 7 days (delta of averages).
    balance_score_trend_delta: Optional[float] = None
    # Days in window with score >= 70. Drives "you had 5 strong days".
    balance_score_7d_good_days: int = 0
    balance_score_7d_total_days: int = 0


class DailySnapshotData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    snapshot_date: str

    # Health
    step_count_for_one_day: Optional[int] = Field(default=None, alias="h_steps")
    active_minutes_for_one_day: Optional[int] = Field(default=None, alias="h_active_minutes")
    total_calories_burned_for_one_day_kcal: Optional[float] = Field(default=None, alias="h_calories_burned")
    exercise_session_count_for_one_day: Optional[int] = Field(default=None, alias="h_exercise_sessions")
    total_workout_duration_for_one_day_minutes: Optional[float] = Field(default=None, alias="h_total_workout_min")
    average_heart_rate_for_one_day_bpm: Optional[int] = Field(default=None, alias="h_avg_heart_rate")
    resting_heart_rate_for_one_day_bpm: Optional[int] = Field(default=None, alias="h_resting_heart_rate")
    sleep_duration_for_one_night_hours: Optional[float] = Field(default=None, alias="h_sleep_hours")
    sleep_quality_score_for_one_night: Optional[int] = Field(default=None, alias="h_sleep_quality")
    water_intake_for_one_day_liters: Optional[float] = Field(default=None, alias="h_water_liters")
    health_score_for_one_day: Optional[int] = Field(default=None, alias="h_health_score")
    sleep_score_for_one_night: Optional[int] = Field(default=None, alias="h_sleep_score")
    steps_goal_completion_percentage_for_one_day: Optional[float] = Field(default=None, alias="h_steps_goal_pct")

    # Alias attributes for backward-compatible property access
    h_steps: Optional[int] = Field(default=None, alias="step_count_for_one_day")
    h_active_minutes: Optional[int] = Field(default=None, alias="active_minutes_for_one_day")
    h_calories_burned: Optional[float] = Field(default=None, alias="total_calories_burned_for_one_day_kcal")
    h_exercise_sessions: Optional[int] = Field(default=None, alias="exercise_session_count_for_one_day")
    h_total_workout_min: Optional[float] = Field(default=None, alias="total_workout_duration_for_one_day_minutes")
    h_avg_heart_rate: Optional[int] = Field(default=None, alias="average_heart_rate_for_one_day_bpm")
    h_resting_heart_rate: Optional[int] = Field(default=None, alias="resting_heart_rate_for_one_day_bpm")
    h_sleep_hours: Optional[float] = Field(default=None, alias="sleep_duration_for_one_night_hours")
    h_sleep_quality: Optional[int] = Field(default=None, alias="sleep_quality_score_for_one_night")
    h_water_liters: Optional[float] = Field(default=None, alias="water_intake_for_one_day_liters")
    h_health_score: Optional[int] = Field(default=None, alias="health_score_for_one_day")
    h_sleep_score: Optional[int] = Field(default=None, alias="sleep_score_for_one_night")
    h_steps_goal_pct: Optional[float] = Field(default=None, alias="steps_goal_completion_percentage_for_one_day")

    # Productivity
    meeting_duration_for_one_day_minutes: Optional[int] = Field(default=None, alias="p_meeting_minutes")
    longest_meeting_duration_for_one_day_minutes: Optional[int] = Field(default=None, alias="p_longest_meeting_min")
    work_span_for_one_day_minutes: Optional[float] = Field(default=None, alias="p_work_span_minutes")
    focus_blocks_30min: Optional[int] = Field(default=None, alias="p_focus_blocks_30min")
    focus_blocks_60min: Optional[int] = Field(default=None, alias="p_focus_blocks_60min")
    task_completion_rate: Optional[float] = Field(default=None, alias="p_task_completion_rate")
    reminders_completed: Optional[int] = Field(default=None, alias="p_reminders_completed")
    events_after_9_pm_for_one_day_count: Optional[int] = Field(default=None, alias="p_events_after_9pm")
    total_events_for_one_day: Optional[int] = Field(default=None, alias="p_total_events")

    # Alias attributes for backward-compatible property access
    p_meeting_minutes: Optional[int] = Field(default=None, alias="meeting_duration_for_one_day_minutes")
    p_longest_meeting_min: Optional[int] = Field(default=None, alias="longest_meeting_duration_for_one_day_minutes")
    p_work_span_minutes: Optional[float] = Field(default=None, alias="work_span_for_one_day_minutes")
    p_focus_blocks_30min: Optional[int] = Field(default=None, alias="focus_blocks_30min")
    p_focus_blocks_60min: Optional[int] = Field(default=None, alias="focus_blocks_60min")
    p_task_completion_rate: Optional[float] = Field(default=None, alias="task_completion_rate")
    p_reminders_completed: Optional[int] = Field(default=None, alias="reminders_completed")
    p_events_after_9pm: Optional[int] = Field(default=None, alias="events_after_9_pm_for_one_day_count")
    p_total_events: Optional[int] = Field(default=None, alias="total_events_for_one_day")

    # Mood
    mood_primary_mood: Optional[str] = Field(default=None, alias="m_primary_mood")
    mood_first_mood: Optional[str] = Field(default=None, alias="m_first_mood")
    mood_score_avg: Optional[float] = Field(default=None, alias="m_mood_score_avg")
    mood_score_min: Optional[float] = Field(default=None, alias="m_mood_score_min")
    mood_score_max: Optional[float] = Field(default=None, alias="m_mood_score_max")
    mood_variance: Optional[float] = Field(default=None, alias="m_mood_variance")
    mood_sequence: Optional[List[str]] = Field(default=None, alias="m_mood_sequence")
    mood_log_count: Optional[int] = Field(default=None, alias="m_log_count")
    mood_has_notes: Optional[bool] = Field(default=None, alias="m_has_notes")

    # Alias attributes for backward-compatible property access
    m_primary_mood: Optional[str] = Field(default=None, alias="mood_primary_mood")
    m_first_mood: Optional[str] = Field(default=None, alias="mood_first_mood")
    m_mood_score_avg: Optional[float] = Field(default=None, alias="mood_score_avg")
    m_mood_score_min: Optional[float] = Field(default=None, alias="mood_score_min")
    m_mood_score_max: Optional[float] = Field(default=None, alias="mood_score_max")
    m_mood_variance: Optional[float] = Field(default=None, alias="mood_variance")
    m_mood_sequence: Optional[List[str]] = Field(default=None, alias="mood_sequence")
    m_log_count: Optional[int] = Field(default=None, alias="mood_log_count")
    m_has_notes: Optional[bool] = Field(default=None, alias="mood_has_notes")

    # Finance
    total_income: Optional[float] = Field(default=None, alias="f_total_income")
    total_expense: Optional[float] = Field(default=None, alias="f_total_expense")
    net_cashflow: Optional[float] = Field(default=None, alias="f_net_cashflow")
    budget_utilization: Optional[float] = Field(default=None, alias="f_budget_utilization")
    bills_due_today: Optional[int] = Field(default=None, alias="f_bills_due_today")
    goals_progress_avg: Optional[float] = Field(default=None, alias="f_goals_progress_avg")

    # Alias attributes for backward-compatible property access
    f_total_income: Optional[float] = Field(default=None, alias="total_income")
    f_total_expense: Optional[float] = Field(default=None, alias="total_expense")
    f_net_cashflow: Optional[float] = Field(default=None, alias="net_cashflow")
    f_budget_utilization: Optional[float] = Field(default=None, alias="budget_utilization")
    f_bills_due_today: Optional[int] = Field(default=None, alias="bills_due_today")
    f_goals_progress_avg: Optional[float] = Field(default=None, alias="goals_progress_avg")

    # Composite Scores
    wellness_score_for_one_day: Optional[float] = Field(default=None, alias="wellness_score")
    productivity_score: Optional[float] = None
    financial_health_score: Optional[float] = None
    overall_day_score: Optional[float] = None

    # Alias attributes for backward-compatible property access
    wellness_score: Optional[float] = Field(default=None, alias="wellness_score_for_one_day")

    # Data quality flag surfaced from BE snapshots so the narrative
    # can mention when a day's data is incomplete.
    data_completeness: Optional[float] = None


class HistoricalTrendsBlock(BaseModel):
    recent_days: List[DailySnapshotData] = Field(default_factory=list)
    averages_3d: Optional[DailySnapshotData] = None
    trends: Dict[str, str] = Field(default_factory=dict)


class UserProfileBlock(BaseModel):
    user_id: Optional[str] = None
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None
    region: Optional[str] = None
    timezone: Optional[str] = None
    climate_type: Optional[str] = None
    job_title: Optional[str] = None
    work_status: Optional[str] = None
    primary_goal: Optional[str] = None
    diet_type: Optional[str] = None
    smoking_status: Optional[str] = None
    alcohol_consumption: Optional[str] = None
    current_health_conditions: List[str] = Field(default_factory=list)


class ExtractedSignals(BaseModel):
    meta: Optional[MetaBlock] = None
    health_signals: Optional[HealthSignalsBlock] = None
    calendar_intelligence: Optional[CalendarIntelligenceBlock] = None
    goal_signals: Optional[GoalSignalsBlock] = None
    behavioral_patterns: Optional[BehavioralPatternsBlock] = None
    historical_trends: Optional[HistoricalTrendsBlock] = None
    user_profile: Optional[UserProfileBlock] = None
    productivity_signals: Optional[ProductivitySignalsBlock] = None
    raw_data: Optional[Dict[str, Any]] = None
    balance_score: Optional[Dict[str, Any]] = None  # legacy passthrough
    balance_snapshot: Optional[BalanceSnapshotBlock] = None  # new structured
    finance_signals: Optional[FinanceSignalsBlock] = None
    narrative_context: Optional[Dict[str, str]] = None


class StaticSignalsBlock(BaseModel):

    # Historical trends — daily snapshot rows strictly < today
    historical_trends: Optional[HistoricalTrendsBlock] = None

    # Behavioral patterns (7d trend strings — derived from historical)
    behavioral_patterns: Optional[BehavioralPatternsBlock] = None

    # Productivity 7d aggregates (meeting_minutes_7d_avg, chronic_overload_days, etc.)
    productivity_7d: Optional[Dict[str, Any]] = None
    # Reminder-derived weekly stats
    productivity_reminders_7d: Optional[Dict[str, Any]] = None
    # Work-hours weekly aggregates
    productivity_work_hours_7d: Optional[Dict[str, Any]] = None

    # Balance 7d aggregates (avg/std/trend_delta/good_days)
    balance_7d: Optional[Dict[str, Any]] = None

    # Health 7d aggregates derived from snapshot rows
    health_7d_aggregates: Optional[Dict[str, Any]] = None

    # User profile — almost immutable
    user_profile: Optional[UserProfileBlock] = None

    # Date stamp identifying the "cutoff" day these aggregates are valid
    # for. Used by the cache layer to refuse stale data (e.g. yesterday's
    # static block must not be served today).
    cutoff_date: Optional[str] = None


class DynamicSignalsBlock(BaseModel):

    # Meta (time_phase, current_time, day_of_week, is_weekend, ...)
    meta: Optional[MetaBlock] = None

    # Calendar — current/next/previous event, free windows, meetings today
    calendar_intelligence: Optional[CalendarIntelligenceBlock] = None

    # Health — last night sleep, steps_today, RHR, live stress, etc.
    health_signals: Optional[HealthSignalsBlock] = None

    # Productivity — today's level, meeting_fatigue, focus_score, tasks
    # completion_rate, today's reminders
    productivity_today: Optional[Dict[str, Any]] = None

    # Today's balance scores (4 fields from /api/balance/score)
    balance_today: Optional[Dict[str, Any]] = None

    # Goal signals — depend on current date vs goal due date
    goal_signals: Optional[GoalSignalsBlock] = None

    # Finance signals — current month rollup, bills_due_today
    finance_signals: Optional[FinanceSignalsBlock] = None

    # The raw_data dict for ad-hoc lookups (e.g. balance_score legacy)
    raw_data: Optional[Dict[str, Any]] = None


# ── NEW: Phase 3 Components ──────────────────────────────────────────
class InsightCaseRef(BaseModel):
    case_id: str
    domain: str
    narrative_hook: str
    emotional_angle: str
    style_guardrails: List[str]
    expert_technique: Optional[str] = None
    recommended_action_bias: Optional[str] = None
    base_urgency: str
    base_score: int
CalendarIntelligenceBlock.model_rebuild()
