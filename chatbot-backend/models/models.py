import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from utils.type_helpers import (
    safe_bool,
    safe_datetime,
    safe_float,
    safe_int,
    safe_list,
    safe_str,
)

Category = Literal["Great Job", "Opportunity", "Need Attention"]
PatternFamily = Literal["A", "B", "C", "D", "E", "F", "X"]
PhaseFocus = Literal["1", "2", "3", "all", "cross"]
InitSessionRequestType = Literal["greeting", "productivity", "health", "finance"]


class ChatMessage(BaseModel):
    message: str
    request_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    app_id: Optional[str] = None
    flow: str = "general"  # "general" or "onboarding"
    timezone: Optional[str] = None  # IANA timezone (e.g., 'Asia/Ho_Chi_Minh')
    quick_reply: Optional[Dict[str, str]] = Field(
        default=None,
        description="Quick reply button info when user clicks a quick reply button (e.g., {'value': 'confirm_request', 'label': 'Đồng ý'})",
    )
    persist_history: bool = Field(
        default=True,
        description="If False, skip persisting this turn into chat history and external conversation DB (used by internal/system prompts).",
    )


class InitSessionRequest(BaseModel):
    request_id: str
    session_id: Optional[str] = None
    user_id: str
    app_id: str
    timezone: str
    request_type: InitSessionRequestType
    request_content: str
    language: str


class ChatResponse(BaseModel):
    response: str
    session_id: str
    user_id: Optional[str] = None
    flow: str
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    connect_action: Dict[str, bool] = Field(default_factory=dict)
    flow_status: str = Field(default="in_progress")
    set_goal: Dict[str, Any] = Field(default_factory=dict)
    quick_reply: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Quick reply buttons (e.g., [{'value': 'confirm_request', 'label': 'Đồng ý'}, {'value': 'cancel_request', 'label': 'Huỷ bỏ'}])",
    )
    action_completed: str = Field(
        default="",
        description="Action completed (e.g., 'calendar', 'health_goal', 'user_profile')",
    )


class RabbitMQMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    user_id: str
    session_id: str
    app_id: Optional[str] = None
    message: str = ""
    flow: str = "general"
    timezone: Optional[str] = None  # IANA timezone (e.g., 'Asia/Ho_Chi_Minh')
    timestamp: datetime = Field(default_factory=datetime.now)
    quick_reply: Optional[Dict[str, str]] = Field(
        default=None,
        description="Quick reply button info when user clicks a quick reply button (e.g., {'value': 'confirm_request', 'label': 'Đồng ý'})",
    )
    request_type: Optional[str] = None
    request_content: Optional[str] = None
    language: Optional[str] = None
    is_auto_user_message: Optional[bool] = False
    persist_history: Optional[bool] = True


class ChatHistory(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]
    user_id: Optional[str] = None
    flow: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class PaginatedChatHistory(BaseModel):
    """Paginated chat history response"""

    messages: List[Dict[str, Any]]
    total: int
    page: int
    limit: int
    total_pages: int


class HealthCheck(BaseModel):
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, str]


class ConnectionData(BaseModel):
    user_id: str
    connection_type: str  # "health_app", "calendar", "email"
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class ConnectionResponse(BaseModel):
    status: str
    message: str
    user_id: str
    connection_type: str


class ProductivityInsightRequest(BaseModel):
    """Request model for productivity insight analysis"""

    user_id: str
    timezone: Optional[str] = None  # IANA timezone (e.g., 'Asia/Ho_Chi_Minh')
    provider_name: Optional[str] = ""  # Calendar provider (default: empty string)
    language: Optional[str] = "en-US"  # Language code (default: "en-US")
    force_update: Optional[bool] = False  # If true, bypass cache and recompute insight


class ProductivityInsightResponse(BaseModel):
    """Response model for productivity insight analysis"""

    status: str  # "success" or "error"
    user_id: str
    insight: Optional[str] = (
        None  # Q&A flow: rendered prose insight (post-render + post-polish)
    )
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthInsightRequest(BaseModel):
    """Request model for health insight analysis."""

    user_id: str
    timezone: Optional[str] = None  # IANA timezone (e.g., 'Asia/Ho_Chi_Minh')
    provider_name: Optional[str] = (
        ""  # Included for consistency with other insight endpoints
    )
    language: Optional[str] = "en-US"  # Language code
    force_update: Optional[bool] = False  # If true, bypass cache and recompute insight


class HealthInsightResponse(BaseModel):
    """Response model for health insight analysis."""

    status: str  # "success" or "error"
    user_id: str
    insight: Optional[str] = (
        None  # Q&A flow: rendered prose insight (post-render + post-polish)
    )
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class CalendarCategorizeEventInput(BaseModel):
    """Single event payload for calendar category detection."""

    summary: str = ""
    description: Optional[str] = ""
    location: Optional[str] = ""
    event_type: Optional[str] = ""


class CalendarCategorizeRequest(BaseModel):
    """Request model for calendar event auto-category."""

    user_id: str
    events: List[CalendarCategorizeEventInput] = Field(..., min_length=1, max_length=50)


class CalendarCategoryResult(BaseModel):
    """Category assignment for one event in the batch."""

    index: int
    category: str


class CalendarCategorizeResponse(BaseModel):
    """Response model for calendar event auto-category."""

    status: str
    user_id: str
    categories: List[CalendarCategoryResult]


class ConversationPayload(BaseModel):
    """Payload for saving conversation to database"""

    profileId: str = Field(..., description="User profile ID")
    timestamp: str = Field(
        ...,
        description="ISO format timestamp with Z (e.g., '2025-12-26T03:45:43.230Z')",
    )
    userMessage: str = Field(..., description="User's message")
    botResponse: str = Field(..., description="Bot's response")
    flow: str = Field(
        default="general", description="Flow type (general, onboarding, etc.)"
    )
    language: str = Field(..., description="Language code")


class ExternalAPIOnboardingPayload(BaseModel):
    """Payload for external API onboarding data"""

    user_id: str
    name: str
    profile_name: Optional[str] = None
    age: Optional[int] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = ""
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None
    region: Optional[str] = ""
    climate_type: Optional[str] = ""
    job_title: Optional[str] = None
    industry: Optional[str] = None
    employment_type: Optional[str] = None
    payment_type: Optional[str] = None
    annual_salary: Optional[float] = None
    prefer_currency: Optional[str] = None
    work_start_time: Optional[str] = ""
    work_end_time: Optional[str] = ""
    health_enabled: bool = False
    calendar_enabled: bool = False
    gmail_enabled: bool = False
    steps_per_day: Optional[float] = None
    heart_rate_avg: Optional[float] = None
    sleep_duration_last_night: Optional[float] = None
    calories_burned_per_day: Optional[float] = None
    current_health_conditions: List[str] = Field(default_factory=list)
    diet_type: Optional[str] = "balanced"
    smoking_status: Optional[str] = ""
    alcohol_consumption: Optional[str] = ""
    stress_level: Optional[int] = None
    target_steps_per_day: Optional[float] = None
    target_heart_rate_bpm: Optional[float] = None
    target_sleep_hours: Optional[float] = None
    target_water_intake_liters: Optional[float] = None
    break_interval_hours: Optional[float] = None
    break_duration_min: Optional[int] = None
    target_tasks_per_day: Optional[int] = None
    target_active_hours: Optional[float] = None
    active_hours_start_time: Optional[str] = None
    active_hours_end_time: Optional[str] = None
    bedtime_start: Optional[str] = None
    bedtime_end: Optional[str] = None
    daily_habit: List[str] = Field(default_factory=list)
    checkin_times: List[str] = Field(default_factory=list)
    primary_goal: Optional[str] = ""
    notifications_enabled: Optional[bool] = None
    critical_alerts_enabled: Optional[bool] = None
    reminders_enabled: Optional[bool] = None
    onboarding_completed: bool = True
    last_updated: Optional[str] = ""
    timezone: Optional[str] = None
    work_status: Optional[str] = None
    sync_email: Optional[str] = None
    sync_provider: Optional[str] = None

    @field_validator("last_updated", mode="before")
    def fix_timestamp(cls, v):
        if isinstance(v, str) and not v.endswith("Z"):
            v += "Z"
        return v

    @classmethod
    def from_user_data(
        cls, user_data: Dict[str, Any]
    ) -> "ExternalAPIOnboardingPayload":
        """Create payload from user data dictionary"""
        name = user_data.get("name", "")
        profile_name = user_data.get("profile_name", "")

        # If profile_name is not provided, create one from name + random UUID
        if not profile_name and name:
            random_uuid = str(uuid.uuid4())[:8]  # Take first 8 characters
            profile_name = f"{name}_{random_uuid}"
        elif not profile_name:
            profile_name = name  # Fallback to name if no name either

        # Use utility functions for type casting

        return cls(
            user_id=safe_str(user_data.get("user_id"), ""),
            name=safe_str(name, ""),
            profile_name=safe_str(profile_name, ""),
            age=safe_int(user_data.get("age")),
            date_of_birth=safe_str(user_data.get("date_of_birth")),
            gender=safe_str(user_data.get("gender"), ""),
            height_cm=safe_float(user_data.get("height_cm")),
            weight_kg=safe_float(user_data.get("weight_kg")),
            bmi=safe_float(user_data.get("bmi")),
            region=safe_str(user_data.get("region"), ""),
            climate_type=safe_str(user_data.get("climate_type"), ""),
            job_title=safe_str(user_data.get("job_title"), ""),
            industry=safe_str(user_data.get("industry"), ""),
            employment_type=safe_str(user_data.get("employment_type"), ""),
            payment_type=safe_str(user_data.get("payment_type"), ""),
            annual_salary=safe_float(user_data.get("annual_salary")),
            prefer_currency=safe_str(user_data.get("prefer_currency"), ""),
            work_start_time=safe_str(user_data.get("work_start_time"), ""),
            work_end_time=safe_str(user_data.get("work_end_time"), ""),
            health_enabled=safe_bool(user_data.get("health_enabled"), False),
            calendar_enabled=safe_bool(user_data.get("calendar_enabled"), False),
            gmail_enabled=safe_bool(user_data.get("gmail_enabled"), False),
            steps_per_day=safe_float(user_data.get("steps_per_day"), 0.0),
            heart_rate_avg=safe_float(user_data.get("heart_rate_avg"), 0.0),
            sleep_duration_last_night=safe_float(
                user_data.get("sleep_duration_last_night"), 0.0
            ),
            calories_burned_per_day=safe_float(
                user_data.get("calories_burned_per_day"), 0.0
            ),
            current_health_conditions=safe_list(
                user_data.get("current_health_conditions"), []
            ),
            diet_type=safe_str(user_data.get("diet_type"), "balanced"),
            smoking_status=safe_str(user_data.get("smoking_status"), ""),
            alcohol_consumption=safe_str(user_data.get("alcohol_consumption"), ""),
            stress_level=safe_int(user_data.get("stress_level")),
            target_steps_per_day=safe_float(user_data.get("target_steps_per_day")),
            target_heart_rate_bpm=safe_float(user_data.get("target_heart_rate_bpm")),
            target_sleep_hours=safe_float(user_data.get("target_sleep_hours")),
            target_water_intake_liters=safe_float(
                user_data.get("target_water_intake_liters")
            ),
            break_interval_hours=safe_float(user_data.get("break_interval_hours")),
            break_duration_min=safe_int(user_data.get("break_duration_min")),
            target_tasks_per_day=safe_int(user_data.get("target_tasks_per_day")),
            target_active_hours=safe_float(user_data.get("target_active_hours")),
            active_hours_start_time=safe_str(user_data.get("active_hours_start_time")),
            active_hours_end_time=safe_str(user_data.get("active_hours_end_time")),
            bedtime_start=safe_str(user_data.get("bedtime_start")),
            bedtime_end=safe_str(user_data.get("bedtime_end")),
            daily_habit=safe_list(user_data.get("daily_habit"), []),
            checkin_times=safe_list(user_data.get("checkin_times"), []),
            primary_goal=safe_str(user_data.get("primary_goal"), ""),
            notifications_enabled=safe_bool(user_data.get("notifications_enabled")),
            critical_alerts_enabled=safe_bool(user_data.get("critical_alerts_enabled")),
            reminders_enabled=safe_bool(user_data.get("reminders_enabled")),
            onboarding_completed=safe_bool(user_data.get("onboarding_completed"), True),
            last_updated=safe_str(user_data.get("last_updated"), ""),
            work_status=safe_str(user_data.get("work_status")),
            sync_email=safe_str(user_data.get("sync_email")),
            sync_provider=safe_str(user_data.get("sync_provider")),
            timezone=safe_str(user_data.get("timezone")),
        )


class HealthMetricEntry(BaseModel):
    """Individual health metric entry"""

    value: Optional[Any] = (
        None  # Can be int, float, or string (e.g., "120/80" for blood pressure)
    )
    date: Optional[str] = None  # ISO format datetime string
    source: Optional[str] = (
        None  # Source of the data (e.g., "iPhone", "Apple Watch", "Health App")
    )


class HealthMetrics(BaseModel):
    """Health metrics data"""

    steps: Optional[List[HealthMetricEntry]] = None
    heart_rate: Optional[List[HealthMetricEntry]] = None
    blood_pressure: Optional[List[HealthMetricEntry]] = None
    weight: Optional[List[HealthMetricEntry]] = None
    sleep: Optional[List[HealthMetricEntry]] = None
    calories: Optional[List[HealthMetricEntry]] = None
    distance: Optional[List[HealthMetricEntry]] = None
    flights_climbed: Optional[List[HealthMetricEntry]] = None
    active_energy: Optional[List[HealthMetricEntry]] = None
    resting_heart_rate: Optional[List[HealthMetricEntry]] = None
    blood_oxygen: Optional[List[HealthMetricEntry]] = None
    body_temperature: Optional[List[HealthMetricEntry]] = None


class HealthGoal(BaseModel):
    """Health goal configuration"""

    target_value: Optional[float] = None
    unit: Optional[str] = None
    set_date: Optional[str] = None  # ISO format datetime string
    metric_name: Optional[str] = None


class HealthGoals(BaseModel):
    """Health goals data"""

    steps: Optional[HealthGoal] = None
    weight: Optional[HealthGoal] = None
    heart_rate: Optional[HealthGoal] = None
    sleep: Optional[HealthGoal] = None
    calories: Optional[HealthGoal] = None
    distance: Optional[HealthGoal] = None
    active_energy: Optional[HealthGoal] = None
    blood_pressure: Optional[HealthGoal] = None


class HealthConsents(BaseModel):
    """Health data sharing consents"""

    steps: Optional[bool] = None
    heart_rate: Optional[bool] = None
    blood_pressure: Optional[bool] = None
    weight: Optional[bool] = None
    sleep: Optional[bool] = None
    calories: Optional[bool] = None
    distance: Optional[bool] = None
    flights_climbed: Optional[bool] = None
    active_energy: Optional[bool] = None
    resting_heart_rate: Optional[bool] = None
    blood_oxygen: Optional[bool] = None
    body_temperature: Optional[bool] = None


class HealthDataPayload(BaseModel):
    """Complete health data payload"""

    user_id: str
    metrics: HealthMetrics = Field(default_factory=HealthMetrics)
    goals: HealthGoals = Field(default_factory=HealthGoals)
    consents: HealthConsents = Field(default_factory=HealthConsents)


class InsightItem(BaseModel):
    """Individual insight item"""

    id: str
    type: str  # "warning", "success", "opportunity"
    title: str  # "Need Attention", "Great Job", "Opportunity"
    message: str
    highlight_text: List[str] = Field(default_factory=list)
    action_link: Optional[Dict[str, Any]] = None
    data_evidence: Optional[Dict[str, Any]] = None


class OverallInsightRequest(BaseModel):
    """Request model for overall insight analysis"""

    user_id: str
    timezone: Optional[str] = None  # IANA timezone (e.g., 'Asia/Ho_Chi_Minh')
    provider_name: Optional[str] = ""  # Calendar provider (default: empty string)
    language: Optional[str] = "en-US"  # Language code (default: "en-US")
    force_update: Optional[bool] = False  # If true, bypass cache and recompute insight


class OverallInsightResponse(BaseModel):
    """Response model for overall insight analysis"""

    status: str  # "success" or "error"
    user_id: str
    insight: Optional[Dict[str, Optional[str]]] = (
        None  # {great_job, need_attention, opportunity}
    )
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class MoodDTO(BaseModel):

    id: str = Field(..., description="Unique mood entry ID")
    mood: str = Field(
        ..., description="Mood value: terrible, sad, okay, happy, amazing"
    )
    mood_display_name: str = Field(
        ..., alias="moodDisplayName", description="Human-readable mood name"
    )
    notes: Optional[str] = Field(None, description="Optional notes about the mood")
    mood_date: str = Field(..., alias="moodDate", description="Mood date in ISO format")

    class Config:
        populate_by_name = True


class BalanceScoreDTO(BaseModel):
    """Balance score data per day from /api/balance endpoints."""

    date: Optional[str] = Field(
        None, alias="balance_score_date", description="Score date in YYYY-MM-DD"
    )
    timezone: Optional[str] = None
    productivityScore: Optional[float] = None
    healthScore: Optional[float] = None
    financeScore: Optional[float] = None
    balanceScore: Optional[float] = Field(
        None, alias="balance_health_score"
    )

    class Config:
        populate_by_name = True


# ============================================================================
# Health Summary API Models
# ============================================================================


class DateRange(BaseModel):
    """Date range for health summary"""

    startDate: str = Field(..., description="Start date in ISO format")
    endDate: str = Field(..., description="End date in ISO format")


class HeartRateRange(BaseModel):
    """Heart rate range"""

    max: int
    min: int


class SleepHeartRate(BaseModel):
    """Sleep heart rate data"""

    max: int
    min: int
    lastTimeRecord: Optional[str] = Field(
        None, description="Last time record in ISO format"
    )


class WorkoutHeartRate(BaseModel):
    """Workout heart rate data"""

    avgHR: Optional[int] = Field(None, description="Average heart rate during workout")
    maxHR: Optional[int] = Field(None, description="Maximum heart rate during workout")
    minHR: Optional[int] = Field(None, description="Minimum heart rate during workout")
    duration: Optional[int] = Field(None, description="Workout duration in seconds")
    activityType: Optional[str] = Field(
        None, description="Type of activity (e.g., 'Running', 'Cycling', 'Walking')"
    )


class MonthlyAverage(BaseModel):
    """Monthly average for steps"""

    steps: int
    distance: float


# ============================================================================
# Daily Data Entry Models (varies by type)
# ============================================================================


class EnergyDataEntry(BaseModel):
    """Daily energy data entry"""

    date: str = Field(..., description="Date in ISO format")
    energyBurn: int
    healthScore: int


class HRDataEntry(BaseModel):
    """Daily heart rate data entry"""

    date: str = Field(..., description="Date in ISO format")
    latestHR: int
    healthScore: int
    latestHRTime: Optional[str] = Field(
        None, description="Latest HR time in ISO format"
    )
    sleepHeartRate: Optional[SleepHeartRate] = None
    restingHeartRate: int
    workoutHeartRate: Optional[WorkoutHeartRate | int] = None
    walkingHeartRateAverage: Optional[int] = None


class SleepDataEntry(BaseModel):
    """Daily sleep data entry"""

    date: str = Field(..., description="Date in ISO format")
    rem: float
    core: float
    deep: float
    total: float
    sleepScore: Optional[int] = Field(None, alias="sleepScore")
    healthScore: int
    lastWakeTime: Optional[str] = Field(
        None, description="Last wake time in ISO format (e.g., '2026-02-04T23:35:00Z')"
    )
    firstSleepTime: Optional[str] = Field(
        None,
        description="First sleep time in ISO format (e.g., '2026-02-04T19:33:00Z')",
    )

    class Config:
        populate_by_name = True


class StepsDataEntry(BaseModel):
    """Daily steps data entry"""

    date: str = Field(..., description="Date in ISO format")
    steps: int
    distance: float
    healthScore: int


# ============================================================================
# Summary Data Models (varies by type)
# ============================================================================


class EnergySummaryData(BaseModel):

    totalRestingEnergy: int
    currentMonthAvg: int
    totalActiveEnergy: int
    avgCurrentWeekEnergyBurn: int
    previousMonthAvg: int


class HRSummaryData(BaseModel):
    """Heart rate summary data"""

    heartRateRange: HeartRateRange
    average_heart_rate_variability_current: int = Field(
        validation_alias="avgHRV"
    )


class SleepSummaryData(BaseModel):
    """Sleep summary data"""

    average_light_sleep_hours: float = Field(validation_alias="avgCore")
    average_awake_hours: float = Field(validation_alias="avgAwake")
    average_time_in_bed_hours: float = Field(validation_alias="avgTimeInBed")
    average_total_sleep_hours: float = Field(validation_alias="avgTimeAsleep")
    average_deep_sleep_hours: float = Field(validation_alias="avgDeep")
    average_rem_sleep_hours: float = Field(validation_alias="avgRem")


class StepsSummaryData(BaseModel):
    """Steps summary data"""

    dailyTarget: int
    currentMonthAvg: MonthlyAverage
    totalSteps: int
    totalDistance: float
    previousMonthAvg: MonthlyAverage


# ============================================================================
# Health Summary Models (base and type-specific)
# ============================================================================


class HealthSummaryBase(BaseModel):
    """Base model for health summary"""

    id: str
    type: str  # "ENERGY", "HR", "SLEEP", "STEPS"
    dateRange: DateRange
    daysWithData: int
    createdAt: str = Field(..., description="Created at timestamp in ISO format")
    updatedAt: str = Field(..., description="Updated at timestamp in ISO format")


class EnergyHealthSummary(HealthSummaryBase):
    """Energy health summary"""

    type: str = Field(default="ENERGY", frozen=True)
    summaryData: EnergySummaryData
    data: List[EnergyDataEntry]


class HRHealthSummary(HealthSummaryBase):
    """Heart rate health summary"""

    type: str = Field(default="HR", frozen=True)
    summaryData: HRSummaryData
    data: List[HRDataEntry]


class SleepHealthSummary(HealthSummaryBase):
    """Sleep health summary"""

    type: str = Field(default="SLEEP", frozen=True)
    summaryData: SleepSummaryData
    data: List[SleepDataEntry]
    sleepScore: Optional[int] = Field(
        None, description="Overall sleep score for the period"
    )


class StepsHealthSummary(HealthSummaryBase):
    """Steps health summary"""

    type: str = Field(default="STEPS", frozen=True)
    summaryData: StepsSummaryData
    data: List[StepsDataEntry]


# Union type for all health summary types
HealthSummary = (
    EnergyHealthSummary | HRHealthSummary | SleepHealthSummary | StepsHealthSummary
)


# ============================================================================
# Monthly Insight Models
# ============================================================================


def _be_snapshot_alias(snake: str) -> str:
    
    parts = snake.split("_")
    if len(parts) >= 2 and parts[0] in ("h", "m", "f", "c", "p"):
        prefix, head, *rest = parts
        return f"{prefix}{head}" + "".join(p.capitalize() for p in rest)
    head, *tail = parts
    return head + "".join(p.capitalize() for p in tail)


class DailySnapshot(BaseModel):
    
    model_config = ConfigDict(
        extra="ignore",
        from_attributes=True,
        alias_generator=_be_snapshot_alias,
        populate_by_name=True,
    )

    profile_id: str
    snapshot_date: date
    user_id: Optional[str] = None

    # Health (h_)
    h_steps: Optional[int] = None
    h_active_minutes: Optional[int] = None
    h_calories_burned: Optional[float] = None
    h_exercise_sessions: Optional[int] = None
    h_total_workout_min: Optional[int] = None
    h_avg_heart_rate: Optional[float] = None
    h_resting_heart_rate: Optional[float] = None
    h_sleep_hours: Optional[float] = None
    h_sleep_quality: Optional[float] = None
    h_water_liters: Optional[float] = None
    h_calories_consumed: Optional[int] = None
    h_weight_kg: Optional[float] = None
    h_health_score: Optional[float] = None
    h_sleep_score: Optional[float] = None
    h_steps_goal_pct: Optional[float] = None
    h_activity_level: Optional[str] = None
    h_active_conditions: Optional[List[str]] = None

    # Productivity (p_)
    p_total_events: Optional[int] = None
    p_confirmed_events: Optional[int] = None
    p_meeting_minutes: Optional[float] = None
    p_longest_meeting_min: Optional[float] = None
    p_avg_meeting_min: Optional[float] = None
    p_work_span_minutes: Optional[float] = None
    p_unique_collaborators: Optional[int] = None
    p_online_meetings: Optional[int] = None
    p_focus_blocks_30min: Optional[int] = None
    p_focus_blocks_60min: Optional[int] = None
    p_short_events_count: Optional[int] = None
    p_events_after_9pm: Optional[int] = None
    p_reminders_due: Optional[int] = None
    p_reminders_completed: Optional[int] = None
    p_reminders_overdue: Optional[int] = None
    p_task_completion_rate: Optional[float] = None

    # Mood (m_)
    m_primary_mood: Optional[str] = None
    m_first_mood: Optional[str] = None
    m_mood_score_avg: Optional[float] = None
    m_mood_score_min: Optional[float] = None
    m_mood_score_max: Optional[float] = None
    m_mood_variance: Optional[float] = None
    m_mood_sequence: Optional[List[str]] = None
    m_log_count: Optional[int] = None
    m_has_notes: Optional[bool] = None

    # Finance (f_)
    f_total_income: Optional[float] = None
    f_total_expense: Optional[float] = None
    f_net_cashflow: Optional[float] = None
    f_transaction_count: Optional[int] = None
    f_expense_by_category: Optional[Dict[str, float]] = None
    f_budget_utilization: Optional[float] = None
    f_budgets_over_limit: Optional[int] = None
    f_budgets_critical: Optional[int] = None
    f_goals_progress_avg: Optional[float] = None
    f_goals_on_track: Optional[int] = None
    f_goals_behind: Optional[int] = None
    f_bills_due_today: Optional[int] = None
    f_bills_due_today_amount: Optional[float] = None
    f_bills_overdue: Optional[int] = None
    f_expense_logged: Optional[bool] = None
    f_active_goals_count: Optional[int] = None

    # Composite scores
    wellness_score: Optional[float] = None
    productivity_score: Optional[float] = None
    financial_health_score: Optional[float] = None
    overall_day_score: Optional[float] = None

    # Data quality
    has_health_data: bool = False
    has_calendar_data: bool = False
    has_mood_data: bool = False
    has_finance_data: bool = False
    data_completeness: float = 0.0


class PhaseMetrics(BaseModel):
    """Aggregated metrics for one of the 3 month phases (days 1-10, 11-20, 21-EOM)."""

    model_config = ConfigDict(extra="forbid")

    phase: Literal[1, 2, 3]
    phase_start_date: date
    phase_end_date: date
    days_with_data: int

    h_steps_avg: Optional[float] = None
    h_sleep_hours_avg: Optional[float] = None
    h_sleep_hours_std: Optional[float] = None
    h_sleep_quality_avg: Optional[float] = None
    h_health_score_avg: Optional[float] = None
    h_active_minutes_avg: Optional[float] = None
    h_total_workout_min_sum: Optional[int] = None
    p_meeting_minutes_avg: Optional[float] = None
    p_total_events_sum: Optional[int] = None
    p_focus_blocks_30min_sum: Optional[int] = None
    p_focus_blocks_60min_sum: Optional[int] = None
    p_task_completion_rate_avg: Optional[float] = None
    p_late_evening_event_days: Optional[int] = None
    p_heavy_meeting_days: Optional[int] = None
    p_reminders_due_sum: Optional[int] = None
    p_reminders_completed_sum: Optional[int] = None
    p_reminders_overdue_sum: Optional[int] = None
    m_mood_score_avg: Optional[float] = None
    m_mood_score_std: Optional[float] = None
    m_mood_range_avg: Optional[float] = None  # daily (max - min) averaged
    f_total_expense_sum: Optional[float] = None
    f_transaction_count_sum: Optional[int] = None
    f_bills_overdue_sum: Optional[int] = None
    f_bills_due_today_amount_sum: Optional[float] = None
    f_budgets_over_limit_max: Optional[int] = None  # peak day count
    f_budgets_critical_max: Optional[int] = None
    wellness_score_avg: Optional[float] = None
    productivity_score_avg: Optional[float] = None
    financial_health_score_avg: Optional[float] = None
    overall_day_score_avg: Optional[float] = None
    overall_day_score_std: Optional[float] = None


class MonthlySnapshot(BaseModel):
    """Aggregated month-level snapshot, includes 3 phases + cross-module correlations."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    year: int
    month: int
    month_start_date: date
    month_end_date: date
    days_with_data: int
    phases: List[PhaseMetrics] = Field(min_length=3, max_length=3)

    h_sleep_hours_avg: Optional[float] = None
    h_health_score_avg: Optional[float] = None
    h_total_workout_min_sum: Optional[int] = None
    p_meeting_minutes_avg: Optional[float] = None
    p_task_completion_rate_avg: Optional[float] = None
    p_total_events_sum: Optional[int] = None
    productivity_score_avg: Optional[float] = None
    p_reminders_due_sum: Optional[int] = None
    p_reminders_completed_sum: Optional[int] = None
    p_reminders_overdue_sum: Optional[int] = None
    m_mood_score_avg: Optional[float] = None
    m_mood_range_avg: Optional[float] = None
    f_total_expense_sum: Optional[float] = None
    financial_health_score_avg: Optional[float] = None
    f_bills_overdue_total: Optional[int] = None
    f_bills_due_today_amount_total: Optional[float] = None
    f_budgets_over_limit_max: Optional[int] = None
    f_expense_by_category_sum: Optional[Dict[str, float]] = None
    overall_day_score_avg: Optional[float] = None
    overall_day_score_std: Optional[float] = None

    # 6 cross-module Pearson r
    corr_work_sleep: Optional[float] = None
    corr_work_health: Optional[float] = None
    corr_work_spend: Optional[float] = None
    corr_sleep_mood: Optional[float] = None
    corr_workout_mood: Optional[float] = None
    corr_meeting_mood: Optional[float] = None
    corr_workout_productivity: Optional[float] = None

    # Module balance
    balance_score: Optional[float] = None
    most_concentrated_area: Optional[str] = None
    module_scores: Optional[Dict[str, float]] = None
    module_max_excess_pct: Optional[float] = None

    # Goal-vs-Reality module averages (Rules §3.D) — None when source data missing.
    goal_alignment_health_pct: Optional[float] = None  # h_steps_goal_pct
    goal_alignment_productivity_pct: Optional[float] = None  # p_task_completion_rate
    goal_alignment_finance_pct: Optional[float] = None  # f_goals_progress_avg


class InsightCandidate(BaseModel):
    """Pre-language-layer detector output."""

    model_config = ConfigDict(extra="forbid")

    pattern_family: PatternFamily
    pattern_type: str
    tentative_category: Category
    phase_focus: PhaseFocus
    raw_template_key: str
    raw_signal: Dict[str, Any]
    signal_strength: float = Field(ge=0.0, le=1.0)
    cross_module_impact: float = Field(ge=0.0, le=1.0)


class Insight(BaseModel):
    """Final ranked insight ready for client + persistence."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    category: Category
    pattern_family: PatternFamily
    pattern_type: str
    phase_focus: PhaseFocus
    text: str

    score: float
    signal_strength: float
    novelty: float
    cross_module_impact: float
    raw_signal: Dict[str, Any]
    insight_hash: str


class MonthlyInsightRequest(BaseModel):
    
    user_id: str  # passed as path segment under `/api/users/{user_id}/...`
    timezone: Optional[str] = None  # IANA TZ (e.g. "Asia/Ho_Chi_Minh")
    provider_name: Optional[str] = ""  # reserved for symmetry with other insight reqs
    language: Optional[str] = "en-US"  # IETF tag — see agents.prompt.get_language_name
    force_update: Optional[bool] = False
    year: Optional[int] = None  # override auto-derive when set
    month: Optional[int] = None  # override auto-derive when set (1..12)


class MonthlyInsightResponse(BaseModel):
    
    status: str  # "success" or "error"
    user_id: str  # echoes request profile_id (kept name "user_id" for FE consistency)
    insight: Optional[Dict[str, Optional[str]]] = (
        None  # {great_job, need_attention, opportunity}; value is None when category empty
    )
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class DebugPipelineRequest(BaseModel):
    """Request model for debugging the V3 Insight pipeline block-by-block"""

    raw_data: Dict[str, Any]
    domain: str = "health"
    language: str = "vi-VN"


class DomainDebugResult(BaseModel):
    """Debug output for a specific domain."""

    phase2_selected_keyword: Optional[Dict[str, Any]] = None
    phase2_action_hint: Optional[str] = None
    phase3_reasoning_plan: Optional[Dict[str, Any]] = None
    phase4_native_draft: Optional[Dict[str, Any]] = None
    phase5_localized_insight: Optional[str] = (
        None  # Prose paragraph after Phase 5 Renderer
    )


class DebugPipelineResponse(BaseModel):
    """Response model for pipeline debugging, returning all intermediate states for all domains"""

    status: str
    phase1_extracted_signals: Optional[Dict[str, Any]] = None
    phase1b_user_narrative: Optional[str] = None
    phase2_active_cases: Optional[List[Dict[str, Any]]] = None
    domains: Optional[Dict[str, DomainDebugResult]] = None
    error: Optional[str] = None
