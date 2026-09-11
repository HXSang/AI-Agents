"""Keep-field maps for extract → processor projection.

After each API fetch (and again in the extract normalize step), payloads are
filtered to only the keys listed for that API path key.

Example::

    FIELD_KEEP_MAP = {
        "balance": [
            {"key": "date"},
            {"key": "timezone"},
            {"key": "healthScore"},
        ],
        "calendar/event": [
            {"key": "summary"},
            {"key": "completed"},
            {"key": "startTime"},
            {"key": "endTime"},
            {"key": "eventType"},
            {"key": "category"},
            {"key": "location"},
            {"key": "participants"},
            {"key": "allDay"},
        ],
        "health/summaries": [
            {"key": "ENERGY.type"},
            {"key": "ENERGY.dateRange"},
            # …
        ],
    }
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from insights.health.canonical_field_mapping import CanonicalField

# ---------------------------------------------------------------------------
# Shared block fields for GET /api/health/summaries metric buckets
# ---------------------------------------------------------------------------
_HEALTH_SUMMARY_BLOCK_FIELDS: Tuple[str, ...] = (
    "dateRange",
    "daysWithData",
    "summaryData",
    "data",
)

_HEALTH_SUMMARY_METRICS: Tuple[str, ...] = ("ENERGY", "HR", "SLEEP", "STEPS")

_HEALTH_SUMMARY_DATA_FIELDS: Dict[str, Tuple[str, ...]] = {
    "ENERGY": (
        "date",
        "energyBurn",
    ),
    "HR": (
        "date",
        "latestHR",
        "restingHeartRate",
        "sleepHeartRate",
        "workoutHeartRate",
        "walkingHeartRateAverage",
    ),
    "SLEEP": (
        "date",
        "rem",
        "core",
        "deep",
        "total",
        "sleepScore",
        "lastWakeTime",
        "firstSleepTime",
    ),
    "STEPS": (
        "date",
        "steps",
        "distance",
    ),
}

_PROFILE_GOAL_FIELDS: Tuple[str, ...] = (
    "category_code",
    "action_code",
    "goal_name",
    "target_value",
    "unit",
)

_PROFILE_GOAL_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "category_code": ("category_code", "user_goal_category", "goal_category"),
    "action_code": ("action_code", "user_goal_action", "goal_action"),
    "goal_name": ("goal_name", "user_goal_name"),
    "target_value": ("target_value", "user_goal_target_value", "goal_target_value"),
    "unit": ("unit", "user_goal_unit", "goal_unit"),
}

_REMINDER_RECURRENCE_FIELDS: Tuple[str, ...] = (
    "frequency",
    "interval",
)

def _health_summaries_keep_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for metric in _HEALTH_SUMMARY_METRICS:
        for field in _HEALTH_SUMMARY_BLOCK_FIELDS:
            entries.append({"key": f"{metric}.{field}"})
    return entries

def _clean_profile_goals(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    clean_goals: List[Dict[str, Any]] = []

    for goal in value:
        goal_row = _coerce_row(goal)

        if not goal_row:
            continue

        clean_goal: Dict[str, Any] = {}
        for out_key in _PROFILE_GOAL_FIELDS:
            aliases = _PROFILE_GOAL_FIELD_ALIASES.get(out_key, (out_key,))
            for src in aliases:
                if src in goal_row and goal_row[src] is not None:
                    clean_goal[out_key] = goal_row[src]
                    break

        if clean_goal:
            clean_goals.append(clean_goal)

    return clean_goals


def _clean_profile_health_conditions(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []

    out: List[str] = []
    for item in value:
        if isinstance(item, str):
            if item:
                out.append(item)
            continue
        row = _coerce_row(item)
        if not row:
            continue
        label = (
            row.get("condition_name")
            or row.get("condition_code")
            or row.get("name")
        )
        if isinstance(label, str) and label:
            out.append(label)
    return out

# ---------------------------------------------------------------------------
# Per-API keep maps (api path key → list of {key} entries)
# GET /api/balance            → "balance"
# GET /api/calendar/events    → "calendar/event"
# GET /api/health/summaries   → "health/summaries"
# ---------------------------------------------------------------------------
FIELD_KEEP_MAP: Dict[str, List[Dict[str, Any]]] = {

    "daily-user-snapshots": [
        {"key": "snapshot_date"},
        {"key": "h_steps"},
        {"key": "h_active_minutes"},
        {"key": "h_calories_burned"},
        {"key": "h_exercise_sessions"},
        {"key": "h_total_workout_min"},
        {"key": "h_avg_heart_rate"},
        {"key": "h_resting_heart_rate"},
        {"key": "h_sleep_hours"},
        {"key": "h_sleep_quality"},
        {"key": "h_water_liters"},
        {"key": "h_calories_consumed"},
        {"key": "h_weight_kg"},
        {"key": "h_health_score"},
        {"key": "h_sleep_score"},
        {"key": "h_steps_goal_pct"},
        {"key": "h_activity_level"},
        {"key": "h_active_conditions"},
        {"key": "p_meeting_minutes"},
        {"key": "p_longest_meeting_min"},
        {"key": "p_avg_meeting_min"},
        {"key": "p_work_span_minutes"},
        {"key": "p_unique_collaborators"},
        {"key": "p_short_events_count"},
        {"key": "p_events_after_9pm"},
        {"key": "p_task_completion_rate"},
        {"key": "p_focus_blocks_60min"},
        {"key": "wellness_score"},
        {"key": "h_bedtime"},
        {"key": "h_wake_time"},
    ],
    "balance": [
        {"key": "date"},
        {"key": "healthScore"},
    ],
    "calendar/event": [
        {"key": "summary"},
        {"key": "completed"},
        {"key": "startTime"},
        {"key": "endTime"},
        {"key": "eventType"},
        {"key": "category"},
        {"key": "location"},
        {"key": "participants"},
        {"key": "allDay"},
    ],
    "calendar/reminders": [
        {"key": "title"},
        {"key": "dueDate"},
        {"key": "completed"},
        {"key": "recurrence"},
    ],
    "moods/latest": [
        {"key": "mood"},
        {"key": "moodDate"},
        {"key": "moodScore"},
    ],
    "moods": [
        {"key": "mood"},
        {"key": "moodDate"},
        {"key": "moodScore"},
    ],
    "productivity/summaries": [
        {"key": "date"},
        {"key": "meetingPercent"},
        {"key": "meetingCount"},
        {"key": "completedMeetingSoFar"},
        {"key": "reminderCount"},
        {"key": "completedReminderSoFar"},
        {"key": "commitmentsDue"},
        {"key": "commitmentsCompleted"},
    ],
    "productivity/summary": [
        {"key": "date"},
        {"key": "meetingPercent"},
        {"key": "meetingCount"},
        {"key": "completedMeetingSoFar"},
        {"key": "reminderCount"},
        {"key": "completedReminderSoFar"},
        {"key": "commitmentsDue"},
        {"key": "commitmentsCompleted"},
    ],
    "calendar/work-hours": [
        {"key": "date"},
        {"key": "totalHours"},
        {"key": "eventCount"},
    ],
    "balance_score": [
        {"key": "date"},
        {"key": "healthScore"},
        {"key": "productivityScore"},
        {"key": "financeScore"},
        {"key": "balanceScore"},
        {"key": "timezone"},
    ],
    "health/summaries": _health_summaries_keep_entries(),
    "onboarding/users/profiles": [
        {"key": "language"},
        {"key": "height_cm"},
        {"key": "weight_kg"},
        {"key": "date_of_birth"},
        {"key": "job_title"},
        {"key": "employment_type"},
        {"key": "work_status"},
        {"key": "active_hours_start_time"},
        {"key": "active_hours_end_time"},
        {"key": "bedtime_start"},
        {"key": "bedtime_end"},
        {"key": "primary_goal"},
        {"key": "current_health_conditions"},
        {"key": "age"},
        {"key": "gender"},
        {"key": "bmi"},
        {"key": "industry"},
        {"key": "goals"},
        {"key": "timezone"},
    ],
}

# Optional source aliases when upstream uses snake_case / alternate names.
# Flat maps: out_key → source candidates.
# Nested maps (health/summaries): "METRIC.field" → source candidates on the block.
FIELD_KEY_ALIASES: Dict[str, Dict[str, Sequence[str]]] = {
    "daily-user-snapshots": {
        "snapshot_date": ("snapshot_date", "snapshotDate"),
        "h_steps": ("h_steps", "hSteps", "h_step_count"),
        "h_active_minutes": ("h_active_minutes", "hActiveMinutes"),
        "h_calories_burned": ("h_calories_burned", "hCaloriesBurned"),
        "h_exercise_sessions": ("h_exercise_sessions", "hExerciseSessions"),
        "h_total_workout_min": ("h_total_workout_min", "hTotalWorkoutMin", "h_total_workout_minutes"),
        "h_avg_heart_rate": ("h_avg_heart_rate", "hAvgHeartRate", "h_average_heart_rate"),
        "h_resting_heart_rate": ("h_resting_heart_rate", "hRestingHeartRate"),
        "h_sleep_hours": ("h_sleep_hours", "hSleepHours"),
        "h_sleep_quality": ("h_sleep_quality", "hSleepQuality", "h_sleep_quality_score"),
        "h_water_liters": ("h_water_liters", "hWaterLiters", "h_water_intake_liters"),
        "h_calories_consumed": ("h_calories_consumed", "hCaloriesConsumed"),
        "h_weight_kg": ("h_weight_kg", "hWeightKg"),
        "h_health_score": ("h_health_score", "hHealthScore"),
        "h_sleep_score": ("h_sleep_score", "hSleepScore"),
        "h_steps_goal_pct": ("h_steps_goal_pct", "hStepsGoalPct", "h_step_goal_completion_pct"),
        "h_activity_level": ("h_activity_level", "hActivityLevel"),
        "h_active_conditions": ("h_active_conditions", "hActiveConditions", "h_health_conditions"),
        "p_meeting_minutes": ( "p_meeting_minutes", "pMeetingMinutes", "p_total_meeting_minutes"),
        "p_longest_meeting_min": ( "p_longest_meeting_min", "pLongestMeetingMin", "p_longest_meeting_minutes"),
        "p_avg_meeting_min": ("p_avg_meeting_min", "pAvgMeetingMin"),
        "p_work_span_minutes": ("p_work_span_minutes", "pWorkSpanMinutes"),
        "p_unique_collaborators": ("p_unique_collaborators", "pUniqueCollaborators"),
        "p_short_events_count": ("p_short_events_count", "pShortEventsCount"),
        "p_events_after_9pm": ("p_events_after_9pm", "pEventsAfter9pm"),
        "p_task_completion_rate": ("p_task_completion_rate", "pTaskCompletionRate"),
        "p_focus_blocks_60min": ("p_focus_blocks_60min", "pFocusBlocks60min"),
        "wellness_score": ("wellness_score", "wellnessScore"),
        "h_bedtime": ("h_bedtime", "hBedtime", "bedtime"),
        "h_wake_time": ("h_wake_time", "hWakeTime", "wake_time"),
    },
    "balance": {
        "healthScore": ("healthScore", "health_score"),
        "date": ("date",),
    },
    "calendar/event": {
        "summary": ("summary", "calendar_event_title"),
        "completed": ("completed", "calendar_event_completion_status"),
        "startTime": ("startTime", "start_time", "calendar_event_start_time"),
        "endTime": ("endTime", "end_time", "calendar_event_end_time"),
        "eventType": ("eventType", "event_type", "calendar_event_type"),
        "category": ("category",),
        "location": ("location",),
        "participants": ("participants",),
        "allDay": ("allDay", "all_day"),
    },
    "health/summaries": {
        "HR": ("HR", "HEART_RATE", "RESTING_HEART_RATE"),
        "ENERGY": ("ENERGY", "ACTIVE_ENERGY"),
        "SLEEP": ("SLEEP",),
        "STEPS": ("STEPS",),
        "dateRange": ("dateRange", "date_range"),
        "daysWithData": ("daysWithData", "days_with_data"),
        "createdAt": ("createdAt", "created_at"),
        "updatedAt": ("updatedAt", "updated_at"),
        "summaryData": ("summaryData", "summary_data"),
        "sleepScore": ("sleepScore", "sleep_score"),
        # HR summaryData fields
        CanonicalField.AVERAGE_HEART_RATE_VARIABILITY_CURRENT: (CanonicalField.AVERAGE_HEART_RATE_VARIABILITY_CURRENT,),
        CanonicalField.MINIMUM_HEART_RATE_CURRENT_SUMMARY_PERIOD_BPM: (CanonicalField.MINIMUM_HEART_RATE_CURRENT_SUMMARY_PERIOD_BPM,),
        CanonicalField.MAXIMUM_HEART_RATE_CURRENT_SUMMARY_PERIOD_BPM: (CanonicalField.MAXIMUM_HEART_RATE_CURRENT_SUMMARY_PERIOD_BPM,),
    },
    "calendar/reminders": {
        "title": ("title", "Title", "reminder_title"),
        "dueDate": ("dueDate", "due_date", "due_at", "dueAt", "reminder_due_time"),
        "completed": ("completed", "Completed", "isCompleted", "reminder_completion_status"),
        "recurrence": ("recurrence", "Recurrence"),
    },
    "moods/latest": {
        "mood": ("mood", "Mood"),
        "moodDate": ("moodDate", "mood_date"),
        "moodScore": ("moodScore", "mood_score"),
    },
    "moods": {
        "mood": ("mood", "Mood"),
        "moodDate": ("moodDate", "mood_date"),
        "moodScore": ("moodScore", "mood_score"),
    },
    "productivity/summaries": {
        "date": ("date",),
        "meetingPercent": ("meetingPercent", "meeting_percent", CanonicalField.MEETING_TIME_PERCENTAGE_TODAY),
        "meetingCount": ("meetingCount", "meeting_count", CanonicalField.MEETING_COUNT_TODAY),
        "completedMeetingSoFar": ("completedMeetingSoFar", "completed_meeting_so_far", CanonicalField.COMPLETED_MEETING_COUNT_TODAY),
        "reminderCount": ("reminderCount", "reminder_count", CanonicalField.REMINDER_COUNT_TODAY),
        "completedReminderSoFar": ("completedReminderSoFar", "completed_reminder_so_far", CanonicalField.COMPLETED_REMINDER_COUNT_TODAY),
        "commitmentsDue": ("commitmentsDue", "commitments_due"),
        "commitmentsCompleted": ("commitmentsCompleted", "commitments_completed"),
    },
    "productivity/summary": {
        "date": ("date",),
        "meetingPercent": ("meetingPercent", "meeting_percent", CanonicalField.MEETING_TIME_PERCENTAGE_TODAY),
        "meetingCount": ("meetingCount", "meeting_count", CanonicalField.MEETING_COUNT_TODAY),
        "completedMeetingSoFar": ("completedMeetingSoFar", "completed_meeting_so_far", CanonicalField.COMPLETED_MEETING_COUNT_TODAY),
        "reminderCount": ("reminderCount", "reminder_count", CanonicalField.REMINDER_COUNT_TODAY),
        "completedReminderSoFar": ("completedReminderSoFar", "completed_reminder_so_far", CanonicalField.COMPLETED_REMINDER_COUNT_TODAY),
        "commitmentsDue": ("commitmentsDue", "commitments_due"),
        "commitmentsCompleted": ("commitmentsCompleted", "commitments_completed"),
    },
    "balance_score": {
        "date": ("date",),
        "healthScore": ("healthScore", "health_score"),
        "productivityScore": ("productivityScore", "productivity_score"),
        "financeScore": ("financeScore", "finance_score"),
        "balanceScore": ("balanceScore", "balance_score"),
        "timezone": ("timezone",),
    },
    "calendar/work-hours": {
        "date": ("date", "work_hours_date"),
        "totalHours": ("totalHours", "total_hours", "scheduled_work_hours"),
        "eventCount": ("eventCount", "event_count", "scheduled_work_event_count"),
    },
    "onboarding/users/profiles": {
        "language": ("language", "user_language"),
        "height_cm": ("height_cm", "heightCm", "user_height_centimeters"),
        "weight_kg": ("weight_kg", "weightKg", "user_weight_kilograms"),
        "date_of_birth": ("date_of_birth", "dateOfBirth", "user_date_of_birth"),
        "job_title": ("job_title", "jobTitle", "user_job_title"),
        "employment_type": ("employment_type", "employmentType", "user_employment_type"),
        "work_status": ("work_status", "workStatus", "user_work_status"),
        "active_hours_start_time": ("active_hours_start_time", "activeHoursStartTime", "user_active_hours_start_time"),
        "active_hours_end_time": ("active_hours_end_time","activeHoursEndTime", "user_active_hours_end_time"),
        "bedtime_start": ("bedtime_start", "bedtimeStart", "user_bedtime_window_start"),
        "bedtime_end": ("bedtime_end", "bedtimeEnd", "user_bedtime_window_end"),
        "primary_goal": ("primary_goal", "primaryGoal", "user_primary_goal"),
        "current_health_conditions": ("current_health_conditions","currentHealthConditions", "user_current_health_conditions"),
        "age": ("age", "user_age_years"),
        "gender": ("gender", "user_gender"),
        "bmi": ("bmi", "user_body_mass_index"),
        "industry": ("industry", "user_industry"),
        "goals": ("user_goals", "goals", "user_health_goals"),
        "timezone": ("timezone", "user_timezone"),
    },
}


def _coerce_row(item: Any) -> Optional[Dict[str, Any]]:
    if item is None:
        return None
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        try:
            dumped = item.model_dump(by_alias=True)
            return dumped if isinstance(dumped, dict) else None
        except Exception:
            pass
    if hasattr(item, "__dict__"):
        try:
            return dict(vars(item))
        except Exception:
            return None
    return None


def _keep_keys_for(api_key: str, keep_map: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[str]:
    entries = keep_map.get(api_key) or []
    keys: List[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        key = entry.get("key")
        if isinstance(key, str) and key:
            keys.append(key)
    return keys


def _is_nested_keep(keep_keys: Sequence[str]) -> bool:
    return any("." in k for k in keep_keys)


def _group_nested_keep_keys(keep_keys: Sequence[str]) -> Dict[str, List[str]]:
    """Split ``METRIC.field`` keys into ``{METRIC: [field, …]}``."""
    grouped: Dict[str, List[str]] = {}
    for key in keep_keys:
        if "." not in key:
            continue
        metric, field = key.split(".", 1)
        if not metric or not field:
            continue
        grouped.setdefault(metric, []).append(field)
    return grouped


def _pick_field(
    row: Mapping[str, Any],
    out_key: str,
    *,
    alias_table: Mapping[str, Sequence[str]],
) -> Tuple[bool, Any]:
    candidates = alias_table.get(out_key) or (out_key,)
    for src in candidates:
        # Keep falsy-but-present values (e.g. completed=False, daysWithData=0)
        if src in row and row[src] is not None:
            return True, row[src]
    return False, None


def filter_row_by_keep_map(
    item: Any,
    api_key: str,
    *,
    keep_map: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    aliases: Optional[Mapping[str, Mapping[str, Sequence[str]]]] = None,
) -> Optional[Dict[str, Any]]:
    """Project one row to keys listed for `api_key` in the keep map."""
    map_ = keep_map if keep_map is not None else FIELD_KEEP_MAP
    keep_keys = _keep_keys_for(api_key, map_)
    if not keep_keys:
        row = _coerce_row(item)
        return dict(row) if row else None

    row = _coerce_row(item)
    if not row:
        return None

    alias_table = ((aliases if aliases is not None else FIELD_KEY_ALIASES).get(api_key) or {})
    out: Dict[str, Any] = {}
    for out_key in keep_keys:
        found, value = _pick_field(row, out_key, alias_table=alias_table)

        if not found:
            continue
        if api_key == "onboarding/users/profiles" and out_key == "goals":
            value = _clean_profile_goals(value)

        if (api_key == "onboarding/users/profiles" and out_key == "current_health_conditions"):
            value = _clean_profile_health_conditions(value)
            
        if (api_key == "calendar/reminders" and out_key == "recurrence" and isinstance(value, Mapping)):
            value = {
                key: value[key]
                for key in _REMINDER_RECURRENCE_FIELDS
                if key in value and value[key] is not None
            }
        out[out_key] = value

    return out if out else None

def _filter_nested_summaries(
    payload: Any,
    api_key: str,
    *,
    keep_map: Mapping[str, Sequence[Mapping[str, Any]]],
    aliases: Optional[Mapping[str, Mapping[str, Sequence[str]]]] = None,
) -> Any:
    root = _coerce_row(payload)
    if not root:
        return payload if not isinstance(payload, dict) else {}

    keep_keys = _keep_keys_for(api_key, keep_map)
    grouped = _group_nested_keep_keys(keep_keys)
    if not grouped:
        return payload

    alias_table = (aliases if aliases is not None else FIELD_KEY_ALIASES).get(api_key) or {}
    out: Dict[str, Any] = {}

    for metric, fields in grouped.items():
        metric_candidates = alias_table.get(metric) or (metric,)
        block = None

        for src in metric_candidates:
            if src in root and root[src] is not None:
                block = root[src]
                break
        if block is None:
            continue

        block_row = _coerce_row(block)
        if not block_row:
            continue

        projected: Dict[str, Any] = {}
        for field in fields:
            found, value = _pick_field(block_row, field, alias_table=alias_table)

            if not found:
                continue

            if field == "data" and isinstance(value, list):
                allowed_fields = _HEALTH_SUMMARY_DATA_FIELDS.get(
                    metric, ()
                )

                clean_data = []

                for item in value:
                    item_row = _coerce_row(item)

                    if not item_row:
                        continue

                    clean_item = {
                        key: item_row[key]
                        for key in allowed_fields
                        if key in item_row
                        and item_row[key] is not None
                    }

                    if clean_item:
                        clean_data.append(clean_item)

                projected[field] = clean_data
            elif field == "summaryData" and isinstance(value, Mapping):
                # Extract nested fields from summaryData using alias table
                summary_data_row = _coerce_row(value)
                if summary_data_row:
                    # Get all summaryData nested field aliases from alias_table
                    nested_fields: Dict[str, Any] = {}
                    for alias_key, candidates in alias_table.items():
                        if "." in alias_key:
                            # This is a summaryData nested field: "average_heart_rate_variability_current"
                            out_key = alias_key.split(".", 1)[1] if "." in alias_key else alias_key
                            for candidate in candidates:
                                if candidate in summary_data_row and summary_data_row[candidate] is not None:
                                    nested_fields[alias_key] = summary_data_row[candidate]
                                    break
                    projected["summaryData"] = nested_fields if nested_fields else value
                else:
                    projected[field] = value
            else:
                projected[field] = value
        if projected:
            out[metric] = projected

    return out


def filter_by_keep_map(
    payload: Any,
    api_key: str,
    *,
    keep_map: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    aliases: Optional[Mapping[str, Mapping[str, Sequence[str]]]] = None,
) -> Any:
    """Filter a fetch payload using ``FIELD_KEEP_MAP[api_key]``.

    - flat map (balance, calendar/event): dict/list of rows → keep listed keys
    - nested map (health/summaries): ``METRIC.field`` keys → project each block
    - unknown api_key → return payload unchanged
    """
    map_ = keep_map if keep_map is not None else FIELD_KEEP_MAP
    if api_key not in map_:
        return payload

    keep_keys = _keep_keys_for(api_key, map_)
    if _is_nested_keep(keep_keys):
        return _filter_nested_summaries(
            payload, api_key, keep_map=map_, aliases=aliases
        )

    if isinstance(payload, list):
        out: List[Dict[str, Any]] = []
        for item in payload:
            projected = filter_row_by_keep_map(
                item, api_key, keep_map=map_, aliases=aliases
            )
            if projected is not None:
                out.append(projected)
        return out

    if isinstance(payload, dict) or hasattr(payload, "model_dump") or hasattr(payload, "__dict__"):
        return filter_row_by_keep_map(
            payload, api_key, keep_map=map_, aliases=aliases
        )

    return payload
