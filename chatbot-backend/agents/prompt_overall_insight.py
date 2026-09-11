"""Rule-based overall insight prompts (great_job, need_attention, opportunity)"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from agents.prompt import (
    build_preserve_entity_titles_prompt,
    build_reminders_awareness_prompt,
    get_language_name,
    important_language_prompt,
)
from agents.prompt_insight_reasoning import format_for_overall as _overall_block


def _is_sleep_from_last_night(
    first_sleep_time: str,
    current_time: str,
    timezone: Optional[str] = None,
) -> bool:
    """
    Check if first_sleep_time is from last night's sleep (from 18:00 yesterday onwards).

    Args:
        first_sleep_time: First sleep time in ISO format (YYYY-MM-DDTHH:MM:SS)
        current_time: Current time in ISO format
        timezone: Optional IANA timezone (default: UTC)

    Returns:
        True if first_sleep_time is from last night's sleep (from 18:00 yesterday onwards),
        False otherwise
    """
    try:
        tz = ZoneInfo(timezone)

        # Parse current_time
        if current_time.endswith("Z"):
            current_dt = datetime.fromisoformat(current_time.replace("Z", "+00:00"))
        else:
            current_dt = datetime.fromisoformat(current_time)
        if current_dt.tzinfo is None:
            current_dt = current_dt.replace(tzinfo=tz)

        # Parse first_sleep_time
        if first_sleep_time.endswith("Z"):
            first_sleep_dt = datetime.fromisoformat(
                first_sleep_time.replace("Z", "+00:00")
            )
        else:
            first_sleep_dt = datetime.fromisoformat(first_sleep_time)
        if first_sleep_dt.tzinfo is None:
            first_sleep_dt = first_sleep_dt.replace(tzinfo=tz)

        # Convert both to user timezone for comparison
        current_dt = current_dt.astimezone(tz)
        first_sleep_dt = first_sleep_dt.astimezone(tz)

        # Calculate "18:00 today" (evening threshold for today)
        evening_threshold_today = current_dt.replace(
            hour=18, minute=0, second=0, microsecond=0
        )

        # Check: first_sleep_time must be <= 18:00 today to be from last night's sleep
        # If first_sleep_time is after 18:00 today, it might be current sleep (user just went to bed)
        return first_sleep_dt <= evening_threshold_today
    except Exception as e:
        # If parsing fails, return False to be safe
        return False


def _build_conditional_line(
    label: str,
    value: Any,
    suffix: str = "",
    formatter: Optional[callable] = None,
) -> str:
    """
    Build a conditional line that only appears if value is not None/empty.

    Args:
        label: Label for the line (e.g., "Bedtime Streak")
        value: Value to check (None/empty means don't show line)
        suffix: Optional suffix (e.g., " days", " hours")
        formatter: Optional function to format the value

    Returns:
        Formatted line string or empty string if value is None/empty
    """
    if value is None:
        return ""

    # Handle empty strings, empty lists, empty dicts
    if isinstance(value, (str, list, dict)) and not value:
        return ""

    # Format value if formatter provided
    if formatter:
        formatted_value = formatter(value)
    else:
        formatted_value = value

    return f"- {label}: {formatted_value} {suffix}"


_OVERALL_OUTPUT_RULES = """
**OVERALL OUTPUT RULES (ALWAYS APPLY):**
- Respond with valid JSON only: exactly three keys `great_job`, `need_attention`, `opportunity`.
- Use empty string `""` for any category that does not apply. Do NOT invent filler to fill keys.
- Only include content supported by data in this message (health, calendar, mood, productivity context).
- One fact or issue → at most ONE category. Do not praise and warn about the same issue.
- At most ONE bullet about sleep across the whole response.
- Align with productivity context when provided; do not copy the same sentence into multiple categories.
- Be concise: 1-2 sentences per non-empty category.

|HARD DATA-FAITHFULNESS RULES (NO HALLUCINATION):
|- Numbers that are NOT in the provided context MUST NOT be invented. If a duration, BMI, HR, steps, sleep minutes, meeting count, or any other number is not explicitly present in the input you were given, you MUST NOT cite it. NEVER fabricate values like "187 minutes", "BMI 27.8", "150 bpm" etc. unless those exact numbers appear in your input.
|- Do NOT infer "X minutes of continuous sitting/work" unless the input explicitly provides a continuous-sitting or continuous-work metric. Stating "you have been sitting for X minutes" without that data is a fabrication.
|- BMI is a body-composition ratio. It does NOT directly indicate recent activity level. A high BMI MUST NOT be phrased as evidence of "low activity today" or "sedentary state" unless the input also gives a current steps/activity metric to back it up. If you mention BMI, describe it ONLY as a body-composition data point, never as a proxy for today's behavior.

|STRICT TIME & FUTURE-ONLY RULES:
|- Suggestions in `need_attention` and `opportunity` MUST be framed in the FUTURE relative to the user's current time. NEVER recommend an action at a specific past or already-elapsed time slot (e.g. "from 14:30 to 14:35" when current_time is 17:20 is INVALID).
|- When mentioning a specific clock time for an action, it MUST be AFTER the current time given in your input. If current_time is 17:20, the earliest valid start time for an action you suggest is later than 17:20.
|- If current_time is not provided or you are unsure, use qualitative language only ("a short break", "the next free window") and DO NOT invent specific clock times.
|- Opportunity MUST be something the user can actually do in the present or near future. Past actions belong in `great_job` only (with past tense), never in `opportunity` or `need_attention`.
"""

_MULTI_DOMAIN_CATEGORY_RULES = """
**MULTI-DOMAIN CATEGORY RULES:**
- Each non-empty category should reflect a DIFFERENT domain when possible (steps, sleep, calendar, mood, productivity/finance).
- **great_job**: Past achievements already done today (e.g. steps_today >= steps_goal, streak, mood, productivity wins). Past tense only.
- **need_attention**: The single most important concern actionable or relevant NOW (one domain).
- **opportunity**: A constructive next step in a DIFFERENT domain than need_attention, only if feasible now (see time actionability).
- **Sleep debt** (sleep_lastnight < 85% of sleep_goal, or sleep_quality is low/very_low):
  - Put sleep debt in need_attention.
  - Do NOT praise sleep timing, sleep phase, or "biological clock" in great_job.
  - Do not repeat the same sleep advice in opportunity.
- If steps_today < steps_goal: do NOT put step deficit in great_job. Only mention step gap in need_attention/opportunity when catch-up is allowed for this time window (see group rules).
"""

_TIME_ACTIONABILITY_BY_GROUP: Dict[str, str] = {
    "bedtime_window": """
**TIME ACTIONABILITY (bedtime_window — user should be resting now):**
- Allowed now: rest, sleep, calm wind-down (dim lights, avoid screens).
- FORBIDDEN in ALL categories: walking, step catch-up, workouts, meal prep, planning, inbox, "prepare tonight", "prep for tomorrow tonight".
- Do NOT mention steps behind goal in any category (sounds like "go walk now").
- opportunity: usually `""`. If used, one short calm rest line only, not duplicating need_attention.
- great_job: only non-sleep wins already achieved today (e.g. steps goal met earlier).""",
    "wind_down_window": """
**TIME ACTIONABILITY (wind_down_window):**
- Allowed now: calm wind-down, rest preparation. No work, planning, or stimulating tasks.
- FORBIDDEN: step catch-up, "go walk" for steps goal, evening work prep, inbox.
- Do NOT mention steps behind goal in any category.
- opportunity: calm actions only (lights, screens, relax); no "10-min tomorrow prep".""",
    "evening_flexible": """
**TIME ACTIONABILITY (evening_flexible):**
- Light evening recharge OK. No intense work or late step catch-up pressure.
- Do NOT nudge step deficit if user is close to wind-down/bedtime (use time_to_bedtime if short).""",
    "evening_after_work": """
**TIME ACTIONABILITY (evening_after_work):**
- Recovery and light evening activities OK. No step catch-up if time_to_bedtime is short.
- Calendar may inform need_attention (schedule affecting rest); opportunity = calm evening, not tasks for tomorrow tonight.""",
    "active_window": """
**TIME ACTIONABILITY (active_window):**
- May suggest actions in current free slot (free_slot_length). No past tasks. Respect calendar conflicts.
- Step catch-up only if steps_today < steps_goal AND slot is sufficient; gentle wording.""",
    "morning_window": """
**TIME ACTIONABILITY (morning_window):**
- Planning and day setup OK. Past sleep informs tone; focus forward from current time.""",
    "health_goal": """
**TIME ACTIONABILITY (health_goal):**
- Use free_slot_length and time_to_bedtime: step catch-up only before wind-down with enough slot.""",
    "contextual_suggestion": """
**TIME ACTIONABILITY (contextual_suggestion):**
- Use in_active_window, free_slot_length, time_to_bedtime from context. No actions in the past.""",
}

_DEFAULT_TIME_ACTIONABILITY = """
**TIME ACTIONABILITY:**
- Match suggestions to the current time window implied by this insight group.
- need_attention must not sound like "do X now" when the user should be sleeping or winding down.
- If no feasible opportunity exists, use `""` for opportunity."""


def _get_shared_overall_rules_block(group: str) -> str:
    """Shared rules appended in get_overall_insight_prompt (no new API params)."""
    time_rules = _TIME_ACTIONABILITY_BY_GROUP.get(group, _DEFAULT_TIME_ACTIONABILITY)
    return (
        _OVERALL_OUTPUT_RULES
        + _MULTI_DOMAIN_CATEGORY_RULES
        + time_rules
        + f"\n**Detected insight group:** `{group}`\n"
    )


def _get_bedtime_window_overall_insight_no_data_prompt(
    calendar_data: str = "",
    current_time: str = "",
    timezone: Optional[str] = None,
    bedtime_start: Optional[str] = None,
    bedtime_end: Optional[str] = None,
    language_name: str = "English",
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for bedtime window overall insight when sleep timing data is not available.

    This handles the case when user is not wearing a device or hasn't slept yet.
    """
    system_prompt = f"""You are a bedtime window analyzer. The user is in their bedtime window, but sleep timing data is not available (they may not be wearing their device or haven't slept yet).

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Output Format:**
You MUST respond with a valid JSON object containing exactly three fields:
{{
  "great_job": "Summary of achievements and positive patterns",
  "need_attention": "Summary of areas needing attention and actionable guidance",
  "opportunity": "Summary of opportunities and suggestions for optimal actions"
}}

**User Schedule:**
- Bedtime Window: {bedtime_start or 'Not specified'} to {bedtime_end or 'Not specified'}

**Context:**
- Sleep timing data (last wake time, first sleep time) is not available
- This could mean: user is not wearing their wearable device, device is not synced, or user hasn't slept yet
- DO NOT make assumptions about sleep patterns or timing
- However, you CAN analyze the current time's position within the bedtime window

**Categories (no sleep timing data — use calendar + shared rules):**
- **great_job**: A win from another domain if present in calendar/productivity context (not sleep-phase praise). Often `""`.
- **need_attention**: One rest-focused note if user should switch off (bedtime window). Often `""` if nothing concrete.
- **opportunity**: Usually `""` at bedtime. At most one calm rest line if need_attention is empty.
- Do NOT assign a different sleep phase to each category.

{_build_productivity_context_prompt(productivity_context)}

IMPORTANT TIME RULES:
- DO NOT mention exact clock times (e.g. 8:58 AM, 9:00, current time)
- DO NOT repeat or paraphrase the provided current_time value
- DO NOT suggest any work, planning, or stimulating activities
- DO NOT suggest screens, inbox, or admin tasks
- DO NOT make assumptions about sleep patterns or timing when data is unavailable
- Use ONLY relative time expressions such as:
  "at the start of your sleep window", "in the middle of your rest period",
  "as you approach wake time", "during your deep sleep phase",
  "near the end of your sleep window"
- Focus on sleep phase-appropriate activities:
  - Early: dim lights, avoid screens, gentle stretching, meditation, reading (non-stimulating)
  - Middle: maintain sleep environment, avoid disruptions, deep rest
  - Late: gentle awakening preparation, light morning thoughts, maintaining rest

IMPORTANT:
- relative to calendar events (if have):
    - Pay attention to event titles, descriptions, and participants to better understand the context and intent behind each calendar event.
    - Use this contextual understanding to produce more relevant, coherent, and human-like insights.
- Three JSON keys are required; use empty string for categories that do not apply (do not invent filler)
- Be concise - each category 1-2 sentences maximum
- Use natural, conversational language
- Acknowledge data limitation without being technical
- Focus on sleep phase analysis and appropriate suggestions
- Respond with valid JSON only
{important_language_prompt(language_name)}
"""

    user_prompt = f"""**Bedtime Window Analysis with Sleep Phase Detection (No Sleep Timing Data)**

**Current Context:**
- Current Time: {current_time}
- Bedtime Window: {bedtime_start} to {bedtime_end}

**Note:** Sleep timing data (last wake time, first sleep time) is not available.
This could mean:
- User is not wearing their wearable device
- Device is not synced
- User hasn't slept yet

**Calendar Events:**
{calendar_data}

Use calendar and productivity context for great_job (non-sleep wins). Rest-focused need_attention only if relevant. opportunity usually empty.

{important_language_prompt(language_name)}

Return JSON (use "" for unused categories). Example partial: {{"great_job":"","need_attention":"...","opportunity":""}}"""

    return system_prompt, user_prompt


def _build_productivity_context_prompt(
    productivity_context: Optional[str] = "",
) -> str:
    """
    Build prompt section about productivity context to ensure consistency.

    Args:
        productivity_context: Productivity insight context (can be JSON string or dict string)

    Returns:
        Prompt section string, or empty string if no context
    """
    event_semantic_policy = (
        "\n**EVENT SEMANTIC POLICY (ALWAYS APPLY):**\n"
        "- Use event semantics from title + description + participants + location.\n"
        "- If context is social/personal (e.g., hang out, friend, family), describe it as social/personal event.\n"
        "- Use neutral term 'event' when intent is unclear.\n"
    )
    reminders_awareness = build_reminders_awareness_prompt()
    title_policy = build_preserve_entity_titles_prompt()
    if not productivity_context:
        return event_semantic_policy + title_policy + reminders_awareness

    # Convert to string if not already
    context_str = str(productivity_context) if productivity_context else ""

    # Format the context for prompt
    context_section = "\n\n**Productivity Insight Context (MUST ALIGN WITH):**\n"
    context_section += (
        "The following productivity insights have already been provided to the user. "
    )
    context_section += "Your overall insights (great_job, need_attention, opportunity) Must be consistent and aligned, and should reinforce and complement these insights.\n\n"
    context_section += f"{context_str}\n"

    context_section += "\n**CRITICAL CONSISTENCY REQUIREMENTS:**\n"
    context_section += "- DO NOT provide conflicting information with the productivity insights above\n"
    context_section += "- DO NOT contradict or create gaps between productivity insights and overall insights\n"
    context_section += "- DO NOT repeat the same information, but ensure your insights complement and align with productivity insights\n"
    context_section += "- If productivity insights mention specific facts (e.g., 'back-to-back events'), your overall insights should acknowledge or build upon these facts, not contradict them\n"
    context_section += "- Ensure great_job, need_attention, and opportunity are consistent with the productivity context\n"
    context_section += "- If productivity insights focus on a specific area, ensure your overall insights don't create confusion by addressing the same area differently\n"

    return context_section + event_semantic_policy + title_policy + reminders_awareness


def get_safety_risk_overall_insight_prompt(
    latest_heart_rate: Optional[int] = None,
    resting_heart_rate: Optional[int] = None,
    baseline_resting_hr: Optional[float] = None,
    sleep_lastnight: Optional[float] = None,
    sleep_goal: Optional[float] = None,
    sleep_quality: Optional[str] = None,
    sleep_last3nights: Optional[List[Optional[float]]] = None,
    calendar_density: Optional[float] = None,
    back_to_back_count: Optional[int] = None,
    meeting_minutes: Optional[int] = None,
    calendar_events_next3h: Optional[List[Dict[str, Any]]] = None,
    current_time: str = "",
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for safety risk overall insight (Group 1).

    Args:
        latest_heart_rate: Latest heart rate
        resting_heart_rate: Resting heart rate
        sleep_lastnight: Sleep hours last night
        sleep_goal: Daily sleep goal
        sleep_quality: Sleep quality level (very_high, high, ok, low, very_low)
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    system_prompt = f"""You are a safety and health risk detector. Analyze the user's health data and provide insights in exactly three categories: great_job, need_attention, and opportunity.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Output Format:**
You MUST respond with a valid JSON object containing exactly three fields:
{{
  "great_job": "Summary of achievements and positive patterns (can be empty string if none)",
  "need_attention": "Summary of areas needing attention and actionable guidance (can be empty string if none)",
  "opportunity": "Summary of opportunities and suggestions for optimal actions (can be empty string if none)"
}}

**Safety Rules:**
1. **Dangerous HR**: If HR < 40 OR HR > 180 → Block all suggestions, show safety guidance
2. **High Stress**: If resting HR > baseline + 10 bpm → Block intense exercise (baseline: {baseline_resting_hr or "Not available"})
3. **Sleep Debt**: If sleep_lastnight < sleep_goal → Avoid late intense workouts
4. **Sleep Quality + Calendar Density**: If sleep quality is "low" or "very_low" AND calendar density high → Suggest lighter schedule
5. **Sleep Quality + Back-to-back**: If sleep quality is "low" or "very_low" AND back-to-back events → Suggest recovery breaks
6. **Medical Conflicts**: If medical conditions conflict with suggestions → Offer safe alternatives

**Categories:**
- **great_job**: (usually empty for safety risks)
- **need_attention**: Safety/risk summary with actionable safety guidance
- **opportunity**: (usually empty for safety risks)

IMPORTANT TIME & SAFETY RULES:
- DO NOT mention or restate the exact current time
- DO NOT imply real-time monitoring (e.g. "right now", "currently", "at this moment")
- DO NOT present guidance as urgent commands unless the condition is explicitly dangerous
- When referencing time-based rules (evening, caffeine cutoff), describe them qualitatively (e.g. "later in the day", "toward the evening")
- Safety guidance should be precautionary and conditional, not definitive diagnoses
- Avoid medical certainty language; use supportive and cautious phrasing

IMPORTANT:
- Three JSON keys are required; use empty string for categories that do not apply (do not invent filler)
- If a category doesn't apply, use an empty string ""
- Be concise and actionable
- Prioritize safety over productivity
- Use natural, supportive language
- Respond with valid JSON only
{important_language_prompt(language_name)}"""

    # Format sleep_last3nights
    sleep_last3nights_str = "Not available"
    if sleep_last3nights:
        sleep_hours_list = []
        for i, h in enumerate(sleep_last3nights):
            if h is not None:
                sleep_hours_list.append(f"Night {i+1}: {h:.1f}h")
            else:
                sleep_hours_list.append(f"Night {i+1}: No data")
        sleep_last3nights_str = " | ".join(sleep_hours_list)

    # Format calendar_events_next3h
    events_next3h_str = "No events"
    if calendar_events_next3h:
        events_list = []
        for event in calendar_events_next3h[:5]:  # Limit to 5 events
            summary = event.get("summary", "Untitled")
            start_time = event.get("startTime", "")
            description = str(event.get("description", "") or "").strip()
            desc_hint = f" | Description: {description[:120]}" if description else ""
            events_list.append(f"- {summary} ({start_time}){desc_hint}")
        events_next3h_str = "\n".join(events_list) if events_list else "No events"

    # Build conditional health data lines
    health_lines = []
    if latest_heart_rate is not None:
        health_lines.append(f"- Latest Heart Rate: {latest_heart_rate}")
    if resting_heart_rate is not None:
        health_lines.append(f"- Resting Heart Rate: {resting_heart_rate}")
    if baseline_resting_hr is not None:
        health_lines.append(f"- Baseline Resting HR: {baseline_resting_hr}")
    if sleep_lastnight is not None:
        health_lines.append(f"- Sleep Last Night: {sleep_lastnight} hours")
    if sleep_goal is not None:
        health_lines.append(f"- Sleep Goal: {sleep_goal} hours")
    if sleep_quality:
        health_lines.append(
            f"- Sleep Quality: {sleep_quality} (very_high, high, ok, low, very_low)"
        )
    if sleep_last3nights_str:
        health_lines.append(f"- Sleep Last 3 Nights: {sleep_last3nights_str}")
    health_section = "\n".join(health_lines) if health_lines else ""

    # Build conditional calendar context lines
    calendar_lines = []
    if calendar_density is not None:
        calendar_lines.append(
            f"- Calendar Density: {calendar_density} event blocks in active hours"
        )
    if back_to_back_count is not None:
        calendar_lines.append(
            f"- Back-to-back Count: {back_to_back_count} consecutive events"
        )
    if meeting_minutes is not None:
        calendar_lines.append(
            f"- Event Minutes (Last 3 Hours): {meeting_minutes} minutes"
        )
    if calendar_events_next3h:
        calendar_lines.append(
            f"- Events Next 3 Hours: {len(calendar_events_next3h)} events"
        )
        if events_next3h_str and events_next3h_str != "No events":
            calendar_lines.append(events_next3h_str)
    calendar_section = "\n".join(calendar_lines) if calendar_lines else ""

    user_prompt = f"""**Safety & Risk Analysis**

**Health Data:**
{health_section if health_section else ""}

**Calendar Context:**
{calendar_section if calendar_section else ""}

**Safety Analysis Rules:**
1. If sleep quality is "low" or "very_low" AND calendar_density high → Suggest lighter schedule
2. If sleep quality is "low" or "very_low" AND back_to_back_count >= 3 → Suggest recovery breaks
3. If sleep_last3nights shows consistent sleep debt (any night < sleep_goal) → Prioritize recovery
4. If resting_heart_rate > baseline_resting_hr + 10 bpm → Block intense exercise
5. If calendar_events_next3h shows heavy load → Suggest lighter schedule

Analyze safety risks and provide guidance. Focus on:
- Dangerous heart rate readings (< 40 or > 180)
- High stress signals (resting HR > baseline + 10 bpm)
- Severe sleep debt (from sleep_last3nights)
- Sleep quality + calendar density interactions
- Sleep quality + back-to-back events
- Upcoming events load impact
- Medical condition conflicts

{important_language_prompt(language_name)}

Return JSON with three categories:
{{
  "great_job": "",
  "need_attention": "Safety/risk summary with actionable safety guidance",
  "opportunity": "Safety/risk opportunities"
}}"""

    return system_prompt, user_prompt


def get_time_window_overall_insight_prompt(
    calendar_data: str = "",
    time_to_bedtime: Optional[int] = None,
    time_to_next_event: Optional[int] = None,
    in_active_window: Optional[bool] = None,
    free_slot_length: Optional[int] = None,
    bedtime_start: Optional[str] = None,
    bedtime_end: Optional[str] = None,
    active_hours_start: Optional[str] = None,
    active_hours_end: Optional[str] = None,
    current_time: str = "",
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for time window overall insight (Group 2).

    Args:
        calendar_data: Formatted calendar events string
        time_to_bedtime: Minutes until bedtime
        time_to_next_event: Minutes until next event
        in_active_window: Whether in active hours window
        free_slot_length: Current free slot length in minutes
        bedtime_start: Bedtime window start
        bedtime_end: Bedtime window end
        active_hours_start: Active hours start
        active_hours_end: Active hours end
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    system_prompt = f"""You are a time context analyzer. Analyze the user's current time context and provide insights in exactly three categories: great_job, need_attention, and opportunity.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Output Format:**
You MUST respond with a valid JSON object containing exactly three fields:
{{
  "great_job": "Summary of achievements and positive patterns",
  "need_attention": "Summary of areas needing attention and actionable guidance",
  "opportunity": "Summary of opportunities and suggestions for optimal actions"
}}

**User Schedule:**
- Bedtime Window: {bedtime_start or 'Not specified'} to {bedtime_end or 'Not specified'}
- Active Hours (Work Focus): {active_hours_start or 'Not specified'} to {active_hours_end or 'Not specified'}
- Time to Next Event: {time_to_next_event} minutes (if applicable)
- Current Free Slot: {free_slot_length} minutes (if applicable)

**Time Window Detection Rules:**
1. **WIND_DOWN_WINDOW**: Within 1 hours before bedtime_start → Suggest calm activities
2. **BEDTIME_WINDOW**: Within bedtime_start to bedtime_end → Suggest sleep prep
3. **ACTIVE_FOCUS_WINDOW**: Within active hours with free slot >= 45 mins → Suggest focus work
4. **MORNING_START**: Within 60 mins after wake → Suggest day planning
5. **PREP_WINDOW**: Less than 30 mins before next event → Suggest prep for that event
6. **SHORT_FOCUS**: In active hours but slot < 45 mins → Suggest mini focus
7. **OPEN_FLEX**: Outside active hours with long free slot >= 90 mins → Suggest flexible activities

**Categories:**
- **great_job**: Positive time window patterns (e.g., "Great timing – you're winding down right on schedule")
- **need_attention**: Time window issues (e.g., "Bedtime is soon – shift anything non-urgent and start winding down")
- **opportunity**: Time window opportunities (e.g., "Perfect timing – you're in your active window")

{_build_productivity_context_prompt(productivity_context)}


IMPORTANT TIME RULES:
- DO NOT mention exact clock times (e.g. 8:58 AM, 9:00, current time)
- DO NOT repeat or paraphrase the provided current_time value
- DO NOT suggest starting new projects, big planning, or scheduling events within 90 minutes of bedtime.
- DO NOT suggest inbox work, admin tasks, or stimulating media during wind-down periods.
- Use ONLY relative time expressions such as:
  "early in your day", "just before active hours",
  "with a long free window ahead",
  "before your next event",
  "outside work hours"
- If time context is needed, describe it abstractly, not numerically

IMPORTANT:
- relative to calendar events (if have):
    - Pay attention to event titles, descriptions, and participants to better understand the context and intent behind each calendar event.
    - Use this contextual understanding to produce more relevant, coherent, and human-like insights.
- Three JSON keys are required; use empty string for categories that do not apply (do not invent filler)
- Be concise - each category 1-2 sentences maximum
- Use natural, conversational language
- Focus on actionable insights
- Respond with valid JSON only
{important_language_prompt(language_name)}"""

    user_prompt = f"""**Time Window Analysis**

**Current Context:**
- Time to Bedtime: {time_to_bedtime} minutes (if applicable)
- Time to Next Event: {time_to_next_event} minutes (if applicable)
- In Active Window: {in_active_window}
- Free Slot Length: {free_slot_length} minutes (if applicable)
- Bedtime Window: {bedtime_start or "Not specified"} to {bedtime_end or "Not specified"}
- Active Hours: {active_hours_start or "Not specified"} to {active_hours_end or "Not specified"}

**Calendar Events:**
{calendar_data}

Analyze time windows and provide insights:
- Wind-down window (1 hour before bedtime)
- Bedtime window
- Active hours window
- Morning start window
- Prep window (before events)

{important_language_prompt(language_name)}

Return JSON with three categories:
{{
  "great_job": "Positive time window patterns",
  "need_attention": "Time window issues",
  "opportunity": "Time window opportunities"
}}"""

    return system_prompt, user_prompt


def get_wind_down_window_overall_insight_prompt(
    calendar_data: str = "",
    bedtime_start: Optional[str] = None,
    bedtime_end: Optional[str] = None,
    time_to_bedtime: Optional[int] = None,
    wind_down_buffer_mins: Optional[int] = None,
    bedtime_streak: Optional[int] = None,
    sleep_quality: Optional[str] = None,
    sleep_last3nights: Optional[List[Optional[float]]] = None,
    sleep_goal: Optional[float] = None,
    sleep_lastnight: Optional[float] = None,
    steps_today: Optional[int] = None,
    steps_goal: Optional[float] = None,
    current_time: str = "",
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for wind-down window overall insight.

    Args:
        calendar_data: Formatted calendar events string
        bedtime_start: Bedtime window start
        bedtime_end: Bedtime window end
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    system_prompt = f"""You are a wind-down window analyzer. Analyze the user's wind-down period (within 1 hours before bedtime) and provide insights in exactly three categories: great_job, need_attention, and opportunity.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Output Format:**
You MUST respond with a valid JSON object containing exactly three fields:
{{
  "great_job": "Summary of achievements and positive patterns",
  "need_attention": "Summary of areas needing attention and actionable guidance",
  "opportunity": "Summary of opportunities and suggestions for optimal actions"
}}

**User Schedule:**
- Bedtime Window: {bedtime_start or 'Not specified'} to {bedtime_end or 'Not specified'}

**Wind-Down Window Rules:**
- **WIND_DOWN_WINDOW**: Within {wind_down_buffer_mins or 60} minutes (1 hours) before bedtime_start → Suggest calm activities
- **LATE_WIND_DOWN**: Within 30 minutes before bedtime_start → Prioritize relaxation
- **BEDTIME_STREAK**: If bedtime_streak >= 3 days → Celebrate consistency
- **TIME_TO_BEDTIME**: {time_to_bedtime or "Not available"} minutes until bedtime

**Categories (multi-domain — use Sleep Context + Calendar below):**
- **great_job**: Best **completed** win today (steps if at goal in data, mood, productivity) — NOT sleep-timing praise if sleep debt.
- **need_attention**: Top concern: sleep debt, heavy calendar, or wind-down (one domain). No step-deficit nudging.
- **opportunity**: Different domain than need_attention if feasible now; calm only (reading, dim lights). No walk for steps, no "prep tomorrow tonight".

{_build_productivity_context_prompt(productivity_context)}

IMPORTANT TIME RULES:
- DO NOT mention exact clock times (e.g. 8:58 AM, 9:00, current time)
- DO NOT repeat or paraphrase the provided current_time value
- DO NOT suggest starting new projects, big planning, or scheduling meetings
- DO NOT suggest inbox work, admin tasks, or stimulating media
- DO NOT suggest intense exercise or cognitively heavy tasks
- Use ONLY relative time expressions such as:
  "as bedtime approaches", "during your wind-down period",
  "before you rest", "as you prepare for sleep"
- Focus on calm, relaxing activities: reading, stretching, meditation (no step catch-up, no tomorrow prep tonight)

IMPORTANT:
- relative to calendar events (if have):
    - Pay attention to event titles, descriptions, and participants to better understand the context and intent behind each calendar event.
    - Use this contextual understanding to produce more relevant, coherent, and human-like insights.
- Three JSON keys are required; use empty string for categories that do not apply (do not invent filler)
- Be concise - each category 1-2 sentences maximum
- Use natural, conversational language
- Focus on relaxation and calm activities
- Respond with valid JSON only
{important_language_prompt(language_name)}"""

    # Format sleep_last3nights
    sleep_last3nights_str = "Not available"
    if sleep_last3nights:
        sleep_hours_list = []
        for i, h in enumerate(sleep_last3nights):
            if h is not None:
                sleep_hours_list.append(f"Night {i+1}: {h:.1f}h")
            else:
                sleep_hours_list.append(f"Night {i+1}: No data")
        sleep_last3nights_str = " | ".join(sleep_hours_list)

    # Build conditional context lines
    context_lines = []
    if bedtime_start and bedtime_end:
        context_lines.append(f"- Bedtime Window: {bedtime_start} to {bedtime_end}")
    if time_to_bedtime is not None:
        context_lines.append(f"- Time to Bedtime: {time_to_bedtime} minutes")
    if wind_down_buffer_mins is not None:
        context_lines.append(f"- Wind-Down Buffer: {wind_down_buffer_mins} minutes")
    if bedtime_streak is not None:
        context_lines.append(f"- Bedtime Streak: {bedtime_streak} days")
    context_section = "\n".join(context_lines) if context_lines else ""

    # Build conditional sleep context lines
    sleep_lines = []
    if sleep_quality:
        sleep_lines.append(
            f"- Sleep Quality: {sleep_quality} (very_high, high, ok, low, very_low)"
        )
    if sleep_last3nights_str and sleep_last3nights_str != "Not available":
        sleep_lines.append(f"- Sleep Last 3 Nights: {sleep_last3nights_str}")
    if sleep_goal is not None:
        sleep_lines.append(f"- Sleep Goal: {sleep_goal} hours")
    if sleep_lastnight is not None:
        sleep_lines.append(f"- Sleep Last Night: {sleep_lastnight} hours")
    sleep_section = "\n".join(sleep_lines) if sleep_lines else ""

    steps_section = ""
    if steps_today is not None and steps_goal is not None:
        steps_section = f"- Steps Today: {steps_today} / Goal: {steps_goal}"
    elif steps_today is not None:
        steps_section = f"- Steps Today: {steps_today}"

    user_prompt = f"""**Wind-Down Window Analysis**

**Current Context:**
{context_section if context_section else ""}

**Sleep Context:**
{sleep_section if sleep_section else ""}

**Steps (for great_job only if goal met — do not mention step gap):**
{steps_section if steps_section else "Not provided"}

**Calendar Events:**
{calendar_data}

**Wind-Down Analysis:**
- Sleep debt → need_attention (one message). Do not also praise sleep phase in great_job.
- If bedtime_streak >= 3 and no sleep debt → great_job may celebrate consistency.
- Use calendar for schedule/rest conflicts in need_attention if stronger than sleep.
- opportunity: calm wind-down only or "".

{important_language_prompt(language_name)}

Return JSON (use "" for unused categories)."""

    return system_prompt, user_prompt


def get_bedtime_window_overall_insight_prompt(
    calendar_data: str = "",
    bedtime_start: Optional[str] = None,
    bedtime_end: Optional[str] = None,
    bedtime_streak: Optional[int] = None,
    sleep_last3nights: Optional[List[Optional[float]]] = None,
    sleep_lastnight: Optional[float] = None,
    last_wake_time: Optional[str] = None,
    first_sleep_time: Optional[str] = None,
    sleep_goal: Optional[float] = None,
    sleep_quality: Optional[str] = None,
    steps_today: Optional[int] = None,
    steps_goal: Optional[float] = None,
    current_time: str = "",
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for bedtime window overall insight.

    Args:
        calendar_data: Formatted calendar events string
        bedtime_start: Bedtime window start
        bedtime_end: Bedtime window end
        bedtime_streak: Consecutive days sleeping within bedtime window
        sleep_last3nights: List of sleep hours for last 3 nights
        sleep_lastnight: Hours of sleep last night
        last_wake_time: Last wake time in ISO format (YYYY-MM-DDTHH:MM:SS)
        first_sleep_time: First sleep time in ISO format (YYYY-MM-DDTHH:MM:SS)
        sleep_goal: Target sleep hours per night
        sleep_quality: Sleep quality level (very_high, high, ok, low, very_low)
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    # Determine if we have sleep timing data
    # Check: both times must exist AND first_sleep_time must be from last night (from 18:00 yesterday onwards)
    has_sleep_timing_data = (
        last_wake_time
        and first_sleep_time
        and _is_sleep_from_last_night(first_sleep_time, current_time, timezone)
    )

    if not has_sleep_timing_data:
        # Case: No sleep timing data (user not wearing device or hasn't slept yet)
        return _get_bedtime_window_overall_insight_no_data_prompt(
            calendar_data=calendar_data,
            current_time=current_time,
            timezone=timezone,
            bedtime_start=bedtime_start,
            bedtime_end=bedtime_end,
            language_name=language_name,
            productivity_context=productivity_context,
        )

    system_prompt = f"""You are a bedtime window analyzer. Analyze the user's bedtime period using sleep timing data and provide insights in exactly three categories: great_job, need_attention, and opportunity.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Output Format:**
You MUST respond with a valid JSON object containing exactly three fields:
{{
  "great_job": "Summary of achievements and positive patterns",
  "need_attention": "Summary of areas needing attention and actionable guidance",
  "opportunity": "Summary of opportunities and suggestions for optimal actions"
}}

**User Schedule:**
- Bedtime Window: {bedtime_start or 'Not specified'} to {bedtime_end or 'Not specified'}
- Sleep Goal: {sleep_goal or 'Not set'} hours

**Sleep Timing Data:**
{_build_conditional_line("First Sleep Time", first_sleep_time, "time")}
{_build_conditional_line("Last Wake Time", last_wake_time, "time")}
{_build_conditional_line("Sleep Last Night", sleep_lastnight, "hours")}
{_build_conditional_line("Sleep Quality", sleep_quality)}

**Sleep analysis (inform ONE category — do not split across three):**
- Compare sleep_lastnight to sleep_goal for debt; check sleep_quality.
- Use timing fields for context only, not to fill great_job + need_attention + opportunity separately.

**Categories (priority order):**
1. If sleep debt (short sleep_lastnight vs goal or low quality) → **need_attention** = sleep debt (one clear sentence). **great_job** = non-sleep win today only (e.g. steps met) or `""`. **opportunity** = `""` or one non-duplicative calm rest line.
2. If adequate sleep + bedtime_streak >= 3 → **great_job** may celebrate consistency; others often `""`.
3. Do NOT praise "mid sleep phase / biological clock OK" while sleep debt is present.
4. Do NOT put step deficit in any category at bedtime.

{_build_productivity_context_prompt(productivity_context)}

IMPORTANT TIME RULES:
- DO NOT mention exact clock times (e.g. 8:58 AM, 9:00, current time)
- DO NOT repeat or paraphrase the provided current_time value
- DO NOT suggest any work, planning, or stimulating activities
- DO NOT suggest screens, inbox, or admin tasks
- Use ONLY relative time expressions such as:
  "at the start of your sleep window", "in the middle of your rest period",
  "as you approach wake time", "during your deep sleep phase",
  "near the end of your sleep window"
- Focus on sleep phase-appropriate activities:
  - Early: dim lights, avoid screens, gentle stretching, meditation, reading (non-stimulating)
  - Middle: maintain sleep environment, avoid disruptions, deep rest
  - Late: gentle awakening preparation, light morning thoughts, maintaining rest

IMPORTANT:
- relative to calendar events (if have):
    - Pay attention to event titles, descriptions, and participants to better understand the context and intent behind each calendar event.
    - Use this contextual understanding to produce more relevant, coherent, and human-like insights.
- Three JSON keys are required; use empty string for categories that do not apply (do not invent filler)
- Be concise - each category 1-2 sentences maximum
- Use natural, conversational language
- Focus on sleep phase analysis and appropriate suggestions
- Respond with valid JSON only
{important_language_prompt(language_name)}"""

    # Format sleep_last3nights for display
    def _format_sleep_last3nights(
        sleep_last3nights: Optional[List[Optional[float]]],
    ) -> str:
        """Format sleep_last3nights for display."""
        if not sleep_last3nights:
            return ""
        sleep_hours_list = [
            f"{h:.1f}" if h is not None else "N/A" for h in sleep_last3nights
        ]
        return f"[{', '.join(sleep_hours_list)}] hours"

    # Build conditional context lines
    context_lines = []
    if bedtime_start and bedtime_end:
        context_lines.append(f"- Bedtime Window: {bedtime_start} to {bedtime_end}")
    if bedtime_streak is not None:
        context_lines.append(f"- Bedtime Streak: {bedtime_streak} days")
    context_section = "\n".join(context_lines) if context_lines else ""

    # Build conditional sleep timing data lines
    sleep_timing_lines = []
    sleep_timing_lines.append(
        _build_conditional_line("First Sleep Time", first_sleep_time)
    )
    sleep_timing_lines.append(_build_conditional_line("Last Wake Time", last_wake_time))
    sleep_timing_lines.append(
        _build_conditional_line("Sleep Last Night", sleep_lastnight, "hours")
    )
    sleep_timing_lines.append(
        _build_conditional_line("Sleep Goal", sleep_goal, "hours")
    )
    sleep_timing_lines.append(_build_conditional_line("Sleep Quality", sleep_quality))
    if sleep_last3nights:
        formatted = _format_sleep_last3nights(sleep_last3nights)
        if formatted:
            sleep_timing_lines.append(f"- Sleep Last 3 Nights: {formatted}")
    sleep_timing_section = (
        "\n".join([line for line in sleep_timing_lines if line])
        if sleep_timing_lines
        else ""
    )

    user_prompt = f"""**Bedtime Window Analysis**

**Current Context:**
{context_section if context_section else ""}

**Sleep Timing Data:**
{sleep_timing_section if sleep_timing_section else ""}

**Calendar Events:**
{calendar_data}

**Steps (great_job only if steps_today >= steps_goal — never mention step gap at bedtime):**
{f"- Steps Today: {steps_today} / Goal: {steps_goal}" if steps_today is not None and steps_goal is not None else (f"- Steps Today: {steps_today}" if steps_today is not None else "Not provided")}

Assign categories per priority rules in system prompt. User is in bedtime — prioritize rest now. Use "" when a category does not apply.

{important_language_prompt(language_name)}

Return JSON (use "" for unused categories)."""

    return system_prompt, user_prompt


def get_morning_window_overall_insight_prompt(
    calendar_data: str = "",
    bedtime_end: Optional[str] = None,
    active_hours_start: Optional[str] = None,
    time_to_next_event: Optional[int] = None,
    sleep_lastnight: Optional[float] = None,
    sleep_goal: Optional[float] = None,
    sleep_quality: Optional[str] = None,
    weekly_health_progress: Optional[Dict[str, Any]] = None,
    free_slot_length: Optional[int] = None,
    current_time: str = "",
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for morning window overall insight.

    Morning start window: current_time > bedtime_end and < active_hours_start
    Within 60 mins after wake → Suggest day planning

    Args:
        calendar_data: Formatted calendar events string
        bedtime_end: Bedtime window end
        active_hours_start: Active hours start
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    system_prompt = f"""You are a morning start analyzer. Analyze the user's morning period (after bedtime window, before active hours) and provide insights in exactly three categories: great_job, need_attention, and opportunity.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Output Format:**
You MUST respond with a valid JSON object containing exactly three fields:
{{
  "great_job": "Summary of achievements and positive patterns",
  "need_attention": "Summary of areas needing attention and actionable guidance",
  "opportunity": "Summary of opportunities and suggestions for optimal actions"
}}

**User Schedule:**
- Bedtime End: {bedtime_end or 'Not specified'}
- Active Hours Start: {active_hours_start or 'Not specified'}

**Morning Start Rules:**
- **MORNING_START**: After bedtime window, before active hours start → Suggest day planning
- Within 60 mins after wake → Focus on planning and preparation for the day
- If good sleep (sleep_lastnight >= sleep_goal AND sleep_quality is "very_high" or "high") → Suggest morning planning and high-impact tasks
- If sleep quality is "low" or "very_low" → Lighter morning, suggest gentle activation
- If first event soon (time_to_next_event <= 30 mins) → Suggest quick prep
- If weekly_health_progress shows good progress → Reinforce positive patterns

**Categories:**
- **great_job**: Positive morning patterns (e.g., "Great start to your day – fresh after rest")
- **need_attention**: Morning issues (e.g., "Early in your day – want to plan ahead?")
- **opportunity**: Morning opportunities (e.g., "Perfect time to plan your day – review priorities and set your focus")

{_build_productivity_context_prompt(productivity_context)}

IMPORTANT TIME RULES:
- DO NOT mention exact clock times (e.g. 8:58 AM, 9:00, current time)
- DO NOT repeat or paraphrase the provided current_time value
- Use ONLY relative time expressions such as:
  "early in your day", "as your day begins",
  "after your rest period", "before your active hours start",
  "at the start of your day"
- If time context is needed, describe it abstractly, not numerically

IMPORTANT:
- relative to calendar events (if have):
    - Pay attention to event titles, descriptions, and participants to better understand the context and intent behind each calendar event.
    - Use this contextual understanding to produce more relevant, coherent, and human-like insights.
- Three JSON keys are required; use empty string for categories that do not apply (do not invent filler)
- Be concise - each category 1-2 sentences maximum
- Use natural, conversational language
- Focus on day planning and morning preparation
- Respond with valid JSON only
{important_language_prompt(language_name)}"""

    # Format weekly_health_progress
    weekly_progress_str = "Not available"
    if weekly_health_progress:
        total_steps = weekly_health_progress.get("total_steps", 0)
        avg_daily = weekly_health_progress.get("avg_daily_steps", 0)
        days_met = weekly_health_progress.get("days_met_goal", 0)
        progress_pct = weekly_health_progress.get("progress_percentage", 0)
        weekly_progress_str = f"Total: {total_steps} steps | Avg: {avg_daily:.0f}/day | Days Met Goal: {days_met} | Progress: {progress_pct:.0f}%"

    # Build conditional context lines
    context_lines = []
    if bedtime_end:
        context_lines.append(f"- Bedtime End: {bedtime_end}")
    if active_hours_start:
        context_lines.append(f"- Active Hours Start: {active_hours_start}")
    if sleep_lastnight is not None:
        context_lines.append(f"- Sleep Last Night: {sleep_lastnight} hours")
    if sleep_goal is not None:
        context_lines.append(f"- Sleep Goal: {sleep_goal} hours")
    if time_to_next_event is not None:
        context_lines.append(f"- Time to Next Event: {time_to_next_event} minutes")
    if free_slot_length is not None:
        context_lines.append(f"- Free Slot Length: {free_slot_length} minutes")
    context_section = "\n".join(context_lines) if context_lines else ""

    # Build conditional morning context lines
    morning_lines = []
    if sleep_quality:
        morning_lines.append(
            f"- Sleep Quality Last Night: {sleep_quality} (very_high, high, ok, low, very_low)"
        )
    if weekly_progress_str:
        morning_lines.append(f"- Weekly Health Progress: {weekly_progress_str}")
    morning_section = "\n".join(morning_lines) if morning_lines else ""

    user_prompt = f"""**Morning Window Analysis**

**Current Context:**
{context_section if context_section else ""}

**Morning Context:**
{morning_section if morning_section else ""}

**Calendar Events:**
{calendar_data}

**Morning Analysis Rules:**
- If sleep_quality is "very_high" or "high" AND sleep_lastnight >= sleep_goal → Great morning start, suggest high-impact tasks
- If sleep_quality is "low" or "very_low" → Lighter morning, suggest gentle activation
- If weekly_health_progress shows good progress → Reinforce positive patterns
- Morning start context (after rest, beginning of day)
- Day planning suggestions
- First event prep (if soon)
- First event timing (if soon, suggest quick prep)
- Day planning suggestions

{important_language_prompt(language_name)}

Return JSON with three categories:
{{
  "great_job": "Positive morning patterns",
  "need_attention": "Morning issues",
  "opportunity": "Morning opportunities"
}}"""

    return system_prompt, user_prompt


def get_active_window_overall_insight_prompt(
    calendar_data: str = "",
    time_to_next_event: Optional[int] = None,
    free_slot_length: Optional[int] = None,
    active_hours_start: Optional[str] = None,
    active_hours_end: Optional[str] = None,
    bedtime_start: Optional[str] = None,
    wind_down_buffer_mins: Optional[int] = None,
    sleep_quality: Optional[str] = None,
    calendar_density: Optional[float] = None,
    back_to_back_count: Optional[int] = None,
    current_time: str = "",
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for active window overall insight.

    Args:
        calendar_data: Formatted calendar events string
        time_to_next_event: Minutes until next event
        free_slot_length: Current free slot length in minutes
        active_hours_start: Active hours start
        active_hours_end: Active hours end
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    system_prompt = f"""You are an active window analyzer. Analyze the user's active work hours and provide insights in exactly three categories: great_job, need_attention, and opportunity.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Output Format:**
You MUST respond with a valid JSON object containing exactly three fields:
{{
  "great_job": "Summary of achievements and positive patterns",
  "need_attention": "Summary of areas needing attention and actionable guidance",
  "opportunity": "Summary of opportunities and suggestions for optimal actions from current time {current_time} onward"
}}

**User Schedule:**
- Active Hours (Work Focus): {active_hours_start or 'Not specified'} to {active_hours_end or 'Not specified'}
- Time to Next Event: {time_to_next_event} minutes (if applicable)
- Current Free Slot: {free_slot_length} minutes (if applicable)

**Active Window Rules:**
1. **ACTIVE_FOCUS_WINDOW**: Within active hours with free slot >= 45 mins AND NOT in wind-down → Suggest focus work
2. **PREP_WINDOW**: Less than 30 mins before next event → Suggest prep for that event
3. **SHORT_FOCUS**: In active hours but slot < 45 mins → Suggest mini focus
4. **OPEN_FLEX**: Outside active hours with long free slot >= 90 mins → Suggest flexible activities


**Categories:**
- **great_job**: Active window achievements (e.g., "Nice focus session – you completed 90 minutes with minimal interruptions")
- **need_attention**: Active window issues (e.g., "Your active hours are getting crowded")
- **opportunity**: Active window opportunities (e.g., "Perfect timing – you're in your active window with a good free window ahead")

{_build_productivity_context_prompt(productivity_context)}

IMPORTANT TIME RULES:
- DO NOT mention exact clock times (e.g. 8:58 AM, 9:00, current time)
- DO NOT repeat or paraphrase the provided current_time value
- Use ONLY relative time expressions such as:
  "in your active hours",
  "with a long free window ahead", "before your next event",
  "during your focus time"
- If time context is needed, describe it abstractly, not numerically
- DO NOT suggest to need_attention or opportunity fields in past time


IMPORTANT:
- relative to calendar events (if have):
    - Pay attention to event titles, descriptions, and participants to better understand the context and intent behind each calendar event.
    - Use this contextual understanding to produce more relevant, coherent, and human-like insights.
- Three JSON keys are required; use empty string for categories that do not apply (do not invent filler)
- Be concise - each category 1-2 sentences maximum
- Use natural, conversational language
- Focus on actionable work and focus insights
- Respond with valid JSON only
{important_language_prompt(language_name)}"""

    # Build conditional context lines
    context_lines = []
    if active_hours_start and active_hours_end:
        context_lines.append(
            f"- Active Hours: {active_hours_start} to {active_hours_end}"
        )
    if time_to_next_event is not None:
        context_lines.append(f"- Time to Next Event: {time_to_next_event} minutes")
    if free_slot_length is not None:
        context_lines.append(f"- Free Slot: {free_slot_length} minutes")
    context_section = "\n".join(context_lines) if context_lines else ""

    # Build conditional active window context lines
    active_lines = []
    if sleep_quality:
        active_lines.append(
            f"- Sleep Quality: {sleep_quality} (affects focus capacity: very_high, high, ok, low, very_low)"
        )
    if calendar_density is not None:
        active_lines.append(
            f"- Calendar Density: {calendar_density} event blocks in active hours"
        )
    if back_to_back_count is not None:
        active_lines.append(
            f"- Back-to-back Count: {back_to_back_count} consecutive events"
        )
    active_section = "\n".join(active_lines) if active_lines else ""

    user_prompt = f"""**Active Window Analysis**

**Current Context:**
{context_section if context_section else ""}

**Active Window Context:**
{active_section if active_section else ""}

**Calendar Events:**
{calendar_data}

**Active Window Analysis:**
- If sleep_quality is "low" or "very_low" → Suggest lighter focus tasks, avoid deep work
- If calendar_density high → Suggest protecting remaining focus blocks
- If back_to_back_count >= 3 → Suggest micro-breaks between events
- Active window context (focus time, prep window, etc.)
- Work/focus suggestions

{important_language_prompt(language_name)}

Return JSON with three categories:
{{
  "great_job": "Active window achievements",
  "need_attention": "Active window issues",
  "opportunity": "Active window opportunities from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


def get_evening_flexible_overall_insight_prompt(
    calendar_data: str = "",
    bedtime_start: Optional[str] = None,
    active_hours_end: Optional[str] = None,
    wind_down_buffer_mins: Optional[int] = None,
    free_slot_length: Optional[int] = None,
    time_to_bedtime: Optional[int] = None,
    current_hour: Optional[int] = None,
    mood: Optional[str] = None,
    mood_not_logged_today: bool = False,
    current_time: str = "",
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for evening flexible window overall insight.

    Evening flexible window: from 19:00 (7 PM) to before wind_down_window
    Wind_down_window starts 1 hours before bedtime_start
    This is a flexible time for recharge, connection, hobbies, or light activities

    Args:
        calendar_data: Formatted calendar events string
        bedtime_start: Bedtime window start
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        mood: Current mood ("terrible", "sad", "okay", "happy", "amazing")
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    mood_section = ""
    if mood:
        mood_section = f"""
**Current Mood:** {mood}

**Mood-Aware Evening Suggestions:**
- **Mood: Terrible/Sad**: Offer gentle support, low-effort activities, reduce load, focus on comfort and self-care
- **Mood: Okay**: Suggest meaningful but not demanding activities, light progress, gentle recharge
- **Mood: Happy/Awesome**: Channel positive energy into enjoyable activities, but avoid overcommitment, maintain balance
- **Mood + Evening Time**: Combine mood state with evening flexible window to suggest appropriate activities
"""
    elif mood_not_logged_today:
        mood_section = """
**Mood Today:** Not logged yet
- Do NOT assume the user's mood; do NOT use any previously logged mood as current state.
- You MAY occasionally include a soft, optional invitation to log mood (preferably inside `opportunity`, max one short sentence). Skip it if the field already has a stronger, more relevant suggestion.
- Never make the mood-log reminder the main focus, and never repeat it across multiple fields.
"""

    system_prompt = f"""You are an evening flexible window analyzer. Analyze the user's evening flexible period (from 7 PM to before wind-down) and provide insights in exactly three categories: great_job, need_attention, and opportunity.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}
{mood_section}
**Output Format:**
You MUST respond with a valid JSON object containing exactly three fields:
{{
  "great_job": "Summary of achievements and positive patterns",
  "need_attention": "Summary of areas needing attention and actionable guidance",
  "opportunity": "Summary of opportunities and suggestions for optimal actions from current time {current_time} onward"
}}

**User Schedule:**
- Bedtime Window Start: {bedtime_start or 'Not specified'}
- Active Hours End: {active_hours_end or 'Not specified'}
- Wind-Down Buffer: {wind_down_buffer_mins or 60} minutes
- Evening Flexible Window: From 19:00 to before wind-down ({wind_down_buffer_mins or 60} minutes before bedtime)
- Time to Bedtime: {time_to_bedtime or "Not available"} minutes
- Current Hour: {current_hour or "Not available"}

**Evening Flexible Window Rules:**
- **EVENING_FLEXIBLE**: From 19:00 to before wind-down window → Suggest flexible activities
- **AFTER_WORK**: If current_time > active_hours_end → After work hours, suggest recovery/fitness/social/hobbies
- **EARLY_EVENING**: If 18:00 <= current_hour <= 19:00 AND slot >= 30 mins → Suggest early evening meal
- **IN_WIND_DOWN**: If time_to_bedtime <= {wind_down_buffer_mins or 60} mins → Already in wind-down, suggest calm activities
- This is a flexible time for recharge, connection, hobbies, or light activities
- Focus on activities that help transition from work to rest
- **Mood Integration**: Adjust suggestions based on user's current mood state

**Categories (multi-domain + mood; use Calendar + time_to_bedtime):**
- **great_job**: Completed wins today (steps at goal, mood, productivity) — not generic "great balance".
- **need_attention**: One priority: calendar/load, sleep debt, or boundaries. No step-deficit if close to bedtime.
- **opportunity**: Feasible this evening only; if in_wind_down or time_to_bedtime short → calm rest or "". No step catch-up, no prep tomorrow tonight.

{_build_productivity_context_prompt(productivity_context)}

IMPORTANT TIME RULES:
- DO NOT mention exact clock times (e.g. 7:00 PM, 8:00, current time)
- DO NOT repeat or paraphrase the provided current_time value
- DO NOT suggest intense work, heavy planning, or cognitively demanding tasks
- DO NOT suggest activities that interfere with wind-down (avoid stimulating media, intense exercise)
- Use ONLY relative time expressions such as:
  "in your evening", "during your flexible evening time",
  "as evening approaches", "in this evening window",
  "before your wind-down period"
- Focus on flexible evening activities: recharge, connection, hobbies, light activities, meal planning, gentle movement

IMPORTANT MOOD RULES:
- If mood is Terrible/Sad: Offer gentle support, low-effort options, reduce load, focus on comfort
- If mood is Okay: Suggest meaningful but not demanding activities, light progress
- If mood is Happy/Awesome: Channel positive energy but avoid overcommitment
- NEVER push high-pressure activities when mood is Terrible/Sad
- NEVER interpret/diagnose mental health – use supportive language only
- Adjust evening suggestions to match mood state appropriately

IMPORTANT:
- relative to calendar events (if have):
    - Pay attention to event titles, descriptions, and participants to better understand the context and intent behind each calendar event.
    - Use this contextual understanding to produce more relevant, coherent, and human-like insights.
- Three JSON keys are required; use empty string for categories that do not apply (do not invent filler)
- Be concise - each category 1-2 sentences maximum
- Use natural, conversational, empathetic language
- Focus on flexible evening activities that match mood state
- Respond with valid JSON only

{_overall_block("evening_flexible")}

{important_language_prompt(language_name)}"""

    # Build conditional context lines
    context_lines = []
    if bedtime_start:
        context_lines.append(f"- Bedtime Start: {bedtime_start}")
    if active_hours_end:
        context_lines.append(f"- Active Hours End: {active_hours_end}")
    if wind_down_buffer_mins is not None:
        context_lines.append(f"- Wind-Down Buffer: {wind_down_buffer_mins} minutes")
    if time_to_bedtime is not None:
        context_lines.append(f"- Time to Bedtime: {time_to_bedtime} minutes")
    if free_slot_length is not None:
        context_lines.append(f"- Free Slot Length: {free_slot_length} minutes")
    if current_hour is not None:
        context_lines.append(f"- Current Hour: {current_hour}")
        after_work = (
            "Yes"
            if (
                active_hours_end and current_hour >= int(active_hours_end.split(":")[0])
                if active_hours_end
                else False
            )
            else "No"
        )
        early_evening = "Yes" if (17 <= current_hour <= 19) else "No"
        in_wind_down = (
            "Yes"
            if (
                time_to_bedtime is not None
                and time_to_bedtime <= (wind_down_buffer_mins or 60)
            )
            else "No"
        )
        context_lines.append(f"- After Work Hours: {after_work}")
        context_lines.append(f"- Early Evening (17:00-19:00): {early_evening}")
        context_lines.append(f"- In Wind-Down: {in_wind_down}")
    context_section = "\n".join(context_lines) if context_lines else ""

    if mood:
        mood_line = f"**Current Mood:** {mood}\n"
    elif mood_not_logged_today:
        mood_line = "**Mood Today:** Not logged yet (optional soft reminder allowed)\n"
    else:
        mood_line = ""

    user_prompt = f"""**Evening Flexible Window Analysis**

{mood_line}**Current Context:**
{context_section if context_section else ""}

**Calendar Events:**
{calendar_data}

Analyze evening flexible window with mood awareness and provide insights:
- Evening flexible context (flexible evening time, before wind-down)
- After work detection (if after active hours end)
- Early evening detection (17:00-19:00)
- Wind-down detection (if approaching bedtime)
- Flexible evening suggestions (recharge, connection, hobbies, light activities) adjusted for mood
- Mood-appropriate evening activities

{important_language_prompt(language_name)}

Return JSON (use "" for unused categories)."""

    return system_prompt, user_prompt


def get_calendar_pattern_overall_insight_prompt(
    calendar_data: str = "",
    calendar_density: Optional[float] = None,
    back_to_back_count: Optional[int] = None,
    meeting_minutes: Optional[int] = None,
    sleep_quality: Optional[str] = None,
    weekly_health_progress: Optional[Dict[str, Any]] = None,
    active_hours_start: Optional[str] = None,
    active_hours_end: Optional[str] = None,
    calendar_events_next48h: Optional[List[Dict[str, Any]]] = None,
    current_time: str = "",
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for calendar pattern overall insight (Group 3).

    Args:
        calendar_data: Formatted calendar events string
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    system_prompt = f"""You are a calendar pattern analyzer. Analyze the user's calendar patterns and provide insights in exactly three categories: great_job, need_attention, and opportunity.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Output Format:**
You MUST respond with a valid JSON object containing exactly three fields:
{{
  "great_job": "Summary of achievements and positive patterns",
  "need_attention": "Summary of areas needing attention and actionable guidance",
  "opportunity": "Summary of opportunities and suggestions for optimal actions from current time {current_time} onward"
}}

**Patterns to Detect:**
1. **Back-to-back events**: Consecutive events with no gap (Count: {back_to_back_count or "Not available"})
2. **Calendar density**: High number of event blocks in active hours (Density: {calendar_density or "Not available"})
3. **Event minutes (last 3 hours)**: {meeting_minutes or "Not available"} minutes
4. **Fragmentation**: Many small free slots (high context switching)
5. **Event streak**: Continuous event blocks for >= 1 hours
6. **Schedule optimization**: No focus blocks in active hours for >= 3 days
7. **Sleep Quality + Calendar**: If sleep quality is low AND (calendar density high OR back-to-back events) → Suggest lighter schedule

**Calendar Context:**
- Active Hours: {active_hours_start or "Not specified"} to {active_hours_end or "Not specified"}
- Sleep Quality: {sleep_quality or "Not available"} (very_high, high, ok, low, very_low)
- Events Next 48 Hours: {len(calendar_events_next48h) if calendar_events_next48h else 0} events

**Categories:**
- **great_job**: Positive calendar patterns (e.g., "Well done – those short breaks help a lot")
- **need_attention**: Calendar pattern issues (e.g., "Low sleep quality + back-to-back events – add micro-breaks")
- **opportunity**: Calendar optimization opportunities (e.g., "Your active hours are getting crowded")

{_build_productivity_context_prompt(productivity_context)}

IMPORTANT TIME RULES:
- DO NOT reference the current moment (e.g. "right now", "currently", "at this time")
- DO NOT restate or paraphrase the provided current_time
- DO NOT use relative phrases tied to now (e.g. "about to", "coming up next", "just finished")
- You MAY reference event time ranges that exist within the calendar data (e.g. "from 1–4 PM")
- Prefer pattern-based and structural descriptions over time-sensitive language
- Frame insights as general schedule observations, not real-time alerts
- DO NOT suggest any task, slot, or action in the past to need_attention or opportunity fields
- DO NOT suggest anything that overlaps with existing calendar events to need_attention or opportunity fields
- DO NOT suggest a task longer than the available free slot to need_attention or opportunity fields
- DO NOT suggest deep work, workouts, or cognitively heavy tasks within 20 minutes before the next event to need_attention or opportunity fields
- DO NOT squeeze new meetings or tasks between consecutive back-to-back events
- DO NOT suggest to need_attention or opportunity fields in past time

IMPORTANT:
- Pay attention to event titles, descriptions, and participants to better understand the context and intent behind each calendar event.
- Use this contextual understanding to produce more relevant, coherent, and human-like insights.
- Three JSON keys are required; use empty string for categories that do not apply (do not invent filler)
- Be concise - each category 1-2 sentences maximum
- Use natural, conversational language
- Focus on actionable insights
- Respond with valid JSON only
{important_language_prompt(language_name)}"""

    # Format weekly_health_progress
    weekly_progress_str = "Not available"
    if weekly_health_progress:
        total_steps = weekly_health_progress.get("total_steps", 0)
        avg_daily = weekly_health_progress.get("avg_daily_steps", 0)
        days_met = weekly_health_progress.get("days_met_goal", 0)
        progress_pct = weekly_health_progress.get("progress_percentage", 0)
        weekly_progress_str = f"Total: {total_steps} steps | Avg: {avg_daily:.0f}/day | Days Met Goal: {days_met} | Progress: {progress_pct:.0f}%"

    # Format calendar_events_next48h
    events_next48h_summary = "Not available"
    if calendar_events_next48h:
        work_events = [
            e
            for e in calendar_events_next48h
            if any(
                k
                in (
                    f"{e.get('summary', '')} {e.get('description', '')} {e.get('location', '')}"
                ).lower()
                for k in [
                    "work",
                    "meeting",
                    "standup",
                    "sync",
                    "client",
                    "project",
                    "office",
                    "họp",
                ]
            )
        ]
        events_next48h_summary = f"Total: {len(calendar_events_next48h)} events | Work-related: {len(work_events)} events"

    # Build conditional calendar metrics lines
    calendar_lines = []
    if calendar_density is not None:
        calendar_lines.append(
            f"- Calendar Density: {calendar_density} event blocks in active hours"
        )
    if back_to_back_count is not None:
        calendar_lines.append(
            f"- Back-to-back Count: {back_to_back_count} consecutive events"
        )
    if meeting_minutes is not None:
        calendar_lines.append(
            f"- Event Minutes (Last 3 Hours): {meeting_minutes} minutes"
        )
    if sleep_quality:
        calendar_lines.append(
            f"- Sleep Quality: {sleep_quality} (affects schedule capacity: very_high, high, ok, low, very_low)"
        )
    if active_hours_start and active_hours_end:
        calendar_lines.append(
            f"- Active Hours: {active_hours_start} to {active_hours_end}"
        )
    if events_next48h_summary:
        calendar_lines.append(f"- Events Next 48 Hours: {events_next48h_summary}")
    calendar_section = "\n".join(calendar_lines) if calendar_lines else ""

    # Build conditional health context lines
    health_lines = []
    if weekly_progress_str:
        health_lines.append(f"- Weekly Health Progress: {weekly_progress_str}")
    health_section = "\n".join(health_lines) if health_lines else ""

    user_prompt = f"""**Calendar Pattern Analysis**

**Calendar Metrics:**
{calendar_section if calendar_section else ""}

**Health Context:**
{health_section if health_section else ""}

**Calendar Events:**
{calendar_data}

**Pattern Analysis Rules:**
- If sleep_quality is "low" or "very_low" AND calendar_density high → Suggest lighter schedule
- If sleep_quality is "low" or "very_low" AND back_to_back_count >= 3 → Suggest recovery breaks
- If calendar_events_next48h shows heavy load → Suggest protecting breaks and recovery time
- Back-to-back meetings
- Calendar density
- Fragmentation
- Meeting streaks
- Schedule optimization needs

{important_language_prompt(language_name)}

Return JSON with three categories:
{{
  "great_job": "Positive calendar patterns",
  "need_attention": "Calendar pattern issues",
  "opportunity": "Calendar opportunities from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


def get_health_goal_overall_insight_prompt(
    steps_today: Optional[int] = None,
    steps_goal: Optional[float] = None,
    steps_streak: Optional[int] = None,
    sleep_lastnight: Optional[float] = None,
    sleep_goal: Optional[float] = None,
    sleep_quality: Optional[str] = None,
    weekly_health_progress: Optional[Dict[str, Any]] = None,
    daily_health_progress: Optional[Dict[str, Any]] = None,
    sleep_last3nights: Optional[List[Optional[float]]] = None,
    free_slot_length: Optional[int] = None,
    time_to_bedtime: Optional[int] = None,
    current_hour: Optional[int] = None,
    is_weekend: Optional[bool] = None,
    current_time: str = "",
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for health goal overall insight (Group 4).

    Args:
        steps_today: Today's step count
        steps_goal: Daily steps goal
        sleep_lastnight: Sleep hours last night
        sleep_goal: Daily sleep goal
        sleep_quality: Sleep quality level (very_high, high, ok, low, very_low)
        free_slot_length: Current free slot length in minutes
        time_to_bedtime: Minutes until bedtime
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    # Calculate progress
    steps_percentage = None
    sleep_percentage = None
    if steps_today and steps_goal:
        steps_percentage = (
            int((steps_today / steps_goal) * 100) if steps_goal > 0 else 0
        )
    if sleep_lastnight and sleep_goal:
        sleep_percentage = (
            int((sleep_lastnight / sleep_goal) * 100) if sleep_goal > 0 else 0
        )

    system_prompt = f"""You are a health goal progress tracker. Analyze the user's health data against their goals and provide insights in exactly three categories: great_job, need_attention, and opportunity.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Output Format:**
You MUST respond with a valid JSON object containing exactly three fields:
{{
  "great_job": "Summary of achievements and positive patterns",
  "need_attention": "Summary of areas needing attention and actionable guidance",
  "opportunity": "Summary of opportunities and suggestions for optimal actions from current time {current_time} onward"
}}

**Health Goals:**
- Steps Goal: {steps_goal or 'Not set'} steps/day
- Sleep Goal: {sleep_goal or 'Not set'} hours/day

**Current Health Data:**
- Steps Today: {steps_today or 'N/A'} / Goal: {steps_goal or 'N/A'} ({steps_percentage or 'N/A'}%)
- Steps Streak: {steps_streak or 'N/A'} days
- Sleep Last Night: {sleep_lastnight or 'N/A'} hours / Goal: {sleep_goal or 'N/A'} hours ({sleep_percentage or 'N/A'}%)
- Sleep Quality: {sleep_quality or 'N/A'} (very_high, high, ok, low, very_low)
- Sleep Last 3 Nights: {', '.join([f"{h:.1f}" if h is not None else "N/A" for h in sleep_last3nights]) if sleep_last3nights else "Not available"}
- Sleep goal: {sleep_goal or 'N/A'} hours/day

**Progress Metrics:**
- Weekly Health Progress: {weekly_health_progress or ""}
- Daily Health Progress: {daily_health_progress or ""}

**Context:**
- Free Slot: {free_slot_length} minutes (if applicable)
- Time to Bedtime: {time_to_bedtime} minutes (if applicable)
- Current Hour: {current_hour or "Not available"}
- Is Weekend: {is_weekend or False}

**Health Catch-up Rules:**
1. If behind on goals AND (catch-up window OR weekend) AND slot >= 30 mins → Suggest health catch-up
2. If goals met → Celebrate and suggest maintenance
3. If weekend AND behind on goals → Suggest weekend catch-up
4. If steps_streak >= 5 days → Celebrate consistency
5. If sleep quality is "very_high" or "high" → Celebrate strong sleep
6. If sleep quality is "low" or "very_low" → Suggest lighter day
7. If sleep_last3nights shows sleep debt → Suggest recovery

**Categories:**
- **great_job**: Health achievements (e.g., "Great job – you've hit your goal today" or "Nice steps streak – {steps_streak or 0} days strong!" or "Sleep quality is strong – {sleep_quality or 'N/A'}")
- **need_attention**: Health issues (e.g., "Your sleep quality was {sleep_quality or 'low'} – let's keep today lighter" or "Sleep debt from last 3 nights – prioritize recovery")
- **opportunity**: Health catch-up opportunities (e.g., "Quick movement will help you stay on track" or "Weekend catch-up window – perfect for health goals")

{_build_productivity_context_prompt(productivity_context)}

IMPORTANT TIME RULES:
- DO NOT mention or restate the exact current time
- DO NOT use phrases implying real-time urgency (e.g. "now", "right now", "at this moment")
- DO NOT suggest immediate actions tied to the current minute
- When referencing catch-up windows, describe them qualitatively (e.g. "later in the day", "during typical catch-up hours")
- Health progress numbers (steps, sleep, heart rate) are allowed
- Frame suggestions as flexible opportunities, not real-time commands

IMPORTANT:
- Three JSON keys are required; use empty string for categories that do not apply (do not invent filler)
- Be concise - each category 1-2 sentences maximum
- Use natural, conversational language
- Focus on actionable health suggestions
- Respond with valid JSON only
{important_language_prompt(language_name)}"""

    # Format sleep_last3nights for display
    sleep_last3nights_str = "Not available"
    if sleep_last3nights:
        sleep_hours_list = []
        for i, h in enumerate(sleep_last3nights):
            if h is not None:
                sleep_hours_list.append(f"Night {i+1}: {h:.1f}h")
            else:
                sleep_hours_list.append(f"Night {i+1}: No data")
        sleep_last3nights_str = " | ".join(sleep_hours_list)

    # Format weekly_health_progress
    weekly_progress_str = None
    if weekly_health_progress:
        total_steps = weekly_health_progress.get("total_steps", 0)
        avg_daily = weekly_health_progress.get("avg_daily_steps", 0)
        days_met = weekly_health_progress.get("days_met_goal", 0)
        progress_pct = weekly_health_progress.get("progress_percentage", 0)
        weekly_progress_str = f"Total: {total_steps} steps | Avg: {avg_daily:.0f}/day | Days Met Goal: {days_met} | Progress: {progress_pct:.0f}%"

    # Format daily_health_progress
    daily_progress_str = None
    if daily_health_progress:
        steps_today_val = daily_health_progress.get("steps_today", 0)
        steps_goal_val = daily_health_progress.get("steps_goal", 0)
        progress_pct = daily_health_progress.get("progress_percentage", 0)
        remaining = daily_health_progress.get("remaining_steps", 0)
        met_goal = daily_health_progress.get("met_goal", False)
        daily_progress_str = f"{steps_today_val}/{steps_goal_val} steps ({progress_pct:.0f}%) | Remaining: {remaining} | Goal Met: {met_goal}"

    # Build conditional health data lines
    health_lines = []
    if steps_today is not None and steps_goal is not None:
        health_lines.append(f"- Steps Today: {steps_today} / Goal: {steps_goal}")
    elif steps_today is not None:
        health_lines.append(f"- Steps Today: {steps_today}")
    elif steps_goal is not None:
        health_lines.append(f"- Steps Goal: {steps_goal}")
    if steps_streak is not None:
        health_lines.append(f"- Steps Streak: {steps_streak} days")
    if sleep_lastnight is not None and sleep_goal is not None:
        health_lines.append(
            f"- Sleep Last Night: {sleep_lastnight} hours / Goal: {sleep_goal} hours"
        )
    elif sleep_lastnight is not None:
        health_lines.append(f"- Sleep Last Night: {sleep_lastnight} hours")
    elif sleep_goal is not None:
        health_lines.append(f"- Sleep Goal: {sleep_goal} hours")
    if sleep_quality:
        health_lines.append(
            f"- Sleep Quality: {sleep_quality} (very_high, high, ok, low, very_low)"
        )
    if sleep_last3nights_str:
        health_lines.append(f"- Sleep Last 3 Nights: {sleep_last3nights_str}")
    health_section = "\n".join(health_lines) if health_lines else ""

    # Build conditional progress metrics lines
    progress_lines = []
    if weekly_progress_str:
        progress_lines.append(f"- Weekly Health Progress: {weekly_progress_str}")
    if daily_progress_str:
        progress_lines.append(f"- Daily Health Progress: {daily_progress_str}")
    progress_section = "\n".join(progress_lines) if progress_lines else ""

    # Build conditional context lines
    context_lines = []
    if free_slot_length is not None:
        context_lines.append(f"- Free Slot Length: {free_slot_length} minutes")
    if time_to_bedtime is not None:
        context_lines.append(f"- Time to Bedtime: {time_to_bedtime} minutes")
    if current_hour is not None:
        context_lines.append(f"- Current Hour: {current_hour}")
    if is_weekend is not None:
        context_lines.append(f"- Is Weekend: {is_weekend}")
    context_section = "\n".join(context_lines) if context_lines else ""

    user_prompt = f"""**Health Goal Analysis**

**Health Data:**
{health_section if health_section else ""}

**Progress Metrics:**
{progress_section if progress_section else ""}

**Current Context:**
{context_section if context_section else ""}

**Health Catch-up Analysis:**
- If daily_health_progress shows < 60% progress AND current_hour >= 16 → Suggest afternoon catch-up
- If daily_health_progress shows < 40% progress AND current_hour >= 17 → Suggest evening catch-up
- If weekly_health_progress shows < 70% progress AND is_weekend → Suggest weekend catch-up

Analyze health goal progress and provide insights:
- Health targets behind/ahead
- Steps goal progress and streak
- Sleep goal progress and quality
- Sleep debt tracking (last 3 nights)
- Weekly and daily health progress
- Health catch-up opportunities (considering weekend, catch-up window, free slot)

{important_language_prompt(language_name)}

Return JSON with three categories:
{{
  "great_job": "Health achievements",
  "need_attention": "Health issues",
  "opportunity": "Health catch-up opportunities from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


def get_recovery_break_overall_insight_prompt(
    calendar_data: str = "",
    time_to_next_event: Optional[int] = None,
    free_slot_length: Optional[int] = None,
    continuous_events_minutes: Optional[int] = None,
    back_to_back_count: Optional[int] = None,
    meeting_minutes: Optional[int] = None,
    calendar_density: Optional[float] = None,
    sleep_quality: Optional[str] = None,
    is_weekend: Optional[bool] = None,
    current_time: str = "",
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for recovery break overall insight (Group 5).

    Args:
        calendar_data: Formatted calendar events string
        time_to_next_event: Minutes until next event
        free_slot_length: Current free slot length in minutes
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    system_prompt = f"""You are a recovery and break advisor. Detect when user needs breaks and provide recovery suggestions in exactly three categories: great_job, need_attention, and opportunity.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Output Format:**
You MUST respond with a valid JSON object containing exactly three fields:
{{
  "great_job": "Summary of achievements and positive patterns",
  "need_attention": "Summary of areas needing attention and actionable guidance",
  "opportunity": "Summary of opportunities and suggestions for optimal actions from current time {current_time} onward"
}}

**Recovery Rules:**
1. **Continuous Work**: If continuous work/meetings >= 90 mins without break → Suggest break (Current: {continuous_events_minutes or "Not available"} mins)
2. **Back-to-back**: If >= 3 back-to-back meetings → Suggest micro-recovery (Current: {back_to_back_count or "Not available"} back-to-back)
3. **Meeting Streak**: If meeting minutes in last 3 hours >= 120 → Suggest recovery window (Current: {meeting_minutes or "Not available"} mins)
4. **Calendar Density**: If calendar density high → Suggest breaks between meetings (Current: {calendar_density or "Not available"})
5. **Recovery Missing**: If continuous events >= 120 mins OR breaks_today < 2 → Break first
6. **Weekend Recovery**: If weekend AND work-heavy → Suggest recovery + move flexible work

**Categories:**
- **great_job**: (usually empty for recovery needs)
- **need_attention**: Recovery needs (e.g., "You've been going non-stop – take a 5-minute reset?")
- **opportunity**: Recovery opportunities (e.g., "Back-to-back meetings – quick reset?")

{_build_productivity_context_prompt(productivity_context)}

IMPORTANT TIME RULES:
- DO NOT mention or restate the exact current time
- DO NOT use real-time urgency language (e.g. "now", "right now", "immediately")
- DO NOT reference rolling time windows tied to the present (e.g. "in the last 3 hours")
- Describe work intensity and recovery needs using qualitative, pattern-based language
- Frame break suggestions as flexible opportunities, not immediate commands

IMPORTANT:
- relative to calendar events (if have):
    - Pay attention to event titles, descriptions, and participants to better understand the context and intent behind each calendar event.
    - Use this contextual understanding to produce more relevant, coherent, and human-like insights.
- Three JSON keys are required; use empty string for categories that do not apply (do not invent filler)
- Be concise - each category 1-2 sentences maximum
- Use natural, conversational language
- Focus on recovery and wellbeing
- Respond with valid JSON only
{important_language_prompt(language_name)}"""

    # Build conditional recovery context lines
    recovery_lines = []
    if sleep_quality:
        recovery_lines.append(
            f"- Sleep Quality: {sleep_quality} (affects recovery needs: very_high, high, ok, low, very_low)"
        )
    if continuous_events_minutes is not None:
        recovery_lines.append(
            f"- Continuous Events: {continuous_events_minutes} minutes without break"
        )
    if back_to_back_count is not None:
        recovery_lines.append(
            f"- Back-to-back Count: {back_to_back_count} consecutive meetings"
        )
    if meeting_minutes is not None:
        recovery_lines.append(
            f"- Meeting Minutes (Last 3 Hours): {meeting_minutes} minutes"
        )
    if calendar_density is not None:
        recovery_lines.append(
            f"- Calendar Density: {calendar_density} meetings in active hours"
        )
    recovery_section = "\n".join(recovery_lines) if recovery_lines else ""

    # Build conditional current context lines
    context_lines = []
    if time_to_next_event is not None:
        context_lines.append(f"- Time to Next Event: {time_to_next_event} minutes")
    if free_slot_length is not None:
        context_lines.append(f"- Free Slot Length: {free_slot_length} minutes")
    if is_weekend is not None:
        context_lines.append(f"- Is Weekend: {is_weekend}")
    context_section = "\n".join(context_lines) if context_lines else ""

    user_prompt = f"""**Recovery & Break Analysis**

**Recovery Context:**
{recovery_section if recovery_section else ""}

**Current Context:**
{context_section if context_section else ""}

**Recovery Analysis:**
- If sleep_quality is "low" or "very_low" AND continuous_events_minutes >= 90 → Prioritize recovery breaks
- If sleep_quality is "low" or "very_low" AND back_to_back_count >= 3 → Suggest micro-breaks
- If continuous_events_minutes >= 120 → Critical break needed

**Calendar Events:**
{calendar_data}

Analyze recovery needs:
- Continuous work without breaks (>= 90 mins threshold)
- Back-to-back meetings (>= 3 threshold)
- Meeting streak (>= 120 mins in last 3 hours)
- Calendar density impact
- Weekend recovery needs
- Recovery missing

{important_language_prompt(language_name)}

Return JSON with three categories:
{{
  "great_job": "Recovery achievements",
  "need_attention": "Recovery needs",
  "opportunity": "Recovery opportunities from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


def get_task_type_overall_insight_prompt(
    calendar_data: str = "",
    free_slot_length: Optional[int] = None,
    time_to_next_event: Optional[int] = None,
    time_to_bedtime: Optional[int] = None,
    in_active_window: Optional[bool] = None,
    current_time: str = "",
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for task type overall insight (Group 6).

    Args:
        calendar_data: Formatted calendar events string
        free_slot_length: Current free slot length in minutes
        time_to_next_event: Minutes until next event
        time_to_bedtime: Minutes until bedtime
        in_active_window: Whether in active hours window
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    system_prompt = f"""You are a task type matcher. Determine which task types are appropriate for the current context and provide insights in exactly three categories: great_job, need_attention, and opportunity.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Output Format:**
You MUST respond with a valid JSON object containing exactly three fields:
{{
  "great_job": "Summary of achievements and positive patterns",
  "need_attention": "Summary of areas needing attention and actionable guidance",
  "opportunity": "Summary of opportunities and suggestions for optimal actions from current time {current_time} onward"
}}

**Context:**
- Slot Length: {free_slot_length} minutes
- Time to Next Event: {time_to_next_event} minutes
- Time to Bedtime: {time_to_bedtime} minutes
- In Active Window: {in_active_window}

**Task Type Rules:**
1. **Deep Work**: Requires >= 45 mins AND in active window AND >= 20 mins before next event
2. **Intense Tasks**: Block if time_to_bedtime <= 60 mins
3. **New Commitments**: Block if time_to_bedtime <= 30 mins
4. **Deep Work Outside Active**: Block if NOT in_active_window
5. **Meeting Prep**: Appropriate if time_to_next_event <= 30 mins
6. **Quick Tasks**: Appropriate if slot_length < 45 mins

**Categories:**
- **great_job**: Task completion achievements (e.g., "Nice focus session")
- **need_attention**: (usually empty for task type)
- **opportunity**: Task opportunities (e.g., "Short focus window – want a 25-min sprint?")

{_build_productivity_context_prompt(productivity_context)}

IMPORTANT TIME RULES:
- DO NOT mention exact durations or countdowns (e.g. "60 minutes", "45 mins left")
- DO NOT restate time_to_next_event or time_to_bedtime values
- DO NOT use real-time urgency language (e.g. "now", "right now", "start immediately")
- Describe time availability qualitatively (e.g. "a long free window", "a short gap before the next event")
- Frame task suggestions as appropriate options, not immediate commands

IMPORTANT:
- relative to calendar events (if have):
    - Pay attention to event titles, descriptions, and participants to better understand the context and intent behind each calendar event.
    - Use this contextual understanding to produce more relevant, coherent, and human-like insights.
- Three JSON keys are required; use empty string for categories that do not apply (do not invent filler)
- Be concise - each category 1-2 sentences maximum
- Use natural, conversational language
- Focus on appropriate task matching
- Respond with valid JSON only
{important_language_prompt(language_name)}"""

    user_prompt = f"""**Task Type Matching Analysis**

**Current Context:**
- Free Slot Length: {free_slot_length} minutes (if applicable)
- Time to Next Event: {time_to_next_event} minutes (if applicable)
- Time to Bedtime: {time_to_bedtime} minutes (if applicable)
- In Active Window: {in_active_window}

**Calendar Events:**
{calendar_data}

Analyze task type matching:
- Deep work requires >= 45 mins
- Avoid deep work < 20 mins before meeting
- Avoid intense tasks 1 hours before bedtime
- Short focus windows

{important_language_prompt(language_name)}

Return JSON with three categories:
{{
  "great_job": "Task completion achievements",
  "need_attention": "Task areas needing attention",
  "opportunity": "Task opportunities from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


def get_mood_based_overall_insight_prompt(
    mood: Optional[str] = None,
    latest_heart_rate: Optional[int] = None,
    sleep_lastnight: Optional[float] = None,
    sleep_goal: Optional[float] = None,
    in_active_window: Optional[bool] = None,
    free_slot_length: Optional[int] = None,
    is_weekend: Optional[bool] = None,
    current_hour: Optional[int] = None,
    back_to_back_count: Optional[int] = None,
    calendar_events_next3h: Optional[List[Dict[str, Any]]] = None,
    current_time: str = "",
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for mood-based overall insight (Group 7).

    Args:
        mood: Current mood ("terrible", "sad", "okay", "happy", "amazing")
        latest_heart_rate: Latest heart rate
        sleep_lastnight: Sleep hours last night
        sleep_goal: Daily sleep goal
        in_active_window: Whether in active hours window
        free_slot_length: Current free slot length in minutes
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    system_prompt = f"""You are a mood-based insight analyzer. Analyze the user's mood and provide emotion-aware insights in exactly three categories: great_job, need_attention, and opportunity.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Output Format:**
You MUST respond with a valid JSON object containing exactly three fields:
{{
  "great_job": "Summary of achievements and positive patterns",
  "need_attention": "Summary of areas needing attention and actionable guidance",
  "opportunity": "Summary of opportunities and suggestions for optimal actions from current time {current_time} onward"
}}

**Mood-Based Rules:**
1. **Mood: Terrible/Sad**: Need attention → Offer gentle support, reduce load, low-effort options
   - If weekend → Suggest social connection or gentle activities
   - If back-to-back events → Suggest micro-breaks
   - If calendar_events_next3h heavy → Suggest lighter schedule
2. **Mood: Okay**: Opportunity → Suggest meaningful tasks if in active window, light progress if outside
   - If weekend morning → Suggest health activities
3. **Mood: Happy/Awesome**: Great job → Celebrate and channel momentum (but avoid overcommitment)
4. **Mood + Abnormal HR**: If mood=terrible AND (HR<40 OR HR>180) → Safety override

**Categories:**
- **great_job**: Positive mood reinforcement (e.g., "Love your mood – keep the momentum")
- **need_attention**: Mood support needed (e.g., "Looks like today is tough. Want a 3-minute reset?")
- **opportunity**: Mood-based opportunities (e.g., "Nice – want to use this slot for something meaningful?")

{_build_productivity_context_prompt(productivity_context)}

IMPORTANT TIME RULES:
- DO NOT mention exact times, hours, or minutes (e.g. "5 PM", "2 hours", "60 mins")
- DO NOT restate or paraphrase current_time
- DO NOT use real-time urgency language (e.g. "now", "right now", "perfect time")
- Describe time context only in qualitative terms (e.g. "later in the day", "approaching bedtime", "during a free window")
- When combining multiple signals, prioritize relevance over precision
- Frame mood-based suggestions as supportive and flexible, not commands

IMPORTANT:
- Three JSON keys are required; use empty string for categories that do not apply (do not invent filler)
- Be concise - each category 1-2 sentences maximum
- Use natural, supportive, empathetic language
- Focus on emotion-aware insights
- Respond with valid JSON only
{important_language_prompt(language_name)}"""

    # Format calendar_events_next3h
    events_next3h_str = "No events"
    if calendar_events_next3h:
        events_list = []
        for event in calendar_events_next3h[:5]:  # Limit to 5 events
            summary = event.get("summary", "Untitled")
            start_time = event.get("startTime", "")
            description = str(event.get("description", "") or "").strip()
            desc_hint = f" | Description: {description[:120]}" if description else ""
            events_list.append(f"- {summary} ({start_time}){desc_hint}")
        events_next3h_str = "\n".join(events_list) if events_list else "No events"

    # Build conditional health data lines
    health_lines = []
    if latest_heart_rate is not None:
        health_lines.append(f"- Latest Heart Rate: {latest_heart_rate}")
    if sleep_lastnight is not None:
        health_lines.append(f"- Sleep Last Night: {sleep_lastnight} hours")
    if sleep_goal is not None:
        health_lines.append(f"- Sleep Goal: {sleep_goal} hours")
    health_section = "\n".join(health_lines) if health_lines else ""

    # Build conditional context lines
    context_lines = []
    if in_active_window is not None:
        context_lines.append(f"- In Active Window: {in_active_window}")
    if free_slot_length is not None:
        context_lines.append(f"- Free Slot Length: {free_slot_length} minutes")
    if is_weekend is not None:
        context_lines.append(f"- Is Weekend: {is_weekend}")
    if current_hour is not None:
        context_lines.append(f"- Current Hour: {current_hour}")
    if back_to_back_count is not None:
        context_lines.append(f"- Back-to-back Count: {back_to_back_count}")
    context_section = "\n".join(context_lines) if context_lines else ""

    mood_line = f"**Current Mood:** {mood}\n" if mood else ""
    events_section = (
        f"**Upcoming Events (Next 3 Hours):**\n{events_next3h_str}"
        if events_next3h_str
        else ""
    )

    user_prompt = f"""**Mood-Based Analysis**

{mood_line}**Health Data:**
{health_section if health_section else ""}

**Current Context:**
{context_section if context_section else ""}

{events_section if events_section else ""}

**Mood Analysis:**
- If mood is "terrible" or "sad" AND calendar_events_next3h shows heavy load → Suggest lighter schedule
- If mood is "terrible" or "sad" AND back_to_back_count >= 3 → Suggest micro-breaks
- If mood is "okay" AND in_active_window AND free_slot_length >= 45 → Suggest meaningful tasks
- Mood: Terrible/Sad (need attention)
  - Weekend context (if weekend, suggest social/gentle activities)
  - Back-to-back meetings impact (if heavy, suggest breaks)
  - Upcoming events load (if heavy, suggest lighter schedule)
- Mood: Okay (opportunity)
  - Weekend morning context (if weekend morning, suggest health activities)
- Mood: Happy/Awesome (great job)

{important_language_prompt(language_name)}

Return JSON with three categories:
{{
  "great_job": "Positive mood reinforcement",
  "need_attention": "Mood support needed",
  "opportunity": "Mood-based opportunities from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


def get_inbox_reminders_overall_insight_prompt(
    free_slot_length: Optional[int] = None,
    in_active_window: Optional[bool] = None,
    current_time: str = "",
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for inbox reminders overall insight (Group 8).

    Args:
        free_slot_length: Current free slot length in minutes
        in_active_window: Whether in active hours window
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    system_prompt = f"""You are an inbox and reminders analyzer. Analyze the user's inbox and reminders status and provide insights in exactly three categories: great_job, need_attention, and opportunity.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Output Format:**
You MUST respond with a valid JSON object containing exactly three fields:
{{
  "great_job": "Summary of achievements and positive patterns",
  "need_attention": "Summary of areas needing attention and actionable guidance",
  "opportunity": "Summary of opportunities and suggestions for optimal actions from current time {current_time} onward"
}}

**Inbox & Reminders Rules:**
1. **Inbox Overload**: If inbox_flagged_count >= threshold AND no_admin_block_next24h → Schedule inbox batch
2. **Inbox High + Active Window**: If inActiveWindow AND inbox_flagged_count >= threshold → Defer inbox, protect focus
3. **Reminders Overdue**: If reminders_overdue_count >= threshold → Suggest triage
4. **Inbox Hygiene**: If inbox_flagged_count decreased significantly → Celebrate achievement

**Categories:**
- **great_job**: Inbox/reminders achievements (e.g., "Nice work – inbox is lighter")
- **need_attention**: Inbox/reminders issues (e.g., "A few emails need action – block 30 mins to clear them?")
- **opportunity**: Inbox/reminders opportunities (e.g., "Good time for a quick admin batch")

{_build_productivity_context_prompt(productivity_context)}

IMPORTANT TIME RULES:
- DO NOT mention exact times, hours, or minutes (e.g. "5 PM", "2 hours", "60 mins")
- DO NOT restate or paraphrase current_time
- DO NOT use real-time urgency language (e.g. "now", "right now", "perfect time")
- Describe time context only in qualitative terms
- When combining multiple signals, prioritize relevance over precision

IMPORTANT:
- Three JSON keys are required; use empty string for categories that do not apply (do not invent filler)
- Be concise - each category 1-2 sentences maximum
- Use natural, conversational language
- Focus on actionable insights
- Respond with valid JSON only
{important_language_prompt(language_name)}"""

    user_prompt = f"""**Inbox & Reminders Analysis**

**Current Context:**
- Free Slot Length: {free_slot_length} minutes (if applicable)
- In Active Window: {in_active_window}

Analyze inbox and reminders:
- Inbox overload
- Reminders overdue
- Inbox hygiene achievements

{important_language_prompt(language_name)}

Return JSON with three categories:
{{
  "great_job": "Inbox/reminders achievements",
  "need_attention": "Inbox/reminders issues",
  "opportunity": "Inbox/reminders opportunities from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


def get_weekend_lifestyle_overall_insight_prompt(
    calendar_data: str = "",
    steps_today: Optional[int] = None,
    steps_goal: Optional[float] = None,
    free_slot_length: Optional[int] = None,
    weekly_health_progress: Optional[Dict[str, Any]] = None,
    work_events_hours_weekend: Optional[float] = None,
    work_load_high: Optional[bool] = None,
    calendar_events_next7d: Optional[List[Dict[str, Any]]] = None,
    sleep_quality: Optional[str] = None,
    current_hour: Optional[int] = None,
    is_weekend: Optional[bool] = None,
    current_time: str = "",
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for weekend lifestyle overall insight (Group 9).

    Args:
        calendar_data: Formatted calendar events string
        steps_today: Today's step count
        steps_goal: Daily steps goal
        free_slot_length: Current free slot length in minutes
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    system_prompt = f"""You are a weekend and lifestyle analyzer. Analyze the user's weekend and lifestyle patterns and provide insights in exactly three categories: great_job, need_attention, and opportunity.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Output Format:**
You MUST respond with a valid JSON object containing exactly three fields:
{{
  "great_job": "Summary of achievements and positive patterns",
  "need_attention": "Summary of areas needing attention and actionable guidance",
  "opportunity": "Summary of opportunities and suggestions for optimal actions from current time {current_time} onward"
}}

**Weekend & Lifestyle Rules:**
1. **Weekend Health**: If weekend AND morning (current_hour < 12) AND slot >= 45 mins → Prioritize workout/outdoors/meal prep
2. **Weekend Balance**: If weekend AND steps high AND weekly_health_progress good → Reinforce lifestyle balance
3. **Weekend Burnout**: If weekend AND work_events_hours_weekend > threshold → Suggest recovery + move flexible work
4. **Weekend Social**: If weekend AND evening (current_hour >= 17) AND not inWindDown → Suggest social or hobby
5. **Weekend Work Load**: If weekend AND work_load_high → Suggest protecting recovery time
6. **Weekend Planning**: If weekend AND calendar_events_next7d available → Suggest week reset and planning
7. **No Upcoming Events**: If there are no upcoming events, provide flexible opportunity suggestions (social connection, enjoyment, reading, recovery, or weekly organization) tailored to context.

**Categories:**
- **great_job**: Weekend/lifestyle achievements (e.g., "Great balance this weekend – movement + connection")
- **need_attention**: Weekend/lifestyle issues (e.g., "Your weekend is work-heavy – protect recovery time")
- **opportunity**: Weekend/lifestyle opportunities (e.g., "Weekend morning is perfect for your health goals")

{_build_productivity_context_prompt(productivity_context)}

IMPORTANT TIME RULES:
- DO NOT mention exact times, hours, or minutes (e.g. "5 PM", "2 hours", "60 mins")
- DO NOT restate or paraphrase current_time
- DO NOT use real-time urgency language (e.g. "now", "right now", "perfect time")
- Describe time context only in qualitative terms (e.g. "later in the day", "approaching bedtime", "during a free window")
- When combining multiple signals, prioritize relevance over precision

IMPORTANT:
- RELATIVE TO CALENDAR EVENTS:
    - Pay attention to event titles, descriptions, and participants to better understand the context and intent behind each calendar event.
    - Use this contextual understanding to produce more relevant, coherent, and human-like insights.
- Three JSON keys are required; use empty string for categories that do not apply (do not invent filler)
- If no upcoming events are present, keep suggestions varied and natural; avoid rigid or repetitive "to-do list" phrasing.
- Be concise - each category 1-2 sentences maximum
- Use natural, conversational language
- Focus on lifestyle balance and wellbeing
- Respond with valid JSON only
{important_language_prompt(language_name)}"""

    # Format weekly_health_progress
    weekly_progress_str = None
    if weekly_health_progress:
        total_steps = weekly_health_progress.get("total_steps", 0)
        avg_daily = weekly_health_progress.get("avg_daily_steps", 0)
        days_met = weekly_health_progress.get("days_met_goal", 0)
        progress_pct = weekly_health_progress.get("progress_percentage", 0)
        weekly_progress_str = f"Total: {total_steps} steps | Avg: {avg_daily:.0f}/day | Days Met Goal: {days_met} | Progress: {progress_pct:.0f}%"

    # Format calendar_events_next7d
    events_next7d_summary = None
    if calendar_events_next7d:
        work_events = [
            e
            for e in calendar_events_next7d
            if any(
                k
                in (
                    f"{e.get('summary', '')} {e.get('description', '')} {e.get('location', '')}"
                ).lower()
                for k in [
                    "work",
                    "meeting",
                    "standup",
                    "sync",
                    "client",
                    "project",
                    "office",
                    "họp",
                ]
            )
        ]
        events_next7d_summary = f"Total: {len(calendar_events_next7d)} events | Work-related: {len(work_events)} events"

    morning = "Yes" if (current_hour and current_hour < 12) else "No"
    evening = "Yes" if (current_hour and current_hour >= 17) else "No"

    # Build conditional context lines
    context_lines = []
    if is_weekend is not None:
        context_lines.append(f"- Is Weekend: {is_weekend}")
    if current_hour is not None:
        context_lines.append(f"- Current Hour: {current_hour}")
        context_lines.append(f"- Morning (before 12:00): {morning}")
        context_lines.append(f"- Evening (after 17:00): {evening}")
    if free_slot_length is not None:
        context_lines.append(f"- Free Slot Length: {free_slot_length} minutes")
    context_section = "\n".join(context_lines) if context_lines else ""

    # Build conditional health data lines
    health_lines = []
    if steps_today is not None and steps_goal is not None:
        health_lines.append(f"- Steps Today: {steps_today} / Goal: {steps_goal}")
    elif steps_today is not None:
        health_lines.append(f"- Steps Today: {steps_today}")
    elif steps_goal is not None:
        health_lines.append(f"- Steps Goal: {steps_goal}")
    if weekly_progress_str:
        health_lines.append(f"- Weekly Health Progress: {weekly_progress_str}")
    health_section = "\n".join(health_lines) if health_lines else ""

    # Build conditional calendar metrics lines
    calendar_lines = []
    if work_events_hours_weekend is not None:
        calendar_lines.append(
            f"- Work Events Hours (Weekend): {work_events_hours_weekend} hours"
        )
    if work_load_high is not None:
        calendar_lines.append(f"- Work Load High: {work_load_high}")
    if events_next7d_summary:
        calendar_lines.append(f"- Events Next 7 Days: {events_next7d_summary}")
    calendar_section = "\n".join(calendar_lines) if calendar_lines else ""

    user_prompt = f"""**Weekend & Lifestyle Analysis**

**Current Context:**
{context_section if context_section else ""}

**Health Data:**
{health_section if health_section else ""}

**Calendar Metrics:**
{calendar_section if calendar_section else ""}

**Calendar Events:**
{calendar_data}

**Weekend Analysis:**
- If weekly_health_progress shows good progress → Celebrate and suggest maintaining balance
- If work_events_hours_weekend > 8 hours → Suggest protecting recovery time
- If work_load_high AND is_weekend → Suggest prioritizing rest and recovery

Analyze weekend and lifestyle:
- Weekend health opportunities (morning workout/outdoors/meal prep)
- Weekend balance (steps + weekly health progress)
- Weekend burnout (work events hours)
- Weekend work load (if high, protect recovery)
- Weekend planning (week reset, next 7 days events)
- Weekend social (evening activities)
- If no upcoming events, emphasize flexible, enjoyable, and non-rigid options.

{important_language_prompt(language_name)}

Return JSON with three categories:
{{
  "great_job": "Weekend/lifestyle achievements",
  "need_attention": "Weekend/lifestyle issues",
  "opportunity": "Weekend/lifestyle opportunities from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


def get_evening_after_work_overall_insight_prompt(
    calendar_data: str = "",
    time_to_bedtime: Optional[int] = None,
    free_slot_length: Optional[int] = None,
    active_hours_end: Optional[str] = None,
    wind_down_buffer_mins: Optional[int] = None,
    bedtime_start: Optional[str] = None,
    sleep_quality: Optional[str] = None,
    in_active_window: Optional[bool] = None,
    current_hour: Optional[int] = None,
    current_time: str = "",
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for evening after work overall insight (Group 10).

    Args:
        calendar_data: Formatted calendar events string
        time_to_bedtime: Minutes until bedtime
        free_slot_length: Current free slot length in minutes
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    system_prompt = f"""You are an evening and after work analyzer. Analyze the user's evening and after work patterns and provide insights in exactly three categories: great_job, need_attention, and opportunity.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Output Format:**
You MUST respond with a valid JSON object containing exactly three fields:
{{
  "great_job": "Summary of achievements and positive patterns",
  "need_attention": "Summary of areas needing attention and actionable guidance",
  "opportunity": "Summary of opportunities and suggestions for optimal actions from current time {current_time} onward"
}}

**Evening & After Work Rules:**
1. **After Work Free**: If current_time > active_hours_end AND slot >= 30 mins AND not inWindDown → Suggest recovery/fitness/social/hobbies
2. **Evening Boundaries**: If after_work_hours AND frequent_late_work_blocks → Suggest boundary and move work to active window
3. **Late-night Work**: If inWindDown (time_to_bedtime <= {wind_down_buffer_mins or 60} mins) AND work_activity_detected → Suggest stopping work + rescheduling
4. **Early Evening Meal**: If early_evening (17:00 <= current_hour <= 19:00) AND slot >= 30 mins → Suggest healthy meal plan
5. **In Wind-Down**: If time_to_bedtime <= {wind_down_buffer_mins or 60} mins → Already in wind-down, suggest calm activities

**Categories (multi-domain):**
- **great_job**: Concrete wins today (productivity/finance/steps met) from data — not vague praise if calendar shows rest issues.
- **need_attention**: Schedule affecting rest OR sleep debt (one domain). No step deficit nudging at night.
- **opportunity**: If in_wind_down or bedtime soon → calm wind-down only or ""; else light recharge (not tomorrow prep tonight).

{_build_productivity_context_prompt(productivity_context)}

IMPORTANT TIME RULES:
- DO NOT mention exact times, hours, or minutes (e.g. "5 PM", "2 hours", "60 mins")
- DO NOT restate or paraphrase current_time
- DO NOT use real-time urgency language (e.g. "now", "right now", "perfect time")
- Describe time context only in qualitative terms (e.g. "later in the day", "approaching bedtime", "during a free window")
- When combining multiple signals, prioritize relevance over precision

IMPORTANT:
- relative to calendar events (if have):
    - Pay attention to event titles, descriptions, and participants to better understand the context and intent behind each calendar event.
    - Use this contextual understanding to produce more relevant, coherent, and human-like insights.
- Three JSON keys are required; use empty string for categories that do not apply (do not invent filler)
- Be concise - each category 1-2 sentences maximum
- Use natural, conversational language
- Focus on work-life balance and boundaries
- Respond with valid JSON only
{important_language_prompt(language_name)}"""

    after_work = (
        "Yes"
        if (
            active_hours_end
            and current_hour
            and current_hour >= int(active_hours_end.split(":")[0])
            if active_hours_end
            else False
        )
        else "No"
    )
    early_evening = "Yes" if (current_hour and 17 <= current_hour <= 19) else "No"
    in_wind_down = (
        "Yes"
        if (
            time_to_bedtime is not None
            and time_to_bedtime <= (wind_down_buffer_mins or 60)
        )
        else "No"
    )

    # Build conditional evening context lines
    evening_lines = []
    if sleep_quality:
        evening_lines.append(
            f"- Sleep Quality: {sleep_quality} (affects evening capacity: very_high, high, ok, low, very_low)"
        )
    if in_active_window is not None:
        evening_lines.append(f"- In Active Window: {in_active_window}")
    if after_work == "Yes":
        evening_lines.append(f"- After Work Hours: {after_work}")
    if in_wind_down == "Yes":
        evening_lines.append(
            f"- In Wind-Down: {in_wind_down} (time_to_bedtime <= {wind_down_buffer_mins or 60} mins)"
        )
    evening_section = "\n".join(evening_lines) if evening_lines else ""

    # Build conditional current context lines
    context_lines = []
    if active_hours_end:
        context_lines.append(f"- Active Hours End: {active_hours_end}")
    if bedtime_start:
        context_lines.append(f"- Bedtime Start: {bedtime_start}")
    if wind_down_buffer_mins is not None:
        context_lines.append(f"- Wind-Down Buffer: {wind_down_buffer_mins} minutes")
    if time_to_bedtime is not None:
        context_lines.append(f"- Time to Bedtime: {time_to_bedtime} minutes")
    if free_slot_length is not None:
        context_lines.append(f"- Free Slot Length: {free_slot_length} minutes")
    if current_hour is not None:
        context_lines.append(f"- Current Hour: {current_hour}")
    if early_evening == "Yes":
        context_lines.append(f"- Early Evening (17:00-19:00): {early_evening}")
    context_section = "\n".join(context_lines) if context_lines else ""

    user_prompt = f"""**Evening & After Work Analysis**

**Evening Context:**
{evening_section if evening_section else ""}

**Current Context:**
{context_section if context_section else ""}

**Calendar Events:**
{calendar_data}

Analyze evening and after work:
- After work free time (if after active hours end)
- Early evening meal opportunity (17:00-19:00)
- Wind-down detection (if approaching bedtime)
- Evening boundaries
- Late-night work detected

{important_language_prompt(language_name)}

Return JSON (use "" for unused categories)."""

    return system_prompt, user_prompt


def get_contextual_overall_insight_prompt(
    calendar_data: str = "",
    health_data: Optional[Dict[str, Dict[str, Any]]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
    steps_today: Optional[int] = None,
    steps_goal: Optional[float] = None,
    sleep_lastnight: Optional[float] = None,
    sleep_goal: Optional[float] = None,
    time_to_next_event: Optional[int] = None,
    free_slot_length: Optional[int] = None,
    time_to_bedtime: Optional[int] = None,
    in_active_window: Optional[bool] = None,
    current_time: str = "",
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for contextual overall insight (Group 11 - default).

    Args:
        calendar_data: Formatted calendar events string
        health_data: Health data dict
        user_profile: User profile dict
        steps_today: Today's step count
        steps_goal: Daily steps goal
        sleep_lastnight: Sleep hours last night
        sleep_goal: Daily sleep goal
        time_to_next_event: Minutes until next event
        free_slot_length: Current free slot length in minutes
        time_to_bedtime: Minutes until bedtime
        in_active_window: Whether in active hours window
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    system_prompt = f"""You are a contextual productivity advisor. Generate personalized suggestions based on all available context and provide insights in exactly three categories: great_job, need_attention, and opportunity.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Output Format:**
You MUST respond with a valid JSON object containing exactly three fields:
{{
  "great_job": "Summary of achievements and positive patterns",
  "need_attention": "Summary of areas needing attention and actionable guidance",
  "opportunity": "Summary of opportunities and suggestions for optimal actions from current time {current_time} onward"
}}

**AI SUGGESTIONS Rules (from AI Logics for Overall Insights):**
- Health catch-up (catch-up window, behind on goals)
- Wind-down (1 hours before bedtime)
- Bedtime window (sleep prep)
- Morning start (day planning)
- Active focus window (deep work)
- Recovery/break needs
- Schedule optimization
- And 30+ other contextual suggestions

**Categories:**
- **great_job**: Contextual positive observations
- **need_attention**: Contextual areas needing attention
- **opportunity**: Contextual opportunities and suggestions

{_build_productivity_context_prompt(productivity_context)}

IMPORTANT TIME RULES:
- DO NOT mention exact times, hours, or minutes (e.g. "5 PM", "2 hours", "60 mins")
- DO NOT restate or paraphrase current_time
- DO NOT use real-time urgency language (e.g. "now", "right now", "perfect time")
- Describe time context only in qualitative terms (e.g. "later in the day", "approaching bedtime", "during a free window")
- When combining multiple signals, prioritize relevance over precision
- This is a cached, non-real-time contextual suggestion
- DO NOT suggest any task, slot, or action in the past to need_attention or opportunity fields
- DO NOT suggest anything that overlaps with existing calendar events to need_attention or opportunity fields
- DO NOT suggest a task longer than the available free slot to need_attention or opportunity fields
- DO NOT suggest deep work, workouts, or cognitively heavy tasks within 20 minutes before the next meeting to need_attention or opportunity fields
- DO NOT squeeze new meetings or tasks between consecutive back-to-back meetings

IMPORTANT
- relative to calendar events (if have):
    - Pay attention to event titles, descriptions, and participants to better understand the context and intent behind each calendar event.
    - Use this contextual understanding to produce more relevant, coherent, and human-like insights.
- Three JSON keys are required; use empty string for categories that do not apply (do not invent filler)
- Be concise - each category 1-2 sentences maximum
- Use natural, conversational language
- Follow all NOT DO rules strictly
- Focus on most relevant suggestion for current context
- Respond with valid JSON only
{important_language_prompt(language_name)}"""

    # Build conditional context lines
    context_lines = []
    if time_to_next_event is not None:
        context_lines.append(f"- Time to Next Event: {time_to_next_event} minutes")
    if free_slot_length is not None:
        context_lines.append(f"- Free Slot Length: {free_slot_length} minutes")
    if time_to_bedtime is not None:
        context_lines.append(f"- Time to Bedtime: {time_to_bedtime} minutes")
    if in_active_window is not None:
        context_lines.append(f"- In Active Window: {in_active_window}")
    context_section = "\n".join(context_lines) if context_lines else ""

    # Build conditional health data lines
    health_lines = []
    if steps_today is not None and steps_goal is not None:
        health_lines.append(f"- Steps Today: {steps_today} / Goal: {steps_goal}")
    elif steps_today is not None:
        health_lines.append(f"- Steps Today: {steps_today}")
    elif steps_goal is not None:
        health_lines.append(f"- Steps Goal: {steps_goal}")
    if sleep_lastnight is not None and sleep_goal is not None:
        health_lines.append(
            f"- Sleep Last Night: {sleep_lastnight} hours / Goal: {sleep_goal} hours"
        )
    elif sleep_lastnight is not None:
        health_lines.append(f"- Sleep Last Night: {sleep_lastnight} hours")
    elif sleep_goal is not None:
        health_lines.append(f"- Sleep Goal: {sleep_goal} hours")
    health_section = "\n".join(health_lines) if health_lines else ""

    user_prompt = f"""**Contextual Suggestion Analysis**

**Current Context:**
{context_section if context_section else ""}

**Calendar Events:**
{calendar_data}

**Health Data:**
{health_section if health_section else ""}

Analyze overall context and provide general insights.

{important_language_prompt(language_name)}

Return JSON with three categories:
{{
  "great_job": "Contextual positive observations",
  "need_attention": "Contextual areas needing attention",
  "opportunity": "Contextual opportunities and suggestions from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


def get_overall_insight_prompt(
    group: str,
    calendar_data: str = "",
    health_data: Optional[Dict[str, Dict[str, Any]]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
    current_time: str = "",
    timezone: Optional[str] = None,
    steps_today: Optional[int] = None,
    steps_goal: Optional[float] = None,
    sleep_lastnight: Optional[float] = None,
    sleep_goal: Optional[float] = None,
    sleep_quality: Optional[str] = None,
    latest_heart_rate: Optional[int] = None,
    resting_heart_rate: Optional[int] = None,
    baseline_resting_hr: Optional[float] = None,
    bedtime_streak: Optional[int] = None,
    sleep_last3nights: Optional[List[Optional[float]]] = None,
    steps_streak: Optional[int] = None,
    weekly_health_progress: Optional[Dict[str, Any]] = None,
    daily_health_progress: Optional[Dict[str, Any]] = None,
    wind_down_buffer_mins: Optional[int] = None,
    time_to_next_event: Optional[int] = None,
    free_slot_length: Optional[int] = None,
    time_to_bedtime: Optional[int] = None,
    in_active_window: Optional[bool] = None,
    is_weekend: Optional[bool] = None,
    current_hour: Optional[int] = None,
    bedtime_start: Optional[str] = None,
    bedtime_end: Optional[str] = None,
    active_hours_start: Optional[str] = None,
    active_hours_end: Optional[str] = None,
    calendar_density: Optional[float] = None,
    back_to_back_count: Optional[int] = None,
    meeting_minutes: Optional[int] = None,
    continuous_events_minutes: Optional[int] = None,
    work_events_hours_weekend: Optional[float] = None,
    work_load_high: Optional[bool] = None,
    calendar_events_next7d: Optional[List[Dict[str, Any]]] = None,
    calendar_events_next48h: Optional[List[Dict[str, Any]]] = None,
    calendar_events_next3h: Optional[List[Dict[str, Any]]] = None,
    mood: Optional[str] = None,
    mood_not_logged_today: bool = False,
    last_wake_time: Optional[str] = None,
    first_sleep_time: Optional[str] = None,
    language: Optional[str] = None,
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for overall insight analysis based on detected group.

    This is a router function that calls the appropriate group-specific prompt function.

    Args:
        group: Detected group name
        calendar_data: Formatted calendar events string
        health_data: Health data dict
        user_profile: User profile dict
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        steps_today: Today's step count
        steps_goal: Daily steps goal
        sleep_lastnight: Sleep hours last night
        sleep_goal: Daily sleep goal
        sleep_quality: Sleep quality level (very_high, high, ok, low, very_low)
        latest_heart_rate: Latest heart rate
        resting_heart_rate: Resting heart rate
        time_to_next_event: Minutes until next event
        free_slot_length: Current free slot length in minutes
        time_to_bedtime: Minutes until bedtime
        in_active_window: Whether in active hours window
        bedtime_start: Bedtime window start
        bedtime_end: Bedtime window end
        active_hours_start: Active hours start
        active_hours_end: Active hours end
        mood: Current mood ("terrible", "sad", "okay", "happy", "amazing")
        last_wake_time: Last wake time in ISO format (YYYY-MM-DDTHH:MM:SS)
        first_sleep_time: First sleep time in ISO format (YYYY-MM-DDTHH:MM:SS)
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    if group == "safety_risk":
        system_prompt, user_prompt = get_safety_risk_overall_insight_prompt(
            latest_heart_rate=latest_heart_rate,
            resting_heart_rate=resting_heart_rate,
            baseline_resting_hr=baseline_resting_hr,
            sleep_lastnight=sleep_lastnight,
            sleep_goal=sleep_goal,
            sleep_quality=sleep_quality,
            sleep_last3nights=sleep_last3nights,
            calendar_density=calendar_density,
            back_to_back_count=back_to_back_count,
            meeting_minutes=meeting_minutes,
            calendar_events_next3h=calendar_events_next3h,
            current_time=current_time,
            timezone=timezone,
            language=language,
            productivity_context=productivity_context,
        )

    elif group == "wind_down_window":
        system_prompt, user_prompt = get_wind_down_window_overall_insight_prompt(
            calendar_data=calendar_data,
            bedtime_start=bedtime_start,
            bedtime_end=bedtime_end,
            time_to_bedtime=time_to_bedtime,
            wind_down_buffer_mins=wind_down_buffer_mins,
            bedtime_streak=bedtime_streak,
            sleep_quality=sleep_quality,
            sleep_last3nights=sleep_last3nights,
            sleep_goal=sleep_goal,
            sleep_lastnight=sleep_lastnight,
            steps_today=steps_today,
            steps_goal=steps_goal,
            current_time=current_time,
            timezone=timezone,
            language=language,
            productivity_context=productivity_context,
        )

    elif group == "bedtime_window":
        system_prompt, user_prompt = get_bedtime_window_overall_insight_prompt(
            calendar_data=calendar_data,
            bedtime_start=bedtime_start,
            bedtime_end=bedtime_end,
            bedtime_streak=bedtime_streak,
            sleep_last3nights=sleep_last3nights,
            sleep_lastnight=sleep_lastnight,
            last_wake_time=last_wake_time,
            first_sleep_time=first_sleep_time,
            sleep_goal=sleep_goal,
            sleep_quality=sleep_quality,
            steps_today=steps_today,
            steps_goal=steps_goal,
            current_time=current_time,
            timezone=timezone,
            language=language,
            productivity_context=productivity_context,
        )

    elif group == "morning_window":
        system_prompt, user_prompt = get_morning_window_overall_insight_prompt(
            calendar_data=calendar_data,
            bedtime_end=bedtime_end,
            active_hours_start=active_hours_start,
            time_to_next_event=time_to_next_event,
            sleep_lastnight=sleep_lastnight,
            sleep_goal=sleep_goal,
            sleep_quality=sleep_quality,
            weekly_health_progress=weekly_health_progress,
            free_slot_length=free_slot_length,
            current_time=current_time,
            timezone=timezone,
            language=language,
            productivity_context=productivity_context,
        )

    elif group == "active_window":
        system_prompt, user_prompt = get_active_window_overall_insight_prompt(
            calendar_data=calendar_data,
            time_to_next_event=time_to_next_event,
            free_slot_length=free_slot_length,
            active_hours_start=active_hours_start,
            active_hours_end=active_hours_end,
            bedtime_start=bedtime_start,
            wind_down_buffer_mins=wind_down_buffer_mins,
            sleep_quality=sleep_quality,
            calendar_density=calendar_density,
            back_to_back_count=back_to_back_count,
            current_time=current_time,
            timezone=timezone,
            language=language,
            productivity_context=productivity_context,
        )

    elif group == "evening_flexible":
        system_prompt, user_prompt = get_evening_flexible_overall_insight_prompt(
            calendar_data=calendar_data,
            bedtime_start=bedtime_start,
            active_hours_end=active_hours_end,
            wind_down_buffer_mins=wind_down_buffer_mins,
            free_slot_length=free_slot_length,
            time_to_bedtime=time_to_bedtime,
            current_hour=current_hour,
            mood=mood,
            mood_not_logged_today=mood_not_logged_today,
            current_time=current_time,
            timezone=timezone,
            language=language,
            productivity_context=productivity_context,
        )

    elif group == "calendar_pattern":
        system_prompt, user_prompt = get_calendar_pattern_overall_insight_prompt(
            calendar_data=calendar_data,
            calendar_density=calendar_density,
            back_to_back_count=back_to_back_count,
            meeting_minutes=meeting_minutes,
            sleep_quality=sleep_quality,
            weekly_health_progress=weekly_health_progress,
            active_hours_start=active_hours_start,
            active_hours_end=active_hours_end,
            calendar_events_next48h=calendar_events_next48h,
            current_time=current_time,
            timezone=timezone,
            language=language,
            productivity_context=productivity_context,
        )

    elif group == "health_goal":
        system_prompt, user_prompt = get_health_goal_overall_insight_prompt(
            steps_today=steps_today,
            steps_goal=steps_goal,
            steps_streak=steps_streak,
            sleep_lastnight=sleep_lastnight,
            sleep_goal=sleep_goal,
            sleep_quality=sleep_quality,
            weekly_health_progress=weekly_health_progress,
            daily_health_progress=daily_health_progress,
            sleep_last3nights=sleep_last3nights,
            free_slot_length=free_slot_length,
            time_to_bedtime=time_to_bedtime,
            current_hour=current_hour,
            is_weekend=is_weekend,
            current_time=current_time,
            timezone=timezone,
            language=language,
            productivity_context=productivity_context,
        )

    elif group == "recovery_break":
        system_prompt, user_prompt = get_recovery_break_overall_insight_prompt(
            calendar_data=calendar_data,
            time_to_next_event=time_to_next_event,
            free_slot_length=free_slot_length,
            continuous_events_minutes=continuous_events_minutes,
            back_to_back_count=back_to_back_count,
            meeting_minutes=meeting_minutes,
            calendar_density=calendar_density,
            sleep_quality=sleep_quality,
            is_weekend=is_weekend,
            current_time=current_time,
            timezone=timezone,
            language=language,
            productivity_context=productivity_context,
        )

    elif group == "task_type":
        system_prompt, user_prompt = get_task_type_overall_insight_prompt(
            calendar_data=calendar_data,
            free_slot_length=free_slot_length,
            time_to_next_event=time_to_next_event,
            time_to_bedtime=time_to_bedtime,
            in_active_window=in_active_window,
            current_time=current_time,
            timezone=timezone,
            language=language,
            productivity_context=productivity_context,
        )

    elif group == "mood_based":
        system_prompt, user_prompt = get_mood_based_overall_insight_prompt(
            mood=mood,
            latest_heart_rate=latest_heart_rate,
            sleep_lastnight=sleep_lastnight,
            sleep_goal=sleep_goal,
            in_active_window=in_active_window,
            free_slot_length=free_slot_length,
            is_weekend=is_weekend,
            current_hour=current_hour,
            back_to_back_count=back_to_back_count,
            calendar_events_next3h=calendar_events_next3h,
            current_time=current_time,
            timezone=timezone,
            language=language,
            productivity_context=productivity_context,
        )

    elif group == "inbox_reminders":
        system_prompt, user_prompt = get_inbox_reminders_overall_insight_prompt(
            free_slot_length=free_slot_length,
            in_active_window=in_active_window,
            current_time=current_time,
            timezone=timezone,
            language=language,
            productivity_context=productivity_context,
        )

    elif group == "weekend_lifestyle":
        system_prompt, user_prompt = get_weekend_lifestyle_overall_insight_prompt(
            calendar_data=calendar_data,
            steps_today=steps_today,
            steps_goal=steps_goal,
            free_slot_length=free_slot_length,
            weekly_health_progress=weekly_health_progress,
            work_events_hours_weekend=work_events_hours_weekend,
            work_load_high=work_load_high,
            calendar_events_next7d=calendar_events_next7d,
            sleep_quality=sleep_quality,
            current_hour=current_hour,
            is_weekend=is_weekend,
            current_time=current_time,
            timezone=timezone,
            language=language,
            productivity_context=productivity_context,
        )

    elif group == "evening_after_work":
        system_prompt, user_prompt = get_evening_after_work_overall_insight_prompt(
            calendar_data=calendar_data,
            time_to_bedtime=time_to_bedtime,
            free_slot_length=free_slot_length,
            active_hours_end=active_hours_end,
            wind_down_buffer_mins=wind_down_buffer_mins,
            bedtime_start=bedtime_start,
            sleep_quality=sleep_quality,
            in_active_window=in_active_window,
            current_hour=current_hour,
            current_time=current_time,
            timezone=timezone,
            language=language,
            productivity_context=productivity_context,
        )

    else:  # contextual_suggestion (default)
        system_prompt, user_prompt = get_contextual_overall_insight_prompt(
            calendar_data=calendar_data,
            health_data=health_data,
            user_profile=user_profile,
            steps_today=steps_today,
            steps_goal=steps_goal,
            sleep_lastnight=sleep_lastnight,
            sleep_goal=sleep_goal,
            time_to_next_event=time_to_next_event,
            free_slot_length=free_slot_length,
            time_to_bedtime=time_to_bedtime,
            in_active_window=in_active_window,
            current_time=current_time,
            timezone=timezone,
            language=language,
            productivity_context=productivity_context,
        )

    system_prompt = system_prompt + "\n" + _get_shared_overall_rules_block(group)
    # Inject universal reasoning block for all groups
    reasoning = _overall_block(group)
    if reasoning:
        system_prompt = system_prompt + "\n\n" + reasoning
    return system_prompt, user_prompt
