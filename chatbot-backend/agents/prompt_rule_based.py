"""Rule-based productivity insight prompts"""

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
from agents.prompt_insight_reasoning import format_for_point12 as _insight_block
from services.executor.constant import HealthDataConstants


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


def _get_bedtime_window_no_data_prompt(
    calendar_data: str,
    current_time: str,
    timezone: Optional[str] = None,
    bedtime_start: Optional[str] = None,
    bedtime_end: Optional[str] = None,
    language: str = "English",
    overall_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for bedtime window when sleep timing data is not available.

    This handles the case when user is not wearing a device or hasn't slept yet.
    """
    system_prompt = f"""You are a bedtime window analyzer. The user is in their bedtime window, but sleep timing data is not available (they may not be wearing their device or haven't slept yet).

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language}

**User Schedule:**
- Bedtime Window: {bedtime_start or 'Not specified'} to {bedtime_end or 'Not specified'}

**Context:**
- Sleep timing data (last wake time, first sleep time) is not available
- This could mean: user is not wearing their wearable device, device is not synced, or user hasn't slept yet
- DO NOT make assumptions about sleep patterns or timing based on unavailable data
- However, you CAN analyze the current time's position within the bedtime window

**CRITICAL ANALYSIS REQUIREMENT:**
You MUST analyze the current time's position within the bedtime window to determine:
1. **EARLY SLEEP PHASE** (near bedtime_start): User is just entering sleep window → Focus on sleep preparation, winding down, transitioning to rest
2. **MIDDLE SLEEP PHASE** (middle of bedtime window): User is likely in deep sleep → Focus on maintaining sleep quality, avoiding disruptions
3. **LATE SLEEP PHASE** (near bedtime_end): User is approaching wake time → Focus on gentle awakening, morning preparation, maintaining rest until wake time

**Bedtime Window Analysis Rules:**
- Calculate the position of current_time relative to bedtime_start and bedtime_end
- Consider wrap-around cases (e.g., bedtime_start = 22:00, bedtime_end = 08:00 means sleep spans midnight)
- Determine if current_time is:
  - Within first 25% of sleep window → EARLY SLEEP PHASE
  - Within middle 50% of sleep window → MIDDLE SLEEP PHASE
  - Within last 25% of sleep window → LATE SLEEP PHASE

**Point 1:** Identify bedtime context with sleep phase analysis (1-2 sentences)
- Specify if user is at the beginning, middle, or end of sleep window
- Describe the sleep phase context (entering sleep, deep sleep, approaching wake)
- Acknowledge that sleep timing data is not available

**Point 2:** Provide sleep phase-appropriate suggestion (1-2 sentences)
- **Early phase**: Sleep preparation, winding down, relaxation techniques
- **Middle phase**: Sleep maintenance, avoiding disruptions, deep rest
- **Late phase**: Gentle awakening preparation, maintaining rest until wake time

{_build_overall_context_prompt(overall_context)}

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
- Be concise - each point 1-2 sentences maximum
- Use natural, conversational language
- Acknowledge data limitation without being technical
- Focus on sleep phase analysis and appropriate suggestions
- Respond with valid JSON only

Example (Early Phase):
{{
  "point1": "You're at the start of your sleep window - time to transition into rest",
  "point2": "Dim the lights, avoid screens, and try gentle stretching or meditation to help you wind down"
}}

Example (Middle Phase):
{{
  "point1": "You're in the middle of your sleep window - deep rest period",
  "point2": "Maintain your sleep environment and avoid any disruptions to preserve your rest quality"
}}

Example (Late Phase):
{{
  "point1": "You're approaching the end of your sleep window - gentle awakening ahead",
  "point2": "Maintain rest until wake time, and prepare for a gentle transition to your active day"
}}"""

    user_prompt = f"""Analyze the bedtime window context with sleep phase analysis (no sleep timing data available):

**Current Context:**
**Language:** Respond in {language}
- Current Time: {current_time}
- Bedtime Window: {bedtime_start} to {bedtime_end}

**Note:** Sleep timing data (last wake time, first sleep time) is not available.
This could mean:
- User is not wearing their wearable device
- Device is not synced
- User hasn't slept yet

**CRITICAL TASK:**
1. Calculate the position of current_time within the bedtime window
2. Determine sleep phase:
   - If current_time is within first 25% of window → EARLY SLEEP PHASE
   - If current_time is within middle 50% of window → MIDDLE SLEEP PHASE
   - If current_time is within last 25% of window → LATE SLEEP PHASE
3. Consider wrap-around cases (bedtime may span midnight)
4. Acknowledge that sleep timing data is not available (device may not be worn or synced)

**Calendar Events:**
{calendar_data}

Determine:
1. Bedtime context with sleep phase (early/middle/late sleep phase) - acknowledge that sleep timing data is not available
2. Sleep phase-appropriate suggestion (consider sleep phase position, but avoid assumptions about sleep patterns)

**Language:** Respond in {language}

Return JSON with point1 (context with sleep phase, acknowledging no sleep data) and point2 (phase-appropriate suggestion):
{{
  "point1": "bedtime context with sleep phase analysis, acknowledging no sleep timing data",
  "point2": "sleep phase-appropriate suggestion"
}}"""

    return system_prompt, user_prompt


def _format_sleep_last3nights(
    sleep_last3nights: Optional[List[Optional[float]]],
) -> str:
    """Format sleep_last3nights for display."""
    if not sleep_last3nights:
        return ""
    sleep_hours_list = [f"{h}" if h is not None else "N/A" for h in sleep_last3nights]
    return f"[{', '.join(sleep_hours_list)}] hours"


def _build_overall_context_prompt(
    overall_context: Optional[str] = "",
) -> str:
    """
    Build prompt section about overall context to ensure consistency.

    Args:
        overall_context: Overall insight context as string

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
    if not overall_context:
        return event_semantic_policy + title_policy + reminders_awareness

    # Convert to string if not already
    context_str = str(overall_context) if overall_context else ""

    # Format the context for prompt
    context_section = "\n\n**Overall Insight Context (MUST ALIGN WITH):**\n"
    context_section += (
        "The following overall insights have already been provided to the user. "
    )
    context_section += "Your productivity insights (point1, point2) Must be consistent and aligned, and should reinforce and complement these insights.\n\n"
    context_section += f"{context_str}\n"

    context_section += "\n**CRITICAL CONSISTENCY REQUIREMENTS:**\n"
    context_section += (
        "- DO NOT provide conflicting information with the overall insights above\n"
    )
    context_section += "- DO NOT contradict or create gaps between overall insights and productivity insights\n"
    context_section += "- DO NOT repeat the same information, but ensure your insights complement and align with overall insights\n"
    context_section += "- If overall insights mention specific facts (e.g., 'back-to-back events'), your productivity insights should acknowledge or build upon these facts, not contradict them\n"
    context_section += (
        "- Ensure point1 and point2 are consistent with the overall context\n"
    )
    context_section += "- If overall insights focus on a specific area, ensure your productivity insights don't create confusion by addressing the same area differently\n"

    return context_section + event_semantic_policy + title_policy + reminders_awareness


def get_wind_down_window_insight_prompt(
    calendar_data: str,
    current_time: str,
    timezone: Optional[str] = None,
    bedtime_start: Optional[str] = None,
    bedtime_end: Optional[str] = None,
    time_to_bedtime: Optional[int] = None,
    sleep_lastnight: Optional[float] = None,
    sleep_quality: Optional[str] = None,
    sleep_goal: Optional[float] = None,
    wind_down_buffer_mins: Optional[int] = None,
    language: Optional[str] = None,
    overall_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for wind-down window insights.

    Args:
        calendar_data: Formatted calendar events string
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        bedtime_start: Bedtime window start (e.g., "22:00")
        bedtime_end: Bedtime window end (e.g., "08:00")
        time_to_bedtime: Minutes until bedtime_start
        sleep_lastnight: Hours of sleep last night
        sleep_quality: Sleep quality level (very_high, high, ok, low, very_low)
        sleep_goal: Target sleep hours per night
        wind_down_buffer_mins: Wind-down buffer minutes (default: 60)
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    system_prompt = f"""You are a wind-down window analyzer. Analyze the user's wind-down period (within 1 hours before bedtime) and provide insights in exactly two points.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**User Schedule:**
- Bedtime Window: {bedtime_start or 'Not specified'} to {bedtime_end or 'Not specified'}

**Wind-Down Window Rules:**
- **WIND_DOWN_WINDOW**: Within 1 hours before bedtime_start → Suggest calm activities
- **LATE_WIND_DOWN**: Within 30 minutes before bedtime_start → Prioritize relaxation

**Point 1:** Identify wind-down context (1-2 sentences)
**Point 2:** Provide calm, relaxing suggestion (1-2 sentences)

{_build_overall_context_prompt(overall_context)}

IMPORTANT TIME RULES:
- DO NOT mention exact clock times (e.g. 8:58 AM, 9:00, current time)
- DO NOT repeat or paraphrase the provided current_time value
- DO NOT suggest starting new projects, big planning, or scheduling events
- DO NOT suggest inbox work, admin tasks, or stimulating media
- DO NOT suggest intense exercise or cognitively heavy tasks
- Use ONLY relative time expressions such as:
  "as bedtime approaches", "during your wind-down period",
  "before you rest", "as you prepare for sleep"
- Focus on calm, relaxing activities: gentle walk, reading, stretching, meditation, light prep for tomorrow

IMPORTANT:
- Be concise - each point 1-2 sentences maximum
- Use natural, conversational language
- Focus on relaxation and calm activities
- Respond with valid JSON only

{important_language_prompt(language_name)}

Example:
{{
  "point1": "You're in your wind-down window as bedtime is approaching",
  "point2": "Time to relax - try a gentle walk, reading, or 10-min tomorrow prep"
}}"""

    user_prompt = f"""Analyze the wind-down window context:

Current Time: {current_time}
Bedtime: {bedtime_start} to {bedtime_end}
Time to Bedtime: {time_to_bedtime} minutes
{_build_conditional_line("Sleep Last Night", sleep_lastnight, "hours")}
{_build_conditional_line("Sleep Quality", sleep_quality)}
{_build_conditional_line("Sleep Goal", sleep_goal, "hours")}
{_build_conditional_line("Wind-Down Buffer", wind_down_buffer_mins, "minutes")}

Calendar Events:
{calendar_data}

Determine:
1. Wind-down context (approaching bedtime, time to relax)
2. Calm, relaxing suggestion (consider sleep quality and recent sleep patterns)

{important_language_prompt(language_name)}

Return JSON with point1 (context) and point2 (suggestion):
{{
  "point1": "wind-down context summary",
  "point2": "calm, relaxing suggestion"
}}"""

    return system_prompt, user_prompt


def get_bedtime_window_insight_prompt(
    calendar_data: str,
    current_time: str,
    timezone: Optional[str] = None,
    bedtime_start: Optional[str] = None,
    bedtime_end: Optional[str] = None,
    sleep_lastnight: Optional[float] = None,
    sleep_quality: Optional[str] = None,
    sleep_last3nights: Optional[List[Optional[float]]] = None,
    bedtime_streak: Optional[int] = None,
    last_wake_time: Optional[str] = None,
    first_sleep_time: Optional[str] = None,
    sleep_goal: Optional[float] = None,
    language: Optional[str] = None,
    overall_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for bedtime window insights.

    Args:
        calendar_data: Formatted calendar events string
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        bedtime_start: Bedtime window start (e.g., "22:00")
        bedtime_end: Bedtime window end (e.g., "08:00")
        sleep_lastnight: Hours of sleep last night
        sleep_quality: Sleep quality level (very_high, high, ok, low, very_low)
        sleep_last3nights: List of sleep hours for last 3 nights
        bedtime_streak: Consecutive days sleeping within bedtime window
        last_wake_time: Last wake time in ISO format (YYYY-MM-DDTHH:MM:SS)
        first_sleep_time: First sleep time in ISO format (YYYY-MM-DDTHH:MM:SS)
        sleep_goal: Target sleep hours per night
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
        return _get_bedtime_window_no_data_prompt(
            calendar_data=calendar_data,
            current_time=current_time,
            timezone=timezone,
            bedtime_start=bedtime_start,
            bedtime_end=bedtime_end,
            language=language_name,
            overall_context=overall_context,
        )

    system_prompt = f"""You are a bedtime window analyzer. Analyze the user's bedtime period using sleep timing data and provide insights in exactly two points.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**User Schedule:**
- Bedtime Window: {bedtime_start or 'Not specified'} to {bedtime_end or 'Not specified'}
- Sleep Goal: {sleep_goal or 'Not set'} hours

**Sleep Timing Data:**
{_build_conditional_line("First Sleep Time", first_sleep_time, "time")}
{_build_conditional_line("Last Wake Time", last_wake_time, "time")}
{_build_conditional_line("Sleep Last Night", sleep_lastnight, "hours")}
{_build_conditional_line("Sleep Quality", sleep_quality)}

**CRITICAL ANALYSIS REQUIREMENT:**
You MUST analyze sleep patterns based on first_sleep_time and last_wake_time to determine:

1. **BEDTIME POSITION** (based on first_sleep_time relative to bedtime window):
   - **Early bedtime** (near bedtime_start): Circadian rhythm well aligned → Celebrate and suggest maintaining
   - **Mid bedtime** (middle of window): Good alignment → Suggest consistency
   - **Late bedtime** (near bedtime_end or after): Early signs of drift → Suggest gradual shift earlier

2. **SLEEP DURATION** (based on sleep_lastnight vs sleep_goal):
   - **Continuous sleep with short duration**: Sleep quality stable but insufficient → Suggest increasing duration or adding restorative breaks
   - **Adequate duration**: Good recovery time → Celebrate

3. **WAKE TIME PATTERNS** (based on last_wake_time):
   - **Waking earlier than usual**: Risk of low energy later → Suggest light morning workload
   - **Waking very early or near dawn**: Sleep cycle disrupted → Suggest prioritizing recovery and earlier bedtime

4. **CURRENT TIME POSITION** within bedtime window:
   - **EARLY SLEEP PHASE** (first 25%): Entering sleep → Focus on sleep preparation
   - **MIDDLE SLEEP PHASE** (middle 50%): Deep sleep → Focus on maintaining quality
   - **LATE SLEEP PHASE** (last 25%): Approaching wake → Focus on gentle awakening

**Point 1:** Identify sleep context summary (1-2 sentences)
- Analyze bedtime position (early/mid/late) based on first_sleep_time
- Analyze sleep duration (adequate vs short)
- Analyze wake time patterns (normal vs early)
- Describe current position in bedtime window

**Point 2:** Provide tailored suggestion based on analysis (1-2 sentences)
- **Early bedtime + adequate sleep**: Maintain current bedtime
- **Mid bedtime**: Keep sleep timing consistent
- **Late bedtime**: Gradually shift bedtime earlier (20-30 minutes)
- **Short duration**: Increase sleep duration or add restorative breaks
- **Early wake**: Light morning workload, avoid high-intensity activities
- **Very early wake**: Prioritize recovery, aim for earlier bedtime tonight

{_build_overall_context_prompt(overall_context)}

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
- Be concise - each point 1-2 sentences maximum
- Use natural, conversational language
- Focus on sleep phase analysis and appropriate suggestions
- Respond with valid JSON only

{important_language_prompt(language_name)}

Example (Early Phase):
{{
  "point1": "You're at the start of your sleep window - time to transition into rest",
  "point2": "Dim the lights, avoid screens, and try gentle stretching or meditation to help you wind down"
}}

Example (Middle Phase):
{{
  "point1": "You're in the middle of your sleep window - deep rest period",
  "point2": "Maintain your sleep environment and avoid any disruptions to preserve your rest quality"
}}

Example (Late Phase):
{{
  "point1": "You're approaching the end of your sleep window - gentle awakening ahead",
  "point2": "Maintain rest until wake time, and prepare for a gentle transition to your active day"
}}"""

    user_prompt = f"""Analyze the bedtime window context using sleep timing data:
**Sleep Timing Data:**
- First Sleep Time: {first_sleep_time}
- Last Wake Time: {last_wake_time}
{_build_conditional_line("Sleep Last Night", sleep_lastnight, "hours")}
{_build_conditional_line("Sleep Goal", sleep_goal, "hours")}
{_build_conditional_line("Sleep Quality", sleep_quality)}

**Current Context:**
- Current Time: {current_time}
- Bedtime Window: {bedtime_start} to {bedtime_end}
{_build_conditional_line("Bedtime Streak", bedtime_streak, "days")}
{_build_conditional_line("Sleep Last 3 Nights", _format_sleep_last3nights(sleep_last3nights) if sleep_last3nights else None)}

**CRITICAL ANALYSIS TASKS:**

1. **BEDTIME POSITION ANALYSIS** (based on first_sleep_time):
   - Compare first_sleep_time to bedtime_start and bedtime_end
   - Determine if bedtime is:
     - **Early** (near bedtime_start): Circadian rhythm well aligned
     - **Mid** (middle of window): Good alignment with natural patterns
     - **Late** (near bedtime_end or after): Early signs of circadian drift

2. **SLEEP DURATION ANALYSIS** (based on sleep_lastnight vs sleep_goal):
   - If sleep_lastnight < sleep_goal: Continuous sleep with short duration
   - If sleep_lastnight >= sleep_goal: Adequate duration

3. **WAKE TIME PATTERN ANALYSIS** (based on last_wake_time):
   - Compare last_wake_time to typical wake time (bedtime_end or usual wake time)
   - Determine if:
     - **Earlier than usual**: Risk of low energy later
     - **Very early or near dawn**: Sleep cycle significantly disrupted
     - **Normal**: Regular wake pattern

4. **CURRENT TIME POSITION** within bedtime window:
   - Calculate position of current_time relative to bedtime_start and bedtime_end
   - Determine sleep phase:
     - First 25% of window → EARLY SLEEP PHASE
     - Middle 50% of window → MIDDLE SLEEP PHASE
     - Last 25% of window → LATE SLEEP PHASE
   - Consider wrap-around cases (bedtime may span midnight)

**Calendar Events:**
{calendar_data}

**Determine:**
1. Sleep context summary combining:
   - Bedtime position (early/mid/late)
   - Sleep duration (adequate vs short)
   - Wake time patterns (normal vs early)
   - Current position in bedtime window

2. Tailored suggestion based on analysis:
   - Early bedtime + adequate sleep → Maintain current bedtime
   - Mid bedtime → Keep sleep timing consistent
   - Late bedtime → Gradually shift bedtime earlier (20-30 minutes)
   - Short duration → Increase sleep duration or add restorative breaks
   - Early wake → Light morning workload, avoid high-intensity activities
   - Very early wake → Prioritize recovery, aim for earlier bedtime tonight

{important_language_prompt(language_name)}

Return JSON with point1 (sleep context summary) and point2 (tailored suggestion):
{{
  "point1": "sleep context summary (bedtime position, duration, wake patterns, current phase)",
  "point2": "tailored suggestion based on analysis"
}}"""

    return system_prompt, user_prompt


def get_morning_start_insight_prompt(
    calendar_data: str,
    current_time: str,
    timezone: Optional[str] = None,
    bedtime_end: Optional[str] = None,
    active_hours_start: Optional[str] = None,
    sleep_lastnight: Optional[float] = None,
    sleep_quality: Optional[str] = None,
    sleep_goal: Optional[float] = None,
    time_to_next_event: Optional[int] = None,
    free_slot_length: Optional[int] = None,
    language: Optional[str] = None,
    overall_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for morning start insights.

    Morning start window: current_time > bedtime_end and < active_hours_start
    Within 60 mins after wake → Suggest day planning

    Args:
        calendar_data: Formatted calendar events string
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        bedtime_end: Bedtime window end (e.g., "08:00")
        active_hours_start: Active hours start (e.g., "07:00")
        sleep_lastnight: Hours of sleep last night
        sleep_quality: Sleep quality level (very_high, high, ok, low, very_low)
        sleep_goal: Target sleep hours per night
        time_to_next_event: Minutes until next event
        free_slot_length: Current free slot length in minutes
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    system_prompt = f"""You are a morning start analyzer. Analyze the user's morning period (after bedtime window, before active hours) and provide insights in exactly two points.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**User Schedule:**
- Bedtime End: {bedtime_end or 'Not specified'}
- Active Hours Start: {active_hours_start or 'Not specified'}

**Morning Start Rules:**
- **MORNING_START**: After bedtime window, before active hours start → Suggest day planning
- Within 60 mins after wake → Focus on planning and preparation for the day

**Point 1:** Identify morning start context (1-2 sentences)
**Point 2:** Provide day planning suggestion (1-2 sentences)

{_build_overall_context_prompt(overall_context)}

IMPORTANT TIME RULES:
- DO NOT mention exact clock times (e.g. 8:58 AM, 9:00, current time)
- DO NOT repeat or paraphrase the provided current_time value
- Use ONLY relative time expressions such as:
  "early in your day", "as your day begins",
  "after your rest period", "before your active hours start",
  "at the start of your day"
- If time context is needed, describe it abstractly, not numerically

IMPORTANT:
- Be concise - each point 1-2 sentences maximum
- Use natural, conversational language
- Focus on day planning and morning preparation
- Respond with valid JSON only

{important_language_prompt(language_name)}

Example:
{{
  "point1": "You're at the start of your day, fresh after rest",
  "point2": "Perfect time to plan your day - review priorities and set your focus for the hours ahead"
}}"""

    user_prompt = f"""Analyze the morning start context:
**Language:** Respond in {language}
Current Time: {current_time}
Bedtime End: {bedtime_end}
Active Hours Start: {active_hours_start}
{_build_conditional_line("Sleep Last Night", sleep_lastnight, "hours")}
{_build_conditional_line("Sleep Quality", sleep_quality)}
{_build_conditional_line("Sleep Goal", sleep_goal, "hours")}
{_build_conditional_line("Time to Next Event", time_to_next_event, "minutes")}
{_build_conditional_line("Free Slot Length", free_slot_length, "minutes")}

Calendar Events:
{calendar_data}

Determine:
1. Morning start context (after rest, beginning of day, consider sleep quality)
2. Day planning suggestion (consider available time and upcoming events)

{important_language_prompt(language_name)}

Return JSON with point1 (context) and point2 (suggestion):
{{
  "point1": "morning start context summary",
  "point2": "day planning suggestion"
}}"""

    return system_prompt, user_prompt


def get_active_window_insight_prompt(
    calendar_data: str,
    current_time: str,
    timezone: Optional[str] = None,
    active_hours_start: Optional[str] = None,
    active_hours_end: Optional[str] = None,
    time_to_next_event: Optional[int] = None,
    free_slot_length: Optional[int] = None,
    calendar_density: Optional[int] = None,
    back_to_back_count: Optional[int] = None,
    language: Optional[str] = None,
    overall_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for active window insights.

    Args:
        calendar_data: Formatted calendar events string
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        active_hours_start: Active hours start (e.g., "07:00")
        active_hours_end: Active hours end (e.g., "18:00")
        time_to_next_event: Minutes until next event
        free_slot_length: Current free slot length in minutes
        calendar_density: Number of events in active hours today
        back_to_back_count: Number of back-to-back event blocks
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    system_prompt = f"""You are an active window analyzer. Analyze the user's active work hours and provide insights in exactly two points.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**User Schedule:**
- Active Hours (Work Focus): {active_hours_start or 'Not specified'} to {active_hours_end or 'Not specified'}
- Time to Next Event: {time_to_next_event} minutes (if applicable)
- Current Free Slot: {free_slot_length} minutes (if applicable)

**Active Window Rules:**
1. **ACTIVE_FOCUS_WINDOW**: Within active hours with free slot >= 45 mins → Suggest focus work
2. **PREP_WINDOW**: Less than 30 mins before next event → Suggest prep for that event
3. **SHORT_FOCUS**: In active hours but slot < 45 mins → Suggest mini focus

**Point 1:** Identify active window context (1-2 sentences)
**Point 2:** Provide actionable work/focus suggestion (1-2 sentences)

{_build_overall_context_prompt(overall_context)}

IMPORTANT TIME RULES:
- DO NOT mention exact clock times (e.g. 8:58 AM, 9:00, current time)
- DO NOT repeat or paraphrase the provided current_time value
- Use ONLY relative time expressions such as:
  "in your active hours",
  "with a long free window ahead", "before your next event",
  "during your focus time"
- If time context is needed, describe it abstractly, not numerically

IMPORTANT:
- Be concise - each point 1-2 sentences maximum
- Use natural, conversational language
- Focus on actionable work and focus insights
- Respond with valid JSON only

{important_language_prompt(language_name)}

Example:
{{
  "point1": "You're in your active hours with a good free window ahead",
  "point2": "Perfect time for focused work - start your most important task"
}}"""

    user_prompt = f"""Analyze the active window context:
**Language:** Respond in {language}
Current Time: {current_time}
Active Hours: {active_hours_start} to {active_hours_end}
{_build_conditional_line("Time to Next Event", time_to_next_event, "minutes")}
{_build_conditional_line("Free Slot", free_slot_length, "minutes")}
{_build_conditional_line("Calendar Density", calendar_density, "events today")}
{_build_conditional_line("Back-to-Back Events", back_to_back_count)}

Calendar Events:
{calendar_data}

Determine:
1. Active window context (focus time, prep window, consider calendar load)
2. Work/focus suggestion (consider if user is overloaded with calendar events)

{important_language_prompt(language_name)}

Return JSON with point1 (context) and point2 (suggestion):
{{
  "point1": "active window context summary",
  "point2": "work/focus suggestion from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


def get_evening_flexible_insight_prompt(
    calendar_data: str,
    current_time: str,
    timezone: Optional[str] = None,
    bedtime_start: Optional[str] = None,
    time_to_bedtime: Optional[int] = None,
    wind_down_buffer_mins: Optional[int] = None,
    sleep_lastnight: Optional[float] = None,
    sleep_quality: Optional[str] = None,
    free_slot_length: Optional[int] = None,
    language: Optional[str] = None,
    overall_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for evening flexible window insights.

    Evening flexible window: from 19:00 (7 PM) to before wind_down_window
    Wind_down_window starts 1 hours before bedtime_start
    This is a flexible time for recharge, connection, hobbies, or light activities

    Args:
        calendar_data: Formatted calendar events string
        current_time: Current time in ISO format
        timezone: Optional IANA timezone
        bedtime_start: Bedtime window start (e.g., "22:00")
        time_to_bedtime: Minutes until bedtime_start
        wind_down_buffer_mins: Wind-down buffer minutes (default: 60)
        sleep_lastnight: Hours of sleep last night
        sleep_quality: Sleep quality level (very_high, high, ok, low, very_low)
        free_slot_length: Current free slot length in minutes
        language: Optional language code

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    language_name = get_language_name(language)

    system_prompt = f"""You are an evening flexible window analyzer. Analyze the user's evening flexible period (from 7 PM to before wind-down) and provide insights in exactly two points.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**User Schedule:**
- Bedtime Window Start: {bedtime_start or 'Not specified'}
- Evening Flexible Window: From 19:00 to before wind-down (1 hours before bedtime)

**Evening Flexible Window Rules:**
- **EVENING_FLEXIBLE**: From 19:00 to before wind-down window → Suggest flexible activities
- This is a flexible time for recharge, connection, hobbies, or light activities
- Focus on activities that help transition from work to rest

**Point 1:** Identify evening flexible context (1-2 sentences)
**Point 2:** Provide flexible evening suggestion (1-2 sentences)

{_build_overall_context_prompt(overall_context)}

IMPORTANT TIME RULES:
- DO NOT mention exact clock times (e.g. 7:00 PM, 8:00, current time)
- DO NOT repeat or paraphrase the provided current_time value
- DO NOT suggest intense work, heavy planning, or cognitively demanding tasks
- DO NOT suggest activities that interfere with wind-down (avoid stimulating media, intense exercise)
- Use ONLY relative time expressions such as:
  "in your evening", "during your flexible evening time",
  "as evening approaches", "in this evening window",
  "before your wind-down period"
- If time context is needed, describe it abstractly, not numerically

IMPORTANT:
- Be concise - each point 1-2 sentences maximum
- Use natural, conversational language
- Focus on flexible evening activities: recharge, connection, hobbies, light activities, meal planning, gentle movement
- Do NOT default to meditation, mindfulness, or stretch routines unless time_to_bedtime is under 60 minutes
- Respond with valid JSON only

{_insight_block("evening_flexible")}

{important_language_prompt(language_name)}


Example:
{{
  "point1": "You're in your flexible evening window - time to recharge and connect",
  "point2": "Perfect for a light hobby, connecting with friends, or planning a relaxing evening ahead"
}}"""

    user_prompt = f"""Analyze the evening flexible window context:
Current Time: {current_time}
Bedtime Start: {bedtime_start}
{_build_conditional_line("Time to Bedtime", time_to_bedtime, "minutes")}
{_build_conditional_line("Wind-Down Buffer", wind_down_buffer_mins, "minutes")}
{_build_conditional_line("Sleep Last Night", sleep_lastnight, "hours")}
{_build_conditional_line("Sleep Quality", sleep_quality)}
{_build_conditional_line("Free Slot Length", free_slot_length, "minutes")}

Calendar Events:
{calendar_data}

Determine:
1. Evening flexible context (flexible evening time, before wind-down, consider sleep needs)
2. Flexible evening suggestion (recharge, connection, hobbies, light activities, consider available time)

{important_language_prompt(language_name)}

Return JSON with point1 (context) and point2 (suggestion):
{{
  "point1": "evening flexible context summary",
  "point2": "flexible evening suggestion from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


def get_calendar_pattern_insight_prompt(
    calendar_data: str,
    current_time: str,
    timezone: Optional[str] = None,
    active_hours_start: Optional[str] = None,
    active_hours_end: Optional[str] = None,
    calendar_density: Optional[int] = None,
    back_to_back_count: Optional[int] = None,
    meeting_minutes: Optional[int] = None,
    language: Optional[str] = None,
    overall_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for calendar pattern analysis insights."""
    language_name = get_language_name(language)

    system_prompt = f"""You are a calendar pattern analyzer. Analyze the user's calendar patterns and provide insights in exactly two points.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Active Hours:** {active_hours_start or 'Not specified'} to {active_hours_end or 'Not specified'}

**Calendar Metrics:**
- Back-to-back Count: {back_to_back_count or 'N/A'}
- Calendar Density: {calendar_density or 'N/A'} event blocks in active hours
- Event Minutes (Last 3 Hours): {meeting_minutes or 'N/A'} minutes

**Patterns to Detect:**
1. **Back-to-back events**: Consecutive event blocks with no gap (backToBack >= 2)
2. **Calendar density**: High number of event blocks in active hours (density >= 5)
3. **Fragmentation**: Many small free slots (high context switching)
4. **Event streak**: Continuous event blocks for >= 2 hours (meetingMinutesLast3Hours >= 120)
5. **Schedule optimization**: No focus blocks in active hours for >= 3 days

**Point 1:** Summarize calendar patterns (back-to-back, density, fragmentation, etc.) (1-2 sentences)
**Point 2:** Suggest schedule optimization (add breaks, protect focus blocks, reschedule, etc.) (1-2 sentences)

{_build_overall_context_prompt(overall_context)}

IMPORTANT TIME RULES:
- DO NOT reference the current moment (e.g. "right now", "currently", "at this time")
- DO NOT restate or paraphrase the provided current_time
- DO NOT use relative phrases tied to now (e.g. "about to", "coming up next", "just finished")
- You MAY reference event time ranges that exist within the calendar data (e.g. "from 1–4 PM")
- Prefer pattern-based and structural descriptions over time-sensitive language
- Frame insights as general schedule observations, not real-time alerts
- DO NOT suggest any task, slot, or action in the past
- DO NOT suggest anything that overlaps with existing calendar events
- DO NOT suggest a task longer than the available free slot
- DO NOT suggest deep work, workouts, or cognitively heavy tasks within 20 minutes before the next event
- DO NOT squeeze new meetings or tasks between consecutive back-to-back events

IMPORTANT:
- relative to calendar events (if have):
    - Pay attention to event titles, descriptions, and participants to better understand the context and intent behind each calendar event.
    - Use this contextual understanding to produce more relevant, coherent, and human-like insights.
- Be concise - each point 1-2 sentences maximum
- Use natural, conversational language
- Focus on actionable insights
- Respond with valid JSON only

{important_language_prompt(language_name)}

Example:
{{
  "point1": "You have 3 back-to-back events from 1-4 PM with no breaks",
  "point2": "Consider adding 5-min buffers between events and protecting your active hours for focus work"
}}"""

    user_prompt = f"""Analyze calendar patterns:
Calendar Events:
{calendar_data}

Current Time: {current_time}
Active Hours: {active_hours_start} to {active_hours_end}

Calculate:
1. Number of back-to-back events
2. Calendar density (events in active hours)
3. Fragmentation score (many small slots = high)
4. Event streak (continuous events >= 2 hours)
5. Whether focus blocks exist in active hours

{important_language_prompt(language_name)}

Return JSON with point1 (pattern summary) and point2 (optimization suggestion):
{{
  "point1": "pattern summary",
  "point2": "optimization suggestion from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


def get_health_goal_insight_prompt(
    steps_today: Optional[int],
    steps_goal: Optional[float],
    sleep_lastnight: Optional[float],
    sleep_goal: Optional[float],
    latest_heart_rate: Optional[int],
    resting_heart_rate: Optional[int],
    current_time: str,
    timezone: Optional[str] = None,
    free_slot_length: Optional[int] = None,
    daily_health_progress: Optional[Dict[str, Any]] = None,
    weekly_health_progress: Optional[Dict[str, Any]] = None,
    is_weekend: Optional[bool] = None,
    language: Optional[str] = None,
    overall_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for health goal progress insights.

    Args:
        steps_today: Steps taken today
        steps_goal: Daily steps goal
        sleep_lastnight: Sleep hours last night
        sleep_goal: Daily sleep goal
        latest_heart_rate: Latest heart rate reading
        resting_heart_rate: Resting heart rate
        current_time: Current time string
        timezone: Optional timezone
        free_slot_length: Optional free slot length in minutes
        language: Optional language code
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
    daily_health_progress_percentage = None
    weekly_health_progress_percentage = None
    if daily_health_progress:
        daily_health_progress_percentage = daily_health_progress.get(
            HealthDataConstants.KEY_PROGRESS_PERCENTAGE
        )
    if weekly_health_progress:
        weekly_health_progress_percentage = weekly_health_progress.get(
            HealthDataConstants.KEY_PROGRESS_PERCENTAGE
        )

    system_prompt = f"""You are a health goal progress tracker. Analyze the user's health data against their goals and provide insights in exactly two points.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Health Goals:**
- Steps Goal: {steps_goal or 'Not set'} steps/day
- Sleep Goal: {sleep_goal or 'Not set'} hours/day

**Current Health Data:**
- Steps Today: {steps_today or 'N/A'} / Goal: {steps_goal or 'N/A'} ({steps_percentage or 'N/A'}%)
- Sleep Last Night: {sleep_lastnight or 'N/A'} hours / Goal: {sleep_goal or 'N/A'} hours ({sleep_percentage or 'N/A'}%)
- Latest Heart Rate: {latest_heart_rate or 'N/A'} bpm
- Resting Heart Rate: {resting_heart_rate or 'N/A'} bpm
- Daily Health Progress: {daily_health_progress_percentage if daily_health_progress_percentage is not None else 'N/A'}%
- Weekly Health Progress: {weekly_health_progress_percentage * 100 if weekly_health_progress_percentage is not None else 'N/A'}%

**Free Slot:** {free_slot_length} minutes (if applicable)
**Is Weekend:** {is_weekend if is_weekend is not None else 'N/A'}

**Health Catch-up Rules:**
1. If dailyHealthProgress < target AND (5 PM - 7 PM OR weekend) AND slot >= 30 mins → Suggest health catch-up
2. If weeklyHealthProgress < target AND weekend AND slot >= 30 mins → Suggest weekend catch-up
3. If goals met → Celebrate and suggest maintenance

**Point 1:** Summarize health goal progress (steps, sleep, etc.) (1-2 sentences)
**Point 2:** Provide health catch-up suggestion OR celebration if goals met (1-2 sentences)

{_build_overall_context_prompt(overall_context)}

IMPORTANT TIME RULES:
- DO NOT mention or restate the exact current time
- DO NOT use phrases implying real-time urgency (e.g. "now", "right now", "at this moment")
- DO NOT suggest immediate actions tied to the current minute
- When referencing catch-up windows, describe them qualitatively (e.g. "later in the day", "during typical catch-up hours")
- Health progress numbers (steps, sleep, heart rate) are allowed
- Frame suggestions as flexible opportunities, not real-time commands

IMPORTANT:
- Be concise - each point 1-2 sentences maximum
- Use natural, conversational language
- Focus on actionable health suggestions
- Respond with valid JSON only

{important_language_prompt(language_name)}

Example:
{{
  "point1": "You're at 2,240 steps today, 760 steps behind your 3,000 step goal",
  "point2": "Perfect time for a 20-min walk to catch up - want to start now or schedule it?"
}}"""

    user_prompt = f"""Analyze health goal progress:
**Language:** Respond in {language}
Current Health Data:
- Steps Today: {steps_today or 'N/A'} / Goal: {steps_goal or 'N/A'}
- Sleep Last Night: {sleep_lastnight or 'N/A'} hours / Goal: {sleep_goal or 'N/A'} hours
- Latest Heart Rate: {latest_heart_rate or 'N/A'} bpm
- Resting Heart Rate: {resting_heart_rate or 'N/A'} bpm

Current Time: {current_time}
Free Slot: {free_slot_length} minutes

Calculate:
1. Progress for each goal (percentage)
2. Whether user is behind on goals
3. Whether it's catch-up window (5 PM - 7 PM) or weekend
4. Appropriate suggestion (catch-up OR celebration)

{important_language_prompt(language_name)}

Return JSON with point1 (progress summary) and point2 (suggestion):
{{
  "point1": "progress summary",
  "point2": "suggestion from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


def get_safety_risk_insight_prompt(
    latest_heart_rate: Optional[int],
    resting_heart_rate: Optional[int],
    sleep_lastnight: Optional[float],
    sleep_goal: Optional[float],
    current_time: str,
    timezone: Optional[str] = None,
    baseline_resting_hr: Optional[int] = None,
    sleep_last3nights: Optional[List[Optional[float]]] = None,
    language: Optional[str] = None,
    overall_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for safety risk detection insights.

    Args:
        latest_heart_rate: Latest heart rate reading
        resting_heart_rate: Resting heart rate
        sleep_lastnight: Sleep hours last night
        sleep_goal: Daily sleep goal
        current_time: Current time string
        timezone: Optional timezone
        baseline_resting_hr: Baseline resting heart rate (for stress signal detection)
        sleep_last3nights: Sleep data for last 3 nights (dict with sleep_hours list and all_3_nights_have_data flag)
        language: Optional language code
    """
    language_name = get_language_name(language)

    # Check current hour for caffeine cutoff (default 2 PM)
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        if timezone:
            tz = ZoneInfo(timezone)
            current_dt = datetime.now(tz)
        else:
            current_dt = datetime.now(ZoneInfo("UTC"))
        current_hour = current_dt.hour
        caffeine_cutoff_passed = current_hour >= 14  # 2 PM
    except:
        caffeine_cutoff_passed = False

    system_prompt = f"""You are a safety and health risk detector. Identify dangerous health conditions and provide safety guidance in exactly two points.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Health Data:**
- Latest Heart Rate: {latest_heart_rate or 'N/A'} bpm
- Resting Heart Rate: {resting_heart_rate or 'N/A'} bpm
- Baseline Resting HR: {baseline_resting_hr or 'N/A'} bpm
- Sleep Last Night: {sleep_lastnight or 'N/A'} hours
- Sleep Goal: {sleep_goal or 'Not set'} hours
- Sleep Last 3 Nights: {_format_sleep_last3nights(sleep_last3nights) if sleep_last3nights else 'N/A'}

**Safety Rules:**
1. **Dangerous HR**: If HR < 40 OR HR > 180 → Block all suggestions, show safety guidance
2. **High Stress**: If resting HR > baseline + 10 bpm → Block intense exercise
3. **Sleep Debt**: If sleep_lastnight < sleep_goal OR sleep_last3nights shows insufficient sleep → Avoid late intense workouts
4. **Caffeine Cutoff**: If now_time >= 14:00 (2 PM) → Avoid caffeine suggestions

**Point 1:** Identify safety risks (dangerous HR, sleep debt, high stress, etc.) (1-2 sentences)
**Point 2:** Provide safety guidance or risk mitigation (rest, seek medical advice, avoid intense exercise, etc.) (1-2 sentences)

{_build_overall_context_prompt(overall_context)}

IMPORTANT TIME & SAFETY RULES:
- DO NOT mention or restate the exact current time
- DO NOT imply real-time monitoring (e.g. "right now", "currently", "at this moment")
- DO NOT present guidance as urgent commands unless the condition is explicitly dangerous
- When referencing time-based rules (evening, caffeine cutoff), describe them qualitatively (e.g. "later in the day", "toward the evening")
- Safety guidance should be precautionary and conditional, not definitive diagnoses
- Avoid medical certainty language; use supportive and cautious phrasing

IMPORTANT:
- Be concise - each point 1-2 sentences maximum
- Use natural, conversational language
- Prioritize safety over productivity
- Respond with valid JSON only

{important_language_prompt(language_name)}

Example:
{{
  "point1": "Your heart rate is 185 bpm, which is dangerously high",
  "point2": "It may be best to pause strenuous activity and seek medical advice if this level continues"
}}"""

    user_prompt = f"""Check for safety risks:
Current Health Data:
- Latest Heart Rate: {latest_heart_rate or 'N/A'} bpm
- Resting Heart Rate: {resting_heart_rate or 'N/A'} bpm
- Sleep Last Night: {sleep_lastnight or 'N/A'} hours / Goal: {sleep_goal or 'N/A'} hours

Current Time: {current_time}

Apply safety rules:
1. Check if HR is dangerous (< 40 or > 180)
2. Check if sleep debt exists (sleep_lastnight < sleep_goal)
3. Check if stress signal is high (resting HR > baseline + 10)
4. Check if caffeine cutoff passed

{important_language_prompt(language_name)}

Return JSON with point1 (risk summary) and point2 (safety guidance):
{{
  "point1": "risk summary",
  "point2": "safety guidance from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


def get_recovery_break_insight_prompt(
    calendar_data: str,
    current_time: str,
    timezone: Optional[str] = None,
    break_interval: Optional[float] = None,
    break_duration: Optional[float] = None,
    back_to_back_count: Optional[int] = None,
    meeting_minutes: Optional[int] = None,
    continuous_events_minutes: Optional[int] = None,
    language: Optional[str] = None,
    overall_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for recovery and break management insights."""
    language_name = get_language_name(language)

    system_prompt = f"""You are a recovery and break advisor. Detect when user needs breaks and provide recovery suggestions in exactly two points.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Break Settings:**
- Break Interval: Every {break_interval or 'Not set'} hours
- Break Duration: {break_duration or 'Not set'} minutes

**Recovery Metrics:**
- Back-to-back Count: {back_to_back_count or 'N/A'}
- Meeting Minutes (Last 3 Hours): {meeting_minutes or 'N/A'} minutes
- Continuous Events: {continuous_events_minutes or 'N/A'} minutes

**Recovery Rules:**
1. **Continuous Work**: If continuous work/meetings >= 90 mins without break → Suggest break
2. **Back-to-back**: If backToBack >= 2 → Suggest micro-recovery
3. **Meeting Streak**: If meeting minutes in last 3 hours >= 120 → Suggest recovery window
4. **Recovery Missing**: If continuous events >= 120 mins → Break first

**Point 1:** Summarize work/meeting patterns (continuous work, back-to-back, etc.) (1-2 sentences)
**Point 2:** Provide recovery/break suggestion (take break, add buffer, hydration, etc.) (1-2 sentences)

{_build_overall_context_prompt(overall_context)}

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
- Be concise - each point 1-2 sentences maximum
- Use natural, conversational language
- Focus on recovery and wellbeing
- Respond with valid JSON only

{important_language_prompt(language_name)}

Example:
{{
  "point1": "You've been in back-to-back meetings for 2 hours with no breaks",
  "point2": "A short break could help reset your energy — try breathing, hydrating, and adding buffers between future meetings"
}}"""

    user_prompt = f"""Analyze recovery needs:

Calendar Events:
{calendar_data}

Current Time: {current_time}
Break Interval: Every {break_interval} hours

Calculate:
1. Continuous work/meeting time (no breaks)
2. Number of back-to-back meetings
3. Meeting minutes in last 3 hours
4. Whether break is needed

{important_language_prompt(language_name)}

Return JSON with point1 (pattern summary) and point2 (recovery suggestion):
{{
  "point1": "pattern summary",
  "point2": "recovery suggestion from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


def get_task_type_insight_prompt(
    calendar_data: str,
    current_time: str,
    timezone: Optional[str] = None,
    slot_length: Optional[int] = None,
    time_to_next_event: Optional[int] = None,
    time_to_bedtime: Optional[int] = None,
    in_active_window: Optional[bool] = None,
    language: Optional[str] = None,
    overall_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for task type matching insights."""
    language_name = get_language_name(language)

    system_prompt = f"""You are a task type matcher. Determine which task types are appropriate for the current context and provide insights in exactly two points.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Context:**
- Slot Length: {slot_length} minutes
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

**Point 1:** Summarize available time slot and context (1-2 sentences)
**Point 2:** Suggest appropriate task type (deep work, admin, prep, etc.) or alternative if not suitable (1-2 sentences)

{_build_overall_context_prompt(overall_context)}

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
- Be concise - each point 1-2 sentences maximum
- Use natural, conversational language
- Focus on appropriate task matching
- Respond with valid JSON only

{important_language_prompt(language_name)}

Example:
{{
  "point1": "You have 60 minutes free in your active window with 45 mins until your next meeting",
  "point2": "Perfect for a focused work block - start your most important task now"
}}"""

    user_prompt = f"""Match task types to current context:

Calendar Events:
{calendar_data}

Slot Length: {slot_length} minutes
Time to Next Event: {time_to_next_event} minutes
Time to Bedtime: {time_to_bedtime} minutes
In Active Window: {in_active_window}

Determine:
1. Which task types are appropriate (deep work, admin, prep, etc.)
2. Which task types should be blocked (intense tasks, deep work outside active, etc.)
3. Best suggestion for current context

{important_language_prompt(language_name)}

Return JSON with point1 (context summary) and point2 (task type suggestion):
{{
  "point1": "context summary",
  "point2": "task type suggestion from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


def get_weekend_lifestyle_insight_prompt(
    calendar_data: str,
    current_time: str,
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    free_slot_length: Optional[int] = None,
    steps_today: Optional[int] = None,
    steps_goal: Optional[float] = None,
    overall_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for weekend lifestyle insights."""
    language_name = get_language_name(language)

    system_prompt = f"""You are a weekend and lifestyle analyzer. Analyze the user's weekend and lifestyle patterns and provide insights in exactly two points.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Weekend & Lifestyle Rules:**
1. **Weekend Health**: If weekend AND morning AND slot >= 45 mins → Prioritize workout/outdoors/meal prep
2. **Weekend Balance**: If weekend AND steps high AND social goal met → Reinforce lifestyle balance
3. **Weekend Burnout**: If weekend AND workEventsHours > threshold → Suggest recovery + move flexible work
4. **Weekend Social**: If weekend AND evening AND not inWindDown → Suggest social or hobby
5. **No Upcoming Events**: If there are no upcoming events, suggest flexible lifestyle options (social connection, light recreation, reading, weekly reset/planning, or recovery) based on current context.

**Point 1:** Summarize weekend/lifestyle patterns (balance, health opportunities, work overload, etc.) (1-2 sentences)
**Point 2:** Provide weekend/lifestyle suggestion (workout, recovery, social, hobby, etc.) (1-2 sentences)

{_build_overall_context_prompt(overall_context)}

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
- If no upcoming events are present, provide opportunity-style suggestions that feel natural and varied.
- Be concise - each point 1-2 sentences maximum
- Use natural, conversational language
- Focus on lifestyle balance and wellbeing
- Respond with valid JSON only

{important_language_prompt(language_name)}

Example:
{{
  "point1": "Great balance this weekend - movement + connection",
  "point2": "Weekend morning is perfect for your health goals - consider a workout or outdoor activity"
}}"""

    user_prompt = f"""Analyze weekend and lifestyle patterns:

**Current Context:**
- Is Weekend: True
- Free Slot Length: {free_slot_length} minutes (if applicable)

**Health Data:**
- Steps Today: {steps_today or "Not available"}
- Steps Goal: {steps_goal or "Not available"}

**Calendar Events:**
{calendar_data}

Analyze:
- Weekend health opportunities
- Weekend balance
- Weekend social goals
- Weekend burnout
- If no upcoming events, prioritize flexible and enjoyable suggestions instead of rigid scheduling.

{important_language_prompt(language_name)}

Return JSON with point1 (pattern summary) and point2 (suggestion):
{{
  "point1": "weekend/lifestyle pattern summary",
  "point2": "weekend/lifestyle suggestion from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


def get_contextual_suggestion_insight_prompt(
    calendar_data: str,
    health_params: Optional[Dict[str, Any]],
    user_profile: Optional[Dict[str, Any]],
    current_time: str,
    timezone: Optional[str] = None,
    time_context: Optional[Dict[str, Any]] = None,
    calendar_patterns: Optional[Dict[str, Any]] = None,
    language: Optional[str] = None,
    overall_context: Optional[str] = "",
) -> tuple[str, str]:
    """Get prompts for contextual suggestion insights."""
    language_name = get_language_name(language)

    system_prompt = f"""You are a contextual productivity advisor. Generate personalized suggestions based on all available context in exactly two points.

**Current Time:** {current_time}
**Timezone:** {timezone}
**Language:** Respond in {language_name}

**Context Summary:**
- Time Window: {json.dumps(time_context, indent=2) if time_context else 'Not available'}
- Calendar Patterns: {json.dumps(calendar_patterns, indent=2) if calendar_patterns else 'Not available'}
- Health Progress: Available in health data
- User Profile: Available in user profile

**AI SUGGESTIONS Rules:**
- Health catch-up (5 PM - 7 PM, behind on goals)
- Wind-down (1 hours before bedtime)
- Bedtime window (sleep prep)
- Morning start (day planning)
- Active focus window (deep work)
- Recovery/break needs
- Schedule optimization

**Point 1:** Summarize current context (time, calendar, health, etc.) (1-2 sentences)
**Point 2:** Provide contextual suggestion based on AI SUGGESTIONS rules (1-2 sentences)

{_build_overall_context_prompt(overall_context)}

IMPORTANT TIME RULES:
- DO NOT mention exact times, hours, or minutes (e.g. "5 PM", "2 hours", "60 mins")
- DO NOT restate or paraphrase current_time
- DO NOT use real-time urgency language (e.g. "now", "right now", "perfect time")
- Describe time context only in qualitative terms (e.g. "later in the day", "approaching bedtime", "during a free window")
- When combining multiple signals, prioritize relevance over precision
- This is a cached, non-real-time contextual suggestion
- DO NOT suggest any task, slot, or action in the past
- DO NOT suggest anything that overlaps with existing calendar events
- DO NOT suggest a task longer than the available free slot
- DO NOT suggest deep work, workouts, or cognitively heavy tasks within 20 minutes before the next event
- DO NOT squeeze new meetings or tasks between consecutive back-to-back events

|HARD DATA-FAITHFULNESS RULES:
|- Cite ONLY numbers that appear in the inputs provided to you (calendar events, health data, user profile). If a duration, count, or measurement is not in your input, do NOT quote a number for it. NEVER invent values like "187 minutes", "150 bpm", "3 meetings" unless those exact values appear in the inputs above.
|- Time slot suggestions (start/end) MUST be in the FUTURE relative to current_time. Never suggest an action at a clock time earlier than current_time. If current_time is 17:20, the earliest valid start time is later than 17:20.
|- BMI is a body-composition value, not a direct measure of today's activity. Do NOT use BMI as evidence of "low activity" or "sedentary today" unless steps/activity metrics in the inputs explicitly support that claim.
|- DO NOT describe the user's thoughts, mental state, or cognitive functions with negative or stigmatizing language such as "scattered", "messy", "junk", "garbage", "cluttered", "noisy", "sluggish thinking", or any framing that suggests the user's mind or thinking is somehow broken or worthless. Use neutral, factual language: "focus is lower today", "your attention is divided", "your mind feels pulled in several directions" — never imply the user or their thoughts are a problem to be fixed.

IMPORTANT:
- relative to calendar events (if have):
    - Pay attention to event titles, descriptions, and participants to better understand the context and intent behind each calendar event.
    - Use this contextual understanding to produce more relevant, coherent, and human-like insights.
- Be concise - each point 1-2 sentences maximum
- Use natural, conversational language
- Follow all NOT DO rules strictly
- Focus on most relevant suggestion for current context
- Respond with valid JSON only

{important_language_prompt(language_name)}

Example:
{{
  "point1": "You're approaching the end of your day with some flexible time and room to support your health goals",
  "point2": "A gentle walk could help you stay active while easing into a calmer wind-down routine"
}}"""

    user_prompt = f"""Generate contextual suggestion:

Calendar Events:
{calendar_data}

Health Data:
{json.dumps(health_params, indent=2) if health_params else 'No health data available'}

User Profile:
{json.dumps(user_profile, indent=2) if user_profile else 'No user profile available'}

Time Context: {json.dumps(time_context, indent=2) if time_context else 'Not available'}
Calendar Patterns: {json.dumps(calendar_patterns, indent=2) if calendar_patterns else 'Not available'}
Current Time: {current_time}

Follow AI SUGGESTIONS rules and generate the most relevant suggestion for current context.

{important_language_prompt(language_name)}

Return JSON with point1 (context summary) and point2 (contextual suggestion):
{{
  "point1": "context summary",
  "point2": "contextual suggestion from current time {current_time} onward"
}}"""

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Group → builder mapping + reasoning injection router
# ---------------------------------------------------------------------------

_PRODUCTIVITY_BUILDERS = {
    "wind_down_window": get_wind_down_window_insight_prompt,
    "bedtime_window": get_bedtime_window_insight_prompt,
    "morning_window": get_morning_start_insight_prompt,
    "active_window": get_active_window_insight_prompt,
    "evening_flexible": get_evening_flexible_insight_prompt,
    "calendar_pattern": get_calendar_pattern_insight_prompt,
    "health_goal": get_health_goal_insight_prompt,
    "safety_risk": get_safety_risk_insight_prompt,
    "recovery_break": get_recovery_break_insight_prompt,
    "task_type": get_task_type_insight_prompt,
    "weekend_lifestyle": get_weekend_lifestyle_insight_prompt,
    "contextual_suggestion": get_contextual_suggestion_insight_prompt,
}


def get_productivity_insight_prompt(group: str, **kwargs) -> tuple[str, str]:
    """Route to the correct productivity builder and append the universal reasoning block.

    This is the single injection point for prompt_insight_reasoning. Callers may
    continue to use individual builders directly; this router exists so that new code
    and future callers can get reasoning injection in one place.
    """
    builder = _PRODUCTIVITY_BUILDERS.get(
        group, get_contextual_suggestion_insight_prompt
    )
    system_prompt, user_prompt = builder(**kwargs)
    reasoning = _insight_block(group)
    if reasoning:
        system_prompt = system_prompt + "\n\n" + reasoning
    return system_prompt, user_prompt
