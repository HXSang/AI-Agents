"""Pre-reasoning day context prompt.

Generates a lightweight shared 'day profile' (dominant_pattern, cross_domain,
forward_context) that is cached once per user per ~5 minutes and injected into
all three insight types (productivity, overall, health) to ensure they start from
the same narrative frame regardless of generation order.
"""

from typing import Any, Dict, List, Optional

from agents.prompt import get_language_name


def get_day_context_prompt(
    health_params: Optional[Dict[str, Any]] = None,
    time_data: Optional[Dict[str, Any]] = None,
    calendar_metrics: Optional[Dict[str, Any]] = None,
    calendar_events: Optional[List[Dict[str, Any]]] = None,
    language: str = "en",
) -> tuple[str, str]:
    """Build prompts for the lightweight pre-reasoning day profile synthesis.

    Output shape (JSON, < 200 tokens):
    {
        "dominant_pattern": "what best describes today so far across all domains",
        "cross_domain": "how the domains are interacting",
        "forward_context": "what is ahead that matters (next event, free window, bedtime proximity)"
    }

    Args:
        health_params: Flattened health params dict from PrepareHealthData.
        time_data: Time data dict from PrepareTimeData.
        calendar_metrics: Calendar metrics dict from PrepareCalendarData.
        calendar_events: List of today's calendar event dicts.
        language: Language code (used for output locale).

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    health_params = health_params or {}
    time_data = time_data or {}
    calendar_metrics = calendar_metrics or {}
    calendar_events = calendar_events or []

    language_name = get_language_name(language)

    # --- Extract key signals for the prompt ---
    current_time = time_data.get("time_data_current_time_iso", "")
    timezone = time_data.get("time_data_timezone", "UTC")
    current_hour = time_data.get("time_data_current_hour")
    is_weekend = time_data.get("time_data_is_weekend", False)
    time_to_bedtime = time_data.get("time_data_time_to_bedtime")
    time_to_next_event = time_data.get("time_data_time_to_next_event")
    free_slot = time_data.get("time_data_free_slot_length")
    in_active_window = time_data.get("time_data_in_active_window", False)

    sleep_lastnight = health_params.get("sleep_lastnight")
    sleep_goal = health_params.get("sleep_goal")
    sleep_quality = health_params.get("sleep_quality")
    steps_today = health_params.get("steps_today")
    steps_goal = health_params.get("steps_goal")
    resting_hr = health_params.get("resting_heart_rate")
    baseline_hr = health_params.get("baseline_resting_hr")
    stress_signal_high = health_params.get("stress_signal_high", False)
    mood = health_params.get("mood")

    back_to_back = calendar_metrics.get("back_to_back_count", 0)
    calendar_density = calendar_metrics.get("calendar_density")
    continuous_events = calendar_metrics.get("continuous_events_minutes", 0)

    # Build context lines
    def _cline(label: str, value: Any, unit: str = "") -> str:
        if value is None:
            return ""
        return f"- {label}: {value}{' ' + unit if unit else ''}"

    health_lines = "\n".join(
        line
        for line in [
            _cline("Sleep last night", sleep_lastnight, "h") if sleep_lastnight else "",
            _cline("Sleep goal", sleep_goal, "h") if sleep_goal else "",
            _cline("Sleep quality", sleep_quality) if sleep_quality else "",
            _cline("Steps today", steps_today) if steps_today is not None else "",
            _cline("Steps goal", steps_goal) if steps_goal else "",
            _cline("Resting HR", resting_hr, "bpm") if resting_hr else "",
            _cline("Baseline resting HR", baseline_hr, "bpm") if baseline_hr else "",
            f"- Stress signal high: {stress_signal_high}" if stress_signal_high else "",
            _cline("Mood", mood) if mood else "",
        ]
        if line
    )

    calendar_lines = "\n".join(
        line
        for line in [
            _cline("Back-to-back count", back_to_back) if back_to_back else "",
            _cline("Calendar density", calendar_density) if calendar_density else "",
            (
                _cline("Continuous events", continuous_events, "min")
                if continuous_events
                else ""
            ),
            (
                _cline("Time to next event", time_to_next_event, "min")
                if time_to_next_event is not None
                else ""
            ),
            _cline("Free slot", free_slot, "min") if free_slot is not None else "",
        ]
        if line
    )

    time_lines = "\n".join(
        line
        for line in [
            _cline("Current time", current_time) if current_time else "",
            _cline("Timezone", timezone) if timezone else "",
            _cline("Current hour", current_hour) if current_hour is not None else "",
            f"- Weekend: {is_weekend}",
            f"- In active window: {in_active_window}",
            (
                _cline("Time to bedtime", time_to_bedtime, "min")
                if time_to_bedtime is not None
                else ""
            ),
        ]
        if line
    )

    # Brief event summary (titles only, max 5)
    event_titles = []
    for ev in (calendar_events or [])[:5]:
        title = ev.get("summary") or ev.get("title", "")
        start = ev.get("startTime", "")
        if title:
            event_titles.append(f"  - {title}" + (f" ({start[:16]})" if start else ""))
    events_section = (
        "Today's events:\n" + "\n".join(event_titles)
        if event_titles
        else "No events today."
    )

    system_prompt = f"""You are a day-profile synthesizer. Your job is to produce a concise, factual summary
of the user's current day across all domains — no suggestions, no advice.

**Language:** Respond in {language_name}

Output a valid JSON object with exactly three fields:
{{
  "dominant_pattern": "1-2 sentences: the most meaningful cross-domain pattern happening today",
  "cross_domain": "1 sentence: how the domains are currently interacting (health ↔ productivity ↔ calendar ↔ mood)",
  "forward_context": "1 sentence: the most important thing ahead (next event, free window, bedtime proximity)"
}}

Rules:
- Be factual and specific — name the actual signals, not generic labels.
- Do NOT make suggestions. This is analysis only.
- Keep total output under 150 tokens.
- Respond with valid JSON only.
- If data is missing for a field, write a brief honest statement ("Insufficient data to assess X").
"""

    user_prompt = f"""Synthesize a day profile from the following data:

**Time context:**
{time_lines if time_lines else "No time data available."}

**Health signals:**
{health_lines if health_lines else "No health data available."}

**Calendar load:**
{calendar_lines if calendar_lines else "No calendar metrics available."}

**{events_section}**

Return JSON with dominant_pattern, cross_domain, and forward_context."""

    return system_prompt, user_prompt
