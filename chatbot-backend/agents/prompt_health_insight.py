"""Health insight prompts with dedicated group prompt builders."""

import json
from typing import Any, Dict, Optional

from agents.prompt import get_language_name, important_language_prompt
from agents.prompt_insight_reasoning import format_for_point12 as _insight_block
from services.executor.constant import HealthDataConstants, TimeDataConstants

HEALTH_EVENING_GROUPS = frozenset(
    {
        "wind_down_window",
        "bedtime_window",
        "evening_flexible",
    }
)

_HEALTH_OUTPUT_RULES = """
**HEALTH OUTPUT RULES (ALWAYS APPLY):**
- Return valid JSON only: {"point1": "...", "point2": "..."}.
- Each point 1-2 sentences; evidence-based; no medical diagnosis.
- Follow mood freshness rules from system prompt.
- Respect `is_daytime` in Time context (see user message): when true, point2 must NOT reference tonight's sleep routine.
"""

_HEALTH_TIME_ACTIONABILITY_BY_GROUP: Dict[str, str] = {
    "wind_down_window": """
**TIME ACTIONABILITY (wind_down_window — evening):**
- point2 MAY suggest calm pre-sleep actions, wind-down, mood logging before sleep if relevant.""",
    "bedtime_window": """
**TIME ACTIONABILITY (bedtime_window — evening):**
- point2 MAY suggest rest, sleep timing, bedtime routine for tonight.""",
    "evening_flexible": """
**TIME ACTIONABILITY (evening_flexible — evening):**
- point2 = light evening energy (walk, water, meal timing); sleep prep ONLY if time_to_bedtime <= 60.
- FORBIDDEN as default point2: meditation, mindfulness, "transition to sleep", long stretch routines.""",
    "active_window": """
**TIME ACTIONABILITY (active_window — daytime):**
- point2: in-day actions only (steps, walk, hydration, micro-breaks). FORBIDDEN: before bed, wind-down tonight, sleep prep tonight, log mood before sleep.""",
    "morning_window": """
**TIME ACTIONABILITY (morning_window):**
- point2: morning habits only. FORBIDDEN: tonight bedtime routine. Sleep in point1 = last night / recovery only.""",
    "health_goal": """
**TIME ACTIONABILITY (health_goal):**
- If is_daytime=true: catch-up steps/movement only in point2; FORBIDDEN sleep routine tonight.
- If is_daytime=false: may include gentle sleep-timing guidance in point2.""",
    "contextual_suggestion": """
**TIME ACTIONABILITY (contextual_suggestion):**
- If is_daytime=true: choose highest-impact IN-DAY health action; FORBIDDEN bedtime/wind-down wording in point2.""",
    "calendar_pattern": """
**TIME ACTIONABILITY (calendar_pattern — daytime):**
- point2: schedule/energy framing for today; FORBIDDEN before-bed sleep prep.""",
    "recovery_break": """
**TIME ACTIONABILITY (recovery_break — daytime):**
- point2: break/recovery now; FORBIDDEN linking to bedtime tonight.""",
    "task_type": """
**TIME ACTIONABILITY (task_type — daytime):**
- point2: task/energy match for current hours; FORBIDDEN before sleep.""",
    "weekend_lifestyle": """
**TIME ACTIONABILITY (weekend_lifestyle):**
- point2: weekend reset; sleep consistency as habit OK but not "tonight before bed" unless is_daytime=false.""",
    "safety_risk": """
**TIME ACTIONABILITY (safety_risk):**
- Safety first. Sleep debt as risk OK in point1/point2; tonight sleep routine only if is_daytime=false.""",
}

_HEALTH_DAYTIME_POINT2_FORBIDDEN = """
**DAYTIME point2 FORBIDDEN PHRASES (when is_daytime=true):**
- "trước khi ngủ", "before bed", "before sleep", "wind-down", "wind down", "chuẩn bị giấc ngủ", "đi ngủ", "sleep tonight", "log mood before bed", "ghi cảm xúc trước khi ngủ", "thư giãn ... trước khi ngủ".
**DAYTIME point2 ALLOWED:** walking/steps in the current part of day, hydration, breaks, HR/steps progress — not tied to bedtime.
**DAYTIME point1:** May mention sleep GOAL or last night's sleep; do NOT prescribe tonight's bedtime routine.
"""


def _is_health_daytime(group: str, time_data: Dict[str, Any]) -> bool:
    """True when health point2 should avoid sleep/bedtime routine language."""
    if group in HEALTH_EVENING_GROUPS:
        return False
    hour = time_data.get(TimeDataConstants.KEY_CURRENT_HOUR)
    if hour is not None and int(hour) >= 19:
        return False
    return True


def _get_shared_health_rules_block(group: str) -> str:
    time_rules = _HEALTH_TIME_ACTIONABILITY_BY_GROUP.get(
        group, _HEALTH_DAYTIME_POINT2_FORBIDDEN
    )
    return (
        _HEALTH_OUTPUT_RULES
        + time_rules
        + _HEALTH_DAYTIME_POINT2_FORBIDDEN
        + f"\n**Detected health insight group:** `{group}`\n"
    )


def _build_time_context_block(group: str, time_data: Dict[str, Any]) -> str:
    is_daytime = _is_health_daytime(group, time_data)
    hour = time_data.get(TimeDataConstants.KEY_CURRENT_HOUR)
    in_active = time_data.get(TimeDataConstants.KEY_IN_ACTIVE_WINDOW)
    time_to_bed = time_data.get(TimeDataConstants.KEY_TIME_TO_BEDTIME)
    lines = [
        f"- insight_group: {group}",
        f"- is_daytime: {is_daytime} (if true, point2 must NOT suggest sleep/bedtime routine for tonight)",
        f"- current_hour: {hour if hour is not None else 'unknown'}",
        f"- in_active_window: {in_active if in_active is not None else 'unknown'}",
        f"- time_to_bedtime_minutes: {time_to_bed if time_to_bed is not None else 'unknown'}",
    ]
    return "**Time context:**\n" + "\n".join(lines)


def _has_value(value: Any) -> bool:
    return value is not None and value != ""


def _build_health_snapshot(health_params: Dict[str, Any]) -> str:
    fields = [
        ("Steps today", HealthDataConstants.KEY_STEPS_TODAY),
        ("Steps goal", HealthDataConstants.KEY_STEPS_GOAL),
        ("Remaining steps", HealthDataConstants.KEY_REMAINING_STEPS),
        ("Sleep last night (hours)", HealthDataConstants.KEY_SLEEP_LASTNIGHT),
        ("Sleep goal (hours)", HealthDataConstants.KEY_SLEEP_GOAL),
        ("Sleep quality score", HealthDataConstants.KEY_SLEEP_QUALITY_SCORE),
        ("Sleep quality", HealthDataConstants.KEY_SLEEP_QUALITY),
        ("Latest heart rate", HealthDataConstants.KEY_LATEST_HEART_RATE),
        ("Resting heart rate", HealthDataConstants.KEY_RESTING_HEART_RATE),
        ("Baseline resting HR", HealthDataConstants.KEY_BASELINE_RESTING_HR),
        ("Last wake time", HealthDataConstants.KEY_LAST_WAKE_TIME),
        ("First sleep time", HealthDataConstants.KEY_FIRST_SLEEP_TIME),
        ("Bedtime streak", HealthDataConstants.KEY_BEDTIME_STREAK),
        ("Sleep last 3 nights", HealthDataConstants.KEY_SLEEP_LAST3NIGHTS),
        ("All 3 nights have data", HealthDataConstants.KEY_ALL_3_NIGHTS_HAVE_DATA),
        ("Steps streak", HealthDataConstants.KEY_STEPS_STREAK),
        ("Workout heart rate", HealthDataConstants.KEY_WORKOUT_HEART_RATE),
        ("Workout avg HR", HealthDataConstants.KEY_WORKOUT_AVG_HR),
        ("Workout max HR", HealthDataConstants.KEY_WORKOUT_MAX_HR),
        ("Workout min HR", HealthDataConstants.KEY_WORKOUT_MIN_HR),
        ("Workout duration", HealthDataConstants.KEY_WORKOUT_DURATION),
        ("Workout activity type", HealthDataConstants.KEY_WORKOUT_ACTIVITY_TYPE),
        ("Weekly health progress", HealthDataConstants.KEY_WEEKLY_HEALTH_PROGRESS),
        ("Daily health progress", HealthDataConstants.KEY_DAILY_HEALTH_PROGRESS),
        ("Wind-down buffer (mins)", HealthDataConstants.KEY_WIND_DOWN_BUFFER_MINS),
    ]

    lines = []
    for label, key in fields:
        value = health_params.get(key)
        if _has_value(value):
            lines.append(f"- {label}: {value}")

    if not lines:
        return "No health snapshot data available."
    return "\n".join(lines)


def _build_mood_snapshot(latest_mood: Dict[str, Any]) -> Dict[str, Any]:
    # `not_logged_today` is set upstream when the user hasn't logged a mood
    # for the current local date. Surface it so the prompt can softly nudge
    # the user (see rule below) instead of relying on stale moods.
    return {
        "mood": latest_mood.get("mood") or latest_mood.get("current_mood"),
        "notes": latest_mood.get("notes"),
        "not_logged_today": bool(latest_mood.get("not_logged_today", False)),
    }


def _build_system_prompt(
    *,
    role: str,
    rules_title: str,
    rules_text: str,
    point1_instruction: str,
    point2_instruction: str,
    language_name: str,
    alignment_context: str,
) -> str:
    return f"""You are a {role}. Analyze the user's health context and provide insights in exactly two points.

Return valid JSON only with exactly two keys:
{{"point1": "...", "point2": "..."}}

Output constraints:
- point1 and point2 must each be 1-2 sentences.
- Do not diagnose disease or make medical claims.
- Reflect mood when available (empathetic for negative mood).
- Mood input shows `mood: null` and `not_logged_today: true` when the user has not logged a mood for today. In that case:
  - Do NOT infer or assume a current mood.
  - You MAY occasionally include a soft, optional reminder to log mood (preferably inside point2, at most one short clause). Skip it when point2 already has a more relevant health action.
  - Never make the mood-log reminder the main focus, and never repeat it across point1 and point2.
- Avoid exact clock times; use qualitative timing.
- Keep recommendations feasible and low-friction.
- Health-only scope: point1/point2 must focus only on health signals and health actions.
- Do NOT mention calendar/schedule/meeting/inbox/task-planning details in the final answer.
- If alignment context includes calendar facts, use them only as background reasoning and convert them into health framing (stress, recovery load, energy), without mentioning calendar entities directly.

**{rules_title}:**
{rules_text}

**Point 1 (MANDATORY STRUCTURE):**
- Summarize key health highlights from available data (heart rate, steps, sleep last night, recovery/progress).
- MUST include both: (1) what is going well (2) gaps / below target.
- If Time context shows is_daytime=true: sleep may appear as LAST NIGHT or as a GOAL only — not tonight's bedtime routine.
- If is_daytime=false (evening groups): sleep/wind-down context is allowed in point1.
- Keep it concise (1-2 sentences). Stay factual; no diagnosis.
- Group-specific focus for point1: {point1_instruction}

**Point 2 (MANDATORY STRUCTURE):**
- MUST align with the alignment context (overall/productivity insights) and avoid any contradiction.
- SHOULD provide practical health suggestions based on the user's current health data.
- MAY add additional necessary suggestions that were not mentioned before, as long as they are consistent with alignment context.
- Keep suggestions actionable, realistic, and supportive (1-2 sentences).
- MUST remain health-only and avoid explicit calendar/schedule references.
- Group-specific focus for point2: {point2_instruction}

IMPORTANT TIME RULES:
- DO NOT mention or restate the exact current time.
- DO NOT use urgency phrases like "right now" or "at this moment".
- Describe timing qualitatively.

{alignment_context}

{important_language_prompt(language_name)}
"""


def _build_user_prompt(
    *,
    group: str,
    current_time: str,
    timezone: Optional[str],
    mood_snapshot: Dict[str, Any],
    health_snapshot: str,
    time_data: Dict[str, Any],
    calendar_metrics: Dict[str, Any],
    user_profile: Dict[str, Any],
    extra_focus: str,
    overall_context: str,
    productivity_context: str,
) -> str:
    time_context_block = _build_time_context_block(group, time_data)
    return f"""Use the input below to produce health insight JSON.

Group-specific focus:
{extra_focus}

{time_context_block}

Current time: {current_time}
Timezone: {timezone or "UTC"}

Mood:
{json.dumps(mood_snapshot, ensure_ascii=False, indent=2)}

Health:
{health_snapshot}

User profile:
{json.dumps(user_profile, ensure_ascii=False, indent=2)}

Overall insight context (if available):
{overall_context or "N/A"}

Productivity insight context (if available):
{productivity_context or "N/A"}

Return JSON only:
{{
  "point1": "health summary",
  "point2": "suggestion"
}}"""


def _build_group_prompt(
    *,
    group: str = "contextual_suggestion",
    role: str,
    rules_title: str,
    rules_text: str,
    point1_instruction: str,
    point2_instruction: str,
    extra_focus: str,
    health_params: Dict[str, Any],
    latest_mood: Dict[str, Any],
    user_profile: Dict[str, Any],
    time_data: Dict[str, Any],
    calendar_metrics: Dict[str, Any],
    current_time: str,
    timezone: Optional[str],
    language: Optional[str],
    overall_context: str,
    productivity_context: str,
) -> tuple[str, str]:
    language_name = get_language_name(language)
    is_daytime = _is_health_daytime(group, time_data)
    alignment_context = _build_alignment_context_prompt(
        overall_context=overall_context,
        productivity_context=productivity_context,
        is_daytime=is_daytime,
    )
    system_prompt = _build_system_prompt(
        role=role,
        rules_title=rules_title,
        rules_text=rules_text,
        point1_instruction=point1_instruction,
        point2_instruction=point2_instruction,
        language_name=language_name,
        alignment_context=alignment_context,
    )
    user_prompt = _build_user_prompt(
        group=group,
        current_time=current_time,
        timezone=timezone,
        mood_snapshot=_build_mood_snapshot(latest_mood),
        health_snapshot=_build_health_snapshot(health_params),
        time_data=time_data,
        calendar_metrics=calendar_metrics,
        user_profile=user_profile,
        extra_focus=extra_focus,
        overall_context=overall_context,
        productivity_context=productivity_context,
    )
    return system_prompt, user_prompt


def _build_alignment_context_prompt(
    overall_context: Optional[str] = "",
    productivity_context: Optional[str] = "",
    is_daytime: bool = True,
) -> str:
    if not overall_context and not productivity_context:
        return ""
    section = "\n**Cross-Insight Consistency Context (MUST ALIGN WITH):**\n"
    if overall_context:
        section += f"- Overall insight:\n{overall_context}\n"
    if productivity_context:
        section += f"- Productivity insight:\n{productivity_context}\n"
    section += (
        "\n**CRITICAL CONSISTENCY REQUIREMENTS:**\n"
        "- DO NOT conflict with overall/productivity insights\n"
        "- DO NOT contradict existing facts already communicated to the user\n"
        "- Keep health guidance complementary and coherent with prior insights\n"
        "- Avoid redundant repetition; add health-specific value while staying aligned\n"
    )
    if is_daytime:
        section += (
            "- When is_daytime=true: use overall/productivity only for consistency — "
            "do NOT copy evening sleep, wind-down, or 'before bed' suggestions into point2.\n"
        )
    return section


def get_health_safety_risk_prompt(**kwargs) -> tuple[str, str]:
    return _build_group_prompt(
        role="safety and health risk detector",
        rules_title="Safety Risk Rules",
        rules_text=f"""1. PRIORITY FIRST: Always prioritize safety and physiological risk over optimization.
2. RISK TRIGGERS: Pay attention to severe sleep debt, abnormal heart-rate signals, persistent fatigue, or overload patterns.
3. TONE CONTROL: Keep tone calm, protective, and non-alarming; avoid panic language.
4. ACTION STYLE: Recommend low-risk immediate steps (rest, hydration, workload reduction, light movement only if safe).
5. ESCALATION: If signs look high-risk, suggest professional support in a cautious and non-diagnostic way.
6. NO OVER-PROMISE: Do not claim certainty or medical conclusions.""",
        point1_instruction="Summarize key risk signals and why they matter now (1-2 sentences).",
        point2_instruction="Provide immediate but gentle risk-mitigation guidance (1-2 sentences).",
        extra_focus="Focus on safety and recovery protection first.",
        **kwargs,
    )


def get_health_wind_down_window_prompt(**kwargs) -> tuple[str, str]:
    return _build_group_prompt(
        role="wind-down health advisor",
        rules_title="Wind-Down Rules",
        rules_text=f"""1. WINDOW INTENT: Focus on transition from activity mode to recovery mode.
2. STIMULATION CONTROL: Recommend reducing cognitive/sensory load (screens, intense tasks, heavy discussions).
3. SLEEP PROTECTION: Prioritize sleep quality over late productivity gains.
4. ACTION QUALITY: Prefer calming, low-effort actions.
5. REALISM: Suggest practical actions user can apply in short time.
6. CONSISTENCY: Reinforce repeatable pre-sleep routines for long-term benefit.
7. ACTION DIVERSITY (CRITICAL): The user may have already received breathing/stretching/walking suggestions earlier today. You MUST rotate to a DIFFERENT family of actions. Examples of distinct families:
   - sensory: warm drink, herbal tea, scent/aroma (candle, diffuser)
   - environment: dim lights, cool/dark room, screen-off ritual, fresh air
   - body: warm shower, light stretch (only if not recently suggested), journaling/voice memo, gratitude note
   - sound: ambient music, lo-fi playlist, calm podcast, white noise
   - reflection: brief gratitude note, brain dump to clear mind, plan tomorrow briefly then stop
   Pick one family that the user has likely NOT heard recently. NEVER default to "hít thở sâu" (deep breathing) or "đi bộ nhẹ quanh nhà" (walk around the house) if these have been suggested today.""",
        point1_instruction="Summarize wind-down readiness and current recovery context (1-2 sentences).",
        point2_instruction="Suggest ONE calm pre-sleep action from a DIFFERENT family than recent suggestions (1-2 sentences). Be specific — name the exact drink, music, or ritual.",
        extra_focus="Evening wind-down window only — pre-sleep actions allowed. Rotate action type from prior suggestions.",
        **kwargs,
    )


def get_health_bedtime_window_prompt(**kwargs) -> tuple[str, str]:
    return _build_group_prompt(
        role="bedtime consistency coach",
        rules_title="Bedtime Rules",
        rules_text=f"""1. TIMING CHECK: Evaluate bedtime alignment with user's typical sleep targets/routine.
2. RECOVERY LINK: Explain how bedtime consistency affects next-day energy and recovery.
3. NON-JUDGMENTAL: Use supportive coaching language, avoid guilt framing.
4. TONIGHT PLAN: Recommend one realistic corrective step for tonight only.
5. STABILITY OVER INTENSITY: Favor consistency improvements over drastic changes.
6. AVOID MEDICAL CLAIMS: Keep suggestions behavioral, not diagnostic.""",
        point1_instruction="Summarize bedtime alignment or misalignment (1-2 sentences).",
        point2_instruction="Suggest one practical step to improve sleep timing (1-2 sentences).",
        extra_focus="Evening bedtime window only — sleep routine for tonight is allowed.",
        **kwargs,
    )


def get_health_morning_window_prompt(**kwargs) -> tuple[str, str]:
    return _build_group_prompt(
        role="morning recovery planner",
        rules_title="Morning Start Rules",
        rules_text=f"""1. READINESS FIRST: Infer morning readiness from sleep/recovery indicators before suggesting intensity.
2. ENERGY RAMP: Favor gradual energy ramp-up and stable pacing.
3. LOW-RECOVERY MODE: If sleep/recovery is low, recommend lighter start and fewer high-load tasks.
4. HEALTH ANCHORS: Include hydration, movement, and recovery-supportive morning habits where relevant.
5. PRACTICALITY: Keep recommendations simple and executable.
6. SUSTAINABILITY: Optimize for all-day stability, not short spikes.""",
        point1_instruction="Summarize morning readiness from sleep and recovery signals (1-2 sentences).",
        point2_instruction="Suggest a healthy morning plan for stable energy — not tonight's bedtime.",
        extra_focus="Morning only; sleep refers to last night or recovery, not evening routine.",
        **kwargs,
    )


def get_health_active_window_prompt(**kwargs) -> tuple[str, str]:
    return _build_group_prompt(
        role="active window health balancer",
        rules_title="Active Window Rules",
        rules_text=f"""1. BALANCE RULE: Balance output demands with recovery capacity during active hours.
2. LOAD AWARENESS: Factor in meeting density, fragmentation, and limited buffer time.
3. FATIGUE PREVENTION: Suggest micro-breaks, hydration, posture resets, and pacing when strain increases.
4. INTENSITY MATCH: Match task intensity to current physiological/mental capacity.
5. ACTIONABLE GUIDANCE: Give one concrete in-day adjustment (steps, walk, hydration) — never bedtime or wind-down tonight.
6. DO NO HARM: Avoid recommendations that increase overload risk.""",
        point1_instruction="Summarize current strain/load versus health capacity (1-2 sentences).",
        point2_instruction="Suggest one in-day action (e.g. walk, steps, break) feasible now — never 'before bed' or sleep prep tonight.",
        extra_focus="Active hours: in-day health only; respect is_daytime in Time context.",
        **kwargs,
    )


def get_health_calendar_pattern_prompt(**kwargs) -> tuple[str, str]:
    return _build_group_prompt(
        role="calendar-health pattern analyzer",
        rules_title="Calendar Pattern Rules",
        rules_text=f"""1. PATTERN ANALYSIS: Interpret calendar structure through health impact (energy drain, stress, recovery debt).
2. RISK PATTERNS: Highlight prolonged meeting streaks, fragmentation, and back-to-back overload.
3. HEALTH COST: Connect schedule patterns to likely effects on focus, fatigue, and recovery.
4. RESTRUCTURE STRATEGY: Suggest healthier schedule shaping (buffers, focus blocks, break slots).
5. REAL-WORLD FEASIBILITY: Recommendations must be practical with current constraints.
6. PRIORITIZE IMPACT: Emphasize one highest-impact schedule change.""",
        point1_instruction="Summarize health impact from calendar patterns (1-2 sentences).",
        point2_instruction="Suggest schedule optimization to reduce health friction (1-2 sentences).",
        extra_focus="Translate calendar load patterns into recovery consequences.",
        **kwargs,
    )


def get_health_goal_progress_prompt(**kwargs) -> tuple[str, str]:
    return _build_group_prompt(
        role="health goal progress tracker",
        rules_title="Health Goal Rules",
        rules_text=f"""1. GOAL COMPARISON: Compare current metrics with health goals (steps, sleep, recovery trend).
2. STATUS CLARITY: Clearly indicate whether user is on track, near target, or behind.
3. POSITIVE REINFORCEMENT: Celebrate wins when progress is strong.
4. CATCH-UP DESIGN: If behind and is_daytime=true, suggest steps/movement catch-up only — NOT sleep routine tonight.
5. If is_daytime=false: may include gentle sleep-timing guidance in point2.
6. AVOID EXTREMES: Do not propose aggressive compensation behaviors.""",
        point1_instruction="Summarize goal progress (steps and sleep as goals or last night — not tonight routine if daytime).",
        point2_instruction="If is_daytime: steps/walk catch-up only. If evening: may add sleep prep if time_to_bedtime is short.",
        extra_focus="Respect is_daytime: no 'before bed' wording during day.",
        **kwargs,
    )


def get_health_evening_flexible_prompt(**kwargs) -> tuple[str, str]:

    system, user = _build_group_prompt(
        role="evening flexible health coach",
        rules_title="Evening Flexible Rules",
        rules_text=f"""1. EVENING ≠ BEDTIME: Before wind-down, focus on hydration, light movement, posture — not sleep rituals.
2. UPCOMING EVENTS: If event soon, suggest physical readiness (water, brief walk, breathing) — not meditation or stretch-as-default.
3. SLEEP DEBT: Mention in point1 only; point2 = meal/walk/hydration unless time_to_bedtime is short.
4. SMALL-WIN: One concrete body action (not a list of wellness options).
5. DIFFERENTIATE: Do not repeat productivity/overall wording (no duplicate "wind down before sleep").""",
        point1_instruction="One causal pattern linking mood, steps, sleep, and today's load (1-2 sentences).",
        point2_instruction="One body-focused action for this evening moment (hydration, short walk, posture) — not mindfulness unless wind-down imminent.",
        extra_focus="Evening flexible — living evening, not pre-sleep routine.",
        **kwargs,
    )
    block = _insight_block("evening_flexible")
    return f"{system}\n\n{block}", user


def get_health_recovery_break_prompt(**kwargs) -> tuple[str, str]:
    return _build_group_prompt(
        role="recovery and break advisor",
        rules_title="Recovery Break Rules",
        rules_text=f"""1. OVERLOAD DETECTION: Identify signs of cumulative strain (long focus streaks, meeting stacks, low recovery signals).
2. EARLY INTERVENTION: Encourage short breaks before fatigue escalates.
3. BREAK QUALITY: Prefer restorative break types (movement, breathing, hydration, visual reset).
4. SPECIFICITY: Recommend concrete break actions, not vague advice.
5. CONTEXT MATCH: Adjust suggestion to available time and workload constraints.
6. SUSTAINED PERFORMANCE: Frame recovery as performance protection, not interruption.""",
        point1_instruction="Summarize current recovery need (1-2 sentences).",
        point2_instruction="Suggest a concrete break/recovery intervention (1-2 sentences).",
        extra_focus="Target overload prevention and short recovery loops.",
        **kwargs,
    )


def get_health_task_type_prompt(**kwargs) -> tuple[str, str]:
    return _build_group_prompt(
        role="energy-task matching advisor",
        rules_title="Task Type Rules",
        rules_text=f"""1. ENERGY-TASK MATCH: Align task complexity with current energy/recovery state.
2. LOW-ENERGY SAFETY: In low-energy state, shift to lighter, structured, lower-stress tasks.
3. HIGH-ENERGY USE: In strong state, allow deeper work while preserving recovery buffers.
4. MISMATCH AVOIDANCE: Explicitly avoid recommendations that can worsen fatigue.
5. PRACTICAL REPLAN: Provide one actionable task-shaping recommendation.
6. HEALTH PRESERVATION: Maintain sustainable effort over peak effort.""",
        point1_instruction="Summarize energy-task fit context (1-2 sentences).",
        point2_instruction="Suggest task style/intensity that protects health (1-2 sentences).",
        extra_focus="Optimize task load for health sustainability.",
        **kwargs,
    )


def get_health_weekend_lifestyle_prompt(**kwargs) -> tuple[str, str]:
    return _build_group_prompt(
        role="weekend lifestyle optimizer",
        rules_title="Weekend Lifestyle Rules",
        rules_text=f"""1. LIFESTYLE BALANCE: Balance restoration, movement, social/personal life, and optional productivity.
2. RESET PRIORITY: Encourage weekly reset behaviors that improve next-week readiness.
3. OVERLOAD GUARDRAIL: Avoid over-scheduling or high-pressure weekend plans.
4. HOLISTIC HEALTH: Include sleep consistency, light activity, and mental decompression.
5. LOW-FRICTION EXECUTION: Keep suggestions realistic and enjoyable.
6. SUSTAINABILITY: Favor repeatable lifestyle patterns over one-off intensity.""",
        point1_instruction="Summarize weekend lifestyle balance from health view (1-2 sentences).",
        point2_instruction="Suggest one restorative weekend-oriented action (1-2 sentences).",
        extra_focus="Lean toward reset quality and lifestyle sustainability.",
        **kwargs,
    )


def get_health_contextual_suggestion_prompt(**kwargs) -> tuple[str, str]:
    return _build_group_prompt(
        role="contextual health advisor",
        rules_title="Contextual Health Rules",
        rules_text=f"""1. CONTEXT SYNTHESIS: Use mood, health metrics, Time context (is_daytime), and signals together.
2. IMPACT PRIORITY: Choose the single most impactful health angle for this moment.
3. If is_daytime=true: point2 must be an in-day action — never default to sleep/bedtime tonight.
4. ACTION FEASIBILITY: Recommendations must be immediately doable at current time of day.
5. SUSTAINABLE CHANGE: Prefer steady behavior improvements over extreme interventions.
6. FALLBACK LOGIC: If data is sparse, provide safe in-day health guidance when is_daytime=true.""",
        point1_instruction="Summarize most relevant current health context (1-2 sentences).",
        point2_instruction="Best next-step for NOW: if is_daytime, steps/movement/hydration — not wind-down or before bed.",
        extra_focus="Use is_daytime flag; do not mention sleep prep tonight during day.",
        **kwargs,
    )


def get_health_insight_prompt(
    group: str,
    health_params: Optional[Dict[str, Any]],
    latest_mood: Optional[Dict[str, Any]],
    user_profile: Optional[Dict[str, Any]],
    time_data: Optional[Dict[str, Any]],
    calendar_metrics: Optional[Dict[str, Any]],
    current_time: str,
    timezone: Optional[str] = None,
    language: Optional[str] = None,
    overall_context: Optional[str] = "",
    productivity_context: Optional[str] = "",
) -> tuple[str, str]:
    """Dispatch to dedicated group prompt builders (health insight)."""
    kwargs = {
        "health_params": health_params or {},
        "latest_mood": latest_mood or {},
        "user_profile": user_profile or {},
        "time_data": time_data or {},
        "calendar_metrics": calendar_metrics or {},
        "current_time": current_time,
        "timezone": timezone,
        "language": language,
        "overall_context": overall_context or "",
        "productivity_context": productivity_context or "",
    }
    prompt_by_group = {
        "safety_risk": get_health_safety_risk_prompt,
        "wind_down_window": get_health_wind_down_window_prompt,
        "bedtime_window": get_health_bedtime_window_prompt,
        "morning_window": get_health_morning_window_prompt,
        "active_window": get_health_active_window_prompt,
        "calendar_pattern": get_health_calendar_pattern_prompt,
        "health_goal": get_health_goal_progress_prompt,
        "evening_flexible": get_health_evening_flexible_prompt,
        "recovery_break": get_health_recovery_break_prompt,
        "task_type": get_health_task_type_prompt,
        "weekend_lifestyle": get_health_weekend_lifestyle_prompt,
        "contextual_suggestion": get_health_contextual_suggestion_prompt,
    }
    kwargs["group"] = group
    builder = prompt_by_group.get(group, get_health_contextual_suggestion_prompt)
    system_prompt, user_prompt = builder(**kwargs)
    system_prompt = system_prompt + "\n" + _get_shared_health_rules_block(group)
    # Inject universal reasoning block for all groups
    reasoning = _insight_block(group)
    if reasoning:
        system_prompt = system_prompt + "\n\n" + reasoning
    return system_prompt, user_prompt
