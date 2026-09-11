"""Universal insight reasoning framework for all insight groups across the full day.

This module provides data-driven reasoning lenses, constraint guards, cross-domain
coherence rules, and suggestion palettes that are injected into every insight prompt
(productivity, overall, health). It replaces narrow per-group reasoning frames with
a universal approach where the LLM picks the most meaningful lens based on actual data.

Evening-specific extras (SUGGESTION_CONTEXT_PICKER, EVENING_FLEXIBLE_ACTION_GUARD,
and the original reasoning texts for wind_down/bedtime/evening groups) are imported
from prompt_evening_reasoning.py and appended on top of the universal blocks.
"""

from typing import Dict, Optional

from agents.prompt_evening_reasoning import (
    _OVERALL_CATEGORY_GUIDANCE as _EVENING_OVERALL_CATEGORY_GUIDANCE,
)
from agents.prompt_evening_reasoning import (
    AFTER_WORK_REASONING,
    BEDTIME_REASONING,
    EVENING_FLEXIBLE_ACTION_GUARD,
    EVENING_REASONING,
    SUGGESTION_CONTEXT_PICKER,
    SUGGESTION_PALETTE_AFTER_WORK,
    SUGGESTION_PALETTE_BEDTIME,
    SUGGESTION_PALETTE_EVENING,
    SUGGESTION_PALETTE_WIND_DOWN,
    WIND_DOWN_REASONING,
)

# ---------------------------------------------------------------------------
# Block 1: UNIVERSAL REASONING LENSES
# ---------------------------------------------------------------------------

UNIVERSAL_REASONING_LENSES = """
**UNIVERSAL REASONING LENSES — pick 1-2 that best match the actual data:**

Examine the available data through these lenses. Pick the 1-2 that reveal the most
meaningful pattern for this user right now. Do NOT default to the most obvious or generic one.

1. **Trade-off lens**: Something improved while another area was compromised.
   Examples: busy meetings → fewer steps; high output day → no recovery; good health day → low focus time.

2. **Reinforcing lens**: One positive behavior supported another, compounding benefit.
   Examples: good sleep → better focus → more steps; recovery break → sustained output later.

3. **Hidden imbalance lens**: Surface looks fine but one area is quietly underserved.
   Examples: steps goal met but resting HR elevated; productive day but zero recovery; good mood but sleep debt building.

4. **Constraint lens**: Calendar, energy, mood, or health limited what was actually possible today.
   Examples: meeting stack prevented deep work; sleep debt capped cognitive capacity; mood constrained activity choice.

5. **Momentum lens**: Multiple signals compounding in the same direction — the trend matters more than today's snapshot.
   Examples: steps streak + consistent bedtime = virtuous cycle building; 3-day back-to-back overload = escalating strain.

6. **Carry-over lens**: Today's pattern will create a specific effect tomorrow if nothing changes.
   Examples: sleep debt → tomorrow's focus at risk; meeting-heavy day → recovery need will carry into tomorrow.

7. **Recovery readiness lens**: Is the body/mind in a state where more output helps or hurts?
   Examples: high resting HR + back-to-back meetings → more load = diminishing returns; well-recovered → safe to push.

8. **Goal alignment lens**: Is what's happening aligned with this user's own stated goals?
   Examples: active person with near-zero steps today; step goal met but sleep goal far off; weekend work against rest preference.

**Reasoning rules:**
- Explain WHY the pattern is happening, not just WHAT is happening.
- Prefer the lens that produces the most non-obvious, specific insight for this data.
- If two lenses are equally strong, layer them: "X is happening [lens A] because of Y, which will affect Z tomorrow [lens B]."
- Never report raw metrics. Never say "your steps are 4,000 today." Say what the pattern means.
"""

# ---------------------------------------------------------------------------
# Block 2: GROUP SIGNAL WEIGHTS
# ---------------------------------------------------------------------------

_GROUP_SIGNAL_WEIGHTS: Dict[str, str] = {
    "morning_window": (
        "**SIGNAL PRIORITY (morning):** Sleep quality from last night and timing of the first event "
        "carry the most context. Everything else builds from the recovery state the user is starting with."
    ),
    "active_window": (
        "**SIGNAL PRIORITY (active window):** Free slot length and meeting density shape what is actually "
        "achievable right now. Energy and mood signals determine whether that capacity can be used well."
    ),
    "wind_down_window": (
        "**SIGNAL PRIORITY (wind-down):** The user can no longer meaningfully change today's outcomes. "
        "Recovery readiness and what has been accumulated today are the primary signals."
    ),
    "bedtime_window": (
        "**SIGNAL PRIORITY (bedtime):** Sleep timing consistency and the 3-night sleep debt trend carry "
        "more weight than any single metric."
    ),
    "health_goal": (
        "**SIGNAL PRIORITY (health goal):** Goal progress percentage and available time window together "
        "determine urgency. Sleep quality and mood modulate how hard to push."
    ),
    "recovery_break": (
        "**SIGNAL PRIORITY (recovery break):** Cumulative load signals (back_to_back, continuous_events, "
        "breaks_today) are the primary signals. Calendar structure explains WHY the load built."
    ),
    "calendar_pattern": (
        "**SIGNAL PRIORITY (calendar pattern):** The structural shape of the day (density, fragmentation, "
        "back-to-back chains) is the signal. Health and mood state determine the impact severity."
    ),
    "task_type": (
        "**SIGNAL PRIORITY (task type):** Available slot length and proximity to next event are hard constraints. "
        "Energy and recovery state determine what task type fits within those constraints."
    ),
    "evening_flexible": (
        "**SIGNAL PRIORITY (evening flexible):** Time remaining until wind-down and the shape of the day "
        "(load, mood, sleep debt) together determine what kind of evening activity is appropriate."
    ),
    "evening_after_work": (
        "**SIGNAL PRIORITY (evening after work):** The transition from work to personal time is the signal. "
        "Mood and remaining energy determine what kind of recovery or engagement fits."
    ),
    "weekend_lifestyle": (
        "**SIGNAL PRIORITY (weekend lifestyle):** The balance between work bleed, recovery activity, and "
        "personal engagement signals are primary. Weekday health trends provide the 'why it matters' context."
    ),
    "mood_based": (
        "**SIGNAL PRIORITY (mood-based):** Mood state is the primary signal and acts as a constraint override "
        "for all other suggestions. Cross-check with HR and sleep to confirm the mood interpretation."
    ),
    "safety_risk": (
        "**SIGNAL PRIORITY (safety risk):** Body signals (HR, resting HR vs baseline, sleep debt) take full "
        "priority. No other lens applies — safety first, everything else deferred."
    ),
    "contextual_suggestion": (
        "**SIGNAL PRIORITY (contextual):** No single signal dominates — find the cross-domain pattern that "
        "has the most forward impact if addressed now. Avoid listing; synthesize into ONE observation."
    ),
}


def _get_signal_weight(group: str) -> str:
    return _GROUP_SIGNAL_WEIGHTS.get(group, "")


# ---------------------------------------------------------------------------
# Block 3: NOT_DO CONSTRAINT GUARDS
# ---------------------------------------------------------------------------

NOT_DO_CONSTRAINT_GUARDS = """
**NOT_DO CONSTRAINT GUARDS — check ALL before suggesting anything:**

TIME CONSTRAINTS:
- Never suggest a task or activity in a past time slot.
- Never suggest anything that overlaps an existing calendar event.
- Never suggest a task longer than the available free slot.
- No deep work or focus suggestions if free slot < 45 min (suggest admin/prep/micro-task instead).
- No heavy tasks (deep work, intense study, hard conversations) within 120 min of bedtime.
- No new commitments (new project, meeting, big planning) within 90 min of bedtime.
- No inbox/admin/stimulating media during wind-down window (outside active hours).
- No deep work suggestions outside active hours window.

HEALTH CONSTRAINTS:
- If HR < 40 OR HR > 180: block ALL non-safety suggestions; safety guidance only.
- If resting HR > baseline + delta OR stress_signal_high: block intense exercise; suggest breathing/walk/hydration.
- If sleep_last3nights < sleep_goal AND current time >= 18:00: no late intense workouts; suggest mobility/light walk.
- If steps or workout goals already met: never push for more intensity; celebrate and suggest maintenance/recovery.
- No caffeine suggestions after the user's caffeine cutoff time.
- If medical_condition_flag = TRUE: do not suggest anything that conflicts with medical conditions.

CALENDAR CONSTRAINTS:
- If back_to_back >= 2: no new meetings or meeting suggestions in gaps; suggest break buffers only.
- If calendar_density_high: no new commitments; suggest reschedule or load reduction.
- If time_to_next_event <= 20: no deep work or workout suggestions; meeting prep + quick reset only.
- If continuous_events >= 120 OR breaks_today < 2: suggest break first before any productivity nudge.
- Do not suggest scheduling anything over meal windows (breakfast 07–10, lunch 12–14, dinner 18–20).

MOOD CONSTRAINTS:
- If mood in {terrible, sad}: block deep work, intense study, hard conversations, high-intensity workout.
- If mood = terrible: block social events, networking, group activities (unless user_opt_in = TRUE).
- If mood in {happy, amazing} AND (sleep_lastnight < sleep_goal OR stress_signal_high OR HR out of range):
  channel positivity into ONE meaningful task + ONE recovery action; no late high-intensity suggestions.
- NEVER use diagnostic or clinical language based on mood signals — non-clinical, supportive language only.
"""

# ---------------------------------------------------------------------------
# Block 4: CROSS-DOMAIN COHERENCE RULES
# ---------------------------------------------------------------------------

CROSS_DOMAIN_COHERENCE_RULES = """
**CROSS-DOMAIN COHERENCE — how this insight relates to the other two types:**

The three insight types together form a coherent picture of the same day, not three separate reports:
- **Overall** = whole-person synthesis (multi-domain story of the day)
- **Productivity** = the time/work/calendar angle of that story
- **Health** = the body/energy/recovery angle of that story

Rules:
1. **Same narrative frame, different domain angle.**
   If overall says "high-load day with good focus momentum", productivity addresses the work angle of that load;
   health addresses the body angle — neither starts an unrelated new topic.

2. **No direct contradiction.**
   If overall has already noted sleep debt as a risk, productivity must not ignore it.
   If health notes recovery need, productivity must not push harder output.

3. **No duplication.**
   Do not repeat the same observation word-for-word. Add your domain's perspective.
   If overall said "sleep quality was low", health should explain WHY it matters for the body today —
   not just repeat "sleep quality was low".

4. **Mood alignment.**
   All three types must respect the same mood state. A mood in {terrible, sad} constrains what all three
   types can suggest — not just the mood-based group.

5. **Genuine tension is allowed — don't hide it.**
   "Today was productive, but the body signals suggest the pace has a cost — worth watching tomorrow."
   This is better than pretending there is no tension between the domains.
"""

# ---------------------------------------------------------------------------
# Block 5: SUGGESTION DIVERSITY RULES
# ---------------------------------------------------------------------------

SUGGESTION_DIVERSITY_RULES = """
**SUGGESTION DIVERSITY RULES (applies to opportunity / point2 / any suggestion field):**
- Pick ONE specific suggestion; do not list multiple options ("try X or Y or Z").
- Do NOT default to reading, stretching, or meditation unless context strongly supports it.
- Tie the suggestion to today's actual pattern (meeting load, mood, sleep debt, free slot, time to next event).
- Vary activity type and phrasing across productivity / health / overall — they must NOT give the same suggestion.
- The suggestion must feel specific to THIS user's day, not a reusable script.
"""

# ---------------------------------------------------------------------------
# Suggestion palettes (daytime groups)
# ---------------------------------------------------------------------------

SUGGESTION_PALETTE_MORNING = """
**MORNING SUGGESTION PALETTE (pick ONE for point2 / opportunity):**
- set one clear priority for the day before opening anything else
- hydration: start with water before coffee/tea
- 5-min light movement to shift out of sleep mode
- quick prep for the first event (review agenda, gather materials)
- scan what didn't finish yesterday and decide: carry forward or drop
- 2-min breathwork to set a calm entry into the day
"""

SUGGESTION_PALETTE_ACTIVE = """
**ACTIVE WINDOW SUGGESTION PALETTE (pick ONE for point2 / opportunity):**
- start a focused work block on the highest-priority task
- quick meeting prep if next event is within 45 min
- energy snack or water refill to sustain focus
- micro-break between back-to-back blocks (2–5 min, away from screen)
- batch small admin tasks into one slot to protect focus time
- wrap up a loose thread before active hours end
"""

SUGGESTION_PALETTE_HEALTH_GOAL = """
**HEALTH GOAL SUGGESTION PALETTE (pick ONE for point2 / opportunity):**
- short outdoor walk (10–20 min) if steps are behind and time allows
- hydration — drink water now, not later
- brief movement snack (standing, walking to a room, light stretching)
- note a streak milestone to reinforce the momentum
- if evening and bedtime is near: sleep hygiene cue instead of activity push
"""

SUGGESTION_PALETTE_RECOVERY = """
**RECOVERY BREAK SUGGESTION PALETTE (pick ONE for point2 / opportunity):**
- 5-min walk away from the desk
- breathing reset (box breathing or 4-7-8, 2 min)
- water refill and look out a window
- step outside briefly for natural light
- brief mental unload: write down one thing that's still open so the brain can let it go
"""

SUGGESTION_PALETTE_CALENDAR = """
**CALENDAR PATTERN SUGGESTION PALETTE (pick ONE for point2 / opportunity):**
- block a focus slot tomorrow before meetings fill it
- add a 10-min buffer after the densest block of the day
- batch all admin tasks into one scheduled slot
- protect one meal window that is currently at risk
- reduce load: identify one meeting that could be async
"""

SUGGESTION_PALETTE_TASK_TYPE = """
**TASK TYPE SUGGESTION PALETTE (pick ONE for point2 / opportunity):**
- deep work block (only if slot >= 45 min AND outside 120 min of bedtime)
- light admin batch (reply, triage, file — no deep thinking required)
- communication sweep (batch messages/emails into one focused block)
- creative task (lower cognitive barrier, but still meaningful output)
- review or triage pass (process what's accumulated, clear the backlog lightly)
- wrap-up: close out today's open items before the active window ends
"""

SUGGESTION_PALETTE_WEEKEND = """
**WEEKEND LIFESTYLE SUGGESTION PALETTE (pick ONE for point2 / opportunity):**
- outdoor activity (walk, run, hike, cycling — something with natural light)
- social connection (call or meet someone, low-pressure)
- hobby or creative pursuit with no performance pressure
- cooking or meal prep as a mindful, screen-free activity
- physical movement that you wouldn't do on a workday
- quality downtime: no agenda, no output — just rest
"""

SUGGESTION_PALETTE_MOOD = """
**MOOD-BASED SUGGESTION PALETTE (pick ONE appropriate for the mood state):**
- terrible/sad → comfort-first: warm drink, favourite music, no-effort low-stimulation activity
- terrible/sad → reduce load: defer one non-urgent commitment, protect the rest of the day
- okay → gentle meaningful activity: something you want to do, not something you should do
- okay → light progress on one small thing that has been waiting
- happy/amazing → channel momentum into one meaningful task (not a new commitment)
- happy/amazing → share the energy: connect with someone briefly
"""

SUGGESTION_PALETTE_SAFETY = """
**SAFETY RISK SUGGESTION PALETTE (pick ONE, safety only):**
- rest immediately; stop all physical activity
- hydrate and sit/lie down
- avoid any intense activity until signals normalise
- if HR is dangerously out of range: rest and seek medical advice if unwell
- if sleep debt is high: do not schedule late intense workouts; light walk or rest only
"""

SUGGESTION_PALETTE_CONTEXTUAL = """
**CONTEXTUAL SUGGESTION PALETTE (pick ONE based on the strongest signal):**
- address the highest-impact cross-domain signal first (health, calendar, or mood)
- one small action that improves the most constrained area
- a micro-recovery if strain signals are present
- a forward-planning action if today's load is manageable
- acknowledgment of a positive pattern that deserves continuation
"""

_DAYTIME_PALETTES: Dict[str, str] = {
    "morning_window": SUGGESTION_PALETTE_MORNING,
    "active_window": SUGGESTION_PALETTE_ACTIVE,
    "health_goal": SUGGESTION_PALETTE_HEALTH_GOAL,
    "recovery_break": SUGGESTION_PALETTE_RECOVERY,
    "calendar_pattern": SUGGESTION_PALETTE_CALENDAR,
    "task_type": SUGGESTION_PALETTE_TASK_TYPE,
    "weekend_lifestyle": SUGGESTION_PALETTE_WEEKEND,
    "mood_based": SUGGESTION_PALETTE_MOOD,
    "safety_risk": SUGGESTION_PALETTE_SAFETY,
    "contextual_suggestion": SUGGESTION_PALETTE_CONTEXTUAL,
}

_EVENING_PALETTES: Dict[str, str] = {
    "evening_after_work": SUGGESTION_PALETTE_AFTER_WORK,
    "evening_flexible": SUGGESTION_PALETTE_EVENING,
    "wind_down_window": SUGGESTION_PALETTE_WIND_DOWN,
    "bedtime_window": SUGGESTION_PALETTE_BEDTIME,
}

ALL_PALETTES: Dict[str, str] = {**_DAYTIME_PALETTES, **_EVENING_PALETTES}


def _get_suggestion_palette(group: str) -> str:
    return ALL_PALETTES.get(group, "")


# ---------------------------------------------------------------------------
# Output guidance blocks
# ---------------------------------------------------------------------------

_POINT12_OUTPUT_GUIDANCE = """
**OUTPUT RULES (point1 / point2):**
- **point1**: ONE central pattern with causal reasoning (WHY, not just WHAT). Do not list metrics.
- **point2**: ONE contextual suggestion from the suggestion palette — do not repeat point1's observation.
- If no meaningful action applies for point2, keep it supportive and brief (never generic metric advice).
"""

_OVERALL_CATEGORY_OUTPUT_GUIDANCE = """
**OUTPUT RULES (great_job / need_attention / opportunity):**
- Identify ONE central pattern across domains (do not split into three unrelated topics).
- **need_attention**: the main causal insight (trade-off, imbalance, carry-over, or constraint).
- **great_job**: positive momentum supporting the same narrative, or `""` if not applicable.
- **opportunity**: ONE contextual suggestion from the suggestion palette, or `""` if not applicable.
- Do NOT force three unrelated domains across categories.
- Use `""` for categories that do not apply; do not invent filler.
"""

# ---------------------------------------------------------------------------
# Evening-group extras (appended on top of universal blocks for evening groups)
# ---------------------------------------------------------------------------

_EVENING_EXTRA_BY_GROUP: Dict[str, str] = {
    "evening_flexible": EVENING_FLEXIBLE_ACTION_GUARD
    + "\n"
    + SUGGESTION_CONTEXT_PICKER,
    "evening_after_work": SUGGESTION_CONTEXT_PICKER,
    "wind_down_window": SUGGESTION_CONTEXT_PICKER,
    "bedtime_window": "",
}

_EVENING_REASONING_BY_GROUP: Dict[str, str] = {
    "evening_flexible": EVENING_REASONING,
    "evening_after_work": AFTER_WORK_REASONING,
    "wind_down_window": WIND_DOWN_REASONING,
    "bedtime_window": BEDTIME_REASONING,
}


def _get_evening_extra(group: str) -> str:
    return _EVENING_EXTRA_BY_GROUP.get(group, "")


def _get_evening_reasoning(group: str) -> str:
    return _EVENING_REASONING_BY_GROUP.get(group, "")


# ---------------------------------------------------------------------------
# Public API — format_for_point12 and format_for_overall
# ---------------------------------------------------------------------------

EVENING_GROUPS = frozenset(
    {"evening_after_work", "evening_flexible", "wind_down_window", "bedtime_window"}
)


def format_for_point12(group: str) -> str:
    """Assemble the full reasoning block for point1/point2 output format (productivity & health)."""
    parts = [
        UNIVERSAL_REASONING_LENSES,
        _get_signal_weight(group),
        NOT_DO_CONSTRAINT_GUARDS,
        CROSS_DOMAIN_COHERENCE_RULES,
        _get_suggestion_palette(group),
        SUGGESTION_DIVERSITY_RULES,
        _POINT12_OUTPUT_GUIDANCE,
    ]
    # Append evening-specific reasoning and action guards for evening groups
    if group in EVENING_GROUPS:
        evening_reasoning = _get_evening_reasoning(group)
        if evening_reasoning:
            parts.append(evening_reasoning)
        evening_extra = _get_evening_extra(group)
        if evening_extra:
            parts.append(evening_extra)
    return "\n\n".join(p for p in parts if p)


def format_for_overall(group: str) -> str:
    """Assemble the full reasoning block for great_job/need_attention/opportunity format (overall)."""
    parts = [
        UNIVERSAL_REASONING_LENSES,
        _get_signal_weight(group),
        NOT_DO_CONSTRAINT_GUARDS,
        CROSS_DOMAIN_COHERENCE_RULES,
        _get_suggestion_palette(group),
        SUGGESTION_DIVERSITY_RULES,
        _OVERALL_CATEGORY_OUTPUT_GUIDANCE,
    ]
    # Append evening-specific reasoning and action guards for evening groups
    if group in EVENING_GROUPS:
        evening_reasoning = _get_evening_reasoning(group)
        if evening_reasoning:
            parts.append(evening_reasoning)
        evening_extra = _get_evening_extra(group)
        if evening_extra:
            parts.append(evening_extra)
        # Keep original evening category guidance from prompt_evening_reasoning for compatibility
        parts.append(_EVENING_OVERALL_CATEGORY_GUIDANCE)
    return "\n\n".join(p for p in parts if p)


def is_evening_insight_group(group: str) -> bool:
    """Return True if this group belongs to the evening window."""
    return group in EVENING_GROUPS
