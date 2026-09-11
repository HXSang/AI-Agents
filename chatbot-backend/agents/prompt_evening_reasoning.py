"""Shared evening reasoning frameworks for overall, productivity, and health insights."""

from typing import Dict

EVENING_INSIGHT_GROUPS = frozenset(
    {
        "evening_after_work",
        "evening_flexible",
        "wind_down_window",
        "bedtime_window",
    }
)

DEEP_INSIGHT_RULES = """
**DEEP INSIGHT RULES (ALWAYS APPLY):**
- Never simply restate data.
- Every insight must contain one of the following pattern types:
  1. **Trade-off** — something improved while another area received less attention
  2. **Reinforcing Pattern** — one positive behavior supported another
  3. **Hidden Imbalance** — progress exists but is not evenly distributed
  4. **Constraint** — calendar, energy, mood, or recovery limited available choices
  5. **Momentum** — multiple positive behaviors are reinforcing each other
  6. **Carry-over Effect** — today's behavior is likely to influence tomorrow
- Always explain WHY the pattern occurred.
- Avoid metric reporting; prefer causal reasoning over observation.
"""

SUGGESTION_DIVERSITY_RULES = """
**SUGGESTION DIVERSITY RULES (evening action fields — opportunity / point2):**
- Pick ONE specific suggestion; do not list multiple options ("try X or Y or Z").
- Do NOT default to reading, stretching, or meditation unless context strongly supports it.
- Tie the suggestion to today's pattern (meeting load, mood, sleep debt, free slot, time to bedtime).
- Vary activity type and phrasing; avoid generic wellness templates.
- The suggestion must feel specific to THIS user's day, not a reusable script.
"""

SUGGESTION_CONTEXT_PICKER = """
**SUGGESTION CONTEXT PICKER (choose ONE palette item that fits):**
- mood terrible/sad → comfort-first (music, warm drink, low effort); no productive nudges
- back_to_back_count or meeting_minutes high → mental unload (journaling, breathing, music); not new hobbies
- time_to_next_event <= 45 AND calendar has upcoming event → event_prep (logistics, agenda, arrive calm); NOT stretch/meditation/wind-down
- sleep_quality low OR sleep debt AND time_to_bedtime > 120 → energy/social/meal/hobby first; sleep hygiene only in point1, NOT default point2
- sleep_quality low AND time_to_bedtime <= 90 → may suggest light wind-down cues
- free_slot_length >= 30 (after work / evening only) → walk, meal, connect, hobby — not meditation by default
- time_to_bedtime <= 30 → micro-actions only (breathing, music, lights); no walks or meals
"""

EVENING_FLEXIBLE_ACTION_GUARD = """
**EVENING FLEXIBLE (19:00–wind-down) — NOT wind-down, NOT bedtime:**
- point2 / opportunity = still **living the evening**: meals, social, hobbies, event prep, light movement.
- FORBIDDEN as primary point2 unless time_to_bedtime <= 60: "wind-down", "before sleep", "transition to sleep", meditation, mindfulness, gentle stretch routine.
- If upcoming calendar events exist: prioritize **event_prep** or **enjoyable gap** between events — use event titles.
- point1 = synthesize the **shape of the day** (wins, meeting load, trade-offs); point2 = **one** concrete next action now.
- Pick ONE item from EVENING palette (cooking, podcast, social, hobby, meal prep) — vary across APIs (prod ≠ health ≠ overall).
"""

SUGGESTION_PALETTE_AFTER_WORK = """
**AFTER WORK SUGGESTION PALETTE (pick ONE for opportunity / point2):**
- light unhurried meal
- short walk to decompress
- call or message someone you care about
- warm shower to mark end of work
- 5-minute journal or brain dump
- chill playlist to shift out of work mode
- digital boundary (mute work notifications)
- quiet hobby with no performance pressure
"""

SUGGESTION_PALETTE_EVENING = """
**EVENING SUGGESTION PALETTE (pick ONE for opportunity / point2):**
- simple cooking or mindful eating
- podcast or music unwind
- light social connection
- gentle movement (not a workout)
- creative hobby
- light meal prep for tomorrow (only if early enough in evening)
- quality downtime without screens
"""

SUGGESTION_PALETTE_WIND_DOWN = """
**WIND-DOWN SUGGESTION PALETTE (pick ONE for opportunity / point2):**
- ambient or lo-fi music
- body scan or progressive muscle relaxation
- 4-7-8 or box breathing
- warm caffeine-free drink
- brief gratitude or release note
- screen-off ritual
- cool/dark room preparation
- quiet breathing without labeling it as "meditation"
"""

SUGGESTION_PALETTE_BEDTIME = """
**BEDTIME SUGGESTION PALETTE (pick ONE for opportunity / point2):**
- sleep hygiene (dark, cool, quiet room)
- consistency cue (same small ritual nightly)
- minimal tomorrow prep then stop
- avoid late scroll or stimulating content
- protect bedtime streak with one small step
- if wind-down is done → rest only, no new tasks
"""

SUGGESTION_PALETTE_BY_GROUP: Dict[str, str] = {
    "evening_after_work": SUGGESTION_PALETTE_AFTER_WORK,
    "evening_flexible": SUGGESTION_PALETTE_EVENING,
    "wind_down_window": SUGGESTION_PALETTE_WIND_DOWN,
    "bedtime_window": SUGGESTION_PALETTE_BEDTIME,
}

AFTER_WORK_REASONING = """
**AFTER WORK REASONING (Work ↔ Life):**
You are helping the user transition from work mode into personal time.
Do not summarize metrics.
Instead identify ONE meaningful pattern that explains how today's work, health, mood, and goals interacted.
Focus on: trade-offs, momentum, balance, constraints.
Questions to answer internally:
1. What helped the user make progress today?
2. What area received less attention because of another area?
3. Is the user ending work with balanced energy or accumulated strain?
Explain WHY the pattern happened, not just WHAT happened.
The goal is reflection, not evaluation.
"""

EVENING_REASONING = """
**EVENING REASONING (Whole Day Balance):**
Analyze the day as a whole — life-centric, not work-centric.
Do not focus on individual metrics.
Look for broader patterns across: health, productivity, mood, calendar, goals.
Identify one of: reinforcing pattern, hidden imbalance, sustainable habit, emerging trend.
Explain how multiple domains interacted.
The goal is understanding the shape of the day, not reporting statistics.
"""

WIND_DOWN_REASONING = """
**WIND-DOWN REASONING (Recovery Readiness):**
The user is no longer optimizing today. The focus shifts toward recovery.
Do not discuss: unfinished goals, missed steps, productivity improvements.
Instead identify: mental load, recovery readiness, accumulated fatigue, emotional carry-over.
Ask internally: "What is most likely to influence tomorrow if nothing changes?"
The goal is helping the user let go of the day.
"""

BEDTIME_REASONING = """
**BEDTIME REASONING (Tomorrow Readiness):**
The user can no longer meaningfully change today's outcomes.
Do not evaluate: missed goals, unfinished tasks, low step count.
Shift perspective toward tomorrow.
Focus on: recovery value, readiness for tomorrow, consistency of healthy habits, protecting momentum.
Ask internally: "What action right now would create the biggest positive effect on tomorrow?"
The goal is preparation, not reflection.
"""

_REASONING_BY_GROUP: Dict[str, str] = {
    "evening_after_work": AFTER_WORK_REASONING,
    "evening_flexible": EVENING_REASONING,
    "wind_down_window": WIND_DOWN_REASONING,
    "bedtime_window": BEDTIME_REASONING,
}

_OVERALL_CATEGORY_GUIDANCE = """
**EVENING PATTERN CATEGORY RULES:**
- Identify ONE central pattern across domains (do not split into unrelated domains).
- **need_attention**: main causal insight (trade-off, imbalance, carry-over, or constraint).
- **great_job**: positive momentum supporting the same narrative, or `""`.
- **opportunity**: ONE contextual suggestion from the evening suggestion palette, or `""`.
- Do NOT force three unrelated domains across categories.
- Use `""` for categories that do not apply; do not invent filler.
"""

_POINT12_GUIDANCE = """
**EVENING PATTERN OUTPUT RULES:**
- **point1**: ONE central pattern with causal reasoning (WHY, not just WHAT). Do not list metrics.
- **point2**: ONE contextual suggestion from the evening suggestion palette — do not repeat point1.
- If no meaningful action applies, keep point2 supportive and brief (never generic metric advice).
"""


def get_reasoning_block(group: str) -> str:
    """Return the reasoning framework text for an evening insight group."""
    return _REASONING_BY_GROUP.get(group, "")


def get_suggestion_palette(group: str) -> str:
    """Return the suggestion palette text for an evening insight group."""
    return SUGGESTION_PALETTE_BY_GROUP.get(group, "")


def _format_suggestion_blocks(group: str) -> str:
    palette = get_suggestion_palette(group)
    if not palette:
        return ""
    return f"{palette}\n{SUGGESTION_CONTEXT_PICKER}\n{SUGGESTION_DIVERSITY_RULES}"


def format_for_overall(group: str) -> str:
    """Reasoning + category guidance for great_job / need_attention / opportunity."""
    reasoning = get_reasoning_block(group)
    if not reasoning:
        return ""
    suggestion = _format_suggestion_blocks(group)
    extra = EVENING_FLEXIBLE_ACTION_GUARD if group == "evening_flexible" else ""
    return f"{reasoning}\n{_OVERALL_CATEGORY_GUIDANCE}\n{suggestion}\n{extra}\n{DEEP_INSIGHT_RULES}"


def format_for_point12(group: str) -> str:
    """Reasoning + point1/point2 guidance for productivity and health insights."""
    reasoning = get_reasoning_block(group)
    if not reasoning:
        return ""
    suggestion = _format_suggestion_blocks(group)
    extra = EVENING_FLEXIBLE_ACTION_GUARD if group == "evening_flexible" else ""
    return (
        f"{reasoning}\n{_POINT12_GUIDANCE}\n{suggestion}\n{extra}\n{DEEP_INSIGHT_RULES}"
    )


def is_evening_insight_group(group: str) -> bool:
    return group in EVENING_INSIGHT_GROUPS
