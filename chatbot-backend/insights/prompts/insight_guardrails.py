"""Per-domain insight GUARDRAILS (health / productivity / overall).

Used by daily Sylo Q&A prompts. Keep domain-scoped — do not dump
cross-domain NOT_DO lists into every insight.
"""

from __future__ import annotations

from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Shared (all domains)
# ─────────────────────────────────────────────────────────────────────────────

_SHARED_DATA_GUARD = """
**DATA HONESTY (ALL DOMAINS):**
- Never invent metrics, meetings, mood labels, or timestamps.
- Treat 0 / null / missing as "no data" — do not report as a real measurement.
- If data is stale (staleness_days >= 1), say the date / that it is not from today.
- Hedge when uncertain (may / seems / appears). No fake precision.
- Never use clinical / diagnostic language (disease names, medical prescriptions).
- TEMPORAL: last-night sleep (bedtime/duration/quality) is closed history — never blame it on today's calendar events.
  Today's events only affect tonight's upcoming wind-down / bedtime_goal.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

HEALTH_INSIGHT_GUARDRAIL = """
**HEALTH INSIGHT GUARDRAIL — check before writing:**

SCOPE:
- Body & recovery & mood only (sleep, steps/energy, HR/HRV, stress feel, emotions).
- Do NOT make productivity jargon the main frame (meetings, inbox, deep-work blocks).
- Calendar may inform timing realism for the next action — not the insight topic.

METRICS & MOOD:
- Use exact mood labels from data; if mood missing → say no mood logged (do not invent).
- Never escalate mood (okay ≠ sad; sad ≠ terrible).
- Prefer personal baseline / week / month over population norms.
- If goal already met (steps/sleep): celebrate / maintain — do not push intensity.

SAFETY:
- Extreme HR / clear risk signals → safety-first, calm language only; no intense exercise push.
- No medication, diagnosis, or clinical claims.
- No caffeine late / no intense workout jammed against bedtime when sleep debt is clear.

FORBIDDEN PHRASING:
- Clinical alarmism ("nervous system overloaded", disease framing).
- Invented workout protocols with hard minute counts as the insight itself.
- Blaming LAST NIGHT's bedtime/duration on TODAY's calendar events
  (e.g. "giờ ngủ trễ vì cf với Phương vừa xong" when 22:00 is last-night sleep).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Productivity
# ─────────────────────────────────────────────────────────────────────────────

PRODUCTIVITY_INSIGHT_GUARDRAIL = """
**PRODUCTIVITY INSIGHT GUARDRAIL — check before writing:**

SCOPE (PRIMARY LENS — CALENDAR FIRST):
- FIRST look at TODAY's calendar: events (title/time), denseness, buffers,
  free windows, next/upcoming events, work-hours fit vs active hours.
- HIGHEST PRIORITY: schedule irrationality vs bedtime goal
  (e.g. event "làm việc" at 22:00 while bedtime_goal is 21:00) — lead with this
  before free-window tips, steps, or focus scores.
- The main story should be about today's schedule when calendar data is present.
- Reminders / task completion / focus metrics are SECONDARY support only —
  do not lead with "focus level low" or task % if today's calendar has a clearer story.
- Call out unreasonable windows ONLY when calendar/work-hours data supports it.
- Do NOT frame as physiology (HR, sleep debt, HRV, step counts) as the main story.

CALENDAR / WORK:
- Never invent meetings, tasks, deadlines, or free windows.
- Name real event titles from data when relevant (keep original titles in quotes).
- If free slot is short or next event soon: no "deep work marathon" claims.
- If schedule is already light/OK: acknowledge — do not force a problem.
- Reschedule / protect-focus suggestions stay conceptual (no fake event titles).
- Reschedule / move / cluster advice applies ONLY to remaining / upcoming events —
  NEVER to past / already-ended events.

THRESHOLDS:
- Task completion ~70–80% is normal — do not call it "low".
- Focus moderate/high = OK; only "low" needs concern — and still not ahead of calendar.

FORBIDDEN:
- Turning productivity into a health coaching session.
- Hard Pomodoro / inbox-minute protocols as the insight.
- Suggesting work deep into wind-down without data that the user works then.
- Asserting how the user feels ("you are feeling…", "bạn đang cảm thấy…") from focus/load metrics.
- Leaking system labels ("hệ thống báo…", "mức tập trung thấp", "focus level low").
- Leading with abstract focus/task scores while ignoring today's calendar.
- Leading with free-window / steps tips while ignoring events that overlap bedtime.
- Calling a calendar EVENT a "call" / "cuộc gọi" / "meeting" unless title or event_type says so.
  Use "sự kiện" for events and "ghi nhớ" for reminders.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Overall
# ─────────────────────────────────────────────────────────────────────────────

OVERALL_INSIGHT_GUARDRAIL = """
**OVERALL INSIGHT GUARDRAIL — check before writing:**

SCOPE:
- Cross-domain synthesis ONLY when the connection is material (body ↔ work rhythm).
- Prefer one causal bridge — do not restate a pure health or pure productivity insight.
- If both domains look fine: acknowledge balance — do not invent conflict.

LANGUAGE:
- Plain everyday causal language ("tired makes focus harder").
- FORBIDDEN: cortisol, amygdala, nervous-system jargon, clinical diagnosis.
- Prefer common words over academic ones; anyone should understand on first read.

<<<<<<< HEAD
PRIORITY:
- If calendar events overlap bedtime / sleep goals → that cross-domain conflict is the story.
- Do not bury it under steps or generic free-window walking tips.

=======
>>>>>>> dev
ACTION BOUNDARY:
- The one next action must not simply copy a health-only tip or productivity-only tip.
- Keep it generalized and evidence-linked.
- Do not invent calendar or health numbers to force a bridge.
"""

_DOMAIN_GUARDRAILS = {
    "health": HEALTH_INSIGHT_GUARDRAIL,
    "productivity": PRODUCTIVITY_INSIGHT_GUARDRAIL,
    "overall": OVERALL_INSIGHT_GUARDRAIL,
}


def get_insight_guardrail(
    domain: str,
    *,
    active_hours_end_time: Optional[str] = None,
    include_shared: bool = True,
) -> str:
    """Return the GUARDRAIL block for a daily insight domain."""
    key = domain if domain in _DOMAIN_GUARDRAILS else "health"
    parts: list[str] = []
    if include_shared:
        parts.append(_SHARED_DATA_GUARD.strip())
    parts.append(_DOMAIN_GUARDRAILS[key].strip())
    _ = active_hours_end_time
    return "\n\n".join(parts)
