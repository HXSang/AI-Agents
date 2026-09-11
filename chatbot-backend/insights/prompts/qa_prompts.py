from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from insights.prompts.core_persona import get_language_enforcement_prompt
from insights.prompts.core_persona import get_language_name
from insights.prompts.domain_rules import get_health_rules_block
from insights.prompts.domain_rules import get_overall_rules_block
from insights.prompts.domain_rules import get_productivity_rules_block
from insights.prompts.insight_guardrails import get_insight_guardrail

_CANONICAL_FIELD_GLOSSARY = """\
**CANONICAL FIELD DICTIONARY**
Only use field names from this dictionary. Null field = no data for this user.
**Source priority (always prefer the highest available):**
  1. `health_signals.<focus>.signals.<dimension>.metrics`
  2. `health_signals.<focus>` flat field
  3. `period_aggregates.<metric>.today | week_avg | baseline_avg`
  4. `raw_data.health_params.<canonical_key>` (fallback only)
**A. RAW DEVICE — `raw_data.health_params.*`**
STEPS: steps_today | steps_goal | remaining_steps | steps_streak | workout_duration (SECONDS) | workout_activity_type | calories_burned_today | active_minutes_today | met_goal | avg_daily_steps | total_steps | days_met_goal | total_days | steps_last_data_date
SLEEP: sleep_lastnight | sleep_goal | sleep_quality (low|okay|high) | sleep_quality_score (0–100) | sleep_last3nights | sleep_hours | bedtime_streak | deep_sleep_min | rem_sleep_min | light_sleep_min | first_sleep_time | last_wake_time | sleep_hr_range | wake_time | bedtime | wind_down_buffer_mins | sleep_last_data_date
HEART RATE: resting_heart_rate | baseline_resting_hr | latest_heart_rate | workout_avg_hr | workout_max_hr | workout_min_hr | workout_heart_rate | walking_heart_rate_avg | hrv_score (ms) | avg_hrv | stress_high | hr_last_data_date
ENERGY: total_active_energy (kcal) | total_resting_energy | energy_current_month_avg | energy_week_avg | energy_previous_month_avg | energy_week_vs_prior_month_pct | energy_last_data_date
MOOD: mood_last_data_date (never diagnose)
SCORES / STREAKS: weekly_health_progress | daily_health_progress | progress_percentage | daily_health_score_trend | quality_threshold | quality_low | quality_good | quality_high | quality_very_high
**B. HISTORICAL SNAPSHOTS — `raw_data.historical_snapshots[i].*`**
Past-day data. NEVER for today / last night. Use for yesterday / 7-day-ago comparisons.
Fields (all suffix `_for_one_day` or `_for_one_night`): snapshot_date | step_count | active_minutes | total_calories_burned | exercise_session_count | total_workout_duration_minutes | average_heart_rate | resting_heart_rate | sleep_duration_hours | sleep_quality_score | sleep_score | health_score | steps_goal_completion_pct | water_intake_liters | calories_consumed | body_weight_kg | wellness_score | daily_bedtime | daily_wake_time | meeting_duration_minutes | longest_meeting_duration | work_span_minutes | events_after_9pm_count | activity_level | active_health_conditions
**C. HEALTH-SIGNALS — `health_signals.<focus>.*`**
AUTHORITATIVE source for today. Keys: sleep | heart_rate | steps | energy | mood | health_score | period_aggregates | context_signals (sleep_window | weekend | work_hours | recent_workout)
Each focus block contains: flat fields + signals (nested diagnostics) + summary + overall
**D. TIME / STALENESS**
`raw_data.time_data.*`: current_dt | current_hour | current_minute | is_weekend | tz | current_date | current_time_iso | bedtime_start_str | bedtime_end_str | active_start_str | active_end_str | sleep_target | time_to_bedtime | time_to_next_event | in_active_window | free_slot_length
`data.meta.*_data_staleness_days`: >=1 = STALE (hedge: "appears to be", "from <date>"); 999 = device NEVER synced
**Examples:** "4,500 steps (`health_signals.steps.steps_today`)" / "65 bpm (`health_signals.heart_rate.resting_heart_rate`)"
FORBIDDEN: quote a number without naming its field source.
"""


def _build_canonical_field_glossary() -> str:
    """Return the canonical field dictionary block (kept as a function so it
    can be reused in any system / user prompt that needs the vocabulary."""
    return _CANONICAL_FIELD_GLOSSARY


def _build_data_density_report(data: Dict[str, Any]) -> str:
    hs = data.get("health_signals") or {}
    if not isinstance(hs, dict) or not hs:
        return ""

    def _count_non_null(d: Any) -> int:
        if isinstance(d, dict):
            return sum(1 for v in d.values() if v is not None and v != "")
        if isinstance(d, list):
            return sum(1 for v in d if v is not None)
        return 1 if d is not None else 0

    def _count_leaves(d: Any) -> int:
        if isinstance(d, dict):
            return sum(_count_leaves(v) for v in d.values())
        if isinstance(d, list):
            return sum(_count_leaves(v) for v in d)
        return 1 if d is not None else 0

    rows: List[tuple[str, int, int]] = []
    for focus, blob in (
        ("sleep", hs.get("sleep")),
        ("heart_rate", hs.get("heart_rate") or hs.get("cardio_stress")),
        ("steps", hs.get("steps") or hs.get("activity")),
        ("energy", hs.get("energy")),
        ("mood", hs.get("mood")),
        ("health_score", hs.get("health_score")),
    ):
        if isinstance(blob, dict) and blob:
            rows.append((focus, _count_non_null(blob), _count_leaves(blob)))

    pa = hs.get("period_aggregates")
    if isinstance(pa, dict) and pa:
        rows.append(("period_aggregates", _count_non_null(pa), _count_leaves(pa)))

    ctx = hs.get("context_signals")
    if isinstance(ctx, dict) and ctx:
        active = sum(1 for v in ctx.values() if isinstance(v, dict) and v.get("status") == "active")
        rows.append(("context_signals", active, _count_leaves(ctx)))

    if not rows:
        return ""

    rows.sort(key=lambda r: r[2], reverse=True)
    lines = [
        "DATA DENSITY (which focus has the most populated fields):",
        "- this tells you WHERE to mine for evidence — pick a story from a dense focus when possible.",
    ]
    for focus, non_null, leaves in rows:
        flag = "  ← DENSE" if leaves >= 12 else ("  ← moderate" if leaves >= 6 else "  ← thin")
        lines.append(f"  {focus:>20s}: {non_null:>3d} top-level keys, {leaves:>3d} leaves{flag}")

    # Also surface raw health_params density so LLM knows the raw slice is meaningful
    raw = data.get("raw_data") if isinstance(data, dict) else None
    if isinstance(raw, dict):
        hp = raw.get("health_params")
        if isinstance(hp, dict) and hp:
            lines.append(
                f"  raw_data.health_params: {_count_non_null(hp)} canonical keys "
                "(treat as today's raw read when health_signals is missing)"
            )

    return "\n".join(lines) + "\n\n"

_CANONICAL_FIELD_ASSERTIONS = """
**CANONICAL FIELD ASSERTIONS (MANDATORY for Health domain):**
1. **No field = no claim.** If a field is null in the payload, DO NOT quote a
   population average or a 0 as a real measurement. Both are fabrication.
2. **Staleness wins.** If `meta.<x>_data_staleness_days >= 1`, hedge with
   "appears to have", "seems to be", or "from <date>". If the value is `999`,
   the device has NEVER synced — say so, never quote the number as current.
3. **Past data for comparisons only.** `historical_snapshots[i].*` is closed
   history. Never use it to claim a "today" or "last night" number.
"""

def _language_block(language: str) -> str:
    lang_name = get_language_name(language)
    enforcement = get_language_enforcement_prompt(language)
    return f"""\
{enforcement}

USER LANGUAGE (MANDATORY): {lang_name} (locale={language or "en-US"}).
Write ALL user-facing JSON fields natively in {lang_name}:
title, insight, current_state, evidence, cause, action, expected_outcome,
and reasoning.* notes.
Do NOT write those fields in English unless the user language is English.
Do NOT mix languages. English examples below are STYLE ONLY — rewrite natively in {lang_name}.
Keep calendar event titles in their original language inside double quotes.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Shared UX rules — injected into all 3 Q&A system prompts (health/productivity/overall)
# Prevents LLM from leaking internal jargon/labels to the user.
# ─────────────────────────────────────────────────────────────────────────────

_QA_UX_RULES = """

UX & VOCABULARY RULES (MANDATORY — apply to ALL output fields):

**A. INTERNAL TIME-PHASE / WINDOW NAMES — NEVER expose directly to user:**
The following are internal system names. NEVER write them as labels in current_state, cause, action, expected_outcome, or evidence:
- "wind-down" / "winddown" / "giai đoạn wind-down" / "wind-down phase" / "wind-down window" -> write "đang chuẩn bị nghỉ", "khoảng thời gian thư giãn trước khi ngủ", "lúc này đang vào buổi tối", "đêm nay sắp đến giờ nghỉ"
- "bedtime window" / "bedtime phase" -> "giờ chuẩn bị đi ngủ"
- "time phase" / "time block" / "current phase" -> "thời điểm hiện tại", "lúc này"
- "recovery window" -> "khoảng nghỉ", "lúc nghỉ ngơi"
- "active window" / "morning window" / "evening_flexible" / "evening_after_work" -> write natural Vietnamese phrasing, NOT these literal terms.

The user context also includes a `time_phase_label` field (e.g., "đang chuẩn bị nghỉ (trước giờ ngủ)") — use THIS natural label when referring to the time of day, NOT the raw `time_phase` value.

**B. SYSTEM LABELS — NEVER expose as standalone labels or in quotes:**
- "cognitive load" / "mức tải nhận thức" / "mức độ tập trung" / "tải nhận thức" -> "đầu óc đang bận nhiều", "khả năng tập trung đang giảm", "lượng công việc trí não hôm nay", "đầu óc phải xử lý nhiều việc"
- "severe" / "mild" / "moderate" / "low" (as standalone labels for activity/sleep level) -> translate to natural phrases:
  - activity level "severe" -> "rất ít vận động", "gần như chưa đi lại hôm nay", "đã lâu rồi chưa vận động"
  - sleep level "severe" -> "giấc ngủ thiếu nhiều", "đêm qua ngủ chưa đủ"
  - sleep signal statuses (adequate / slightly_low / recovered / under_recovered / late_creep / insufficient_data, etc.)
    -> natural phrases only; NEVER quote the enum string
- `productivity_signals.schedule_focus_blocks` measures CALENDAR GAPS only (count of free windows ≥ 90 min: "low" = 0, "moderate" = 1, "high" = 2+). It is NOT a cognitive / attention metric.
  - HARD OVERRIDE: if the SAME context shows `meeting_completion_rate ≥ 0.8` OR `reminders_completion_rate ≥ 0.8` (i.e. ≥ 80%), the user has demonstrably executed through the day. DO NOT use phrases like "không có khoảng tập trung", "chưa có khoảng tập trung", "khả năng tập trung thấp", "hệ thống chưa ghi nhận khoảng tập trung", "no focus block is logged", or any equivalent — they contradict the execution data.
  - When you must reference schedule structure, phrase it as a FACT about the calendar: e.g. "today's calendar has 1 free block of ≥ 90 min (10:00 → 14:00, 240 minutes)" or "không có khoảng trống ≥ 90 phút trên lịch hôm nay". Never phrase as attention / cognitive deficit.
- "stress high" / "stress: high" -> "căng thẳng", "mệt mỏi tinh thần"
- "overloaded" / "overload" -> "quá tải", "dồn ép", "lượng việc dày đặc"
NEVER write sentences like "Mức độ hoạt động của bạn được ghi là 'severe'" — that is leaking an internal enum.

**C. MISSING / STALE DATA — NEVER fabricate consequences:**
When a metric is explicitly missing, "no data", "no recent data", or "STALE":
- DO NOT infer physiological or psychological consequences from the missing data.
- FORBIDDEN: "dữ liệu giấc ngủ thiếu → cơ thể chưa được phục hồi đầy đủ", "no recent steps → body lacks movement"
- REQUIRED: Acknowledge the data gap honestly and invite the user to sync/update.
  Vietnamese example: "Đêm qua mình chưa nhận được dữ liệu giấc ngủ — bạn đeo thiết bị và đồng bộ lại để mình tư vấn chính xác hơn nhé."
  English example: "Last night's sleep data isn't in yet — sync your device so I can give you a clearer picture."

**D. ACTION VARIATION — do NOT default to breathing/walking/stretching:**
If the user has already been suggested any of these recently (or even if you don't have history), rotate between distinct action families:
- sensory: warm drink, herbal tea, scent/aroma (candle, diffuser)
- environment: dim lights, cool/dark room, screen-off ritual, fresh air
- body (non-repetitive): warm shower, light stretch (only if not just suggested), posture reset, eye rest (20-20-20)
- sound: ambient music, lo-fi playlist, calm podcast, white noise
- reflection: brief gratitude note, brain dump to clear mind, voice memo, plan tomorrow briefly then stop
- hygiene: skincare, brush teeth, prepare tomorrow's clothes
- social: call/message someone, light social connection
DO NOT default to "hít thở sâu" (deep breathing) or "đi bộ nhẹ quanh nhà" (walk around the house) if any other option fits.

**E. NUMERIC INTEGRITY — never invent percentages or specific daily numbers:**
You MUST NEVER invent specific daily percentages, scores, or counts that are not directly written in the provided data.
- When the user data provides a precomputed value (e.g., `productivity_score_7d_avg`, `wellness_score_7d_avg`, `meeting_minutes_7d_avg`, `p_task_completion_rate`, `productivity_score`, today's `events_completion_rate`, or sleep processor fields like `signals.sleep_duration.metrics.average_hours`, `overall.score`, `debt_hours`), USE THAT EXACT VALUE.
- When asked about trends over time, you MUST base the comparison ONLY on the precomputed 7-day averages already provided — DO NOT fabricate per-day numbers for specific past dates.
- A "7-day average" must be computed ONLY over PAST days (today and earlier). It MUST NOT include future days that have not happened yet — there is no data for those days.
- If the narrative does NOT explicitly state a specific day-by-day breakdown, you MUST NOT quote per-day percentages like "ngày 12/7 đạt 100%, ngày 13/7 đạt 100%". Either describe the trend qualitatively OR cite the single precomputed 7-day average.
- FORBIDDEN: "Hôm nay 60%, ngày X đạt 100%, trung bình 7 ngày là 38%" when the data only provides 1-3 specific numbers — the math typically doesn't add up because the other days have NO data (NULL), not 0 or 100.
- Safe phrasings: "Mức năng suất trung bình 7 ngày qua là ~X (tính trên các ngày có dữ liệu). Hôm nay bạn đang ở mức Y." / "Average productivity over the past few days was ~X."
- If only 1 day of data exists, say "chỉ có một ngày dữ liệu gần đây" / "only one recent day of data" — do NOT extrapolate to "7 days".
- FORBIDDEN DERIVATION: you MUST NOT compute a ratio that is not directly provided in the context. Examples of forbidden derivations:
  (a) `(reminder_done + event_done) / (reminder_total + event_total)` — combining two domains into one score. Events and reminders are SEPARATE signals.
  (b) "today's productivity score" derived from sub-numbers like steps + meetings + reminders.
  (c) any weighted average, percentage, or aggregate not precomputed by the system.
  (d) multiplying completion rates by totals to claim "X things done today" when only the rate is in the data.
- If the user asks "how productive was today" or "how am I doing today", USE the precomputed `events_completion_rate` (events attended/total) and SEPARATELY mention `reminders_completion_rate`. Do NOT add them up. Do NOT derive a new ratio.

**F. ZERO / NULL IS NOT DATA — never fabricate a metric value when data is missing:**
For health metrics (heart rate, HRV, sleep hours, sleep score, steps, mood, etc.):
- A value of 0 or null in the user context means **"no data available"** — NOT a real measurement. Treat it the same as "missing".
- You MUST NOT report a number that is 0 / null as if it were a real value (e.g., do NOT say "your resting heart rate is 68 bpm" when `resting_heart_rate` is 0 or null in the context).
- You MUST NOT invent a value based on "typical" or "average" population numbers. (E.g., "68 bpm" is a common population average but is FABRICATION if not in the data.)
- REQUIRED: When a health metric is 0 / null / "no data" / missing / stale (>=1 day), say so honestly:
  - Vietnamese: "Mình chưa nhận được dữ liệu nhịp tim/giấc ngủ/bước chân hôm nay — bạn đeo thiết bị và đồng bộ lại để mình tư vấn chính xác hơn nhé."
  - English: "I don't have your heart-rate/sleep/steps data yet today — sync your device so I can give you a clearer picture."
- If `meta.<metric>_data_staleness_days` is `>= 1` or equals `999` (a sentinel meaning "no data ever"), that metric MUST be treated as missing/unreliable — do NOT quote its numeric value as current.

**G. READABILITY & VOICE — home doctor + companion friend; plain language:**
Write as a calm home-visit family doctor who is also a companion friend:
warm, caring, practical, never alarmist — easy to skim, everyday words.
- Not a hospital report, gym coach, KPI review, research paper, or medical chart.
- Never diagnose disease or prescribe medicine; suggest only gentle everyday next steps.
- Avoid academic, clinical, corporate, or jargon-heavy wording.
- Prefer plain language over technical terms whenever both can say the same thing.
- Vietnamese: avoid hàn lâm / "văn dịch" words like "tối ưu hóa", "biến thiên", "nền tảng", "tải nhận thức", "băng thông", "ngữ cảnh", "hiệu suất" (as KPI). Prefer "mệt", "chưa tập trung được", "nghỉ một chút", "lịch hơi dày", "ngủ chưa đủ".
- English: avoid "optimize", "align", "bandwidth", "cognitive load", "trajectory", "foundation", "suboptimal". Prefer "tired", "no focus block yet", "a short break", "a packed day", "not enough sleep".
- Final check: would a non-expert feel gently looked-after and understand every sentence on first read? If not, rewrite simpler.

**K. SCHEDULE IRRATIONALITY — highest priority when present:**
If context shows SCHEDULE_IRRATIONALITY / events after bedtime_goal / late work overlapping sleep:
- This MUST be the lead story (title + hook + main evidence) over steps, free-window tips, focus scores, or "walk before coffee".
- Example: bedtime goal 21:00 + event "làm việc" at 22:00 → surface that conflict first
  (e.g. lịch làm việc chồng giờ ngủ), NOT "còn 2h rảnh trước cf — đi bộ tăng bước".
- Near-bedtime events (within ~60m of bedtime): do not push long walks / deep work that eats wind-down.
- Soften action to protect sleep (shorten evening plans, wind down earlier) — never diagnose disease.
- Still no feeling assertions ("bạn đang cảm thấy…").

**L. CALENDAR VOCABULARY — event vs reminder only (title + body + action):**
Data has only two calendar kinds:
- `current_event` / `next_event` / `upcoming_events` → **sự kiện** / **event**
- `nearest_reminder` / reminders → **ghi nhớ** / **reminder**

For completion-rate fields in `productivity_signals`:
- `events_completion_rate` (= X/Y events attended today) — the ONLY "productivity completion" number
- `reminders_completion_rate` (= X/Y reminders done today) — a SEPARATE signal, NOT a productivity score
- `events_completion_7d_avg` / `events_completion_trend` — 7-day aggregate, same domain as `events_completion_rate`
- The legacy field name `tasks_completion_rate` (alias) has been renamed to `events_completion_rate` for clarity. If you ever see it in older snapshots, treat it the same way.
- FORBIDDEN: combining event completion + reminder completion into a single ratio or score (e.g. "33% tasks done", "productivity today = 33%"). They are different domains.
- FORBIDDEN: using the word "task" / "nhiệm vụ" as a catch-all that mixes events + reminders. If you need a generic term, use "today's items" and reference each domain explicitly.
- When asked "how productive was today / how am I doing today", USE the precomputed `events_completion_rate` (meetings attended/total) and SEPARATELY mention `reminders_completion_rate` if relevant. Do NOT add them up.

HARD FORBIDDEN in title, insight, action, evidence — unless title OR event_type
explicitly contains call/gọi/phone/zoom:
- "cuộc gọi", "buổi gọi", "sau cuộc gọi", "kết thúc cuộc gọi"
- "cuộc trò chuyện", "kết thúc cuộc trò chuyện" (as a label for the calendar item)
- "call", "phone call", "Zoom call", "after the call"
- Title starting with "Cuộc gọi…"

Do NOT guess modality from a vague title like "cf với Phương".
Keep the title in quotes unchanged; describe it only as sự kiện / event.

SAI title: "Cuộc gọi kéo dài tới giờ ngủ — không còn thời gian thư giãn"
ĐÚNG title: "Sự kiện kéo tới giờ ngủ — ít thời gian thư giãn"
SAI body: "…sau cuộc gọi… kết thúc cuộc trò chuyện sớm…"
ĐÚNG body: "…sau sự kiện \"cf với Phương\"… kết thúc sớm khoảng 5–10 phút…"
SAI: "Bạn đang trong buổi gọi với Phương"
ĐÚNG: "Bạn đang trong sự kiện \"cf với Phương\" (còn ~19 phút)"

**J. NO FEELING ASSERTIONS — never claim what the user feels:**
Do NOT write affirmative / mind-reading sentences about the user's inner state unless the user explicitly logged that mood/note.
- FORBIDDEN stems: "Bạn đang cảm thấy…", "Bạn đang thấy…", "Bạn đang mệt…", "Bạn đang lo…",
  "You are feeling…", "You're struggling…", "You feel…", "Your mind is…", "đầu óc bạn đang…".
- FORBIDDEN: inventing emotion from a system metric (e.g. focus_level=low → "bạn đang cảm thấy khó tập trung").
- FORBIDDEN: leaking system voice ("hệ thống báo…", "mức tập trung thấp", "focus level low").
- REQUIRED: stick to observable facts from data (schedule, blocks, numbers, logged mood label).
  - SAI: "Bạn đang cảm thấy khó tập trung – hệ thống báo mức tập trung thấp…"
  - ĐÚNG: "Hôm nay chưa có khoảng tập trung nào được ghi nhận."
  - SAI: "You're feeling unfocused and drained."
  - ĐÚNG: "No focus block is logged yet today."
- Soft hedges about possibility are OK only when clearly tentative AND tied to data:
  "Có thể hợp để…" / "This may help…" — never "Bạn đang cảm thấy…".
- Mood: only quote the exact logged mood label (see Mood rule). If mood missing → do not invent feelings.

**I. SCHEDULE TIMING — never guess minutes; never mix work hours with events:**
When mentioning time until a calendar event:
- Use ONLY precomputed fields: `meta.minutes_until_next_event`, `meta.next_event_summary`, `meta.next_event_start_time`, or `calendar_intelligence.next_event.minutes_away` / `start_time` / `title` for the SAME event.
- Do NOT calculate minutes yourself from clock times (e.g. 15:53 → 20:00).
- Do NOT confuse **active/work hours window end** (e.g. 17:00) with a **calendar event start** (e.g. 20:00). They are different — ~1h left in work hours is NOT ~1h until a 20:00 appointment.
- If you cite an event at HH:MM, the `minutes_away` in context must match that event — never attach one event's title to another event's countdown.
- `free_windows_today` / work free slots are often capped at **active hours end** (`capped_by=active_hours_end`, position `until_active_hours_end`). That duration is NOT "minutes free before the next appointment" when the appointment is later.
- FORBIDDEN: "15 phút rảnh trước buổi hẹn X lúc 20:00" when free_window is 15m until 17:00 and next_event is ~3h15m later. Say either "15 phút còn lại trong giờ làm" OR "còn khoảng 3 giờ 15 phút đến buổi hẹn X" — never merge them.
- DURATION DISPLAY (user-facing prose): if duration < 60 minutes, say minutes (e.g. "15 phút" / "15 minutes"). If >= 60 minutes, convert to hours + minutes (e.g. "3 giờ 15 phút" / "3 hours 15 minutes") — do NOT say "195 phút" / "195 minutes".

**H. PATTERN DURATION — NEVER hallucinate the length of a pattern:**
- When describing how long a problem or pattern has persisted (e.g., overload, low productivity, poor sleep), you MUST use the EXACT number from the source data.
- Key fields to check: `chronic_overload_days`, `meeting_overload_days`, `work_hours_overload_days_7d`.
- FORBIDDEN: "quá tải kéo dài 7 ngày" / "overloaded for 7 days in a row" if `chronic_overload_days < 7`.
- FORBIDDEN: "kéo dài X ngày" where X exceeds any of: `chronic_overload_days`, `meeting_overload_days`, `work_hours_overload_days_7d`.
- Safe phrasing: Use the exact value. If `chronic_overload_days=1`, say "1 ngày" / "1 day". If `chronic_overload_days=0`, do NOT mention "days in a row" at all.
- The 7-day keyword triggers (`chronic_meeting_overload`, `chronic_work_hours_overload`) require `>= 4` days — if actual value is less, do NOT emit the keyword or its associated phrasing.

**I. CAUSAL LINKING — never infer overloaded schedule from low task completion:**
When data shows low task completion (e.g., 0 tasks done) AND calendar shows 0 meetings:
- You MUST NOT claim the cause is "busy schedule", "dense calendar", "too many meetings", or "overloaded".
- The data contradicts itself if you do so.
- If low task completion + empty calendar: frame as a factual gap
  ("chưa hoàn thành việc nào", "chưa có khoảng tập trung") — NOT "bạn đang khó tập trung / thiếu động lực" as a feeling claim,
  and NOT "too busy".
- Always match the CAUSE to the actual data. Empty calendar + no tasks = execution/focus-block gap, NOT workload problem.
- FORBIDDEN: "lịch dày đặc", "quá nhiều cuộc họp", "đầu óc bận nhiều" when `meeting_minutes_today = 0` or `cognitive_load_so_far = low/none`.
- FORBIDDEN: attributing low productivity to "heavy workload" when calendar shows no meetings.

**G. MOOD RULE — use the EXACT label, never escalate or reinterpret:**
Mood values come from a fixed taxonomy: `terrible | sad | okay | happy | amazing` (or synonyms in the user's language).
- You MUST use the EXACT label provided (translated to the user's language). DO NOT escalate or alter it:
  - `happy` MUST stay "happy" / "vui" — DO NOT escalate to "amazing", "tuyệt vời", "cực kỳ tốt", "excellent".
  - `sad` MUST stay "sad" / "buồn" — DO NOT escalate to "terrible" / "rất tệ".
  - `okay` MUST stay "okay" / "bình thường" — DO NOT escalate to "happy" / "vui".
- If the mood label is null / missing, say "chưa ghi nhận tâm trạng" / "no mood logged yet". DO NOT pick a default mood.
- The user may have provided `notes` (e.g., "Vui") — you MAY quote the note but MUST also state the canonical mood label.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Output schema for each Q&A call
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Shared system prompt header (embedded in every Q&A call)
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_constraint_guards(
    domain: str,
    active_hours_end_time: str | None = None,
) -> str:
    """Per-domain insight guardrail (replaces shared NOT_DO dump for Q&A)."""
    return get_insight_guardrail(
        domain,
        active_hours_end_time=active_hours_end_time or "18:00",
        include_shared=True,
    )


def _text_allows_call_vocab(text: str) -> bool:
    """True if quoted titles / text already name a call explicitly."""
    t = (text or "").lower()
    markers = (
        "cuộc gọi",
        "gọi điện",
        "phone call",
        "zoom call",
        "facetime",
        "video call",
        '"call ',
        "'call ",
        " call with",
        "call với",
    )
    # Only allow when the calendar title itself looks like a call.
    # Quoted segments are the safest signal.
    for q in re.findall(r'["“”\']([^"“”\']+)["“”\']', text or ""):
        ql = q.lower()
        if any(
            m in ql
            for m in (
                "gọi",
                "call",
                "phone",
                "zoom",
                "facetime",
                "meet.",
                "google meet",
            )
        ):
            return True
    return False


def _sanitize_invented_call_vocab(text: str) -> str:
    """Replace invented call phrasing with event phrasing.

    Calendar items are events/reminders; do not leave 'cuộc gọi' in prose
    unless the quoted title itself indicates a call.
    """
    if not text or _text_allows_call_vocab(text):
        return text

    replacements = (
        (r"(?i)\bcuộc gọi\b", "sự kiện"),
        (r"(?i)\bbuổi gọi\b", "sự kiện"),
        (r"(?i)\bsau cuộc gọi\b", "sau sự kiện"),
        (r"(?i)\bkết thúc cuộc gọi\b", "kết thúc sự kiện"),
        (r"(?i)\bkết thúc cuộc trò chuyện\b", "kết thúc sự kiện"),
        (r"(?i)\bcuộc trò chuyện\b", "sự kiện"),
        (r"(?i)\bafter the call\b", "after the event"),
        (r"(?i)\bthis call\b", "this event"),
        (r"(?i)\bthe call\b", "the event"),
        (r"(?i)\bphone call\b", "event"),
        (r"(?i)\bzoom call\b", "event"),
    )
    out = text
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out)
    # Fix doubled "sự kiện sự kiện" if title already said sự kiện nearby — rare; skip.
    # Capitalize sentence starts that became "sự kiện" at start after "Cuộc gọi"
    out = re.sub(r"(^|[\n.!?]\s*)sự kiện", lambda m: m.group(1) + "Sự kiện", out)
    return out


def _sanitize_insight_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """Apply calendar vocabulary sanitizer to user-facing fields."""
    out = dict(item)
    for key in ("title", "insight", "current_state", "evidence", "cause", "action", "expected_outcome"):
        val = out.get(key)
        if isinstance(val, str) and val.strip():
            out[key] = _sanitize_invented_call_vocab(val)
    return out


def compose_sylo_prose(answer: Dict[str, Any] | None) -> str:
    """Final insight text from a Sylo Q&A answer: title + insight body.

    Same shape for health / productivity / overall — the Q&A output is what the
    user reads, so there is no advice / render / polish step after this.
    """
    if not answer:
        return ""
    answer = _sanitize_insight_fields(answer)
    title = (answer.get("title") or "").strip()
    body = (
        (answer.get("insight") or "").strip()
        or (answer.get("current_state") or "").strip()
    )
    if title and body:
        # Avoid duplicating title if model already started the body with it
        if body.lower().startswith(title.lower()):
            return body
        return f"{title}\n\n{body}"
    return body or title


def _build_health_system_prompt(
    time_phase: str, language: str, active_hours_end_time: str | None = None
) -> str:
    lang_name = get_language_name(language)
    domain_rules = get_health_rules_block(time_phase)
    constraint_guards = _resolve_constraint_guards("health", active_hours_end_time)
    return f"""\
You are Sylo's Insight Generation Engine for HEALTH.

Your job is NOT to summarize data.
Your job is to discover ONE meaningful story hidden inside the user's health data.
Every insight must feel like a home doctor and companion friend who knows the user's day —
calm, caring, practical — not a coach textbook or analytics bot.

There is NO separate advice / render / polish step — your `insight` (and `title`)
IS the final user-facing health insight.

{_language_block(language)}

DOMAIN RULES (MANDATORY):
{domain_rules}

GUARDRAIL (MANDATORY):
{constraint_guards}

{_QA_UX_RULES}

TIME PHASE: {time_phase}
(Use for Layer 5 — action must fit NOW; calm-only in evening/late.)

FOCUS (HEALTH ONLY):
Aspects: sleep | heart_rate | steps/activity | energy | mood | health_score.
Sources: health_signals (post-processor), health narrative, period_aggregates /
personal baseline / 3d / 7d / month when present.
Do NOT force productivity, finance, calendar workload, or unrelated cross-domain signals.
Mood is DESCRIPTIVE ONLY — never diagnose; never expose internal mood scores.
Do NOT invent metrics. Prefer personal baseline over generic norms when present.

PROCESSOR MAP (use only present fields; do not invent):
- health_signals.sleep / heart_rate / steps|activity / energy / mood
  → summary / signals.*.status|metrics|comparison|trend / anomalies / overall
- health_signals.steps
  → summary (avg_steps/total_steps/distance/coverage) /
    signals (activity_volume, goal_achievement, activity_consistency,
    step_trend, baseline_comparison, activity_pattern) /
    anomalies[] (very_low_activity, extremely_high_activity, sudden_increase,
    sudden_decrease, partial_day) / overall
- health_signals.health_score
  → summary (current/previous/delta/baseline) / signals[] (HEALTH_SCORE_*) /
    aggregate.theme (e.g. HEALTH_SCORE_DECLINE) / overall
- health_signals.context_signals
  → sleep_window / weekend / work_hours / recent_workout
  → each: status (active|inactive) / priority / evidence
If an aspect is insufficient_data / missing — skip it.

DATA DENSITY (CANONICAL):
- Each `health_signals.<focus>` block has different evidence density.
- A `DENSE` focus (>12 leaves populated) is usually the safest story source.
- A `thin` focus (<6 leaves) often yields fabricated stories — prefer not
  to lead with it.
- The HEALTH METRICS PANEL above lists what each focus contains right now;
  re-check before choosing your lead focus.

{_CANONICAL_FIELD_ASSERTIONS}

{_build_canonical_field_glossary()}

CONTEXT SIGNALS (MANDATORY when present):
Treat active context signals as situational constraints, not as the insight topic.
Higher priority wins when recommendations would conflict:
  critical > high > medium > low
When sleep_window is active (critical): prioritize sleep/recovery; do NOT make
exercise the primary recommendation; encourage rest/sleep when appropriate.
When weekend is active (low): avoid unnecessary exercise pressure; prefer
balanced / recovery-oriented guidance — still respect health evidence.
When work_hours is active (medium): keep suggested actions realistic for a
workday context.
When recent_workout is active (high): interpret elevated RHR / high activity /
recovery dips in light of the recent session — do not treat them in isolation.
Do NOT invent context. Do NOT ignore a higher-priority active context signal.

DATA STALENESS WARNING (CRITICAL):
If sleep/hr/steps/energy staleness_days >= 1: data is NOT from today — state the date.
If staleness >= 2: cautious language (may / seems / appears). Never invent metrics.

--------------------------------------------------
STEP 1 — Collect Evidence
--------------------------------------------------
Read every available HEALTH source in context + narrative + HEALTH METRICS PANEL.
Scan ALL focuses when present: sleep, heart_rate, steps, energy, mood,
health_score, period_aggregates, context_signals.
Possible inputs (only when present): health metrics, mood (descriptive), lifestyle
signals tied to body recovery, goals related to sleep/activity, historical baseline,
weekly trends, monthly trends, user habits / profile fragments in context.
Never assume. Every statement must be backed by evidence.
Do NOT skip non-steps focuses just because steps fields are denser.

--------------------------------------------------
STEP 2 — Find Candidate Stories
--------------------------------------------------
Search for situations such as:
Behavior change · Habit recovery · Habit decline · Momentum · Warning signals ·
Positive progress · Hidden opportunity · Upcoming risk · Consistency · Recovery ·
Stress accumulation · Energy changes · Sleep patterns · Motivation · Personal routines.

Only choose ONE story — the one with the highest expected value for the user TODAY.

--------------------------------------------------
STEP 3 — Multi-layer Reasoning
--------------------------------------------------
Reason in this order (fill `reasoning` fields; do not invent missing layers):

Layer 1 — Current Context
What is happening today? What numbers prove it?

Layer 2 — Trends
Compare when data exists: today vs yesterday · vs weekly average · vs monthly average ·
vs personal baseline. Improving / declining / stable?

Layer 3 — Personal Context
What is normal for this user? Which habits matter? What usually causes this?
Have similar situations happened before? What usually happens afterwards?
(Only from available history/habits — never invent memories.)

Layer 4 — Cross-domain Intelligence
SKIP for Health Insight. Set reasoning.cross_domain to "".

Layer 5 — Best Next Action
Suggest ONE action only. It must be realistic, fit today's schedule/energy/habits,
respect active context_signals (especially sleep_window / recent_workout / weekend),
and be actionable immediately. Explain WHY it has the highest expected benefit
(in reasoning.next_action_reason). Respect time_phase / bedtime / active-hours guards.

--------------------------------------------------
STEP 4 — Emotional Writing
--------------------------------------------------
Do NOT sound like analytics, a research paper, a hospital chart, or a coach textbook.
Write like a home-visit family doctor who is also a companion friend —
warm, calm, practical, plain words everyone understands.
Never assert the user's feelings ("you are feeling…", "bạn đang cảm thấy…") unless they logged that mood.
Prefer facts: schedule gaps, logged blocks, concrete numbers.
Never diagnose disease or prescribe medicine.

Avoid: "Your HRV is..." / "Your recovery trajectory is suboptimal..." / "You're feeling unfocused..."
Prefer: "Sleep was short last night." / "No focus block is logged yet today."

The title should create curiosity. Examples (rewrite into {lang_name}):
- Your body is asking for an early night
- Coffee is covering for something else
- You're improving for the right reasons
- Your brain hasn't had a real break

--------------------------------------------------
STEP 5 — Output Structure
--------------------------------------------------
Return JSON: {{"insights": [ ... ]}} with exactly one insight object.

Each insight MUST include:
  - question_id: string
  - title: curiosity-creating headline in {lang_name} (short; no "Insight:" prefix)
  - reasoning: object with:
      current_context, trend, personal_context, cross_domain ("" for health),
      next_action_reason
    (internal coach notes in {lang_name}; evidence-backed; may be brief)
  - insight: FINAL user-facing paragraph in {lang_name}, 80–180 words.
    Natural flow (do NOT label sections):
      Hook → Evidence → Interpretation → Future implication → One specific action
    Use concrete numbers. Always compare against personal baseline when present.
    Never recommend generic advice. Never invent data.
    Every recommendation must directly connect to the evidence.
    Must feel like something only Sylo could generate.
  - current_state: MUST be identical to `insight` (pipeline compatibility)
  - evidence: strongest 1–3 concrete data points IN {lang_name}
  - cause: brief interpretation note (optional; may echo reasoning)
  - action: the ONE specific next action (short phrase in {lang_name})
  - expected_outcome: brief hedged benefit of that action
  - action_family: short family tag if clear, else ""
  - tone: encouraging | urgent | gentle | celebratory | informative
  - confidence: 0.5–1.0

RULES:
- Prefer personal 3d / 7d / month / baseline over generic population norms — only if present.
- Never invent metrics, trends, habits, goals, or medical conclusions.
- Never mention productivity jargon (meetings, inbox, focus blocks, deadlines).
- Do NOT write section labels (Hook / Evidence / Action) inside title or insight.
- If little data is present: honest story from what exists + cautious small next step; lower confidence.
"""

def _build_productivity_system_prompt(
    time_phase: str, language: str, active_hours_end_time: str | None = None
) -> str:
    lang_name = get_language_name(language)
    domain_rules = get_productivity_rules_block(time_phase)
    constraint_guards = _resolve_constraint_guards(
        "productivity", active_hours_end_time
    )
    return f"""\
You are Sylo's Insight Generation Engine for PRODUCTIVITY.

Your job is NOT to summarize data.
Your job is to discover ONE meaningful productivity story hidden inside the user's data.
Every insight must feel like a home doctor and companion friend who knows the user's day —
calm, caring, practical — not a coach textbook or analytics bot.

There is NO separate advice / render / polish step — your `insight` (and `title`)
IS the final user-facing productivity insight.

{_language_block(language)}

DOMAIN RULES (MANDATORY):
{domain_rules}

GUARDRAIL (MANDATORY):
{constraint_guards}

{_QA_UX_RULES}

TIME PHASE: {time_phase}
Use for Layer 5 action timing: action must fit NOW and respect active hours.

FOCUS (PRODUCTIVITY ONLY — CALENDAR FIRST):
1) TODAY's calendar first (mandatory when present):
   calendar_intelligence / schedule narrative / free_windows_today /
   next_event / upcoming_events / meeting load / denseness / buffers.
2) Then secondary: reminders, task completion, focus blocks, utilization trends.
Prefer a today-schedule story over a pure task-score / focus-score story.
Also use when present:
- productivity_signals.summary / signals.* / anomalies / overall
- productivity narrative section
Do NOT force health/finance cross-domain intelligence.
Do NOT invent meetings, reminders, deadlines, or goals.

--------------------------------------------------
STEP 1 — Collect Evidence (calendar today → then rest)
--------------------------------------------------
FIRST read today's calendar picture:
events (title + time), denseness, free windows, next/upcoming events,
work-hours fit, buffers / back-to-back load.
THEN read reminder execution, commitments, focus blocks, utilization,
trends, baseline comparisons when present.
Every statement must be evidence-backed.

--------------------------------------------------
STEP 2 — Find Candidate Stories
--------------------------------------------------
Prefer calendar-based stories for today — IRRATIONALITY FIRST when present:
1) events overlapping / after bedtime goal (e.g. "làm việc" 22:00 vs bedtime 21:00)
2) events crowding wind-down (within ~60m of bedtime)
3) dense / fragmented / unreasonable late windows vs active hours
4) next event soon / useful free window (only if no sleep-schedule conflict)
Secondary stories (only if calendar has little to say):
execution decline, backlog risk, commitment recovery, focus opportunity, momentum.
Choose ONE story with highest expected value for today —
schedule irrationality (esp. sleep overlap) ALWAYS wins over steps/walk tips and focus scores.

--------------------------------------------------
STEP 3 — Multi-layer Reasoning
--------------------------------------------------
Layer 1 — Current context: what is happening now? which numbers prove it?
Layer 2 — Trends: today vs yesterday / 7d / month / baseline when present.
Layer 3 — Personal context: user routine, work-hour boundaries, recurring patterns.
Layer 4 — Cross-domain: SKIP for Productivity Insight. Set reasoning.cross_domain to "".
Layer 5 — Best next action: ONE realistic action for now + why highest benefit.

--------------------------------------------------
STEP 4 — Emotional Writing
--------------------------------------------------
Write like a home doctor + companion friend — warm, calm, practical —
not an analytics dump, KPI review, hospital chart, or coach textbook.
Keep prose plain: short everyday words everyone understands — never academic or jargon-heavy.
Title should create curiosity.

--------------------------------------------------
STEP 5 — Output Structure
--------------------------------------------------
Return JSON: {{"insights": [ ... ]}} with exactly one insight object.

Each insight object MUST include:
  - question_id: string
  - title: curiosity-creating headline in {lang_name}
  - reasoning: object with
      current_context, trend, personal_context, cross_domain (""), next_action_reason
  - insight: FINAL user-facing paragraph in {lang_name}, 80–180 words.
    Natural flow (no labels): Hook → Evidence → Interpretation → Future implication → One specific action
  - current_state: MUST be identical to `insight` (pipeline compatibility)
  - evidence: strongest 1–3 concrete productivity data points IN {lang_name}
  - cause: brief interpretation note (optional)
  - action: the ONE specific next action (short phrase)
  - expected_outcome: brief hedged benefit
  - action_family: short family tag if clear, else ""
  - tone: encouraging | urgent | gentle | celebratory | informative
  - confidence: 0.5–1.0

RULES:
- Use concrete numbers from processor signals.
- Prefer personal baseline / 7d / month comparisons when available.
- Never invent data.
- No section labels in title or insight.
- Keep productivity-only; no health physiology framing.
"""


def _build_overall_system_prompt(
    time_phase: str, language: str, active_hours_end_time: str | None = None
) -> str:
    lang_name = get_language_name(language)
    domain_rules = get_overall_rules_block()
    constraint_guards = _resolve_constraint_guards("overall", active_hours_end_time)
    return f"""\
You are Sylo's Insight Generation Engine for HOME / OVERALL.

Your job is NOT to summarize data.
Your job is to discover ONE meaningful story hidden inside the user's life data.
Every insight must feel like a home doctor and companion friend who knows the user's day —
calm, caring, practical — not a coach textbook or analytics bot.

There is NO separate advice / render / polish step — your `insight` (and `title`)
IS the final user-facing overall insight.

{_language_block(language)}

DOMAIN RULES (MANDATORY):
{domain_rules}

GUARDRAIL (MANDATORY):
{constraint_guards}

{_QA_UX_RULES}

TIME PHASE: {time_phase}
(Use for Layer 5 action timing and feasibility.)

DATA STALENESS WARNING (CRITICAL):
If sleep/hr/steps/energy staleness_days >= 1: data is NOT from today — state the date.
If staleness >= 2: use cautious language (may / seems / appears). Never invent metrics.

OVERALL SIGNAL MAP (use only present fields):
- health_signals: sleep / heart_rate / steps / energy / mood
  (summary / signals / anomalies / overall)
- productivity_signals: summary / signals / anomalies / overall
- finance_signals
- balance_snapshot (today + 7d aggregates)
- goal_signals
- behavioral_patterns / historical_trends
- narrative_context

--------------------------------------------------
STEP 1 — Collect Evidence
--------------------------------------------------
FIRST check schedule vs bedtime / sleep goals for irrationality
(events after bedtime, late work overlapping sleep window).
THEN read all available cross-domain sources above.
Never assume. Every statement must be evidence-backed.

--------------------------------------------------
STEP 2 — Find Candidate Stories
--------------------------------------------------
PRIORITY ORDER:
1) Schedule irrationality (calendar overlapping bedtime / sleep) — always first when present
2) Other cross-domain situations: overload, recovery momentum, life-balance drift,
   goal progress risk, hidden opportunity, upcoming risk, consistency, resilience
Do NOT lead with steps/walk tips when a sleep-schedule conflict is present
(e.g. bedtime 21:00 + "làm việc" 22:00 beats "free window before coffee — walk more").
Choose ONE story with highest expected value for today.

--------------------------------------------------
STEP 3 — Multi-layer Reasoning
--------------------------------------------------
Layer 1 — Current context: what is happening now? numbers?
Layer 2 — Trends: today vs yesterday / 7d / month / baseline when present.
Layer 3 — Personal context: routines, habits, goals, historical patterns.
Layer 4 — Cross-domain intelligence: explain WHY multiple domains support same conclusion.
Layer 5 — Best next action: ONE realistic action for now + why highest benefit.

--------------------------------------------------
STEP 4 — Emotional Writing
--------------------------------------------------
Write like a home doctor + companion friend — warm, calm, practical —
not an analytics dump, KPI review, hospital chart, or coach textbook.
Keep prose plain: short everyday words everyone understands — never academic or jargon-heavy.
Title should create curiosity.

--------------------------------------------------
STEP 5 — Output Structure
--------------------------------------------------
Return JSON: {{"insights": [ ... ]}} with exactly one insight object.

Each insight MUST include:
  - question_id: string
  - title: curiosity-creating headline in {lang_name}
  - reasoning: object with:
      current_context, trend, personal_context, cross_domain, next_action_reason
  - insight: FINAL user-facing paragraph in {lang_name}, 80–180 words.
    Natural flow (no labels):
      Hook → Evidence → Interpretation → Future implication → One specific action
  - current_state: MUST be identical to `insight` (pipeline compatibility)
  - evidence: strongest 1–3 concrete cross-domain data points in {lang_name}
  - cause: brief interpretation note (optional)
  - action: the ONE specific next action (short phrase)
  - expected_outcome: brief hedged benefit
  - action_family: short family tag if clear, else ""
  - tone: encouraging | urgent | gentle | celebratory | informative
  - confidence: 0.5–1.0

RULES:
- Never invent metrics, trends, goals, events, or medical conclusions.
- Cross-domain is required only when supported by evidence.
- Do NOT write section labels inside title or insight.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Question groups — each group is one LLM call
# ─────────────────────────────────────────────────────────────────────────────

# Health: Sylo overview — one story → title + insight (Hook→Evidence→…→Action).
_HEALTH_GROUP_OVERVIEW = {
    "group_id": "h_group_overview",
    "focus": "overview",
    "description": (
        "HEALTH OVERVIEW (Sylo) — discover ONE meaningful health story from "
        "health_signals (sleep, heart_rate, steps/activity, energy, mood) plus "
        "context_signals (sleep_window, weekend, work_hours, recent_workout) "
        "and the health narrative. Collect evidence → pick one story → reason "
        "Layers 1–3 + 5 (SKIP Layer 4 cross-domain). Respect active "
        "context_signals by priority. Output title + insight "
        "(Hook → Evidence → Interpretation → Future implication → One action). "
        "Only present fields; never invent; mood descriptive only."
    ),
    "questions": [
        {
            "id": "h_overview_infer",
            "text": (
                "Đọc TOÀN BỘ health_signals hiện có "
                "(sleep, heart_rate, steps/activity, energy, mood, health_score — "
                "summary / signals.*.status|metrics|comparison|trend / anomalies / overall) "
                "và health_signals.context_signals "
                "(sleep_window / weekend / work_hours / recent_workout — "
                "status + priority + evidence) "
                "và phần health trong narrative. "
                "Thu thập evidence (không giả định); tìm các candidate stories "
                "(recovery, sleep patterns, energy changes, activity volume / goal / "
                "step trend, habit decline/recovery, "
                "warning signals, progress, …) rồi CHỌN ĐÚNG MỘT story "
                "có giá trị cao nhất cho user hôm nay. "
                "Reason: Layer 1 current context → Layer 2 trends "
                "(vs yesterday / 7d / month / baseline nếu có) → "
                "Layer 3 personal context → SKIP Layer 4 → "
                "Layer 5 một action khả thi ngay (phù hợp time_phase). "
                "Viết title gây tò mò + insight 80–180 từ (giọng bác sĩ tại gia + bạn đồng hành, "
                "phổ thông ai cũng hiểu, có số liệu, so baseline cá nhân; flow Hook→Evidence→Interpretation→"
                "Future implication→One action; không gắn nhãn section; tránh từ hàn lâm; không đoán cảm xúc). "
                "current_state = insight. Không bịa data; không chẩn đoán; "
                "mood chỉ mô tả."
            ),
            "metrics": ["sleep", "heart_rate", "activity", "energy", "mood", "health_score", "context"],
            "signal_keys": [
                "sleep",
                "heart_rate",
                "steps",
                "energy",
                "mood",
                "health_score",
                "context_signals",
            ],
        },
    ],
}

# Productivity: Sylo overview — one story from processor signals.
_PRODUCTIVITY_GROUP_OVERVIEW = {
    "group_id": "p_group_overview",
    "focus": "overview",
    "description": (
        "PRODUCTIVITY OVERVIEW (Sylo) — CALENDAR FIRST: start from TODAY's schedule "
        "(events, denseness, free windows, next/upcoming), then secondary reminder/task signals. "
        "Discover ONE meaningful story. Collect evidence → pick one story → "
        "reason Layers 1–3 + 5 (SKIP Layer 4 cross-domain). "
        "Output title + insight (Hook → Evidence → Interpretation → Future implication → One action). "
        "Use only present processor fields; never invent."
    ),
    "questions": [
        {
            "id": "p_overview_infer",
            "text": (
                "ƯU TIÊN LỊCH HÔM NAY trước: đọc calendar_intelligence / schedule facts / "
                "free_windows_today / next_event / upcoming_events / phần lịch trong narrative "
                "(sự kiện, độ dày, khoảng trống, lịch tiếp theo) và BEDTIME_GOALS. "
                "Nếu có SCHEDULE_IRRATIONALITY (ví dụ sự kiện chồng / sau giờ ngủ như "
                "'làm việc' 22:00 trong khi bedtime 21:00) → ĐÓ LÀ STORY #1, "
                "không dẫn bằng đi bộ tăng bước / focus score / khoảng rảnh trước cf. "
                "Sau đó mới xem productivity_signals còn lại "
                "(summary, signals.*, anomalies, overall) và reminder/task nếu cần hỗ trợ. "
                "Thu thập evidence; ưu tiên candidate stories: "
                "(1) lịch bất hợp lý vs giờ ngủ, (2) lịch dày/xé nhỏ/sắp tới sự kiện, "
                "(3) khoảng rảnh hữu ích — hơn story thuần focus-score / task%. "
                "CHỌN ĐÚNG MỘT story có giá trị cao nhất hôm nay. "
                "Reason: Layer 1 current context → Layer 2 trends "
                "(vs yesterday / 7d / month / baseline nếu có) → Layer 3 personal context "
                "→ SKIP Layer 4 → Layer 5 một action khả thi ngay (phù hợp time_phase). "
                "Viết title gây tò mò + insight 80–180 từ (giọng bác sĩ tại gia + bạn đồng hành, "
                "phổ thông ai cũng hiểu, có số liệu cụ thể, "
                "flow Hook→Evidence→Interpretation→Future implication→One action; "
                "không gắn nhãn section; tránh từ hàn lâm; không đoán cảm xúc). "
                "Hook/evidence chính nên bám lịch hôm nay khi có dữ liệu lịch. "
                "current_state = insight. Không bịa data."
            ),
            "metrics": ["productivity", "calendar", "reminder"],
            "signal_keys": [
                "calendar_intelligence",
                "summary",
                "signals",
                "anomalies",
                "overall",
            ],
        }
    ],
}

# Overall: Sylo overview — one cross-domain story from standardized processor signals.
_OVERALL_GROUP_OVERVIEW = {
    "group_id": "o_group_overview",
    "focus": "overview",
    "description": (
        "OVERALL OVERVIEW (Sylo) — prioritize SCHEDULE IRRATIONALITY first "
        "(e.g. calendar events overlapping bedtime), then other cross-domain stories. "
        "Discover ONE meaningful story from health_signals, productivity_signals, "
        "finance_signals, balance_snapshot, goal_signals, behavioral_patterns, historical_trends. "
        "Collect evidence → pick one story → reason Layers 1–5. "
        "Output title + insight (Hook → Evidence → Interpretation → Future implication → One action)."
    ),
    "questions": [
        {
            "id": "o_overview_infer",
            "text": (
                "ƯU TIÊN phát hiện bất hợp lý lịch trước (đặc biệt chồng giờ ngủ): "
                "đọc SCHEDULE FACTS / BEDTIME_GOALS / calendar upcoming "
                "(ví dụ bedtime 21:00 mà có 'làm việc' 22:00). "
                "Nếu có conflict này → CHỌN ĐÓ LÀ story chính; "
                "KHÔNG dẫn bằng 'khoảng rảnh trước cf — đi bộ tăng bước'. "
                "Nếu không có irrationality, mới đọc health_signals, productivity_signals, "
                "finance_signals, balance_snapshot, goal_signals, behavioral_patterns, historical_trends "
                "và narrative để chọn MỘT story liên miền giá trị cao nhất hôm nay. "
                "Reason theo Layer 1→2→3→4→5 "
                "(Layer 4 phải giải thích vì sao nhiều domain cùng ủng hộ kết luận). "
                "Viết title gây tò mò + insight 80–180 từ (giọng bác sĩ tại gia + bạn đồng hành, "
                "phổ thông ai cũng hiểu, có số liệu cụ thể, "
                "flow Hook→Evidence→Interpretation→Future implication→One action; "
                "không gắn nhãn section; tránh từ hàn lâm; không đoán cảm xúc). "
                "current_state = insight. Không bịa data."
            ),
            "metrics": [
                "health",
                "productivity",
                "finance",
                "balance",
                "goals",
                "behavior",
            ],
            "signal_keys": [
                "health_signals",
                "productivity_signals",
                "finance_signals",
                "balance_snapshot",
                "goal_signals",
                "behavioral_patterns",
                "historical_trends",
                "calendar_intelligence",
            ],
        }
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API — build prompts for each group
# ─────────────────────────────────────────────────────────────────────────────

def _goals_block_from_context(context_json: str) -> str:
    """Attach bedtime_goals (+ work_hours_goals) for productivity Q&A context."""
    try:
        from insights.health.health_processor.context.goal_context import build_bedtime_goals
        from insights.health.health_processor.context.goal_context import build_work_hours_goals
        from insights.schemas.processed_context import ExtractedSignals

        data = json.loads(context_json) if isinstance(context_json, str) else context_json
        if not isinstance(data, dict):
            return ""
        signals = ExtractedSignals.model_validate(data)
        bedtime = build_bedtime_goals(signals)
        work_hours = build_work_hours_goals(signals)
        parts: list[str] = []
        if bedtime:
            parts.append(
                "BEDTIME_GOALS (TONIGHT's upcoming schedule boundary — NOT last night's sleep):\n"
                "Distinct from health_signals.sleep.bedtime / LAST_NIGHT_BEDTIME "
                "(that is closed history from the night already slept).\n"
                "If any TODAY event starts at/after bedtime_start, that is SCHEDULE_IRRATIONALITY "
                "for TONIGHT — do not claim it caused last night's bedtime.\n"
                + json.dumps(bedtime, ensure_ascii=False, default=str)
            )
        if work_hours:
            parts.append(
                "WORK_HOURS_GOALS (active / work-hours window — weigh when relevant):\n"
                + json.dumps(work_hours, ensure_ascii=False, default=str)
            )
        return ("\n\n" + "\n\n".join(parts)) if parts else ""
    except Exception:
        return ""


def _clock_mins(value: Any) -> int | None:
    """Parse HH:MM / HH:MM:SS / ISO datetime into minutes-from-midnight."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # ISO / datetime → take time portion
    if "T" in s:
        s = s.split("T", 1)[1]
    s = s.replace("Z", "").split("+", 1)[0].split("-", 1)[0]  # strip tz if +07:00
    m = re.search(r"(\d{1,2}):(\d{2})", s)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 23 or mi > 59:
        return None
    return h * 60 + mi


def _bedtime_start_from_data(data: dict) -> str | None:
    """Best-effort bedtime goal start from context dict."""
    try:
        from insights.health.health_processor.context.goal_context import build_bedtime_goals
        from insights.schemas.processed_context import ExtractedSignals

        signals = ExtractedSignals.model_validate(data)
        bedtime = build_bedtime_goals(signals)
        if isinstance(bedtime, dict) and bedtime.get("bedtime_start"):
            return str(bedtime["bedtime_start"])
    except Exception:
        pass
    meta = data.get("meta") or {}
    for key in ("bedtime_start", "bedtime_start_str", "bedtime"):
        if meta.get(key):
            return str(meta[key])
    raw = data.get("raw_data") if isinstance(data.get("raw_data"), dict) else {}
    td = raw.get("time_data") if isinstance(raw.get("time_data"), dict) else {}
    return (
        td.get("bedtime_start_str")
        or td.get("bedtime_start")
        or None
    )


def _collect_today_event_briefs(data: dict) -> list[dict]:
    """Unique brief event dicts with title + start_time from calendar/meta."""
    seen: set[tuple] = set()
    out: list[dict] = []

    def _add(title: Any, start: Any, category: Any = None) -> None:
        if not title or start is None:
            return
        key = (str(title), str(start))
        if key in seen:
            return
        seen.add(key)
        item = {"title": str(title), "start_time": str(start)}
        if category:
            item["category"] = str(category)
        out.append(item)

    cal = data.get("calendar_intelligence") or {}
    if isinstance(cal, dict):
        for key in ("next_event", "current_event"):
            ev = cal.get(key)
            if isinstance(ev, dict):
                _add(ev.get("title"), ev.get("start_time"), ev.get("category"))
        for u in cal.get("upcoming_events") or []:
            if isinstance(u, dict):
                _add(u.get("title"), u.get("start_time"), u.get("category"))
    meta = data.get("meta") or {}
    _add(
        meta.get("next_event_summary"),
        meta.get("next_event_start_time"),
        meta.get("next_event_category"),
    )
    return out


def _append_schedule_irrationality(lines: list[str], data: dict) -> None:
    """Flag high-priority schedule conflicts (e.g. events after bedtime)."""
    bedtime_raw = _bedtime_start_from_data(data)
    bedtime_mins = _clock_mins(bedtime_raw)
    if bedtime_mins is None:
        return

    lines.append(f"BEDTIME_GOAL_START: {bedtime_raw} (schedule boundary)")

    overlaps: list[str] = []
    crowds: list[str] = []
    for ev in _collect_today_event_briefs(data):
        start_mins = _clock_mins(ev.get("start_time"))
        if start_mins is None:
            continue
        title = ev["title"]
        start = ev.get("start_time")
        # Evening bedtime (>=18:00): event at/after bedtime → overlaps sleep
        # Also early-morning events before typical wake (00:00–04:59) after evening bedtime
        overlaps_sleep = False
        if bedtime_mins >= 18 * 60:
            if start_mins >= bedtime_mins or start_mins < 5 * 60:
                overlaps_sleep = True
        elif start_mins >= bedtime_mins:
            overlaps_sleep = True

        if overlaps_sleep:
            overlaps.append(f"{title!r}@{start}")
            continue

        # Starts within 60 minutes before bedtime → crowds wind-down
        gap = bedtime_mins - start_mins
        if 0 < gap <= 60:
            crowds.append(f"{title!r}@{start} (~{_format_duration_hm(gap)} before bedtime)")

    if overlaps:
        lines.append(
            "SCHEDULE_IRRATIONALITY (PRIORITY #1 — surface this before steps/focus/free-window tips): "
            f"event(s) overlap / after bedtime_goal {bedtime_raw}: "
            + "; ".join(overlaps)
            + ". Prefer this story over 'walk more before coffee' or abstract productivity scores."
        )
    if crowds:
        lines.append(
            "SCHEDULE_NEAR_BEDTIME (high priority): "
            + "; ".join(crowds)
            + ". Do not push long exercise / deep work that eats into wind-down."
        )


def _format_duration_hm(minutes: Any) -> str:
    """Format duration for prompts: <60 → Nm; else Nh / NhMm."""
    if minutes is None:
        return "n/a"
    try:
        total = int(round(float(minutes)))
    except (TypeError, ValueError):
        return "n/a"
    if total < 0:
        total = abs(total)
    if total < 60:
        return f"{total}m"
    h, m = divmod(total, 60)
    return f"{h}h{m}m" if m else f"{h}h"


def _build_schedule_facts_banner(data: dict) -> str:
    """Precomputed schedule facts so the LLM does not guess event countdowns."""
    lines: list[str] = []
    meta = data.get("meta") or {}
    if meta.get("current_time"):
        lines.append(f"CURRENT_TIME: {meta['current_time']}")

    cal = data.get("calendar_intelligence") or {}
    curr = cal.get("current_event") if isinstance(cal, dict) else None
    if isinstance(curr, dict) and curr.get("title"):
        lines.append(
            "CURRENT_CALENDAR_EVENT (kind=EVENT / sự kiện — NOT a call/meeting unless title or "
            "event_type says so; NOT a reminder): "
            f"title={curr.get('title')!r} "
            f"category={curr.get('category')!r} "
            f"event_type={curr.get('event_type')!r} "
            f"start={curr.get('start_time')!r} "
            f"end={curr.get('end_time')!r} "
            f"left={_format_duration_hm(curr.get('minutes_left'))} "
            f"(raw_minutes_left={curr.get('minutes_left')}). "
            "VOCAB LOCK: in title+body say only 'sự kiện'/'event' for this item — "
            "never 'cuộc gọi'/'call'/'cuộc trò chuyện'."
        )

    nxt = cal.get("next_event") if isinstance(cal, dict) else None
    if isinstance(nxt, dict) and nxt.get("title"):
        cat = nxt.get("category")
        et = nxt.get("event_type")
        lines.append(
            "NEXT_CALENDAR_EVENT (kind=EVENT / sự kiện — not a reminder; do not invent call/meeting): "
            f"title={nxt.get('title')!r} "
            f"category={cat!r} "
            f"event_type={et!r} "
            f"start={nxt.get('start_time')!r} "
            f"in={_format_duration_hm(nxt.get('minutes_away'))} "
            f"(raw_minutes={nxt.get('minutes_away')})"
        )
    elif meta.get("next_event_summary"):
        lines.append(
            "NEXT_CALENDAR_EVENT (kind=EVENT / sự kiện — not a reminder; do not invent call/meeting): "
            f"title={meta.get('next_event_summary')!r} "
            f"category={meta.get('next_event_category')!r} "
            f"event_type={meta.get('next_event_type')!r} "
            f"start={meta.get('next_event_start_time')!r} "
            f"in={_format_duration_hm(meta.get('minutes_until_next_event'))} "
            f"(raw_minutes={meta.get('minutes_until_next_event')})"
        )

    upcoming = cal.get("upcoming_events") if isinstance(cal, dict) else None
    if isinstance(upcoming, list) and upcoming:
        brief_bits = []
        for u in upcoming[:5]:
            if not isinstance(u, dict):
                continue
            bit = f"{u.get('title')}"
            if u.get("category"):
                bit += f"[{u.get('category')}]"
            if u.get("minutes_away") is not None:
                bit += f"@{_format_duration_hm(u.get('minutes_away'))}"
            brief_bits.append(bit)
        if brief_bits:
            lines.append(
                "UPCOMING_EVENTS (kind=EVENT / sự kiện; title[category]@duration): "
                + "; ".join(brief_bits)
            )

    nearest_rem = cal.get("nearest_reminder") if isinstance(cal, dict) else None
    if isinstance(nearest_rem, dict) and nearest_rem.get("title"):
        lines.append(
            "NEAREST_REMINDER (kind=REMINDER / ghi nhớ — NOT an event, NOT a call): "
            f"title={nearest_rem.get('title')!r} "
            f"due_in={_format_duration_hm(nearest_rem.get('due_in_mins') or nearest_rem.get('minutes_away'))} "
            f"priority={nearest_rem.get('priority')!r}"
        )
    overdue = cal.get("overdue_reminder_titles") if isinstance(cal, dict) else None
    if isinstance(overdue, list) and overdue:
        lines.append(
            "OVERDUE_REMINDERS (kind=REMINDER / ghi nhớ): "
            + "; ".join(f"{t!r}" for t in overdue[:5])
        )

    ctx = (data.get("health_signals") or {}).get("context_signals") or {}
    wh = ctx.get("work_hours") if isinstance(ctx, dict) else None
    if isinstance(wh, dict) and wh.get("status") == "active":
        ev = wh.get("evidence") or {}
        lines.append(
            "WORK_HOURS_ACTIVE (NOT a calendar event): "
            f"{ev.get('work_start')}–{ev.get('work_end')} "
            f"current={ev.get('current_time')} — "
            "do NOT treat work_end as the next appointment time"
        )

    free_windows = cal.get("free_windows_today") if isinstance(cal, dict) else None
    if isinstance(free_windows, list) and free_windows:
        fw_bits = []
        for fw in free_windows[:3]:
            if not isinstance(fw, dict):
                continue
            bit = f"{_format_duration_hm(fw.get('duration_mins'))}"
            if fw.get("position"):
                bit += f"/{fw.get('position')}"
            if fw.get("capped_by"):
                bit += f"/capped_by={fw.get('capped_by')}"
            if fw.get("ends_at"):
                bit += f"/ends_at={fw.get('ends_at')}"
            fw_bits.append(bit)
        if fw_bits:
            lines.append(
                "WORK_FREE_SLOTS (often end at active_hours — NOT time until a later event): "
                + "; ".join(fw_bits)
            )
            nxt_mins = None
            if isinstance(nxt, dict):
                nxt_mins = nxt.get("minutes_away")
            elif meta.get("minutes_until_next_event") is not None:
                nxt_mins = meta.get("minutes_until_next_event")
            first_dur = free_windows[0].get("duration_mins") if isinstance(free_windows[0], dict) else None
            first_cap = free_windows[0].get("capped_by") if isinstance(free_windows[0], dict) else None
            if (
                isinstance(nxt_mins, (int, float))
                and isinstance(first_dur, (int, float))
                and first_cap == "active_hours_end"
                and nxt_mins > first_dur + 5
            ):
                lines.append(
                    "WARNING: free-slot duration ≠ time until next calendar event. "
                    f"Use next_event.in={_format_duration_hm(nxt_mins)} for 'before appointment'; "
                    f"use free-slot {_format_duration_hm(first_dur)} only as remainder of work hours."
                )

    _append_schedule_irrationality(lines, data)

    if not lines:
        return ""
    return (
        "SCHEDULE FACTS (precomputed — do not invent or recalculate minutes):\n"
        + "\n".join(f"- {line}" for line in lines)
        + "\n\n"
    )


def _dim_status_bits(signals_obj: Any, pairs: List[tuple]) -> List[str]:
    """Collect ``name=status`` for dimension signals that are present."""
    if not signals_obj or not isinstance(signals_obj, dict):
        return []
    bits: List[str] = []
    for name, key in pairs:
        dim = signals_obj.get(key)
        if not isinstance(dim, dict):
            continue
        st = dim.get("status")
        if st and st not in ("unknown", "insufficient_data", None, ""):
            bits.append(f"{name}={st}")
    return bits


def _health_metrics_panel(data: Dict[str, Any]) -> str:
    """Balanced digest of ALL health focuses for LLM reasoning (pre-JSON)."""
    hs = data.get("health_signals") or {}
    if not isinstance(hs, dict):
        return ""
    meta = data.get("meta") or {}
    lines: List[str] = [
        "HEALTH METRICS PANEL — read EVERY focus below before choosing the story.",
        "Do NOT default to steps/activity; pick the highest-value story across all metrics.",
    ]

    # Sleep
    sleep = hs.get("sleep") or {}
    if isinstance(sleep, dict) and sleep:
        overall = sleep.get("overall") or {}
        bits = [
            f"overall={overall.get('status')}" if overall.get("status") else None,
            f"last_night_h={sleep.get('last_night_h')}"
            if sleep.get("last_night_h") is not None
            else None,
            f"debt_h={sleep.get('debt_hours')}"
            if sleep.get("debt_hours") is not None
            else None,
            f"score={sleep.get('sleep_score')}"
            if sleep.get("sleep_score") is not None
            else None,
            f"level={sleep.get('level')}" if sleep.get("level") else None,
        ]
        sig_bits = _dim_status_bits(
            sleep.get("signals"),
            [
                ("duration", "sleep_duration"),
                ("efficiency", "sleep_efficiency"),
                ("consistency", "sleep_consistency"),
                ("deep", "deep_sleep"),
                ("rem", "rem_sleep"),
                ("debt", "sleep_debt"),
                ("trend", "sleep_trend"),
                ("recovery", "recovery"),
            ],
        )
        if sig_bits:
            bits.append("signals[" + ", ".join(sig_bits) + "]")
        stale = meta.get("sleep_data_staleness_days")
        if stale is not None:
            bits.append(f"staleness_days={stale}")
        lines.append("- SLEEP: " + "; ".join(b for b in bits if b) if any(bits) else "- SLEEP: present")

    # Heart rate
    hr = hs.get("heart_rate") or hs.get("cardio_stress") or {}
    if isinstance(hr, dict) and hr:
        overall = hr.get("overall") or {}
        bits = [
            f"overall={overall.get('status')}" if overall.get("status") else None,
            f"rhr={hr.get('resting_heart_rate')}"
            if hr.get("resting_heart_rate") is not None
            else None,
            f"baseline_rhr={hr.get('baseline_resting_hr')}"
            if hr.get("baseline_resting_hr") is not None
            else None,
            f"hrv={hr.get('hrv_score')}" if hr.get("hrv_score") is not None else None,
            f"latest={hr.get('latest_heart_rate')}"
            if hr.get("latest_heart_rate") is not None
            else None,
            "stress=elevated" if hr.get("stress_high") else None,
        ]
        sig_bits = _dim_status_bits(
            hr.get("signals"),
            [
                ("rhr", "resting_heart_rate"),
                ("hrv", "heart_rate_variability"),
                ("recovery", "recovery_state"),
                ("stability", "heart_rate_stability"),
                ("baseline", "baseline_comparison"),
                ("stress", "stress_indicator"),
            ],
        )
        if sig_bits:
            bits.append("signals[" + ", ".join(sig_bits) + "]")
        stale = meta.get("hr_data_staleness_days")
        if stale is not None:
            bits.append(f"staleness_days={stale}")
        lines.append("- HEART_RATE: " + "; ".join(b for b in bits if b) if any(bits) else "- HEART_RATE: present")

    # Steps
    steps = hs.get("steps") or hs.get("activity") or {}
    if isinstance(steps, dict) and steps:
        overall = steps.get("overall") or {}
        summary = steps.get("summary") or {}
        bits = [
            f"overall={overall.get('status')}" if overall.get("status") else None,
            f"today={steps.get('steps_today')}"
            if steps.get("steps_today") is not None
            else None,
            f"goal={steps.get('steps_goal')}"
            if steps.get("steps_goal") is not None
            else None,
            f"pace={steps.get('pace_ratio')}"
            if steps.get("pace_ratio") is not None
            else None,
            f"avg_7d={summary.get('avg_steps')}"
            if isinstance(summary, dict) and summary.get("avg_steps") is not None
            else None,
            f"level={steps.get('level')}" if steps.get("level") else None,
        ]
        sig_bits = _dim_status_bits(
            steps.get("signals"),
            [
                ("volume", "activity_volume"),
                ("goal", "goal_achievement"),
                ("consistency", "activity_consistency"),
                ("trend", "step_trend"),
                ("baseline", "baseline_comparison"),
                ("pattern", "activity_pattern"),
            ],
        )
        if sig_bits:
            bits.append("signals[" + ", ".join(sig_bits) + "]")
        anoms = steps.get("anomalies") or []
        if isinstance(anoms, list) and anoms:
            kinds = [a.get("kind") for a in anoms if isinstance(a, dict) and a.get("kind")]
            # partial_day is common midday noise — list but de-emphasize in panel
            if kinds:
                bits.append("anomalies[" + ", ".join(kinds[:4]) + "]")
        stale = meta.get("steps_data_staleness_days")
        if stale is not None:
            bits.append(f"staleness_days={stale}")
        lines.append("- STEPS: " + "; ".join(b for b in bits if b) if any(bits) else "- STEPS: present")

    # Energy
    energy = hs.get("energy") or {}
    if isinstance(energy, dict) and energy:
        overall = energy.get("overall") or {}
        bits = [
            f"overall={overall.get('status')}" if overall.get("status") else None,
            f"active_kcal={energy.get('total_active_energy') or energy.get('calories_burned_today')}"
            if (energy.get("total_active_energy") or energy.get("calories_burned_today"))
            is not None
            else None,
            f"week_vs_prior_pct={energy.get('week_vs_prior_month_pct')}"
            if energy.get("week_vs_prior_month_pct") is not None
            else None,
            f"level={energy.get('level')}" if energy.get("level") else None,
        ]
        sig_bits = _dim_status_bits(
            energy.get("signals"),
            [
                ("volume", "activity_volume"),
                ("consistency", "energy_consistency"),
                ("trend", "energy_trend"),
                ("baseline", "baseline_comparison"),
                ("composition", "energy_composition"),
                ("pattern", "activity_pattern"),
            ],
        )
        if sig_bits:
            bits.append("signals[" + ", ".join(sig_bits) + "]")
        stale = meta.get("energy_data_staleness_days")
        if stale is not None:
            bits.append(f"staleness_days={stale}")
        lines.append("- ENERGY: " + "; ".join(b for b in bits if b) if any(bits) else "- ENERGY: present")

    # Mood
    mood = hs.get("mood") or {}
    if isinstance(mood, dict) and mood:
        overall = mood.get("overall") or {}
        bits = [
            f"overall={overall.get('status')}" if overall.get("status") else None,
            f"primary={mood.get('primary_mood')}" if mood.get("primary_mood") else None,
            f"trend_7d={mood.get('mood_trend_7d')}" if mood.get("mood_trend_7d") else None,
        ]
        sig_bits = _dim_status_bits(
            mood.get("signals"),
            [
                ("snapshot", "snapshot"),
                ("baseline", "baseline"),
                ("trend", "trend"),
                ("stability", "stability"),
                ("change", "change"),
            ],
        )
        if sig_bits:
            bits.append("signals[" + ", ".join(sig_bits) + "]")
        stale = meta.get("mood_data_staleness_days")
        if stale is not None:
            bits.append(f"staleness_days={stale}")
        lines.append("- MOOD: " + "; ".join(b for b in bits if b) if any(bits) else "- MOOD: present")

    # Health score
    hscore = hs.get("health_score") or {}
    if isinstance(hscore, dict) and hscore:
        summary = hscore.get("summary") or {}
        agg = hscore.get("aggregate") or {}
        overall = hscore.get("overall") or {}
        fired = hscore.get("signals") or []
        keys = []
        if isinstance(fired, list):
            keys = [f.get("key") for f in fired if isinstance(f, dict) and f.get("key")]
        bits = [
            f"overall={overall.get('status')}" if overall.get("status") else None,
            f"current={summary.get('current_score')}"
            if isinstance(summary, dict) and summary.get("current_score") is not None
            else None,
            f"previous={summary.get('previous_score')}"
            if isinstance(summary, dict) and summary.get("previous_score") is not None
            else None,
            f"delta={summary.get('delta')}"
            if isinstance(summary, dict) and summary.get("delta") is not None
            else None,
            f"theme={agg.get('theme')}"
            if isinstance(agg, dict) and agg.get("theme")
            else None,
            ("fired[" + ", ".join(keys[:6]) + "]") if keys else None,
        ]
        lines.append(
            "- HEALTH_SCORE: " + "; ".join(b for b in bits if b)
            if any(bits)
            else "- HEALTH_SCORE: present"
        )

    # Period aggregates (compact)
    period = hs.get("period_aggregates") or {}
    if isinstance(period, dict) and period:
        pa_bits: List[str] = []
        for metric in (
            "steps",
            "sleep_hours",
            "resting_heart_rate",
            "active_minutes",
            "calories_burned",
        ):
            m = period.get(metric)
            if not isinstance(m, dict):
                continue
            today = m.get("today")
            week = m.get("week_avg")
            base = m.get("baseline_avg")
            chunk = []
            if today is not None:
                chunk.append(f"today={today}")
            if week is not None:
                chunk.append(f"week={week}")
            if base is not None:
                chunk.append(f"baseline={base}")
            if chunk:
                pa_bits.append(f"{metric}(" + ", ".join(chunk) + ")")
        flags = []
        if period.get("rhr_elevated"):
            flags.append("rhr_elevated")
        if period.get("workout_high_recent_load"):
            flags.append("workout_high_recent_load")
        if flags:
            pa_bits.append("flags[" + ", ".join(flags) + "]")
        if pa_bits:
            lines.append("- PERIOD: " + "; ".join(pa_bits))

    # Context signals
    ctx_sigs = hs.get("context_signals") or {}
    if isinstance(ctx_sigs, dict) and ctx_sigs:
        active = []
        for name, item in ctx_sigs.items():
            if not isinstance(item, dict):
                continue
            if item.get("status") == "active":
                pri = item.get("priority") or "?"
                active.append(f"{name}[{pri}]")
        if active:
            lines.append("- CONTEXT_ACTIVE: " + ", ".join(active))
        else:
            lines.append("- CONTEXT_ACTIVE: (none)")

    lines.append(
        "Use the full health_signals JSON below for numbers/evidence; "
        "this panel is the coverage checklist."
    )
    return "\n".join(lines) + "\n\n"


def _slim_health_context_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep health-reasoning fields; drop bulky non-health / raw noise."""
    keep_keys = (
        "meta",
        "health_signals",
        "user_profile",
        "historical_trends",
        "behavioral_patterns",
        "goal_signals",
        "balance_score",
        "balance_snapshot",
        "narrative_context",
    )
    slim: Dict[str, Any] = {k: data[k] for k in keep_keys if k in data and data[k] is not None}

    # Tiny raw slice: health_params only (device numbers the processors used)
    raw = data.get("raw_data")
    if isinstance(raw, dict):
        tiny: Dict[str, Any] = {}
        hp = raw.get("health_params")
        if isinstance(hp, dict) and hp:
            tiny["health_params"] = hp
        if tiny:
            slim["raw_data"] = tiny
    return slim


def _serialize_health_context_block(context_json: str) -> str:
    """Health QA context: metrics panel + density report + staleness + slim JSON (all focuses)."""
    try:
        data = json.loads(context_json) if isinstance(context_json, str) else context_json
        if not isinstance(data, dict):
            return _serialize_context_block(context_json)

        panel = _health_metrics_panel(data)
        density = _build_data_density_report(data)

        # Reuse staleness / schedule banner from general serializer by
        # running it on the slim payload (still has meta + health_signals).
        slim = _slim_health_context_payload(data)
        body = _serialize_context_block(json.dumps(slim, ensure_ascii=False))
        return panel + density + body
    except Exception:
        return _serialize_context_block(context_json)


def _serialize_context_block(context_json: str) -> str:
    """Compact JSON for prompt context — strip None values to reduce noise."""
    try:
        data = json.loads(context_json) if isinstance(context_json, str) else context_json

        # Extract staleness info for explicit warning with last known values
        staleness_warning_lines = []
        meta = data.get("meta", {}) or {}
        health_signals = data.get("health_signals", {}) or {}

        # Sleep staleness - include last known value
        sleep_staleness = meta.get("sleep_data_staleness_days")
        sleep_last_date = meta.get("sleep_last_data_date")
        # Sentinel 999 = "no data ever"; otherwise >=1 means stale.
        if sleep_staleness == 999:
            staleness_warning_lines.append(
                "- SLEEP: NO DATA EVER — do NOT report sleep hours/quality/score as a fact. Tell the user to sync their device."
            )
        elif sleep_staleness is not None and sleep_staleness >= 1:
            sleep_signal = health_signals.get("sleep", {}) or {}
            last_h = sleep_signal.get("last_night_h")
            quality = sleep_signal.get("quality")
            score = sleep_signal.get("sleep_score")
            overall = sleep_signal.get("overall") or {}
            overall_status = overall.get("status") if isinstance(overall, dict) else None
            value_str = f"{last_h}h" if last_h else "N/A"
            if quality:
                value_str += f" ({quality})"
            if score:
                value_str += f", score: {score}"
            if overall_status:
                value_str += f", overall: {overall_status}"
            staleness_warning_lines.append(
                f"- SLEEP: last known value from {sleep_last_date} ({sleep_staleness} day(s) ago): {value_str} - ⚠️ NOT today's data. Use hedging language. Prefer sleep.signals / summary / overall if present — do not treat flat fields as fresh."
            )

        # HR staleness - include last known value
        hr_staleness = meta.get("hr_data_staleness_days")
        hr_last_date = meta.get("hr_last_data_date")
        if hr_staleness == 999:
            staleness_warning_lines.append(
                "- HEART RATE: NO DATA EVER — do NOT report resting HR / latest HR / HRV as a fact. Tell the user to sync their device."
            )
        elif hr_staleness is not None and hr_staleness >= 1:
            hr = health_signals.get("heart_rate") or health_signals.get("cardio_stress") or {}
            hr_val = hr.get("resting_heart_rate")
            hrv_val = hr.get("hrv_score")
            value_str = f"HR: {hr_val} bpm" if hr_val else ""
            if hrv_val:
                value_str += f", HRV: {hrv_val}"
            overall = hr.get("overall") or {}
            if isinstance(overall, dict) and overall.get("status"):
                value_str += f", overall: {overall.get('status')}"
            if value_str:
                value_str += " - "
            value_str += f"STALE from {hr_staleness}d ago"
            staleness_warning_lines.append(
                f"- HEART RATE: {value_str}. Prefer heart_rate.signals / summary / overall if present — do not treat flat fields as fresh."
            )

        # Steps staleness - include last known value
        steps_staleness = meta.get("steps_data_staleness_days")
        steps_last_date = meta.get("steps_last_data_date")
        if steps_staleness == 999:
            staleness_warning_lines.append(
                "- STEPS: NO DATA EVER — do NOT report steps as a fact. Tell the user to sync their device."
            )
        elif steps_staleness is not None and steps_staleness >= 1:
            activity = (
                health_signals.get("steps")
                or health_signals.get("activity")
                or {}
            )
            steps_val = activity.get("steps_today")
            goal_val = activity.get("steps_goal")
            value_str = f"{steps_val} steps"
            if goal_val:
                value_str += f" (goal: {goal_val})"
            staleness_warning_lines.append(
                f"- STEPS: {value_str} from {steps_last_date} ({steps_staleness} day(s) ago) - ⚠️ NOT today's data"
            )

        # Mood staleness - include last known value
        mood_staleness = meta.get("mood_data_staleness_days")
        mood_last_date = meta.get("mood_last_data_date")
        if mood_staleness == 999:
            staleness_warning_lines.append(
                "- MOOD: NO DATA EVER — do NOT report a mood label. Say 'chưa ghi nhận tâm trạng'."
            )
        elif mood_staleness is not None and mood_staleness >= 1:
            mood = health_signals.get("mood") or health_signals.get("cardio_stress") or {}
            mood_val = mood.get("mood")
            mood_score = mood.get("mood_score_avg")
            value_str = ""
            if mood_val:
                value_str = f"mood: {mood_val}"
            if mood_score:
                value_str += f" (avg score: {mood_score})"
            if value_str:
                value_str += " - "
            value_str += f"STALE from {mood_staleness}d ago ({mood_last_date})"
            staleness_warning_lines.append(f"- MOOD: {value_str}")

        # Energy staleness - include last known value
        energy_staleness = meta.get("energy_data_staleness_days")
        energy_last_date = meta.get("energy_last_data_date")
        if energy_staleness == 999:
            staleness_warning_lines.append(
                "- ENERGY: NO DATA EVER — do NOT report today's energy burn / active-minutes as a fact."
            )
        elif energy_staleness is not None and energy_staleness >= 1:
            energy = health_signals.get("energy") or {}
            cardio = health_signals.get("cardio_stress", {}) or {}
            period = health_signals.get("period_aggregates", {}) or {}
            energy_pct = energy.get("week_vs_prior_month_pct")
            if energy_pct is None:
                energy_pct = cardio.get("energy_week_vs_prior_month_pct")
            active_today = energy.get("active_minutes_today")
            if active_today is None:
                active_today = ((period.get("active_minutes") or {}).get("today"))
            calories_today = energy.get("calories_burned_today")
            if calories_today is None:
                calories_today = ((period.get("calories_burned") or {}).get("today"))
            workout_today = energy.get("workout_duration_min")
            if workout_today is None:
                workout_today = ((period.get("total_workout_min") or {}).get("today"))
            bits = []
            if isinstance(active_today, (int, float)):
                bits.append(f"active_minutes: {round(float(active_today), 1)}")
            if isinstance(calories_today, (int, float)):
                bits.append(f"calories_burned: {round(float(calories_today), 1)}")
            if isinstance(workout_today, (int, float)):
                bits.append(f"workout_min: {round(float(workout_today), 1)}")
            if isinstance(energy_pct, (int, float)):
                bits.append(f"week_vs_prior_month: {round(float(energy_pct), 1)}%")
            overall = energy.get("overall") or {}
            if isinstance(overall, dict) and overall.get("status"):
                bits.append(f"overall: {overall.get('status')}")
            value_str = ", ".join(bits) if bits else "last energy payload exists"
            staleness_warning_lines.append(
                f"- ENERGY: {value_str} from {energy_last_date} ({energy_staleness} day(s) ago) - ⚠️ NOT today's energy data. Prefer energy.signals / summary / overall if present — do not treat flat fields as fresh."
            )

        # Build warning block if any staleness exists
        staleness_warning = ""
        if staleness_warning_lines:
            staleness_warning = f"""
DATA STALENESS ALERT - READ CAREFULLY:
The following metrics have OLD data (not from today):
{chr(10).join(staleness_warning_lines)}

RULES:
- Do NOT say "last night", "hôm qua", "hôm nay" for stale data.
- Do NOT present stale data as if it's current/recent.
- For ENERGY stale data: never claim "today's burn/active minutes" as fresh.
- Use hedging language for stale data: "appears to have", "seems to be", "may be".
- For stale data older than 2 days: express clear uncertainty, acknowledge data is outdated.
- You MAY reference stale data but must clearly indicate it is historical, not current.
"""

        # Remove None values to reduce noise
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items() if v is not None and v != ""}
            elif isinstance(obj, list):
                return [clean(item) for item in obj if item is not None]
            return obj

        narrative_ctx = data.get("narrative_context")
        if narrative_ctx and isinstance(narrative_ctx, dict):
            cleaned_data["narrative_context"] = {
                k: str(v) for k, v in narrative_ctx.items() if v
            }

        cleaned_data = clean(data)

        # Translate internal time_phase enum to natural language label so LLM
        # does NOT echo the raw value (e.g. "wind_down") to the user.
        _TIME_PHASE_LABEL = {
            "morning": "buổi sáng (sáng sớm)",
            "afternoon": "buổi chiều",
            "after_work": "sau giờ làm việc",
            "evening": "buổi tối (đầu giờ)",
            "wind_down": "đang chuẩn bị nghỉ (trước giờ ngủ)",
            "bedtime": "đang là giờ chuẩn bị đi ngủ",
            "late_night": "đã khuya (qua giữa đêm)",
        }
        try:
            meta_block = cleaned_data.get("meta") or {}
            tp = meta_block.get("time_phase")
            if tp in _TIME_PHASE_LABEL:
                meta_block["time_phase_label"] = _TIME_PHASE_LABEL[tp]
                cleaned_data["meta"] = meta_block
        except Exception:
            pass

        json_str = json.dumps(cleaned_data, ensure_ascii=False)

        schedule_banner = _build_schedule_facts_banner(cleaned_data)
        return staleness_warning + schedule_banner + json_str
    except Exception:
        return context_json


def build_health_qa_user_prompt(
    context_json: str,
    narrative: str,
    group: Dict[str, Any],
    language: str,
) -> str:
    """Build user prompt for health Sylo Q&A → title + insight story."""
    focus = group.get("focus") or "overview"
    questions_text = "\n".join(
        f"{i+1}. [{q['id']}] {q['text']}"
        + (
            f"\n   → Read signals: {', '.join(q['signal_keys'])}"
            if q.get("signal_keys")
            else ""
        )
        for i, q in enumerate(group["questions"])
    )
    overview_guide = """
HEALTH EVIDENCE MAP (post-processor — only use what is present):
- health_signals.sleep | heart_rate | steps | energy | mood
  → summary / signals.*.status|metrics|comparison|trend / anomalies / overall
- health_signals.health_score
  → summary / signals[] (HEALTH_SCORE_CRITICAL|DROP|ANOMALY|…) /
    aggregate (HEALTH_SCORE_DECLINE|IMPROVEMENT) / overall
- health_signals.period_aggregates
  → today / week_avg / month_avg / baseline_avg + deltas per metric
- health_signals.context_signals
  → sleep_window / weekend / work_hours / recent_workout
  → status + priority + evidence (constraints for Layer 5 / framing)
- USER NARRATIVE + HEALTH METRICS PANEL = coverage checklist across ALL focuses
- Prefer personal baseline / 3d / 7d / month when present
- Mood: descriptive only; never diagnose; never expose internal scores

COVERAGE RULE (MANDATORY before STEP 2):
- Mentally scan SLEEP, HEART_RATE, STEPS, ENERGY, MOOD, HEALTH_SCORE, CONTEXT.
- Candidate stories may come from ANY focus with real evidence.
- Do NOT default to steps/pace/goal just because those fields are dense.
- `partial_day` on steps is normal midday noise — not automatically the lead story.
- Prefer recovery / sleep / HR / health_score decline when severity is higher
  than routine activity pace.

CONTEXT RULES:
- Higher-priority active context overrides conflicting lower-priority framing.
- sleep_window active → prioritize sleep/recovery; no primary exercise push.
- weekend active → avoid unnecessary exercise pressure.
- recent_workout active → read recovery/activity through that lens.
- If SCHEDULE_IRRATIONALITY / events after bedtime_goal appear in schedule facts:
  do NOT push long walks or late exercise as the main health story — protect sleep first.
- Context describes state only — you still choose the single best health story.

**SLEEP TIME TRAVEL (CRITICAL):**
- `sleep.bedtime` / `LAST_NIGHT_BEDTIME` / last-night duration/quality = the night that ALREADY ended
  (user already slept; woke this morning). Closed history.
- `BEDTIME_GOALS.bedtime_start` / `TONIGHT bedtime goal` = upcoming target for TONIGHT.
- Today's calendar (`current_event` / `previous_event` / "cf với Phương") happens on the calendar day
  AFTER last night's sleep — it CANNOT have caused last night's bedtime.
- FORBIDDEN: "Giờ ngủ trễ vì sự kiện X vừa xong" when X is today's event and bedtime 22:00 is LAST NIGHT.
- OK: if tonight's wind-down is crowded by a late event NOW, talk about protecting TONIGHT's bedtime goal —
  without claiming it caused last night's 22:00 bedtime.
- SAI: title "Giờ ngủ trễ vì cf với Phương vừa xong" + body tying last-night 22:00 to today's event.
- ĐÚNG: either (a) last-night story without today's calendar as cause, or (b) tonight wind-down vs bedtime_goal,
  without rewriting last night's session as caused by today.

SYLO FLOW (internal — then write title + insight):
STEP 1 Collect evidence from ALL focuses → STEP 2 One story → STEP 3 Layers 1–3 + 5
(SKIP Layer 4 cross-domain) → STEP 4 home-doctor + companion voice → STEP 5 JSON

insight paragraph flow (no labels):
Hook → Evidence → Interpretation → Future implication → One specific action
Length: 80–180 words. Concrete numbers. Personal baseline when present.
"""
    return f"""\
You are Sylo's Insight Generation Engine for HEALTH.
Do NOT summarize data dumps. Discover ONE meaningful story with the highest
expected value for the user today — after reviewing ALL health metrics.

FOCUS GROUP: {focus}
{group.get("description", "")}
Health-only — SKIP cross-domain Layer 4. Do NOT force productivity/finance signals.
{overview_guide}
CONTEXT (user's current day — evidence sources):
{_serialize_health_context_block(context_json)}
{_goals_block_from_context(context_json)}

USER NARRATIVE (multi-aspect digest from processors):
{narrative}

QUESTION:
{questions_text}

Return JSON:
{{"insights": [{{
  "question_id": "...",
  "title": "...",
  "reasoning": {{
    "current_context": "...",
    "trend": "...",
    "personal_context": "...",
    "cross_domain": "",
    "next_action_reason": "..."
  }},
  "insight": "...",
  "current_state": "...",
  "evidence": "...",
  "cause": "...",
  "action": "...",
  "expected_outcome": "...",
  "action_family": "...",
  "tone": "...",
  "confidence": 0.7
}}]}}

Field mapping:
- title = curiosity headline (home-doctor + companion voice)
- reasoning = Layers 1–3 + 5 notes; cross_domain MUST be ""
- insight = final paragraph 80–180 words (Hook→Evidence→Interpretation→Future→Action)
- current_state = MUST equal insight (pipeline)
- evidence = 1–3 concrete numbers used
- action = the ONE next action (also woven into insight)
- expected_outcome = brief hedged benefit

Rules:
- Exactly {len(group['questions'])} insight object(s).
- Never invent data / trends / medical conclusions.
- Never write section labels inside title or insight.
- Confidence lower if data sparse/stale.
- reasoning.current_context SHOULD briefly note which focuses were considered
  (e.g. sleep/HR/steps/energy/mood/score) even if the story leads with one.
"""

def build_productivity_qa_user_prompt(
    context_json: str,
    narrative: str,
    group: Dict[str, Any],
    language: str,
) -> str:
    """Build user prompt for productivity Sylo Q&A → title + insight story."""
    focus = group.get("focus") or "overview"
    questions_text = "\n".join(
        f"{i+1}. [{q['id']}] {q['text']}"
        + (
            f"\n   → Read signals: {', '.join(q['signal_keys'])}"
            if q.get("signal_keys")
            else ""
        )
        for i, q in enumerate(group["questions"])
    )
    goals_block = _goals_block_from_context(context_json)
    overview_guide = """
PRODUCTIVITY EVIDENCE MAP — CALENDAR FIRST (only use what is present):
1) TODAY's calendar (read first):
- calendar_intelligence.next_event / upcoming_events / current_event (title, category, duration)
- free_windows_today (if capped_by=active_hours_end, duration is NOT
  "time before the next appointment" when that appointment is later)
- schedule denseness / meeting load / buffers from productivity_signals when present
- USER NARRATIVE productivity / schedule section (today's events)

2) Secondary (only after calendar picture is clear):
- productivity_signals.summary / signals.* / anomalies / overall
- reminder / task completion / focus metrics as support — do not lead with these
  if today's calendar already has a clear story

SCHEDULE RULE (critical):
To say "before appointment X", use next_event duration for X — never free_window duration.
In user prose: <60m keep minutes; >=60m write hours + minutes (never "195 phút").
Prefer a today-calendar story over a pure focus-score / task% story when both exist.
If SCHEDULE_IRRATIONALITY is present (events after bedtime_goal): that is PRIORITY #1 —
do NOT lead with free-window walking / steps / focus tips.

SYLO FLOW:
STEP 1 Collect evidence (calendar today → then rest) → STEP 2 One story →
STEP 3 Layers 1–3 + 5 (SKIP Layer 4 cross-domain) →
STEP 4 home-doctor + companion voice → STEP 5 JSON

insight paragraph flow (no labels):
Hook → Evidence → Interpretation → Future implication → One specific action
Length: 80–180 words. Concrete numbers. Hook/evidence should start from today's schedule when present.
"""
    return f"""\
You are Sylo's Insight Generation Engine for PRODUCTIVITY.
Do NOT summarize data dumps. Discover ONE meaningful productivity story with
the highest expected value for the user today.

FOCUS GROUP: {focus}
{group.get("description", "")}
Productivity-only — SKIP cross-domain Layer 4.
{overview_guide}
If bedtime_goals / work_hours_goals are present, weigh late or dense schedule
against them as schedule boundaries.

CONTEXT (user's current day — evidence sources):
{_serialize_context_block(context_json)}
{goals_block}

USER NARRATIVE (productivity digest from processors):
{narrative}

QUESTION:
{questions_text}

Return JSON:
{{"insights": [{{
  "question_id": "...",
  "title": "...",
  "reasoning": {{
    "current_context": "...",
    "trend": "...",
    "personal_context": "...",
    "cross_domain": "",
    "next_action_reason": "..."
  }},
  "insight": "...",
  "current_state": "...",
  "evidence": "...",
  "cause": "...",
  "action": "...",
  "expected_outcome": "...",
  "action_family": "...",
  "tone": "...",
  "confidence": 0.7
}}]}}

Field mapping:
- title = curiosity headline (home-doctor + companion voice)
- reasoning = Layers 1–3 + 5 notes; cross_domain MUST be ""
- insight = final paragraph 80–180 words (Hook→Evidence→Interpretation→Future→Action)
- current_state = MUST equal insight (pipeline)
- evidence = 1–3 concrete numbers used
- action = the ONE next action (also woven into insight)
- expected_outcome = brief hedged benefit

Rules:
- Exactly {len(group['questions'])} insight object(s).
- Never invent data / trends.
- Never write section labels inside title or insight.
- Confidence lower if data sparse/stale.
"""


def build_overall_qa_user_prompt(
    context_json: str,
    narrative: str,
    group: Dict[str, Any],
    language: str,
) -> str:
    """Build user prompt for overall Sylo Q&A → title + insight story."""
    focus = group.get("focus") or "overview"
    questions_text = "\n".join(
        f"{i+1}. [{q['id']}] {q['text']}"
        + (
            f"\n   → Read signals: {', '.join(q['signal_keys'])}"
            if q.get("signal_keys")
            else ""
        )
        for i, q in enumerate(group["questions"])
    )
    overview_guide = """
OVERALL EVIDENCE MAP (post-processor — only use what is present):
- SCHEDULE FACTS / BEDTIME_GOALS / calendar_intelligence (check FIRST)
- health_signals (summary / signals / anomalies / overall)
- productivity_signals (summary / signals / anomalies / overall)
- finance_signals
- balance_snapshot (today + 7d)
- goal_signals
- behavioral_patterns + historical_trends
- USER NARRATIVE

PRIORITY: If SCHEDULE_IRRATIONALITY exists (events after bedtime_goal, e.g. work at 22:00
vs bedtime 21:00), that is the story — not free-window steps tips.

SYLO FLOW:
STEP 1 Collect evidence (irrationality first) → STEP 2 One story → STEP 3 Layers 1–5
→ STEP 4 home-doctor + companion voice → STEP 5 JSON

insight paragraph flow (no labels):
Hook → Evidence → Interpretation → Future implication → One specific action
Length: 80–180 words. Concrete numbers. Baseline/trend when present.
"""
    goals_block = _goals_block_from_context(context_json)
    return f"""\
You are Sylo's Insight Generation Engine for HOME / OVERALL.
Do NOT summarize data dumps. Discover ONE meaningful cross-domain story with
the highest expected value for the user today.

FOCUS GROUP: {focus}
{group.get("description", "")}
{overview_guide}

CONTEXT (user's current day — evidence sources):
{_serialize_context_block(context_json)}
{goals_block}

USER NARRATIVE (cross-domain digest from processors):
{narrative}

QUESTION:
{questions_text}

Return JSON:
{{"insights": [{{
  "question_id": "...",
  "title": "...",
  "reasoning": {{
    "current_context": "...",
    "trend": "...",
    "personal_context": "...",
    "cross_domain": "...",
    "next_action_reason": "..."
  }},
  "insight": "...",
  "current_state": "...",
  "evidence": "...",
  "cause": "...",
  "action": "...",
  "expected_outcome": "...",
  "action_family": "...",
  "tone": "...",
  "confidence": 0.7
}}]}}

Field mapping:
- title = curiosity headline (home-doctor + companion voice)
- reasoning = Layers 1–5 notes (cross_domain is required when evidence supports)
- insight = final paragraph 80–180 words (Hook→Evidence→Interpretation→Future→Action)
- current_state = MUST equal insight (pipeline)
- evidence = 1–3 concrete data points used
- action = the ONE next action (also woven into insight)
- expected_outcome = brief hedged benefit

Rules:
- Exactly {len(group['questions'])} insight object(s).
- Never invent data / trends / goals / events.
- Never write section labels inside title or insight.
- Confidence lower if data sparse/stale.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Group registries — used by the service to iterate over groups
# ─────────────────────────────────────────────────────────────────────────────

HEALTH_GROUPS = [
    _HEALTH_GROUP_OVERVIEW,
]

PRODUCTIVITY_GROUPS = [_PRODUCTIVITY_GROUP_OVERVIEW]
OVERALL_GROUPS = [
    _OVERALL_GROUP_OVERVIEW,
]

DOMAIN_GROUP_REGISTRY: Dict[str, List[Dict[str, Any]]] = {
    "health": HEALTH_GROUPS,
    "productivity": PRODUCTIVITY_GROUPS,
    "overall": OVERALL_GROUPS,
}

_SYSTEM_PROMPT_BUILDERS: Dict[str, Any] = {
    "health": _build_health_system_prompt,
    "productivity": _build_productivity_system_prompt,
    "overall": _build_overall_system_prompt,
}

_USER_PROMPT_BUILDERS: Dict[str, Any] = {
    "health": build_health_qa_user_prompt,
    "productivity": build_productivity_qa_user_prompt,
    "overall": build_overall_qa_user_prompt,
}


def resolve_overview(domain: str) -> Optional[tuple[Dict[str, Any], Dict[str, Any]]]:
    """Fixed Sylo overview group + question (one per domain — no random)."""
    groups = DOMAIN_GROUP_REGISTRY.get(domain) or []
    if not groups:
        return None
    group = groups[0]
    questions = group.get("questions") or []
    if not questions:
        return None
    return group, questions[0]


def build_sylo_qa_prompts(
    domain: str,
    *,
    context_json: str,
    narrative: str,
    language: str,
    time_phase: str,
    active_hours_end_time: str | None = None,
) -> Optional[Dict[str, Any]]:
    """Assemble system + user prompts for the fixed overview question.

    Returns ``{system, user, group, question}`` or None if domain has no overview.
    """
    resolved = resolve_overview(domain)
    if not resolved:
        return None
    group, question = resolved
    slim_group = {**group, "questions": [question]}
    system_fn = _SYSTEM_PROMPT_BUILDERS.get(domain)
    user_fn = _USER_PROMPT_BUILDERS.get(domain)
    if not system_fn or not user_fn:
        return None
    return {
        "system": system_fn(time_phase, language, active_hours_end_time),
        "user": user_fn(
            context_json=context_json,
            narrative=narrative,
            group=slim_group,
            language=language,
        ),
        "group": group,
        "question": question,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parse LLM output
# ─────────────────────────────────────────────────────────────────────────────

def parse_qa_response(content: str) -> List[Dict[str, Any]]:
    """
    Parse LLM JSON output from a Q&A call.
    Returns a list of insight dicts with ``prose`` (title+insight) already composed.
    """
    from agents.llm_helper import parse_json_object_from_llm_text

    raw = (content or "").strip()
    parsed = parse_json_object_from_llm_text(raw)

    if isinstance(parsed, dict) and "insights" in parsed:
        insights = parsed["insights"]
    elif isinstance(parsed, list):
        insights = parsed
    else:
        return []

    if not isinstance(insights, list):
        return []

    # Sylo: keep insight/current_state in sync; fold compose into parse
    normalized: List[Dict[str, Any]] = []
    for item in insights:
        if not isinstance(item, dict):
            continue
        insight = (item.get("insight") or "").strip()
        current = (item.get("current_state") or "").strip()
        if insight and not current:
            item = {**item, "current_state": insight}
        elif current and not insight:
            item = {**item, "insight": current}
        item = _sanitize_insight_fields(item)
        item["prose"] = compose_sylo_prose(item)
        normalized.append(item)
    return normalized