from __future__ import annotations

import re
from typing import Any, Dict, Optional

_NUM = r"\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?"

_UNIT_PATTERNS = [
    # English
    (re.compile(rf"({_NUM})\s*(?:bpm|BPM)"), "bpm"),
    (re.compile(rf"({_NUM})\s*(?:steps|step)\b"), "steps"),
    (re.compile(rf"({_NUM})\s*(?:hours|hrs|h)\b"), "hours"),
    (re.compile(rf"({_NUM})\s*(?:minutes|min|mins|m)\b"), "minutes"),
    (re.compile(rf"({_NUM})\s*%"), "percent"),
    # Vietnamese
    (re.compile(rf"({_NUM})\s*(?:nhịp|phút|giờ|bước|ngày)\b"), "value_vi"),
    (re.compile(rf"({_NUM})\s*%/"), "percent"),
]

_ALLOWED_GENERIC_NUMBERS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 24, 30, 60, 100, 365}


def _collect_real_values(health_params: Dict[str, Any]) -> Dict[str, set]:
    """Bucket every non-None numeric value in health_params by unit."""
    bpm: set = set()
    steps: set = set()
    hours: set = set()
    minutes: set = set()
    percent: set = set()
    generic: set = set()

    for key, val in (health_params or {}).items():
        if val is None:
            continue
        if isinstance(val, bool):
            continue
        if not isinstance(val, (int, float)):
            continue
        v = int(val) if isinstance(val, float) and val.is_integer() else val
        if v == 0:
            # Treat 0 as missing — see ZERO/NULL IS NOT DATA rule.
            continue
        kl = key.lower()
        if "heart" in kl or "hr" in kl or "hrv" in kl:
            bpm.add(int(v))
        elif "step" in kl:
            steps.add(int(v))
        elif "sleep" in kl or "hour" in kl:
            hours.add(int(v))
        elif "minute" in kl or "duration" in kl:
            minutes.add(int(v))
        elif "score" in kl or "percent" in kl or "pct" in kl:
            percent.add(int(v))
        else:
            generic.add(int(v))
    return {
        "bpm": bpm,
        "steps": steps,
        "hours": hours,
        "minutes": minutes,
        "percent": percent,
        "generic": generic,
    }


_DURATION_VI_PATTERNS = [
    re.compile(rf"({_NUM})\s*ngày", re.IGNORECASE),
    re.compile(rf"({_NUM})\s*ngày\s*(?:qua|liên tiếp|liên tục)", re.IGNORECASE),
    re.compile(rf"(?:từ|kéo dài|suốt)\s*({_NUM})\s*ngày", re.IGNORECASE),
]

_DURATION_EN_PATTERNS = [
    re.compile(rf"({_NUM})\s*days?\s*(?:in\s*a\s*row|consecutive|straight)?", re.IGNORECASE),
    re.compile(rf"(?:for|over|over\s*the\s*past)\s*({_NUM})\s*days?", re.IGNORECASE),
    re.compile(rf"({_NUM})-day", re.IGNORECASE),
]


def _check_pattern_duration_violations(
    prose: str,
    productivity_params: Optional[Dict[str, Any]] = None,
) -> list:
    if not prose or not productivity_params:
        return []

    violations: list = []
    chronic_overload_days = productivity_params.get("chronic_overload_days", 0) or 0
    meeting_overload_days = productivity_params.get("meeting_overload_days", 0) or 0
    work_hours_overload_days = productivity_params.get("work_hours_overload_days_7d", 0) or 0
    max_overload_days = max(chronic_overload_days, meeting_overload_days, work_hours_overload_days)

    duration_phrases = [
        "kéo dài", "liên tiếp", "liên tục", "suốt", "trong suốt",
        "in a row", "consecutive", "back.to.back", "straight", "for days",
        "over the past", "over the last", "persisting", "chronic",
        "days in a row", "days straight",
    ]

    all_patterns = _DURATION_VI_PATTERNS + _DURATION_EN_PATTERNS

    for pattern in all_patterns:
        for match in pattern.finditer(prose):
            claimed_days_raw = match.group(1)
            try:
                claimed_days = int(claimed_days_raw)
            except (ValueError, IndexError):
                continue
            if claimed_days <= 4:
                continue
            start = max(0, match.start() - 30)
            end = min(len(prose), match.end() + 30)
            snippet = prose[start:end].strip()
            is_overload_claim = any(
                phrase in prose[max(0, match.start() - 100):match.end() + 100].lower()
                for phrase in duration_phrases
            )
            if is_overload_claim and claimed_days > max_overload_days:
                violations.append({
                    "claimed_days": claimed_days,
                    "actual_max": max_overload_days,
                    "snippet": snippet,
                    "type": "pattern_duration_exceeded",
                })

    return violations


_CALENDAR_EMPTY_VI = ["lịch không có cuộc họp", "lịch trống", "không có cuộc học", "mật độ lịch bằng 0", "lịch hôm nay không", "không có lịch họp"]
_CALENDAR_EMPTY_EN = ["no meetings", "empty calendar", "no scheduled", "calendar is empty", "no meetings today"]
_CALENDAR_BUSY_VI = ["lịch dày đặc", "dày đặc", "quá tải", "quá nhiều việc", "bận rộn", "lịch họp nhiều"]
_CALENDAR_BUSY_EN = ["busy schedule", "heavy schedule", "dense calendar", "back-to-back", "overloaded", "too many meetings"]


def _check_calendar_contradiction(
    prose: str,
    calendar_intelligence: Optional[Dict[str, Any]] = None,
) -> list:
    if not prose or not calendar_intelligence:
        return []

    violations: list = []
    meeting_minutes = calendar_intelligence.get("meeting_minutes_today") or 0
    cognitive_load = calendar_intelligence.get("cognitive_load_so_far") or "low"
    is_empty = meeting_minutes == 0 or cognitive_load in ("low", "none")

    if not is_empty:
        return []

    prose_lower = prose.lower()
    has_empty_claim = any(p in prose_lower for p in _CALENDAR_EMPTY_VI + _CALENDAR_EMPTY_EN)
    has_busy_claim = any(p in prose_lower for p in _CALENDAR_BUSY_VI + _CALENDAR_BUSY_EN)

    if has_empty_claim and has_busy_claim:
        empty_snippets = [p for p in _CALENDAR_EMPTY_VI + _CALENDAR_EMPTY_EN if p in prose_lower]
        busy_snippets = [p for p in _CALENDAR_BUSY_VI + _CALENDAR_BUSY_EN if p in prose_lower]
        violations.append({
            "type": "calendar_contradiction",
            "empty_claims": empty_snippets,
            "busy_claims": busy_snippets,
            "actual_meetings": meeting_minutes,
            "actual_load": cognitive_load,
            "snippet": prose[:200],
        })

    return violations


def validate_insight_against_health_params(
    prose: str,
    health_params: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    language: Optional[str] = None,
    productivity_params: Optional[Dict[str, Any]] = None,
    calendar_intelligence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not prose or not prose.strip():
        return {
            "ok": True,
            "suspicious": [],
            "date_violations": [],
            "language_violations": [],
            "pattern_duration_violations": [],
            "contradiction_violations": [],
            "advice": "",
        }

    real = _collect_real_values(health_params or {})
    suspicious: list = []

    for pattern, unit in _UNIT_PATTERNS:
        for match in pattern.finditer(prose):
            raw_n = match.group(1)
            cleaned = raw_n.replace(",", "").replace(".", "")
            try:
                n = int(cleaned)
            except (IndexError, ValueError):
                continue
            if n in _ALLOWED_GENERIC_NUMBERS:
                continue
            bucket = real.get(unit, real["generic"])
            if n in bucket:
                continue
            if unit == "bpm" and n in real["bpm"]:
                continue
            if unit == "steps" and n in real["steps"]:
                continue
            if unit == "percent" and n in real["percent"]:
                continue
            if unit == "hours" and n in real["hours"]:
                continue
            if unit == "minutes" and n in real["minutes"]:
                continue
            if unit == "value_vi" and any(n in real[k] for k in real):
                continue
            snippet_start = max(0, match.start() - 25)
            snippet_end = min(len(prose), match.end() + 25)
            suspicious.append(
                {
                    "value": n,
                    "raw": raw_n,
                    "unit": unit,
                    "snippet": prose[snippet_start:snippet_end],
                }
            )

    date_violations = _check_stale_as_today(prose, meta, health_params or {})
    pattern_duration_violations = _check_pattern_duration_violations(
        prose, productivity_params
    )
    contradiction_violations = _check_calendar_contradiction(prose, calendar_intelligence)

    language_violations: list = []
    consistency_violations: list = []
    if language and language.lower().startswith("vi"):
        # Legacy mood-only check (kept for backward compatibility)
        language_violations = _check_english_mood_labels(prose)
        # New comprehensive check: any English leak in Vietnamese prose
        consistency_violations = _check_language_consistency(prose, language)

    advice = ""
    if suspicious:
        advice = (
            "The prose mentions numbers that are NOT in the source data. "
            "Treat 0 / null health values as missing and DO NOT fabricate a "
            "real measurement. Common culprits: resting HR, latest HR, HRV, "
            "steps_today, sleep_lastnight, sleep_score."
        )
    if date_violations:
        advice += (
            " The prose talks about TODAY but the underlying field is stale "
            "(last data is from an earlier date) or comes from a weekly "
            "rolling average. Do NOT quote historical numbers as today's "
            "reading. Tell the user explicitly that you have no fresh data "
            "for today."
        )
    if pattern_duration_violations:
        for v in pattern_duration_violations:
            advice += (
                f" CLAIMED {v['claimed_days']} DAYS but actual max is {v['actual_max']} days. "
                f"Snippet: '{v['snippet'][:80]}'. "
                "NEVER claim a duration (e.g. '7 days in a row', 'kéo dài 7 ngày') "
                "that exceeds the actual data count. Use the actual number from data."
            )
    if language_violations:
        advice += (
            " The prose contains English mood labels (amazing/happy/sad/...) "
            "but the insight is written in Vietnamese. Translate the mood to "
            "Vietnamese before emitting prose: amazing=tuyệt vời, happy=vui, "
            "sad=buồn, okay=bình thường, terrible=tệ, neutral=bình thường, "
            "stressed=căng thẳng, calm=bình yên, anxious=lo lắng, "
            "tired=mệt, energetic=tràn năng lượng."
        )
    if contradiction_violations:
        for v in contradiction_violations:
            advice += (
                f" SELF-CONTRADICTION: prose claims calendar is empty/busy simultaneously. "
                f"Empty claims: {v['empty_claims']}, Busy claims: {v['busy_claims']}. "
                f"Actual meetings={v['actual_meetings']}, load={v['actual_load']}. "
                "Do NOT claim a busy/heavy schedule when calendar data shows 0 meetings."
            )
    if consistency_violations:
        sample = [v["token"] for v in consistency_violations[:8]]
        advice += (
            " LANGUAGE CONSISTENCY VIOLATION: the prose contains English "
            f"leak(s) inside Vietnamese text. Forbidden tokens: {sample}. "
            "REWRITE the entire prose in pure Vietnamese. Do NOT embed English "
            "field/parameter names, mood labels (amazing/happy/sad/okay/...), "
            "or technical jargon (focus_blocks, resting_heart_rate, "
            "meeting_completion_rate, cognitive_load, bandwidth, ...). "
            "Use natural Vietnamese phrasing for everything except calendar "
            "event titles quoted in double quotes. Numbers and units "
            "(giờ/phút/bước/lần) are allowed."
        )

    return {
        "ok": not (
            suspicious
            or date_violations
            or language_violations
            or consistency_violations
            or pattern_duration_violations
            or contradiction_violations
        ),
        "suspicious": suspicious,
        "date_violations": date_violations,
        "language_violations": language_violations,
        "consistency_violations": consistency_violations,
        "pattern_duration_violations": pattern_duration_violations,
        "contradiction_violations": contradiction_violations,
        "advice": advice,
    }

_TODAY_FRAMING_PHRASES_VI = (
    "hôm nay",
    "đêm qua",
    "đêm hôm qua",
    "sáng nay",
    "tối nay",
    "ngày hôm nay",
    "hôm qua",
)
_TODAY_FRAMING_PHRASES_EN = (
    "today",
    "tonight",
    "last night",
    "this morning",
    "this evening",
    "yesterday",
    "right now",
    "currently",
)


def _check_stale_as_today(
    prose: str,
    meta: Optional[Dict[str, Any]],
    health_params: Dict[str, Any],
) -> list:
    """Detect ``"<today-claim> ... <number> <unit>"`` where the source field
    is stale or aggregated.

    Returns a list of violation dicts (empty when ``meta`` is missing or the
    prose doesn't frame any number as today).
    """
    if not prose or meta is None:
        return []

    prose_lower = prose.lower()
    has_today_framing = any(
        phrase in prose_lower
        for phrase in _TODAY_FRAMING_PHRASES_VI + _TODAY_FRAMING_PHRASES_EN
    )
    if not has_today_framing:
        return []

    violations: list = []
    sleep_stale_days = meta.get("sleep_data_staleness_days")
    sleep_last_date = meta.get("sleep_last_data_date")
    sleep_lastnight = health_params.get("sleep_lastnight")
    if (
        sleep_stale_days is not None
        and sleep_stale_days >= 1
        and sleep_lastnight is None
    ):
        snippet = _find_today_number_for_field(
            prose, units=("giờ", "hours", "h", "hrs")
        )
        if snippet:
            violations.append(
                {
                    "field": "sleep_lastnight",
                    "reason": "framed_as_today_but_stale",
                    "staleness_days": sleep_stale_days,
                    "last_data_date": sleep_last_date,
                    "snippet": snippet,
                }
            )

    if sleep_lastnight is None:
        avg_asleep = health_params.get("average_total_sleep_hours")
        if avg_asleep is not None and avg_asleep > 0:
            snippet = _find_today_number_for_field(
                prose, units=("giờ", "hours", "h", "hrs")
            )
            if snippet and any(word in snippet for word in ("8", "8.0", str(int(avg_asleep)))):
                violations.append(
                    {
                        "field": "average_total_sleep_hours",
                        "reason": "historical_avg_framed_as_today",
                        "actual_date": sleep_last_date or "unknown (no data)",
                        "snippet": snippet,
                    }
                )

    # ── steps_today stale? ──────────────────────────────────────────────────
    steps_stale_days = meta.get("steps_data_staleness_days")
    steps_last_date = meta.get("steps_last_data_date")
    steps_today = health_params.get("steps_today")
    if (
        steps_stale_days is not None
        and steps_stale_days >= 1
        and steps_today is None
    ):
        snippet = _find_today_number_for_field(
            prose, units=("bước", "steps", "step")
        )
        if snippet:
            violations.append(
                {
                    "field": "steps_today",
                    "reason": "framed_as_today_but_stale",
                    "staleness_days": steps_stale_days,
                    "last_data_date": steps_last_date,
                    "snippet": snippet,
                }
            )

    # ── resting_heart_rate stale? ──────────────────────────────────────
    hr_stale_days = meta.get("hr_data_staleness_days")
    hr_last_date = meta.get("hr_last_data_date")
    rhr = health_params.get("resting_heart_rate")
    if (
        hr_stale_days is not None
        and hr_stale_days >= 1
        and rhr is None
    ):
        snippet = _find_today_number_for_field(
            prose, units=("nhịp", "bpm", "BPM")
        )
        if snippet:
            violations.append(
                {
                    "field": "resting_heart_rate",
                    "reason": "framed_as_today_but_stale",
                    "staleness_days": hr_stale_days,
                    "last_data_date": hr_last_date,
                    "snippet": snippet,
                }
            )

    # ── mood stale? ─────────────────────────────────────────────────────
    mood_stale_days = meta.get("mood_data_staleness_days")
    mood_last_date = meta.get("mood_last_data_date")
    latest_mood = health_params.get("latest_mood")
    if (
        mood_stale_days is not None
        and mood_stale_days >= 1
        and not latest_mood
    ):
        snippet = _find_today_phrase_without_date(
            prose,
            keywords=("tâm trạng", "mood", "cảm xúc"),
        )
        if snippet:
            violations.append(
                {
                    "field": "latest_mood",
                    "reason": "framed_as_today_but_stale",
                    "staleness_days": mood_stale_days,
                    "last_data_date": mood_last_date,
                    "snippet": snippet,
                }
            )

    return violations


def _find_today_number_for_field(prose: str, units: tuple) -> Optional[str]:
    # Split into sentences (Vietnamese + English punctuation).
    sentences = re.split(r"[.!?\n]+", prose)
    today_signals = _TODAY_FRAMING_PHRASES_VI + _TODAY_FRAMING_PHRASES_EN
    units_alt = "|".join(re.escape(u) for u in units)
    number_with_unit = re.compile(
        rf"({_NUM})\s*(?:{units_alt})\b",
        re.IGNORECASE,
    )
    for sentence in sentences:
        sent_lower = sentence.lower()
        if not any(p in sent_lower for p in today_signals):
            continue
        if number_with_unit.search(sentence):
            return sentence.strip()[:120]
    return None


def _find_today_phrase_without_date(
    prose: str,
    keywords: tuple,
) -> Optional[str]:
    sentences = re.split(r"[.!?\n]+", prose)
    today_signals = _TODAY_FRAMING_PHRASES_VI + _TODAY_FRAMING_PHRASES_EN
    date_signal = re.compile(
        r"(?:ngày\s*\d|\d{{1,2}}/\d|\d{{4}}-\d{{2}}-\d{{2}}|from\s+\d{{4}})",
        re.IGNORECASE,
    )
    disclaimer_patterns = (
        r"chưa\s+c[oó]",
        r"không\s+c[oó]",
        r"no\s+data",
        r"no\s+mood",
        r"không\s+c[oó]\s+dữ\s*liệu",
        r"chưa\s+ghi\s*nhận",
        r"chưa\s+c[oó]\s+ghi\s*nhận",
    )
    for sentence in sentences:
        sent_lower = sentence.lower()
        if not any(p in sent_lower for p in today_signals):
            continue
        if not any(k in sent_lower for k in keywords):
            continue
        if date_signal.search(sentence):
            continue
        if any(
            re.search(pat, sent_lower) for pat in disclaimer_patterns
        ):
            continue
        return sentence.strip()[:120]
    return None

_EN_MOOD_LABELS = {
    "amazing", "happy", "good", "great", "sad", "okay", "ok", "terrible",
    "neutral", "stressed", "calm", "anxious", "tired", "energetic",
    "excited", "frustrated", "angry", "depressed", "relaxed", "focused",
    "overwhelmed", "grateful", "hopeful", "lonely", "content", "proud",
    "confident", "irritated", "worried", "stressed out", "meh",
}


_INTERNAL_EN_TOKENS = frozenset({
    # Health field names
    "resting_heart_rate", "latest_heart_rate", "baseline_resting_hr",
    "hrv_score", "walking_heart_rate_avg", "workout_avg_hr", "workout_max_hr",
    "workout_min_hr", "stress_high", "hrv", "rhr",
    "steps_today", "steps_goal", "remaining_steps", "steps_streak",
    "workout_duration", "workout_activity_type", "calories_burned",
    "active_minutes_today", "met_goal", "avg_daily_steps", "total_steps",
    "sleep_lastnight", "sleep_goal", "sleep_quality", "sleep_quality_score",
    "sleep_hours", "bedtime_streak", "deep_sleep_min", "rem_sleep_min",
    "light_sleep_min", "wake_time", "wind_down_buffer_mins",
    "weekly_health_progress", "daily_health_progress", "progress_percentage",
    "health_score", "healthscore", "health_score_trend",
    "total_active_energy", "total_resting_energy", "energy_current_month_avg",
    "energy_week_avg", "energy_previous_month_avg",
    "energy_week_vs_prior_month_pct",
    # Mood / activity level
    "mood_last_data_date", "current_mood_score", "baseline_mood_score",
    "mood_streak", "happy_streak", "sad_streak", "okay_streak", "amazing_streak",
    "activity_level", "activity_volume", "goal_achievement",
    "activity_consistency", "step_trend", "baseline_comparison", "activity_pattern",
    "primary_mood", "mood_first", "mood_sequence", "moodScore",
    # Productivity / calendar internals
    "focus_blocks", "focus_score", "focus_block", "focus level",
    "cognitive_load", "cognitive", "context_switching", "context switching",
    "meeting_completion_rate", "reminders_completion_rate", "events_completion_rate",
    "tasks_completion_rate", "task_completion_rate", "reminders_completion",
    "meetings_due_today", "reminders_due_today", "overdue_reminders_count",
    "meeting_fatigue", "meeting_minutes", "meeting_overload_days",
    "work_hours_overload", "chronic_overload_days", "chronic_meeting_overload",
    "productivity_score", "productivity_score_7d_avg",
    "wellness_score", "wellness_score_7d_avg", "p_task_completion_rate",
    "meeting_minutes_7d_avg", "events_completion_7d_avg",
    "events_completion_trend", "workload",
    # Calendar / time phase
    "calendar_intelligence", "current_event", "next_event", "upcoming_events",
    "nearest_reminder", "free_windows_today", "free_window",
    "free_slot_length", "minutes_until_next_event", "next_event_summary",
    "next_event_start_time", "time_to_bedtime", "time_to_next_event",
    "in_active_window", "bedtime_start", "bedtime_end", "bedtime_goal",
    "active_start", "active_end", "work_hours", "active_hours",
    "time_phase", "time_phase_label", "time_block", "current phase",
    "wind_down", "wind-down", "winddown", "wind_down_window",
    "bedtime_window", "recovery_window", "active_window", "morning_window",
    "evening_flexible", "evening_after_work", "operating_window",
    "schedule_focus_blocks", "schedule_irrationality",
    "context_signals", "context_signal",
    # Generic UX/corporate jargon that LLM echoes from internal data
    "bandwidth", "alignment", "optimize", "trajectory", "suboptimal",
    "recovery debt", "foundation", "inbox", "deep-work", "deep work",
    "Pomodoro", "pomodoro", "cognitive load", "cognitive",
    "nervous system", "cortisol", "amygdala",
    "data_staleness_days", "staleness_days",
})


def _split_sentences_keep_quoted(prose: str) -> list[tuple[str, bool]]:
    if not prose:
        return []
    out: list[tuple[str, bool]] = []
    buf: list[str] = []
    in_quote = False
    for ch in prose:
        if ch == '"':
            buf.append(ch)
            if buf:
                out.append(("".join(buf), in_quote))
                buf = []
            in_quote = not in_quote
        elif ch in ".!?\n" and not in_quote and buf:
            out.append(("".join(buf), in_quote))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append(("".join(buf), in_quote))
    return [(s.strip(), q) for s, q in out if s.strip()]


def _check_language_consistency(prose: str, language: str) -> list:
    violations: list = []
    if not prose:
        return violations

    is_vi = (language or "").lower().startswith("vi")
    is_en = (language or "").lower().startswith("en")
    if is_en:
        # English output is allowed to be entirely English; no violations here.
        return violations
    if not is_vi:
        # Other languages: skip this check (no dictionary yet).
        return violations

    sentences = _split_sentences_keep_quoted(prose)
    for sent, in_quote in sentences:
        if in_quote:
            continue
        sent_lower = sent.lower()

        for label in _EN_MOOD_LABELS:
            if re.search(rf"\b{re.escape(label)}\b", sent_lower):
                snippet_start = max(0, sent.find(label) - 25)
                snippet_end = min(len(sent), sent.find(label) + len(label) + 25)
                violations.append({
                    "kind": "english_mood",
                    "token": label,
                    "snippet": sent[max(0, snippet_start - 25):snippet_end] if snippet_start < 0 else sent[snippet_start:snippet_end],
                })

        # 2) Internal English tokens (field/parameter names)
        for tok in _INTERNAL_EN_TOKENS:
            # word boundary match (handles underscores, hyphens)
            pattern = re.compile(
                r"(?<![A-Za-z0-9])" + re.escape(tok) + r"(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
            m = pattern.search(sent)
            if m:
                snippet_start = max(0, m.start() - 25)
                snippet_end = min(len(sent), m.end() + 25)
                violations.append({
                    "kind": "english_token",
                    "token": m.group(0),
                    "snippet": sent[snippet_start:snippet_end],
                })

        vn_chars = sum(1 for c in sent if c in "ăâđêôơưĂÂĐÊÔƠƯáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữữýỳỷỹỵÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỮÝỲỶỸỴ")
        if vn_chars >= 5:
            # Look for English words (≥4 chars, only ASCII letters)
            en_words = re.findall(r"\b[A-Za-z]{4,}\b", sent)
            # Allowlist: common Vietnamese-friendly words that may appear in VN prose
            vn_allowlist = {
                "team", "sync", "standup", "retro", "demo", "review", "sprint",
                "feedback", "offline", "online", "app", "data", "score",
                "video", "audio", "ai", "ml", "ok", "plus", "pro",
                "team sync", "all hands", "one on one", "1on1",
            }
            bad_words = [w for w in en_words if w.lower() not in vn_allowlist]
            # Don't flag common abbreviations / brand names
            brand_allow = {"sylo", "sy"}
            bad_words = [w for w in bad_words if w.lower() not in brand_allow]
            if bad_words:
                violations.append({
                    "kind": "mixed_sentence",
                    "token": ", ".join(bad_words[:5]),
                    "snippet": sent[:120],
                })

    # De-dup by (kind, token, snippet[:40])
    seen: set = set()
    deduped: list = []
    for v in violations:
        key = (v["kind"], v["token"], v["snippet"][:40])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(v)
    return deduped


_EN_TOKEN_TO_VI = {
    "resting_heart_rate": "nhịp tim nghỉ",
    "latest_heart_rate": "nhịp tim gần nhất",
    "baseline_resting_hr": "nhịp tim nghỉ nền",
    "hrv_score": "biến thiên nhịp tim",
    "walking_heart_rate_avg": "nhịp tim đi bộ trung bình",
    "workout_avg_hr": "nhịp tim tập trung bình",
    "stress_high": "căng thẳng",
    "steps_today": "bước chân hôm nay",
    "steps_goal": "mục tiêu bước chân",
    "remaining_steps": "bước chân còn lại",
    "steps_streak": "chuỗi ngày đạt bước",
    "workout_duration": "thời gian tập",
    "active_minutes_today": "phút hoạt động hôm nay",
    "met_goal": "đạt mục tiêu",
    "total_steps": "tổng bước",
    "sleep_lastnight": "giấc ngủ đêm qua",
    "sleep_goal": "mục tiêu giấc ngủ",
    "sleep_quality": "chất lượng giấc ngủ",
    "sleep_quality_score": "điểm chất lượng giấc ngủ",
    "sleep_hours": "giờ ngủ",
    "bedtime_streak": "chuỗi ngày đúng giờ ngủ",
    "deep_sleep_min": "phút ngủ sâu",
    "rem_sleep_min": "phút ngủ REM",
    "light_sleep_min": "phút ngủ nông",
    "wake_time": "giờ thức",
    "wind_down_buffer_mins": "thời gian thư giãn trước ngủ",
    "weekly_health_progress": "tiến triển sức khỏe tuần",
    "daily_health_progress": "tiến triển sức khỏe hôm nay",
    "progress_percentage": "phần trăm tiến triển",
    "health_score": "điểm sức khỏe",
    "total_active_energy": "tổng năng lượng hoạt động",
    "energy_current_month_avg": "năng lượng trung bình tháng",
    "energy_week_avg": "năng lượng trung bình tuần",
    "energy_week_vs_prior_month_pct": "năng lượng tuần so với tháng trước",
    "current_mood_score": "điểm tâm trạng hiện tại",
    "baseline_mood_score": "điểm tâm trạng nền",
    "mood_streak": "chuỗi tâm trạng",
    "happy_streak": "chuỗi ngày vui",
    "sad_streak": "chuỗi ngày buồn",
    "okay_streak": "chuỗi ngày bình thường",
    "amazing_streak": "chuỗi ngày rất vui",
    "activity_level": "mức vận động",
    "activity_volume": "lượng vận động",
    "goal_achievement": "mức đạt mục tiêu",
    "activity_consistency": "tính đều đặn vận động",
    "step_trend": "xu hướng bước chân",
    "baseline_comparison": "so với mức nền",
    "primary_mood": "tâm trạng chính",
    "moodScore": "điểm tâm trạng",
    "focus_blocks": "khoảng tập trung",
    "focus_block": "khoảng tập trung",
    "focus_score": "điểm tập trung",
    "cognitive_load": "đầu óc đang bận",
    "context_switching": "bị ngắt giữa chừng",
    "meeting_completion_rate": "tỉ lệ hoàn thành sự kiện",
    "reminders_completion_rate": "tỉ lệ hoàn thành ghi nhớ",
    "events_completion_rate": "tỉ lệ hoàn thành sự kiện",
    "tasks_completion_rate": "tỉ lệ hoàn thành việc",
    "reminders_completion": "hoàn thành ghi nhớ",
    "meetings_due_today": "sự kiện hôm nay",
    "reminders_due_today": "ghi nhớ hôm nay",
    "overdue_reminders_count": "số ghi nhớ quá hạn",
    "meeting_fatigue": "mệt vì lịch họp",
    "meeting_minutes": "phút họp",
    "meeting_overload_days": "số ngày quá tải họp",
    "work_hours_overload": "quá tải giờ làm",
    "chronic_overload_days": "số ngày quá tải liên tiếp",
    "chronic_meeting_overload": "quá tải họp kéo dài",
    "productivity_score": "điểm năng suất",
    "productivity_score_7d_avg": "điểm năng suất trung bình 7 ngày",
    "wellness_score": "điểm cân bằng",
    "meeting_minutes_7d_avg": "phút họp trung bình 7 ngày",
    "events_completion_7d_avg": "tỉ lệ hoàn thành sự kiện trung bình 7 ngày",
    "events_completion_trend": "xu hướng hoàn thành sự kiện",
    "workload": "lượng việc",
    "calendar_intelligence": "phân tích lịch",
    "current_event": "sự kiện hiện tại",
    "next_event": "sự kiện tiếp theo",
    "upcoming_events": "sự kiện sắp tới",
    "nearest_reminder": "ghi nhớ gần nhất",
    "free_windows_today": "khoảng trống hôm nay",
    "free_window": "khoảng trống",
    "free_slot_length": "độ dài khoảng trống",
    "minutes_until_next_event": "phút đến sự kiện tiếp theo",
    "next_event_summary": "tóm tắt sự kiện tiếp theo",
    "next_event_start_time": "giờ bắt đầu sự kiện tiếp theo",
    "time_to_bedtime": "thời gian đến giờ ngủ",
    "time_to_next_event": "thời gian đến sự kiện tiếp theo",
    "in_active_window": "trong giờ hoạt động",
    "bedtime_start": "giờ bắt đầu ngủ",
    "bedtime_end": "giờ kết thúc ngủ",
    "bedtime_goal": "mục tiêu giờ ngủ",
    "active_start": "giờ bắt đầu hoạt động",
    "active_end": "giờ kết thúc hoạt động",
    "work_hours": "giờ làm việc",
    "active_hours": "giờ hoạt động",
    "time_phase": "thời điểm hiện tại",
    "time_phase_label": "nhãn thời điểm",
    "time_block": "khung giờ",
    "wind_down": "thư giãn trước ngủ",
    "wind-down": "thư giãn trước ngủ",
    "winddown": "thư giãn trước ngủ",
    "wind_down_window": "khoảng thư giãn trước ngủ",
    "bedtime_window": "khung giờ ngủ",
    "recovery_window": "khoảng nghỉ",
    "active_window": "khung giờ hoạt động",
    "morning_window": "đầu ngày",
    "evening_flexible": "buổi tối linh hoạt",
    "evening_after_work": "tối sau giờ làm",
    "operating_window": "khung giờ làm việc chính",
    "schedule_focus_blocks": "các khoảng trống lịch",
    "schedule_irrationality": "lịch bất hợp lý",
    "context_signals": "tín hiệu ngữ cảnh",
    "context_signal": "tín hiệu ngữ cảnh",
    "bandwidth": "đầu óc còn chỗ",
    "alignment": "ăn khớp",
    "optimize": "làm gọn hơn",
    "trajectory": "xu hướng",
    "suboptimal": "chưa ổn lắm",
    "recovery debt": "thiếu ngủ bù",
    "foundation": "nền tảng",
    "inbox": "hộp thư",
    "deep-work": "việc cần tập trung sâu",
    "deep work": "việc cần tập trung sâu",
    "pomodoro": "25 phút tập trung",
    "Pomodoro": "25 phút tập trung",
    "nervous system": "cơ thể",
    "cortisol": "căng thẳng",
    "amygdala": "cơ thể",
    "data_staleness_days": "số ngày dữ liệu cũ",
    "staleness_days": "số ngày dữ liệu cũ",
    # Mood label translations
    "amazing": "rất vui",
    "happy": "vui",
    "good": "tốt",
    "great": "tốt",
    "sad": "buồn",
    "okay": "bình thường",
    "ok": "ok",
    "terrible": "rất tệ",
    "neutral": "bình thường",
    "stressed": "căng thẳng",
    "calm": "bình yên",
    "anxious": "lo lắng",
    "tired": "mệt",
    "energetic": "tràn năng lượng",
    "excited": "hào hứng",
    "frustrated": "bực bội",
    "angry": "tức giận",
    "depressed": "buồn nặng",
    "relaxed": "thư giãn",
    "focused": "tập trung",
    "overwhelmed": "quá tải",
    "grateful": "biết ơn",
    "hopeful": "hy vọng",
    "lonely": "cô đơn",
    "content": "hài lòng",
    "proud": "tự hào",
    "confident": "tự tin",
    "irritated": "khó chịu",
    "worried": "lo lắng",
    "stressed out": "căng thẳng",
    "meh": "chán",
}


def _strip_mixed_language_tokens(
    prose: str,
    language: str,
) -> tuple[str, list[dict]]:
    if not prose:
        return prose, []
    is_vi = (language or "").lower().startswith("vi")
    if not is_vi:
        return prose, []

    cleaned = prose
    replacements: list[dict] = []

    tokens = sorted(_EN_TOKEN_TO_VI.items(), key=lambda kv: len(kv[0]), reverse=True)

    for tok, vi_repl in tokens:
        if not tok:
            continue
        if len(tok) < 3:
            continue
        pattern = re.compile(
            r"(?<![A-Za-z0-9])" + re.escape(tok) + r"(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        new_cleaned, n = pattern.subn(vi_repl, cleaned)
        if n > 0:
            replacements.append({"token": tok, "replacement": vi_repl, "count": n})
            cleaned = new_cleaned

    return cleaned, replacements


def _check_english_mood_labels(prose: str) -> list:
    violations: list = []
    quoted = re.findall(r'"([^"]+)"', prose)
    for raw in quoted:
        token = raw.strip().lower()
        if token in _EN_MOOD_LABELS:
            violations.append(
                {
                    "label": token,
                    "raw": raw,
                    "snippet": f"\"{raw}\"",
                    "kind": "quoted_mood_label",
                }
            )
    # Also scan for the bare tokens, but only as whole words.
    lowered = prose.lower()
    for label in _EN_MOOD_LABELS:
        pattern = re.compile(rf"\b{re.escape(label)}\b", re.IGNORECASE)
        for match in pattern.finditer(lowered):
            snippet_start = max(0, match.start() - 25)
            snippet_end = min(len(prose), match.end() + 25)
            snippet = prose[snippet_start:snippet_end]
            violations.append(
                {
                    "label": label,
                    "raw": match.group(0),
                    "snippet": snippet,
                    "kind": "bare_mood_word",
                }
            )
    # De-dup by (label, snippet).
    seen: set = set()
    deduped: list = []
    for v in violations:
        key = (v["label"], v["snippet"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(v)
    return deduped
