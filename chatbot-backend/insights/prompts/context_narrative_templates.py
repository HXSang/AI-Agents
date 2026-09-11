from __future__ import annotations

from typing import Any, List, Optional, Tuple

from insights.health.canonical_field_mapping import CanonicalField

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _v(val: Any, default: str = "unknown") -> str:
    if val is None:
        return default
    if isinstance(val, bool):
        return "yes" if val else "no"
    if isinstance(val, float):
        return f"{val:.1f}"
    return str(val)


def _mins(minutes: int | None) -> str:
    """Format duration: under 60 → minutes only; otherwise hours + minutes."""
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


def _sg(obj: Any, *keys, default=None) -> Any:
    """Safe nested attribute/dict access."""
    for k in keys:
        if obj is None:
            return default
        obj = obj.get(k) if isinstance(obj, dict) else getattr(obj, k, None)
    return obj if obj is not None else default


def _p(items: List[str]) -> str:
    c = [s.strip() for s in items if s and s.strip()]
    return ". ".join(c) + "." if c else ""


# ─── Label maps ────────────────────────────────────────────────────────────────

_SLEEP_L = {"ok": "well-rested", "mild": "mild shortfall", "moderate": "moderate deficit", "severe": "significant deficit"}
_STEP_L = {"not_started": "no steps yet", "ok": "on pace", "mild": "slightly behind", "moderate": "behind pace", "severe": "significantly behind"}
_MEET_L = {"none": "no meetings", "low": "light schedule", "moderate": "moderate load", "high": "heavy day", "overloaded": "back-to-back"}
_MOOD_L = {"positive": "positive", "negative": "low/negative", "neutral": "neutral", "stressed": "stressed", "happy": "happy", "calm": "calm", "anxious": "anxious", "tired": "tired", "energetic": "energetic"}

_MOOD_DISPLAY_NAMES = {
    "vi": {
        "terrible": "rất tệ",
        "sad": "buồn",
        "okay": "bình thường",
        "happy": "vui",
        "amazing": "rất vui / tuyệt vời",
    },
    "en": {
        "terrible": "terrible",
        "sad": "sad",
        "okay": "okay",
        "happy": "happy",
        "amazing": "amazing",
    },
}


def _mood_label(raw_mood: str, language: str = "en") -> str:
    """Translate the canonical mood enum to the user's language.

    If the value is not in the canonical taxonomy, fall back to the raw
    value (with a best-effort pass through the legacy EN label map). Never
    escalate or modify — the LLM must see the EXACT label.
    """
    if not raw_mood:
        return ""
    lang_key = "vi" if (language or "").lower().startswith("vi") else "en"
    return _MOOD_DISPLAY_NAMES.get(lang_key, _MOOD_DISPLAY_NAMES["en"]).get(
        raw_mood, _MOOD_L.get(raw_mood, raw_mood)
    )

_PROD_L = {"ok": "balanced", "overloaded": "overloaded", "underutilized": "light"}
_FOCUS_L = {"low": "low", "moderate": "moderate", "high": "high"}

_CONTEXT_PRIORITY_RANK = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


def _format_evidence_bits(evidence: Any) -> str:
    if not evidence:
        return ""
    if hasattr(evidence, "model_dump"):
        evidence = evidence.model_dump(exclude_none=True)
    if not isinstance(evidence, dict):
        return ""
    skip = {"timezone"}
    bits: List[str] = []
    for key, val in evidence.items():
        if key in skip or val is None or val == "" or val is False:
            continue
        bits.append(f"{key}={val}")
        if len(bits) >= 4:
            break
    return "; ".join(bits)


def _format_active_context_signals(ctx: Any) -> Optional[str]:
    """Human-readable digest of active processor context signals (state only)."""
    raw = _sg(ctx, "health_signals", "context_signals")
    if not raw:
        return None
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(exclude_none=True)
    if not isinstance(raw, dict) or not raw:
        return None

    items: List[Tuple[int, str, str, str]] = []
    for name, payload in raw.items():
        if payload is None:
            continue
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(exclude_none=True)
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "").lower()
        if status != "active":
            continue
        priority = str(payload.get("priority") or "medium").lower()
        rank = _CONTEXT_PRIORITY_RANK.get(priority, 9)
        evidence_txt = _format_evidence_bits(payload.get("evidence") or {})
        items.append((rank, priority, str(name), evidence_txt))

    if not items:
        return None

    items.sort(key=lambda x: (x[0], x[2]))
    lines = [
        "Active context signals (deterministic; higher priority constrains framing):"
    ]
    for _, priority, name, evidence_txt in items:
        line = f"- {name} [{priority}]"
        if evidence_txt:
            line += f": {evidence_txt}"
        lines.append(line)
    lines.append(
        "When recommendations conflict, prefer higher-priority context "
        "(critical > high > medium > low). Do not invent context."
    )
    return "\n".join(lines)


# ─── Part 1: Background ────────────────────────────────────────────────────────


def build_background_template(
    ctx: Any, language: str = "en", domain: Optional[str] = None
) -> str:
    domain_key = (domain or "").strip().lower()
    is_health_only = domain_key == "health"
    parts: List[str] = []

    # ── Sleep ──────────────────────────────────────────────────────────────
    s = _sg(ctx, "health_signals", "sleep")
    meta = _sg(ctx, "meta")
    if s:
        # Check staleness - if last sleep data is not from today/yesterday
        staleness_days = _sg(meta, "sleep_data_staleness_days")
        last_data_date = _sg(meta, "sleep_last_data_date")

        if staleness_days == 999:
            # Sentinel: NO DATA EVER.
            p = "Sleep: no data available yet (device not synced)"
            parts.append(p)
        elif staleness_days is not None and staleness_days >= 1:
            if last_data_date:
                date_part = f"from {last_data_date}"
            else:
                date_part = f"from {staleness_days}d ago"
            p = (
                f"Sleep: NO DATA TODAY — last sleep record is {date_part}; "
                f"do NOT assume current sleep status"
            )
            parts.append(p)
        else:
            # Read from summary + signals (no flat duplication).
            duration_sig = s.signals.sleep_duration if s.signals else None
            debt_sig = s.signals.sleep_debt if s.signals else None
            quality_sig = s.signals.sleep_quality if s.signals else None
            trend_sig = s.signals.sleep_trend if s.signals else None
            timing_sig = s.signals.bedtime_timing if s.signals else None
            duration_metrics = duration_sig.metrics if duration_sig else {}
            debt_metrics = debt_sig.metrics if debt_sig else {}
            quality_metrics = quality_sig.metrics if quality_sig else {}
            trend_metrics = trend_sig.metrics if trend_sig else {}
            timing_metrics = timing_sig.metrics if timing_sig else {}

            level = _SLEEP_L.get(_v(debt_sig.status if debt_sig else None), _v(debt_sig.status if debt_sig else None))
            trend = _v(trend_sig.status if trend_sig else None)
            debt = _v(debt_metrics.get("last_night_debt_h"), "0")
            last = _v(duration_metrics.get("last_night_hours"))
            goal = _v(duration_metrics.get("target_hours"))
            score = _v(quality_metrics.get("sleep_score"))
            streak = getattr(s, "bedtime_streak", None)

            p = f"Sleep: {level} (3d trend: {trend})"
            if last and goal:
                p += f". LAST NIGHT slept {last}h (duration goal: {goal}h)"
            elif last:
                p += f". LAST NIGHT slept {last}h"
            bed = timing_metrics.get("bedtime")
            wake = timing_metrics.get("wake_time")
            bed_source = timing_metrics.get("bedtime_source") or "raw"
            if bed and bed_source == "raw":
                p += (
                    f". LAST_NIGHT_BEDTIME {_v(bed)}"
                    + (f" → wake {_v(wake)}" if wake else "")
                    + " (closed history — NOT tonight; NOT caused by today's calendar)"
                )
            goal_start = timing_metrics.get("goal_start")
            if goal_start:
                p += f". TONIGHT bedtime goal: {_v(goal_start)} (upcoming — distinct from last night)"
            try:
                debt_val = float(debt)
            except (TypeError, ValueError):
                debt_val = 0.0
            if debt_val > 0.5:
                p += f". Sleep debt: {debt}h"
            if score and score != "unknown":
                p += f". Sleep score: {score}/100"
            if streak:
                p += f". Bedtime streak: {streak} nights"
            # Prefer processor signals/overall when present (single source of truth)
            overall = getattr(s, "overall", None)
            if overall is not None:
                o_status = _v(getattr(overall, "status", None))
                o_score = _v(getattr(overall, "score", None))
                o_conf = _v(getattr(overall, "confidence", None))
                if o_status and o_status != "unknown":
                    p += f". Overall: {o_status}"
                    if o_score and o_score != "unknown":
                        p += f" ({o_score}/100, {o_conf})"
            summary = getattr(s, "summary", None)
            if summary is not None:
                avg_d = _v(getattr(summary, "avg_duration_h", None))
                measured = getattr(summary, "measured_days", None)
                if avg_d and avg_d != "unknown":
                    p += f". Weekly avg: {avg_d}h"
                    if measured:
                        p += f" over {measured}d"
            sigs = getattr(s, "signals", None)
            if sigs is not None:
                bits = []
                for name, attr in (
                    ("duration", "sleep_duration"),
                    ("efficiency", "sleep_efficiency"),
                    ("deep", "deep_sleep"),
                    ("REM", "rem_sleep"),
                    ("recovery", "recovery"),
                ):
                    dim = getattr(sigs, attr, None)
                    st = _v(getattr(dim, "status", None)) if dim else None
                    if st and st not in ("unknown", "insufficient_data"):
                        bits.append(f"{name}={st}")
                if bits:
                    p += ". Signals: " + ", ".join(bits)
            parts.append(p)

    # ── Steps / activity ──────────────────────────────────────────────────
    a = _sg(ctx, "health_signals", "steps") or _sg(ctx, "health_signals", "activity")
    meta = _sg(ctx, "meta")
    if a:
        # Check steps staleness
        steps_staleness = _sg(meta, "steps_data_staleness_days")
        steps_last_date = _sg(meta, "steps_last_data_date")

        if steps_staleness == 999:
            # Sentinel: NO DATA EVER. Don't quote a steps number.
            p = "Activity: no steps data available yet (device not synced)"
        elif steps_staleness is not None and steps_staleness >= 1:
            # Data is stale - show no data status
            p = f"Activity: no recent steps data (last data from {steps_last_date})"
        else:
            level = _STEP_L.get(_v(a.level), _v(a.level))
            streak = a.steps_streak
            workout = a.workout_activity_type
            dur = a.workout_duration_min
            p = f"Activity: {level}"
            if getattr(a, "steps_today", None) is not None and getattr(a, "steps_goal", None):
                p += f". Today {a.steps_today}/{a.steps_goal}"
                if getattr(a, "pace_ratio", None) is not None:
                    p += f" ({a.pace_ratio * 100:.0f}% of goal)"
            elif getattr(a, "steps_today", None) is not None:
                p += f". Today {a.steps_today} steps"
            if streak:
                p += f". Step streak: {streak}d"
            if workout:
                p += f". Regular workout: {workout}"
                if dur:
                    p += f" ({_mins(dur)})"
            overall = getattr(a, "overall", None)
            if overall is not None:
                o_status = _v(getattr(overall, "status", None))
                o_score = _v(getattr(overall, "score", None))
                if o_status and o_status not in ("unknown", "insufficient_data"):
                    p += f". Overall: {o_status}"
                    if o_score and o_score != "unknown":
                        p += f" ({o_score})"
            summary = getattr(a, "summary", None)
            if summary is not None:
                avg = getattr(summary, "avg_steps", None)
                measured = getattr(summary, "days_measured", None)
                if avg is not None:
                    p += f". Weekly avg: {avg:.0f} steps"
                    if measured:
                        p += f" over {measured}d"
            sigs = getattr(a, "signals", None)
            if sigs is not None:
                bits = []
                for name, attr in (
                    ("volume", "activity_volume"),
                    ("goal", "goal_achievement"),
                    ("consistency", "activity_consistency"),
                    ("trend", "step_trend"),
                    ("baseline", "baseline_comparison"),
                    ("pattern", "activity_pattern"),
                ):
                    dim = getattr(sigs, attr, None)
                    st = _v(getattr(dim, "status", None)) if dim else None
                    if st and st not in ("unknown", "insufficient_data"):
                        bits.append(f"{name}={st}")
                if bits:
                    p += ". Signals: " + ", ".join(bits)
        parts.append(p)

    # ── Heart rate / mood / energy (5-focus block, cardio fallback) ───────
    hr = _sg(ctx, "health_signals", "heart_rate") or _sg(
        ctx, "health_signals", "cardio_stress"
    )
    mood_sig = _sg(ctx, "health_signals", "mood")
    energy_sig = _sg(ctx, "health_signals", "energy")
    c = _sg(ctx, "health_signals", "cardio_stress")
    if hr or mood_sig or energy_sig or c:
        # Check HR staleness
        hr_staleness = _sg(meta, "hr_data_staleness_days")
        hr_last_date = _sg(meta, "hr_last_data_date")

        p = ""
        if hr_staleness == 999:
            # Sentinel: NO DATA EVER. Don't quote any HR number.
            p += "HR: no data available yet (device not synced). "
        elif hr_staleness is not None and hr_staleness >= 1:
            # HR is stale — DO NOT quote the stale numbers. Surface only the
            # gap + last-known date.
            if hr_last_date:
                date_part = f"from {hr_last_date}"
            else:
                date_part = f"from {hr_staleness}d ago"
            p += (
                f"HR: NO HR DATA TODAY — last reading is {date_part}; "
                f"do NOT assume current resting HR or HRV. "
            )
        elif hr:
            if hr.resting_heart_rate:
                p += f"Resting HR: {hr.resting_heart_rate} bpm. "
            if hr.hrv_score:
                p += f"HRV: {hr.hrv_score}. "
            overall = getattr(hr, "overall", None)
            if overall is not None:
                o_status = _v(getattr(overall, "status", None))
                o_score = _v(getattr(overall, "score", None))
                if o_status and o_status != "unknown":
                    p += f"HR overall: {o_status}"
                    if o_score and o_score != "unknown":
                        p += f" ({o_score})"
                    p += ". "
            sigs = getattr(hr, "signals", None)
            if sigs is not None:
                bits = []
                for name, attr in (
                    ("resting", "resting_heart_rate"),
                    ("HRV", "heart_rate_variability"),
                    ("recovery", "recovery_state"),
                    ("stability", "heart_rate_stability"),
                ):
                    dim = getattr(sigs, attr, None)
                    st = _v(getattr(dim, "status", None)) if dim else None
                    if st and st not in ("unknown", "insufficient_data"):
                        bits.append(f"{name}={st}")
                if bits:
                    p += "HR signals: " + ", ".join(bits) + ". "
        if hr and _v(hr.stress_high) == "yes":
            p += "Stress: elevated. "
        # Mood staleness check: skip stale mood to avoid citing outdated state.
        mood_staleness = _sg(meta, "mood_data_staleness_days")
        mood_last_date = _sg(meta, "mood_last_data_date")
        mood_src = mood_sig or c
        if mood_staleness == 999:
            p += "Mood: no data available yet (device not synced). "
        elif mood_staleness is not None and mood_staleness >= 1:
            if mood_last_date:
                date_part = f"from {mood_last_date}"
            else:
                date_part = f"from {mood_staleness}d ago"
            p += (
                f"Mood: NO MOOD DATA TODAY — last entry is {date_part}; "
                f"do NOT assume current mood. "
            )
        elif mood_src:
            smry = mood_src.summary
            mood = _mood_label(_v(smry.current_mood), language)
            if smry.current_mood:
                p += f"Mood: {mood}. "
            overall = getattr(mood_src, "overall", None)
            if overall is not None:
                o_status = _v(getattr(overall, "status", None))
                if o_status and o_status not in ("unknown", "insufficient_data"):
                    p += f"Mood overall: {o_status}. "
            sigs = getattr(mood_src, "signals", None)
            if sigs is not None:
                bits = []
                for name, attr in (
                    ("baseline", "baseline"),
                    ("trend", "trend"),
                    ("stability", "stability"),
                    ("frequency", "frequency"),
                ):
                    dim = getattr(sigs, attr, None)
                    st = _v(getattr(dim, "status", None)) if dim else None
                    if st and st not in ("unknown", "insufficient_data"):
                        bits.append(f"{name}={st}")
                if bits:
                    p += "Mood signals: " + ", ".join(bits) + ". "

        energy_staleness = _sg(meta, "energy_data_staleness_days")
        energy_last_date = _sg(meta, "energy_last_data_date")
        if energy_staleness == 999:
            p += "Energy burn: no data available yet (device not synced). "
        elif energy_staleness is not None and energy_staleness >= 1:
            if energy_last_date:
                date_part = f"from {energy_last_date}"
            else:
                date_part = f"from {energy_staleness}d ago"
            p += (
                f"Energy burn: NO ENERGY DATA TODAY — last reading is {date_part}; "
                f"do NOT assume today's burn. "
            )
        else:
            cal = _sg(energy_sig, "calories_burned_today")
            pct = _sg(energy_sig, "week_vs_prior_month_pct")
            if pct is None:
                pct = _sg(c, "energy_week_vs_prior_month_pct")
            bits = []
            if cal is not None:
                bits.append(f"{cal:.0f} kcal today")
            if pct is not None:
                sign = "+" if pct > 0 else ""
                bits.append(f"{sign}{pct:.1f}% vs prior month")
            overall = getattr(energy_sig, "overall", None)
            if overall is not None:
                o_status = _v(getattr(overall, "status", None))
                if o_status and o_status != "unknown":
                    bits.append(f"overall {o_status}")
            summary = getattr(energy_sig, "summary", None)
            if summary is not None:
                avg = getattr(summary, "avg_energy_burn", None)
                measured = getattr(summary, "days_measured", None)
                if avg is not None:
                    bits.append(f"weekly avg {avg:.0f} kcal")
                    if measured:
                        bits.append(f"{measured}d measured")
            sigs = getattr(energy_sig, "signals", None)
            if sigs is not None:
                for name, attr in (
                    ("volume", "activity_volume"),
                    ("trend", "energy_trend"),
                    ("baseline", "baseline_comparison"),
                ):
                    dim = getattr(sigs, attr, None)
                    st = _v(getattr(dim, "status", None)) if dim else None
                    if st and st not in ("unknown", "insufficient_data"):
                        bits.append(f"{name}={st}")
            if bits:
                p += "Energy burn: " + ", ".join(bits) + ". "
        if p:
            parts.append(p.rstrip(". "))

    # ── Health score (composite) ──────────────────────────────────────────
    hscore = _sg(ctx, "health_signals", "health_score")
    if hscore:
        summary = getattr(hscore, "summary", None) or {}
        overall = getattr(hscore, "overall", None)
        agg = getattr(hscore, "aggregate", None)
        bits = []
        cur = _sg(summary, "current_score")
        prev = _sg(summary, "previous_score")
        delta = _sg(summary, "delta")
        if cur is not None:
            bits.append(f"current {cur}")
        if prev is not None:
            bits.append(f"previous {prev}")
        if delta is not None:
            sign = "+" if delta > 0 else ""
            bits.append(f"delta {sign}{delta}")
        if overall is not None:
            o_status = _v(getattr(overall, "status", None))
            if o_status and o_status not in ("unknown", "insufficient_data"):
                bits.append(f"overall {o_status}")
        theme = _sg(agg, "theme") if agg is not None else None
        if theme:
            bits.append(f"theme {theme}")
        fired = getattr(hscore, "signals", None) or []
        keys = []
        for item in fired:
            key = _sg(item, "key")
            if key:
                keys.append(str(key))
        if keys:
            bits.append("fired " + ", ".join(keys[:5]))
        if bits:
            parts.append("Health score: " + "; ".join(bits))

    # ── Period aggregates (today vs week/month/baseline) ─────────────────
    period = _sg(ctx, "health_signals", "period_aggregates")
    if period:
        pa_bits = []
        for label, attr in (
            ("steps", "steps"),
            ("sleep", "sleep_hours"),
            ("RHR", "resting_heart_rate"),
            ("active_min", "active_minutes"),
            ("calories", "calories_burned"),
        ):
            m = getattr(period, attr, None) if not isinstance(period, dict) else period.get(attr)
            if m is None:
                continue
            today = _sg(m, "today")
            week = _sg(m, "week_avg")
            base = _sg(m, "baseline_avg")
            chunk = []
            if today is not None:
                chunk.append(f"today {today}")
            if week is not None:
                chunk.append(f"week {week}")
            if base is not None:
                chunk.append(f"baseline {base}")
            if chunk:
                pa_bits.append(f"{label}: " + ", ".join(chunk))
        flags = []
        if _sg(period, "rhr_elevated"):
            flags.append("RHR elevated")
        if _sg(period, "workout_high_recent_load"):
            flags.append("high recent workout load")
        if flags:
            pa_bits.append("flags: " + ", ".join(flags))
        if pa_bits:
            parts.append("Period aggregates: " + "; ".join(pa_bits))

    # ── Historical deep-dive (from DailySnapshotData) ─────────────────────
    ht = _sg(ctx, "historical_trends")
    recent_days = _sg(ht, "recent_days") or []
    a3 = _sg(ht, "averages_3d") or {}
    meta = _sg(ctx, "meta")
    steps_staleness = _sg(meta, "steps_data_staleness_days")

    if a3 and steps_staleness is not None and steps_staleness < 1:
        p = "3d averages:"
        if _sg(a3, CanonicalField.STEP_COUNT_ONE_DAY):
            p += f" steps {_sg(a3, CanonicalField.STEP_COUNT_ONE_DAY)}/day"
        if _sg(a3, CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS):
            p += f", sleep {_sg(a3, CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS)}h"
        if _sg(a3, CanonicalField.MEETING_DURATION_ONE_DAY_MINUTES) is not None:
            p += f", meetings {_mins(int(_sg(a3, CanonicalField.MEETING_DURATION_ONE_DAY_MINUTES)))}"
        if _sg(a3, "mood_score_avg"):
            p += f", mood {_sg(a3, 'mood_score_avg'):.1f}/10"
        if p != "3d averages:":
            parts.append(p)

    # 7d activity: exercise sessions & workout minutes
    if recent_days:
        exercise_days = sum(1 for d in recent_days if _sg(d, CanonicalField.EXERCISE_SESSION_COUNT_ONE_DAY) or _sg(d, "h_exercise_sessions"))
        total_workout_mins = sum(
            _sg(d, CanonicalField.TOTAL_WORKOUT_DURATION_ONE_DAY_MINUTES) or _sg(d, "h_total_workout_min") or 0 for d in recent_days
        )
        avg_workout = (
            _mins(int(total_workout_mins / len(recent_days)))
            if total_workout_mins and len(recent_days) > 0
            else None
        )
        if exercise_days > 0 or total_workout_mins:
            p = "7d exercise:"
            if exercise_days:
                p += f" {exercise_days}/{len(recent_days)} days"
            if avg_workout:
                p += f", avg {avg_workout}/day"
            if total_workout_mins >= 60:
                p += f" (total {_mins(int(total_workout_mins))})"
            parts.append(p)

        # Focus blocks from historical
        focus_blocks = [
            (_sg(d, "p_focus_blocks_60min") or 0) + (_sg(d, "p_focus_blocks_30min") or 0) / 2
            for d in recent_days
            if _sg(d, "p_focus_blocks_60min") or _sg(d, "p_focus_blocks_30min")
        ]
        if focus_blocks:
            avg_focus = sum(focus_blocks) / len(focus_blocks)
            max_focus = max(focus_blocks)
            if avg_focus > 0:
                parts.append(f"Focus blocks: avg {avg_focus:.1f}/day, best day {max_focus:.0f}")

        # Meeting longest block
        longest_meetings = [_sg(d, "p_longest_meeting_min") for d in recent_days if _sg(d, "p_longest_meeting_min")]
        if longest_meetings:
            avg_longest = sum(longest_meetings) / len(longest_meetings)
            parts.append(f"Avg longest meeting: {_mins(int(avg_longest))}")

        # Hydration
        water_vals = [_sg(d, CanonicalField.WATER_INTAKE_ONE_DAY_LITERS) or _sg(d, "h_water_liters") for d in recent_days if _sg(d, CanonicalField.WATER_INTAKE_ONE_DAY_LITERS) or _sg(d, "h_water_liters")]
        if water_vals:
            avg_water = sum(water_vals) / len(water_vals)
            parts.append(f"Hydration: avg {avg_water:.1f}L/day")

        # Wellness/productivity scores from snapshots (canonical + legacy fallbacks)
        wellness_scores = [_sg(d, CanonicalField.WELLNESS_SCORE_ONE_DAY) or _sg(d, "wellness_score") for d in recent_days if _sg(d, CanonicalField.WELLNESS_SCORE_ONE_DAY) or _sg(d, "wellness_score")]
        prod_scores = [_sg(d, "productivity_score") for d in recent_days if _sg(d, "productivity_score")]
        fin_scores = [_sg(d, "financial_health_score") for d in recent_days if _sg(d, "financial_health_score")]

        score_parts = []
        if wellness_scores:
            score_parts.append(f"wellness {sum(wellness_scores)/len(wellness_scores):.0f}")
        if prod_scores:
            score_parts.append(f"productivity {sum(prod_scores)/len(prod_scores):.0f}")
        if fin_scores and not is_health_only:
            score_parts.append(f"financial {sum(fin_scores)/len(fin_scores):.0f}")
        if score_parts:
            parts.append(f"7d scores: {', '.join(score_parts)}")

        # Finance cashflow
        incomes = [_sg(d, "total_income") or _sg(d, "f_total_income") for d in recent_days if _sg(d, "total_income") or _sg(d, "f_total_income")]
        expenses = [_sg(d, "total_expense") or _sg(d, "f_total_expense") for d in recent_days if _sg(d, "total_expense") or _sg(d, "f_total_expense")]
        if incomes and expenses and not is_health_only:
            total_in = sum(incomes)
            total_out = sum(expenses)
            net = total_in - total_out
            cashflow_label = "positive" if net > 0 else "negative" if net < 0 else "neutral"
            parts.append(f"Cashflow: {cashflow_label} ({net:+.0f} over {len(recent_days)}d)")

    # ── Behavioral patterns ───────────────────────────────────────────────
    bp = _sg(ctx, "behavioral_patterns")
    if bp:
        p = ""
        if bp.sleep_trend_7d:
            p += f"Sleep 7d: {bp.sleep_trend_7d}. "
        if bp.activity_trend_7d:
            p += f"Activity 7d: {bp.activity_trend_7d}. "
        if bp.personal_context and bp.personal_context != "unknown":
            p += f"{bp.personal_context}. "
        if bp.meeting_fragmentation:
            p += f"Meetings: {bp.meeting_fragmentation}. "
        if bp.work_span_trend:
            p += f"Work hours: {bp.work_span_trend}. "
        if bp.task_completion_momentum:
            p += f"Task momentum: {bp.task_completion_momentum}. "
        if bp.collaboration_load:
            p += f"Collaboration: {bp.collaboration_load}. "
        if bp.mood_avg_7d_label:
            p += f"Mood 7d: {bp.mood_avg_7d_label}. "
        if p:
            parts.append(p.rstrip(". "))

    # ── Productivity signals ──────────────────────────────────────────────
    ps = _sg(ctx, "productivity_signals")
    if ps and not is_health_only:
        level = _PROD_L.get(_v(ps.level), _v(ps.level))
        focus_raw = _v(ps.focus_score)
        focus = _FOCUS_L.get(focus_raw, focus_raw)
        p = f"Productivity: {level}. Focus opportunity: {focus}"
        if ps.meetings_due_today and ps.meeting_completion_rate is not None:
            m_done = int(round(ps.meeting_completion_rate * ps.meetings_due_today))
            p += f". Meetings attended: {m_done}/{ps.meetings_due_today}"
        elif ps.meetings_due_today:
            p += f". Meetings today: {ps.meetings_due_today}"
        if ps.reminders_due_today and ps.reminders_completion_rate is not None:
            r_done = int(round(ps.reminders_completion_rate * ps.reminders_due_today))
            p += f". Reminders: {r_done}/{ps.reminders_due_today} done"
        elif ps.reminders_due_today:
            p += f". Reminders due: {ps.reminders_due_today}"

        # 7-day precomputed averages (PAST days only) — these are authoritatively computed
        avg_bits = []
        if ps.events_completion_7d_avg is not None:
            avg_bits.append(f"event completion 7d avg {ps.events_completion_7d_avg * 100:.0f}%")
        if ps.meeting_minutes_7d_avg is not None:
            avg_bits.append(f"meetings/day avg {_mins(int(ps.meeting_minutes_7d_avg))}")
        if ps.focus_minutes_7d_avg is not None:
            avg_bits.append(f"focus/day avg {_mins(int(ps.focus_minutes_7d_avg))}")
        if ps.chronic_overload_days:
            avg_bits.append(f"{ps.chronic_overload_days} heavy-meeting days")
        if avg_bits:
            p += "\n7d precomputed (PAST days only — do NOT divide by 7 to estimate daily averages): " + ", ".join(avg_bits)
        if ps.events_completion_trend:
            p += f". Trend: {ps.events_completion_trend}"

        parts.append(p)

    # ── Finance signals ────────────────────────────────────────────────────
    fs = _sg(ctx, "finance_signals")
    if fs and not is_health_only:
        p = ""
        if fs.budget_utilization is not None:
            p += f"Budget: {fs.budget_utilization:.0f}% used. "
        if fs.cashflow_status:
            p += f"Cashflow: {fs.cashflow_status}. "
        if fs.active_goals_on_track:
            p += f"Goals on track: {fs.active_goals_on_track}. "
        if fs.active_goals_behind:
            p += f"Goals behind: {fs.active_goals_behind}. "
        if fs.critical_alerts:
            p += f"Alerts: {', '.join(fs.critical_alerts[:2])}. "
        if p:
            parts.append(p.rstrip(". "))

    # ── User profile ───────────────────────────────────────────────────────
    up = _sg(ctx, "user_profile")
    if up:
        p = ""
        if _v(up.name) != "unknown":
            p += f"Name: {up.name}. "
        if up.age:
            p += f"Age: {up.age}. "
        if _v(up.job_title) != "unknown":
            p += f"Job: {up.job_title}. "
        if _v(up.primary_goal) != "unknown":
            p += f"Goal: {up.primary_goal}. "
        if up.current_health_conditions:
            p += f"Conditions: {', '.join(up.current_health_conditions)}. "
        if p:
            parts.append(p.rstrip(". "))

    # TODO(narrative-refactor): collapse with ``_build_narrative_context``.
    narrative_ctx = getattr(ctx, "narrative_context", None)
    if narrative_ctx:
        if isinstance(narrative_ctx, dict):
            nc_dict = narrative_ctx
        else:
            # Pydantic model — convert to dict
            try:
                nc_dict = narrative_ctx.model_dump()
            except Exception:
                nc_dict = {}
        if nc_dict:
            bg_lines = []
            for key in [
                "user_intro",
                "health_snapshot",
                "activity_snapshot",
                "cardio_snapshot",
                "week_pattern",
                "mood_narrative",
                "productivity_today",
                "finance_snapshot",
                "goal_progress",
                "calendar_outlook",
                "risk_flags",
                "positive_signals",
            ]:
                v = nc_dict.get(key)
                if v:
                    bg_lines.append(v)
            if bg_lines:
                parts.append("PRE-DIGESTED CONTEXT: " + " | ".join(bg_lines))

    if not parts:
        return "[BACKGROUND]\nInsufficient data available."

    return "[BACKGROUND]\n" + "\n".join(parts)


# ─── Part 2: Current state ─────────────────────────────────────────────────────


def build_current_state_template(ctx: Any, language: str = "en") -> str:
    parts: List[str] = []

    # Meta / time
    m = _sg(ctx, "meta")
    if m:
        p = f"Time: {_v(m.current_time)} ({_v(m.time_phase)}), {_v(m.day_of_week)}"
        if _v(m.is_weekend) == "yes":
            p += " (weekend)"
        parts.append(p)
        if m.next_event_summary and m.minutes_until_next_event is not None:
            evt = f"Next event: {_v(m.next_event_summary)}"
            if getattr(m, "next_event_category", None):
                evt += f" [{_v(m.next_event_category)}]"
            if getattr(m, "next_event_type", None):
                evt += f" ({_v(m.next_event_type)})"
            if getattr(m, "next_event_start_time", None):
                evt += f" at {_v(m.next_event_start_time)}"
            evt += f" (in {_mins(m.minutes_until_next_event)} — precomputed)"
            parts.append(evt)

    # Processor context signals (sleep_window / weekend / work_hours / recent_workout)
    context_block = _format_active_context_signals(ctx)
    if context_block:
        parts.append(context_block)

    # Calendar intelligence
    cal = _sg(ctx, "calendar_intelligence")
    if cal:
        meet_min = cal.meeting_minutes_today or 0
        b2b = cal.back_to_back_count or 0

        # Only include meeting info if there's actual meeting activity
        if meet_min > 0 or b2b > 0 or cal.cognitive_load_so_far in ("moderate", "high", "overloaded"):
            load = _MEET_L.get(_v(cal.cognitive_load_so_far), _v(cal.cognitive_load_so_far))
            p = f"Meeting load: {load}"
            if meet_min > 0:
                p += f". {_mins(int(meet_min))} in meetings so far"
            if b2b > 0:
                p += f". Back-to-back: {b2b}"
            parts.append(p)

        # Current calendar EVENT (not a reminder; do not invent call/meeting)
        curr = cal.current_event
        if curr:
            line = f"Now (calendar EVENT / sự kiện): {_v(curr.title)}"
            if getattr(curr, "category", None):
                line += f" [{_v(curr.category)}]"
            if getattr(curr, "event_type", None):
                line += f" type={_v(curr.event_type)}"
            line += f" ({_mins(curr.minutes_left)} left)"
            parts.append(line)
        else:
            nxt = cal.next_event
            if nxt:
                # Time until the calendar event itself — NOT free-window duration
                line = (
                    f"Time until next calendar EVENT {_v(nxt.title)!r}"
                )
                if getattr(nxt, "category", None):
                    line += f" [{_v(nxt.category)}]"
                line += (
                    f": {_mins(nxt.minutes_away)} "
                    f"(starts {_v(nxt.start_time)} — use THIS for 'before appointment')"
                )
                parts.append(line)

        nearest_rem = getattr(cal, "nearest_reminder", None)
        if nearest_rem and getattr(nearest_rem, "title", None):
            due = getattr(nearest_rem, "due_in_mins", None)
            if due is None:
                due = getattr(nearest_rem, "minutes_away", None)
            rem_line = f"Nearest REMINDER / ghi nhớ: {_v(nearest_rem.title)}"
            if due is not None:
                rem_line += f" (in {_mins(due)})"
            parts.append(rem_line)

        prev = cal.previous_event
        if prev:
            line = f"Just finished: {_v(prev.title)}"
            if getattr(prev, "category", None):
                line += f" [{_v(prev.category)}]"
            line += f" ({_mins(prev.minutes_since)} ago)"
            parts.append(line)

        upcoming = getattr(cal, "upcoming_events", None) or []
        if upcoming:
            up_bits = []
            for u in upcoming[:4]:
                bit = _v(getattr(u, "title", None))
                cat = getattr(u, "category", None)
                if cat:
                    bit += f"[{_v(cat)}]"
                mins = getattr(u, "minutes_away", None)
                if mins is not None:
                    bit += f" in {_mins(mins)}"
                up_bits.append(bit)
            if up_bits:
                parts.append("Upcoming (title[category]): " + "; ".join(up_bits))

        windows = getattr(cal, "free_windows_today", None) or []
        if windows:
            fw_bits = []
            for fw in windows[:3]:
                bit = f"{_mins(fw.duration_mins)} ({_v(fw.position)}"
                capped = getattr(fw, "capped_by", None)
                if capped:
                    bit += f", capped_by={capped}"
                ends = getattr(fw, "ends_at", None)
                if ends:
                    bit += f", ends_at={_v(ends)}"
                bit += ")"
                fw_bits.append(bit)
            parts.append(
                "Work free slots (capped at active hours / bedtime — "
                "NOT equal to time until a later calendar event): "
                + "; ".join(fw_bits)
            )

        breakdown = getattr(cal, "meeting_type_breakdown", None)
        by_cat = getattr(breakdown, "by_category", None) if breakdown else None
        if by_cat:
            if hasattr(by_cat, "items"):
                cat_bits = [f"{k}={v}" for k, v in by_cat.items() if v]
            elif isinstance(by_cat, dict):
                cat_bits = [f"{k}={v}" for k, v in by_cat.items() if v]
            else:
                cat_bits = []
            if cat_bits:
                parts.append("Today's event categories: " + ", ".join(cat_bits))
    act = _sg(ctx, "health_signals", "steps") or _sg(ctx, "health_signals", "activity")
    steps_staleness = _sg(m, "steps_data_staleness_days")
    steps_last_date = _sg(m, "steps_last_data_date")
    if act:
        if steps_staleness == 999:
            parts.append("Steps today: no data available yet (device not synced)")
        elif steps_staleness is not None and steps_staleness >= 1:
            if steps_last_date:
                date_part = f"from {steps_last_date}"
            else:
                date_part = f"from {steps_staleness}d ago"
            parts.append(
                f"Steps today: NO DATA TODAY — last reading is {date_part}; "
                f"do NOT assume current steps"
            )
        elif act.steps_today is not None and act.steps_goal:
            pct = (
                f"{act.pace_ratio * 100:.0f}%"
                if act.pace_ratio is not None
                else "n/a"
            )
            remaining = act.remaining_steps if act.remaining_steps is not None else "?"
            parts.append(
                f"Steps today: {act.steps_today:,} of {act.steps_goal:,}. "
                f"Still need {remaining:,} more steps to reach the goal "
                f"({pct} of goal done so far)."
            )
        elif act.steps_today is not None:
            parts.append(f"Steps today: {act.steps_today:,}")

    # Goals
    goals = _sg(ctx, "goal_signals", "goals")
    if goals:
        goal_parts = []
        for g in (goals or [])[:3]:
            if not g:
                continue
            pct = f"{g.pace_ratio * 100:.0f}%" if g.pace_ratio is not None else "n/a"
            goal_parts.append(f"{_v(g.goal_label)} ({_v(g.urgency_level)}, {_v(g.days_remaining)}d left, pace: {pct})")
        if goal_parts:
            parts.append(f"Active goals: {'; '.join(goal_parts)}")

    # Balance scores (today's + 7d averages from /api/balance range)
    bs = getattr(ctx, "balance_score", None)
    if bs:
        p = "Today's scores:"
        scores = []
        for key in ("wellness_score", "productivity_score", "financial_health_score", "overall_day_score"):
            v = _v(_sg(bs, key), "")
            if v and v != "unknown":
                scores.append(f"{key.replace('_score', '')} {v}")
        if scores:
            p += " " + ", ".join(scores)

        avg_parts = []
        for short, full in [
            ("wellness", "wellness_score_7d_avg"),
            ("productivity", "productivity_score_7d_avg"),
            ("finance", "finance_score_7d_avg"),
            ("balance", "balance_score_7d_avg"),
        ]:
            avg_v = _sg(bs, full)
            if avg_v is not None:
                avg_parts.append(f"{short} 7d avg {avg_v:.0f}")
        if avg_parts:
            total_days = _sg(bs, "balance_score_7d_total_days")
            good_days = _sg(bs, "balance_score_7d_good_days")
            p += "\n7d averages (precomputed from PAST days only): " + ", ".join(avg_parts)
            if total_days:
                p += f". Window: {good_days or 0}/{total_days} strong days (past 7 days)"

        parts.append(p)

    if not parts:
        return "[CURRENT]\nInsufficient data available."

    return "[CURRENT]\n" + "\n".join(parts)


# ─── Part 3: Future ────────────────────────────────────────────────────────────


def build_future_template(ctx: Any, language: str = "en") -> str:
    parts: List[str] = []

    cal = _sg(ctx, "calendar_intelligence")
    if cal:
        nxt = cal.next_event
        if nxt:
            line = f"Next: {_v(nxt.title)}"
            if getattr(nxt, "category", None):
                line += f" [{_v(nxt.category)}]"
            line += f" at {_v(nxt.start_time)} (in {_mins(nxt.minutes_away)})"
            parts.append(line)

        upcoming = getattr(cal, "upcoming_events", None) or []
        if upcoming and len(upcoming) > 1:
            up_bits = []
            for u in upcoming[1:4]:
                bit = _v(getattr(u, "title", None))
                cat = getattr(u, "category", None)
                if cat:
                    bit += f"[{_v(cat)}]"
                up_bits.append(bit)
            if up_bits:
                parts.append("Later today: " + "; ".join(up_bits))

        windows = cal.free_windows_today or []
        if windows:
            fw_parts = []
            for fw in windows[:3]:
                bit = f"{_mins(fw.duration_mins)} ({_v(fw.position)}"
                capped = getattr(fw, "capped_by", None)
                if capped:
                    bit += f"/{capped}"
                bit += ")"
                fw_parts.append(bit)
            parts.append(
                "Work free windows (until active-hours end unless capped_by=next_event): "
                + ", ".join(fw_parts)
            )

    goals = _sg(ctx, "goal_signals", "goals")
    if goals:
        urgent = [g for g in (goals or [])[:3] if g and _v(g.urgency_level) in ("high", "moderate")]
        if urgent:
            g_parts = [f"{_v(g.goal_label)} (due in {_v(g.days_remaining)}d)" for g in urgent]
            parts.append(f"Upcoming deadlines: {', '.join(g_parts)}")

    m = _sg(ctx, "meta")
    if m:
        if m.active_hours_end_time:
            parts.append(f"Active hours end: {m.active_hours_end_time}")
        if m.hard_constraints:
            parts.append(f"Hard constraints: {', '.join(m.hard_constraints[:3])}")

    if not parts:
        return "[FUTURE]\nNo upcoming events or deadlines."

    return "[FUTURE]\n" + "\n".join(parts)


# ─── Public API ────────────────────────────────────────────────────────────────


def build_3part_narrative_parts(
    ctx: Any, language: str = "en", domain: Optional[str] = None
) -> Tuple[str, str, str]:
    """Return three narrative parts (background / current / future)."""
    return (
        build_background_template(ctx, language, domain=domain),
        build_current_state_template(ctx, language),
        build_future_template(ctx, language),
    )
