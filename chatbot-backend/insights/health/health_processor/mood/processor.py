"""Mood signal processor — single source of truth for mood context.

Mood is descriptive only. Never diagnose or infer medical/psychological conditions.
Pipeline: snapshot → baseline → trend → pattern → stability → frequency → change
→ optional correlations → evidence-rich signals.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from insights.health.health_processor.common.utils import health_param_value
from insights.health.health_processor.mood.constants import (
    MOOD_BASELINE_LOOKBACK_DAYS,
    MOOD_DECLINED_DELTA,
    MOOD_FREQUENCY_LOOKBACK_DAYS,
    MOOD_IMPROVED_DELTA,
    MOOD_LOGGING_HIGH_ENTRIES_7D,
    MOOD_LOGGING_MED_ENTRIES_7D,
    MOOD_MIN_BASELINE_ENTRIES,
    MOOD_MIN_TREND_ENTRIES,
    MOOD_NEGATIVE,
    MOOD_OVERALL_WEIGHT_CURRENT,
    MOOD_OVERALL_WEIGHT_LOGGING,
    MOOD_OVERALL_WEIGHT_STABILITY,
    MOOD_OVERALL_WEIGHT_TREND,
    MOOD_POSITIVE,
    MOOD_SCORE_MAP,
    MOOD_STREAK_MIN_DAYS,
    MOOD_TREND_LOOKBACK_DAYS,
    MOOD_VOLATILITY_HIGH_STD,
    MOOD_VOLATILITY_MEDIUM_STD,
)
from insights.schemas.processed_context import (
    MoodAnomaly,
    MoodDimensionSignal,
    MoodOverall,
    MoodSignal,
    MoodSignalsMap,
    MoodSummaryMetrics,
)


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _snapshot_field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _snapshot_date(item: Any) -> Optional[date]:
    raw = _snapshot_field(item, "daily_snapshot_date")
    if not raw:
        return None
    try:
        if isinstance(raw, date) and not isinstance(raw, datetime):
            return raw
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
    except Exception:
        return None


def _mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _stdev(vals: List[float]) -> Optional[float]:
    if len(vals) < 2:
        return None
    return statistics.pstdev(vals)


def _mood_score(label: Optional[str]) -> Optional[int]:
    if not label or not isinstance(label, str):
        return None
    return MOOD_SCORE_MAP.get(label.strip().lower())


def _score_to_mood(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    s = int(round(score))
    rev = {1: "terrible", 2: "sad", 3: "okay", 4: "happy", 5: "amazing"}
    return rev.get(max(1, min(5, s)))


def _display_name(label: Optional[str], override: Optional[str] = None) -> Optional[str]:
    if override:
        return override
    if not label:
        return None
    return label.strip().capitalize()


def _dim(
    status: str,
    metrics: Optional[Dict[str, Any]] = None,
    comparison: Optional[Dict[str, Any]] = None,
    trend: Optional[str] = None,
    evidence_context: Optional[Dict[str, Any]] = None,  # kept for caller compat; ignored
) -> MoodDimensionSignal:
    metrics = metrics or {}
    comparison = comparison or {}
    return MoodDimensionSignal(
        status=status,
        metrics=metrics,
        comparison=comparison,
        trend=trend,
    )


def _is_stale(staleness_days: Optional[int], threshold: int = 0) -> bool:
    if staleness_days is None:
        return True
    return staleness_days > threshold


@dataclass
class MoodEntry:
    entry_date: date
    mood: Optional[str] = None
    score: Optional[float] = None
    display_name: Optional[str] = None
    has_notes: bool = False


class MoodSignalProcessor:
    """Transform latest_mood + history into evidence-rich MoodSignal."""

    @classmethod
    def process(
        cls,
        health_params: Optional[Dict[str, Any]] = None,
        historical_snapshots: Optional[List[Any]] = None,
        staleness_info: Optional[Dict[str, Any]] = None,
        time_data: Optional[Dict[str, Any]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
        *,
        sleep_context: Optional[Dict[str, Any]] = None,
        energy_context: Optional[Dict[str, Any]] = None,
        hr_context: Optional[Dict[str, Any]] = None,
        calendar_context: Optional[Dict[str, Any]] = None,
    ) -> MoodSignal:
        try:
            return cls._process_unsafe(
                health_params=health_params or {},
                historical_snapshots=historical_snapshots or [],
                staleness_info=staleness_info or {},
                time_data=time_data or {},
                raw_data=raw_data or {},
                sleep_context=sleep_context,
                energy_context=energy_context,
                hr_context=hr_context,
                calendar_context=calendar_context,
            )
        except Exception:
            try:
                from utils.logger import logger as _logger

                _logger.warning("MoodSignalProcessor failed", exc_info=True)
            except Exception:
                pass
            return MoodSignal(
                overall=MoodOverall(status="insufficient_data", confidence="low"),
            )

    @classmethod
    def _process_unsafe(
        cls,
        health_params: Dict[str, Any],
        historical_snapshots: List[Any],
        staleness_info: Dict[str, Any],
        time_data: Dict[str, Any],
        raw_data: Dict[str, Any],
        sleep_context: Optional[Dict[str, Any]],
        energy_context: Optional[Dict[str, Any]],
        hr_context: Optional[Dict[str, Any]],
        calendar_context: Optional[Dict[str, Any]],
    ) -> MoodSignal:
        today = cls._resolve_today(time_data)
        now = cls._resolve_now(time_data)
        mood_staleness = staleness_info.get("mood_data_staleness_days")
        stale = _is_stale(mood_staleness) if mood_staleness is not None else False
        if stale and raw_data.get("latest_mood"):
            stale = False

        current = cls._extract_current_mood(raw_data, health_params)
        entries = cls._aggregate_entries(
            historical_snapshots=historical_snapshots,
            raw_data=raw_data,
            today=today,
            current=current if not stale else None,
        )

        has_mood = bool(current.get("mood")) and not (
            mood_staleness == 999
        )
        # If no data ever
        if mood_staleness == 999 or (not has_mood and not entries):
            return cls._empty_no_mood()

        # Soft-stale: keep historical signals, clear "today" framing
        if stale:
            has_mood_today = False
        else:
            has_mood_today = has_mood

        age_hours = cls._mood_age_hours(current.get("mood_date"), now)
        # Prefer user-TZ-normalized local date injected by the collector
        # (see DataCollector._normalize_latest_mood). Falls back to the
        # server-local timezone conversion so callers that bypass the
        # collector (e.g. unit tests, direct invocations) still work.
        local_date_iso = current.get("mood_date_local")
        last_logged_days_ago = None
        if local_date_iso and today is not None:
            try:
                local_date = date.fromisoformat(str(local_date_iso).split("T")[0])
                last_logged_days_ago = (today - local_date).days
            except Exception:
                last_logged_days_ago = None
        if last_logged_days_ago is None and current.get("mood_date") and today is not None:
            try:
                # Convert mood_date to local timezone before extracting date
                # so we don't compare UTC-date with local-date (off by one).
                # NOTE: this fallback uses the SERVER's local timezone and
                # is only kept for backward compatibility with callers that
                # do not go through DataCollector._normalize_latest_mood.
                dt = datetime.fromisoformat(
                    str(current["mood_date"]).replace("Z", "+00:00")
                )
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                local_date = dt.astimezone().date()
                last_logged_days_ago = (today - local_date).days
            except Exception:
                last_logged_days_ago = mood_staleness

        counts = cls._entry_counts(entries, today)
        summary = MoodSummaryMetrics(
            has_mood=has_mood_today,
            current_mood=None if stale else current.get("mood"),
            current_mood_display_name=(
                None if stale else current.get("display_name")
            ),
            current_mood_score=None if stale else current.get("score"),
            current_mood_date=current.get("mood_date"),
            mood_age_hours=None if stale else age_hours,
            entries_last_7_days=counts[7],
            entries_last_14_days=counts[14],
            entries_last_30_days=counts[30],
            last_logged_days_ago=last_logged_days_ago,
            days_measured=counts[7],
            coverage_pct=round(100.0 * counts[7] / MOOD_TREND_LOOKBACK_DAYS, 1),
        )

        baseline = cls._baseline(entries, today, summary)
        trend = cls._trend(entries, today)
        pattern = cls._pattern(entries, today)
        stability = cls._stability(entries, today)
        frequency = cls._frequency(counts, last_logged_days_ago)
        change = cls._change(entries, today, summary, baseline)
        correlation = cls._correlation(
            summary,
            sleep_context=sleep_context,
            energy_context=energy_context,
            hr_context=hr_context,
            calendar_context=calendar_context,
        )
        anomalies = cls._detect_anomalies(entries, today)

        ctx = {
            "entries_last_7_days": counts[7],
            "entries_last_14_days": counts[14],
            "entries_last_30_days": counts[30],
            "last_logged_days_ago": last_logged_days_ago,
            "has_mood": has_mood_today,
        }

        signals = MoodSignalsMap(
            snapshot=_dim(
                "available" if has_mood_today else (
                    "stale" if current.get("mood") else "no_mood"
                ),
                metrics={
                    "current_mood": summary.current_mood,
                    "current_mood_display_name": summary.current_mood_display_name,
                    # score is internal — included for engine, never show to user
                    "current_mood_score": summary.current_mood_score,
                    "current_mood_date": summary.current_mood_date,
                    "mood_age_hours": summary.mood_age_hours,
                },
                evidence_context=ctx,
            ),
            baseline=_dim(
                baseline.get("status") or "insufficient_data",
                metrics={
                    "baseline_mood": baseline.get("baseline_mood"),
                    "baseline_mood_score": baseline.get("baseline_mood_score"),
                    "mood_delta": baseline.get("mood_delta"),
                },
                comparison={
                    "mood_vs_baseline": baseline.get("mood_vs_baseline"),
                },
                evidence_context={
                    **ctx,
                    "baseline_entries": baseline.get("entries"),
                },
            ),
            trend=_dim(
                trend.get("status") or "insufficient_data",
                metrics={
                    "mood_trend": trend.get("status"),
                    "positive_trend": trend.get("positive_trend"),
                    "negative_trend": trend.get("negative_trend"),
                    "stable_trend": trend.get("stable_trend"),
                    "first_half_avg": trend.get("first_half_avg"),
                    "second_half_avg": trend.get("second_half_avg"),
                },
                trend=trend.get("status"),
                evidence_context=ctx,
            ),
            pattern=_dim(
                pattern.get("status") or "insufficient_data",
                metrics={
                    "positive_streak": pattern.get("positive_streak"),
                    "negative_streak": pattern.get("negative_streak"),
                    "happy_streak": pattern.get("happy_streak"),
                    "sad_streak": pattern.get("sad_streak"),
                    "okay_streak": pattern.get("okay_streak"),
                    "amazing_streak": pattern.get("amazing_streak"),
                },
                evidence_context=ctx,
            ),
            stability=_dim(
                stability.get("status") or "insufficient_data",
                metrics={
                    "mood_consistency": stability.get("status"),
                    "mood_variability": stability.get("std"),
                    "volatile_mood": stability.get("volatile_mood"),
                },
                evidence_context=ctx,
            ),
            frequency=_dim(
                frequency.get("status") or "insufficient_data",
                metrics={
                    "logging_consistency": frequency.get("status"),
                    "entries_last_7_days": counts[7],
                    "entries_last_14_days": counts[14],
                    "entries_last_30_days": counts[30],
                    "last_logged_days_ago": last_logged_days_ago,
                },
                evidence_context=ctx,
            ),
            change=_dim(
                change.get("status") or "insufficient_data",
                metrics={
                    "mood_changed_today": change.get("mood_changed_today"),
                    "mood_improved_today": change.get("mood_improved_today"),
                    "mood_declined_today": change.get("mood_declined_today"),
                    "largest_positive_jump": change.get("largest_positive_jump"),
                    "largest_negative_drop": change.get("largest_negative_drop"),
                },
                evidence_context=ctx,
            ),
            correlation=(
                _dim(
                    correlation.get("status") or "none",
                    metrics=correlation.get("flags") or {},
                    evidence_context={
                        **ctx,
                        "evidence": correlation.get("evidence") or [],
                    },
                )
                if correlation
                else None
            ),
        )

        overall = cls._build_overall(
            summary=summary,
            trend=trend,
            stability=stability,
            frequency=frequency,
            has_mood_today=has_mood_today,
        )

        return MoodSignal(
            summary=summary,
            signals=signals,
            anomalies=anomalies,
            overall=overall,
        )

    # ── empty / resolve ─────────────────────────────────────────────────

    @classmethod
    def _empty_no_mood(cls) -> MoodSignal:
        return MoodSignal(
            overall=MoodOverall(status="insufficient_data", confidence="low"),
        )

    @classmethod
    def _resolve_today(cls, time_data: Dict[str, Any]) -> Optional[date]:
        raw = time_data.get("time_data_current_time_iso")
        if not raw:
            return None
        try:
            return date.fromisoformat(str(raw).split("T")[0])
        except Exception:
            return None

    @classmethod
    def _resolve_now(cls, time_data: Dict[str, Any]) -> Optional[datetime]:
        raw = time_data.get("time_data_current_time_iso")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except Exception:
            return None

    @classmethod
    def _mood_age_hours(
        cls, mood_date: Optional[str], now: Optional[datetime]
    ) -> Optional[float]:
        if not mood_date or now is None:
            return None
        try:
            dt = datetime.fromisoformat(str(mood_date).replace("Z", "+00:00"))
            if dt.tzinfo is None and now.tzinfo is not None:
                dt = dt.replace(tzinfo=timezone.utc)
            if now.tzinfo is None and dt.tzinfo is not None:
                now = now.replace(tzinfo=timezone.utc)
            return round((now - dt).total_seconds() / 3600.0, 1)
        except Exception:
            return None

    # ── aggregation ─────────────────────────────────────────────────────

    @classmethod
    def _extract_current_mood(
        cls,
        raw_data: Dict[str, Any],
        health_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        latest = raw_data.get("latest_mood") or {}
        if not isinstance(latest, dict):
            latest = (
                latest.model_dump(by_alias=True)
                if hasattr(latest, "model_dump")
                else {}
            )
        mood = (
            latest.get("current_mood")
            or health_param_value(
                health_params,
                "health_params_mood_last_data_date",
                "mood_last_data_date",
            )
        )
        if isinstance(mood, str):
            mood = mood.strip().lower()
        display = latest.get("mood_display_name")
        mood_date = (
            latest.get("current_mood_date")
            or health_param_value(
                health_params,
                "health_params_mood_last_data_date",
                "mood_last_data_date",
            )
        )
        score = _mood_score(mood)
        notes = bool(latest.get("mood_notes"))
        return {
            "mood": mood,
            "display_name": _display_name(mood, display),
            "score": score,
            "mood_date": str(mood_date) if mood_date else None,
            "mood_date_local": latest.get("mood_date_local"),
            "notes": notes,
        }

    @classmethod
    def _aggregate_entries(
        cls,
        *,
        historical_snapshots: List[Any],
        raw_data: Dict[str, Any],
        today: Optional[date],
        current: Optional[Dict[str, Any]],
    ) -> List[MoodEntry]:
        by_date: Dict[date, MoodEntry] = {}

        history = raw_data.get("moods_7d") or []
        if isinstance(history, list):
            for item in history:
                if not isinstance(item, dict):
                    continue
                date_raw = item.get("current_mood_date") or item.get("mood_date") or item.get("date")
                if not date_raw:
                    continue
                try:
                    d = date.fromisoformat(str(date_raw).split("T")[0])
                except Exception:
                    continue
                mood = item.get("current_mood")
                if isinstance(mood, str):
                    mood = mood.strip().lower()
                score = _mood_score(mood) or _safe_float(item.get("mood_score"))
                by_date[d] = MoodEntry(
                    entry_date=d,
                    mood=mood,
                    score=float(score) if score is not None else None,
                    display_name=item.get("mood_display_name"),
                    has_notes=bool(item.get("mood_has_notes") or item.get("note") or item.get("notes")),
                )

        if today is not None and current and current.get("mood"):
            score = current.get("score")
            by_date[today] = MoodEntry(
                entry_date=today,
                mood=current.get("mood"),
                score=float(score) if score is not None else _mood_score(current.get("mood")),
                display_name=current.get("display_name"),
                has_notes=bool(current.get("has_notes")),
            )

        return sorted(by_date.values(), key=lambda e: e.entry_date)

    @classmethod
    def _window_entries(
        cls, entries: List[MoodEntry], today: Optional[date], days: int
    ) -> List[MoodEntry]:
        if today is None:
            return list(entries[-days:])
        cutoff = today - timedelta(days=days)
        return [e for e in entries if cutoff < e.entry_date <= today]

    @classmethod
    def _entry_counts(
        cls, entries: List[MoodEntry], today: Optional[date]
    ) -> Dict[int, int]:
        out = {}
        for n in MOOD_FREQUENCY_LOOKBACK_DAYS:
            out[n] = len(cls._window_entries(entries, today, n))
        return out

    # ── signal builders ─────────────────────────────────────────────────

    @classmethod
    def _baseline(
        cls, entries: List[MoodEntry], today: Optional[date], summary: MoodSummaryMetrics
    ) -> Dict[str, Any]:
        window = cls._window_entries(entries, today, MOOD_BASELINE_LOOKBACK_DAYS)
        # exclude today for baseline
        hist = [e for e in window if today is None or e.entry_date < today]
        scores = [float(e.score) for e in hist if e.score is not None]
        if len(scores) < MOOD_MIN_BASELINE_ENTRIES:
            return {"status": "insufficient_data", "entries": len(scores)}
        base_score = round(_mean(scores), 2)
        base_mood = _score_to_mood(base_score)
        cur = summary.current_mood_score
        delta = None
        vs = None
        status = "stable"
        if cur is not None:
            delta = int(cur) - int(round(base_score))
            if delta >= MOOD_IMPROVED_DELTA:
                vs = "improved"
                status = "improved"
            elif delta <= MOOD_DECLINED_DELTA:
                vs = "declined"
                status = "declined"
            else:
                vs = "stable"
                status = "stable"
        return {
            "status": status,
            "baseline_mood": base_mood,
            "baseline_mood_score": base_score,
            "mood_delta": delta,
            "mood_vs_baseline": vs,
            "entries": len(scores),
        }

    @classmethod
    def _trend(
        cls, entries: List[MoodEntry], today: Optional[date]
    ) -> Dict[str, Any]:
        window = cls._window_entries(entries, today, MOOD_TREND_LOOKBACK_DAYS)
        scores = [float(e.score) for e in window if e.score is not None]
        if len(scores) < MOOD_MIN_TREND_ENTRIES:
            return {"status": "insufficient_data"}
        mid = len(scores) // 2
        first = scores[:mid] or scores[:1]
        second = scores[mid:] or scores[-1:]
        first_avg = sum(first) / len(first)
        second_avg = sum(second) / len(second)
        delta = second_avg - first_avg
        if delta >= 0.5:
            status = "improving"
        elif delta <= -0.5:
            status = "declining"
        else:
            status = "stable"
        return {
            "status": status,
            "positive_trend": status == "improving",
            "negative_trend": status == "declining",
            "stable_trend": status == "stable",
            "first_half_avg": round(first_avg, 2),
            "second_half_avg": round(second_avg, 2),
        }

    @classmethod
    def _pattern(
        cls, entries: List[MoodEntry], today: Optional[date]
    ) -> Dict[str, Any]:
        window = list(reversed(cls._window_entries(entries, today, 14)))
        if not window:
            return {"status": "insufficient_data"}

        def streak(pred) -> int:
            n = 0
            for e in window:
                if e.mood and pred(e.mood):
                    n += 1
                else:
                    break
            return n

        pos = streak(lambda m: m in MOOD_POSITIVE)
        neg = streak(lambda m: m in MOOD_NEGATIVE)
        happy = streak(lambda m: m == "happy")
        sad = streak(lambda m: m == "sad")
        okay = streak(lambda m: m == "okay")
        amazing = streak(lambda m: m == "amazing")

        if max(pos, neg, happy, sad, okay, amazing) >= MOOD_STREAK_MIN_DAYS:
            status = "streak"
        elif window:
            status = "mixed"
        else:
            status = "insufficient_data"
        return {
            "status": status,
            "positive_streak": pos,
            "negative_streak": neg,
            "happy_streak": happy,
            "sad_streak": sad,
            "okay_streak": okay,
            "amazing_streak": amazing,
        }

    @classmethod
    def _stability(
        cls, entries: List[MoodEntry], today: Optional[date]
    ) -> Dict[str, Any]:
        window = cls._window_entries(entries, today, MOOD_TREND_LOOKBACK_DAYS)
        scores = [float(e.score) for e in window if e.score is not None]
        if len(scores) < MOOD_MIN_TREND_ENTRIES:
            return {"status": "insufficient_data", "volatile_mood": False}
        std = _stdev(scores)
        if std is None:
            return {"status": "insufficient_data", "volatile_mood": False}
        if std >= MOOD_VOLATILITY_HIGH_STD:
            status = "volatile"
            volatile = True
        elif std >= MOOD_VOLATILITY_MEDIUM_STD:
            status = "mixed"
            volatile = False
        elif std <= 0.35:
            status = "very_stable"
            volatile = False
        else:
            status = "stable"
            volatile = False
        return {
            "status": status,
            "std": round(std, 2),
            "volatile_mood": volatile,
        }

    @classmethod
    def _frequency(
        cls, counts: Dict[int, int], last_logged_days_ago: Optional[int]
    ) -> Dict[str, Any]:
        n7 = counts.get(7, 0)
        if n7 >= MOOD_LOGGING_HIGH_ENTRIES_7D:
            status = "high"
        elif n7 >= MOOD_LOGGING_MED_ENTRIES_7D:
            status = "medium"
        else:
            status = "low"
        return {"status": status}

    @classmethod
    def _change(
        cls,
        entries: List[MoodEntry],
        today: Optional[date],
        summary: MoodSummaryMetrics,
        baseline: Dict[str, Any],
    ) -> Dict[str, Any]:
        if today is None or not summary.has_mood:
            return {"status": "insufficient_data"}
        today_e = next((e for e in entries if e.entry_date == today), None)
        prev = [e for e in entries if e.entry_date < today]
        prev_e = prev[-1] if prev else None
        # Tri-state semantics:
        #   None  = chưa đủ dữ liệu để so sánh (chỉ có 1 entry / thiếu score
        #           / không có entry hôm nay / không có entry trước đó)
        #   False = đã so sánh và biết chắc KHÔNG có thay đổi/cải thiện/suy giảm
        #   True  = đã so sánh và biết chắc CÓ thay đổi/cải thiện/suy giảm
        changed = improved = declined = None
        if today_e and prev_e and today_e.score is not None and prev_e.score is not None:
            delta = today_e.score - prev_e.score
            if delta != 0:
                changed = abs(delta) >= 1
                improved = delta >= MOOD_IMPROVED_DELTA
                declined = delta <= MOOD_DECLINED_DELTA
            else:
                changed = False
                improved = False
                declined = False

        # largest jump/drop in 14d
        #   largest_pos = biggest day-over-day INCREASE only (None nếu không có)
        #   largest_neg = biggest day-over-day DECREASE only (None nếu không có)
        # Nếu mọi jumps đều = 0 → trả None cho cả 2 (không có thay đổi nào)
        window = cls._window_entries(entries, today, 14)
        pos_jumps = []
        neg_jumps = []
        for i in range(1, len(window)):
            a, b = window[i - 1].score, window[i].score
            if a is None or b is None:
                continue
            d = b - a
            if d > 0:
                pos_jumps.append(d)
            elif d < 0:
                neg_jumps.append(d)
        largest_pos = max(pos_jumps) if pos_jumps else None
        largest_neg = min(neg_jumps) if neg_jumps else None

        status = "stable"
        # Chỉ set status khi đã có dữ liệu để so sánh (None = không đủ)
        if improved is True:
            status = "improved_today"
        elif declined is True:
            status = "declined_today"
        elif changed is True:
            status = "changed_today"
        elif improved is None or declined is None or changed is None:
            # Không đủ dữ liệu để kết luận ổn định hay thay đổi
            status = "insufficient_data"
        return {
            "status": status,
            "mood_changed_today": changed,
            "mood_improved_today": improved,
            "mood_declined_today": declined,
            "largest_positive_jump": round(largest_pos, 1) if largest_pos is not None else None,
            "largest_negative_drop": round(largest_neg, 1) if largest_neg is not None else None,
        }

    @classmethod
    def _correlation(
        cls,
        summary: MoodSummaryMetrics,
        *,
        sleep_context: Optional[Dict[str, Any]],
        energy_context: Optional[Dict[str, Any]],
        hr_context: Optional[Dict[str, Any]],
        calendar_context: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Only emit flags when supporting evidence exists — never invent causality."""
        if not summary.has_mood or summary.current_mood is None:
            return None
        mood = summary.current_mood
        score = summary.current_mood_score
        flags: Dict[str, bool] = {}
        evidence: List[str] = []

        sleep = sleep_context or {}
        energy = energy_context or {}
        hr = hr_context or {}
        cal = calendar_context or {}

        good_sleep = sleep.get("quality") in ("high", "very_high", "ok") or (
            isinstance(sleep.get("last_night_h"), (int, float))
            and sleep.get("last_night_h") >= 7
        )
        poor_sleep = sleep.get("level") in ("moderate", "severe") or (
            isinstance(sleep.get("debt_hours"), (int, float))
            and sleep.get("debt_hours") >= 1.5
        )
        good_activity = (
            isinstance(energy.get("total_active_energy"), (int, float))
            and energy.get("total_active_energy") > 0
            and energy.get("level") in ("ok", None)
        )
        rest_day = energy.get("level") in ("not_started", "mild") and (
            energy.get("total_active_energy") is not None
            and energy.get("total_active_energy") < 200
        )
        high_stress = bool(hr.get("stress_high"))
        heavy_meetings = (
            isinstance(cal.get("meeting_minutes_today"), (int, float))
            and cal.get("meeting_minutes_today") >= 180
        )

        positive = mood in MOOD_POSITIVE or (score is not None and score >= 4)
        negative = mood in MOOD_NEGATIVE or (score is not None and score <= 2)

        if positive and good_sleep:
            flags["positive_mood_after_good_sleep"] = True
            evidence.append("good_sleep+positive_mood")
        if positive and good_activity:
            flags["positive_mood_after_exercise"] = True
            evidence.append("activity+positive_mood")
        if positive and rest_day:
            flags["positive_mood_after_rest_day"] = True
            evidence.append("rest_day+positive_mood")
        if negative and poor_sleep:
            flags["low_mood_after_poor_sleep"] = True
            evidence.append("poor_sleep+low_mood")
        if negative and heavy_meetings:
            flags["low_mood_after_heavy_meeting_day"] = True
            evidence.append("heavy_meetings+low_mood")
        if negative and high_stress:
            flags["low_mood_after_high_stress"] = True
            evidence.append("high_stress+low_mood")

        if not flags:
            return {"status": "none", "flags": {}, "evidence": []}
        return {"status": "present", "flags": flags, "evidence": evidence}

    @classmethod
    def _detect_anomalies(
        cls, entries: List[MoodEntry], today: Optional[date]
    ) -> List[MoodAnomaly]:
        anomalies: List[MoodAnomaly] = []
        window = cls._window_entries(entries, today, 14)
        prev = None
        for e in window:
            if e.score is None:
                continue
            if e.score <= 1:
                anomalies.append(
                    MoodAnomaly(
                        kind="very_low_mood",
                        day_date=e.entry_date.isoformat(),
                        value=e.score,
                        evidence={"mood": e.mood},
                    )
                )
            if prev is not None and abs(e.score - prev) >= 3:
                anomalies.append(
                    MoodAnomaly(
                        kind="large_mood_swing",
                        day_date=e.entry_date.isoformat(),
                        value=e.score - prev,
                        threshold=3.0,
                        evidence={"from": prev, "to": e.score, "mood": e.mood},
                    )
                )
            prev = e.score
        return anomalies

    @classmethod
    def _build_overall(
        cls,
        *,
        summary: MoodSummaryMetrics,
        trend: Dict[str, Any],
        stability: Dict[str, Any],
        frequency: Dict[str, Any],
        has_mood_today: bool,
    ) -> MoodOverall:
        if not has_mood_today and summary.entries_last_7_days == 0:
            return MoodOverall(status="insufficient_data", confidence="low")

        cur = summary.current_mood_score
        if cur is None:
            cur_pts = 50.0
        else:
            cur_pts = (cur / 5.0) * 100.0

        t = trend.get("status")
        if t == "improving":
            trend_pts = 90.0
        elif t == "stable":
            trend_pts = 75.0
        elif t == "declining":
            trend_pts = 40.0
        else:
            trend_pts = 60.0

        s = stability.get("status")
        if s in ("very_stable", "stable"):
            stab_pts = 85.0
        elif s == "mixed":
            stab_pts = 60.0
        elif s == "volatile":
            stab_pts = 40.0
        else:
            stab_pts = 60.0

        f = frequency.get("status")
        if f == "high":
            log_pts = 90.0
        elif f == "medium":
            log_pts = 70.0
        else:
            log_pts = 45.0

        score = round(
            cur_pts * MOOD_OVERALL_WEIGHT_CURRENT
            + trend_pts * MOOD_OVERALL_WEIGHT_TREND
            + stab_pts * MOOD_OVERALL_WEIGHT_STABILITY
            + log_pts * MOOD_OVERALL_WEIGHT_LOGGING,
            1,
        )
        if not has_mood_today:
            status = "insufficient_data"
        elif cur is not None and cur <= 2:
            status = "low"
        elif cur is not None and cur >= 4:
            status = "positive"
        else:
            status = "neutral"

        if summary.entries_last_7_days >= 5:
            confidence = "high"
        elif summary.entries_last_7_days >= 3:
            confidence = "medium"
        else:
            confidence = "low"

        return MoodOverall(status=status, score=score, confidence=confidence)
