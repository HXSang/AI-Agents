"""Health Score signal engine — techplan health-score.md.

Uses only (date, timezone, healthScore). Null scores are ignored.
Produces deterministic fired signals + optional aggregate theme for the Insight Engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from insights.health.health_processor.health_score.constants import (
    HEALTH_SCORE_ANOMALY_MIN_ABSOLUTE_DEVIATION,
    HEALTH_SCORE_ANOMALY_MIN_RELATIVE_DEVIATION,
    HEALTH_SCORE_BASELINE_LOOKBACK_DAYS,
    HEALTH_SCORE_CRITICAL_THRESHOLD,
    HEALTH_SCORE_DROP_CRITICAL_THRESHOLD,
    HEALTH_SCORE_DROP_HIGH_THRESHOLD,
    HEALTH_SCORE_DROP_LOW_THRESHOLD,
    HEALTH_SCORE_DROP_MEDIUM_THRESHOLD,
    HEALTH_SCORE_LONG_TERM_BASELINE_LOOKBACK_DAYS,
    HEALTH_SCORE_LOW_THRESHOLD,
    HEALTH_SCORE_MINIMUM_DATA_POINTS,
    HEALTH_SCORE_PRIORITY,
    HEALTH_SCORE_RECOVERY_THRESHOLD,
    HEALTH_SCORE_STABLE_MAX_DELTA,
    HEALTH_SCORE_TREND_MIN_CONSECUTIVE_DAYS,
    HEALTH_SCORE_TREND_MIN_DEVIATION,
    HEALTH_SCORE_VOLATILITY_LOOKBACK_DAYS,
    HEALTH_SCORE_VOLATILITY_RANGE_THRESHOLD,
)
from insights.schemas.processed_context import (
    HealthScoreAggregate,
    HealthScoreFiredSignal,
    HealthScoreOverall,
    HealthScoreSignal,
    HealthScoreSummaryMetrics,
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


def _parse_date(raw: Any) -> Optional[date]:
    if raw is None:
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


def _priority(key: str) -> float:
    return float(HEALTH_SCORE_PRIORITY.get(key, 0.0))


@dataclass
class ScorePoint:
    day: date
    score: float
    timezone: Optional[str] = None


class HealthScoreSignalProcessor:
    """Configurable Health Score Signal Engine (healthScore-only)."""

    @classmethod
    def process(
        cls,
        health_params: Optional[Dict[str, Any]] = None,
        historical_snapshots: Optional[List[Any]] = None,
        time_data: Optional[Dict[str, Any]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> HealthScoreSignal:
        try:
            return cls._process_unsafe(
                health_params=health_params or {},
                historical_snapshots=historical_snapshots or [],
                time_data=time_data or {},
                raw_data=raw_data or {},
            )
        except Exception:
            try:
                from utils.logger import logger as _logger

                _logger.warning("HealthScoreSignalProcessor failed", exc_info=True)
            except Exception:
                pass
            return HealthScoreSignal()

    @classmethod
    def _process_unsafe(
        cls,
        health_params: Dict[str, Any],
        historical_snapshots: List[Any],
        time_data: Dict[str, Any],
        raw_data: Dict[str, Any],
    ) -> HealthScoreSignal:
        today = cls._resolve_today(time_data, raw_data)
        tz_name = cls._resolve_timezone(time_data, raw_data)
        points = cls._collect_points(
            raw_data=raw_data,
            today=today,
            timezone=tz_name,
        )
        if not points:
            return HealthScoreSignal(
                overall=HealthScoreOverall(
                    status="insufficient_data", confidence="low"
                )
            )

        # Chronological ascending; current = last valid day ≤ today (prefer today)
        points.sort(key=lambda p: p.day)
        current = cls._pick_current(points, today)
        if current is None:
            return HealthScoreSignal(
                overall=HealthScoreOverall(
                    status="insufficient_data", confidence="low"
                )
            )

        prior_points = [p for p in points if p.day < current.day]
        previous = prior_points[-1] if prior_points else None
        delta = (
            current.score - previous.score if previous is not None else None
        )

        baseline_7d = cls._baseline(
            prior_points, HEALTH_SCORE_BASELINE_LOOKBACK_DAYS
        )
        baseline_30d = cls._baseline(
            prior_points,
            HEALTH_SCORE_LONG_TERM_BASELINE_LOOKBACK_DAYS,
            min_points=HEALTH_SCORE_LONG_TERM_BASELINE_LOOKBACK_DAYS,
        )
        baseline = baseline_7d

        window_days = HEALTH_SCORE_BASELINE_LOOKBACK_DAYS
        series_window = [
            p
            for p in points
            if (current.day - timedelta(days=window_days - 1)) <= p.day <= current.day
        ]
        coverage = (
            round(100.0 * len(series_window) / window_days, 1) if window_days else None
        )

        summary = HealthScoreSummaryMetrics(
            current_score=round(current.score, 2),
            previous_score=round(previous.score, 2) if previous else None,
            current_date=current.day.isoformat(),
            previous_date=previous.day.isoformat() if previous else None,
            timezone=tz_name or current.timezone,
            delta=round(delta, 2) if delta is not None else None,
            baseline_7d=round(baseline_7d, 2) if baseline_7d is not None else None,
            baseline_30d=round(baseline_30d, 2) if baseline_30d is not None else None,
            days_measured=len(points),
            window_days=window_days,
            coverage_pct=coverage,
            series=[
                {
                    "date": p.day.isoformat(),
                    "time_data_timezone": p.timezone or tz_name,
                    "healthScore": round(p.score, 2),
                }
                for p in series_window
            ],
        )

        fired: List[HealthScoreFiredSignal] = []
        fired.extend(cls._detect_state(current.score))
        fired.extend(
            cls._detect_change(
                current_score=current.score,
                previous_score=previous.score if previous else None,
                delta=delta,
            )
        )
        fired.extend(
            cls._detect_trend(
                points_to_current=[p for p in points if p.day <= current.day],
                current=current,
                baseline=baseline,
            )
        )
        fired.extend(
            cls._detect_anomaly(
                current_score=current.score,
                baseline=baseline,
                prior_count=len(prior_points),
            )
        )
        fired.extend(
            cls._detect_stable(delta=delta, prior_count=len(prior_points))
        )
        fired.extend(
            cls._detect_volatility(
                [p for p in points if p.day <= current.day],
                current.day,
            )
        )

        # CRITICAL takes precedence over LOW
        keys = {s.key for s in fired}
        if "HEALTH_SCORE_CRITICAL" in keys:
            fired = [s for s in fired if s.key != "HEALTH_SCORE_LOW"]

        fired.sort(key=lambda s: (-s.priority, s.key))
        aggregate = cls._aggregate(
            fired=fired,
            current_score=current.score,
            previous_score=previous.score if previous else None,
            baseline=baseline,
            delta=delta,
        )
        overall = cls._build_overall(fired, current.score, len(prior_points))

        return HealthScoreSignal(
            summary=summary,
            signals=fired,
            aggregate=aggregate,
            overall=overall,
        )

    # ── collectors ────────────────────────────────────────────────────────

    @classmethod
    def _collect_points(
        cls,
        *,
        raw_data: Dict[str, Any],
        today: date,
        timezone: Optional[str],
    ) -> List[ScorePoint]:
        by_day: Dict[date, ScorePoint] = {}

        def _put(day: Optional[date], score: Optional[float], tz: Optional[str] = None):
            if day is None or score is None:
                return
            by_day[day] = ScorePoint(day=day, score=float(score), timezone=tz or timezone)

        bal = raw_data.get("balance_score")
        if hasattr(bal, "model_dump"):
            try:
                bal = bal.model_dump(by_alias=True)
            except Exception:
                try:
                    bal = bal.model_dump()
                except Exception:
                    bal = None
        if isinstance(bal, dict):
            score_raw = bal.get("balance_health_score")
            if score_raw is None:
                score_raw = bal.get("healthScore")
            score = _safe_float(score_raw)
            d = _parse_date(
                bal.get("balance_score_date")
                or bal.get("date")
                or bal.get("balance_date")
            )
            tz_field = bal.get("time_data_timezone") or bal.get("timezone")
            _put(d, score, tz_field)

        # 2) Past 30 days (excluding today) from /api/balance.
        for item in raw_data.get("balance_scores_30d") or []:
            entry = item
            if hasattr(item, "model_dump"):
                try:
                    entry = entry.model_dump(by_alias=True)
                except Exception:
                    try:
                        entry = item.model_dump()
                    except Exception:
                        continue
            if not isinstance(entry, dict):
                continue
            score_raw = entry.get("balance_health_score")
            if score_raw is None:
                score_raw = entry.get("healthScore")
            score = _safe_float(score_raw)
            d = _parse_date(
                entry.get("balance_score_date")
                or entry.get("date")
                or entry.get("balance_date")
            )
            tz_field = entry.get("time_data_timezone") or entry.get("timezone")
            _put(d, score, tz_field)

        return list(by_day.values())

    @staticmethod
    def _pick_current(points: List[ScorePoint], today: date) -> Optional[ScorePoint]:
        eligible = [p for p in points if p.day <= today]
        if not eligible:
            return None
        # Prefer exact today; else most recent prior day
        for p in reversed(eligible):
            if p.day == today:
                return p
        return eligible[-1]

    @staticmethod
    def _baseline(
        prior: List[ScorePoint],
        lookback_days: int,
        min_points: Optional[int] = None,
    ) -> Optional[float]:
        if not prior:
            return None
        end = prior[-1].day
        start = end - timedelta(days=lookback_days - 1)
        window = [p.score for p in prior if start <= p.day <= end]
        threshold = (
            lookback_days if min_points is None else min_points
        )
        if len(window) < threshold:
            return None
        return _mean(window)

    # ── detectors ─────────────────────────────────────────────────────────

    @classmethod
    def _detect_state(cls, score: float) -> List[HealthScoreFiredSignal]:
        out: List[HealthScoreFiredSignal] = []
        if score < HEALTH_SCORE_CRITICAL_THRESHOLD:
            out.append(
                HealthScoreFiredSignal(
                    key="HEALTH_SCORE_CRITICAL",
                    category="state",
                    severity="CRITICAL",
                    priority=_priority("HEALTH_SCORE_CRITICAL"),
                    evidence={
                        "currentScore": score,
                        "threshold": HEALTH_SCORE_CRITICAL_THRESHOLD,
                    },
                )
            )
        elif score < HEALTH_SCORE_LOW_THRESHOLD:
            out.append(
                HealthScoreFiredSignal(
                    key="HEALTH_SCORE_LOW",
                    category="state",
                    severity="LOW",
                    priority=_priority("HEALTH_SCORE_LOW"),
                    evidence={
                        "currentScore": score,
                        "threshold": HEALTH_SCORE_LOW_THRESHOLD,
                    },
                )
            )
        return out

    @classmethod
    def _detect_change(
        cls,
        *,
        current_score: float,
        previous_score: Optional[float],
        delta: Optional[float],
    ) -> List[HealthScoreFiredSignal]:
        out: List[HealthScoreFiredSignal] = []
        if delta is None or previous_score is None:
            return out

        if delta <= -HEALTH_SCORE_DROP_LOW_THRESHOLD:
            abs_drop = abs(delta)
            if abs_drop >= HEALTH_SCORE_DROP_CRITICAL_THRESHOLD:
                severity = "CRITICAL"
            elif abs_drop >= HEALTH_SCORE_DROP_HIGH_THRESHOLD:
                severity = "HIGH"
            elif abs_drop >= HEALTH_SCORE_DROP_MEDIUM_THRESHOLD:
                severity = "MEDIUM"
            else:
                severity = "LOW"
            out.append(
                HealthScoreFiredSignal(
                    key="HEALTH_SCORE_DROP",
                    category="change",
                    severity=severity,
                    priority=_priority("HEALTH_SCORE_DROP"),
                    evidence={
                        "currentScore": current_score,
                        "previousScore": previous_score,
                        "delta": delta,
                        "threshold": -HEALTH_SCORE_DROP_LOW_THRESHOLD,
                    },
                )
            )

        if (
            delta >= HEALTH_SCORE_RECOVERY_THRESHOLD
            and previous_score < HEALTH_SCORE_LOW_THRESHOLD
        ):
            out.append(
                HealthScoreFiredSignal(
                    key="HEALTH_SCORE_RECOVERY",
                    category="change",
                    priority=_priority("HEALTH_SCORE_RECOVERY"),
                    evidence={
                        "currentScore": current_score,
                        "previousScore": previous_score,
                        "delta": delta,
                        "recoveryThreshold": HEALTH_SCORE_RECOVERY_THRESHOLD,
                        "previousBelowLow": True,
                    },
                )
            )
        return out

    @classmethod
    def _detect_trend(
        cls,
        *,
        points_to_current: List[ScorePoint],
        current: ScorePoint,
        baseline: Optional[float],
    ) -> List[HealthScoreFiredSignal]:
        out: List[HealthScoreFiredSignal] = []
        if len(points_to_current) < HEALTH_SCORE_MINIMUM_DATA_POINTS:
            return out
        if baseline is None:
            return out

        need = HEALTH_SCORE_TREND_MIN_CONSECUTIVE_DAYS
        if len(points_to_current) < need:
            return out

        window = points_to_current[-need:]
        downs = all(
            window[i].score < window[i - 1].score for i in range(1, len(window))
        )
        ups = all(
            window[i].score > window[i - 1].score for i in range(1, len(window))
        )
        deviation = abs(current.score - baseline)
        if deviation < HEALTH_SCORE_TREND_MIN_DEVIATION:
            return out

        if downs and current.score < baseline:
            out.append(
                HealthScoreFiredSignal(
                    key="HEALTH_SCORE_TREND_DOWN",
                    category="trend",
                    priority=_priority("HEALTH_SCORE_TREND_DOWN"),
                    evidence={
                        "consecutiveDays": need,
                        "scores": [p.score for p in window],
                        "baseline": baseline,
                        "deviation": round(deviation, 2),
                    },
                )
            )
        elif ups and current.score > baseline:
            out.append(
                HealthScoreFiredSignal(
                    key="HEALTH_SCORE_TREND_UP",
                    category="trend",
                    priority=_priority("HEALTH_SCORE_TREND_UP"),
                    evidence={
                        "consecutiveDays": need,
                        "scores": [p.score for p in window],
                        "baseline": baseline,
                        "deviation": round(deviation, 2),
                    },
                )
            )
        return out

    @classmethod
    def _detect_anomaly(
        cls,
        *,
        current_score: float,
        baseline: Optional[float],
        prior_count: int,
    ) -> List[HealthScoreFiredSignal]:
        if baseline is None or prior_count < HEALTH_SCORE_MINIMUM_DATA_POINTS:
            return []
        abs_dev = abs(current_score - baseline)
        rel_dev = abs_dev / baseline if baseline else 0.0
        if (
            abs_dev >= HEALTH_SCORE_ANOMALY_MIN_ABSOLUTE_DEVIATION
            or rel_dev >= HEALTH_SCORE_ANOMALY_MIN_RELATIVE_DEVIATION
        ):
            return [
                HealthScoreFiredSignal(
                    key="HEALTH_SCORE_ANOMALY",
                    category="pattern",
                    priority=_priority("HEALTH_SCORE_ANOMALY"),
                    evidence={
                        "currentScore": current_score,
                        "baseline": baseline,
                        "absoluteDeviation": round(abs_dev, 2),
                        "relativeDeviation": round(rel_dev, 4),
                    },
                )
            ]
        return []

    @classmethod
    def _detect_stable(
        cls, *, delta: Optional[float], prior_count: int
    ) -> List[HealthScoreFiredSignal]:
        if delta is None or prior_count < 1:
            return []
        if abs(delta) <= HEALTH_SCORE_STABLE_MAX_DELTA:
            return [
                HealthScoreFiredSignal(
                    key="HEALTH_SCORE_STABLE",
                    category="state",
                    priority=_priority("HEALTH_SCORE_STABLE"),
                    evidence={
                        "delta": delta,
                        "maxDelta": HEALTH_SCORE_STABLE_MAX_DELTA,
                        "supporting_only": True,
                    },
                )
            ]
        return []

    @classmethod
    def _detect_volatility(
        cls, points: List[ScorePoint], current_day: date
    ) -> List[HealthScoreFiredSignal]:
        lookback = HEALTH_SCORE_VOLATILITY_LOOKBACK_DAYS
        start = current_day - timedelta(days=lookback - 1)
        window = [p for p in points if start <= p.day <= current_day]
        if len(window) < HEALTH_SCORE_MINIMUM_DATA_POINTS:
            return []
        scores = [p.score for p in window]
        score_range = max(scores) - min(scores)
        if score_range < HEALTH_SCORE_VOLATILITY_RANGE_THRESHOLD:
            return []

        # Distinguish from a consistent mono trend
        diffs = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
        all_down = all(d < 0 for d in diffs)
        all_up = all(d > 0 for d in diffs)
        if all_down or all_up:
            return []

        return [
            HealthScoreFiredSignal(
                key="HEALTH_SCORE_VOLATILITY_HIGH",
                category="pattern",
                priority=_priority("HEALTH_SCORE_VOLATILITY_HIGH"),
                evidence={
                    "range": round(score_range, 2),
                    "threshold": HEALTH_SCORE_VOLATILITY_RANGE_THRESHOLD,
                    "lookbackDays": lookback,
                    "min": min(scores),
                    "max": max(scores),
                    "scores": scores,
                },
            )
        ]

    @classmethod
    def _aggregate(
        cls,
        *,
        fired: List[HealthScoreFiredSignal],
        current_score: float,
        previous_score: Optional[float],
        baseline: Optional[float],
        delta: Optional[float],
    ) -> Optional[HealthScoreAggregate]:
        keys = [s.key for s in fired]
        keyset = set(keys)
        decline_keys = {
            "HEALTH_SCORE_CRITICAL",
            "HEALTH_SCORE_DROP",
            "HEALTH_SCORE_TREND_DOWN",
            "HEALTH_SCORE_ANOMALY",
            "HEALTH_SCORE_LOW",
        }
        improve_keys = {
            "HEALTH_SCORE_RECOVERY",
            "HEALTH_SCORE_TREND_UP",
        }
        decline_hits = sorted(keyset & decline_keys)
        improve_hits = sorted(keyset & improve_keys)

        theme = None
        band = None
        theme_signals: List[str] = []
        if len(decline_hits) >= 2:
            theme = "HEALTH_SCORE_DECLINE"
            theme_signals = decline_hits
            top = max((_priority(k) for k in decline_hits), default=0)
            band = "high" if top >= 0.85 else "medium"
        elif len(improve_hits) >= 1 and not (
            "HEALTH_SCORE_CRITICAL" in keyset or "HEALTH_SCORE_DROP" in keyset
        ):
            theme = "HEALTH_SCORE_IMPROVEMENT"
            theme_signals = improve_hits
            band = "medium"

        if theme is None:
            return None
        return HealthScoreAggregate(
            theme=theme,
            priority=band,
            current_score=round(current_score, 2),
            previous_score=round(previous_score, 2) if previous_score is not None else None,
            baseline=round(baseline, 2) if baseline is not None else None,
            delta=round(delta, 2) if delta is not None else None,
            signals=theme_signals,
        )

    @classmethod
    def _build_overall(
        cls,
        fired: List[HealthScoreFiredSignal],
        current_score: float,
        prior_count: int,
    ) -> HealthScoreOverall:
        keys = {s.key for s in fired}
        confidence = (
            "high"
            if prior_count >= HEALTH_SCORE_BASELINE_LOOKBACK_DAYS
            else (
                "medium"
                if prior_count >= HEALTH_SCORE_MINIMUM_DATA_POINTS
                else "low"
            )
        )
        if "HEALTH_SCORE_CRITICAL" in keys:
            status = "critical"
        elif "HEALTH_SCORE_DROP" in keys or "HEALTH_SCORE_TREND_DOWN" in keys:
            status = "declining"
        elif "HEALTH_SCORE_RECOVERY" in keys or "HEALTH_SCORE_TREND_UP" in keys:
            status = "recovering" if "HEALTH_SCORE_RECOVERY" in keys else "improving"
        elif "HEALTH_SCORE_VOLATILITY_HIGH" in keys:
            status = "volatile"
        elif "HEALTH_SCORE_LOW" in keys:
            status = "low"
        elif "HEALTH_SCORE_STABLE" in keys:
            status = "stable"
        else:
            status = "stable" if prior_count else "insufficient_data"
        return HealthScoreOverall(
            status=status,
            score=round(current_score, 2),
            confidence=confidence,
        )

    @staticmethod
    def _resolve_today(
        time_data: Dict[str, Any], raw_data: Dict[str, Any]
    ) -> date:
        for key in ("today_date", "local_date", "date"):
            d = _parse_date(time_data.get(key))
            if d:
                return d
        cur = time_data.get("current_time") or raw_data.get("current_time")
        d = _parse_date(cur)
        if d:
            return d
        return date.today()

    @staticmethod
    def _resolve_timezone(
        time_data: Dict[str, Any], raw_data: Dict[str, Any]
    ) -> Optional[str]:
        tz = time_data.get("timezone") or raw_data.get("timezone")
        if tz is None:
            return None
        return str(tz)
