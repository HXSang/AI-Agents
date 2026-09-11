import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from insights.health.canonical_field_mapping import CanonicalField
from insights.schemas.processed_context import DailySnapshotData
from insights.schemas.processed_context import HistoricalTrendsBlock

logger = logging.getLogger(__name__)


class HistoricalTrendsProcessor:
    @staticmethod
    def _snapshot_date(item: Any) -> Optional[date]:
        raw = (
            item.get("snapshot_date")
            if isinstance(item, dict)
            else getattr(item, "snapshot_date", None)
        )
        if not raw:
            return None
        try:
            if isinstance(raw, date) and not isinstance(raw, datetime):
                return raw
            return datetime.fromisoformat(str(raw)).date()
        except Exception:
            return None

    @staticmethod
    def _snapshot_field(item: Any, field: str) -> Any:
        return item.get(field) if isinstance(item, dict) else getattr(item, field, None)

    @staticmethod
    def _compute_trend_string(values: List[float]) -> str:
        if len(values) < 4:
            return "stable"
        mid = len(values) // 2
        first_avg = sum(values[:mid]) / len(values[:mid])
        second_avg = sum(values[mid:]) / len(values[mid:])
        if first_avg == 0:
            return "stable"
        delta = second_avg - first_avg
        if delta >= 0.05 * abs(first_avg):
            return "improving"
        if delta <= -0.05 * abs(first_avg):
            return "declining"
        return "stable"

    @staticmethod
    def _parse_iso_date(raw: Any) -> Optional[date]:
        if raw is None:
            return None
        try:
            if isinstance(raw, date) and not isinstance(raw, datetime):
                return raw
            s = str(raw)
            return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
        except Exception:
            return None

    @staticmethod
    def _build_recent_snapshots_from_health_stats(
        today: date,
        today_health_stats: Optional[Dict[str, Any]],
        productivity_summary: Optional[Dict[str, Any]] = None,
        latest_mood: Optional[Dict[str, Any]] = None,
        productivity_summaries_7d: Optional[List[Dict[str, Any]]] = None,
        window_days: int = 7,
    ) -> Dict[str, "DailySnapshotData"]:
        by_date: Dict[date, Dict[str, Any]] = {}

        if today_health_stats:
            steps_entry = today_health_stats.get("STEPS") or {}
            for e in steps_entry.get("data") or []:
                d = HistoricalTrendsProcessor._parse_iso_date(e.get("date"))
                if d is None:
                    continue
                if (today - d).days < 0 or (today - d).days > window_days:
                    continue
                # Aggregate per-day in case multiple entries (e.g. partial sync).
                row = by_date.setdefault(d, {})
                row[CanonicalField.STEP_COUNT_ONE_DAY] = row.get(CanonicalField.STEP_COUNT_ONE_DAY, 0) + int(e.get("steps") or 0)
                if e.get("distance") is not None:
                    row["h_distance_km"] = round(
                        float(row.get("h_distance_km", 0) or 0)
                        + float(e.get("distance") or 0)
                        / 1000.0,
                        2,
                    )

            sleep_entry = today_health_stats.get("SLEEP") or {}
            for e in sleep_entry.get("data") or []:
                d = HistoricalTrendsProcessor._parse_iso_date(e.get("date"))
                if d is None:
                    continue
                if (today - d).days < 0 or (today - d).days > window_days:
                    continue
                row = by_date.setdefault(d, {})
                hours = e.get("hours") or e.get("totalSleepHours")
                if hours is not None:
                    row[CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS] = float(hours)
                quality = e.get("quality") or e.get("averageQuality")
                if quality is not None:
                    row[CanonicalField.SLEEP_QUALITY_SCORE_ONE_NIGHT] = quality

            hr_entry = today_health_stats.get("HR") or {}
            for e in hr_entry.get("data") or []:
                d = HistoricalTrendsProcessor._parse_iso_date(e.get("date"))
                if d is None:
                    continue
                if (today - d).days < 0 or (today - d).days > window_days:
                    continue
                row = by_date.setdefault(d, {})
                if e.get("restingHeartRate") is not None:
                    row[CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM] = e.get("restingHeartRate")

        # Productivity summaries — one row per date.
        for s in productivity_summaries_7d or []:
            d = HistoricalTrendsProcessor._parse_iso_date(s.get("date"))
            if d is None:
                continue
            if (today - d).days < 0 or (today - d).days > window_days:
                continue
            row = by_date.setdefault(d, {})
            if s.get("meetingCount") is not None:
                row[CanonicalField.TOTAL_EVENTS_ONE_DAY] = int(s["meetingCount"])
            if s.get("overallPercent") is not None:
                row["task_completion_rate"] = round(
                    float(s["overallPercent"]) / 100.0, 2
                )

        # Today-specific overrides from productivity_summary and latest_mood.
        today_row = by_date.setdefault(today, {})
        if productivity_summary:
            if productivity_summary.get("meetingCount") is not None:
                today_row[CanonicalField.TOTAL_EVENTS_ONE_DAY] = int(
                    productivity_summary["meetingCount"]
                )
            if productivity_summary.get("overallPercent") is not None:
                today_row["task_completion_rate"] = round(
                    float(productivity_summary["overallPercent"]) / 100.0, 2
                )
        if isinstance(latest_mood, dict):
            md = latest_mood.get("moodDate") or latest_mood.get("mood_date")
            d = HistoricalTrendsProcessor._parse_iso_date(md)
            if d == today and latest_mood.get("mood"):
                today_row["mood"] = latest_mood["mood"]

        return {
            d.isoformat(): DailySnapshotData(snapshot_date=d.isoformat(), **fields)
            for d, fields in by_date.items()
        }

    @staticmethod
    def process(
        historical_snapshots: List[Any],
        today: date,
        today_health_stats: Optional[Dict[str, Any]] = None,
        productivity_summary: Optional[Dict[str, Any]] = None,
        latest_mood: Optional[Dict[str, Any]] = None,
        productivity_summaries_7d: Optional[List[Dict[str, Any]]] = None,
        window_days: int = 4,
    ) -> Optional[HistoricalTrendsBlock]:
        if not historical_snapshots and not today_health_stats:
            return None
        synthesized = (
            HistoricalTrendsProcessor._build_recent_snapshots_from_health_stats(
                today=today,
                today_health_stats=today_health_stats,
                productivity_summary=productivity_summary,
                latest_mood=latest_mood,
                productivity_summaries_7d=productivity_summaries_7d,
                window_days=window_days,
            )
        )

        # Parse BE snapshots (strictly < today).
        valid_snapshots = []
        for r in historical_snapshots or []:
            d = HistoricalTrendsProcessor._snapshot_date(r)
            if d is not None and d < today:
                valid_snapshots.append((d, r))
        valid_snapshots.sort(key=lambda x: x[0], reverse=True)
        window_dates = [today - timedelta(days=i) for i in range(window_days)]
        window_iso = {d.isoformat() for d in window_dates}
        recent_days: List[DailySnapshotData] = []
        be_dates: set = set()
        for d, r in valid_snapshots:
            iso = d.isoformat()
            if iso not in window_iso:
                continue
            row_dict = r if isinstance(r, dict) else r.__dict__
            snapshot = DailySnapshotData(
                snapshot_date=iso,
                step_count_for_one_day=row_dict.get(CanonicalField.STEP_COUNT_ONE_DAY) or row_dict.get("h_steps"),
                active_minutes_for_one_day=row_dict.get(CanonicalField.ACTIVE_MINUTES_ONE_DAY) or row_dict.get("h_active_minutes"),
                total_calories_burned_for_one_day_kcal=row_dict.get(CanonicalField.TOTAL_CALORIES_BURNED_ONE_DAY_KCAL) or row_dict.get("h_calories_burned"),
                exercise_session_count_for_one_day=row_dict.get(CanonicalField.EXERCISE_SESSION_COUNT_ONE_DAY) or row_dict.get("h_exercise_sessions"),
                total_workout_duration_for_one_day_minutes=row_dict.get(CanonicalField.TOTAL_WORKOUT_DURATION_ONE_DAY_MINUTES) or row_dict.get("h_total_workout_min"),
                average_heart_rate_for_one_day_bpm=row_dict.get(CanonicalField.AVERAGE_HEART_RATE_ONE_DAY_BPM) or row_dict.get("h_avg_heart_rate"),
                resting_heart_rate_for_one_day_bpm=row_dict.get(CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM) or row_dict.get("h_resting_heart_rate"),
                sleep_duration_for_one_night_hours=row_dict.get(CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS) or row_dict.get("h_sleep_hours"),
                sleep_quality_score_for_one_night=row_dict.get(CanonicalField.SLEEP_QUALITY_SCORE_ONE_NIGHT) or row_dict.get("h_sleep_quality"),
                water_intake_for_one_day_liters=row_dict.get(CanonicalField.WATER_INTAKE_ONE_DAY_LITERS) or row_dict.get("h_water_liters"),
                health_score_for_one_day=row_dict.get(CanonicalField.HEALTH_SCORE_ONE_DAY) or row_dict.get("h_health_score"),
                sleep_score_for_one_night=row_dict.get(CanonicalField.SLEEP_SCORE_ONE_NIGHT) or row_dict.get("h_sleep_score"),
                steps_goal_completion_percentage_for_one_day=row_dict.get(CanonicalField.STEPS_GOAL_COMPLETION_PERCENTAGE_ONE_DAY) or row_dict.get("h_steps_goal_pct"),
                total_events_for_one_day=row_dict.get(CanonicalField.TOTAL_EVENTS_ONE_DAY) or row_dict.get("p_total_events"),
                meeting_duration_for_one_day_minutes=row_dict.get(CanonicalField.MEETING_DURATION_ONE_DAY_MINUTES) or row_dict.get("p_meeting_minutes"),
                longest_meeting_duration_for_one_day_minutes=row_dict.get(CanonicalField.LONGEST_MEETING_DURATION_ONE_DAY_MINUTES) or row_dict.get("p_longest_meeting_min"),
                work_span_for_one_day_minutes=row_dict.get(CanonicalField.WORK_SPAN_ONE_DAY_MINUTES) or row_dict.get("p_work_span_minutes"),
                focus_blocks_30min=row_dict.get("focus_blocks_30min") or row_dict.get("p_focus_blocks_30min"),
                focus_blocks_60min=row_dict.get("focus_blocks_60min") or row_dict.get("p_focus_blocks_60min"),
                task_completion_rate=row_dict.get("task_completion_rate") or row_dict.get("p_task_completion_rate"),
                reminders_completed=row_dict.get("reminders_completed") or row_dict.get("p_reminders_completed"),
                mood_primary_mood=row_dict.get("mood") or row_dict.get("mood_primary_mood") or row_dict.get("m_primary_mood"),
                mood_first_mood=row_dict.get("mood_first_mood") or row_dict.get("m_first_mood"),
                mood_score_avg=row_dict.get("mood_score_avg") or row_dict.get("m_mood_score_avg"),
                mood_score_min=row_dict.get("mood_score_min") or row_dict.get("m_mood_score_min"),
                mood_score_max=row_dict.get("mood_score_max") or row_dict.get("m_mood_score_max"),
                mood_variance=row_dict.get("mood_variance") or row_dict.get("m_mood_variance"),
                mood_sequence=row_dict.get("mood_sequence") or row_dict.get("m_mood_sequence"),
                mood_log_count=row_dict.get("mood_log_count") or row_dict.get("m_log_count"),
                mood_has_notes=row_dict.get("mood_has_notes") or row_dict.get("m_has_notes"),
                total_income=row_dict.get("total_income") or row_dict.get("f_total_income"),
                total_expense=row_dict.get("total_expense") or row_dict.get("f_total_expense"),
                net_cashflow=row_dict.get("net_cashflow") or row_dict.get("f_net_cashflow"),
                budget_utilization=row_dict.get("budget_utilization") or row_dict.get("f_budget_utilization"),
                bills_due_today=row_dict.get("bills_due_today") or row_dict.get("f_bills_due_today"),
                goals_progress_avg=row_dict.get("goals_progress_avg") or row_dict.get("f_goals_progress_avg"),
                wellness_score_for_one_day=row_dict.get(CanonicalField.WELLNESS_SCORE_ONE_DAY) or row_dict.get("wellness_score"),
                productivity_score=row_dict.get("productivity_score"),
                financial_health_score=row_dict.get("financial_health_score"),
                overall_day_score=row_dict.get("overall_day_score"),
                data_completeness=row_dict.get("data_completeness"),
            )
            recent_days.append(snapshot)
            be_dates.add(iso)

        for d in window_dates:
            iso = d.isoformat()
            if iso in synthesized and iso not in be_dates:
                recent_days.append(synthesized[iso])

        if not recent_days:
            return None
        recent_days.sort(key=lambda s: s.snapshot_date)

        cutoff_3d = today - timedelta(days=3)
        recent_3d = [(d, r) for d, r in valid_snapshots if d >= cutoff_3d]

        avg_data = None
        trends = {}

        if recent_3d:
            fields_to_avg = [
                CanonicalField.STEP_COUNT_ONE_DAY,
                CanonicalField.ACTIVE_MINUTES_ONE_DAY,
                CanonicalField.TOTAL_CALORIES_BURNED_ONE_DAY_KCAL,
                CanonicalField.TOTAL_WORKOUT_DURATION_ONE_DAY_MINUTES,
                CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM,
                CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS,
                CanonicalField.SLEEP_QUALITY_SCORE_ONE_NIGHT,
                CanonicalField.HEALTH_SCORE_ONE_DAY,
                CanonicalField.MEETING_DURATION_ONE_DAY_MINUTES,
                CanonicalField.WORK_SPAN_ONE_DAY_MINUTES,
                "task_completion_rate",
                "mood_score_avg",
                "mood_variance",
                "net_cashflow",
                "budget_utilization",
                "total_income",
                "total_expense",
                "goals_progress_avg",
                CanonicalField.WELLNESS_SCORE_ONE_DAY,
                "productivity_score",
                "financial_health_score",
                "overall_day_score",
            ]

            INT_FIELDS = {
                CanonicalField.STEP_COUNT_ONE_DAY,
                CanonicalField.ACTIVE_MINUTES_ONE_DAY,
                CanonicalField.SLEEP_QUALITY_SCORE_ONE_NIGHT,
                CanonicalField.HEALTH_SCORE_ONE_DAY,
                CanonicalField.MEETING_DURATION_ONE_DAY_MINUTES,
                CanonicalField.WORK_SPAN_ONE_DAY_MINUTES,
                CanonicalField.RESTING_HEART_RATE_ONE_DAY_BPM,
            }
            avgs = {}
            for field in fields_to_avg:
                vals = []
                for d, r in recent_3d:
                    v = HistoricalTrendsProcessor._snapshot_field(r, field)
                    if v is not None:
                        try:
                            vals.append(float(v))
                        except (ValueError, TypeError):
                            pass
                if vals:
                    avg_val = sum(vals) / len(vals)
                    avgs[field] = (
                        int(round(avg_val))
                        if field in INT_FIELDS
                        else round(avg_val, 2)
                    )
                else:
                    avgs[field] = None

            avg_data = DailySnapshotData(snapshot_date="3_day_average", **avgs)

            # Compute trends for core scores
            for trend_field, dict_key in [
                ("wellness_score_trend", CanonicalField.WELLNESS_SCORE_ONE_DAY),
                ("productivity_score_trend", "productivity_score"),
                ("finance_score_trend", "financial_health_score"),
            ]:
                # Sort ascending for trend calculation
                sorted_3d = sorted(recent_3d, key=lambda x: x[0])
                vals = []
                for d, r in sorted_3d:
                    v = HistoricalTrendsProcessor._snapshot_field(r, dict_key)
                    if v is not None:
                        try:
                            vals.append(float(v))
                        except Exception:
                            pass
                trends[trend_field] = HistoricalTrendsProcessor._compute_trend_string(
                    vals
                )

        return HistoricalTrendsBlock(
            recent_days=recent_days, averages_3d=avg_data, trends=trends
        )
