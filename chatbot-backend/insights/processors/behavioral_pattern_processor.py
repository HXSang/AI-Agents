from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from insights.health.canonical_field_mapping import CanonicalField
from insights.insight_config import BehaviorConfig
from insights.insight_config import TrendStatus
from insights.processors.calendar_intelligence_processor import _event_category
from insights.processors.calendar_intelligence_processor import _event_end_str
from insights.processors.calendar_intelligence_processor import _event_start_str
from insights.schemas.processed_context import WeekComparison
from insights.schemas.processed_context import BehavioralPatternsBlock

_DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

_MEETING_TYPES = {"MEETING", "MEETINGS", "CONFERENCE"}
_SHORT_MEETING_MINUTES = 30
_LATE_EVENT_HOUR = 21
_STEPS_VALUE_KEYS = ("steps", CanonicalField.STEP_COUNT_ONE_DAY)
_SLEEP_VALUE_KEYS = (
    "hours",
    "totalSleepHours",
    CanonicalField.SLEEP_DURATION_FROM_SUMMARY_ONE_NIGHT_HOURS,
)


def _parse_any_date(raw: Any) -> Optional[date]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    try:
        text = str(raw).replace("Z", "+00:00")
        if "T" in text:
            return date.fromisoformat(text.split("T")[0])
        return date.fromisoformat(text[:10])
    except Exception:
        return None


def _parse_iso_dt(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except Exception:
        return None


def _align_dt_pair(
    start: datetime, end: datetime
) -> Optional[Tuple[datetime, datetime]]:
    if start.tzinfo is not None and end.tzinfo is None:
        end = end.replace(tzinfo=start.tzinfo)
    elif start.tzinfo is None and end.tzinfo is not None:
        start = start.replace(tzinfo=end.tzinfo)
    elif start.tzinfo is not None and end.tzinfo is not None:
        end = end.astimezone(start.tzinfo)
    if end <= start:
        return None
    return start, end


def _is_all_day(event: Dict[str, Any]) -> bool:
    v = event.get("allDay")
    if v is None:
        v = event.get("all_day")
    return bool(v)


def _is_meeting_event(event: Dict[str, Any]) -> bool:
    if _event_category(event) == "MEETINGS":
        return True
    et = (
        event.get("eventType")
        or event.get("event_type")
        or event.get("calendar_event_type")
    )
    if et is not None and str(et).strip().upper() in _MEETING_TYPES:
        return True
    participants = event.get("participants")
    return isinstance(participants, list) and len(participants) >= 2


def _participant_id(participant: Any) -> Optional[str]:
    if isinstance(participant, str) and participant.strip():
        return participant.strip().lower()
    if isinstance(participant, dict):
        for key in ("email", "emailAddress", "id", "name"):
            value = participant.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
    return None


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _item_field(item: Any, field: str) -> Any:
    if isinstance(item, dict):
        return item.get(field)
    return getattr(item, field, None)


def _snapshot_date(item: Any) -> Optional[date]:
    return _parse_any_date(
        _item_field(item, "daily_snapshot_date")
        or _item_field(item, "snapshot_date")
    )


def _calendar_events_for_behavior(raw_data: Dict[str, Any]) -> List[Any]:
    week_events = raw_data.get("calendar_events_week")
    if isinstance(week_events, list) and week_events:
        return week_events
    events = raw_data.get("calendar_events")
    return events if isinstance(events, list) else []


def _resolve_user_timezone(raw_data: Dict[str, Any]):
    names = [
        raw_data.get("time_data_timezone"),
        raw_data.get("timezone"),
        raw_data.get("user_timezone"),
    ]
    time_data = raw_data.get("time_data")
    if isinstance(time_data, dict):
        names.extend(
            [
                time_data.get("timezone"),
                time_data.get("user_timezone"),
                time_data.get("time_data_timezone"),
            ]
        )
    profile = raw_data.get("user_profile")
    if isinstance(profile, dict):
        names.append(profile.get("timezone"))
    for name in names:
        if not isinstance(name, str) or not name.strip():
            continue
        try:
            return ZoneInfo(name.strip())
        except Exception:
            continue
    return None


def _fill_from_snapshots(
    series: Dict[date, Any],
    snapshots: List[Any],
    extract,
) -> Dict[date, Any]:
    out = dict(series)
    for item in snapshots or []:
        day = _snapshot_date(item)
        if day is None:
            continue
        value = extract(item)
        if value is None:
            continue
        existing = out.get(day)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged = dict(value)
            merged.update(existing)
            out[day] = merged
        elif existing is None:
            out[day] = value
    return out


def _health_daily_series(
    raw_data: Optional[Dict[str, Any]],
    type_key: str,
    value_keys: Tuple[str, ...],
    aggregate: str = "last",
) -> Dict[date, float]:
    if not isinstance(raw_data, dict):
        return {}
    stats = raw_data.get("today_health_stats")
    if not isinstance(stats, dict):
        return {}
    block = stats.get(type_key)
    if not isinstance(block, dict):
        return {}
    rows = block.get("data") or []
    if not isinstance(rows, list):
        return {}
    out: Dict[date, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        day = _parse_any_date(row.get("date"))
        if day is None:
            continue
        value = None
        for key in value_keys:
            value = _numeric(row.get(key))
            if value is not None:
                break
        if value is None:
            continue
        if aggregate == "sum" and day in out:
            out[day] += value
        else:
            out[day] = value
    return out


def _to_local_naive(dt: datetime, tz) -> datetime:
    if tz is None or dt.tzinfo is None:
        return dt
    return dt.astimezone(tz).replace(tzinfo=None)


def _calendar_day_metrics(events: List[Any], tz=None) -> Dict[date, Dict[str, Any]]:
    grouped: Dict[date, Dict[str, Any]] = {}
    for event in events or []:
        if not isinstance(event, dict):
            continue
        start = _parse_iso_dt(_event_start_str(event))
        if start is None:
            continue
        start = _to_local_naive(start, tz)
        day = start.date()
        slot = grouped.setdefault(
            day,
            {
                "meeting_mins": [],
                "collabs": set(),
                "starts": [],
                "ends": [],
                "late": 0,
                "timed": False,
            },
        )
        participants = event.get("participants")
        if isinstance(participants, list):
            for participant in participants:
                pid = _participant_id(participant)
                if pid:
                    slot["collabs"].add(pid)
        if _is_all_day(event):
            continue
        end = _parse_iso_dt(_event_end_str(event))
        if end is None:
            continue
        end = _to_local_naive(end, tz)
        aligned = _align_dt_pair(start, end)
        if aligned is None:
            continue
        start, end = aligned
        minutes = max(int((end - start).total_seconds() // 60), 0)
        slot["timed"] = True
        slot["starts"].append(start)
        slot["ends"].append(end)
        if start.hour >= _LATE_EVENT_HOUR:
            slot["late"] += 1
        if _is_meeting_event(event):
            slot["meeting_mins"].append(minutes)

    out: Dict[date, Dict[str, Any]] = {}
    for day, slot in grouped.items():
        row: Dict[str, Any] = {}
        meetings = slot["meeting_mins"]
        if meetings:
            row["avg_meeting_min"] = sum(meetings) / len(meetings)
            row["short_meeting_count"] = sum(
                1 for mins in meetings if mins < _SHORT_MEETING_MINUTES
            )
        if slot["collabs"]:
            row["unique_collaborators"] = len(slot["collabs"])
        if slot["timed"] and slot["starts"] and slot["ends"]:
            span = max(slot["ends"]) - min(slot["starts"])
            row["work_span_minutes"] = max(int(span.total_seconds() // 60), 0)
            row["late_event_count"] = slot["late"]
        if row:
            out[day] = row
    return out


def _meeting_from_snapshot(item: Any) -> Optional[Dict[str, float]]:
    row: Dict[str, float] = {}
    avg = _numeric(_item_field(item, "p_avg_meeting_min"))
    short = _numeric(_item_field(item, "p_short_events_count"))
    if avg is not None:
        row["avg_meeting_min"] = avg
    if short is not None:
        row["short_meeting_count"] = short
    return row or None


def _work_span_from_snapshot(item: Any) -> Optional[Dict[str, float]]:
    row: Dict[str, float] = {}
    span = _numeric(
        _item_field(item, CanonicalField.WORK_SPAN_ONE_DAY_MINUTES)
        or _item_field(item, "p_work_span_minutes")
    )
    late = _numeric(
        _item_field(item, CanonicalField.EVENTS_AFTER_9_PM_ONE_DAY_COUNT)
        or _item_field(item, "p_events_after_9pm")
    )
    if span is not None:
        row["work_span_minutes"] = span
    if late is not None:
        row["late_event_count"] = late
    return row or None


def _collaboration_from_snapshot(item: Any) -> Optional[float]:
    return _numeric(_item_field(item, "p_unique_collaborators"))


def _task_rate_from_snapshot(item: Any) -> Optional[float]:
    return _numeric(_item_field(item, "p_task_completion_rate"))


def _steps_from_snapshot(item: Any) -> Optional[float]:
    return _numeric(
        _item_field(item, CanonicalField.STEP_COUNT_ONE_DAY) or _item_field(item, "h_steps")
    )


def _meeting_history(
    calendar_days: Dict[date, Dict[str, Any]], snapshots: List[Any]
) -> Dict[date, Any]:
    series: Dict[date, Any] = {}
    for day, row in calendar_days.items():
        partial: Dict[str, float] = {}
        if "avg_meeting_min" in row:
            partial["avg_meeting_min"] = row["avg_meeting_min"]
        if "short_meeting_count" in row:
            partial["short_meeting_count"] = row["short_meeting_count"]
        if partial:
            series[day] = partial
    return _fill_from_snapshots(series, snapshots, _meeting_from_snapshot)


def _work_span_history(
    calendar_days: Dict[date, Dict[str, Any]], snapshots: List[Any]
) -> Dict[date, Any]:
    series: Dict[date, Any] = {}
    for day, row in calendar_days.items():
        if "work_span_minutes" not in row:
            continue
        series[day] = {
            "work_span_minutes": row["work_span_minutes"],
            "late_event_count": row.get("late_event_count") or 0,
        }
    return _fill_from_snapshots(series, snapshots, _work_span_from_snapshot)


def _collaboration_history(
    calendar_days: Dict[date, Dict[str, Any]], snapshots: List[Any]
) -> Dict[date, Any]:
    series = {
        day: row["unique_collaborators"]
        for day, row in calendar_days.items()
        if "unique_collaborators" in row
    }
    return _fill_from_snapshots(series, snapshots, _collaboration_from_snapshot)


def _task_rate_from_entry(entry: Dict[str, Any]) -> Optional[float]:
    due = entry.get("commitmentsDue")
    done = entry.get("commitmentsCompleted")
    if not isinstance(due, (int, float)):
        event_due = entry.get("eventCommitmentsDue")
        reminder_due = entry.get("reminderCommitmentsDue")
        if isinstance(event_due, (int, float)) or isinstance(reminder_due, (int, float)):
            due = float(event_due or 0) + float(reminder_due or 0)
            done = float(entry.get("eventCommitmentsCompleted") or 0) + float(
                entry.get("reminderCommitmentsCompleted") or 0
            )
    if isinstance(due, (int, float)) and due > 0 and isinstance(done, (int, float)):
        return (float(done) / float(due)) * 100.0
    return None


def _productivity_entry_date(entry: Dict[str, Any]) -> Optional[date]:
    day = _parse_any_date(entry.get("date") or entry.get("Date"))
    if day is not None:
        return day
    date_range = entry.get("dateRange") or entry.get("date_range")
    if isinstance(date_range, dict):
        return _parse_any_date(
            date_range.get("startDate") or date_range.get("start_date")
        )
    return None


def _productivity_daily_rates(raw_data: Optional[Dict[str, Any]]) -> Dict[date, float]:
    if not isinstance(raw_data, dict):
        return {}
    rows: List[Any] = list(raw_data.get("productivity_summaries_7d") or [])
    summary = raw_data.get("productivity_summary")
    if isinstance(summary, dict):
        rows.append(summary)
    out: Dict[date, float] = {}
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        day = _productivity_entry_date(entry)
        rate = _task_rate_from_entry(entry)
        if day is None or rate is None:
            continue
        out[day] = rate
    return out


def _task_history(raw_data: Dict[str, Any], snapshots: List[Any]) -> Dict[date, float]:
    return _fill_from_snapshots(
        _productivity_daily_rates(raw_data), snapshots, _task_rate_from_snapshot
    )


def _activity_history(
    raw_data: Dict[str, Any], snapshots: List[Any]
) -> Dict[date, float]:
    series = _health_daily_series(
        raw_data, "STEPS", _STEPS_VALUE_KEYS, aggregate="sum"
    )
    return _fill_from_snapshots(series, snapshots, _steps_from_snapshot)


def _sleep_from_snapshot(item: Any) -> Optional[float]:
    return _numeric(
        _item_field(item, CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS)
        or _item_field(item, "h_sleep_hours")
    )


def _sleep_history(
    raw_data: Dict[str, Any], snapshots: List[Any]
) -> Dict[date, float]:
    series = _health_daily_series(raw_data, "SLEEP", _SLEEP_VALUE_KEYS)
    return _fill_from_snapshots(series, snapshots, _sleep_from_snapshot)


def _latest_week_values(
    series: Dict[date, Any],
    today: date,
    min_days: int = 3,
    n: int = 3,
) -> Optional[List[Any]]:
    week_start = today - timedelta(days=today.weekday())
    by_date = {
        day: value
        for day, value in series.items()
        if week_start <= day <= today and value is not None
    }
    if len(by_date) < min_days:
        return None
    latest = sorted(by_date.keys(), reverse=True)[:n]
    latest.sort()
    return [by_date[day] for day in latest]


def _trend_from_values(
    values: List[float],
    scale: Optional[float] = None,
    custom_delta: Optional[float] = None,
) -> str:
    mid = len(values) // 2
    first_half_avg = sum(values[:mid]) / mid
    second_half_avg = sum(values[mid:]) / (len(values) - mid)
    delta = second_half_avg - first_half_avg
    if scale:
        delta = delta / scale
    threshold = (
        custom_delta if custom_delta is not None else BehaviorConfig.TREND_DELTA
    )
    if delta > threshold:
        return TrendStatus.IMPROVING
    if delta < -threshold:
        return TrendStatus.DECLINING
    return TrendStatus.STABLE


class BehavioralPatternProcessor:

    @staticmethod
    def _get_field(item: Any, field: str) -> Any:
        if isinstance(item, dict):
            return item.get(field)
        return getattr(item, field, None)

    @staticmethod
    def process(
        historical_snapshots: List[Any], today: date, steps_goal: Optional[int] = None,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> BehavioralPatternsBlock:
        raw = raw_data if isinstance(raw_data, dict) else {}
        history: List[Any] = list(historical_snapshots or [])
        if not history:
            history = list(raw.get("historical_snapshots") or [])

        calendar_days = _calendar_day_metrics(
            _calendar_events_for_behavior(raw), _resolve_user_timezone(raw)
        )
        meeting_history = _meeting_history(calendar_days, history)
        work_span_history = _work_span_history(calendar_days, history)
        collaboration_history = _collaboration_history(calendar_days, history)
        task_history = _task_history(raw, history)
        activity_history = _activity_history(raw, history)
        sleep_history = _sleep_history(raw, history)

        sleep_trend = BehavioralPatternProcessor._compute_sleep_trend(
            sleep_history, today
        )
        activity_trend = BehavioralPatternProcessor._compute_activity_trend(
            activity_history, today, steps_goal
        )
        week_comparison = BehavioralPatternProcessor._compute_week_comparison(
            historical_snapshots=history, today=today
        )
        personal_context = BehavioralPatternProcessor._compute_activity_pattern(
            historical_snapshots=history, today=today
        )
        meeting_frag = BehavioralPatternProcessor._compute_meeting_fragmentation(
            meeting_history, today
        )
        work_span = BehavioralPatternProcessor._compute_work_span_trend(
            work_span_history, today
        )
        collab_load = BehavioralPatternProcessor._compute_collaboration_load(
            collaboration_history, today
        )
        task_momentum = BehavioralPatternProcessor._compute_task_completion_momentum(
            task_history, today
        )
        mood_avg_label = BehavioralPatternProcessor._compute_mood_avg_7d_label(
            raw, today
        )

        return BehavioralPatternsBlock(
            sleep_trend_7d=sleep_trend,
            activity_trend_7d=activity_trend,
            this_week_vs_last=week_comparison,
            personal_context=personal_context,
            meeting_fragmentation=meeting_frag,
            work_span_trend=work_span,
            collaboration_load=collab_load,
            task_completion_momentum=task_momentum,
            mood_avg_7d_label=mood_avg_label,
        )

    @staticmethod
    def _parse_date(item: Any) -> Optional[date]:
        raw = (
            BehavioralPatternProcessor._get_field(item, "daily_snapshot_date")
            or BehavioralPatternProcessor._get_field(item, "snapshot_date")
        )
        if not raw:
            return None
        return _parse_any_date(raw)

    @staticmethod
    def _compute_sleep_trend(
        series: Dict[date, float], today: date
    ) -> Optional[str]:
        values = _latest_week_values(series, today)
        if values is None:
            return None
        return _trend_from_values(values)

    @staticmethod
    def _compute_activity_trend(
        series: Dict[date, float], today: date, steps_goal: Optional[int]
    ) -> Optional[str]:
        values = _latest_week_values(series, today)
        if values is None:
            return None
        return _trend_from_values(
            values, scale=steps_goal, custom_delta=0.10
        )

    @staticmethod
    def _compute_week_comparison(
        historical_snapshots: List[Any], today: date
    ) -> WeekComparison:
        """
        Compare current calendar week (Mon–today) vs last full calendar week (Mon–Sun).
        """
        days_since_monday = today.weekday()
        this_week_start = today - timedelta(days=days_since_monday)
        last_week_start = this_week_start - timedelta(days=7)
        last_week_end = this_week_start

        def week_avg(history, field, start, end):
            vals = [
                BehavioralPatternProcessor._get_field(r, field)
                for r in history
                if BehavioralPatternProcessor._get_field(r, field) is not None
                and BehavioralPatternProcessor._parse_date(r)
                and start <= BehavioralPatternProcessor._parse_date(r) < end
            ]
            return sum(vals) / len(vals) if vals else None

        def compare(current_avg, last_avg) -> str:
            if current_avg is None or last_avg is None:
                return TrendStatus.INSUFFICIENT_DATA
            if last_avg == 0:
                return TrendStatus.INSUFFICIENT_DATA
            ratio = (current_avg - last_avg) / last_avg
            if ratio > BehaviorConfig.SIMILAR_THRESHOLD:
                return TrendStatus.AHEAD
            elif ratio < -BehaviorConfig.SIMILAR_THRESHOLD:
                return TrendStatus.BEHIND
            return TrendStatus.SIMILAR

        steps_this = week_avg(
            historical_snapshots, CanonicalField.STEP_COUNT_ONE_DAY, this_week_start, today + timedelta(days=1)
        )
        steps_last = week_avg(
            historical_snapshots, CanonicalField.STEP_COUNT_ONE_DAY, last_week_start, last_week_end
        )
        sleep_this = week_avg(
            historical_snapshots,
            CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS,
            this_week_start,
            today + timedelta(days=1),
        )
        sleep_last = week_avg(
            historical_snapshots, CanonicalField.SLEEP_DURATION_ONE_NIGHT_HOURS, last_week_start, last_week_end
        )
        focus_this = week_avg(
            historical_snapshots,
            "p_focus_blocks_60min",
            this_week_start,
            today + timedelta(days=1),
        )
        focus_last = week_avg(
            historical_snapshots, "p_focus_blocks_60min", last_week_start, last_week_end
        )

        return WeekComparison(
            steps=compare(steps_this, steps_last),
            sleep=compare(sleep_this, sleep_last),
            focus_hours=compare(focus_this, focus_last),
        )

    @staticmethod
    def _compute_activity_pattern(historical_snapshots: List[Any], today: date) -> str:
        """
        Find historically most active days of week (by steps).
        Compare with this week's performance on those days.
        Uses a 4-week rolling window to keep the baseline adaptive.
        """
        days_since_monday = today.weekday()
        this_week_start = today - timedelta(days=days_since_monday)
        history_cutoff = this_week_start - timedelta(
            days=BehaviorConfig.HISTORY_CUTOFF_DAYS
        )

        historical: Dict[int, List[float]] = defaultdict(list)
        this_week: Dict[int, Optional[float]] = {}

        for r in historical_snapshots:
            d = BehavioralPatternProcessor._parse_date(r)
            steps = BehavioralPatternProcessor._get_field(r, CanonicalField.STEP_COUNT_ONE_DAY)
            if d is None or steps is None:
                continue
            if d >= this_week_start:
                this_week[d.weekday()] = steps
            elif d >= history_cutoff:
                historical[d.weekday()].append(steps)

        if not historical:
            return "Insufficient historical data to determine activity patterns."

        if not this_week:
            return "No activity data logged yet for this week."

        weekday_avgs = {wd: sum(vals) / len(vals) for wd, vals in historical.items()}

        if not weekday_avgs:
            return "Insufficient historical data to determine activity patterns."

        top_days = sorted(weekday_avgs, key=lambda wd: weekday_avgs[wd], reverse=True)[
            :2
        ]
        top_day_names = [_DAY_NAMES[wd] for wd in top_days]

        underperforming = []
        for wd in top_days:
            if wd in this_week:
                actual = this_week[wd]
                historical_avg = weekday_avgs[wd]
                if (
                    historical_avg > 0
                    and actual < historical_avg * BehaviorConfig.UNDERPERFORMING_RATIO
                ):
                    underperforming.append(_DAY_NAMES[wd])

        if underperforming:
            return f"Historically most active on {', '.join(top_day_names)}, but underperformed on {', '.join(underperforming)} this week."
        elif all(wd not in this_week for wd in top_days):
            return f"Historically most active on {', '.join(top_day_names)}. Those days are yet to come this week."
        else:
            return f"On track with historical active days ({', '.join(top_day_names)})."

    @staticmethod
    def _compute_meeting_fragmentation(
        series: Dict[date, Any], today: date
    ) -> Optional[str]:
        complete = {
            day: value
            for day, value in series.items()
            if isinstance(value, dict)
            and "avg_meeting_min" in value
            and "short_meeting_count" in value
        }
        rows = _latest_week_values(complete, today)
        if rows is None:
            return None
        short_counts = [row["short_meeting_count"] for row in rows]
        avg_mins = [row["avg_meeting_min"] for row in rows]
        if (
            sum(short_counts) / len(short_counts) >= 3
            and sum(avg_mins) / len(avg_mins) < 30
        ):
            return "highly_fragmented"
        elif sum(avg_mins) / len(avg_mins) > 45:
            return "focused_blocks"
        return "normal_fragmentation"

    @staticmethod
    def _compute_work_span_trend(
        series: Dict[date, Any], today: date
    ) -> Optional[str]:
        complete = {
            day: value
            for day, value in series.items()
            if isinstance(value, dict) and "work_span_minutes" in value
        }
        rows = _latest_week_values(complete, today)
        if rows is None:
            return None
        spans = [row["work_span_minutes"] for row in rows]
        late_events = [row.get("late_event_count") or 0 for row in rows]
        avg_span = sum(spans) / len(spans)
        if avg_span > 600 or (sum(late_events) >= 2):
            return "extended_hours"
        if avg_span > 0 and sum(late_events) == 0:
            return "stable_boundaries"
        return "normal_boundaries"

    @staticmethod
    def _compute_collaboration_load(
        series: Dict[date, Any], today: date
    ) -> Optional[str]:
        collabs = _latest_week_values(series, today)
        if collabs is None:
            return None
        if sum(collabs) / len(collabs) >= 6:
            return "high_collaboration_tax"
        return "normal_collaboration"

    @staticmethod
    def _compute_task_completion_momentum(
        series: Dict[date, float], today: date
    ) -> Optional[str]:
        rates = _latest_week_values(series, today, min_days=3)
        if rates is None:
            return None
        avg_rate = sum(rates) / len(rates)
        if avg_rate >= 70:
            return "strong_execution"
        elif avg_rate <= 40:
            return "execution_lag"
        return "steady_execution"

    @staticmethod
    def _compute_mood_avg_7d_label(raw_data: Dict[str, Any], today: date) -> Optional[str]:
        moods_7d = raw_data.get("moods_7d") or []
        cutoff = today - timedelta(days=7)
        scores: List[float] = []
        for entry in moods_7d:
            if not isinstance(entry, dict):
                continue
            mood_date_str = (
                entry.get("moodDate")
                or entry.get("mood_date")
                or entry.get("current_mood_date")
            )
            if not mood_date_str:
                continue
            try:
                if "T" in str(mood_date_str):
                    d = date.fromisoformat(str(mood_date_str).split("T")[0])
                else:
                    d = date.fromisoformat(str(mood_date_str))
            except Exception:
                continue
            if d < cutoff or d >= today:
                continue
            score = entry.get("moodScore") or entry.get("mood_score")
            if isinstance(score, (int, float)):
                scores.append(float(score))
        if len(scores) >= 2:
            avg_mood = sum(scores) / len(scores)
            if avg_mood >= 4:
                return "consistently_positive"
            elif avg_mood <= 2:
                return "consistently_negative"
            return "neutral_or_mixed"
        return "insufficient_data"
