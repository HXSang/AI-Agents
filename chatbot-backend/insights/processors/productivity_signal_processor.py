import logging
from datetime import datetime, timezone as tz
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from insights.health.canonical_field_mapping import CanonicalField
from insights.processors.productivity_calendar_processor import (
    ProductivityCalendarProcessor,
)
from insights.processors.productivity_reminder_processor import (
    ProductivityReminderProcessor,
)
from insights.processors.reminder_signal_processor import extract_reminder_signals
from insights.processors.work_hours_processor import compute_work_hours_signals
from insights.schemas.processed_context import ProductivitySignalsBlock

logger = logging.getLogger(__name__)


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _merge_processor_signals(
    reminder_signals: Dict[str, Any],
    calendar_signals: Dict[str, Any],
    combined_anomalies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Flat-merge reminder + calendar signals without silent key collisions.

    Both processors expose ``commitmentCompletion`` and ``anomalies``. Calendar
    would overwrite reminder if merged with ``{**rem, **cal}``. We:
    - strip nested anomaly signals and rebuild from the combined list
    - keep calendar ``commitmentCompletion`` as the schedule-quality primary
    - preserve reminder's richer trend under ``reminderCommitmentCompletion``
    """
    rem = dict(reminder_signals or {})
    cal = dict(calendar_signals or {})
    rem.pop("anomalies", None)
    cal.pop("anomalies", None)

    rem_commit = rem.pop("commitmentCompletion", None)
    cal_commit = cal.pop("commitmentCompletion", None)

    merged: Dict[str, Any] = {**rem, **cal}

    if rem_commit is not None:
        merged["reminderCommitmentCompletion"] = rem_commit
    if cal_commit is not None:
        merged["commitmentCompletion"] = cal_commit
    elif rem_commit is not None:
        merged["commitmentCompletion"] = rem_commit

    merged["anomalies"] = {
        "status": "attention" if combined_anomalies else "normal",
        "metrics": {"count": len(combined_anomalies)},
        "comparison": {},
        "trend": {"direction": "stable"},
        "evidence": {"items": combined_anomalies},
    }
    return merged


def _compute_7d_trend(
    prod_7d: List[Dict[str, Any]],
    historical_snapshots: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Optional[Any]]:
    meeting_counts: List[float] = []
    focus_vals: List[float] = []
    task_rates: List[float] = []
    meeting_completion_rates: List[float] = []
    commitments_rates: List[float] = []
    chronic_overload_days = 0
    meeting_overload_days = 0
    active_days = 0
    commitments_total = 0
    commitments_done = 0

    for entry in prod_7d:
        if not isinstance(entry, dict):
            continue

        meeting_count = entry.get(
            CanonicalField.MEETING_COUNT_TODAY,
            entry.get("meetingCount", 0),
        )
        if isinstance(meeting_count, (int, float)):
            mc_f = float(meeting_count)
            meeting_counts.append(mc_f)
            if mc_f > 5:
                meeting_overload_days += 1

        fm = entry.get("focusMinutes")
        if isinstance(fm, (int, float)):
            focus_vals.append(float(fm))
        fop = entry.get("focusOutputPercent")
        if isinstance(fop, (int, float)):
            focus_vals.append(float(fop) * 1.2)

        completed_meetings = entry.get(
            CanonicalField.COMPLETED_MEETING_COUNT_TODAY,
            entry.get("completedMeetingSoFar", 0),
        )
        if isinstance(meeting_count, (int, float)) and isinstance(completed_meetings, (int, float)):
            mc_f = float(meeting_count)
            if mc_f > 0:
                meeting_completion_rates.append(float(completed_meetings) / mc_f)
                active_days += 1

        done = entry.get(
            CanonicalField.COMPLETED_REMINDER_COUNT_TODAY,
            entry.get("completedTasks"),
        )
        total = entry.get(
            CanonicalField.REMINDER_COUNT_TODAY,
            entry.get("totalTasks"),
        )
        if (
            isinstance(done, (int, float))
            and isinstance(total, (int, float))
            and total > 0
        ):
            task_rates.append(float(done) / float(total))
            active_days += 1
            continue
        op = entry.get(
            CanonicalField.MEETING_TIME_PERCENTAGE_TODAY,
            entry.get("overallPercent"),
        )
        if isinstance(op, (int, float)):
            task_rates.append(float(op) / 100.0)
            if op > 0:
                active_days += 1

        cd = entry.get("commitmentsDue")
        cc = entry.get("commitmentsCompleted")
        if isinstance(cd, (int, float)):
            commitments_total += float(cd)
        if isinstance(cc, (int, float)) and isinstance(cd, (int, float)) and cd > 0:
            commitments_rates.append(float(cc) / float(cd))
            commitments_done += float(cc)

        if entry.get("wholeDayMeetingCount", 0) > 0:
            chronic_overload_days += 1

    if not task_rates and historical_snapshots:
        for snap in historical_snapshots:
            if not isinstance(snap, dict):
                continue
            r = snap.get("p_task_completion_rate")
            if isinstance(r, (int, float)):
                task_rates.append(float(r))

    meeting_7d_avg = _safe_mean(meeting_counts)
    focus_7d_avg = _safe_mean(focus_vals)
    events_7d_avg = _safe_mean(task_rates)
    meeting_completion_7d_avg = _safe_mean(meeting_completion_rates)
    commitments_rate_7d_avg = _safe_mean(commitments_rates)

    trend: Optional[str] = None
    if len(task_rates) >= 4:
        mid = len(task_rates) // 2
        first_half = _safe_mean(task_rates[:mid])
        second_half = _safe_mean(task_rates[mid:])
        if first_half is not None and second_half is not None:
            delta = second_half - first_half
            if delta > 0.10:
                trend = "improving"
            elif delta < -0.10:
                trend = "declining"
            else:
                trend = "stable"

    return {
        "meeting_minutes_7d_avg": meeting_7d_avg,
        "meeting_counts_7d_avg": _safe_mean(meeting_counts),
        "meeting_completion_7d_avg": meeting_completion_7d_avg,
        "focus_minutes_7d_avg": focus_7d_avg,
        "events_completion_7d_avg": events_7d_avg,
        "commitments_rate_7d_avg": commitments_rate_7d_avg,
        "commitments_7d_total": commitments_total,
        "commitments_7d_completed": commitments_done,
        "chronic_overload_days": chronic_overload_days,
        "meeting_overload_days": meeting_overload_days,
        "active_days_7d": active_days,
        "events_completion_trend": trend,
    }


def _resolve_now_and_tz(raw_data: Dict[str, Any]) -> tuple[datetime, ZoneInfo]:
    tz_name = raw_data["timezone"]
    try:
        user_tz = ZoneInfo(tz_name)
    except Exception:
        logger.warning("Unknown timezone %r, falling back to UTC", tz_name)
        user_tz = ZoneInfo("UTC")

    current_time = raw_data.get("current_time")
    if isinstance(current_time, str) and current_time:
        s = current_time.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            now = datetime.fromisoformat(s)
            if now.tzinfo is None:
                now = now.replace(tzinfo=user_tz)
            return now.astimezone(tz.utc), user_tz
        except ValueError:
            pass
    return datetime.now(tz.utc), user_tz


def _merge_overall(reminder_overall: Dict[str, Any], calendar_overall: Dict[str, Any]) -> Dict[str, Any]:
    r_score = float(reminder_overall.get("score") or 0.0)
    c_score = float(calendar_overall.get("score") or 0.0)
    score = round((r_score * 0.5) + (c_score * 0.5), 1)
    status = "good" if score >= 80 else "attention" if score < 60 else "moderate"
    return {
        "status": status,
        "score": score,
        "confidence": min(
            float(reminder_overall.get("confidence") or 0.7),
            float(calendar_overall.get("confidence") or 0.7),
        ),
    }


class ProductivitySignalProcessor:
    @staticmethod
    def process(
        raw_data: Dict[str, Any],
        historical_snapshots: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[ProductivitySignalsBlock]:
        prod_summary = raw_data.get("productivity_summary") or {}
        prod_7d = raw_data.get("productivity_summaries_7d") or []
        today_reminders = raw_data.get("today_reminders") or []
        calendar_events = raw_data.get("calendar_events") or []
        calendar_work_hours = raw_data.get("calendar_work_hours") or []

        if (
            not prod_summary
            and not prod_7d
            and not today_reminders
            and not calendar_events
        ):
            return None

        now_utc, user_tz = _resolve_now_and_tz(raw_data)
        now_local_date = now_utc.astimezone(user_tz).date().isoformat()

        reminder_block = ProductivityReminderProcessor.process(
            today_reminders=today_reminders,
            productivity_summary=prod_summary,
            productivity_summaries_7d=prod_7d,
            now_local_date=now_local_date,
        )
        calendar_block = ProductivityCalendarProcessor.process(
            calendar_events=calendar_events,
            productivity_summary=prod_summary,
            calendar_work_hours=calendar_work_hours,
        )

        reminder_signals = extract_reminder_signals(
            raw_reminders=today_reminders,
            raw_calendar_events=calendar_events,
            now_utc=now_utc,
            user_tz=user_tz,
        )

        summary = {
            "reminder": reminder_block.get("summary", {}),
            "calendar": calendar_block.get("summary", {}),
        }
        anomalies = (reminder_block.get("anomalies", []) or []) + (
            calendar_block.get("anomalies", []) or []
        )
        signals = _merge_processor_signals(
            reminder_block.get("signals", {}) or {},
            calendar_block.get("signals", {}) or {},
            anomalies,
        )
        overall = _merge_overall(
            reminder_block.get("overall", {}) or {},
            calendar_block.get("overall", {}) or {},
        )

        trend = _compute_7d_trend(prod_7d, historical_snapshots)
        meeting_7d_avg = trend["meeting_counts_7d_avg"]
        focus_7d_avg = trend["focus_minutes_7d_avg"]

        meeting_count = (
            (calendar_block.get("summary") or {}).get("meetingCount")
            or prod_summary.get("meetingCount")
            or 0
        )
        focus_output = prod_summary.get("focusOutputPercent", 0)

        meeting_vs_avg_pct = None
        if meeting_7d_avg and meeting_7d_avg > 0 and meeting_count:
            meeting_vs_avg_pct = (float(meeting_count) / meeting_7d_avg) - 1.0

        focus_vs_avg_pct = None
        if focus_7d_avg and focus_7d_avg > 0 and focus_output:
            focus_vs_avg_pct = (float(focus_output) / focus_7d_avg) - 1.0

        raw_work_hours = raw_data.get("calendar_work_hours") or []
        today_date = now_utc.astimezone(user_tz).date()
        work_hours_signals = compute_work_hours_signals(raw_work_hours, today=today_date)

        completion_rate = (
            ((signals.get("taskCompletion") or {}).get("metrics") or {}).get(
                "completionRate"
            )
            or 0.0
        )

        reminder_total = prod_summary.get(CanonicalField.REMINDER_COUNT_TODAY) or 0
        reminder_completed = prod_summary.get(CanonicalField.COMPLETED_REMINDER_COUNT_TODAY) or 0

        meeting_total = prod_summary.get(CanonicalField.MEETING_COUNT_TODAY) or 0
        meeting_completed = prod_summary.get(CanonicalField.COMPLETED_MEETING_COUNT_TODAY) or 0

        # Meeting completion rate is the canonical productivity signal.
        meeting_completion_rate: Optional[float] = None
        if meeting_total > 0:
            meeting_completion_rate = (float(meeting_completed) / float(meeting_total)) * 100.0
        elif completion_rate:
            # Fall back to backend-reported taskCompletion only if no meetings today.
            meeting_completion_rate = float(completion_rate)
        utilization = (
            ((signals.get("calendarUtilization") or {}).get("metrics") or {}).get(
                "utilization"
            )
            or 0.0
        )
        backlog = (
            ((signals.get("planningBacklog") or {}).get("metrics") or {}).get("backlog")
            or 0
        )
        focus_blocks = (
            ((signals.get("focusBlocks") or {}).get("metrics") or {}).get("focusBlocks")
            or 0
        )
        overdue_count = (
            ((signals.get("overdueTasks") or {}).get("metrics") or {}).get("count")
            or 0
        )
        due_today = (
            ((signals.get("dueAnalysis") or {}).get("metrics") or {}).get("dueToday")
            or 0
        )
        next_three_days = (
            ((signals.get("dueAnalysis") or {}).get("metrics") or {}).get("nextThreeDays")
            or 0
        )

        level = "ok"
        if utilization >= 90 or backlog >= 6 or overdue_count >= 4:
            level = "overloaded"
        elif completion_rate >= 80 and utilization <= 60 and focus_blocks >= 2:
            level = "underutilized"

        meeting_hours = (
            ((signals.get("meetingLoad") or {}).get("metrics") or {}).get("meetingHours")
            or 0.0
        )
        if meeting_hours >= 5:
            meeting_fatigue = "high"
        elif meeting_hours >= 2:
            meeting_fatigue = "mild"
        else:
            meeting_fatigue = "none"

        if focus_blocks >= 2:
            schedule_focus_score = "high"
        elif focus_blocks == 1:
            schedule_focus_score = "moderate"
        else:
            schedule_focus_score = "low"

        focus_score = schedule_focus_score
        reminder_completion_rate = (
            (float(reminder_completed) / float(reminder_total)) if reminder_total > 0 else None
        )
        high_execution = (
            (meeting_completion_rate is not None and meeting_completion_rate >= 80.0)
            or (reminder_completion_rate is not None and reminder_completion_rate >= 0.8)
        )
        if high_execution:
            if focus_score == "low":
                focus_score = "moderate"
            elif focus_score == "moderate":
                focus_score = "high"

        return ProductivitySignalsBlock(
            summary=summary,
            signals=signals,
            anomalies=anomalies,
            overall=overall,
            source_processors=["productivity_reminder_processor", "productivity_calendar_processor"],
            level=level,
            meeting_fatigue=meeting_fatigue,
            focus_score=focus_score,
            events_completion_rate=(meeting_completion_rate / 100.0 if meeting_completion_rate is not None else None),
            meeting_completion_rate=(meeting_completion_rate / 100.0 if meeting_completion_rate is not None else None),
            reminders_completion_rate=(
                (float(reminder_completed) / float(reminder_total)) if reminder_total > 0 else None
            ),
            reminders_due_today=int(reminder_total),
            meetings_due_today=int(meeting_total),
            upcoming_deadlines=int(next_three_days),
            meeting_minutes_7d_avg=meeting_7d_avg,
            meeting_minutes_vs_avg_pct=meeting_vs_avg_pct,
            focus_minutes_7d_avg=focus_7d_avg,
            focus_minutes_vs_avg_pct=focus_vs_avg_pct,
            events_completion_7d_avg=trend["events_completion_7d_avg"],
            events_completion_trend=trend["events_completion_trend"],
            chronic_overload_days=trend["chronic_overload_days"] or 0,
            meeting_overload_days=trend["meeting_overload_days"] or 0,
            active_days_7d=trend["active_days_7d"] or 0,
            commitments_7d_avg=trend["commitments_rate_7d_avg"],
            commitments_7d_total=int(trend["commitments_7d_total"] or 0),
            commitments_7d_completed=int(trend["commitments_7d_completed"] or 0),
            meeting_completion_7d_avg=trend["meeting_completion_7d_avg"],
            high_priority_reminders_7d=reminder_signals.get("high_priority_reminders_7d", 0),
            nearest_reminder_title=reminder_signals.get("nearest_reminder_title"),
            nearest_reminder_due_in_mins=reminder_signals.get("nearest_reminder_due_in_mins"),
            nearest_reminder_priority=reminder_signals.get("nearest_reminder_priority"),
            overdue_reminders_count=int(overdue_count),
            reminders_collision=reminder_signals.get("reminders_collision", False),
            reminders_by_type=reminder_signals.get("reminders_by_type", {}),
            reminders_dismissed_7d=reminder_signals.get("reminders_dismissed_7d", 0),
            reminders_pending_7d=reminder_signals.get("reminders_pending_7d", 0),
            work_hours_7d_scheduled_avg=work_hours_signals["work_hours_7d_scheduled_avg"],
            work_hours_7d_target=work_hours_signals["work_hours_7d_target"],
            work_hours_overload_days_7d=work_hours_signals["work_hours_overload_days_7d"],
            work_hours_underload_days_7d=work_hours_signals["work_hours_underload_days_7d"],
        )
