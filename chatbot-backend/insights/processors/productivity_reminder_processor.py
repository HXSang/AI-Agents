from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _to_iso_date(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.date().isoformat()
    except ValueError:
        return raw[:10] if len(raw) >= 10 else None


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def _status_by_rate(rate: float, good: float = 75.0, warn: float = 50.0) -> str:
    if rate >= good:
        return "good"
    if rate >= warn:
        return "moderate"
    return "attention"


class ProductivityReminderProcessor:
    """Doc-aligned reminder processor:
    summary -> signals -> anomalies -> overall.
    """

    @staticmethod
    def process(
        *,
        today_reminders: List[Dict[str, Any]],
        productivity_summary: Dict[str, Any],
        productivity_summaries_7d: List[Dict[str, Any]],
        now_local_date: str,
    ) -> Dict[str, Any]:
        reminders = [r for r in (today_reminders or []) if isinstance(r, dict)]
        total = len(reminders)
        # Support multiple completion field names from different API schemas
        def _is_completed(r: Dict[str, Any]) -> bool:
            return (
                r.get("completed") is True
                or r.get("isCompleted") is True
                or r.get("reminder_completion_status") is True
                or r.get("status") == "completed"
            )
        completed = sum(1 for r in reminders if _is_completed(r))
        open_count = max(total - completed, 0)
        recurring = sum(1 for r in reminders if r.get("isRecurring") is True)

        overdue_tasks: List[Dict[str, Any]] = []
        due_today = 0
        due_tomorrow = 0
        due_this_week = 0
        next_three_days = 0

        for r in reminders:
            due = _to_iso_date(r.get("dueDate"))
            if not due:
                continue
            if due == now_local_date:
                due_today += 1
            try:
                due_d = datetime.fromisoformat(due).date()
                now_d = datetime.fromisoformat(now_local_date).date()
                delta = (due_d - now_d).days
            except ValueError:
                continue

            if delta < 0 and not _is_completed(r):
                overdue_tasks.append(
                    {
                        "title": r.get("title") or "(untitled reminder)",
                        "dueDate": due,
                        "overdueDays": abs(delta),
                    }
                )
            if delta == 1:
                due_tomorrow += 1
            if 0 <= delta <= 6:
                due_this_week += 1
            if 0 <= delta <= 2:
                next_three_days += 1

        overdue_count = len(overdue_tasks)
        oldest_overdue_days = max((t["overdueDays"] for t in overdue_tasks), default=0)
        completion_rate = round(_safe_rate(completed, total), 1) if total else 0.0

        # Commitment completion from productivity summary
        commitments_due = float(productivity_summary.get("commitmentsDue") or 0)
        commitments_done = float(productivity_summary.get("commitmentsCompleted") or 0)
        commitments_rate = round(_safe_rate(commitments_done, commitments_due), 1)

        backlog = open_count + overdue_count
        backlog_status = (
            "high" if backlog >= 5 else "moderate" if backlog >= 3 else "low"
        )

        recurring_completed = sum(
            1 for r in reminders if r.get("isRecurring") is True and _is_completed(r)
        )
        recurring_rate = round(_safe_rate(recurring_completed, recurring), 1) if recurring else 0.0

        # Freshness
        ages: List[int] = []
        now_dt = datetime.fromisoformat(now_local_date).date()
        for r in reminders:
            created = _to_iso_date(r.get("createdAt"))
            if not created:
                continue
            try:
                cdt = datetime.fromisoformat(created).date()
            except ValueError:
                continue
            ages.append(max((now_dt - cdt).days, 0))
        avg_age_days = round((sum(ages) / len(ages)), 1) if ages else None

        # Trend from last 7d summaries
        rates_7d: List[float] = []
        backlog_7d: List[float] = []
        commitments_7d: List[float] = []
        for d in productivity_summaries_7d or []:
            if not isinstance(d, dict):
                continue
            rc = d.get("reminderCount")
            cc = d.get("completedReminderSoFar")
            if isinstance(rc, (int, float)) and rc > 0 and isinstance(cc, (int, float)):
                rates_7d.append(_safe_rate(float(cc), float(rc)))
                backlog_7d.append(max(float(rc) - float(cc), 0.0))
            c_due = d.get("commitmentsDue")
            c_done = d.get("commitmentsCompleted")
            if (
                isinstance(c_due, (int, float))
                and c_due > 0
                and isinstance(c_done, (int, float))
            ):
                commitments_7d.append(_safe_rate(float(c_done), float(c_due)))

        def _direction(values: List[float], tol: float = 5.0) -> str:
            if len(values) < 4:
                return "insufficient_data"
            mid = len(values) // 2
            first = sum(values[:mid]) / len(values[:mid])
            second = sum(values[mid:]) / len(values[mid:])
            delta = second - first
            if delta > tol:
                return "improving"
            if delta < -tol:
                return "declining"
            return "stable"

        completion_direction = _direction(rates_7d)
        backlog_direction = _direction([-x for x in backlog_7d])  # lower backlog => improve
        commitment_direction = _direction(commitments_7d)

        anomalies: List[Dict[str, Any]] = []
        if backlog >= 5:
            anomalies.append({"type": "high_backlog", "count": backlog})
        if overdue_count >= 3:
            anomalies.append({"type": "many_overdue", "count": overdue_count})
        if completed == 0 and total > 0:
            anomalies.append({"type": "no_completed_reminders", "count": total})
        if total >= 12:
            anomalies.append({"type": "high_reminder_volume", "count": total})
        if avg_age_days is not None and avg_age_days >= 7:
            anomalies.append({"type": "stale_reminders", "avgAgeDays": avg_age_days})

        summary = {
            "totalReminders": total,
            "completedReminders": completed,
            "openReminders": open_count,
            "overdueReminders": overdue_count,
            "recurringReminders": recurring,
            "completionRate": completion_rate,
        }

        signals: Dict[str, Dict[str, Any]] = {
            "taskCompletion": {
                "status": _status_by_rate(completion_rate),
                "metrics": {
                    "completed": completed,
                    "total": total,
                    "completionRate": completion_rate,
                },
                "comparison": {"remaining": open_count},
                "trend": {"direction": completion_direction},
                "evidence": {"tasks": total},
            },
            "commitmentCompletion": {
                "status": _status_by_rate(commitments_rate),
                "metrics": {
                    "completed": int(commitments_done),
                    "due": int(commitments_due),
                    "completionRate": commitments_rate,
                },
                "comparison": {},
                "trend": {"direction": commitment_direction},
                "evidence": {},
            },
            "overdueTasks": {
                "status": "high" if overdue_count >= 3 else "moderate" if overdue_count >= 1 else "low",
                "metrics": {
                    "count": overdue_count,
                    "oldestOverdueDays": oldest_overdue_days,
                },
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {"tasks": overdue_tasks[:3]},
            },
            "planningBacklog": {
                "status": backlog_status,
                "metrics": {"backlog": backlog},
                "comparison": {"open": open_count, "overdue": overdue_count},
                "trend": {"direction": backlog_direction},
                "evidence": {},
            },
            "dueAnalysis": {
                "status": "busy" if due_today >= 4 else "normal",
                "metrics": {
                    "dueToday": due_today,
                    "dueTomorrow": due_tomorrow,
                    "dueThisWeek": due_this_week,
                    "nextThreeDays": next_three_days,
                },
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {},
            },
            "reminderLoad": {
                "status": "high" if total >= 12 else "moderate" if total >= 6 else "light",
                "metrics": {"total": total, "open": open_count, "overdue": overdue_count},
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {},
            },
            "recurringTasks": {
                "status": _status_by_rate(recurring_rate, good=80, warn=50) if recurring else "insufficient_data",
                "metrics": {"count": recurring, "completionRate": recurring_rate},
                "comparison": {"openRecurring": max(recurring - recurring_completed, 0)},
                "trend": {"direction": "stable"},
                "evidence": {},
            },
            "reminderFreshness": {
                "status": "stale" if (avg_age_days or 0) >= 7 else "normal",
                "metrics": {"averageAgeDays": avg_age_days},
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {},
            },
            "completionTrend": {
                "status": "ok",
                "metrics": {"direction": completion_direction},
                "comparison": {
                    "backlogDirection": backlog_direction,
                    "commitmentDirection": commitment_direction,
                },
                "trend": {"direction": completion_direction},
                "evidence": {},
            },
            "anomalies": {
                "status": "attention" if anomalies else "normal",
                "metrics": {"count": len(anomalies)},
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {"items": anomalies},
            },
        }

        penalty = (
            min(overdue_count, 5) * 6
            + min(backlog, 8) * 3
            + (8 if completion_rate < 40 else 0)
        )
        score = max(0.0, min(100.0, 100.0 - penalty))
        overall = {
            "status": "good" if score >= 80 else "attention" if score < 60 else "moderate",
            "score": round(score, 1),
            "confidence": 0.9 if total > 0 else 0.5,
        }

        return {
            "summary": summary,
            "signals": signals,
            "anomalies": anomalies,
            "overall": overall,
        }
