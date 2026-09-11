from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

HIGH_PRIORITY_VALUES = {"high", "urgent", "critical", "p1", "p2"}

MAX_OVERDUE_TITLES = 3

MAX_LOOKAHEAD_DAYS = 30

COLLISION_BUFFER_MINS = 5


def _parse_due_dt(
    entry: Dict[str, Any], now_utc: datetime, user_tz: ZoneInfo
) -> Optional[datetime]:
    """Best-effort parse of a reminder's due datetime in user timezone."""
    raw = (
        entry.get("dueDate")
        or entry.get("due_date")
        or entry.get("dueTime")
        or entry.get("due_time")
        or entry.get("reminderDate")
    )
    if not raw:
        return None
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    # Normalize trailing Z (UTC) to +00:00 for fromisoformat.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Some APIs use "YYYY-MM-DD HH:MM:SS" without timezone.
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            logger.debug("Could not parse reminder dueDate=%r", raw)
            return None

    # If naive, assume user TZ.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=user_tz)
    # Convert to UTC for consistent delta math against now_utc.
    return dt.astimezone(timezone.utc)


def _event_window(
    event: Dict[str, Any], user_tz: ZoneInfo
) -> Optional[tuple[datetime, datetime]]:
    """Return (start_utc, end_utc) for a calendar event, or None."""
    start_raw = event.get("startTime") or event.get("start_time")
    end_raw = event.get("endTime") or event.get("end_time")
    if not start_raw or not end_raw:
        return None
    if not isinstance(start_raw, str) or not isinstance(end_raw, str):
        return None
    s, e = start_raw.strip(), end_raw.strip()
    if e.endswith("Z"):
        e = e[:-1] + "+00:00"
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        s_dt = datetime.fromisoformat(s)
        e_dt = datetime.fromisoformat(e)
    except ValueError:
        return None
    if s_dt.tzinfo is None:
        s_dt = s_dt.replace(tzinfo=user_tz)
    if e_dt.tzinfo is None:
        e_dt = e_dt.replace(tzinfo=user_tz)
    return s_dt.astimezone(timezone.utc), e_dt.astimezone(timezone.utc)


def extract_reminder_signals(
    raw_reminders: List[Dict[str, Any]],
    raw_calendar_events: List[Dict[str, Any]],
    now_utc: datetime,
    user_tz: ZoneInfo,
) -> Dict[str, Any]:
    if not raw_reminders:
        return {
            "reminders_due_today": 0,
            "high_priority_reminders_7d": 0,
            "overdue_reminders_count": 0,
            "overdue_reminder_titles": [],
            "nearest_reminder_title": None,
            "nearest_reminder_due_in_mins": None,
            "nearest_reminder_priority": None,
            "reminders_collision": False,
            "nearest_reminder": None,
            "reminders_by_type": {},
            "reminders_dismissed_7d": 0,
            "reminders_pending_7d": 0,
        }

    max_lookahead = now_utc.timestamp() + MAX_LOOKAHEAD_DAYS * 86400

    candidates: List[Dict[str, Any]] = []
    overdue_titles: List[str] = []
    overdue_count = 0
    high_priority_count = 0
    today_date_str = now_utc.astimezone(user_tz).date().isoformat()
    reminders_due_today = 0
    reminders_by_type: Dict[str, int] = {}
    reminders_dismissed_7d = 0
    reminders_pending_7d = 0

    for entry in raw_reminders:
        if not isinstance(entry, dict):
            continue
        # Skip completed — they are no longer actionable.
        # Support multiple completion field names from different API schemas
        if (
            entry.get("completed") is True
            or entry.get("isCompleted") is True
            or entry.get("reminder_completion_status") is True
            or entry.get("status") == "completed"
        ):
            continue
        is_dismissed = entry.get("dismissed") is True
        if is_dismissed:
            reminders_dismissed_7d += 1
        else:
            reminders_pending_7d += 1

        due_utc = _parse_due_dt(entry, now_utc, user_tz)
        if due_utc is None:
            continue
        # Ignore far-future recurring noise.
        if due_utc.timestamp() > max_lookahead:
            continue

        title = (
            entry.get("title")
            or entry.get("name")
            or entry.get("summary")
            or "(unnamed reminder)"
        )
        priority = (
            entry.get("priority")
            or entry.get("priorityLevel")
            or entry.get("priority_level")
        )
        priority_norm = priority.lower() if isinstance(priority, str) else None
        category = (
            entry.get("category")
            or entry.get("type")
            or entry.get("tag")
        )
        notes = entry.get("notes") or entry.get("description")
        if category:
            cat_key = str(category).upper()
            reminders_by_type[cat_key] = (
                reminders_by_type.get(cat_key, 0) + 1
            )

        minutes_until = int((due_utc - now_utc).total_seconds() / 60)
        due_local_iso = due_utc.astimezone(user_tz).isoformat()

        if minutes_until < 0:
            overdue_count += 1
            if len(overdue_titles) < MAX_OVERDUE_TITLES:
                overdue_titles.append(title)

        if priority_norm and priority_norm in HIGH_PRIORITY_VALUES:
            high_priority_count += 1

        if due_utc.astimezone(user_tz).date().isoformat() == today_date_str:
            reminders_due_today += 1

        candidates.append(
            {
                "title": title,
                "priority": priority_norm,
                "category": category,
                "notes": notes,
                "minutes_until": minutes_until,
                "due_local_iso": due_local_iso,
                "due_utc": due_utc,
            }
        )

    # Nearest = entry with smallest |minutes_until| (overdue rises to top).
    nearest: Optional[Dict[str, Any]] = None
    for cand in candidates:
        if nearest is None or abs(cand["minutes_until"]) < abs(
            nearest["minutes_until"]
        ):
            nearest = cand

    # Collision: nearest reminder overlaps any meeting today (with buffer).
    reminders_collision = False
    if nearest is not None:
        reminder_start = nearest["due_utc"]
        reminder_end = reminder_start.replace()  # moment-in-time; treat as 5m
        for event in raw_calendar_events or []:
            if not isinstance(event, dict):
                continue
            window = _event_window(event, user_tz)
            if not window:
                continue
            ev_start, ev_end = window
            # Buffer: meetings that end within buffer of the reminder still
            # count as colliding (no time to act).
            if (
                ev_start
                <= reminder_start
                <= ev_end + _mins_to_delta(COLLISION_BUFFER_MINS)
            ):
                reminders_collision = True
                break

    result: Dict[str, Any] = {
        "reminders_due_today": reminders_due_today,
        "high_priority_reminders_7d": high_priority_count,
        "overdue_reminders_count": overdue_count,
        "overdue_reminder_titles": overdue_titles,
        "nearest_reminder_title": nearest["title"] if nearest else None,
        "nearest_reminder_due_in_mins": (
            nearest["minutes_until"] if nearest else None
        ),
        "nearest_reminder_priority": nearest["priority"] if nearest else None,
        "reminders_collision": reminders_collision,
        # Enrichment — typed breakdown + dismissed/pending split.
        "reminders_by_type": reminders_by_type,
        "reminders_dismissed_7d": reminders_dismissed_7d,
        "reminders_pending_7d": reminders_pending_7d,
    }
    if nearest:
        result["nearest_reminder"] = {
            "title": nearest["title"],
            "due_at": nearest["due_local_iso"],
            "minutes_until": nearest["minutes_until"],
            "priority": nearest["priority"],
            "category": nearest["category"],
            "notes": nearest["notes"],
        }
    else:
        result["nearest_reminder"] = None
    return result


def _mins_to_delta(mins: int):
    from datetime import timedelta

    return timedelta(minutes=mins)